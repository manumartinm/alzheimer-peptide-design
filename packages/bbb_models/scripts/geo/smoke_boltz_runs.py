#!/usr/bin/env python3
"""Smoke-test bbb_geo on structures downloaded under dataset/boltz-experiments."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_REPO = Path(__file__).resolve().parents[3]
_DATASET = _REPO / "dataset"
_BOLTZ_DEFAULT = _DATASET / "boltz-experiments"

sys.path.insert(0, str(_DATASET / "src"))

from tfg_bbb.struct_io import parse_cif_backbone, write_coords_npz  # noqa: E402

from bbb_geo.features.membrane_potential import amphipathicity_score, per_residue_hydrophobicity
from bbb_geo.features.struct_graph import build_struct_graph
from bbb_geo.infer.struct_guidance import BBBGuidanceConfig, compute_bbb_guidance_force
from bbb_geo.models import StructEGNNGeo
from bbb_geo.pipeline.train import run as run_geo_train


def _find_structure_cif(run_dir: Path) -> Path:
    for pattern in ("**/sample_*predicted_structure.cif", "**/sample_*.cif", "outputs/files/**/*.cif"):
        matches = sorted(run_dir.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No CIF under {run_dir}")


def _parse_run(run_json: Path) -> dict:
    payload = json.loads(run_json.read_text(encoding="utf-8"))
    sequence = payload["input"]["entities"][0]["value"].upper()
    sample = payload.get("output", {}).get("best_sample") or {}
    if not sample:
        results = payload.get("output", {}).get("all_sample_results") or []
        sample = results[0] if results else {}
    metrics = sample.get("metrics") or {}
    complex_plddt = float(metrics.get("complex_plddt", float("nan")))
    return {
        "sequence": sequence,
        "plddt": complex_plddt * 100.0 if complex_plddt == complex_plddt else float("nan"),
        "ptm": float(metrics.get("ptm", float("nan"))),
        "structure_confidence": float(metrics.get("structure_confidence", float("nan"))),
        "status": payload.get("status"),
    }


def build_manifest_from_boltz_runs(boltz_dir: Path, out_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    structures_dir = out_dir / "structures"
    structures_dir.mkdir(parents=True, exist_ok=True)

    for run_json in sorted(boltz_dir.glob("bbb-fold-*/run.json")):
        meta = _parse_run(run_json)
        if meta["status"] != "succeeded":
            continue
        cif_path = _find_structure_cif(run_json.parent)
        parsed = parse_cif_backbone(cif_path)
        seq_from_cif = "".join(parsed["sequence"])
        if seq_from_cif != meta["sequence"]:
            raise ValueError(f"Sequence mismatch for {run_json.parent.name}: {meta['sequence']} vs {seq_from_cif}")

        from hashlib import sha256

        seq_hash = sha256(meta["sequence"].encode()).hexdigest()[:16]
        coords_path = structures_dir / seq_hash / "coords.npz"
        coords_path.parent.mkdir(parents=True, exist_ok=True)
        write_coords_npz(coords_path, parsed)

        rows.append(
            {
                "sequence": meta["sequence"],
                "sequence_hash": seq_hash,
                "coords_path": str(coords_path.resolve()),
                "plddt": meta["plddt"],
                "ptm": meta["ptm"],
                "structure_confidence": meta["structure_confidence"],
                "boltz_run_dir": str(run_json.parent.resolve()),
                "cif_path": str(cif_path.resolve()),
            }
        )
    if not rows:
        raise RuntimeError(f"No succeeded runs found in {boltz_dir}")
    return pd.DataFrame(rows)


def check_model_forward(manifest: pd.DataFrame) -> dict[str, float]:
    row = manifest.iloc[0]
    graph = build_struct_graph(
        __import__("numpy").load(row["coords_path"], allow_pickle=True)["coords"],
        row["sequence"],
    )
    model = StructEGNNGeo(hidden_dim=32, num_layers=2)
    model.eval()

    logits = model.forward(graphs=[graph])
    prob = float(torch.sigmoid(logits).item())
    logp = float(model.log_prob([graph]).item())

    coords = graph["coords"].detach().clone().requires_grad_(True)
    graph_grad = dict(graph)
    graph_grad["coords"] = coords
    energy = -model.log_prob([graph_grad]).sum()
    energy.backward()
    grad_norm = float(torch.linalg.norm(coords.grad).item()) if coords.grad is not None else 0.0

    hydro = per_residue_hydrophobicity(row["sequence"], device=coords.device, dtype=coords.dtype)
    amp = float(amphipathicity_score(graph["coords"], hydro).item())

    atom_coords = coords.unsqueeze(0)
    n = coords.shape[0]
    feats = {
        "atom_to_token": torch.eye(n),
        "design_mask": torch.ones(n),
        "res_type": torch.nn.functional.one_hot(torch.arange(n) % 20 + 2, num_classes=33).float(),
    }
    membrane_force = compute_bbb_guidance_force(
        atom_coords,
        feats,
        torch.ones(n),
        sigma=2.0,
        cfg=BBBGuidanceConfig(bbb_weight=0.0, membrane_weight=1.0, ckpt_path="", sigma_gate=8.0),
    )
    membrane_norm = float(torch.linalg.norm(membrane_force).item()) if membrane_force is not None else 0.0

    return {
        "prob": prob,
        "log_prob": logp,
        "grad_norm": grad_norm,
        "amphipathicity": amp,
        "membrane_force_norm": membrane_norm,
        "n_residues": float(len(row["sequence"])),
    }


def micro_train(manifest: pd.DataFrame, preview_csv: Path, work_dir: Path) -> dict:
    labels = pd.read_csv(preview_csv)
    seq = manifest.iloc[0]["sequence"]
    label_rows = labels[labels["sequence"] == seq]
    if label_rows.empty:
        raise RuntimeError(f"No BBB label for {seq} in {preview_csv}")
    label_row = label_rows.iloc[0].to_dict()
    train_row = {**label_row, "fold_id": 1}
    val_row = {**label_row, "fold_id": 0}

    manifest_path = work_dir / "manifest.parquet"
    dataset_path = work_dir / "dataset.parquet"
    manifest.to_parquet(manifest_path, index=False)
    pd.DataFrame([train_row, val_row]).to_parquet(dataset_path, index=False)

    exp_yaml = _REPO / "bbb_models" / "configs" / "experiments" / "exp09_struct_egnn_noise.yaml"
    data_yaml = work_dir / "data.yaml"
    train_yaml = work_dir / "train.yaml"

    import yaml

    data_cfg = yaml.safe_load((_REPO / "bbb_models" / "configs" / "data.yaml").read_text())
    data_cfg["dataset_path"] = str(dataset_path.resolve())
    data_cfg["struct_manifest_path"] = str(manifest_path.resolve())
    data_cfg["fold_col"] = "fold_id"
    data_cfg["test_size"] = 0.5
    data_yaml.write_text(yaml.safe_dump(data_cfg), encoding="utf-8")

    train_cfg = yaml.safe_load((_REPO / "bbb_models" / "configs" / "train.yaml").read_text())
    train_cfg["training"]["epochs"] = 1
    train_cfg["training"]["batch_size"] = 1
    train_cfg["training"]["patience"] = 10
    train_cfg["tracking"] = {"tensorboard": False, "mlflow": False}
    train_yaml.write_text(yaml.safe_dump(train_cfg), encoding="utf-8")

    ds_path = str(dataset_path.resolve())
    artifacts_root = str(work_dir / "artifacts")

    class Args:
        exp = str(exp_yaml.resolve())
        data_config = str(data_yaml)
        train_config = str(train_yaml)
        output_root = artifacts_root
        dataset_path = ds_path

    run_geo_train(Args())
    run_dir = work_dir / "artifacts" / "models" / "exp09_struct_egnn_noise"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    ckpt = run_dir / "checkpoints" / "best.ckpt"
    if not ckpt.exists():
        raise RuntimeError(f"Expected checkpoint at {ckpt}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test bbb_geo on Boltz experiment folders.")
    parser.add_argument("--boltz-dir", default=str(_BOLTZ_DEFAULT))
    parser.add_argument("--preview-csv", default=str(_DATASET / "data/processed/peptides_bbb_preview.csv"))
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    boltz_dir = Path(args.boltz_dir).resolve()
    if not boltz_dir.is_dir():
        raise SystemExit(f"Boltz runs directory not found: {boltz_dir}")

    with tempfile.TemporaryDirectory(prefix="bbb_geo_smoke_") as tmp:
        work = Path(tmp)
        manifest = build_manifest_from_boltz_runs(boltz_dir, work)
        print(f"Loaded {len(manifest)} Boltz run(s):")
        print(manifest[["sequence", "plddt", "structure_confidence", "coords_path"]].to_string(index=False))

        checks = check_model_forward(manifest)
        print("\nForward / guidance checks:")
        for key, value in checks.items():
            print(f"  {key}: {value}")

        assert checks["n_residues"] == len(manifest.iloc[0]["sequence"])
        assert checks["grad_norm"] > 0, "Expected non-zero coordinate gradient from log_prob"
        assert checks["membrane_force_norm"] > 0, "Expected non-zero membrane guidance force"

        if not args.skip_train:
            metrics = micro_train(manifest, Path(args.preview_csv), work)
            print("\nMicro-train metrics:")
            print(json.dumps(metrics, indent=2))
            assert np.isfinite(metrics["raw"]["brier"])

    print("\nbbb_geo smoke test passed on Boltz structures.")


if __name__ == "__main__":
    main()
