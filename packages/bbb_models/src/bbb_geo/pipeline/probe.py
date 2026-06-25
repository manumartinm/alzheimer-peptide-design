from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bbb_geo.features.membrane_potential import amphipathicity_score, per_residue_hydrophobicity
from bbb_geo.features.struct_loader import build_struct_batch, load_struct_manifest, merge_dataset_with_manifest
from bbb_geo.models import StructEGNNGeo
from bbb_geo.train.checkpoints import load_checkpoint


def _grad_norm(model: StructEGNNGeo, sample: dict) -> float:
    coords = sample["coords"].detach().clone().requires_grad_(True)
    sample = dict(sample)
    sample["coords"] = coords
    logp = model.log_prob([sample]).sum()
    logp.backward()
    if coords.grad is None:
        return 0.0
    return float(torch.linalg.norm(coords.grad).item())


def run(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    df = pd.read_parquet(args.dataset)
    manifest = load_struct_manifest(args.manifest)
    merged = merge_dataset_with_manifest(df, manifest, sequence_col="sequence").head(args.max_samples)
    _, samples = build_struct_batch(merged, "sequence")

    state = load_checkpoint(run_dir / "checkpoints" / "best.ckpt")
    model = StructEGNNGeo()
    model.load_state_dict(state["model"])
    model.eval()

    grad_norms = [_grad_norm(model, s) for s in samples]
    amp_scores = []
    for sample in samples:
        hydro = per_residue_hydrophobicity(str(sample["sequence"]), device=sample["coords"].device, dtype=sample["coords"].dtype)
        amp_scores.append(float(amphipathicity_score(sample["coords"], hydro).item()))

    with torch.no_grad():
        probs = [float(torch.sigmoid(model.forward(graphs=[s])).item()) for s in samples]

    corr = float(np.corrcoef(probs, amp_scores)[0, 1]) if len(probs) > 1 else float("nan")
    report = {
        "mean_grad_norm": float(np.mean(grad_norms)),
        "median_grad_norm": float(np.median(grad_norms)),
        "prob_amp_correlation": corr,
        "n_samples": len(samples),
        "gate_pass": float(np.mean(grad_norms)) > 1e-3 and (np.isnan(corr) or abs(corr) > 0.1),
        "recommendation": "hybrid" if float(np.mean(grad_norms)) > 1e-3 else "physics_only",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Geometry-sensitivity gate for geo guidance.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="artifacts/geometry_sensitivity.json")
    parser.add_argument("--max-samples", type=int, default=50)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
