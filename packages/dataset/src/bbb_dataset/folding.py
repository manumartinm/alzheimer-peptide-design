from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .struct_io import parse_cif_backbone, write_coords_npz

SAMPLE_METRIC_KEYS = (
    "structure_confidence",
    "ptm",
    "iptm",
    "complex_plddt",
    "complex_iplddt",
    "complex_pde",
    "complex_ipde",
)


@dataclass
class FoldConfig:
    model: str = "boltz-2.1"
    max_workers: int = 4
    limit: int = 0
    structures_dir: str = "data/structures"
    experiments_dir: str = "boltz-experiments"
    output_name: str = "peptides_struct_manifest.parquet"
    resume: bool = True


def sequence_hash(sequence: str) -> str:
    return sha256(sequence.encode()).hexdigest()[:16]


def run_dir_for_sequence(experiments_dir: Path, sequence: str) -> Path:
    return experiments_dir / f"bbb-fold-{sequence_hash(sequence)}"


def run_status(run_dir: Path) -> str | None:
    run_json = run_dir / "run.json"
    if not run_json.exists():
        return None
    payload = json.loads(run_json.read_text(encoding="utf-8"))
    return payload.get("status")


def _as_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _sample_metrics(sample: dict[str, Any]) -> dict[str, float]:
    raw = sample.get("metrics") or {}
    return {key: _as_float(raw.get(key)) for key in SAMPLE_METRIC_KEYS}


def parse_run_json(run_json_path: Path) -> dict[str, float]:
    payload = json.loads(run_json_path.read_text(encoding="utf-8"))
    output = payload.get("output") or {}
    sample = output.get("best_sample") or {}
    if not sample:
        samples = output.get("all_sample_results") or []
        sample = samples[0] if samples else {}
    return _sample_metrics(sample)


def manifest_fields(metrics: dict[str, float]) -> dict[str, float]:
    complex_plddt = metrics.get("complex_plddt", float("nan"))
    plddt = complex_plddt * 100.0 if np.isfinite(complex_plddt) else float("nan")
    return {
        "plddt": plddt,
        "ptm": metrics.get("ptm", float("nan")),
        "structure_confidence": metrics.get("structure_confidence", float("nan")),
        "iptm": metrics.get("iptm", float("nan")),
        "complex_iplddt": metrics.get("complex_iplddt", float("nan")),
        "complex_pde": metrics.get("complex_pde", float("nan")),
        "complex_ipde": metrics.get("complex_ipde", float("nan")),
    }


def find_structure_cif(run_dir: Path) -> Path:
    for pattern in (
        "outputs/files/**/*.cif",
        "**/sample_*predicted_structure.cif",
        "**/sample_*.cif",
    ):
        matches = sorted(run_dir.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No structure CIF under {run_dir}")


def _write_coords_for_run(run_dir: Path, seq_hash: str, structures_dir: Path) -> Path:
    out_dir = structures_dir / seq_hash
    out_dir.mkdir(parents=True, exist_ok=True)
    coords_path = out_dir / "coords.npz"
    if not coords_path.exists():
        cif_path = find_structure_cif(run_dir)
        write_coords_npz(coords_path, parse_cif_backbone(cif_path))
    return coords_path


def has_local_structure(run_dir: Path) -> bool:
    try:
        find_structure_cif(run_dir)
        return True
    except FileNotFoundError:
        return False


def import_from_run_dir(
    run_dir: Path,
    sequence: str,
    *,
    structures_dir: Path,
) -> dict[str, object]:
    run_json = run_dir / "run.json"
    if not run_json.exists():
        raise FileNotFoundError(f"Missing run.json in {run_dir}")
    if run_status(run_dir) != "succeeded":
        raise RuntimeError(f"Run not succeeded: {run_dir}")
    if not has_local_structure(run_dir):
        raise FileNotFoundError(f"No structure CIF under {run_dir}")

    seq = sequence.upper()
    seq_hash = sequence_hash(seq)
    fields = manifest_fields(parse_run_json(run_json))
    coords_path = _write_coords_for_run(run_dir, seq_hash, structures_dir)
    return {
        "sequence_hash": seq_hash,
        "coords_path": str(coords_path.resolve()),
        **fields,
        "length": len(seq),
    }


def fold_sequence(
    sequence: str,
    *,
    model: str,
    structures_dir: Path,
) -> dict[str, object]:
    from boltz_api import Boltz

    api_key = os.environ.get("BOLTZ_API_KEY")
    if not api_key:
        raise RuntimeError("BOLTZ_API_KEY missing; set it in dataset/.env.local")

    seq = sequence.upper()
    seq_hash = sequence_hash(seq)
    client = Boltz(api_key=api_key)
    run_dir = Path(
        client.predictions.structure_and_binding.run(
            model=model,
            input={
                "entities": [{"type": "protein", "value": seq, "chain_ids": ["A"]}],
                "num_samples": 1,
            },
            name=f"bbb-fold-{seq_hash}",
        )
    )
    return import_from_run_dir(run_dir, seq, structures_dir=structures_dir)


def resolve_or_fold_sequence(
    sequence: str,
    *,
    model: str,
    structures_dir: Path,
    experiments_dir: Path,
    resume: bool,
) -> tuple[dict[str, object], str]:
    seq = sequence.upper()
    run_dir = run_dir_for_sequence(experiments_dir, seq)
    if resume and run_status(run_dir) == "succeeded" and has_local_structure(run_dir):
        return import_from_run_dir(run_dir, seq, structures_dir=structures_dir), "imported"
    return fold_sequence(
        seq,
        model=model,
        structures_dir=structures_dir,
    ), "folded"


@dataclass
class StructureFolder:
    config: FoldConfig

    def rebuild_manifest_from_experiments(
        self,
        df: pd.DataFrame,
        *,
        experiments_dir: Path,
        structures_dir: Path,
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        if "sequence" not in df.columns:
            raise ValueError("Input dataframe must contain a sequence column")

        structures_dir.mkdir(parents=True, exist_ok=True)
        id_col = "peptide_id" if "peptide_id" in df.columns else None
        unique = df.drop_duplicates(subset=["sequence"]).reset_index(drop=True)

        rows: list[dict[str, object]] = []
        missing = 0
        for _, row in unique.iterrows():
            seq = str(row["sequence"]).upper()
            run_dir = run_dir_for_sequence(experiments_dir, seq)
            if run_status(run_dir) != "succeeded" or not has_local_structure(run_dir):
                missing += 1
                continue
            try:
                result = import_from_run_dir(run_dir, seq, structures_dir=structures_dir)
            except (FileNotFoundError, RuntimeError):
                missing += 1
                continue
            rows.append(
                {
                    "peptide_id": row[id_col] if id_col else result["sequence_hash"],
                    "sequence": seq,
                    **result,
                }
            )

        manifest = pd.DataFrame(rows)
        stats = {
            "n_unique_sequences": len(unique),
            "n_folded": len(manifest),
            "n_missing": int(missing),
            "limit": 0,
        }
        return manifest, stats

    def build_manifest(
        self,
        df: pd.DataFrame,
        *,
        structures_dir: Path,
        experiments_dir: Path | None = None,
        manifest_only: bool = False,
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        cfg = self.config
        exp_dir = experiments_dir or structures_dir.parent.parent / cfg.experiments_dir
        if manifest_only:
            return self.rebuild_manifest_from_experiments(
                df,
                experiments_dir=exp_dir,
                structures_dir=structures_dir,
            )

        if "sequence" not in df.columns:
            raise ValueError("Input dataframe must contain a sequence column")

        structures_dir.mkdir(parents=True, exist_ok=True)
        exp_dir.mkdir(parents=True, exist_ok=True)
        id_col = "peptide_id" if "peptide_id" in df.columns else None
        unique = df.drop_duplicates(subset=["sequence"]).reset_index(drop=True)
        if cfg.limit > 0:
            unique = unique.head(cfg.limit)

        def _task(row: pd.Series) -> tuple[dict[str, object], str]:
            seq = str(row["sequence"]).upper()
            result, source = resolve_or_fold_sequence(
                seq,
                model=cfg.model,
                structures_dir=structures_dir,
                experiments_dir=exp_dir,
                resume=cfg.resume,
            )
            return (
                {
                    "peptide_id": row[id_col] if id_col else result["sequence_hash"],
                    "sequence": seq,
                    **result,
                },
                source,
            )

        rows: list[dict[str, object]] = []
        n_imported = 0
        n_folded = 0
        n_errors = 0
        errors: list[str] = []
        with ThreadPoolExecutor(max_workers=max(1, cfg.max_workers)) as pool:
            futures = {pool.submit(_task, row): row for _, row in unique.iterrows()}
            for fut in as_completed(futures):
                try:
                    row_dict, source = fut.result()
                    rows.append(row_dict)
                    if source == "imported":
                        n_imported += 1
                    else:
                        n_folded += 1
                except Exception as exc:
                    n_errors += 1
                    seq = str(futures[fut]["sequence"]).upper()
                    errors.append(f"{seq}: {exc}")

        manifest = pd.DataFrame(rows)
        stats = {
            "n_unique_sequences": len(unique),
            "n_folded": len(manifest),
            "n_imported": int(n_imported),
            "n_api_folded": int(n_folded),
            "n_errors": int(n_errors),
            "limit": int(cfg.limit),
            "experiments_dir": str(exp_dir.resolve()),
        }
        if errors:
            stats["errors"] = errors[:20]
        return manifest, stats


def build_struct_manifest(
    df: pd.DataFrame,
    *,
    structures_dir: Path,
    experiments_dir: Path | None = None,
    fold_cfg: FoldConfig | None = None,
    manifest_only: bool = False,
) -> tuple[pd.DataFrame, dict[str, int]]:
    return StructureFolder(fold_cfg or FoldConfig()).build_manifest(
        df,
        structures_dir=structures_dir,
        experiments_dir=experiments_dir,
        manifest_only=manifest_only,
    )
