#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bbb_classifier.constants import THREE_TO_ONE
from boltzgen_design.filtering.gates import (
    GateThresholds,
    gate_g1,
    gate_g2,
    gate_g3,
    gate_g4,
    gate_g5,
)
from boltzgen_design.filtering.pareto import pareto_front
from boltzgen_design.scoring.bbb_oracle import BBBOracle
from boltzgen_design.utils.paths import BBB_CLASSIFIER


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run G1-G5 cascade and Pareto ranking.")
    p.add_argument("--input-dir", required=True, help="Directory with design CIFs and CSV metrics")
    p.add_argument("--output-csv", required=True)
    p.add_argument("--bbb-repo", default=str(BBB_CLASSIFIER))
    p.add_argument("--bbb-run-dir", required=True)
    p.add_argument("--top-k", type=int, default=30)
    return p.parse_args()


def extract_sequence_from_cif(cif_file: Path) -> str:
    sequence: list[str] = []
    with cif_file.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 4:
                continue
            if not parts[0].startswith(("ATOM", "HETATM")):
                continue
            res_name = parts[3].upper()
            aa = THREE_TO_ONE.get(res_name)
            if aa and (not sequence or sequence[-1] != aa):
                sequence.append(aa)
    return "".join(sequence)


def first_existing(df: pd.DataFrame, names: list[str], default: float = 0.0) -> pd.Series:
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(default)
    return pd.Series([default] * len(df))


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    metrics_files = sorted(input_dir.rglob("aggregate_metrics*.csv"))
    if metrics_files:
        df = pd.read_csv(metrics_files[0])
    else:
        cif_files = sorted(input_dir.rglob("*.cif"))
        df = pd.DataFrame({"structure_path": [str(p) for p in cif_files]})

    if "structure_path" not in df.columns:
        candidate_col = "path" if "path" in df.columns else None
        if candidate_col is None and "name" in df.columns:
            df["structure_path"] = df["name"].map(lambda x: str(input_dir / f"{x}.cif"))
        elif candidate_col is not None:
            df["structure_path"] = df[candidate_col].astype(str)
        else:
            raise ValueError("Could not infer structure paths from metrics CSV")

    df["structure_path"] = df["structure_path"].map(
        lambda p: str((input_dir / p).resolve()) if not Path(p).is_absolute() else p
    )
    df["sequence"] = df["structure_path"].map(
        lambda p: extract_sequence_from_cif(Path(p)) if Path(p).exists() else ""
    )

    bbb_oracle = BBBOracle(
        bbb_repo_root=Path(args.bbb_repo),
        run_dir=Path(args.bbb_run_dir),
    )
    bbb_scores_file = output_csv.with_name(f"{output_csv.stem}.bbb.csv")
    bbb_oracle.score_sequences(df["sequence"].tolist(), bbb_scores_file)
    bbb_df = pd.read_csv(bbb_scores_file)

    prob_col = (
        "p_bbb_calibrated"
        if "p_bbb_calibrated" in bbb_df.columns
        else ("p_bbb_raw" if "p_bbb_raw" in bbb_df.columns else "probability")
    )
    df["bbb_probability"] = pd.to_numeric(bbb_df[prob_col], errors="coerce").fillna(0.0)

    df["hotspot_fraction"] = first_existing(
        df, ["hotspot_fraction", "g1_hotspot_fraction"], default=0.0
    )
    df["atp_repulsion"] = first_existing(df, ["atp_repulsion", "g2_atp_repulsion"], default=1e9)
    df["iptm"] = first_existing(df, ["iptm", "design_iptm"], default=0.0)
    df["plddt"] = first_existing(df, ["plddt", "design_plddt"], default=0.0)
    df["closure_rmsd"] = first_existing(df, ["closure_rmsd", "cyclic_closure_rmsd"], default=1e9)

    # Placeholder liability gate: can be replaced with TANGO and motif checks.
    df["passes_sequence_liability"] = df["sequence"].map(
        lambda s: len(s) > 0 and "KKKK" not in s and "RRRR" not in s
    )

    thresholds = GateThresholds()
    rows = df.to_dict(orient="records")
    gated: list[dict] = []
    for row in rows:
        if not gate_g1(row, thresholds):
            continue
        if not gate_g2(row, thresholds):
            continue
        if not gate_g3(row, thresholds):
            continue
        if not gate_g4(row, thresholds):
            continue
        if not gate_g5(row):
            continue
        row["pareto_hotspot"] = float(row["hotspot_fraction"])
        row["pareto_atp_avoidance"] = float(-row["atp_repulsion"])
        row["pareto_bbb"] = float(row["bbb_probability"])
        row["pareto_iptm"] = float(row["iptm"])
        gated.append(row)

    front = pareto_front(
        gated,
        keys=["pareto_hotspot", "pareto_atp_avoidance", "pareto_bbb", "pareto_iptm"],
    )
    front_df = pd.DataFrame(front).sort_values(
        by=["pareto_hotspot", "pareto_bbb", "pareto_iptm"],
        ascending=False,
    )
    front_df.head(args.top_k).to_csv(output_csv, index=False)
    print(f"Saved {min(args.top_k, len(front_df))} filtered candidates to {output_csv}")


if __name__ == "__main__":
    main()
