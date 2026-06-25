from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bbb_classifier.utils.io import ensure_dir, read_yaml, write_json


def _parse_list(raw: str, cast):
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


def run(args: argparse.Namespace) -> None:
    exp_cfg = read_yaml(args.exp)
    exp_name = exp_cfg.get("name", "exp09_struct_egnn_noise")
    cv_script = Path(__file__).resolve().parents[3] / "scripts" / "geo" / "cv.py"
    output_dir = ensure_dir(Path(args.output_root) / exp_name)

    caps = _parse_list(args.coord_sigma_caps, float)
    aux_weights = _parse_list(args.aux_weights, float)
    rows: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix="bbb_geo_sweep_") as tmp:
        tmp_dir = Path(tmp)
        for cap in caps:
            for aux in aux_weights:
                variant = dict(exp_cfg)
                variant["name"] = f"{exp_name}_cap{cap:g}_aux{aux:g}"
                struct_cfg = dict(variant.get("struct", {}))
                struct_cfg["coord_sigma_cap"] = float(cap)
                struct_cfg["aux_weight"] = float(aux)
                variant["struct"] = struct_cfg

                exp_path = tmp_dir / f"{variant['name']}.yaml"
                exp_path.write_text(yaml.safe_dump(variant), encoding="utf-8")

                cmd = [
                    sys.executable,
                    str(cv_script),
                    "--exp",
                    str(exp_path),
                    "--data-config",
                    args.data_config,
                    "--train-config",
                    args.train_config,
                    "--output-root",
                    str(args.cv_output_root),
                    "--calibration",
                    args.calibration,
                ]
                if args.dataset_path:
                    cmd.extend(["--dataset-path", args.dataset_path])
                subprocess.run(cmd, check=True)

                cv_summary_path = (
                    Path(args.cv_output_root)
                    / variant["name"]
                    / args.calibration
                    / "cv_summary.json"
                )
                summary = json.loads(cv_summary_path.read_text(encoding="utf-8"))
                row = {
                    "experiment": str(variant["name"]),
                    "coord_sigma_cap": float(cap),
                    "aux_weight": float(aux),
                    "raw_pr_auc_mean": summary.get("raw_pr_auc_mean", np.nan),
                    "calibrated_pr_auc_mean": summary.get("calibrated_pr_auc_mean", np.nan),
                    "low_sigma_mean_pr_auc_mean": summary.get("low_sigma_mean_pr_auc_mean", np.nan),
                    "guidance_gate_pass_rate": summary.get("guidance_gate_pass_rate", np.nan),
                }
                rows.append(row)

    if not rows:
        raise RuntimeError("No sweep rows were generated")

    df = pd.DataFrame(rows)
    score_col = args.selection_metric
    if score_col not in df.columns:
        raise ValueError(f"selection_metric '{score_col}' not found in sweep results")
    df = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    df.to_parquet(output_dir / "stability_sweep.parquet", index=False)
    write_json(
        output_dir / "stability_sweep_best.json",
        {"selection_metric": score_col, "best": df.iloc[0].to_dict()},
    )
    print(f"Saved sweep leaderboard to {output_dir / 'stability_sweep.parquet'}")
    print(f"Best config: {df.iloc[0].to_dict()}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Sweep exp09 stability hyperparameters with CV selection."
    )
    parser.add_argument("--exp", default="configs/experiments/exp09_struct_egnn_noise.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--train-config", default="configs/train_cv.yaml")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--coord-sigma-caps", default="8,12,16")
    parser.add_argument("--aux-weights", default="0.1,0.2,0.3")
    parser.add_argument("--calibration", choices=["isotonic", "platt", "none"], default="isotonic")
    parser.add_argument("--selection-metric", default="low_sigma_mean_pr_auc_mean")
    parser.add_argument("--cv-output-root", default="artifacts/cv")
    parser.add_argument("--output-root", default="artifacts/sweeps")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
