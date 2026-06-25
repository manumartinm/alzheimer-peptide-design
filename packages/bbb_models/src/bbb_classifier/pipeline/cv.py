from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from bbb_classifier.data.hf_peptides import load_data_config
from bbb_classifier.utils.io import ensure_dir, read_yaml, write_json


def _reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers, accs = [], []
    for i in range(n_bins):
        left, right = bins[i], bins[i + 1]
        mask = (y_prob >= left) & (y_prob < right if i < n_bins - 1 else y_prob <= right)
        if not np.any(mask):
            continue
        centers.append(float(np.mean(y_prob[mask])))
        accs.append(float(np.mean(y_true[mask])))
    return np.array(centers), np.array(accs)


def run(args: argparse.Namespace) -> None:
    exp_cfg = read_yaml(args.exp)
    data_cfg = load_data_config(args.data_config, ensure=True)
    train_cfg = read_yaml(args.train_config)
    dataset_path = args.dataset_path or data_cfg["dataset_path"]
    df = pd.read_parquet(dataset_path)
    if "fold_id" not in df.columns:
        raise ValueError("Dataset must contain fold_id for cv.")

    train_cfg_mod = train_cfg.copy()
    train_cfg_mod["tracking"] = {"mlflow": False, "tensorboard": False}
    train_cfg_mod["calibration"] = dict(train_cfg.get("calibration", {}))
    if args.calibration == "none":
        train_cfg_mod["calibration"]["enabled"] = False
    else:
        train_cfg_mod["calibration"] = {"enabled": True, "method": args.calibration}

    exp_name = exp_cfg.get("name", "exp")
    cv_root = ensure_dir(Path(args.output_root) / exp_name / args.calibration)
    train_script = Path(__file__).resolve().parents[3] / "scripts" / "classifier" / "train.py"
    all_metrics, all_preds = [], []

    with tempfile.TemporaryDirectory(prefix="bbb_cv_") as tmp:
        tmp_dir = Path(tmp)
        train_cfg_tmp = tmp_dir / "train_cv.yaml"
        train_cfg_tmp.write_text(yaml.safe_dump(train_cfg_mod), encoding="utf-8")
        for fold in range(5):
            fold_df = df.copy()
            fold_df["fold_id"] = (fold_df["fold_id"] == fold).astype(int).replace({1: 0, 0: 1})
            fold_ds = tmp_dir / f"fold_{fold}.parquet"
            fold_df.to_parquet(fold_ds, index=False)
            run_root = cv_root / f"fold_{fold}"
            subprocess.run(
                [
                    sys.executable,
                    str(train_script),
                    "--exp",
                    args.exp,
                    "--data-config",
                    args.data_config,
                    "--train-config",
                    str(train_cfg_tmp),
                    "--output-root",
                    str(run_root),
                    "--dataset-path",
                    str(fold_ds),
                ],
                check=True,
            )
            run_dir = run_root / "models" / exp_name
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            metrics["fold"] = fold
            all_metrics.append(metrics)
            pred_df = pd.read_parquet(run_dir / "predictions" / "val_predictions.parquet")
            pred_df["fold"] = fold
            all_preds.append(pred_df)

    summary = {"experiment": exp_name, "calibration": args.calibration, "folds": all_metrics}
    for split in ("raw", "calibrated"):
        for key in ("pr_auc", "mcc", "brier", "roc_auc", "ece"):
            vals = [m[split][key] for m in all_metrics if key in m.get(split, {})]
            summary[f"{split}_{key}_mean"] = float(np.mean(vals))
            summary[f"{split}_{key}_std"] = float(np.std(vals))

    write_json(cv_root / "cv_summary.json", summary)
    preds = pd.concat(all_preds, ignore_index=True)
    preds.to_parquet(cv_root / "cv_predictions.parquet", index=False)

    y_true = preds[data_cfg["label_col"]].to_numpy(dtype=int)
    y_prob = (
        preds["p_bbb_calibrated"].to_numpy(dtype=float)
        if "p_bbb_calibrated" in preds.columns
        else preds["p_bbb_raw"].to_numpy(dtype=float)
    )
    x, y = _reliability_curve(y_true, y_prob)
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "--", label="ideal")
    if len(x) > 0:
        plt.plot(x, y, marker="o", label="model")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"Reliability ({exp_name}, {args.calibration})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(cv_root / "reliability.png", dpi=180)
    print(f"Saved CV summary to {cv_root / 'cv_summary.json'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="5-fold CV for classifier experiments.")
    parser.add_argument("--exp", required=True)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--output-root", default="artifacts/cv")
    parser.add_argument("--calibration", choices=["isotonic", "platt", "none"], default="isotonic")
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
