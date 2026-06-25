from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bbb_classifier.data.splits import train_val_split
from bbb_classifier.train.calibration import ProbabilityCalibrator
from bbb_classifier.train.metrics import classification_metrics
from bbb_classifier.utils.io import ensure_dir, read_yaml, write_json
from bbb_classifier.utils.logging import get_logger
from bbb_classifier.utils.seed import set_seed

CLASSIFIER_MODEL_TYPES = frozenset(
    {
        "tabular_lgbm",
        "esm_lgbm",
        "esm_tab_mlp",
        "esm_tab_3dfeat",
        "esm_tab_gnn",
    }
)


def add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--exp", required=True, help="Experiment yaml path")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--output-root", default="artifacts")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument(
        "--no-resume", action="store_true", help="Start from scratch even if checkpoints exist"
    )


def load_context(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    exp_cfg = read_yaml(args.exp)
    data_cfg = read_yaml(args.data_config)
    train_cfg = read_yaml(args.train_config)
    run_name = exp_cfg.get("name", exp_cfg["model_type"])
    run_dir = ensure_dir(Path(args.output_root) / "models" / run_name)
    return exp_cfg, data_cfg, train_cfg, run_dir


def prepare_dataframes(
    args: argparse.Namespace,
    exp_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_path = args.dataset_path or data_cfg["dataset_path"]
    df = pd.read_parquet(dataset_path)
    label_col = data_cfg["label_col"]
    train_df, val_df = train_val_split(
        df=df,
        label_col=label_col,
        test_size=float(data_cfg.get("test_size", 0.2)),
        random_state=int(data_cfg.get("random_state", 42)),
        fold_col=data_cfg.get("fold_col"),
    )
    return train_df.copy(), val_df.copy()


def sample_weights(train_df: pd.DataFrame) -> np.ndarray:
    if "sample_weight" in train_df.columns:
        return train_df["sample_weight"].fillna(1.0).to_numpy(dtype=np.float32)
    return np.ones(len(train_df), dtype=np.float32)


def finalize_run(
    *,
    run_dir: Path,
    exp_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    tab_cols: list[str],
    val_df: pd.DataFrame,
    y_val: np.ndarray,
    val_prob: np.ndarray,
    logger,
) -> None:
    run_name = exp_cfg.get("name", exp_cfg["model_type"])
    model_type = exp_cfg["model_type"]
    label_col = data_cfg["label_col"]
    cal_cfg = train_cfg.get("calibration", {})
    p_cal = val_prob
    if cal_cfg.get("enabled", True):
        calibrator = ProbabilityCalibrator(method=cal_cfg.get("method", "isotonic"))
        calibrator.fit(val_prob, y_val)
        p_cal = calibrator.predict(val_prob)
        ensure_dir(run_dir / "calibrators")
        calibrator.save(str(run_dir / "calibrators" / "calibrator.pkl"))

    metrics = classification_metrics(y_val, val_prob)
    cal_metrics = classification_metrics(y_val, p_cal)
    write_json(
        run_dir / "metrics.json",
        {"raw": metrics, "calibrated": cal_metrics, "run_name": run_name, "model_type": model_type},
    )

    pred_df = val_df[[data_cfg["sequence_col"], label_col]].copy()
    pred_df["p_bbb_raw"] = val_prob
    pred_df["p_bbb_calibrated"] = p_cal
    pred_df["decision"] = (pred_df["p_bbb_calibrated"] >= 0.5).astype(int)
    ensure_dir(run_dir / "predictions")
    pred_df.to_parquet(run_dir / "predictions" / "val_predictions.parquet", index=False)
    write_json(
        run_dir / "train_metadata.json",
        {"exp_cfg": exp_cfg, "data_cfg": data_cfg, "train_cfg": train_cfg, "tab_cols": tab_cols},
    )
    logger.info("Training completed for %s", run_name)
    logger.info("Metrics raw=%s calibrated=%s", metrics, cal_metrics)


def setup_run(args: argparse.Namespace) -> tuple[dict, dict, dict, Path, Any]:
    exp_cfg, data_cfg, train_cfg, run_dir = load_context(args)
    set_seed(int(train_cfg.get("seed", 42)))
    logger = get_logger("bbb-classifier", run_dir / "train.log")
    model_type = exp_cfg["model_type"]
    if model_type not in CLASSIFIER_MODEL_TYPES:
        raise ValueError(f"{model_type} is not a classifier model; use scripts/geo/train.py")
    return exp_cfg, data_cfg, train_cfg, run_dir, logger
