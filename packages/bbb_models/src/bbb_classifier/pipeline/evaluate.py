from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch

from bbb_classifier.features.esm_embed import batch_esm_embeddings
from bbb_classifier.features.graph3d import sequence_graph
from bbb_classifier.features.struct3d import feature_matrix_3d
from bbb_classifier.features.tabular import tabular_matrix
from bbb_classifier.models import ESMLGBMModel, ESMTab3DFeatModel, ESMTabGNNModel, ESMTabMLP, TabularLGBMModel
from bbb_classifier.pipeline._common import CLASSIFIER_MODEL_TYPES
from bbb_classifier.train.calibration import ProbabilityCalibrator
from bbb_classifier.train.checkpoints import load_checkpoint
from bbb_classifier.train.engine import TorchData, predict_torch
from bbb_classifier.train.metrics import classification_metrics
from bbb_classifier.utils.io import write_json


def run(args: argparse.Namespace) -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and int(os.environ.get("LOCAL_RANK", "0")) != 0:
        return

    run_dir = Path(args.run_dir)
    meta = json.loads((run_dir / "train_metadata.json").read_text(encoding="utf-8"))
    exp_cfg, data_cfg = meta["exp_cfg"], meta["data_cfg"]
    model_type = exp_cfg["model_type"]
    if model_type not in CLASSIFIER_MODEL_TYPES:
        raise ValueError(f"{model_type} is not a classifier model; use scripts/geo/train.py for evaluation via predict")

    df = pd.read_parquet(args.dataset_path)
    y = df[data_cfg["label_col"]].to_numpy(dtype=int)
    tab = tabular_matrix(df, meta.get("tab_cols", [])) if exp_cfg["features"].get("use_tabular", False) else None
    esm = (
        batch_esm_embeddings(df[data_cfg["sequence_col"]].astype(str).tolist(), dim=int(exp_cfg.get("model", {}).get("esm_dim", 128)))
        if exp_cfg["features"].get("use_esm", False)
        else None
    )
    feat3d = feature_matrix_3d(df, data_cfg.get("three_d_columns", [])) if exp_cfg["features"].get("use_3d", False) else None
    graphs = [sequence_graph(s) for s in df[data_cfg["sequence_col"]].astype(str).tolist()] if exp_cfg["features"].get("use_gnn", False) else None

    if model_type == "tabular_lgbm":
        prob = TabularLGBMModel.load(run_dir / "checkpoints" / "best.pkl").predict_proba(tab)
    elif model_type == "esm_lgbm":
        prob = ESMLGBMModel.load(run_dir / "checkpoints" / "best.pkl").predict_proba(esm)
    else:
        if model_type == "esm_tab_mlp":
            model = ESMTabMLP(d_esm=esm.shape[1], d_tab=tab.shape[1])
        elif model_type == "esm_tab_3dfeat":
            model = ESMTab3DFeatModel(d_esm=esm.shape[1], d_tab=tab.shape[1], d_3d=feat3d.shape[1] if feat3d is not None else 0)
        elif model_type == "esm_tab_gnn":
            model = ESMTabGNNModel(d_esm=esm.shape[1], d_tab=tab.shape[1])
        else:
            raise ValueError(model_type)
        state = load_checkpoint(run_dir / "checkpoints" / "best.ckpt")
        model.load_state_dict(state["model"])
        data = TorchData(y=y, tab=tab, esm=esm, feat3d=feat3d, graphs=graphs)
        prob = predict_torch(model, data, batch_size=128, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    p_cal = prob
    calib_path = run_dir / "calibrators" / "calibrator.pkl"
    if calib_path.exists():
        p_cal = ProbabilityCalibrator.load(str(calib_path)).predict(prob)

    metrics = {"raw": classification_metrics(y, prob), "calibrated": classification_metrics(y, p_cal)}
    write_json(run_dir / "evaluation_metrics.json", metrics)
    out_df = df[[data_cfg["sequence_col"], data_cfg["label_col"]]].copy()
    out_df["p_bbb_raw"] = prob
    out_df["p_bbb_calibrated"] = p_cal
    out_df.to_parquet(run_dir / "evaluation_predictions.parquet", index=False)
    print(metrics)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a classifier run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset-path", required=True)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
