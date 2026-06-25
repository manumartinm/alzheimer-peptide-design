from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from bbb_classifier.features.esm_embed import batch_esm_embeddings
from bbb_classifier.features.graph3d import sequence_graph
from bbb_classifier.features.struct3d import feature_matrix_3d
from bbb_classifier.features.tabular import tabular_matrix
from bbb_classifier.infer.rank import rank_candidates
from bbb_classifier.models import (
    ESMLGBMModel,
    ESMTab3DFeatModel,
    ESMTabGNNModel,
    ESMTabMLP,
    TabularLGBMModel,
)
from bbb_classifier.pipeline._common import CLASSIFIER_MODEL_TYPES
from bbb_classifier.train.calibration import ProbabilityCalibrator
from bbb_classifier.train.checkpoints import load_checkpoint
from bbb_classifier.train.engine import TorchData, predict_torch


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)


def _write_table(df: pd.DataFrame, path: Path) -> None:
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)


def run(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    meta = json.loads((run_dir / "train_metadata.json").read_text(encoding="utf-8"))
    exp_cfg, data_cfg = meta["exp_cfg"], meta["data_cfg"]
    model_type = exp_cfg["model_type"]
    if model_type not in CLASSIFIER_MODEL_TYPES:
        raise ValueError(f"{model_type} is not a classifier model; use scripts/geo/predict.py")

    sequence_col = data_cfg["sequence_col"]
    df = _read_table(Path(args.input))
    tab_cols = [c for c in meta.get("tab_cols", []) if c in df.columns]
    tab = tabular_matrix(df, tab_cols) if exp_cfg["features"].get("use_tabular", False) else None
    esm = (
        batch_esm_embeddings(
            df[sequence_col].astype(str).tolist(),
            dim=int(exp_cfg.get("model", {}).get("esm_dim", 128)),
        )
        if exp_cfg["features"].get("use_esm", False)
        else None
    )
    feat3d = (
        feature_matrix_3d(df, data_cfg.get("three_d_columns", []))
        if exp_cfg["features"].get("use_3d", False)
        else None
    )
    graphs = (
        [sequence_graph(s) for s in df[sequence_col].astype(str).tolist()]
        if exp_cfg["features"].get("use_gnn", False)
        else None
    )

    if model_type == "tabular_lgbm":
        prob = TabularLGBMModel.load(run_dir / "checkpoints" / "best.pkl").predict_proba(tab)
    elif model_type == "esm_lgbm":
        prob = ESMLGBMModel.load(run_dir / "checkpoints" / "best.pkl").predict_proba(esm)
    else:
        if model_type == "esm_tab_mlp":
            model = ESMTabMLP(d_esm=esm.shape[1], d_tab=tab.shape[1])
        elif model_type == "esm_tab_3dfeat":
            model = ESMTab3DFeatModel(
                d_esm=esm.shape[1],
                d_tab=tab.shape[1],
                d_3d=feat3d.shape[1] if feat3d is not None else 0,
            )
        elif model_type == "esm_tab_gnn":
            model = ESMTabGNNModel(d_esm=esm.shape[1], d_tab=tab.shape[1])
        else:
            raise ValueError(model_type)
        state = load_checkpoint(run_dir / "checkpoints" / "best.ckpt")
        model.load_state_dict(state["model"])
        td = TorchData(
            y=df.get(data_cfg["label_col"], pd.Series([0] * len(df))).to_numpy(),
            tab=tab,
            esm=esm,
            feat3d=feat3d,
            graphs=graphs,
        )
        prob = predict_torch(
            model,
            td,
            batch_size=128,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

    p_cal = prob
    calib_path = run_dir / "calibrators" / "calibrator.pkl"
    if calib_path.exists():
        p_cal = ProbabilityCalibrator.load(str(calib_path)).predict(prob)

    out_df = df.copy()
    out_df["p_bbb_raw"] = prob
    out_df["p_bbb_calibrated"] = p_cal
    out_df["decision"] = (out_df["p_bbb_calibrated"] >= args.threshold).astype(int)
    if args.top_k > 0:
        out_df = rank_candidates(out_df, prob_col="p_bbb_calibrated", top_k=args.top_k)
    _write_table(out_df, Path(args.output))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Predict BBB probabilities with a classifier run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args(argv)
    run(args)
    print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
