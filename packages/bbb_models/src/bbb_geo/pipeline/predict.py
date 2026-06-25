from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from bbb_classifier.infer.rank import rank_candidates
from bbb_classifier.train.calibration import ProbabilityCalibrator
from bbb_classifier.train.engine import TorchData, predict_torch
from bbb_geo.features.struct_loader import (
    build_struct_sample,
    load_struct_manifest,
    merge_dataset_with_manifest,
)
from bbb_geo.models import StructEGNNGeo
from bbb_geo.pipeline._common import GEO_MODEL_TYPES
from bbb_geo.train.checkpoints import load_checkpoint


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
    if model_type not in GEO_MODEL_TYPES:
        raise ValueError(f"{model_type} is not a geo model; use scripts/classifier/predict.py")

    sequence_col = data_cfg["sequence_col"]
    df = _read_table(Path(args.input))
    struct_cfg = exp_cfg.get("struct", {})
    manifest_path = (
        args.manifest
        or exp_cfg.get("struct", {}).get("manifest_path")
        or data_cfg.get("struct_manifest_path")
    )

    work_df = df
    if manifest_path and "coords_path" not in df.columns:
        work_df = merge_dataset_with_manifest(
            df, load_struct_manifest(manifest_path), sequence_col=sequence_col
        )

    struct_samples = []
    valid_rows: list[pd.Series] = []
    for _, row in work_df.iterrows():
        if pd.isna(row.get("coords_path")):
            continue
        valid_rows.append(row)
        struct_samples.append(
            build_struct_sample(
                row["coords_path"],
                str(row[sequence_col]),
                radius=float(struct_cfg.get("radius", 10.0)),
                num_rbf=int(struct_cfg.get("num_rbf", 16)),
            )
        )
    if not struct_samples:
        raise ValueError("No structural samples found for prediction.")

    aligned_df = pd.DataFrame(valid_rows).reset_index(drop=True)
    model_cfg = exp_cfg.get("model", {})
    model = StructEGNNGeo(
        hidden_dim=int(model_cfg.get("egnn_hidden", 64)),
        num_layers=int(model_cfg.get("egnn_layers", 3)),
        dropout=float(model_cfg.get("dropout", 0.2)),
    )
    model.load_state_dict(load_checkpoint(run_dir / "checkpoints" / "best.ckpt")["model"])
    td = TorchData(
        y=aligned_df.get(data_cfg["label_col"], pd.Series([0] * len(struct_samples))).to_numpy(),
        tab=None,
        esm=None,
        feat3d=None,
        struct_samples=struct_samples,
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

    out_df = aligned_df.copy()
    out_df["p_bbb_raw"] = prob
    out_df["p_bbb_calibrated"] = p_cal
    out_df["decision"] = (out_df["p_bbb_calibrated"] >= args.threshold).astype(int)
    if args.top_k > 0:
        out_df = rank_candidates(out_df, prob_col="p_bbb_calibrated", top_k=args.top_k)
    _write_table(out_df, Path(args.output))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Predict BBB probabilities with a geo EGNN run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args(argv)
    run(args)
    print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
