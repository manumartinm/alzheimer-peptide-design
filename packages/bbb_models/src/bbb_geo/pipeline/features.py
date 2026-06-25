from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from bbb_geo.features.struct_loader import (
    build_struct_batch,
    load_struct_manifest,
    merge_dataset_with_manifest,
    plddt_sample_weight,
)


def build_features(
    df: pd.DataFrame, data_cfg: dict[str, Any], exp_cfg: dict[str, Any]
) -> dict[str, Any]:
    if exp_cfg.get("model_type") != "struct_egnn_geo":
        raise ValueError(f"Unsupported geo model_type: {exp_cfg.get('model_type')}")

    seq_col = data_cfg["sequence_col"]
    struct_cfg = exp_cfg.get("struct", {})
    manifest_path = struct_cfg.get("manifest_path") or data_cfg.get("struct_manifest_path")
    merged = df.copy()
    if "coords_path" not in merged.columns:
        if "structure_coords_path" in merged.columns:
            dataset_root = Path(str(data_cfg.get("dataset_root", "."))).resolve()
            merged["coords_path"] = merged["structure_coords_path"].map(
                lambda p: str((dataset_root / str(p)).resolve()) if pd.notna(p) else None
            )
        else:
            if not manifest_path:
                raise ValueError(
                    "struct manifest_path required when dataset lacks coords_path columns"
                )
            manifest = load_struct_manifest(manifest_path)
            merged = merge_dataset_with_manifest(df, manifest, sequence_col=seq_col)

    struct_df, struct_samples = build_struct_batch(
        merged,
        seq_col,
        radius=float(struct_cfg.get("radius", 10.0)),
        num_rbf=int(struct_cfg.get("num_rbf", 16)),
        sigma=0.0,
    )

    out: dict[str, Any] = {
        "struct_df": struct_df,
        "struct_samples": struct_samples,
        "tab_cols": [],
        "tab": None,
        "esm": None,
        "feat3d": None,
        "graphs": None,
    }

    return out


def apply_plddt_weights(train_df: pd.DataFrame, exp_cfg: dict[str, Any]) -> pd.DataFrame:
    if "plddt" not in train_df.columns:
        return train_df
    floor = float(exp_cfg.get("struct", {}).get("plddt_weight_floor", 0.1))
    out = train_df.copy()
    base = out["sample_weight"].astype(float) if "sample_weight" in out.columns else 1.0
    out["sample_weight"] = base * out["plddt"].map(lambda v: plddt_sample_weight(v, floor=floor))
    return out
