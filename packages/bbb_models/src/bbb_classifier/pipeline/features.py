from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from bbb_classifier.features.esm_embed import batch_esm_embeddings
from bbb_classifier.features.graph3d import sequence_graph
from bbb_classifier.features.struct3d import feature_matrix_3d
from bbb_classifier.features.tabular import infer_tabular_columns, tabular_matrix


def build_features(
    df: pd.DataFrame, data_cfg: dict[str, Any], exp_cfg: dict[str, Any]
) -> dict[str, Any]:
    seq_col = data_cfg["sequence_col"]
    feats_cfg = exp_cfg.get("features", {})
    esm_cache_dir = exp_cfg.get("esm", {}).get("cache_dir", "artifacts/cache/esm2")
    tab_cols = infer_tabular_columns(df, data_cfg.get("tabular_exclude", []))
    x_tab = tabular_matrix(df, tab_cols) if feats_cfg.get("use_tabular", False) else None
    x_esm = (
        batch_esm_embeddings(
            df[seq_col].astype(str).tolist(),
            dim=int(exp_cfg.get("model", {}).get("esm_dim", 128)),
            cache_dir=esm_cache_dir,
        )
        if feats_cfg.get("use_esm", False)
        else None
    )
    x_3d = (
        feature_matrix_3d(df, data_cfg.get("three_d_columns", []))
        if feats_cfg.get("use_3d", False)
        else None
    )
    graphs = (
        [sequence_graph(s) for s in df[seq_col].astype(str).tolist()]
        if feats_cfg.get("use_gnn", False)
        else None
    )
    return {
        "tab": x_tab,
        "esm": x_esm,
        "feat3d": x_3d,
        "graphs": graphs,
        "tab_cols": tab_cols,
        "df": df,
    }


def select_rows(features: dict[str, Any], idx: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {"tab_cols": features["tab_cols"]}
    for key in ("tab", "esm", "feat3d"):
        out[key] = features[key][idx] if features.get(key) is not None else None
    if features.get("graphs") is not None:
        out["graphs"] = [features["graphs"][i] for i in idx]
    return out
