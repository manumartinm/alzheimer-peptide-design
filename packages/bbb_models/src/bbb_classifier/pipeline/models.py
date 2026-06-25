from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from bbb_classifier.models import (
    ESMLGBMModel,
    ESMTab3DFeatModel,
    ESMTabGNNModel,
    ESMTabMLP,
    TabularLGBMModel,
)
from bbb_classifier.train.engine import TorchData, predict_torch, train_torch_model
from bbb_classifier.utils.io import ensure_dir


def fit_and_predict(
    model_type: str,
    tr_feat: dict[str, Any],
    va_feat: dict[str, Any],
    y_train: np.ndarray,
    y_val: np.ndarray,
    sample_weight: np.ndarray,
    run_dir: Path,
    train_cfg: dict[str, Any],
    exp_cfg: dict[str, Any],
) -> np.ndarray:
    if model_type == "tabular_lgbm":
        model = TabularLGBMModel(random_state=int(train_cfg.get("seed", 42)))
        model.fit(tr_feat["tab"], y_train)
        val_prob = model.predict_proba(va_feat["tab"])
        ensure_dir(run_dir / "checkpoints")
        model.save(run_dir / "checkpoints" / "best.pkl")
        model.save(run_dir / "checkpoints" / "last.pkl")
        return val_prob

    if model_type == "esm_lgbm":
        model = ESMLGBMModel(random_state=int(train_cfg.get("seed", 42)))
        model.fit(tr_feat["esm"], y_train)
        val_prob = model.predict_proba(va_feat["esm"])
        ensure_dir(run_dir / "checkpoints")
        model.save(run_dir / "checkpoints" / "best.pkl")
        model.save(run_dir / "checkpoints" / "last.pkl")
        return val_prob

    model_cfg = exp_cfg.get("model", {})
    if model_type == "esm_tab_mlp":
        model = ESMTabMLP(
            d_esm=tr_feat["esm"].shape[1],
            d_tab=tr_feat["tab"].shape[1],
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            dropout=float(model_cfg.get("dropout", 0.2)),
        )
    elif model_type == "esm_tab_3dfeat":
        model = ESMTab3DFeatModel(
            d_esm=tr_feat["esm"].shape[1],
            d_tab=tr_feat["tab"].shape[1],
            d_3d=tr_feat["feat3d"].shape[1] if tr_feat["feat3d"] is not None else 0,
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            dropout=float(model_cfg.get("dropout", 0.2)),
        )
    elif model_type == "esm_tab_gnn":
        model = ESMTabGNNModel(
            d_esm=tr_feat["esm"].shape[1],
            d_tab=tr_feat["tab"].shape[1],
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            dropout=float(model_cfg.get("dropout", 0.2)),
            gnn_hidden_dim=int(model_cfg.get("gnn_hidden_dim", 64)),
            gnn_layers=int(model_cfg.get("gnn_layers", 2)),
        )
    else:
        raise ValueError(f"Unknown classifier model_type: {model_type}")

    torch_train_cfg = dict(train_cfg.get("training", {}))
    torch_train_cfg["save_periodic_every"] = int(
        train_cfg.get("output", {}).get("save_periodic_every", 0)
    )
    torch_train_cfg["mixup"] = exp_cfg.get("mixup", {})
    tr_data = TorchData(
        y=y_train,
        tab=tr_feat["tab"],
        esm=tr_feat["esm"],
        feat3d=tr_feat["feat3d"],
        sample_weight=sample_weight,
        graphs=tr_feat["graphs"],
    )
    va_data = TorchData(
        y=y_val,
        tab=va_feat["tab"],
        esm=va_feat["esm"],
        feat3d=va_feat["feat3d"],
        graphs=va_feat["graphs"],
    )
    tracking_cfg = train_cfg.get("tracking", {})
    model, _, val_prob = train_torch_model(
        model=model,
        train_data=tr_data,
        val_data=va_data,
        run_dir=run_dir,
        train_cfg=torch_train_cfg,
        tracking_cfg=tracking_cfg,
        primary_metric=train_cfg.get("primary_metric", "pr_auc"),
        maximize_metric=bool(train_cfg.get("maximize_metric", True)),
        resume=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return predict_torch(
        model, va_data, batch_size=int(torch_train_cfg.get("batch_size", 128)), device=device
    )
