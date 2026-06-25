from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn as nn

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

from sklearn.ensemble import HistGradientBoostingClassifier

from bbb_classifier.enums import ModelType
from bbb_classifier.features import FeatureBundle


def mlp(d_in: int, d_hidden: int, d_out: int, dropout: float = 0.2) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_in, d_hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(d_hidden, d_hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(d_hidden, d_out),
    )


class CommonLatentFusion(nn.Module):
    def __init__(
        self,
        d_esm: int,
        d_tab: int,
        d_latent: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.esm = nn.Sequential(nn.LayerNorm(d_esm), nn.Linear(d_esm, d_latent), nn.GELU())
        self.tab = nn.Sequential(nn.LayerNorm(d_tab), nn.Linear(d_tab, d_latent), nn.ReLU())
        self.gate = nn.Sequential(
            nn.Linear(d_latent, d_latent // 2), nn.GELU(), nn.Linear(d_latent // 2, 1)
        )
        self.cls = nn.Sequential(
            nn.LayerNorm(d_latent),
            nn.Linear(d_latent, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, esm: torch.Tensor, tab: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h_esm = self.esm(esm)
        h_tab = self.tab(tab)
        h = torch.stack([h_esm, h_tab], dim=1)
        w = torch.softmax(self.gate(h).squeeze(-1), dim=1)
        z = (h * w.unsqueeze(-1)).sum(dim=1)
        return self.cls(z), {"h_esm": h_esm, "h_tab": h_tab, "z": z, "w": w}


class TabularLGBMModel:
    def __init__(self, random_state: int = 42):
        if LGBMClassifier is not None:
            self.model = LGBMClassifier(
                n_estimators=600,
                learning_rate=0.03,
                max_depth=-1,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=random_state,
            )
        else:
            self.model = HistGradientBoostingClassifier(random_state=random_state)

    def fit(self, x_tab: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(x_tab, y)

    def predict_proba(self, x_tab: np.ndarray) -> np.ndarray:
        p = self.model.predict_proba(x_tab)
        return p[:, 1] if p.ndim == 2 else p

    def save(self, path: str | Path) -> None:
        joblib.dump(self.model, path)

    @staticmethod
    def load(path: str | Path) -> TabularLGBMModel:
        wrapper = TabularLGBMModel()
        wrapper.model = joblib.load(path)
        return wrapper


class ESMLGBMModel:
    def __init__(self, random_state: int = 42):
        if LGBMClassifier is not None:
            self.model = LGBMClassifier(
                n_estimators=800,
                learning_rate=0.025,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=random_state,
            )
        else:
            self.model = HistGradientBoostingClassifier(random_state=random_state)

    def fit(self, x_esm: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(x_esm, y)

    def predict_proba(self, x_esm: np.ndarray) -> np.ndarray:
        p = self.model.predict_proba(x_esm)
        return p[:, 1] if p.ndim == 2 else p

    def save(self, path: str | Path) -> None:
        joblib.dump(self.model, path)

    @staticmethod
    def load(path: str | Path) -> ESMLGBMModel:
        wrapper = ESMLGBMModel()
        wrapper.model = joblib.load(path)
        return wrapper


class ESMTabMLP(nn.Module):
    def __init__(self, d_esm: int, d_tab: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.esm_proj = nn.Sequential(
            nn.LayerNorm(d_esm), nn.Linear(d_esm, hidden_dim // 2), nn.GELU()
        )
        self.tab_proj = nn.Sequential(
            nn.LayerNorm(d_tab), nn.Linear(d_tab, hidden_dim // 2), nn.ReLU()
        )
        self.head = mlp(hidden_dim, hidden_dim, 1, dropout=dropout)

    def forward(self, esm: torch.Tensor, tab: torch.Tensor, **kwargs) -> torch.Tensor:
        z = torch.cat([self.esm_proj(esm), self.tab_proj(tab)], dim=-1)
        return self.head(z)


class ESMTab3DFeatModel(nn.Module):
    def __init__(
        self, d_esm: int, d_tab: int, d_3d: int, hidden_dim: int = 256, dropout: float = 0.2
    ):
        super().__init__()
        self.esm_proj = nn.Sequential(
            nn.LayerNorm(d_esm), nn.Linear(d_esm, hidden_dim // 3), nn.GELU()
        )
        self.tab_proj = nn.Sequential(
            nn.LayerNorm(d_tab), nn.Linear(d_tab, hidden_dim // 3), nn.ReLU()
        )
        self.feat3d_proj = nn.Sequential(
            nn.LayerNorm(max(d_3d, 1)),
            nn.Linear(max(d_3d, 1), hidden_dim // 3),
            nn.ReLU(),
        )
        self.head = mlp(hidden_dim, hidden_dim, 1, dropout=dropout)

    def forward(
        self, esm: torch.Tensor, tab: torch.Tensor, feat3d: torch.Tensor | None = None, **kwargs
    ) -> torch.Tensor:
        if feat3d is None or feat3d.shape[1] == 0:
            feat3d = torch.zeros((esm.shape[0], 1), dtype=esm.dtype, device=esm.device)
        z = torch.cat([self.esm_proj(esm), self.tab_proj(tab), self.feat3d_proj(feat3d)], dim=-1)
        return self.head(z)


class SimpleGraphConv(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.lin_self = nn.Linear(d_in, d_out)
        self.lin_neigh = nn.Linear(d_in, d_out)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h_self = self.lin_self(x)
        if edge_index.numel() == 0:
            return torch.relu(h_self)
        src, dst = edge_index
        agg = torch.zeros_like(h_self)
        agg.index_add_(0, dst, self.lin_neigh(x[src]))
        deg = torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)
        deg.index_add_(0, dst, torch.ones((src.shape[0], 1), dtype=x.dtype, device=x.device))
        agg = agg / torch.clamp(deg, min=1.0)
        return torch.relu(h_self + agg)


class ESMTabGNNModel(nn.Module):
    def __init__(
        self,
        d_esm: int,
        d_tab: int,
        gnn_in_dim: int = 20,
        gnn_hidden_dim: int = 64,
        gnn_layers: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.esm_proj = nn.Sequential(
            nn.LayerNorm(d_esm), nn.Linear(d_esm, hidden_dim // 3), nn.GELU()
        )
        self.tab_proj = nn.Sequential(
            nn.LayerNorm(d_tab), nn.Linear(d_tab, hidden_dim // 3), nn.ReLU()
        )
        gnn_stack = [
            SimpleGraphConv(gnn_in_dim if i == 0 else gnn_hidden_dim, gnn_hidden_dim)
            for i in range(gnn_layers)
        ]
        self.gnn_layers = nn.ModuleList(gnn_stack)
        self.gnn_proj = nn.Sequential(
            nn.LayerNorm(gnn_hidden_dim), nn.Linear(gnn_hidden_dim, hidden_dim // 3), nn.ReLU()
        )
        self.head = mlp(hidden_dim, hidden_dim, 1, dropout=dropout)

    def _graph_batch_embedding(
        self, graphs: list[dict[str, torch.Tensor]], device: torch.device
    ) -> torch.Tensor:
        graph_embs = []
        for g in graphs:
            x = g["x"].to(device)
            edge_index = g["edge_index"].to(device)
            for layer in self.gnn_layers:
                x = layer(x, edge_index)
            graph_embs.append(torch.mean(x, dim=0))
        return torch.stack(graph_embs, dim=0)

    def forward(
        self,
        esm: torch.Tensor,
        tab: torch.Tensor,
        graphs: list[dict[str, torch.Tensor]] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if not graphs:
            g_emb = torch.zeros(
                (esm.shape[0], self.gnn_proj[0].normalized_shape[0]),
                device=esm.device,
                dtype=esm.dtype,
            )
        else:
            g_emb = self._graph_batch_embedding(graphs, esm.device)
        z = torch.cat([self.esm_proj(esm), self.tab_proj(tab), self.gnn_proj(g_emb)], dim=-1)
        return self.head(z)


class ModelRegistry:
    @staticmethod
    def create_torch(
        model_type: ModelType, bundle: FeatureBundle, model_cfg: dict[str, Any]
    ) -> nn.Module:
        d_esm = bundle.esm.shape[1] if bundle.esm is not None else 0
        d_tab = bundle.tab.shape[1] if bundle.tab is not None else 0
        hidden_dim = int(model_cfg.get("hidden_dim", 256))
        dropout = float(model_cfg.get("dropout", 0.2))

        if model_type == ModelType.ESM_TAB_MLP:
            return ESMTabMLP(d_esm=d_esm, d_tab=d_tab, hidden_dim=hidden_dim, dropout=dropout)
        if model_type == ModelType.ESM_TAB_3DFEAT:
            d_3d = bundle.feat3d.shape[1] if bundle.feat3d is not None else 0
            return ESMTab3DFeatModel(
                d_esm=d_esm, d_tab=d_tab, d_3d=d_3d, hidden_dim=hidden_dim, dropout=dropout
            )
        if model_type == ModelType.ESM_TAB_GNN:
            return ESMTabGNNModel(
                d_esm=d_esm,
                d_tab=d_tab,
                hidden_dim=hidden_dim,
                dropout=dropout,
                gnn_hidden_dim=int(model_cfg.get("gnn_hidden_dim", 64)),
                gnn_layers=int(model_cfg.get("gnn_layers", 2)),
            )
        raise ValueError(f"Unknown torch model type: {model_type}")

    @staticmethod
    def load_for_inference(
        model_type: ModelType,
        run_dir: Path,
        bundle: FeatureBundle,
        model_cfg: dict[str, Any],
    ) -> TabularLGBMModel | ESMLGBMModel | nn.Module:
        if model_type == ModelType.TABULAR_LGBM:
            return TabularLGBMModel.load(run_dir / "checkpoints" / "best.pkl")
        if model_type == ModelType.ESM_LGBM:
            return ESMLGBMModel.load(run_dir / "checkpoints" / "best.pkl")
        from bbb_classifier.training import load_checkpoint

        model = ModelRegistry.create_torch(model_type, bundle, model_cfg)
        state = load_checkpoint(run_dir / "checkpoints" / "best.ckpt")
        model.load_state_dict(state["model"])
        return model

    @staticmethod
    def predict_proba(
        model_type: ModelType,
        model: Any,
        bundle: FeatureBundle,
        *,
        y_placeholder: np.ndarray | None = None,
        batch_size: int = 128,
    ) -> np.ndarray:
        if model_type == ModelType.TABULAR_LGBM:
            return model.predict_proba(bundle.tab)
        if model_type == ModelType.ESM_LGBM:
            return model.predict_proba(bundle.esm)
        from bbb_classifier.training import TorchData, predict_torch

        y = y_placeholder if y_placeholder is not None else np.zeros(len(bundle.df))
        td = TorchData(
            y=y,
            tab=bundle.tab,
            esm=bundle.esm,
            feat3d=bundle.feat3d,
            graphs=bundle.graphs,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return predict_torch(model, td, batch_size=batch_size, device=device)
