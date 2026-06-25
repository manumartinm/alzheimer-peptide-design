from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import mlp


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
