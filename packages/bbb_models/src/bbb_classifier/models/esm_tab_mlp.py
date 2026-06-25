from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import mlp


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
