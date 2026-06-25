from __future__ import annotations

import torch
import torch.nn as nn


class CommonLatentFusion(nn.Module):
    def __init__(self, d_esm: int, d_tab: int, d_latent: int = 256, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.esm = nn.Sequential(nn.LayerNorm(d_esm), nn.Linear(d_esm, d_latent), nn.GELU())
        self.tab = nn.Sequential(nn.LayerNorm(d_tab), nn.Linear(d_tab, d_latent), nn.ReLU())
        self.gate = nn.Sequential(nn.Linear(d_latent, d_latent // 2), nn.GELU(), nn.Linear(d_latent // 2, 1))
        self.cls = nn.Sequential(
            nn.LayerNorm(d_latent),
            nn.Linear(d_latent, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, esm: torch.Tensor, tab: torch.Tensor, **kwargs) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h_esm = self.esm(esm)
        h_tab = self.tab(tab)
        h = torch.stack([h_esm, h_tab], dim=1)
        w = torch.softmax(self.gate(h).squeeze(-1), dim=1)
        z = (h * w.unsqueeze(-1)).sum(dim=1)
        return self.cls(z), {"h_esm": h_esm, "h_tab": h_tab, "z": z, "w": w}
