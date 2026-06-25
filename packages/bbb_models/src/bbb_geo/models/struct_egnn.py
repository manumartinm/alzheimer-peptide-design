from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from bbb_geo.features.membrane_potential import (
    amphipathicity_score,
    helical_fraction_proxy,
    per_residue_hydrophobicity,
    radius_of_gyration,
)


class EGNLayer(nn.Module):
    """Single E(n)-equivariant graph layer (Satorras et al., 2021)."""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if edge_index.numel() == 0:
            return h, x
        src, dst = edge_index
        rel = x[src] - x[dst]
        dist2 = (rel.pow(2)).sum(dim=-1, keepdim=True)
        edge_in = torch.cat([h[src], h[dst], edge_attr, dist2], dim=-1)
        m_ij = self.edge_mlp(edge_in)
        agg = torch.zeros(h.shape[0], m_ij.shape[1], dtype=h.dtype, device=h.device)
        agg.index_add_(0, dst, m_ij)
        h_out = h + self.node_mlp(torch.cat([h, agg], dim=-1))
        coord_weights = self.coord_mlp(m_ij)
        coord_delta = torch.zeros_like(x)
        coord_delta.index_add_(0, dst, rel * coord_weights)
        x_out = x + coord_delta
        return h_out, x_out


class SigmaEmbedding(nn.Module):
    def __init__(self, hidden_dim: int, sigma_data: float = 16.0):
        super().__init__()
        self.sigma_data = sigma_data
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        # Mirror BoltzGen c_noise: log(sigma / sigma_data) * 0.25
        if sigma.ndim == 0:
            sigma = sigma.reshape(1)
        noise_emb = torch.log(torch.clamp(sigma, min=1e-6) / self.sigma_data) * 0.25
        return self.mlp(noise_emb.unsqueeze(-1))


class StructEGNNBackbone(nn.Module):
    def __init__(
        self,
        node_dim: int = 22,
        edge_dim: int = 16,
        hidden_dim: int = 64,
        num_layers: int = 3,
        sigma_data: float = 16.0,
    ):
        super().__init__()
        self.node_in = nn.Linear(node_dim, hidden_dim)
        self.sigma_emb = SigmaEmbedding(hidden_dim, sigma_data=sigma_data)
        self.layers = nn.ModuleList(
            [EGNLayer(hidden_dim, edge_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.pool = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU()
        )

    def forward_graph(
        self,
        node_feats: torch.Tensor,
        coords: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        sigma: torch.Tensor | None = None,
        chem_dropout: float = 0.0,
    ) -> torch.Tensor:
        h = self.node_in(node_feats)
        if chem_dropout > 0 and self.training:
            mask = (
                torch.rand(node_feats.shape[0], 1, device=node_feats.device) > chem_dropout
            ).float()
            h = h * mask + self.node_in(torch.zeros_like(node_feats)) * (1.0 - mask)
        x = coords
        if sigma is not None:
            sigma_vec = self.sigma_emb(sigma.to(h.device)).reshape(1, -1)
            h = h + sigma_vec
        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_attr)
        pooled = h.mean(dim=0)
        return self.pool(pooled)


class StructEGNNGeo(nn.Module):
    """Geometry-only classifier p_geo(BBB | x, sigma) for diffusion guidance."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
        chem_dropout: float = 0.2,
        sigma_data: float = 16.0,
    ):
        super().__init__()
        self.chem_dropout = chem_dropout
        self.backbone = StructEGNNBackbone(
            node_dim=22,
            edge_dim=16,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            sigma_data=sigma_data,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.aux_head = nn.Linear(hidden_dim, 3)

    def forward(
        self,
        graphs: list[dict[str, torch.Tensor | str | float]] | None = None,
        chem_dropout: float | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if not graphs:
            device = kwargs.get("device", torch.device("cpu"))
            return torch.zeros((1, 1), device=device)
        logits = []
        for g in graphs:
            emb = self.backbone.forward_graph(
                g["node_feats"],
                g["coords"],
                g["edge_index"],
                g["edge_attr"],
                sigma=torch.tensor(float(g.get("sigma", 0.0)), device=g["coords"].device),
                chem_dropout=chem_dropout if chem_dropout is not None else self.chem_dropout,
            )
            logits.append(self.head(emb))
        return torch.stack(logits, dim=0)

    def forward_with_aux(
        self,
        graphs: list[dict[str, torch.Tensor | str | float]],
        chem_dropout: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        logits = []
        aux = []
        targets = {"amphipathicity": [], "rg": [], "helix": []}
        for g in graphs:
            coords = g["coords"]
            seq = str(g["sequence"])
            hydro = per_residue_hydrophobicity(seq, device=coords.device, dtype=coords.dtype)
            emb = self.backbone.forward_graph(
                g["node_feats"],
                coords,
                g["edge_index"],
                g["edge_attr"],
                sigma=torch.tensor(float(g.get("sigma", 0.0)), device=coords.device),
                chem_dropout=chem_dropout if chem_dropout is not None else self.chem_dropout,
            )
            logits.append(self.head(emb))
            aux_pred = self.aux_head(emb)
            aux.append(aux_pred)
            targets["amphipathicity"].append(torch.log1p(amphipathicity_score(coords, hydro)))
            targets["rg"].append(radius_of_gyration(coords) / 10.0)
            targets["helix"].append(helical_fraction_proxy(coords))
        logit_t = torch.stack(logits, dim=0)
        aux_t = torch.stack(aux, dim=0)
        target_t = torch.stack(
            [
                torch.stack(targets["amphipathicity"]),
                torch.stack(targets["rg"]),
                torch.stack(targets["helix"]),
            ],
            dim=-1,
        )
        return logit_t, aux_t, target_t

    def log_prob(self, graphs: list[dict[str, torch.Tensor | str | float]]) -> torch.Tensor:
        logits = self.forward(graphs=graphs)
        return F.logsigmoid(logits)


def sample_edm_sigma(
    batch_size: int,
    *,
    sigma_data: float = 16.0,
    sigma_min: float = 0.0004,
    sigma_max: float = 160.0,
    low_mid_bias: float = 0.7,
    coord_sigma_cap: float | None = 16.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Sample noise levels from a log-uniform EDM-like range, biased toward low-mid sigma.

    Returned values are in Angstroms (dimensionless sigma * sigma_data), matching BoltzGen AF3.
    coord_sigma_cap limits coordinate noise during EGNN training for numerical stability.
    """
    u = torch.rand(batch_size, device=device)
    biased = u ** (1.0 + low_mid_bias)
    log_min = math.log(sigma_min * sigma_data)
    log_max = math.log(sigma_max * sigma_data)
    sigmas = torch.exp(log_min + biased * (log_max - log_min))
    if coord_sigma_cap is not None:
        sigmas = torch.clamp(sigmas, max=float(coord_sigma_cap))
    return sigmas
