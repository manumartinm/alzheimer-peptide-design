from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import mlp


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
