from __future__ import annotations

import numpy as np
import torch

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_ID = {a: i for i, a in enumerate(AA)}

# Kyte-Doolittle hydrophobicity (canonical order above).
KD_HYDRO = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}

# Charge at pH 7.4 (rough per-residue scalar).
CHARGE_PH7 = {
    "D": -1.0,
    "E": -1.0,
    "K": 1.0,
    "R": 1.0,
    "H": 0.1,
}


def _node_features(sequence: str) -> np.ndarray:
    feats = []
    for aa in sequence.upper():
        one_hot = np.zeros(len(AA), dtype=np.float32)
        idx = AA_TO_ID.get(aa)
        if idx is not None:
            one_hot[idx] = 1.0
        kd = KD_HYDRO.get(aa, 0.0)
        charge = CHARGE_PH7.get(aa, 0.0)
        feats.append(np.concatenate([one_hot, np.array([kd, charge], dtype=np.float32)]))
    return np.stack(feats, axis=0)


def radius_graph(coords: np.ndarray | torch.Tensor, radius: float = 10.0) -> torch.Tensor:
    if isinstance(coords, np.ndarray):
        coords_t = torch.tensor(coords, dtype=torch.float32)
    else:
        coords_t = coords
    n = coords_t.shape[0]
    if n <= 1:
        return torch.zeros((2, 0), dtype=torch.long)
    dist = torch.cdist(coords_t, coords_t)
    mask = (dist <= radius) & (dist > 0)
    src, dst = torch.where(mask)
    if src.numel() == 0 and n > 1:
        idx = torch.arange(n - 1)
        src = torch.cat([idx, idx + 1])
        dst = torch.cat([idx + 1, idx])
    return torch.stack([src, dst], dim=0).long()


def rbf_edge_features(
    coords: torch.Tensor,
    edge_index: torch.Tensor,
    num_rbf: int = 16,
    d_min: float = 0.0,
    d_max: float = 20.0,
) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.zeros((0, num_rbf), dtype=coords.dtype, device=coords.device)
    src, dst = edge_index
    rel = coords[src] - coords[dst]
    dist = torch.linalg.norm(rel, dim=-1, keepdim=True)
    centers = torch.linspace(d_min, d_max, num_rbf, device=coords.device, dtype=coords.dtype)
    gamma = 1.0 / max((d_max - d_min) / num_rbf, 1e-6) ** 2
    return torch.exp(-gamma * (dist - centers) ** 2)


def build_struct_graph(
    coords: np.ndarray,
    sequence: str,
    radius: float = 10.0,
    num_rbf: int = 16,
) -> dict[str, torch.Tensor]:
    """Build a differentiable-ready structural graph from backbone/Ca coordinates."""
    coords_t = torch.tensor(np.asarray(coords, dtype=np.float32), dtype=torch.float32)
    node_feats = torch.tensor(_node_features(sequence), dtype=torch.float32)
    edge_index = radius_graph(coords_t, radius=radius)
    edge_attr = rbf_edge_features(coords_t, edge_index, num_rbf=num_rbf)
    return {
        "coords": coords_t,
        "node_feats": node_feats,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "sequence": sequence,
    }


def apply_coord_noise(
    coords: torch.Tensor, sigma: float, generator: torch.Generator | None = None
) -> torch.Tensor:
    if sigma <= 0:
        return coords
    noise = torch.randn(coords.shape, generator=generator, dtype=coords.dtype, device=coords.device)
    out = coords + float(sigma) * noise
    if out.shape[0] > 0:
        out = out - out.mean(dim=0, keepdim=True)
    return out
