from __future__ import annotations

import torch

from .struct_graph import CHARGE_PH7, KD_HYDRO


def per_residue_hydrophobicity(
    sequence: str, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    values = [KD_HYDRO.get(aa, 0.0) for aa in sequence.upper()]
    return torch.tensor(values, dtype=dtype, device=device)


def per_residue_charge(sequence: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    values = [CHARGE_PH7.get(aa, 0.0) for aa in sequence.upper()]
    return torch.tensor(values, dtype=dtype, device=device)


def hydrophobic_moment(coords: torch.Tensor, hydro: torch.Tensor) -> torch.Tensor:
    """3D hydrophobic moment vector (Eisenberg-style) from coords and per-residue hydrophobicity."""
    if coords.shape[0] == 0:
        return torch.zeros(3, dtype=coords.dtype, device=coords.device)
    center = coords.mean(dim=0, keepdim=True)
    rel = coords - center
    weighted = rel * hydro.unsqueeze(-1)
    return weighted.sum(dim=0)


def amphipathicity_score(coords: torch.Tensor, hydro: torch.Tensor) -> torch.Tensor:
    """Scalar amphipathicity = || hydrophobic moment ||."""
    return torch.linalg.norm(hydrophobic_moment(coords, hydro))


def membrane_potential_energy(
    coords: torch.Tensor,
    sequence: str,
    *,
    maximize: bool = True,
) -> torch.Tensor:
    """Differentiable membrane/amphipathicity potential (higher = more amphipathic face)."""
    hydro = per_residue_hydrophobicity(sequence, device=coords.device, dtype=coords.dtype)
    score = amphipathicity_score(coords, hydro)
    return score if maximize else -score


def radius_of_gyration(coords: torch.Tensor) -> torch.Tensor:
    if coords.shape[0] == 0:
        return torch.zeros((), dtype=coords.dtype, device=coords.device)
    center = coords.mean(dim=0, keepdim=True)
    rel = coords - center
    return torch.sqrt((rel.pow(2).sum(dim=-1)).mean())


def helical_fraction_proxy(coords: torch.Tensor, threshold: float = 3.8) -> torch.Tensor:
    """Proxy for helical content from consecutive CA distances."""
    if coords.shape[0] < 4:
        return torch.zeros((), dtype=coords.dtype, device=coords.device)
    d = torch.linalg.norm(coords[1:] - coords[:-1], dim=-1)
    return (torch.abs(d - 3.8) <= threshold).float().mean()
