from __future__ import annotations

import torch


def min_distance_to_region(
    peptide_coords: torch.Tensor,
    region_coords: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return minimum distance for each region atom to peptide atoms."""
    if peptide_coords.numel() == 0 or region_coords.numel() == 0:
        return torch.empty(0, device=peptide_coords.device)
    dmat = torch.cdist(peptide_coords.unsqueeze(0), region_coords.unsqueeze(0)).squeeze(0)
    return torch.min(dmat, dim=0).values + eps


def hotspot_score(
    peptide_coords: torch.Tensor,
    hotspot_coords: torch.Tensor,
    cutoff: float = 5.0,
    alpha: float = 8.0,
) -> torch.Tensor:
    """Differentiable hotspot contact score in [0, 1]."""
    min_d = min_distance_to_region(peptide_coords, hotspot_coords)
    if min_d.numel() == 0:
        return torch.tensor(0.0, device=peptide_coords.device)
    return torch.sigmoid(alpha * (cutoff - min_d)).mean()


def atp_repulsion_score(
    peptide_coords: torch.Tensor,
    atp_coords: torch.Tensor,
    sigma: float = 3.0,
) -> torch.Tensor:
    """Lennard-Jones-like soft repulsion score."""
    min_d = min_distance_to_region(peptide_coords, atp_coords)
    if min_d.numel() == 0:
        return torch.tensor(0.0, device=peptide_coords.device)
    return ((sigma / min_d) ** 12).mean()


def gate_g1_fraction(
    peptide_coords: torch.Tensor,
    hotspot_coords: torch.Tensor,
    threshold_angstrom: float = 5.0,
) -> float:
    """Hard Gate G1: fraction of hotspot atoms within threshold."""
    min_d = min_distance_to_region(peptide_coords, hotspot_coords)
    if min_d.numel() == 0:
        return 0.0
    return float((min_d <= threshold_angstrom).float().mean().item())

