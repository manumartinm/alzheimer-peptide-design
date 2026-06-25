from __future__ import annotations

from dataclasses import dataclass

import torch

from .potentials import atp_repulsion_score, hotspot_score


@dataclass
class GuidanceConfig:
    hotspot_weight: float = 1.0
    atp_weight: float = 0.7
    bbb_weight: float = 0.0
    membrane_weight: float = 0.0
    bbb_ckpt: str = ""
    bbb_sigma_gate: float = 4.0
    cutoff_angstrom: float = 5.0
    alpha: float = 8.0
    lj_sigma: float = 3.0
    max_force: float = 1.0


def geometric_guidance_force(
    atom_coords: torch.Tensor,
    peptide_atom_mask: torch.Tensor,
    hotspot_atom_indices: torch.Tensor,
    atp_atom_indices: torch.Tensor,
    cfg: GuidanceConfig,
) -> torch.Tensor:
    """Compute force-like tensor added to denoising direction."""
    coords = atom_coords.detach().requires_grad_(True)
    total_energy = coords.new_zeros(())

    for batch_idx in range(coords.shape[0]):
        pep_mask = peptide_atom_mask[batch_idx].bool()
        if not torch.any(pep_mask):
            continue

        pep_coords = coords[batch_idx][pep_mask]
        all_coords = coords[batch_idx]

        hs_idx = hotspot_atom_indices[(hotspot_atom_indices >= 0) & (hotspot_atom_indices < all_coords.shape[0])]
        atp_idx = atp_atom_indices[(atp_atom_indices >= 0) & (atp_atom_indices < all_coords.shape[0])]

        if hs_idx.numel() > 0 and cfg.hotspot_weight > 0:
            hs_coords = all_coords[hs_idx]
            h_score = hotspot_score(
                peptide_coords=pep_coords,
                hotspot_coords=hs_coords,
                cutoff=cfg.cutoff_angstrom,
                alpha=cfg.alpha,
            )
            total_energy = total_energy - (cfg.hotspot_weight * h_score)

        if atp_idx.numel() > 0 and cfg.atp_weight > 0:
            atp_coords = all_coords[atp_idx]
            r_score = atp_repulsion_score(
                peptide_coords=pep_coords,
                atp_coords=atp_coords,
                sigma=cfg.lj_sigma,
            )
            total_energy = total_energy + (cfg.atp_weight * r_score)

    grad = torch.autograd.grad(total_energy, coords, allow_unused=True)[0]
    if grad is None:
        return torch.zeros_like(atom_coords)
    force = -grad
    return torch.clamp(force, min=-cfg.max_force, max=cfg.max_force).detach()

