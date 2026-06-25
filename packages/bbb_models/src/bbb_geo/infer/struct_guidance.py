from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from bbb_classifier.constants import BOLTZ_CANONICAL, THREE_TO_ONE
from bbb_geo.features.membrane_potential import membrane_potential_energy
from bbb_geo.features.struct_graph import build_struct_graph, radius_graph, rbf_edge_features
from bbb_geo.models.struct_egnn import StructEGNNGeo
from bbb_geo.train.checkpoints import load_checkpoint

@dataclass
class BBBGuidanceConfig:
    bbb_weight: float = 0.3
    membrane_weight: float = 0.7
    ckpt_path: str = ""
    sigma_gate: float = 4.0
    max_force: float = 1.0
    model_hidden: int = 64
    model_layers: int = 3


_MODEL_CACHE: dict[str, StructEGNNGeo] = {}


def _load_geo_model(ckpt_path: str, hidden: int, layers: int, device: torch.device) -> StructEGNNGeo:
    if ckpt_path in _MODEL_CACHE:
        return _MODEL_CACHE[ckpt_path]
    model = StructEGNNGeo(hidden_dim=hidden, num_layers=layers).to(device)
    state = load_checkpoint(Path(ckpt_path))
    model.load_state_dict(state["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _MODEL_CACHE[ckpt_path] = model
    return model


def _sequence_from_res_type(res_type: torch.Tensor) -> str:
    idx = int(torch.argmax(res_type).item())
    boltz_idx = idx - 2
    if 0 <= boltz_idx < len(BOLTZ_CANONICAL):
        return THREE_TO_ONE[BOLTZ_CANONICAL[boltz_idx]]
    return "A"


def _group_design_residues(
    atom_coords: torch.Tensor,
    atom_mask: torch.Tensor,
    atom_to_token: torch.Tensor,
    design_mask: torch.Tensor,
    res_type: torch.Tensor,
) -> tuple[torch.Tensor, str] | None:
    design_idx = torch.where(design_mask.bool())[0]
    if design_idx.numel() == 0:
        return None
    coords_list: list[torch.Tensor] = []
    seq_chars: list[str] = []
    valid = atom_mask.bool()
    for tok in design_idx:
        tok = int(tok.item())
        atom_idx = torch.where((atom_to_token[:, tok] > 0.5) & valid)[0]
        if atom_idx.numel() == 0:
            continue
        coords_list.append(atom_coords[atom_idx].mean(dim=0))
        seq_chars.append(_sequence_from_res_type(res_type[tok]))
    if not coords_list:
        return None
    return torch.stack(coords_list, dim=0), "".join(seq_chars)


def compute_bbb_guidance_force(
    atom_coords: torch.Tensor,
    feats: dict[str, Any],
    atom_mask: torch.Tensor,
    sigma: torch.Tensor | float,
    cfg: BBBGuidanceConfig,
) -> torch.Tensor | None:
    if cfg.bbb_weight <= 0 and cfg.membrane_weight <= 0:
        return None
    sigma_val = float(sigma.mean().item()) if isinstance(sigma, torch.Tensor) else float(sigma)
    if sigma_val > cfg.sigma_gate:
        return None
    if cfg.bbb_weight > 0 and not cfg.ckpt_path:
        return None
    if "atom_to_token" not in feats or "design_mask" not in feats or "res_type" not in feats:
        return None

    atom_to_token = feats["atom_to_token"]
    design_mask = feats["design_mask"]
    res_type = feats["res_type"]
    if atom_coords.dim() == 3:
        forces = []
        for b in range(atom_coords.shape[0]):
            force = compute_bbb_guidance_force(
                atom_coords[b],
                {
                    **feats,
                    "atom_to_token": atom_to_token[b] if atom_to_token.dim() == 3 else atom_to_token,
                    "design_mask": design_mask[b] if design_mask.dim() == 2 else design_mask,
                    "res_type": res_type[b] if res_type.dim() == 3 else res_type,
                },
                atom_mask[b] if atom_mask.dim() == 2 else atom_mask,
                sigma,
                cfg,
            )
            if force is None:
                forces.append(torch.zeros_like(atom_coords[b]))
            else:
                forces.append(force)
        return torch.stack(forces, dim=0)

    coords_var = atom_coords.detach().clone().requires_grad_(True)
    design_idx = torch.where(design_mask.bool())[0]
    if design_idx.numel() == 0:
        return None

    res_coords_list: list[torch.Tensor] = []
    seq_chars: list[str] = []
    atom_groups: list[torch.Tensor] = []
    valid = atom_mask.bool()
    for tok in design_idx:
        tok_i = int(tok.item())
        atom_idx = torch.where((atom_to_token[:, tok_i] > 0.5) & valid)[0]
        if atom_idx.numel() == 0:
            continue
        atom_groups.append(atom_idx)
        res_coords_list.append(coords_var[atom_idx].mean(dim=0))
        seq_chars.append(_sequence_from_res_type(res_type[tok_i]))
    if not res_coords_list:
        return None

    res_coords = torch.stack(res_coords_list, dim=0)
    sequence = "".join(seq_chars)
    graph = build_struct_graph(res_coords.detach().cpu().numpy(), sequence)
    graph["coords"] = res_coords
    graph["node_feats"] = graph["node_feats"].to(res_coords.device)
    graph["edge_index"] = radius_graph(res_coords).to(res_coords.device)
    graph["edge_attr"] = rbf_edge_features(res_coords, graph["edge_index"]).to(res_coords.device)
    graph["sequence"] = sequence
    graph["sigma"] = sigma_val

    energy = res_coords.new_zeros(())
    if cfg.bbb_weight > 0:
        model = _load_geo_model(cfg.ckpt_path, cfg.model_hidden, cfg.model_layers, res_coords.device)
        energy = energy - cfg.bbb_weight * model.log_prob([graph]).sum()
    if cfg.membrane_weight > 0:
        energy = energy - cfg.membrane_weight * membrane_potential_energy(res_coords, sequence)

    grad_atoms = torch.autograd.grad(energy, coords_var, allow_unused=True)[0]
    if grad_atoms is None:
        return None
    force = -grad_atoms
    if cfg.max_force > 0:
        force = torch.clamp(force, min=-cfg.max_force, max=cfg.max_force)
    return force.detach()
