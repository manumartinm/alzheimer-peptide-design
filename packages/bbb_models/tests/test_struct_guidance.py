from __future__ import annotations

import torch

from bbb_geo.infer.struct_guidance import BBBGuidanceConfig, compute_bbb_guidance_force


def _minimal_feats(
    atom_coords: torch.Tensor,
    sequence: str,
) -> dict[str, torch.Tensor]:
    n_atoms = atom_coords.shape[0]
    n_tokens = len(sequence)
    atom_to_token = torch.zeros(n_atoms, n_tokens)
    design_mask = torch.zeros(n_tokens)
    res_type = torch.zeros(n_tokens, 33)
    for i in range(n_tokens):
        atom_to_token[i, i] = 1.0
        design_mask[i] = 1.0
        res_type[i, 2 + (i % 20)] = 1.0
    return {
        "atom_to_token": atom_to_token,
        "design_mask": design_mask,
        "res_type": res_type,
    }


def test_membrane_guidance_without_ckpt() -> None:
    coords = torch.tensor(
        [[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [1.9, 2.0, 0.0]],
        dtype=torch.float32,
    )
    feats = _minimal_feats(coords, "ALG")
    cfg = BBBGuidanceConfig(
        bbb_weight=0.0,
        membrane_weight=1.0,
        ckpt_path="",
        sigma_gate=8.0,
    )
    force = compute_bbb_guidance_force(coords, feats, torch.ones(coords.shape[0]), sigma=2.0, cfg=cfg)
    assert force is not None
    assert force.shape == coords.shape
    assert float(torch.linalg.norm(force)) > 0.0


def test_bbb_guidance_requires_ckpt() -> None:
    coords = torch.tensor([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]], dtype=torch.float32)
    feats = _minimal_feats(coords, "AL")
    cfg = BBBGuidanceConfig(bbb_weight=1.0, membrane_weight=0.0, ckpt_path="", sigma_gate=8.0)
    force = compute_bbb_guidance_force(coords, feats, torch.ones(coords.shape[0]), sigma=2.0, cfg=cfg)
    assert force is None
