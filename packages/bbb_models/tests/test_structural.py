from __future__ import annotations

import numpy as np
import torch

from bbb_geo.features.membrane_potential import (
    amphipathicity_score,
    hydrophobic_moment,
    membrane_potential_energy,
    per_residue_hydrophobicity,
    radius_of_gyration,
)
from bbb_geo.features.struct_graph import apply_coord_noise, build_struct_graph, radius_graph
from bbb_geo.features.struct_loader import plddt_sample_weight
from bbb_geo.models.struct_egnn import StructEGNNGeo, sample_edm_sigma


def test_build_struct_graph_shapes() -> None:
    seq = "ACDEFG"
    coords = np.stack([np.arange(6.0), np.zeros(6), np.zeros(6)], axis=1).astype(np.float32)
    graph = build_struct_graph(coords, seq)
    assert graph["coords"].shape == (6, 3)
    assert graph["node_feats"].shape == (6, 22)
    assert graph["edge_index"].shape[0] == 2


def test_coord_noise_changes_coords() -> None:
    coords = torch.ones((4, 3))
    noisy = apply_coord_noise(coords, sigma=0.5, generator=torch.Generator().manual_seed(0))
    assert not torch.allclose(coords, noisy)


def test_amphipathicity_gradient_nonzero() -> None:
    coords = torch.tensor([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [1.9, 2.0, 0.0]], requires_grad=True)
    energy = membrane_potential_energy(coords, "ALG")
    energy.backward()
    assert coords.grad is not None
    assert float(torch.linalg.norm(coords.grad)) > 0.0


def test_hydrophobic_moment_and_rg() -> None:
    coords = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    hydro = per_residue_hydrophobicity("AAA", device=coords.device, dtype=coords.dtype)
    moment = hydrophobic_moment(coords, hydro)
    assert moment.shape == (3,)
    assert float(radius_of_gyration(coords)) >= 0.0
    assert float(amphipathicity_score(coords, hydro)) >= 0.0


def test_struct_egnn_geo_forward() -> None:
    graph = build_struct_graph(
        np.array([[0, 0, 0], [3.8, 0, 0], [7.6, 0, 0]], dtype=np.float32),
        "AAA",
    )
    model = StructEGNNGeo(hidden_dim=32, num_layers=2)
    logits = model.forward(graphs=[graph])
    assert logits.shape == (1, 1)


def test_sample_edm_sigma_range() -> None:
    sigmas = sample_edm_sigma(16, device=torch.device("cpu"))
    assert sigmas.shape == (16,)
    assert torch.all(sigmas > 0)


def test_rotation_invariance_of_amphipathicity() -> None:
    coords = torch.tensor([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [1.9, 2.0, 0.0]])
    hydro = per_residue_hydrophobicity("LLL", device=coords.device, dtype=coords.dtype)
    base = float(amphipathicity_score(coords, hydro).item())
    theta = torch.tensor(torch.pi / 3)
    rot = torch.tensor(
        [
            [torch.cos(theta), -torch.sin(theta), 0.0],
            [torch.sin(theta), torch.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated = coords @ rot.T
    rot_score = float(amphipathicity_score(rotated, hydro).item())
    assert abs(base - rot_score) < 1e-4


def test_plddt_sample_weight() -> None:
    assert plddt_sample_weight(90.0) == 0.9
    assert plddt_sample_weight(5.0, floor=0.1) == 0.1
    assert plddt_sample_weight(None) == 1.0
