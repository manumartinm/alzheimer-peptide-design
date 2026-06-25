from __future__ import annotations

from pathlib import Path

import numpy as np

from bbb_dataset.struct_io import parse_cif_backbone, write_coords_npz


def test_parse_minimal_pdb_style_cif(tmp_path: Path) -> None:
    cif = tmp_path / "mini.cif"
    cif.write_text(
        "\n".join(
            [
                "ATOM 1 CA ALA A 1 0.0 0.0 0.0 1.00 90.0",
                "ATOM 2 CA GLY A 2 3.8 0.0 0.0 1.00 85.0",
                "ATOM 3 CA LEU A 3 7.6 0.0 0.0 1.00 80.0",
            ]
        ),
        encoding="utf-8",
    )
    parsed = parse_cif_backbone(cif)
    assert parsed["coords"].shape == (3, 3)
    assert "".join(parsed["sequence"]) == "AGL"


def test_write_and_load_coords_npz(tmp_path: Path) -> None:
    parsed = {
        "coords": np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        "sequence": ["A", "G"],
        "plddt_per_residue": np.array([90.0, 85.0], dtype=np.float32),
    }
    out = tmp_path / "coords.npz"
    write_coords_npz(out, parsed)
    loaded = np.load(out, allow_pickle=True)
    assert loaded["coords"].shape == (2, 3)
