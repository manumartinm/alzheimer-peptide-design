from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from biotite.structure import distance
from biotite.structure.io.pdbx import CIFFile, get_structure

DEFAULT_GUIDANCE = Path(__file__).resolve().parents[1] / "targets" / "gsk3b" / "guidance.json"


def load_guidance(path: Path | None = None) -> dict:
    guidance_path = path or DEFAULT_GUIDANCE
    if not guidance_path.exists():
        raise FileNotFoundError(f"guidance.json not found: {guidance_path}")
    return json.loads(guidance_path.read_text(encoding="utf-8"))


def _label_seq_ids_as_int(label_seq_id) -> np.ndarray:
    out = np.empty(len(label_seq_id), dtype=np.int32)
    for i, val in enumerate(label_seq_id):
        try:
            out[i] = int(val)
        except (TypeError, ValueError):
            out[i] = -1
    return out


def _region_coords(
    structure, chain_id: str, label_seq_ids: list[int], atom_names: tuple[str, ...] = ("CA",)
) -> np.ndarray:
    coords: list[np.ndarray] = []
    ids = set(int(x) for x in label_seq_ids)
    seq_ids = _label_seq_ids_as_int(structure.label_seq_id)
    for atom_name in atom_names:
        mask = (
            (structure.chain_id == chain_id)
            & np.isin(seq_ids, list(ids))
            & (structure.atom_name == atom_name)
        )
        sub = structure[mask]
        if sub.array_length() > 0:
            coords.append(sub.coord)
    if not coords:
        return np.empty((0, 3), dtype=float)
    return np.concatenate(coords, axis=0)


def _min_distance_per_target(peptide_coords: np.ndarray, target_coords: np.ndarray) -> np.ndarray:
    if peptide_coords.size == 0 or target_coords.size == 0:
        return np.empty(0, dtype=float)
    dmat = np.linalg.norm(peptide_coords[:, None, :] - target_coords[None, :, :], axis=2)
    return dmat.min(axis=0)


def hotspot_fraction(
    peptide_coords: np.ndarray,
    hotspot_coords: np.ndarray,
    threshold_angstrom: float = 5.0,
) -> float:
    min_d = _min_distance_per_target(peptide_coords, hotspot_coords)
    if min_d.size == 0:
        return 0.0
    return float(np.mean(min_d <= threshold_angstrom))


def atp_repulsion(
    peptide_coords: np.ndarray,
    atp_coords: np.ndarray,
    sigma: float = 3.0,
    eps: float = 1e-6,
) -> float:
    min_d = _min_distance_per_target(peptide_coords, atp_coords)
    if min_d.size == 0:
        return 0.0
    return float(np.mean((sigma / (min_d + eps)) ** 12))


def cyclic_closure_rmsd(peptide, cys_resnames: tuple[str, ...] = ("CYS",)) -> float | None:
    sg = peptide[(peptide.atom_name == "SG") & (np.isin(peptide.res_name, list(cys_resnames)))]
    if sg.array_length() < 2:
        return None
    if sg.array_length() > 2:
        dmat = distance(sg.coord, sg.coord)
        np.fill_diagonal(dmat, np.inf)
        i, j = np.unravel_index(np.argmin(dmat), dmat.shape)
        sg_dist = float(dmat[i, j])
    else:
        sg_dist = float(distance(sg.coord[0], sg.coord[1]))
    return abs(sg_dist - 2.05)


def compute_struct_metrics(
    cif_path: Path,
    guidance: dict | None = None,
    peptide_chain: str = "A",
    kinase_chain: str = "B",
) -> dict[str, float]:
    guidance = guidance or load_guidance()
    cutoff = float(guidance.get("guidance", {}).get("cutoff_angstrom", 5.0))
    hotspots = guidance.get("hotspots_primary", []) + guidance.get("hotspots_secondary", [])
    atp_cleft = guidance.get("atp_cleft", [])

    structure = get_structure(CIFFile.read(cif_path), model=1, extra_fields=["label_seq_id"])
    peptide = structure[structure.chain_id == peptide_chain]
    kinase = structure[structure.chain_id == kinase_chain]
    pep_ca = peptide[peptide.atom_name == "CA"]
    hotspot_ca = _region_coords(kinase, kinase_chain, hotspots, atom_names=("CA",))
    atp_ca = _region_coords(kinase, kinase_chain, atp_cleft, atom_names=("CA",))

    closure = cyclic_closure_rmsd(peptide)
    return {
        "hotspot_fraction": hotspot_fraction(pep_ca.coord, hotspot_ca, cutoff),
        "atp_repulsion": atp_repulsion(pep_ca.coord, atp_ca),
        "closure_rmsd": closure if closure is not None else float("nan"),
    }
