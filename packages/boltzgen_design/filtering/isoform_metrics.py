from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from biotite.structure import AtomArray
from biotite.structure.io.pdbx import CIFFile, get_structure

from boltzgen_design.filtering.struct_metrics import _label_seq_ids_as_int

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ISOFORM_MAP = _PACKAGE_ROOT / "targets" / "gsk3a" / "isoform_map.json"
DEFAULT_GSK3A_CIF = _PACKAGE_ROOT / "targets" / "gsk3a" / "gsk3a.cif"


def load_isoform_map(path: Path | None = None) -> dict:
    map_path = path or DEFAULT_ISOFORM_MAP
    if not map_path.exists():
        raise FileNotFoundError(f"isoform_map.json not found: {map_path}")
    return json.loads(map_path.read_text(encoding="utf-8"))


def _kabsch_transform(
    mobile: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mobile_cent = mobile.mean(axis=0)
    target_cent = target.mean(axis=0)
    m = mobile - mobile_cent
    t = target - target_cent
    cov = m.T @ t
    u, _, vt = np.linalg.svd(cov)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1, :] *= -1
        rot = vt.T @ u.T
    return rot, mobile_cent, target_cent


def _apply_transform(
    coords: np.ndarray, rot: np.ndarray, mobile_cent: np.ndarray, target_cent: np.ndarray
) -> np.ndarray:
    return (coords - mobile_cent) @ rot + target_cent


def _residue_atoms(structure: AtomArray, chain_id: str, label_seq_id: int) -> AtomArray:
    seq_ids = _label_seq_ids_as_int(structure.label_seq_id)
    mask = (structure.chain_id == chain_id) & (seq_ids == int(label_seq_id))
    atoms = structure[mask]
    return atoms[atoms.element != "H"]


def _ca_coord(structure: AtomArray, chain_id: str, label_seq_id: int) -> np.ndarray | None:
    seq_ids = _label_seq_ids_as_int(structure.label_seq_id)
    mask = (
        (structure.chain_id == chain_id)
        & (seq_ids == int(label_seq_id))
        & (structure.atom_name == "CA")
    )
    sub = structure[mask]
    if sub.array_length() == 0:
        return None
    return sub.coord[0]


def _peptide_heavy_coords(structure: AtomArray, peptide_chain: str) -> np.ndarray:
    pep = structure[structure.chain_id == peptide_chain]
    heavy = pep[pep.element != "H"]
    if heavy.array_length() == 0:
        return np.empty((0, 3), dtype=float)
    return heavy.coord


def _min_distance(peptide_coords: np.ndarray, target_coords: np.ndarray) -> float:
    if peptide_coords.size == 0 or target_coords.size == 0:
        return float("inf")
    dmat = np.linalg.norm(peptide_coords[:, None, :] - target_coords[None, :, :], axis=2)
    return float(dmat.min())


def compute_isoform_selectivity(
    complex_cif: Path,
    gsk3a_kinase_cif: Path,
    isoform_map: dict | None = None,
    *,
    peptide_chain: str = "A",
    beta_chain: str = "B",
    alpha_chain: str = "A",
    contact_cutoff: float = 5.0,
    interface_shell: float = 12.0,
) -> dict[str, float]:
    isoform_map = isoform_map or load_isoform_map()
    complex_struct = get_structure(
        CIFFile.read(complex_cif), model=1, extra_fields=["label_seq_id"]
    )
    alpha_struct = get_structure(
        CIFFile.read(gsk3a_kinase_cif), model=1, extra_fields=["label_seq_id"]
    )

    peptide_coords = _peptide_heavy_coords(complex_struct, peptide_chain)
    if peptide_coords.size == 0:
        return _empty_metrics()

    anchors = [int(x) for x in isoform_map.get("superposition_anchors", [])]
    mobile_pts: list[np.ndarray] = []
    target_pts: list[np.ndarray] = []
    beta_to_alpha = isoform_map.get("beta_to_alpha", {})

    for beta_id in anchors:
        alpha_id = beta_to_alpha.get(str(beta_id)) or beta_to_alpha.get(beta_id)
        if alpha_id is None:
            continue
        beta_ca = _ca_coord(complex_struct, beta_chain, beta_id)
        alpha_ca = _ca_coord(alpha_struct, alpha_chain, int(alpha_id))
        if beta_ca is None or alpha_ca is None:
            continue
        mobile_pts.append(alpha_ca)
        target_pts.append(beta_ca)

    if len(mobile_pts) < 3:
        return _empty_metrics()

    mobile = np.stack(mobile_pts, axis=0)
    target = np.stack(target_pts, axis=0)
    rot, mobile_cent, target_cent = _kabsch_transform(mobile, target)
    alpha_coords_all = alpha_struct.coord.copy()
    alpha_coords_all = _apply_transform(alpha_coords_all, rot, mobile_cent, target_cent)
    alpha_transformed = alpha_struct.copy()
    alpha_transformed.coord = alpha_coords_all

    position_map = (
        isoform_map.get("position_map") or isoform_map.get("differential_positions") or []
    )
    evaluated = 0
    beta_favored = 0
    beta_contacts = 0
    alpha_contacts = 0

    for entry in position_map:
        beta_id = int(entry["beta_label_seq_id"])
        alpha_id = int(entry["alpha_label_seq_id"])
        beta_ca = _ca_coord(complex_struct, beta_chain, beta_id)
        if beta_ca is None:
            continue
        pep_to_beta_ca = float(np.linalg.norm(peptide_coords - beta_ca, axis=1).min())
        if pep_to_beta_ca > interface_shell:
            continue

        beta_atoms = _residue_atoms(complex_struct, beta_chain, beta_id)
        alpha_atoms = _residue_atoms(alpha_transformed, alpha_chain, alpha_id)
        d_beta = _min_distance(peptide_coords, beta_atoms.coord)
        d_alpha = _min_distance(peptide_coords, alpha_atoms.coord)

        evaluated += 1
        if d_beta <= contact_cutoff:
            beta_contacts += 1
        if d_alpha <= contact_cutoff:
            alpha_contacts += 1
        if d_beta < d_alpha and d_beta <= contact_cutoff:
            beta_favored += 1

    if evaluated == 0:
        return _empty_metrics()

    return {
        "selectivity_margin": float(beta_favored / evaluated),
        "alpha_contact_fraction": float(alpha_contacts / evaluated),
        "beta_contact_fraction": float(beta_contacts / evaluated),
        "n_isoform_interface": float(evaluated),
        "n_beta_favored": float(beta_favored),
    }


def _empty_metrics() -> dict[str, float]:
    return {
        "selectivity_margin": float("nan"),
        "alpha_contact_fraction": float("nan"),
        "beta_contact_fraction": float("nan"),
        "n_isoform_interface": 0.0,
        "n_beta_favored": 0.0,
    }
