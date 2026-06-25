from __future__ import annotations

from pathlib import Path

import numpy as np

from bbb_dataset.aa import THREE_TO_ONE


def _parse_pdb_atom_lines(cif_path: Path) -> dict[str, np.ndarray | list[str]]:
    coords: list[list[float]] = []
    sequence: list[str] = []
    plddt: list[float] = []
    seen: set[tuple[str, str]] = set()
    with cif_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            atom_name = parts[2].upper()
            if atom_name != "CA":
                continue
            res_name = parts[3].upper()
            chain = parts[4]
            res_id = parts[5]
            key = (chain, res_id)
            if key in seen:
                continue
            aa = THREE_TO_ONE.get(res_name)
            if aa is None:
                continue
            x, y, z = float(parts[6]), float(parts[7]), float(parts[8])
            bfac = float(parts[10]) if len(parts) > 10 else float("nan")
            seen.add(key)
            sequence.append(aa)
            coords.append([x, y, z])
            plddt.append(bfac)
    if coords:
        return {
            "coords": np.asarray(coords, dtype=np.float32),
            "sequence": sequence,
            "plddt_per_residue": np.asarray(plddt, dtype=np.float32),
        }
    raise ValueError(f"No CA atoms parsed from {cif_path}")


def write_coords_npz(path: Path, parsed: dict[str, np.ndarray | list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    seq = parsed["sequence"]
    seq_str = "".join(seq) if isinstance(seq, list) else str(seq)
    np.savez_compressed(
        path,
        coords=np.asarray(parsed["coords"], dtype=np.float32),
        sequence=np.array(list(seq_str)),
        plddt_per_residue=np.asarray(parsed.get("plddt_per_residue", []), dtype=np.float32),
    )


def load_coords_npz(path: Path) -> dict[str, np.ndarray | str]:
    payload = np.load(path, allow_pickle=True)
    seq_arr = payload["sequence"]
    if seq_arr.ndim == 0:
        sequence = "".join(str(seq_arr.item()))
    else:
        sequence = "".join(str(x) for x in seq_arr.tolist())
    return {
        "coords": payload["coords"].astype(np.float32),
        "sequence": sequence,
        "plddt_per_residue": payload["plddt_per_residue"].astype(np.float32)
        if "plddt_per_residue" in payload
        else np.array([], dtype=np.float32),
    }


def parse_cif_backbone(cif_path: Path) -> dict[str, np.ndarray | list[str]]:
    """Extract CA coordinates, per-residue sequence, and optional pLDDT from a CIF file."""
    try:
        from Bio.PDB.MMCIFParser import MMCIFParser

        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("peptide", str(cif_path))
        coords: list[list[float]] = []
        sequence: list[str] = []
        plddt: list[float] = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] != " ":
                        continue
                    res_name = residue.get_resname().upper()
                    aa = THREE_TO_ONE.get(res_name)
                    if aa is None:
                        continue
                    if "CA" not in residue:
                        continue
                    atom = residue["CA"]
                    coords.append(atom.coord.astype(float).tolist())
                    sequence.append(aa)
                    plddt.append(float(atom.bfactor))
            break
        if coords:
            return {
                "coords": np.asarray(coords, dtype=np.float32),
                "sequence": sequence,
                "plddt_per_residue": np.asarray(plddt, dtype=np.float32),
            }
    except Exception:
        pass
    return _parse_pdb_atom_lines(cif_path)
