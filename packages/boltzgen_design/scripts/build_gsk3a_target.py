#!/usr/bin/env python3
"""Download, clean GSK3α structure and build β↔α isoform map for G6 selectivity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests
import yaml
from Bio.Align import PairwiseAligner
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.mmcifio import MMCIFIO

DESIGN_ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = DESIGN_ROOT / "targets" / "gsk3a"
BETA_CIF = DESIGN_ROOT / "targets" / "gsk3b" / "gsk3b.cif"
BETA_CONFIG = DESIGN_ROOT / "configs" / "gsk3b_target.yaml"
ALPHA_CONFIG = DESIGN_ROOT / "configs" / "gsk3a_target.yaml"
RCSB_CIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif"

THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def load_mmcif_dict(path: Path) -> dict:
    return MMCIF2Dict(str(path))


def download_pdb_cif(pdb_id: str, dest: Path, force: bool = False) -> Path:
    dest = dest.resolve()
    if dest.exists() and not force:
        print(f"Already exists {dest.name} ({dest.stat().st_size:,} bytes)")
        return dest
    url = RCSB_CIF_URL.format(pdb_id=pdb_id.upper())
    print(f"Downloading {url} ...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    print(f"Saved {dest} ({dest.stat().st_size:,} bytes)")
    return dest


def clean_kinase_cif(src: Path, dest: Path, chain_label: str = "A", force: bool = False) -> Path:
    dest = dest.resolve()
    if dest.exists() and not force:
        print(f"Already exists {dest.name} ({dest.stat().st_size:,} bytes)")
        return dest

    cif_dict = load_mmcif_dict(src)
    n = len(cif_dict["_atom_site.group_PDB"])
    keep = [
        i
        for i in range(n)
        if cif_dict["_atom_site.label_asym_id"][i] == chain_label
        and cif_dict["_atom_site.group_PDB"][i] == "ATOM"
    ]
    if not keep:
        raise ValueError(f"No ATOM rows for label_asym_id={chain_label!r} in {src}")

    atom_keys = [k for k in cif_dict if k.startswith("_atom_site.")]
    out = dict(cif_dict)
    for k in atom_keys:
        out[k] = [cif_dict[k][i] for i in keep]

    dest.parent.mkdir(parents=True, exist_ok=True)
    io = MMCIFIO()
    io.set_dict(out)
    io.save(str(dest))
    print(f"Cleaned {len(keep):,} ATOM rows -> {dest}")
    return dest


def build_label_seq_map(cif_dict: dict, chain_label: str = "A") -> dict[int, str]:
    mapping: dict[int, str] = {}
    n = len(cif_dict["_atom_site.group_PDB"])
    for i in range(n):
        if cif_dict["_atom_site.label_asym_id"][i] != chain_label:
            continue
        if cif_dict["_atom_site.group_PDB"][i] != "ATOM":
            continue
        atom_key = (
            "_atom_site.label_atom_id"
            if "_atom_site.label_atom_id" in cif_dict
            else "_atom_site.atom_name"
        )
        if cif_dict[atom_key][i] != "CA":
            continue
        seq_id = int(cif_dict["_atom_site.label_seq_id"][i])
        res_name = cif_dict["_atom_site.label_comp_id"][i].upper()
        mapping[seq_id] = res_name
    return mapping


def seq_map_to_string(seq_map: dict[int, str], start: int, end: int) -> tuple[str, list[int]]:
    ids = list(range(start, end + 1))
    chars: list[str] = []
    present_ids: list[int] = []
    for rid in ids:
        if rid not in seq_map:
            chars.append("-")
            present_ids.append(-1)
            continue
        aa = THREE_TO_ONE.get(seq_map[rid], "X")
        chars.append(aa)
        present_ids.append(rid)
    return "".join(chars), present_ids


def build_isoform_map(
    beta_map: dict[int, str],
    alpha_map: dict[int, str],
    *,
    domain_start: int = 46,
    domain_end: int = 220,
) -> dict:
    beta_ids_sorted = sorted(beta_map)
    alpha_ids_sorted = sorted(alpha_map)
    beta_seq = "".join(THREE_TO_ONE.get(beta_map[i], "X") for i in beta_ids_sorted)
    alpha_seq = "".join(THREE_TO_ONE.get(alpha_map[i], "X") for i in alpha_ids_sorted)

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aln = aligner.align(beta_seq, alpha_seq)[0]
    beta_aln = str(aln[0])
    alpha_aln = str(aln[1])

    beta_to_alpha: dict[int, int] = {}
    differential: list[dict] = []
    superposition_anchors: list[int] = []
    position_map: list[dict] = []
    i_beta = 0
    i_alpha = 0
    for b_char, a_char in zip(beta_aln, alpha_aln, strict=True):
        b_id = beta_ids_sorted[i_beta] if b_char != "-" else None
        a_id = alpha_ids_sorted[i_alpha] if a_char != "-" else None
        if b_char != "-" and a_char != "-":
            beta_to_alpha[b_id] = a_id
            if domain_start <= b_id <= domain_end:
                entry = {
                    "beta_label_seq_id": b_id,
                    "alpha_label_seq_id": a_id,
                    "beta_aa": beta_map[b_id],
                    "alpha_aa": alpha_map[a_id],
                    "is_differential": beta_map[b_id] != alpha_map[a_id],
                }
                position_map.append(entry)
                if beta_map[b_id] == alpha_map[a_id]:
                    superposition_anchors.append(b_id)
                else:
                    differential.append(
                        {
                            "beta_label_seq_id": b_id,
                            "alpha_label_seq_id": a_id,
                            "beta_aa": beta_map[b_id],
                            "alpha_aa": alpha_map[a_id],
                        }
                    )
        if b_char != "-":
            i_beta += 1
        if a_char != "-":
            i_alpha += 1

    preferred_anchors = [67, 89, 96, 133, 134, 135, 180, 200, 201, 202]
    anchors = [r for r in preferred_anchors if r in superposition_anchors]
    if len(anchors) < 4:
        anchors = superposition_anchors[:20]

    return {
        "beta_pdb": "1Q4L",
        "alpha_pdb": "1Q5K",
        "beta_reference_cif": "targets/gsk3b/gsk3b.cif",
        "alpha_reference_cif": "targets/gsk3a/gsk3a.cif",
        "domain_label_seq_id_range": [domain_start, domain_end],
        "superposition_anchors": anchors,
        "beta_to_alpha": {str(k): v for k, v in beta_to_alpha.items()},
        "position_map": position_map,
        "differential_positions": differential,
        "n_differential": len(differential),
        "n_superposition_anchors": len(anchors),
        "n_mapped_domain_positions": len(position_map),
    }


def write_alpha_config(dest: Path) -> None:
    beta_cfg = yaml.safe_load(BETA_CONFIG.read_text(encoding="utf-8"))
    alpha_cfg = {
        "target": {
            "name": "gsk3a",
            "pdb_id": "1Q5K",
            "chain_id": "A",
            "cif_path": "../targets/gsk3a/gsk3a.cif",
            "rationale": (
                "GSK3α off-target for selectivity gate G6. Same kinase-domain slice as GSK3β "
                "(label_seq_id 46–220) for geometric isoform comparison."
            ),
        },
        "regions": beta_cfg.get("regions", {}),
    }
    dest.write_text(yaml.safe_dump(alpha_cfg, sort_keys=False), encoding="utf-8")
    print(f"Wrote {dest}")


def write_readme(dest: Path) -> None:
    dest.write_text(
        """# GSK3α target (selectivity gate G6)

Source structure: **1Q5K** (GSK-3 alpha kinase domain).

Used only for **geometric isoform selectivity** (G6): GSK3α is superposed onto GSK3β in refolded peptide complexes; peptide contacts to β- vs α-specific surface residues are compared.

**Limitation:** this is a structural proxy, not a cross-target refold or binding affinity measurement.

Regenerate artifacts:

```bash
uv run python boltzgen_design/scripts/build_gsk3a_target.py
```
""",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build GSK3α target and isoform map.")
    p.add_argument("--pdb-id", default="1Q5K")
    p.add_argument("--chain-id", default="A")
    p.add_argument("--force-download", action="store_true")
    p.add_argument("--force-clean", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not BETA_CIF.exists():
        print(f"ERROR: GSK3β reference missing: {BETA_CIF}", file=sys.stderr)
        sys.exit(1)

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    raw_cif = TARGET_DIR / f"{args.pdb_id.upper()}_raw.cif"
    clean_cif = TARGET_DIR / "gsk3a.cif"
    isoform_map_path = TARGET_DIR / "isoform_map.json"

    download_pdb_cif(args.pdb_id, raw_cif, force=args.force_download)
    clean_kinase_cif(raw_cif, clean_cif, chain_label=args.chain_id, force=args.force_clean)

    beta_map = build_label_seq_map(load_mmcif_dict(BETA_CIF), chain_label="A")
    alpha_map = build_label_seq_map(load_mmcif_dict(clean_cif), chain_label=args.chain_id)
    isoform_map = build_isoform_map(beta_map, alpha_map)
    isoform_map_path.write_text(json.dumps(isoform_map, indent=2), encoding="utf-8")
    print(f"Wrote {isoform_map_path} ({isoform_map['n_differential']} differential positions)")

    write_alpha_config(ALPHA_CONFIG)
    write_readme(TARGET_DIR / "README.md")

    print("\nDifferential positions (β label_seq_id: β_aa / α_aa):")
    for entry in isoform_map["differential_positions"][:15]:
        print(f"  {entry['beta_label_seq_id']:3d}: {entry['beta_aa']} / {entry['alpha_aa']}")
    if isoform_map["n_differential"] > 15:
        print(f"  ... and {isoform_map['n_differential'] - 15} more")


if __name__ == "__main__":
    main()
