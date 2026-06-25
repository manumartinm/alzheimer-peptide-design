from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests
from Bio import SeqIO

B3PRED_BASE = "https://webs.iiitd.edu.in/raghava/b3pred/Datasets"


def download_if_missing(url: str, out_path: Path, timeout: int = 60) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return out_path
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    out_path.write_bytes(response.content)
    return out_path


def ensure_b3pred_files(raw_dir: Path) -> dict[str, Path]:
    files = {
        "b3pred_pos_train.fa": (f"{B3PRED_BASE}/pos_train.fasta"),
        "b3pred_pos_val.fa": (f"{B3PRED_BASE}/pos_valid.fasta"),
        "b3pred_neg_train.fa": (f"{B3PRED_BASE}/D1_neg_train.fasta"),
        "b3pred_neg_val.fa": (f"{B3PRED_BASE}/D1_neg_valid.fasta"),
    }
    out: dict[str, Path] = {}
    for name, url in files.items():
        out[name] = download_if_missing(url, raw_dir / name)
    return out


def _load_fasta(path: Path, label: int, split: str, source_db: str, label_tier: str) -> list[dict]:
    rows = []
    for record in SeqIO.parse(path, "fasta"):
        sequence = str(record.seq).strip().upper()
        rows.append(
            {
                "source_id": record.id,
                "sequence": sequence,
                "length": len(sequence),
                "bbb_label": int(label),
                "source_db": source_db,
                "split": split,
                "source_split": split,
                "label_tier": label_tier,
            }
        )
    return rows


def load_b3pred_d1(raw_dir: Path) -> pd.DataFrame:
    files = ensure_b3pred_files(raw_dir)
    rows: list[dict] = []
    rows += _load_fasta(files["b3pred_pos_train.fa"], label=1, split="train", source_db="B3Pred_D1", label_tier="gold")
    rows += _load_fasta(files["b3pred_pos_val.fa"], label=1, split="val", source_db="B3Pred_D1", label_tier="gold")
    rows += _load_fasta(files["b3pred_neg_train.fa"], label=0, split="train", source_db="B3Pred_D1", label_tier="gold")
    rows += _load_fasta(files["b3pred_neg_val.fa"], label=0, split="val", source_db="B3Pred_D1", label_tier="gold")
    return pd.DataFrame(rows)


def load_b3pdb(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, sep="\t")
    cols = {c.lower(): c for c in df.columns}
    seq_col = cols.get("sequence") or cols.get("peptide_sequence")
    if not seq_col:
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "source_id": df.index.astype(str),
            "sequence": df[seq_col].astype(str).str.upper(),
            "bbb_label": 1,
            "source_db": "B3Pdb",
            "split": "train",
            "source_split": "train",
            "label_tier": "gold",
            "assay_method": df[cols["method"]] if "method" in cols else None,
            "reference": df[cols["reference"]] if "reference" in cols else None,
            "organism": df[cols["organism"]] if "organism" in cols else None,
        }
    )
    out["length"] = out["sequence"].str.len()
    return out


def load_brainpeps(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, sep=None, engine="python")
    cols = {c.lower(): c for c in df.columns}
    seq_col = cols.get("sequence") or cols.get("peptide")
    label_col = cols.get("bbb_label") or cols.get("label")
    if not seq_col:
        return pd.DataFrame()
    if label_col:
        labels = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
    else:
        labels = pd.Series([0] * len(df))
    out = pd.DataFrame(
        {
            "source_id": df.index.astype(str),
            "sequence": df[seq_col].astype(str).str.upper(),
            "bbb_label": labels,
            "source_db": "Brainpeps",
            "split": "train",
            "source_split": "train",
            "label_tier": "silver",
            "assay_method": df[cols["assay_method"]] if "assay_method" in cols else None,
            "reference": df[cols["reference"]] if "reference" in cols else None,
            "organism": df[cols["organism"]] if "organism" in cols else None,
        }
    )
    out["length"] = out["sequence"].str.len()
    return out
