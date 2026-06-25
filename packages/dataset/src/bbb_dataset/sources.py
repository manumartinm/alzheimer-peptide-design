from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
from Bio import SeqIO

from .enums import BbbLabel, LabelTier, SourceDb, Split

B3PRED_BASE = "https://webs.iiitd.edu.in/raghava/b3pred/Datasets"


@dataclass
class SourceRegistry:
    raw_dir: Path
    b3pdb_path: Path | None = None
    brainpeps_path: Path | None = None

    def load(self, source: SourceDb) -> pd.DataFrame:
        if source == SourceDb.B3PRED_D1:
            return self._load_b3pred_d1()
        if source == SourceDb.B3PDB:
            return self._load_b3pdb()
        if source == SourceDb.BRAINPEPS:
            return self._load_brainpeps()
        raise ValueError(f"Unknown source: {source}")

    def load_all(self, enabled: frozenset[SourceDb] | None = None) -> pd.DataFrame:
        sources = enabled or frozenset(SourceDb)
        tables: list[pd.DataFrame] = []
        for source in sources:
            df = self.load(source)
            if not df.empty:
                tables.append(df)
        if not tables:
            return pd.DataFrame()
        return pd.concat(tables, ignore_index=True).fillna("")

    def _ensure_b3pred_files(self) -> dict[str, Path]:
        files = {
            "b3pred_pos_train.fa": f"{B3PRED_BASE}/pos_train.fasta",
            "b3pred_pos_val.fa": f"{B3PRED_BASE}/pos_valid.fasta",
            "b3pred_neg_train.fa": f"{B3PRED_BASE}/D1_neg_train.fasta",
            "b3pred_neg_val.fa": f"{B3PRED_BASE}/D1_neg_valid.fasta",
        }
        out: dict[str, Path] = {}
        for name, url in files.items():
            out[name] = self._download_if_missing(url, self.raw_dir / name)
        return out

    @staticmethod
    def _download_if_missing(url: str, out_path: Path, timeout: int = 60) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            return out_path
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        out_path.write_bytes(response.content)
        return out_path

    def _load_fasta(
        self,
        path: Path,
        label: BbbLabel,
        split: Split,
        source_db: SourceDb,
        label_tier: LabelTier,
    ) -> list[dict]:
        rows = []
        for record in SeqIO.parse(path, "fasta"):
            sequence = str(record.seq).strip().upper()
            rows.append(
                {
                    "source_id": record.id,
                    "sequence": sequence,
                    "length": len(sequence),
                    "bbb_label": int(label),
                    "source_db": source_db.value,
                    "split": split.value,
                    "source_split": split.value,
                    "label_tier": label_tier.value,
                }
            )
        return rows

    def _load_b3pred_d1(self) -> pd.DataFrame:
        files = self._ensure_b3pred_files()
        rows: list[dict] = []
        rows += self._load_fasta(
            files["b3pred_pos_train.fa"],
            BbbLabel.POSITIVE,
            Split.TRAIN,
            SourceDb.B3PRED_D1,
            LabelTier.GOLD,
        )
        rows += self._load_fasta(
            files["b3pred_pos_val.fa"],
            BbbLabel.POSITIVE,
            Split.VAL,
            SourceDb.B3PRED_D1,
            LabelTier.GOLD,
        )
        rows += self._load_fasta(
            files["b3pred_neg_train.fa"],
            BbbLabel.NEGATIVE,
            Split.TRAIN,
            SourceDb.B3PRED_D1,
            LabelTier.GOLD,
        )
        rows += self._load_fasta(
            files["b3pred_neg_val.fa"],
            BbbLabel.NEGATIVE,
            Split.VAL,
            SourceDb.B3PRED_D1,
            LabelTier.GOLD,
        )
        return pd.DataFrame(rows)

    def _load_b3pdb(self) -> pd.DataFrame:
        path = self.b3pdb_path
        if path is None or not path.exists():
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
                "bbb_label": int(BbbLabel.POSITIVE),
                "source_db": SourceDb.B3PDB.value,
                "split": Split.TRAIN.value,
                "source_split": Split.TRAIN.value,
                "label_tier": LabelTier.GOLD.value,
                "assay_method": df[cols["method"]] if "method" in cols else None,
                "reference": df[cols["reference"]] if "reference" in cols else None,
                "organism": df[cols["organism"]] if "organism" in cols else None,
            }
        )
        out["length"] = out["sequence"].str.len()
        return out

    def _load_brainpeps(self) -> pd.DataFrame:
        path = self.brainpeps_path
        if path is None or not path.exists():
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
                "source_db": SourceDb.BRAINPEPS.value,
                "split": Split.TRAIN.value,
                "source_split": Split.TRAIN.value,
                "label_tier": LabelTier.SILVER.value,
                "assay_method": df[cols["assay_method"]] if "assay_method" in cols else None,
                "reference": df[cols["reference"]] if "reference" in cols else None,
                "organism": df[cols["organism"]] if "organism" in cols else None,
            }
        )
        out["length"] = out["sequence"].str.len()
        return out


def load_b3pred_d1(raw_dir: Path) -> pd.DataFrame:
    return SourceRegistry(raw_dir=raw_dir).load(SourceDb.B3PRED_D1)


def load_b3pdb(path: Path) -> pd.DataFrame:
    return SourceRegistry(raw_dir=path.parent, b3pdb_path=path).load(SourceDb.B3PDB)


def load_brainpeps(path: Path) -> pd.DataFrame:
    return SourceRegistry(raw_dir=path.parent, brainpeps_path=path).load(SourceDb.BRAINPEPS)
