from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from .enums import ProcessedArtifact
from .folding import find_structure_cif, run_dir_for_sequence, sequence_hash
from .paths import PathLayout

STRUCT_METRIC_COLUMNS = [
    "plddt",
    "ptm",
    "structure_confidence",
    "iptm",
    "complex_iplddt",
    "complex_pde",
    "complex_ipde",
]

MANIFEST_COLUMNS = ["sequence_hash", *STRUCT_METRIC_COLUMNS]


@dataclass
class HFExportConfig:
    base_dir: Path
    variant: Literal["gold", "full"] = "gold"
    output_dir: Path | None = None
    include_cif: bool = True
    copy_structures: bool = True


@dataclass
class HuggingFaceExporter:
    config: HFExportConfig

    def export(self) -> dict[str, object]:
        return self._build_release()

    def _dataset_path(self) -> Path:
        layout = PathLayout(base_dir=self.config.base_dir)
        if self.config.variant == "full":
            return layout.artifact(ProcessedArtifact.COMBINED)
        return layout.artifact(ProcessedArtifact.PEPTIDES_BBB)

    def _manifest_path(self) -> Path:
        return PathLayout(base_dir=self.config.base_dir).artifact(ProcessedArtifact.STRUCT_MANIFEST)

    def _build_release(self) -> dict[str, object]:
        cfg = self.config
        base = cfg.base_dir.resolve()
        out_root = (cfg.output_dir or base / "data" / "hf_release").resolve()
        structures_out = out_root / "structures"
        structures_out.mkdir(parents=True, exist_ok=True)

        peptides_path = self._dataset_path()
        manifest_path = self._manifest_path()
        if not peptides_path.exists():
            raise FileNotFoundError(f"Missing peptides table: {peptides_path}")
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing structure manifest: {manifest_path}")

        peptides = pd.read_parquet(peptides_path)
        manifest = pd.read_parquet(manifest_path)
        merged = self._merge_peptides_with_structures(peptides, manifest)

        src_structures = base / "data" / "structures"
        experiments_dir = base / "boltz-experiments"

        coords_paths: list[str | None] = []
        cif_paths: list[str | None] = []
        if cfg.copy_structures:
            for _, row in merged.iterrows():
                copied = self._copy_structure_assets(
                    row,
                    src_structures_root=src_structures,
                    dst_structures_root=structures_out,
                    experiments_dir=experiments_dir,
                    include_cif=cfg.include_cif,
                )
                coords_paths.append(copied["structure_coords_path"])
                cif_paths.append(copied["structure_cif_path"])
        else:
            for _, row in merged.iterrows():
                seq_hash = str(
                    row.get("sequence_hash") or sequence_hash(str(row["sequence"]).upper())
                )
                coords_paths.append(self._relative_structure_path(seq_hash))
                cif_paths.append(self._relative_cif_path(seq_hash) if cfg.include_cif else None)

        export_df = merged.drop(columns=["coords_path"], errors="ignore")
        export_df["sequence_hash"] = export_df["sequence_hash"].fillna(
            export_df["sequence"].astype(str).str.upper().map(sequence_hash)
        )
        export_df["structure_coords_path"] = coords_paths
        export_df["structure_cif_path"] = cif_paths
        export_df["has_structure"] = export_df["structure_coords_path"].notna()

        parquet_path = out_root / "peptides.parquet"
        PathLayout.write_parquet(export_df, parquet_path)

        stats = {
            "variant": cfg.variant,
            "rows": len(export_df),
            "with_structure": int(export_df["has_structure"].sum()),
            "bbb_positive": int((export_df["bbb_label"] == 1).sum()),
            "bbb_negative": int((export_df["bbb_label"] == 0).sum()),
            "output_dir": str(out_root),
            "parquet_path": str(parquet_path),
        }
        (out_root / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        (out_root / "README.md").write_text(self._render_readme(export_df, stats), encoding="utf-8")
        return {"dataframe": export_df, "stats": stats, "output_dir": out_root}

    @staticmethod
    def _merge_peptides_with_structures(
        peptides: pd.DataFrame, manifest: pd.DataFrame
    ) -> pd.DataFrame:
        manifest = manifest.copy()
        manifest["sequence"] = manifest["sequence"].astype(str).str.upper()
        peptides = peptides.copy()
        peptides["sequence"] = peptides["sequence"].astype(str).str.upper()

        manifest_cols = [c for c in MANIFEST_COLUMNS if c in manifest.columns]
        struct = manifest[["sequence", *manifest_cols, "coords_path"]].drop_duplicates("sequence")
        return peptides.merge(struct, on="sequence", how="left", validate="m:1")

    @staticmethod
    def _relative_structure_path(sequence_hash_value: str) -> str:
        return f"structures/{sequence_hash_value}/coords.npz"

    @staticmethod
    def _relative_cif_path(sequence_hash_value: str) -> str:
        return f"structures/{sequence_hash_value}/structure.cif"

    def _copy_structure_assets(
        self,
        row: pd.Series,
        *,
        src_structures_root: Path,
        dst_structures_root: Path,
        experiments_dir: Path,
        include_cif: bool,
    ) -> dict[str, str | None]:
        seq = str(row["sequence"]).upper()
        seq_hash = str(row.get("sequence_hash") or sequence_hash(seq))
        src_coords = Path(str(row["coords_path"])) if pd.notna(row.get("coords_path")) else None

        out_dir = dst_structures_root / seq_hash
        out_dir.mkdir(parents=True, exist_ok=True)
        dst_coords = out_dir / "coords.npz"

        if src_coords is not None and src_coords.exists():
            if src_coords.resolve() != dst_coords.resolve():
                shutil.copy2(src_coords, dst_coords)
        elif not dst_coords.exists():
            fallback = src_structures_root / seq_hash / "coords.npz"
            if fallback.exists():
                shutil.copy2(fallback, dst_coords)

        cif_rel: str | None = None
        if include_cif:
            run_dir = run_dir_for_sequence(experiments_dir, seq)
            if run_dir.exists():
                try:
                    src_cif = find_structure_cif(run_dir)
                    dst_cif = out_dir / "structure.cif"
                    if src_cif.resolve() != dst_cif.resolve():
                        shutil.copy2(src_cif, dst_cif)
                    cif_rel = self._relative_cif_path(seq_hash)
                except FileNotFoundError:
                    cif_rel = None

        coords_rel = self._relative_structure_path(seq_hash) if dst_coords.exists() else None
        return {"structure_coords_path": coords_rel, "structure_cif_path": cif_rel}

    def _render_readme(self, df: pd.DataFrame, stats: dict[str, object]) -> str:
        chem_cols = [
            c
            for c in df.columns
            if c
            not in {
                "peptide_id",
                "source_id",
                "sequence",
                "sequence_hash",
                "structure_coords_path",
                "structure_cif_path",
                "has_structure",
                "bbb_label",
                "source_db",
                "split",
                "source_split",
                "label_tier",
                "is_gold",
                "cluster_id",
                "external_test",
                "fold_id",
                "is_augmented",
                "parent_peptide_id",
                "length",
                *STRUCT_METRIC_COLUMNS,
            }
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        return f"""---
license: mit
task_categories:
  - tabular-classification
  - graph-ml
tags:
  - peptide
  - blood-brain-barrier
  - bbb
  - protein-design
language:
  - en
size_categories:
  - n<1K
---

# BBB Peptide Dataset

Curated blood–brain barrier (BBB) permeability dataset for peptide sequences with Boltz-predicted 3D structures and physicochemical descriptors.

## Summary

| Field | Value |
|-------|-------|
| Rows | {stats["rows"]} |
| With structure | {stats["with_structure"]} |
| BBB+ | {stats["bbb_positive"]} |
| BBB− | {stats["bbb_negative"]} |
| Variant | `{stats["variant"]}` |

## Files

- `peptides.parquet` — one row per peptide: sequence, label, splits, physicochemical features, structure quality metrics, relative structure paths
- `structures/<sequence_hash>/coords.npz` — CA coordinates + per-residue pLDDT (`coords`, `sequence`, `plddt_per_residue`)
- `structures/<sequence_hash>/structure.cif` — Boltz predicted structure (when available)
- `stats.json` — export statistics

## Columns (main)

- **Identifiers:** `peptide_id`, `source_id`, `sequence`, `sequence_hash`, `length`
- **Label:** `bbb_label` (1 = BBB-permeable, 0 = non-permeable)
- **Splits:** `fold_id`, `cluster_id`, `external_test`, `split`, `source_split`, `is_gold`
- **Structure quality:** {", ".join(f"`{c}`" for c in STRUCT_METRIC_COLUMNS)}
- **Structure paths:** `structure_coords_path`, `structure_cif_path`, `has_structure`
- **Physicochemical:** {", ".join(f"`{c}`" for c in chem_cols[:12])}{"..." if len(chem_cols) > 12 else ""}

## Load with Hugging Face Datasets

```python
from datasets import load_dataset
import numpy as np
from pathlib import Path

repo_dir = "."  # clone of this dataset repo
ds = load_dataset("parquet", data_files=f"{{repo_dir}}/peptides.parquet", split="train")

row = ds[0]
coords = np.load(Path(repo_dir) / row["structure_coords_path"])
ca = coords["coords"]          # (L, 3) float32
seq = "".join(coords["sequence"])  # amino-acid string
plddt = coords["plddt_per_residue"]  # (L,) per-residue confidence
```

## Load with pandas

```python
import pandas as pd
df = pd.read_parquet("peptides.parquet")
```

## Source pipeline

Built with `bbb-dataset` (`bbb-dataset-build`, `bbb-dataset-fold`). Structures predicted with Boltz 2.1 via the Boltz API.

## Citation

If you use this dataset, cite the BBB peptide curation pipeline and Boltz structure prediction.
"""


def export_hf_dataset(
    *,
    base_dir: str | Path = ".",
    variant: Literal["gold", "full"] = "gold",
    output_dir: str | Path | None = None,
    include_cif: bool = True,
    copy_structures: bool = True,
) -> dict[str, object]:
    cfg = HFExportConfig(
        base_dir=Path(base_dir),
        variant=variant,
        output_dir=Path(output_dir) if output_dir else None,
        include_cif=include_cif,
        copy_structures=copy_structures,
    )
    return HuggingFaceExporter(cfg).export()
