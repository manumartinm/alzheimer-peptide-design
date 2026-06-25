# Dataset Pipeline Documentation

This document explains the current `packages/dataset` pipeline end-to-end.

## Scope and Purpose

The dataset pipeline creates a curated BBB peptide dataset for downstream modeling in `packages/bbb_models`.

Primary goals:
- build a clean peptide table with BBB labels (`BBB+` / `BBB-`);
- enrich rows with physicochemical descriptors;
- save reproducible parquet outputs ready for modeling.

## Project Layout

Dataset project root: `packages/dataset`

- `data/raw`: original downloaded sources.
- `data/interim`: temporary outputs per pipeline step.
- `data/processed`: final parquet tables used by training.
- `src/tfg_bbb/cli/build.py`: full pipeline entry point (`tfg-bbb-build`).
- `src/tfg_bbb`: reusable Python helpers used by the CLI.

## Running the pipeline

From the dataset project root:

```bash
uv run tfg-bbb-build
```

Flags: `--skip-eda`, `--skip-augment`, `--skip-fold`, `--base-dir`, `--augment-config`, `--fold-config`.

## Python Package Utilities

Core code in `src/tfg_bbb`:

- `features.py`
  - `compute_features(seq)`: computes feature dictionary from sequence.
  - `add_feature_columns(df)`: vectorizes feature extraction over a dataframe.
  - uses BioPython, modlamp, and pyteomics.
- `clean.py`
  - `filter_sequences(...)`: configurable length and canonical-aa filtering.
  - `resolve_label_conflicts(...)`: removes conflicting labels per sequence.
  - `deduplicate_by_identity(...)`: CD-HIT/mmseqs (fallback python) clustering and representative selection.
- `sources.py`
  - loaders for B3Pred D1, optional B3Pdb and Brainpeps metadata tables.
- `splits.py`
  - cluster-aware folds with `StratifiedGroupKFold`.
- `io.py`
  - `ensure_dirs(base_dir)`: guarantees `raw/interim/processed` directories.
  - `write_parquet` / `read_parquet`: consistent parquet I/O helpers.
- `pipeline.py`
  - `build_gold_dataset`, `run_eda`.
- `data_card.py`
  - generates `DATA_CARD.md` with source and filtering stats.

## Feature Set (Physicochemical + Composition)

The feature generator includes:

- physicochemical properties: molecular weight, pI, net charge, aliphatic index, boman index, GRAVY, instability index, aromaticity;
- composition ratios: hydrophobic/polar/basic/acidic/aromatic percentages;
- derived metrics: charge density, hydrophobic moment (Eisenberg scale), MW consistency (`mw_delta_abs`);
- extinction coefficients and additional computed descriptors.

This produces the tabular descriptors consumed by classifier experiments.

## Environment and Dependencies

Defined in `packages/dataset/pyproject.toml` (Python `>=3.11,<3.13`).

Recommended setup:

```bash
cd packages/dataset
uv sync --group dev
uv run pytest --cov=tfg_bbb --cov-report=term-missing --cov-fail-under=85
```

## Inputs, Outputs, and Handoff

Expected final handoff artifacts:

- `data/processed/peptides_bbb.parquet`
- `data/processed/peptides_bbb_with_augmentation.parquet` (gold + augmented)
- `data/processed/peptides_struct_manifest.parquet` (Boltz folds + metrics)
- `data/processed/DATA_CARD.md`

These tables are consumed by `packages/bbb_models` through configs (`configs/data.yaml`) and DVC stages.

### Hugging Face release (`hf_release`)

CLI: `uv run tfg-bbb-export-hf` (module `tfg_bbb.export_hf`, entry `tfg-bbb-export-hf`).

```bash
cd packages/dataset
uv run tfg-bbb-export-hf --variant gold    # gold only
uv run tfg-bbb-export-hf --variant full    # gold + augmented (825 rows, all with structure)
```

Output: `data/hf_release/`

| File | Description |
|------|-------------|
| `peptides.parquet` | Sequences, labels, physchem, Boltz metrics, relative structure paths |
| `structures/<hash>/coords.npz` | Cα coordinates + per-residue pLDDT |
| `structures/<hash>/structure.cif` | Boltz CIF when available |
| `README.md` | Hugging Face dataset card (`task_categories`: tabular-classification, graph-ml) |
| `stats.json` | Row counts and label balance |

Current release stats (`--variant full`): **825 rows**, 410 BBB+, 415 BBB-, all with structure.

Used by:
- geo training via `structure_coords_path` when `struct_manifest_path` is empty;
- Vast upload (`bbb_models/scripts/vast_launch.sh` requires local `hf_release/`).

Upload to Hugging Face:

```bash
huggingface-cli upload YOUR_USERNAME/bbb-peptides-tfg data/hf_release .
```

## Agent Usage Guide

If you are an agent working on this repository:

1. Use `tfg-bbb-build` as the single pipeline entrypoint.
2. Keep sequence cleaning constraints consistent with downstream training assumptions.
3. Preserve column names used by classifier configs (`sequence`, label/fold identifiers, feature columns).
4. If adding/removing features, update both:
   - dataset documentation;
   - classifier config that selects tabular columns.
5. Keep unit tests green with `pytest --cov` and maintain minimum 85% coverage.

## Current Status

Implemented and usable:
- structured dataset package;
- CLI build flow (`tfg-bbb-build`, `tfg-bbb-export-hf`);
- reusable feature, source, clean, split, schema, and data-card utilities;
- parquet outputs + HF release compatible with classifier and geo training.
