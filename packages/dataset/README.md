# TFG BBB peptide dataset

Dataset curation pipeline for BBB peptide classification used by `TFG/bbb_models`.

## What this builds

- `data/processed/peptides_bbb.parquet`: gold dataset for classifier training.
- `data/processed/peptides_bbb_augmented_extra.parquet`: sequence-augmented training rows only.
- `data/processed/peptides_bbb_with_augmentation.parquet`: gold + augmented rows for training.
- `data/processed/peptides_struct_manifest.parquet`: Boltz structures + confidence metrics.
- `data/processed/peptides_bbb_preview.csv`: quick preview export.
- `data/processed/DATA_CARD.md`: generated summary of sources and filtering stats.
- `data/processed/eda_figures/gold/`: pre-augmentation EDA + fold diagnostics
- `data/processed/eda_figures/augmentation/`: pre/post augmentation comparison

## Pipeline design

The pipeline is implemented in `src/tfg_bbb` and orchestrated from the CLI:

- `tfg-bbb-build` — build, EDA, augmentation, and optional folding

Main modules:

- `tfg_bbb.sources`: load and normalize source datasets.
- `tfg_bbb.clean`: filtering, conflict handling, identity deduplication.
- `tfg_bbb.features`: physicochemical feature generation.
- `tfg_bbb.augment`: conservative sequence perturbation for training augmentation.
- `tfg_bbb.folding`: Boltz API folding and structural manifest builder.
- `tfg_bbb.eda`: EDA plots and summary tables (gold, folds, augmentation).
- `tfg_bbb.splits`: cluster-aware anti-leakage folds.
- `tfg_bbb.pipeline`: end-to-end build orchestration and EDA.
- `tfg_bbb.cli`: command-line entry points (`tfg-bbb-build`, `tfg-bbb-augment`, `tfg-bbb-fold`).
- `tfg_bbb.schema`: output schema validation.

## Length policy

Default filter bounds are `6-30` amino acids (B3Pred/B3Pdb range). Bounds are configurable in `BuildConfig`.

## Quick start

```bash
cd /Users/manumartinm/Documents/ProteinDesign/TFG/dataset
uv sync --group dev
uv run pytest --cov=tfg_bbb --cov-report=term-missing --cov-fail-under=85
```

Run the full pipeline (gold → EDA → augmentation → EDA → optional folding):

```bash
uv run tfg-bbb-build
```

Partial runs:

```bash
# Skip EDA figures
uv run tfg-bbb-build --skip-eda

# Augmentation only
uv run tfg-bbb-augment

# Folding only (requires BOLTZ_API_KEY in .env.local)
uv run tfg-bbb-fold
```

### Data augmentation

After building the gold dataset, generate augmented training rows:

```bash
cd /Users/manumartinm/Documents/ProteinDesign/TFG/dataset
uv run tfg-bbb-augment
```

This writes:
- `peptides_bbb_augmented_extra.parquet` — synthetic rows only (`is_augmented=1`, `parent_peptide_id` set)
- `peptides_bbb_with_augmentation.parquet` — gold + augmented combined file

Configure perturbation probabilities in `configs/augmentation.yaml`. Augmentation is applied only to CV-train rows (`external_test=0` and `fold_id != 0`); validation fold and external holdout peptides are left untouched.

For classifier training with pre-built augmentation, point `bbb_models` `dataset_path` to `peptides_bbb_with_augmentation.parquet`.

### Hugging Face export

Build a self-contained release (parquet + structures + dataset card):

```bash
cd /Users/manumartinm/Documents/ProteinDesign/TFG/dataset
uv run tfg-bbb-export-hf --variant gold
# or: --variant full  (gold + augmented, 825 rows)
```

Writes to `data/hf_release/`:

- `peptides.parquet` — sequence, `bbb_label`, physicochemical descriptors, Boltz metrics, relative structure paths
- `structures/<sequence_hash>/coords.npz` — CA coordinates + per-residue pLDDT
- `structures/<sequence_hash>/structure.cif` — Boltz CIF (when available)
- `README.md` — Hugging Face dataset card
- `stats.json` — row counts

Upload:

```bash
huggingface-cli upload YOUR_USERNAME/bbb-peptides-tfg data/hf_release .
```

### Structure folding

Configure `configs/folding.yaml` (`limit`, `max_workers`, `model`, `experiments_dir`). `tfg-bbb-fold` folds unique sequences from the augmented dataset; with `resume: true` it skips sequences that already have a succeeded run under `boltz-experiments/bbb-fold-<hash>/`.
