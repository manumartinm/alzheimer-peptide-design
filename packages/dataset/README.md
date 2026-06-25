# BBB Peptide Dataset (`bbb-dataset`)

Curated blood–brain barrier (BBB) permeability peptide dataset: multi-source ingestion, cleaning, physicochemical features, cluster-aware CV splits, optional augmentation, and Boltz structure folding.

## Layout

```
packages/dataset/
  src/bbb_dataset/     # Python package
  notebooks/eda.ipynb  # interactive EDA (not part of CLI)
  data/raw/            # source files + downloaded B3Pred FASTAs
  data/processed/      # parquet outputs
  configs/             # augmentation + folding YAML
```

## Pipeline

Implemented in `src/bbb_dataset` and orchestrated from the CLI:

- `bbb-dataset-build` — build gold dataset, optional augmentation and folding
- `bbb-dataset-augment` — augmentation only
- `bbb-dataset-fold` — Boltz folding / manifest rebuild
- `bbb-dataset-export-hf` — Hugging Face release bundle

### Modules

- `bbb_dataset.sources` — load and normalize source datasets (`SourceRegistry`)
- `bbb_dataset.cleaning` — filtering, conflicts, identity deduplication (`SequenceCleaner`)
- `bbb_dataset.features` — physicochemical features (`FeatureComputer`)
- `bbb_dataset.augmentation` — conservative sequence perturbation (`Augmenter`)
- `bbb_dataset.folding` — Boltz API folding (`StructureFolder`)
- `bbb_dataset.splits` — cluster-aware anti-leakage folds (`FoldSplitter`)
- `bbb_dataset.builder` — end-to-end orchestration (`DatasetBuilder`)
- `bbb_dataset.schema` — output schema validation (`DatasetSchema`)

### EDA

Exploratory analysis lives in [`notebooks/eda.ipynb`](notebooks/eda.ipynb). It reads processed parquets and writes figures under `data/processed/eda_figures/`.

## Quick start

```bash
cd packages/dataset
uv sync
uv run bbb-dataset-build --skip-augment --skip-fold
```

## Tests

```bash
uv run pytest --cov=bbb_dataset --cov-report=term-missing --cov-fail-under=85
```

## CLI examples

```bash
# Full pipeline (augment + fold when BOLTZ_API_KEY is set)
uv run bbb-dataset-build

# Gold only
uv run bbb-dataset-build --skip-augment --skip-fold

# Augmentation only
uv run bbb-dataset-augment

# Folding (requires BOLTZ_API_KEY in .env.local)
uv run bbb-dataset-fold

# Rebuild manifest from existing Boltz runs (no API)
uv run bbb-dataset-fold --manifest-only

# Hugging Face export
uv run bbb-dataset-export-hf --variant gold
uv run bbb-dataset-export-hf --variant full
```

Configure `configs/folding.yaml` (`limit`, `max_workers`, `model`, `experiments_dir`). `bbb-dataset-fold` folds unique sequences from the augmented dataset; with `resume: true` it skips sequences that already have a succeeded run under `boltz-experiments/bbb-fold-<hash>/`.
