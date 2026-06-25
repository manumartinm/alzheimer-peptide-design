# BBB training data

Training data for `bbb_models` comes from the public Hugging Face dataset [`manumartinm/bbb-peptides`](https://huggingface.co/datasets/manumartinm/bbb-peptides) (825 peptides with Boltz structures and physicochemical features).

## Download

From `packages/bbb_models`:

```bash
uv run python scripts/data/download.py
```

This writes a local cache under `data/bbb-peptides/` (gitignored):

| Path | Description |
|------|-------------|
| `peptides.parquet` | Main table (labels, splits, descriptors, structure paths) |
| `structures/<hash>/coords.npz` | CA coordinates + per-residue pLDDT |
| `structures/<hash>/structure.cif` | Boltz predicted structure |

Training auto-downloads on first run if the cache is missing (`configs/data.yaml` → `dataset_repo`).

## Vast.ai

Remote training uploads `packages/bbb_models/data/bbb-peptides/` via `infra/vast/bbb_models/upload_workspace.sh`. Download locally before uploading.

## Source pipeline

The dataset is built with `packages/dataset` (`tfg-bbb-build`, `tfg-bbb-export-hf`) and published to Hugging Face. To regenerate from scratch, see [`../../docs/data/dataset-pipeline.md`](../../docs/data/dataset-pipeline.md).
