# alzheimer-peptide-design

Monorepo for *in silico* design of BBB-compatible cyclic phosphomimetic peptides that modulate GSK3β (Alzheimer's disease target).

## Quick start

```bash
git clone --recurse-submodules https://github.com/YOUR_ORG/alzheimer-peptide-design.git
cd alzheimer-peptide-design
uv sync
```

## Packages

| Package | Path | Role |
|---------|------|------|
| `bbb-dataset` | `packages/dataset/` | BBB peptide dataset curation (`bbb-dataset-build`) |
| `bbb-models` | `packages/bbb_models/` | Sequence + structural BBB classifiers |
| `boltzgen-design` | `packages/boltzgen_design/` | GSK3β design orchestration (guidance, filters) |
| `boltzgen` | `packages/boltzgen/` | Diffusion engine (git submodule fork) |

## Data

Large datasets are **not** committed. Download from Hugging Face or regenerate locally:

```bash
cd packages/dataset
uv run bbb-dataset-build
uv run bbb-dataset-export-hf --variant full
```

See [`packages/dataset/data/README.md`](packages/dataset/data/README.md) and the HF dataset [`manumartinm/bbb-peptides`](https://huggingface.co/datasets/manumartinm/bbb-peptides).

## Documentation

Full architecture and pipelines: [`docs/architecture/overview.md`](docs/architecture/overview.md).

## Remote GPU (Vast.ai)

Shared helpers live in `infra/vast/_common.sh`. See [`docs/infrastructure/vast-training.md`](docs/infrastructure/vast-training.md).
