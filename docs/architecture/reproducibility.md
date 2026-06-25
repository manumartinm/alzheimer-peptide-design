# Reproducibility

## v1 (current — Jun 2026)

### What is covered

| Component | Reproducibility |
|-----------|-----------------|
| Gold dataset | `bbb-dataset-build` + parquets in `packages/dataset/data/processed/` |
| HF release | `bbb-dataset-export-hf --variant full` → `data/hf_release/` (825 rows) |
| Classifier | `bbb-classifier train` + configs + `train_metadata.json` |
| Geo EGNN | `bbb-geo train` + `exp09` + `train_geo.yaml` |
| CV | `bbb-classifier cv`, `bbb-geo cv` |
| Remote GPU | `infra/vast/bbb_models/` + `sync_artifacts.sh` |

### Minimal commands

```bash
# Environment
uv sync

# Dataset
cd packages/dataset && uv run bbb-dataset-build
uv run bbb-dataset-export-hf --variant full

# Classifier (local)
cd ../bbb_models
uv run python bbb-classifier train \
  --exp configs/experiments/exp03_esm_tab_mlp.yaml

# Geo (local)
uv run python bbb-geo train \
  --exp configs/experiments/exp09_struct_egnn_noise.yaml \
  --train-config configs/train_geo.yaml

# Tests
uv run pytest
```

### Artifacts (not in git)

- `packages/bbb_models/artifacts/` — checkpoints, metrics, local MLflow
- `packages/dataset/boltz-experiments/` — Boltz folding runs
- `packages/dataset/data/hf_release/structures/` — coords + CIF (large)

Policy: do not commit checkpoints; use `sync_artifacts.sh` from Vast or local ignored paths.

### Tracking

- MLflow: `packages/bbb_models/mlflow.db` (local)
- TensorBoard: under each run directory
- Post-geo: `metrics_multisigma.json`, `guidance_gate.json`

## v2 (roadmap)

> Goal: `uv sync` + `dvc pull` + API keys → regenerate the full pipeline.

| Task | Package |
|------|---------|
| `dataset/dvc.yaml`: fetch_raw → build → augment → fold → export_hf | `packages/dataset/` |
| Fix paths in `bbb_models/dvc.yaml` + geo/CV stages | `packages/bbb_models/` |
| DVC remote (S3/GDrive) for raw, structures, checkpoints | root |
| SHA256 checksums on B3Pred downloads | `packages/dataset/src/bbb_dataset/sources.py` |
| DATA_CARD with provenance (git SHA + dvc.lock) | `packages/dataset/` |
| CI: `dvc repro --dry` | `.github/workflows/` |

### Planned commands (v2)

```bash
uv sync
dvc pull                    # download cached data and models
dvc repro build_gold        # regenerate dataset from raw
dvc repro train             # re-train classifier
dvc repro train_geo         # re-train EGNN
```

See [vast-training.md](../infrastructure/vast-training.md) for remote campaigns while v2 is incomplete.
