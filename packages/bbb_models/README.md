# BBB Models

Reproducible BBB permeability modeling for peptide candidates used in BoltzGen ranking, filtering, and diffusion guidance.

## Packages (`src/`)

| Package | Role |
|---------|------|
| `bbb_classifier` | Tabular / ESM / MLP / LGBM / sequence-GNN classifiers |
| `bbb_geo` | Geometry-only EGNN (`p_geo`), membrane potential, BoltzGen guidance hook |

Both packages use flat modules (`run.py`, `features.py`, `models.py`, …) and a unified CLI.

## Supported experiments

- `exp01`–`exp06` → `bbb-classifier train`
- `exp09` → `bbb-geo train`

## Setup

```bash
uv sync
cd packages/bbb_models
uv run bbb-classifier download-data   # cache HF dataset locally
```

Training data: [`manumartinm/bbb-peptides`](https://huggingface.co/datasets/manumartinm/bbb-peptides) → `data/bbb-peptides/` (see [`data/README.md`](data/README.md)).

## Train

```bash
cd packages/bbb_models

# Tabular / ESM classifier
uv run bbb-classifier train \
  --exp configs/experiments/exp03_esm_tab_mlp.yaml

# Geometry EGNN (guidance model)
uv run bbb-geo train \
  --exp configs/experiments/exp09_struct_egnn_noise.yaml \
  --train-config configs/train_geo.yaml
```

Full documentation: [`../../docs/`](../../docs/README.md) — [structural-classifier.md](../../docs/models/structural-classifier.md), [vast-training.md](../../docs/infrastructure/vast-training.md).

Geo training writes `metrics_multisigma.json` and `guidance_gate.json`.

## Predict

```bash
uv run bbb-classifier predict \
  --run-dir artifacts/models/exp03_esm_tab_mlp \
  --input candidates.parquet \
  --output scored.parquet

uv run bbb-geo predict \
  --run-dir artifacts/models/exp09_struct_egnn_noise \
  --input candidates.parquet \
  --manifest ../dataset/data/processed/peptides_struct_manifest_synthetic.parquet \
  --output scored.parquet
```

## Geometry gate (before enabling diffusion guidance)

```bash
uv run bbb-geo probe \
  --run-dir artifacts/models/exp09_struct_egnn_noise \
  --manifest ../dataset/data/processed/peptides_struct_manifest_synthetic.parquet \
  --dataset ../dataset/data/processed/peptides_bbb_preview.csv
```

## Cross-validation

```bash
uv run bbb-classifier cv --exp configs/experiments/exp06_esm_tab_mlp_aug.yaml
uv run bbb-geo cv --exp configs/experiments/exp09_struct_egnn_noise.yaml
```

## Stability sweep (exp09)

```bash
uv run bbb-geo sweep-stability \
  --exp configs/experiments/exp09_struct_egnn_noise.yaml \
  --coord-sigma-caps 8,12,16 \
  --aux-weights 0.1,0.2,0.3 \
  --selection-metric low_sigma_mean_pr_auc_mean
```

## Vast.ai (upload and train on an existing instance)

From the monorepo root:

```bash
bash infra/vast/bbb_models/upload_workspace.sh <INSTANCE_ID>
bash infra/vast/bbb_models/setup_instance.sh <INSTANCE_ID>
SMOKE=1 bash infra/vast/bbb_models/run_train.sh <INSTANCE_ID>
CV=1 bash infra/vast/bbb_models/run_cv.sh <INSTANCE_ID>
FORCE_CPU=1 bash infra/vast/bbb_models/run_train.sh <INSTANCE_ID>
```

Monitor and sync:

```bash
bash infra/vast/bbb_models/monitor.sh <INSTANCE_ID>
bash infra/vast/bbb_models/sync_artifacts.sh <INSTANCE_ID>
```

## BoltzGen integration (bbb_geo guidance)

```bash
boltzgen run target.yaml \
  --output out/guided \
  --protocol peptide-anything \
  --config design guidance.bbb_weight=0.3 guidance.membrane_weight=0.7 \
  --config design guidance.bbb_ckpt=/abs/path/exp09_struct_egnn_noise/checkpoints/best.ckpt \
  --config design guidance.bbb_sigma_gate=4.0 guidance.max_force=1.0
```
