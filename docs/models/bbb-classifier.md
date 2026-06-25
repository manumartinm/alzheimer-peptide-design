# BBB Classifier Documentation

This document describes the implemented BBB classification system in `packages/bbb_models`, including architecture, experiment matrix, reproducibility flow, and practical usage for agents.

## Scope

The classifier predicts peptide BBB permeability and provides:

- `p_bbb_raw` (uncalibrated probability),
- `p_bbb_calibrated` (isotonic-calibrated probability),
- `decision` (thresholded class).

It is used both as:
- a hard filter for candidate triage;
- a soft scoring signal in later generative workflows.

## Project Layout

Classifier root: `packages/bbb_models`

- `configs/`
  - `data.yaml`: data schema and exclusions.
  - `train.yaml`: optimization and tracking settings.
  - `experiments/*.yaml`: experiment variants.
- `scripts/`
  - `classifier/` — `train.py`, `predict.py`, `evaluate.py`, `cv.py`, `sweep.py`
  - `geo/` — structural EGNN train/predict/probe/cv (see [structural-classifier.md](structural-classifier.md))
  - `data/prepare_data.py`
  - `vast/` — remote training helpers ([vast-training.md](../infrastructure/vast-training.md); scripts in `infra/vast/bbb_models/`)
- `src/bbb_classifier/`
  - `data/`: dataset, collate, split logic.
  - `features/`: ESM, tabular, 3D/graph feature modules.
  - `models/`: tabular/ESM/MLP/LGBM implementations.
  - `train/`: losses, metrics, calibration, checkpoints, engine.
  - `infer/`: ranking and prediction helpers.
- `src/bbb_geo/`
  - structural EGNN, membrane potential, diffusion guidance.
- `tests/`: smoke, dataset, and metrics tests.
- `dvc.yaml`: pipeline orchestration.

## DVC Pipeline

Defined in `dvc.yaml` with three stages:

1. `prepare`
   - reads `${data_input}`;
   - writes `data/processed/peptides_bbb.parquet`.
2. `train`
   - consumes processed data + configs + selected `${experiment}`;
   - writes model artifacts under `artifacts/models`.
3. `evaluate`
   - consumes `${run_dir}` checkpoint and processed dataset;
   - writes metrics file `${run_dir}/evaluation_metrics.json`.

## Data Configuration

From `configs/data.yaml`:

- `sequence_col`: `sequence`
- `label_col`: `bbb_label`
- `fold_col`: `fold_id`
- default dataset path: `/workspace/data/peptides_bbb.parquet`
- tabular feature exclusions include identifiers, metadata, and xref columns.

## Training Configuration

From `configs/train.yaml`:

- seed: `42`
- primary metric: `pr_auc` (maximize)
- batch size: `128`
- epochs: `30`
- optimizer defaults: `lr=1e-3`, `weight_decay=1e-4`
- gradient clipping: `1.0`
- calibration: `isotonic` enabled
- tracking: TensorBoard + MLflow enabled

## Supported Experiments

From `configs/experiments`:

1. `exp01_tabular_lgbm`
   - features: tabular only
   - model type: `tabular_lgbm`
2. `exp02_esm_lgbm`
   - features: ESM only
   - model type: `esm_lgbm`
3. `exp03_esm_tab_mlp`
   - features: ESM + tabular
   - model type: `esm_tab_mlp`
4. `exp04_esm_tab_3dfeat`
   - features: ESM + tabular + 3D features
   - model type: `esm_tab_3dfeat`
5. `exp05_esm_tab_gnn`
   - features: ESM + tabular + graph branch
   - model type: `esm_tab_gnn`

## Core Training Flow (`scripts/classifier/train.py`)

The training script performs:

1. load configs and dataset;
2. split train/val using `train_val_split` (optionally fold-aware);
3. build features according to experiment flags:
   - ESM embeddings (`batch_esm_embeddings`);
   - tabular matrix;
   - optional 3D features;
   - optional sequence graphs;
4. fit chosen model family:
   - LightGBM variants for classic baselines;
   - Torch models for fusion variants;
5. compute raw metrics;
6. fit isotonic calibrator on validation probs;
7. compute calibrated metrics;
8. save:
   - checkpoints,
   - calibrator,
   - metrics JSON,
   - validation predictions parquet,
   - `train_metadata.json` (critical for inference/evaluation reproducibility).

## Evaluation and Inference

### Evaluation (`scripts/classifier/evaluate.py`)

- reloads model and metadata from run directory;
- computes probabilities on provided dataset;
- optionally applies calibrator;
- writes `evaluation_metrics.json` and `evaluation_predictions.parquet`.

### Inference (`scripts/classifier/predict.py`)

- reads input candidates CSV/parquet;
- rebuilds features consistent with training metadata;
- predicts `p_bbb_raw` and `p_bbb_calibrated`;
- applies threshold decision;
- optional top-k ranking via `rank_candidates`.

Expected output fields:
- `sequence`
- `p_bbb_raw`
- `p_bbb_calibrated`
- `decision`

## Reproducibility and Tracking

- DVC controls stage-level reproducibility.
- MLflow logs experiment metadata and metrics.
- TensorBoard stores training curves under run artifacts.
- Checkpoint strategy:
  - `checkpoints/last.*` for resume;
  - `checkpoints/best.*` for deployment/inference.

## Agent Usage Guide

If you are an agent working on this classifier:

1. Prefer DVC stage execution over ad-hoc command chains.
2. Keep feature flags synchronized between:
   - experiment YAML,
   - feature construction in scripts,
   - model input dimensions.
3. Never run inference without `train_metadata.json`; it is required to reconstruct feature space correctly.
4. Treat calibration as mandatory when reporting probabilities used for ranking or gating.
5. If you add a new model type:
   - add experiment YAML;
   - implement model + loader path in train/evaluate/predict;
   - add at least one smoke-level test.

## Current Status

Implemented and operational:
- full training/evaluation/inference scripts;
- multiple experiment families (tabular, ESM, fusion, graph/3D variants);
- probability calibration;
- DVC-based reproducibility and tracking stack.

## Augmentation and Calibration Notes (2026-06)

- CV runs via `scripts/classifier/cv.py` and stores:
  - `cv_summary.json`
  - `cv_predictions.parquet`
  - `reliability.png`
- Geo CV: `scripts/geo/cv.py` (adds `metrics_multisigma.json`, `guidance_gate.json` per fold).
- Augmentation experiment:
  - `configs/experiments/exp06_esm_tab_mlp_aug.yaml`
  - pre-built sequence augmentation in `packages/dataset` + feature-space mixup.
- Dataset for geo (structures): `packages/dataset/data/hf_release/` or `peptides_struct_manifest.parquet`.

### Calibration policy for downstream reward

- Always use `p_bbb_calibrated` for ranking/gating in generative workflows.
- Compare calibration modes with CV:
  - `--calibration isotonic`
  - `--calibration platt`
  - `--calibration none`
- Select calibration by lowest mean `brier` and `ece` in `cv_summary.json`.
- Recommended operational threshold remains `0.6` for high-confidence BBB candidates, but final threshold should be chosen from CV PR tradeoffs.
