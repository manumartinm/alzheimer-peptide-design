# Remote Training on Vast.ai (bbb_models)

> Status: IMPLEMENTED (`infra/vast/bbb_models/` + `infra/vast/_common.sh`).

Train the BBB classifier or geometric EGNN on an **existing Vast instance** (created manually on the web). This workflow does not search offers or provision machines — it only uploads code + dataset and launches the job.

## What gets uploaded

| Local content | Remote destination |
|---------------|---------------------|
| `packages/bbb_models/` (excluding `artifacts/`, `.venv`, cached data) | `/workspace/alzheimer-peptide-design/packages/bbb_models/` |
| `packages/bbb_models/data/bbb-peptides/` | `/workspace/alzheimer-peptide-design/packages/bbb_models/data/bbb-peptides/` |

Geo training needs the HF cache: `peptides.parquet` + `structures/<hash>/coords.npz` (825 peptides). Download locally first:

```bash
cd packages/bbb_models
uv run python bbb-classifier download-data
```

Source: [`manumartinm/bbb-peptides`](https://huggingface.co/datasets/manumartinm/bbb-peptides).

## Main launcher

From the monorepo root:

```bash
bash infra/vast/bbb_models/launch.sh
bash infra/vast/bbb_models/upload_workspace.sh <INSTANCE_ID>
bash infra/vast/bbb_models/setup_instance.sh <INSTANCE_ID>
bash infra/vast/bbb_models/run_train.sh <INSTANCE_ID>
```

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODE` | `geo` | `geo` → `bbb-geo train`; `classifier` → `bbb-classifier train` |
| `EXP` | `exp09_struct_egnn_noise.yaml` | Experiment config |
| `TRAIN_CONFIG` | `train_geo.yaml` | Hyperparameters (geo uses conservative LR/grad_clip) |
| `DATA_CONFIG` | `data.vast.yaml` | Remote paths to HF release |
| `OUTPUT_ROOT` | `artifacts` | Remote output folder |
| `SMOKE=1` | — | Uses `train_smoke.yaml`, 2 epochs, output in `artifacts/smoke_geo` |
| `CV=1` | — | Runs `bbb-geo cv` with `train_cv.yaml` |
| `NO_RESUME=1` | active | Clean training (`--no-resume`) |
| `FORCE_CPU=1` | — | Force CPU if GPU is incompatible (e.g. RTX 5090 + old PyTorch) |

Examples:

```bash
SMOKE=1 bash infra/vast/bbb_models/run_train.sh 42405703
CV=1 bash infra/vast/bbb_models/run_cv.sh 42405703
MODE=classifier EXP=configs/experiments/exp03_esm_tab_mlp.yaml \
  TRAIN_CONFIG=configs/train.yaml \
  bash infra/vast/bbb_models/run_train.sh 42405703
```

## Helper scripts

All in `infra/vast/bbb_models/`:

| Script | Usage |
|--------|-------|
| `monitor.sh <INSTANCE_ID>` | Tail the most recent log in `/workspace/output/` |
| `sync_artifacts.sh <INSTANCE_ID>` | Download remote `artifacts/` to local |
| `status.sh` | Instance status |
| `setup_instance.sh` | Pip install on remote only |
| `upload_workspace.sh` | Upload only (no train) |
| `run_train.sh` / `run_cv.sh` | Train/CV on a prepared instance |
| `destroy.sh` | Destroy instance (destructive) |

Logs remote: `/workspace/output/<exp>_train.log` or `*_cv.log`. PIDs in `last_train.pid` / `last_cv.pid`.

## SSH and keys

Helpers in `infra/vast/_common.sh`:

- SSH user: **`root@`** (not `vastai@`).
- Identity: `~/.ssh/id_ed25519` (`VAST_SSH_IDENTITY` to override).
- Flag: `-o IdentitiesOnly=yes`.
- Prefer Vast proxy endpoint (`ssh9.vast.ai:PORT`) over direct IP.

Register the key in your Vast account with the **contents** of the `.pub` file, not the file path:

```bash
vastai create ssh-key "$(cat ~/.ssh/id_ed25519.pub)"
```

If the key was registered as a path (`/Users/.../id_ed25519.pub`), delete it and recreate. `ensure_vast_ssh_key` detects and fixes this automatically.

## Remote data config

`configs/data.vast.yaml`:

```yaml
dataset_path: /workspace/alzheimer-peptide-design/packages/bbb_models/data/bbb-peptides/peptides.parquet
dataset_root: /workspace/alzheimer-peptide-design/packages/bbb_models/data/bbb-peptides
struct_manifest_path: ""   # empty: geo resolves coords from structure_coords_path
```

`build_features` uses relative `structure_coords_path` in the parquet when no manifest is set.

## Known GPUs / limitations

| Hardware | Notes |
|----------|-------|
| A100 / H100 | Recommended for geo |
| RTX 5090 (sm_120) | PyTorch 2.4.x in the image may fail → `FORCE_CPU=1` or newer image |
| CPU | Valid for smoke; full training very slow (~hours) |

## Expected artifacts (geo)

After training, in `artifacts/models/exp09_struct_egnn_noise/` (or CV subfolder):

- `checkpoints/best.pt`, `checkpoints/last.pt`
- `metrics.json`, `metrics_multisigma.json`
- `guidance_gate.json`
- `val_predictions.parquet`, `train_metadata.json`
- isotonic calibrator (if enabled)

Download:

```bash
bash infra/vast/bbb_models/sync_artifacts.sh <INSTANCE_ID>
```

## Recommended workflow

1. Export HF release locally (`bbb-dataset-export-hf --variant full`).
2. Create Vast instance (A100/H100 GPU, disk ≥ 30 GB).
3. `SMOKE=1 bash infra/vast/bbb_models/run_train.sh <ID>` → verify SSH, data, and pip.
4. `bash infra/vast/bbb_models/run_train.sh <ID>` → full training.
5. `monitor.sh` until convergence; review warnings `skipped N non-finite batches`.
6. `sync_artifacts.sh` → consolidate locally.

See also [`infra/vast/README.md`](../../infra/vast/README.md) for BoltzGen campaign scripts.
