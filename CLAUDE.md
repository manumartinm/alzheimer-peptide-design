# CLAUDE.md

Guidance for Claude Code when working in the **alzheimer-peptide-design** monorepo.

## Project

Bachelor's thesis (TFG): *in silico* design of BBB-compatible cyclic phosphomimetic peptides that modulate **GSK3β** for Tau hyperphosphorylation (Alzheimer's). The peptides must engage substrate-recognition hotspots (R96, R180, K205), avoid the ATP cleft, and cross the blood-brain barrier.

**Source of truth for architecture and SOPs:** [`docs/architecture/agent-context.md`](docs/architecture/agent-context.md)

## TFG sibling workspace

A parallel checkout at `ProteinDesign/TFG/` (flat layout, same packages). Use it for local trained artifacts (`bbb_models/artifacts/models/exp09_*`), thesis deliverables (`entregas/`), and active Vast instance state. **Write new code in this repo** (`alzheimer-peptide-design/`).

| TFG path | Canonical path |
|----------|----------------|
| `TFG/bbb_models/` | `packages/bbb_models/` |
| `TFG/dataset/` | `packages/dataset/` |
| `TFG/boltzgen_design/` | `packages/boltzgen_design/` |
| `TFG/boltzgen/` | `packages/boltzgen/` |
| `TFG/docs/AGENT_CONTEXT.md` | `docs/architecture/agent-context.md` |

TFG commands use the same relative paths without the `packages/` prefix:

```bash
cd TFG
uv run python bbb_models/scripts/geo/train.py --exp configs/experiments/exp09_struct_egnn_noise.yaml
bash boltzgen_design/scripts/vast/run_guided_campaign.sh
```

## Repository layout

```
alzheimer-peptide-design/
├── packages/
│   ├── dataset/           # tfg-bbb-dataset — data curation CLI
│   ├── bbb_models/        # bbb_classifier + bbb_geo
│   ├── boltzgen_design/   # GSK3β orchestration (guidance, filters, TD3B)
│   └── boltzgen/          # diffusion engine (git submodule)
├── docs/                  # architecture, models, data, infrastructure
├── infra/                 # shared Vast.ai helpers (infra/vast/_common.sh)
├── pyproject.toml         # uv workspace root
└── Makefile               # lint, format, typecheck, test
```

## Setup

```bash
git clone --recurse-submodules <repo-url>
cd alzheimer-peptide-design
uv sync --group dev
make pre-commit   # optional: install hooks
```

Python **3.11 or 3.12** only. Use the root uv workspace — never create per-package venvs.

## Common commands

```bash
make lint          # ruff check + format --check
make format        # ruff fix + format
make typecheck     # mypy on workspace packages (not boltzgen submodule)
make test          # pytest across packages

# Dataset pipeline
uv run tfg-bbb-build
uv run tfg-bbb-export-hf --variant full

# BBB classifier training
uv run python packages/bbb_models/scripts/classifier/train.py \
  --exp configs/experiments/exp03_esm_tab_mlp.yaml

# Geometry EGNN (diffusion guidance)
uv run python packages/bbb_models/scripts/geo/train.py \
  --exp configs/experiments/exp09_struct_egnn_noise.yaml \
  --train-config configs/train_geo.yaml

# Geo guidance gate check
uv run python packages/bbb_models/scripts/geo/probe.py \
  --run-dir packages/bbb_models/artifacts/models/exp09_struct_egnn_noise \
  --manifest packages/dataset/data/processed/peptides_struct_manifest_synthetic.parquet \
  --dataset packages/dataset/data/processed/peptides_bbb_preview.csv

# GSK3β design + filtering
uv run python packages/boltzgen_design/scripts/run_baseline_design.py \
  --config packages/boltzgen_design/configs/design_campaign.yaml \
  --output packages/boltzgen/workbench/gsk3b_baseline

uv run python packages/boltzgen_design/scripts/run_filter_cascade.py \
  --input-dir packages/boltzgen/workbench/gsk3b_baseline/final_ranked_designs \
  --output-csv packages/boltzgen/workbench/gsk3b_baseline/gated.csv
```

Run scripts from the package directory when configs use relative paths (e.g. `cd packages/bbb_models` before training), or pass absolute paths.

## Package roles

### `packages/dataset` (`tfg_bbb`)

- CLI entry points: `tfg-bbb-build`, `tfg-bbb-augment`, `tfg-bbb-fold`, `tfg-bbb-export-hf`
- Pipeline modules in `src/tfg_bbb/` (sources, clean, augment, folding, splits, eda)
- Gold dataset: `packages/dataset/data/processed/peptides_bbb.parquet` (build pipeline); **training cache:** `packages/bbb_models/data/bbb-peptides/` from [`manumartinm/bbb-peptides`](https://huggingface.co/datasets/manumartinm/bbb-peptides)
- Docs: [`docs/data/dataset-pipeline.md`](docs/data/dataset-pipeline.md)

### `packages/bbb_models` (`bbb_classifier`, `bbb_geo`)

- **bbb_classifier:** tabular/ESM oracle (`exp03_esm_tab_mlp`) — calibrated `p_bbb_calibrated` for gating and TD3B reward
- **bbb_geo:** geometry-only EGNN (`exp09_struct_egnn_geo`) — differentiable BBB guidance in diffusion
- Logic in `src/*/pipeline/`; thin scripts in `scripts/classifier/` and `scripts/geo/`
- Docs: [`docs/models/bbb-classifier.md`](docs/models/bbb-classifier.md), [`docs/models/structural-classifier.md`](docs/models/structural-classifier.md)

### `packages/boltzgen_design`

- Target prep, geometric guidance helpers, BBB oracle wrapper, TD3B utilities, 5-gate filtering, Pareto selection
- Filtering gates: `filtering/gates.py`
- Vast campaign scripts: `infra/vast/boltzgen_design/`
- Docs: [`docs/design/rl-md-strategy.md`](docs/design/rl-md-strategy.md)

### `packages/boltzgen` (submodule)

- Upstream diffusion engine fork (`manumartinm/boltzgen`, branch `add-bbb-head-and-md`)
- Has its own [`packages/boltzgen/CLAUDE.md`](packages/boltzgen/CLAUDE.md) for BoltzGen internals
- Excluded from root ruff/mypy. Modify only when wiring BBB guidance or MD hooks
- BBB guidance overrides: `guidance.bbb_weight`, `guidance.bbb_ckpt`, `guidance.feats_json`

## Critical architectural constraints

1. **Non-differentiability:** ESM-2 and tabular physicochemical features cannot be backpropagated through 3D coordinates. Use TD3B/RL for sequence-based BBB optimization; use `bbb_geo` EGNN for coordinate-differentiable guidance.

2. **Guidance vs reward split:**
   - SDE gradients: hotspot potential, ATP repulsion, `p_geo` (struct EGNN), membrane potential
   - Post-generation: `exp03` oracle for G3 gate, TD3B reward, Pareto ranking

3. **Geo training stability:** If many `skipped non-finite batches`, lower `coord_sigma_cap` / `aux_weight` or run `scripts/geo/sweep_stability.py` before long Vast jobs.

4. **DVC:** When a `dvc.yaml` stage exists, prefer `dvc repro` over ad-hoc script runs (see [`docs/architecture/reproducibility.md`](docs/architecture/reproducibility.md)).

## Code conventions

- Ruff: py311, line-length 100. Pre-commit runs ruff + mypy.
- New modules: `from __future__ import annotations`
- Keep scripts thin; implement in `src/*/pipeline/`
- Configs in YAML under `configs/`; experiment IDs like `exp03_esm_tab_mlp`, `exp09_struct_egnn_noise`
- Do not commit artifacts, checkpoints, raw data, or workbench outputs (see `.gitignore`)

## Remote GPU (Vast.ai)

Shared helpers: `infra/vast/_common.sh` (resolves SSH via `vastai show instance`, sets `REPO_ROOT`/`REMOTE_ROOT`).

### BBB model training

```bash
bash infra/vast/bbb_models/launch.sh
SMOKE=1 bash infra/vast/bbb_models/run_train.sh <INSTANCE_ID>
bash infra/vast/bbb_models/sync_artifacts.sh <INSTANCE_ID>
```

Upload: `bbb_models/` + `bbb_models/data/bbb-peptides/` only. Docs: [`docs/infrastructure/vast-training.md`](docs/infrastructure/vast-training.md).

### Guided BoltzGen campaign (GSK3β)

Target assets in `packages/boltzgen_design/targets/gsk3b/` (`gsk3b.cif`, `gsk3b_peptide_design.yaml`, `guidance_feats.json`).

```bash
bash infra/vast/boltzgen_design/launch.sh <INSTANCE_ID>
bash infra/vast/boltzgen_design/setup_guided_env.sh <INSTANCE_ID>
# Optional: BBB_CKPT_LOCAL=/abs/path/best.ckpt setup_guided_env.sh ...
SMOKE=1 bash infra/vast/boltzgen_design/run_guided_campaign.sh <INSTANCE_ID>
bash infra/vast/boltzgen_design/sync_results.sh <INSTANCE_ID> gsk3b_guided_smoke
```

Key env vars: `NUM_DESIGNS`, `BBB_CKPT`, `GUIDANCE_FEATS_JSON`, `USE_KERNELS` (auto; force `false` on Blackwell/B200), `REUSE=1`, `ATTACH=1` (tmux attach).

Remote installs editable `boltzgen` + `bbb_models`; runs `boltzgen run` with `guidance.bbb_ckpt` and `guidance.feats_json`.

## Testing and CI

CI (`.github/workflows/ci.yml`): ruff, mypy, pytest on Python 3.11 and 3.12 with recursive submodules.

```bash
make test
uv run pytest packages/dataset/tests -q --cov=tfg_bbb --cov-fail-under=85
```

## When changing architecture

Update [`docs/architecture/agent-context.md`](docs/architecture/agent-context.md) and related docs under `docs/` so future agents reflect the new pipeline reality.

## Key references

| Topic | Document |
|-------|----------|
| Full agent SOPs | `docs/architecture/agent-context.md` |
| TFG flat docs (legacy) | `../TFG/docs/AGENT_CONTEXT.md`, `../TFG/docs/STRUCTURAL_BBB_GUIDANCE.md` |
| Math (SDE, TD3B, gates) | `docs/architecture/theoretical-framework.md` |
| RL + MD outer loop | `docs/design/rl-md-strategy.md` |
| Structural BBB guidance | `docs/models/structural-bbb-guidance.md` |
| BoltzGen internals | `packages/boltzgen/CLAUDE.md` |
