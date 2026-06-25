# BoltzGen Design (GSK3β)

This package orchestrates the TFG-specific GSK3β workflow without forking the whole `boltzgen` codebase.

## What this contains

- `configs/`: target and campaign configuration.
- `guidance/`: geometric hotspot/ATP guidance helpers and BBB diffusion feat mapping.
- `scoring/`: BBB oracle wrapper against `packages/bbb_models`.
- `td3b/`: reward and weighted re-ranking utilities (MVP).
- `filtering/`: 5-gate filtering and Pareto selection.
- `scripts/`: executable entry points for each phase.

## Quick start

0. Prepare kinase target (interactive notebook):

```bash
jupyter notebook boltzgen_design/notebooks/01_prepare_kinase.ipynb
```

### Vast.ai (A100 + PyPI boltzgen)

Scripts in `infra/vast/boltzgen_design/`:

```bash
pip install vastai && vastai set api-key YOUR_KEY

# 1. Upload gsk3b.cif + design yaml
bash infra/vast/boltzgen_design/launch.sh <INSTANCE_ID>

# 2. pip install boltzgen + run campaign on the instance
SMOKE=1 bash infra/vast/boltzgen_design/run_campaign.sh <INSTANCE_ID>
bash infra/vast/boltzgen_design/run_campaign.sh <INSTANCE_ID>

# 3. Download results
bash infra/vast/boltzgen_design/sync_results.sh <INSTANCE_ID>
```

1. Prepare target and masks:

```bash
uv run python boltzgen_design/scripts/run_baseline_design.py \
  --config boltzgen_design/configs/design_campaign.yaml \
  --output boltzgen/workbench/gsk3b_baseline
```

1. Score BBB:

```bash
uv run python boltzgen_design/scripts/run_filter_cascade.py \
  --input-dir boltzgen/workbench/gsk3b_baseline/final_ranked_designs \
  --output-csv boltzgen/workbench/gsk3b_baseline/gated.csv
```
