# BoltzGen Design (GSK3β)

This package orchestrates the TFG-specific GSK3β workflow without forking the whole `boltzgen` codebase.

## What this contains

- `configs/`: target and campaign configuration.
- `guidance/`: geometric hotspot/ATP guidance helpers and BBB diffusion feat mapping.
- `scoring/`: BBB oracle wrapper against `TFG/bbb_models`.
- `td3b/`: reward and weighted re-ranking utilities (MVP).
- `filtering/`: 5-gate filtering and Pareto selection.
- `scripts/`: executable entry points for each phase.

## Quick start

0. Prepare kinase target (interactive notebook):

```bash
jupyter notebook boltzgen_design/notebooks/01_prepare_kinase.ipynb
```

### Vast.ai (A100 + PyPI boltzgen)

Three scripts in `scripts/vast/`:

```bash
pip install vastai && vastai set api-key YOUR_KEY

# 1. Rent A100 + upload gsk3b.cif + design yaml (2 files)
bash boltzgen_design/scripts/vast/launch.sh

# 2. pip install boltzgen + run campaign on the instance
SMOKE=1 bash boltzgen_design/scripts/vast/run_campaign.sh
bash boltzgen_design/scripts/vast/run_campaign.sh

# 3. Download results
bash boltzgen_design/scripts/vast/sync_results.sh
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
