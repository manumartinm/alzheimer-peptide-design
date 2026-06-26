#!/usr/bin/env bash
# Full G1–G6 cascade on gsk3b_guided final designs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CAMPAIGN_DIR="${REPO_ROOT}/packages/boltzgen/workbench/gsk3b_guided"
PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON=python3
fi

cd "${REPO_ROOT}"
exec "${PYTHON}" packages/boltzgen_design/scripts/run_filter_cascade.py \
  --campaign-dir "${CAMPAIGN_DIR}" \
  --download-bbb \
  --output-csv "${CAMPAIGN_DIR}/gated_final_designs.csv"
