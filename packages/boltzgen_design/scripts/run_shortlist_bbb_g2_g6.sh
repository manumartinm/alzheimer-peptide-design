#!/usr/bin/env bash
# G2 (ATP) + G3 (BBB) + G6 (β vs α selectivity) → rank by iPTM.
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
  --require-gates g2,g3,g6 \
  --rank-by iptm \
  --download-bbb \
  --output-csv "${CAMPAIGN_DIR}/shortlist_bbb_g2_g6.csv"
