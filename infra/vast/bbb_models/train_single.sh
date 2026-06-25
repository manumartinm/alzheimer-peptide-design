#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"

EXP="${1:-configs/experiments/exp03_esm_tab_mlp.yaml}"
MODE=classifier EXP="${EXP}" bash "${VAST_DIR}/run_train.sh" "${2:-}"
