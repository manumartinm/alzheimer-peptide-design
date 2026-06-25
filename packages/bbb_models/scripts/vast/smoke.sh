#!/usr/bin/env bash
# shellcheck source=_vast_env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_vast_env.sh"

instance_id="${1:-}"
MODE=classifier \
EXP=configs/experiments/exp01_tabular_lgbm.yaml \
OUTPUT_ROOT=artifacts/smoke \
NO_RESUME=1 \
bash "${VAST_DIR}/run_train.sh" "${instance_id}"
