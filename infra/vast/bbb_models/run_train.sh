#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"

instance_id="$(resolve_instance_id "${1:-}")"
require_vast_session

MODE="${MODE:-geo}" # geo | classifier
EXP="${EXP:-configs/experiments/exp09_struct_egnn_noise.yaml}"
DATA_CONFIG="${DATA_CONFIG:-configs/data.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train_geo.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts}"
DATASET_PATH="${DATASET_PATH:-}"
NO_RESUME="${NO_RESUME:-0}"
FORCE_CPU="${FORCE_CPU:-0}"

if [[ "${MODE}" == "geo" ]]; then
  MODULE="bbb_geo"
else
  MODULE="bbb_classifier"
fi

log_file="/workspace/output/$(basename "${EXP%.yaml}")_train.log"
remote_cmd="set -euo pipefail
cd ${REMOTE_ROOT}/packages/bbb_models
if [[ \"${FORCE_CPU}\" == \"1\" ]]; then export CUDA_VISIBLE_DEVICES=\"\"; fi
cmd=(python3 -m ${MODULE} train --exp ${EXP} --data-config ${DATA_CONFIG} --train-config ${TRAIN_CONFIG} --output-root ${OUTPUT_ROOT})
if [[ -n \"${DATASET_PATH}\" ]]; then cmd+=(--dataset-path ${DATASET_PATH}); fi
if [[ \"${NO_RESUME}\" == \"1\" ]]; then cmd+=(--no-resume); fi
nohup \"\${cmd[@]}\" > ${log_file} 2>&1 &
echo \$! > /workspace/output/last_train.pid
echo \"PID=\$(cat /workspace/output/last_train.pid)\"
echo \"LOG=${log_file}\"
"

vast_ssh "${instance_id}" bash -s <<EOF
${remote_cmd}
EOF

echo "Remote training launched."
echo "Monitor with: bash ${VAST_DIR}/monitor.sh ${instance_id}"
