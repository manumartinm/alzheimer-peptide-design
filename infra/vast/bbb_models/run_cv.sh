#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"

instance_id="$(resolve_instance_id "${1:-}")"
require_vast_session

EXP="${EXP:-configs/experiments/exp09_struct_egnn_noise.yaml}"
DATA_CONFIG="${DATA_CONFIG:-configs/data.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train_cv.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-artifacts/cv}"
CALIBRATION="${CALIBRATION:-isotonic}"
DATASET_PATH="${DATASET_PATH:-}"
FORCE_CPU="${FORCE_CPU:-0}"

log_file="/workspace/output/$(basename "${EXP%.yaml}")_cv.log"
remote_cmd="set -euo pipefail
cd ${REMOTE_ROOT}/packages/bbb_models
if [[ \"${FORCE_CPU}\" == \"1\" ]]; then export CUDA_VISIBLE_DEVICES=\"\"; fi
cmd=(python3 -m bbb_geo cv --exp ${EXP} --data-config ${DATA_CONFIG} --train-config ${TRAIN_CONFIG} --output-root ${OUTPUT_ROOT} --calibration ${CALIBRATION})
if [[ -n \"${DATASET_PATH}\" ]]; then cmd+=(--dataset-path ${DATASET_PATH}); fi
nohup \"\${cmd[@]}\" > ${log_file} 2>&1 &
echo \$! > /workspace/output/last_cv.pid
echo \"PID=\$(cat /workspace/output/last_cv.pid)\"
echo \"LOG=${log_file}\"
"

vast_ssh "${instance_id}" bash -s <<EOF
${remote_cmd}
EOF

echo "Remote CV launched."
echo "Monitor with: bash ${VAST_DIR}/monitor.sh ${instance_id} ${log_file}"
