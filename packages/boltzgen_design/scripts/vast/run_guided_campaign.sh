#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=_vast_env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_vast_env.sh"
# Run BoltzGen campaign on Vast using local editable boltzgen + bbb_geo guidance.
#
# Usage:
#   bash boltzgen_design/scripts/vast/run_guided_campaign.sh
#   bash boltzgen_design/scripts/vast/run_guided_campaign.sh <INSTANCE_ID>
#   SMOKE=1 bash boltzgen_design/scripts/vast/run_guided_campaign.sh
#
# Optional env:
#   NUM_DESIGNS=300
#   OUTPUT_BASENAME=gsk3b_guided
#   REUSE=1
#   SKIP_CHECK=1
#   BBB_CKPT=/workspace/campaign/bbb_geo_best.ckpt
#   GUIDANCE_FEATS_JSON=/workspace/campaign/guidance_feats.json
#   REFOLDING_RMSD_THRESHOLD=3.5
#   USE_KERNELS=auto|false|true  (Blackwell/B200: defaults to false via setup)
#   TMUX_SESSION=guided_gsk3b_guided   (remote session name)
#   ATTACH=1 | WAIT=1                  (attach to tmux after launch; blocks until detach)
#   FOREGROUND=1                       (legacy: run in SSH foreground, no tmux)
set -euo pipefail


STATE_FILE="${VAST_DIR}/.last_instance_id"
OUTPUT_STATE="${VAST_DIR}/.last_output_basename"

INSTANCE_ID="${1:-${INSTANCE_ID:-}}"
if [[ -z "${INSTANCE_ID}" && -f "${STATE_FILE}" ]]; then
  INSTANCE_ID="$(cat "${STATE_FILE}")"
fi
[[ -n "${INSTANCE_ID}" ]] || {
  echo "Usage: $0 [instance_id]  (or run launch.sh first)" >&2
  exit 1
}

REMOTE_CAMPAIGN="${REMOTE_CAMPAIGN:-/workspace/campaign}"
DESIGN_SPEC="${DESIGN_SPEC:-${REMOTE_CAMPAIGN}/gsk3b_peptide_design.yaml}"
GUIDANCE_FEATS_JSON="${GUIDANCE_FEATS_JSON:-${REMOTE_CAMPAIGN}/guidance_feats.json}"
BBB_CKPT="${BBB_CKPT:-${REMOTE_CAMPAIGN}/bbb_geo_best.ckpt}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  OUTPUT_BASENAME="${OUTPUT_BASENAME:-gsk3b_guided_smoke}"
  NUM_DESIGNS="${NUM_DESIGNS:-10}"
else
  OUTPUT_BASENAME="${OUTPUT_BASENAME:-gsk3b_guided}"
  NUM_DESIGNS="${NUM_DESIGNS:-300}"
fi
REMOTE_OUTPUT="/workspace/output/${OUTPUT_BASENAME}"
REMOTE_SCRIPT="${REMOTE_OUTPUT}/run_guided_campaign.sh"
REMOTE_LOG="${REMOTE_OUTPUT}/campaign.log"
TMUX_SESSION="${TMUX_SESSION:-guided_${OUTPUT_BASENAME}}"
echo "${OUTPUT_BASENAME}" > "${OUTPUT_STATE}"

REUSE_FLAG=""
[[ "${REUSE:-0}" == "1" ]] && REUSE_FLAG="--reuse"

USE_KERNELS="${USE_KERNELS:-auto}"

command -v vastai >/dev/null 2>&1 || { echo "pip install vastai" >&2; exit 1; }
ensure_vast_ssh_key

REMOTE_CMD="set -euo pipefail
export HF_HOME=/workspace/.cache/huggingface
USE_KERNELS=${USE_KERNELS:-auto}
if [[ -f /workspace/.guided_python ]]; then
  export PATH=\"\$(dirname \"\$(cat /workspace/.guided_python)\"):\${PATH}\"
fi
mkdir -p \${HF_HOME} ${REMOTE_OUTPUT}
test -f ${DESIGN_SPEC} || { echo 'Missing design spec: ${DESIGN_SPEC}' >&2; exit 1; }
if [[ -f ${GUIDANCE_FEATS_JSON} ]]; then
  echo 'Guidance feats json:' ${GUIDANCE_FEATS_JSON}
else
  echo 'Warning: guidance feats json not found; using inline-only config'
fi
if [[ ! -f ${BBB_CKPT} ]]; then
  echo 'Warning: BBB ckpt not found at ${BBB_CKPT}; BBB guidance may be skipped'
fi
if [[ -f /workspace/.guided_use_kernels ]]; then
  USE_KERNELS=\"\$(cat /workspace/.guided_use_kernels)\"
fi
if python3 -c \"import torch; raise SystemExit(0 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10 else 1)\" 2>/dev/null; then
  USE_KERNELS=false
  echo 'Blackwell GPU: forcing use_kernels=false'
fi
echo \"Using kernels: \${USE_KERNELS}\"
python3 -c \"import boltzgen, bbb_geo, torch; print('imports ok; cuda=', torch.cuda.is_available())\"
boltzgen --version"
bash boltzgen_design/scripts/vast/run_guided_campaign.sh

if [[ "${SKIP_CHECK:-0}" != "1" ]]; then
  REMOTE_CMD+="
boltzgen check ${DESIGN_SPEC} --output ${REMOTE_OUTPUT}/checked"
fi

REMOTE_CMD+="
boltzgen run ${DESIGN_SPEC} --output ${REMOTE_OUTPUT} \\
  --protocol peptide-anything \\
  --use_kernels \${USE_KERNELS} \\
  --num_designs ${NUM_DESIGNS} \\
  --design_checkpoints huggingface:boltzgen/boltzgen-1:boltzgen1_adherence.ckpt \\
  --inverse_fold_num_sequences 4 \\
  --refolding_rmsd_threshold ${REFOLDING_RMSD_THRESHOLD:-3.5} \\
  --config design guidance.feats_json=${GUIDANCE_FEATS_JSON} \\
  --config design guidance.bbb_ckpt=${BBB_CKPT} \\
  --config filtering filter_bindingsite=true peptide_type=cyclic filter_cysteine=true refolding_rmsd_threshold=${REFOLDING_RMSD_THRESHOLD:-3.5} \\
  --config analysis liability_modality=peptide liability_peptide_type=cyclic \\
  --steps design inverse_folding folding analysis filtering ${REUSE_FLAG}"

echo "=== Guided campaign on instance ${INSTANCE_ID} ==="
echo "Output: ${REMOTE_OUTPUT}"
echo "Designs: ${NUM_DESIGNS}"
echo "BBB ckpt: ${BBB_CKPT}"
echo "SSH: $(vast_ssh_url "${INSTANCE_ID}")"

if [[ "${FOREGROUND:-0}" == "1" ]]; then
  echo "Mode: foreground SSH (closing terminal may kill the job)"
  vast_ssh "${INSTANCE_ID}" bash -s <<EOF
${REMOTE_CMD}
EOF
  echo ""
  echo "Done."
else
  echo "Mode: detached tmux session '${TMUX_SESSION}'"
  vast_ssh "${INSTANCE_ID}" bash -s <<EOF
set -euo pipefail
command -v tmux >/dev/null 2>&1 || {
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux
}
TMUX_SESSION='${TMUX_SESSION}'
REMOTE_SCRIPT='${REMOTE_SCRIPT}'
REMOTE_LOG='${REMOTE_LOG}'
mkdir -p '${REMOTE_OUTPUT}'

if tmux has-session -t "\${TMUX_SESSION}" 2>/dev/null; then
  echo "tmux session '\${TMUX_SESSION}' already exists (not starting a duplicate)."
  echo "  attach: tmux attach -t \${TMUX_SESSION}"
  echo "  log:    tail -f \${REMOTE_LOG}"
  exit 0
fi

cat > "\${REMOTE_SCRIPT}" << 'CAMPAIGN_EOF'
${REMOTE_CMD}
CAMPAIGN_EOF
chmod +x "\${REMOTE_SCRIPT}"

tmux new-session -d -s "\${TMUX_SESSION}" "bash \${REMOTE_SCRIPT} 2>&1 | tee -a \${REMOTE_LOG}"
sleep 1
if tmux has-session -t "\${TMUX_SESSION}" 2>/dev/null; then
  echo "Started detached tmux session: \${TMUX_SESSION}"
  echo "Log: \${REMOTE_LOG}"
  echo "Attach: tmux attach -t \${TMUX_SESSION}"
  echo "Detach (keep running): Ctrl+B then D"
else
  echo "ERROR: tmux session failed to start. Check \${REMOTE_LOG}" >&2
  exit 1
fi
EOF

  if [[ "${ATTACH:-0}" == "1" || "${WAIT:-0}" == "1" ]]; then
    echo ""
    echo "Attaching to tmux (Ctrl+B, D to detach without stopping)..."
    vast_ssh "${INSTANCE_ID}" tmux attach -t "${TMUX_SESSION}"
    echo ""
    echo "Detached from tmux. Campaign may still be running on the instance."
  else
    echo ""
    echo "Campaign running in background on Vast."
    echo "  Monitor:  vast_ssh + tail -f ${REMOTE_LOG}"
    echo "  Attach:   vast_ssh + tmux attach -t ${TMUX_SESSION}"
    echo "  Detach:   Ctrl+B, D (job keeps running)"
  fi
fi

echo ""
echo "When finished, download:"
echo "  bash boltzgen_design/scripts/vast/sync_results.sh ${INSTANCE_ID} ${OUTPUT_BASENAME}"
echo "Local filtering-only rerun is still possible using --steps filtering."
