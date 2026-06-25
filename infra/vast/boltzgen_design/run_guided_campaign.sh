#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
# shellcheck source=_campaign.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_campaign.sh"

# Run BoltzGen campaign on Vast using editable boltzgen + bbb_geo guidance.
#
# Usage:
#   bash infra/vast/boltzgen_design/run_guided_campaign.sh [INSTANCE_ID]
#   SMOKE=1 bash infra/vast/boltzgen_design/run_guided_campaign.sh
#
# Optional env: NUM_DESIGNS, OUTPUT_BASENAME, REUSE=1, SKIP_CHECK=1,
#   BBB_CKPT, GUIDANCE_FEATS_JSON, USE_KERNELS, TMUX_SESSION, ATTACH=1, FOREGROUND=1

campaign_init "${1:-}"
campaign_set_output gsk3b_guided_smoke gsk3b_guided

GUIDANCE_FEATS_JSON="${GUIDANCE_FEATS_JSON:-${REMOTE_CAMPAIGN}/guidance_feats.json}"
BBB_CKPT="${BBB_CKPT:-${REMOTE_CAMPAIGN}/bbb_geo_best.ckpt}"
USE_KERNELS="${USE_KERNELS:-auto}"
REMOTE_SCRIPT="${REMOTE_OUTPUT}/run_guided_campaign.sh"
REMOTE_LOG="${REMOTE_OUTPUT}/campaign.log"
TMUX_SESSION="${TMUX_SESSION:-guided_${OUTPUT_BASENAME}}"

REMOTE_CMD="set -euo pipefail
export HF_HOME=/workspace/.cache/huggingface
USE_KERNELS=${USE_KERNELS}
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

if [[ "${SKIP_CHECK:-0}" != "1" ]]; then
  REMOTE_CMD+="
boltzgen check ${DESIGN_SPEC} --output ${REMOTE_OUTPUT}/checked"
fi

REMOTE_CMD+="
boltzgen run ${DESIGN_SPEC} --output ${REMOTE_OUTPUT} \\
  --use_kernels \${USE_KERNELS} \\
$(campaign_remote_run_core) \\
$(campaign_remote_guidance_flags) \\
$(campaign_remote_run_finish)"

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
echo "  bash ${INFRA_VAST}/boltzgen_design/sync_results.sh ${INSTANCE_ID} ${OUTPUT_BASENAME}"
