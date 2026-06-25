#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"
# shellcheck source=_campaign.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_campaign.sh"

# Run BoltzGen campaign on a Vast instance (PyPI boltzgen + uploaded kinase).
#
# Usage:
#   bash infra/vast/boltzgen_design/run_campaign.sh [INSTANCE_ID]
#   SMOKE=1 bash infra/vast/boltzgen_design/run_campaign.sh
#   REUSE=1 bash infra/vast/boltzgen_design/run_campaign.sh

campaign_init "${1:-}"
campaign_set_output gsk3b_smoke gsk3b_baseline

REMOTE_CMD="set -euo pipefail
${CAMPAIGN_REMOTE_PYTHON_SETUP}
${CAMPAIGN_REMOTE_DISK_CHECK}
export HF_HOME=/workspace/.cache/huggingface
mkdir -p \${HF_HOME} ${REMOTE_OUTPUT}
ensure_disk_space 12000
test -f ${REMOTE_CAMPAIGN}/gsk3b.cif || { echo 'Missing gsk3b.cif — run launch.sh first' >&2; exit 1; }
\${BOLTZGEN_PY} -m pip install -q boltzgen
\${BOLTZGEN_PY} -c \"import torch; assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0))\"
boltzgen --version"

if [[ "${SKIP_CHECK:-0}" != "1" ]]; then
  REMOTE_CMD+="
boltzgen check ${DESIGN_SPEC} --output ${REMOTE_OUTPUT}/checked"
fi

REMOTE_CMD+="
boltzgen run ${DESIGN_SPEC} --output ${REMOTE_OUTPUT} \\
$(campaign_remote_run_core) \\
$(campaign_remote_run_finish)"

echo "=== Campaign on instance ${INSTANCE_ID} ==="
echo "Output: ${REMOTE_OUTPUT}"
echo "Designs: ${NUM_DESIGNS}"
echo "SSH: $(vast_ssh_url "${INSTANCE_ID}")"
vast_ssh "${INSTANCE_ID}" bash -s <<EOF
${REMOTE_CMD}
EOF

echo ""
echo "Done."
echo "Download: bash ${INFRA_VAST}/boltzgen_design/sync_results.sh ${INSTANCE_ID} ${OUTPUT_BASENAME}"
