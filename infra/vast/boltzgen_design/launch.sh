#!/usr/bin/env bash
# Upload the GSK3β target to an existing Vast instance.
#
# Usage:
#   bash infra/vast/boltzgen_design/launch.sh [INSTANCE_ID]
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"

INSTANCE_ID="$(resolve_instance_id "${1:-${INSTANCE_ID:-}}")"
REMOTE_CAMPAIGN="/workspace/campaign"
TARGET_DIR="${BOLTZGEN_DESIGN}/targets/gsk3b"
CIF="${TARGET_DIR}/gsk3b.cif"
YAML="${TARGET_DIR}/gsk3b_peptide_design.yaml"

require_vast_session
save_instance_id "${INSTANCE_ID}"

show_instance_summary "${INSTANCE_ID}"

echo "=== Uploading kinase (2 files) ==="
vast_ensure_remote_dirs "${INSTANCE_ID}"
vast_copy_to "${INSTANCE_ID}" "${CIF}" "${REMOTE_CAMPAIGN}/gsk3b.cif"
vast_copy_to "${INSTANCE_ID}" "${YAML}" "${REMOTE_CAMPAIGN}/gsk3b_peptide_design.yaml"

echo ""
echo "Instance ready: ${INSTANCE_ID}"
echo "SSH:    $(vast_ssh_url "${INSTANCE_ID}")"
echo "Next:   bash ${INFRA_VAST}/boltzgen_design/run_campaign.sh ${INSTANCE_ID}"
echo "        (or SMOKE=1 for a 10-design test run)"
