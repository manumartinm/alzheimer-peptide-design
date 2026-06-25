#!/usr/bin/env bash
# Download campaign results via scp.
#
# Usage:
#   bash infra/vast/boltzgen_design/sync_results.sh [INSTANCE_ID] [output_basename]
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"

INSTANCE_ID="$(resolve_instance_id "${1:-}")"
OUTPUT_BASENAME="$(resolve_output_basename gsk3b_baseline "${2:-}")"

REMOTE_DIR="/workspace/output/${OUTPUT_BASENAME}"
LOCAL_DIR="${BOLTZGEN}/workbench/${OUTPUT_BASENAME}"

require_vast_session
mkdir -p "${LOCAL_DIR}"
vast_copy_from "${INSTANCE_ID}" "${REMOTE_DIR}/" "${LOCAL_DIR}"
echo "Done: ${LOCAL_DIR}"
