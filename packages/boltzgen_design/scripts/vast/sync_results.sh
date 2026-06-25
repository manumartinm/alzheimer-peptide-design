#!/usr/bin/env bash
# Download campaign results via scp.
#
# Usage:
#   bash packages/boltzgen_design/scripts/vast/sync_results.sh [INSTANCE_ID] [output_basename]
set -euo pipefail
# shellcheck source=_vast_env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_vast_env.sh"

STATE_FILE="${VAST_DIR}/.last_instance_id"
OUTPUT_STATE="${VAST_DIR}/.last_output_basename"

INSTANCE_ID="${1:-}"
if [[ -z "${INSTANCE_ID}" && -f "${STATE_FILE}" ]]; then
  INSTANCE_ID="$(cat "${STATE_FILE}")"
  echo "Using saved instance: ${INSTANCE_ID}"
fi

if [[ -z "${INSTANCE_ID}" ]]; then
  echo "Usage: $0 [instance_id] [gsk3b_baseline|gsk3b_smoke]" >&2
  exit 1
fi

OUTPUT_BASENAME="${2:-}"
if [[ -z "${OUTPUT_BASENAME}" && -f "${OUTPUT_STATE}" ]]; then
  OUTPUT_BASENAME="$(cat "${OUTPUT_STATE}")"
fi
OUTPUT_BASENAME="${OUTPUT_BASENAME:-gsk3b_baseline}"

REMOTE_DIR="/workspace/output/${OUTPUT_BASENAME}"
LOCAL_DIR="${BOLTZGEN}/workbench/${OUTPUT_BASENAME}"

if ! command -v vastai >/dev/null 2>&1; then
  echo "Install: pip install vastai" >&2
  exit 1
fi
ensure_vast_ssh_key

mkdir -p "${LOCAL_DIR}"
vast_copy_from "${INSTANCE_ID}" "${REMOTE_DIR}/" "${LOCAL_DIR}"
echo "Done: ${LOCAL_DIR}"
