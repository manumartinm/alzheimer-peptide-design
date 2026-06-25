#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=_vast_env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_vast_env.sh"

instance_id="$(resolve_instance_id "${1:-}")"
require_vast_cli
ensure_vast_ssh_key

REMOTE_ROOT="${REMOTE_ROOT}"
LOCAL_BBB="${BBB_MODELS}"
LOCAL_DATASET="${DATASET}/data/hf_release"

[[ -d "${LOCAL_BBB}" ]] || { echo "Missing local bbb_models directory: ${LOCAL_BBB}" >&2; exit 1; }
[[ -d "${LOCAL_DATASET}" ]] || { echo "Missing hf_release dataset: ${LOCAL_DATASET}" >&2; exit 1; }

echo "Preparing remote directories..."
vast_ssh "${instance_id}" "mkdir -p ${REMOTE_ROOT}/packages/bbb_models ${REMOTE_ROOT}/packages/dataset/data"

echo "Uploading bbb_models source..."
tar -C "${PACKAGES_ROOT}" \
  --exclude="bbb_models/artifacts" \
  --exclude="bbb_models/.venv" \
  --exclude="bbb_models/__pycache__" \
  --exclude="bbb_models/.mypy_cache" \
  -czf - bbb_models \
  | vast_ssh "${instance_id}" "tar -xzf - -C ${REMOTE_ROOT}/packages"

echo "Uploading hf_release dataset..."
tar -C "${DATASET}/data" -czf - hf_release \
  | vast_ssh "${instance_id}" "tar -xzf - -C ${REMOTE_ROOT}/packages/dataset/data"

echo "Upload complete."
echo "Next:"
echo "  bash ${BBB_MODELS}/scripts/vast/setup_instance.sh ${instance_id}"
