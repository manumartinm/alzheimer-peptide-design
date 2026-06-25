#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"

instance_id="$(resolve_instance_id "${1:-}")"
require_vast_session

REMOTE_ROOT="${REMOTE_ROOT}"
LOCAL_BBB="${BBB_MODELS}"
LOCAL_DATA="${BBB_MODELS}/data/bbb-peptides"

[[ -d "${LOCAL_BBB}" ]] || { echo "Missing local bbb_models directory: ${LOCAL_BBB}" >&2; exit 1; }
[[ -f "${LOCAL_DATA}/peptides.parquet" ]] || {
  echo "Missing HF dataset cache: ${LOCAL_DATA}/peptides.parquet" >&2
  echo "Download locally first:" >&2
  echo "  cd ${LOCAL_BBB} && uv run python scripts/data/download.py" >&2
  exit 1
}

echo "Preparing remote directories..."
vast_ssh "${instance_id}" "mkdir -p ${REMOTE_ROOT}/packages/bbb_models/data"

echo "Uploading bbb_models source..."
tar -C "${PACKAGES_ROOT}" \
  --exclude="bbb_models/artifacts" \
  --exclude="bbb_models/.venv" \
  --exclude="bbb_models/__pycache__" \
  --exclude="bbb_models/.mypy_cache" \
  --exclude="bbb_models/data/bbb-peptides" \
  -czf - bbb_models \
  | vast_ssh "${instance_id}" "tar -xzf - -C ${REMOTE_ROOT}/packages"

echo "Uploading bbb-peptides dataset..."
tar -C "${BBB_MODELS}/data" -czf - bbb-peptides \
  | vast_ssh "${instance_id}" "tar -xzf - -C ${REMOTE_ROOT}/packages/bbb_models/data"

echo "Upload complete."
echo "Next:"
echo "  bash ${INFRA_VAST}/bbb_models/setup_instance.sh ${instance_id}"
