#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"

instance_id="$(resolve_instance_id "${1:-}")"
remote_dir="${2:-${REMOTE_ROOT}/packages/bbb_models/artifacts/}"
local_dir="${3:-${BBB_MODELS}/artifacts}"

require_vast_session
mkdir -p "${local_dir}"
vast_copy_from "${instance_id}" "${remote_dir}" "${local_dir}"
echo "Artifacts synced to ${local_dir}"
