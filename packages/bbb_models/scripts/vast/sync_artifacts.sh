#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=_vast_env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_vast_env.sh"

instance_id="$(resolve_instance_id "${1:-}")"
remote_dir="${2:-${REMOTE_ROOT}/packages/bbb_models/artifacts/}"
local_dir="${3:-${BBB_MODELS}/artifacts}"

require_vast_cli
ensure_vast_ssh_key
mkdir -p "${local_dir}"
vast_copy_from "${instance_id}" "${remote_dir}" "${local_dir}"
echo "Artifacts synced to ${local_dir}"
