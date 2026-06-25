#!/usr/bin/env bash
# shellcheck source=_vast_env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_vast_env.sh"


instance_id="$(resolve_instance_id "${1:-}")"
log_file="${2:-}"

require_vast_cli
echo "Instance:"
vastai show instance "${instance_id}" || true
echo
if [[ -z "${log_file}" ]]; then
  log_file="$(vast_ssh "${instance_id}" "ls -1t /workspace/output/*_train.log /workspace/output/*_cv.log 2>/dev/null | head -n 1" || true)"
fi
[[ -n "${log_file}" ]] || { echo "No remote train/cv log found under /workspace/output" >&2; exit 1; }
echo "Tailing log: ${log_file}"
vast_ssh "${instance_id}" "tail -n 120 -f ${log_file}"
