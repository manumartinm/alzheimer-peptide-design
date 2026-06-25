#!/usr/bin/env bash
# shellcheck source=_vast_env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_vast_env.sh"


instance_id="$(resolve_instance_id "${1:-}")"
require_vast_cli

echo "Destroying instance ${instance_id}..."
vastai destroy instance "${instance_id}"
if [[ -f "${STATE_FILE}" ]]; then
  rm -f "${STATE_FILE}"
fi
echo "Destroyed."
