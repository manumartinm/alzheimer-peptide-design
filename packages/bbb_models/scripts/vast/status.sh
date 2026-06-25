#!/usr/bin/env bash
# shellcheck source=_vast_env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_vast_env.sh"


instance_id="$(resolve_instance_id "${1:-}")"
require_vast_cli

echo "Instance:"
vastai show instance "${instance_id}" || true
echo
echo "Remote process status:"
vast_ssh "${instance_id}" "set -euo pipefail; ls -l /workspace/output || true; for p in /workspace/output/last_train.pid /workspace/output/last_cv.pid; do if [[ -f \$p ]]; then pid=\$(cat \$p); if ps -p \$pid >/dev/null 2>&1; then echo \"RUNNING pid=\$pid (\$p)\"; else echo \"STOPPED pid=\$pid (\$p)\"; fi; fi; done"
