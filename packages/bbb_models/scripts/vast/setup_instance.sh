#!/usr/bin/env bash
# shellcheck source=_vast_env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_vast_env.sh"


instance_id="$(resolve_instance_id "${1:-}")"
require_vast_cli
ensure_vast_ssh_key

vast_ssh "${instance_id}" bash -s <<'EOF'
set -euo pipefail
cd ${REMOTE_ROOT}/packages/bbb_models
python3 -m pip install -U pip
python3 -m pip install -e .
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), 'GPUs:', torch.cuda.device_count()); assert torch.cuda.is_available()"
EOF

echo "Instance setup complete on ${instance_id}."
