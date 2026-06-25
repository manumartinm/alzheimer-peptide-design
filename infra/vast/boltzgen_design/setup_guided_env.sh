#!/usr/bin/env bash
# Upload local boltzgen + bbb_models code to a Vast instance and
# install both in editable mode for guidance runs.
#
# Usage:
#   bash infra/vast/boltzgen_design/setup_guided_env.sh [INSTANCE_ID]
#
# Optional env:
#   BBB_CKPT_LOCAL=/abs/path/best.ckpt
#   REMOTE_CAMPAIGN=/workspace/campaign
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_common.sh"

INSTANCE_ID="$(resolve_instance_id "${1:-}")"
REMOTE_CAMPAIGN="${REMOTE_CAMPAIGN:-/workspace/campaign}"
REMOTE_BBB_CKPT="${REMOTE_CAMPAIGN}/bbb_geo_best.ckpt"

require_vast_session

[[ -d "${BOLTZGEN}" ]] || { echo "Missing ${BOLTZGEN}" >&2; exit 1; }
[[ -d "${BBB_MODELS}" ]] || { echo "Missing ${BBB_MODELS}" >&2; exit 1; }

GUIDANCE_LOCAL="${BOLTZGEN_DESIGN}/targets/gsk3b/guidance_feats.json"
if [[ ! -f "${GUIDANCE_LOCAL}" ]]; then
  echo "Warning: guidance_feats.json not found at ${GUIDANCE_LOCAL}" >&2
fi

echo "=== Vast guided setup ==="
echo "Instance: ${INSTANCE_ID}"
echo "Remote root: ${REMOTE_ROOT}"
echo "Remote campaign: ${REMOTE_CAMPAIGN}"
echo "SSH: $(vast_ssh_url "${INSTANCE_ID}")"

vast_ensure_remote_dirs "${INSTANCE_ID}"
vast_ssh "${INSTANCE_ID}" "mkdir -p ${REMOTE_ROOT} ${REMOTE_CAMPAIGN} /workspace/output"

echo "=== Upload code (boltzgen + bbb_models) ==="
# COPYFILE_DISABLE: skip macOS AppleDouble (._*) files that break pip on Linux.
export COPYFILE_DISABLE=1
tar -C "${PACKAGES_ROOT}" \
  --exclude="boltzgen/.git" \
  --exclude="boltzgen/.venv" \
  --exclude="boltzgen/**/__pycache__" \
  --exclude="boltzgen/**/._*" \
  --exclude="boltzgen/**/*.egg-info" \
  --exclude="bbb_models/.venv" \
  --exclude="bbb_models/artifacts" \
  --exclude="bbb_models/**/__pycache__" \
  --exclude="bbb_models/**/._*" \
  --exclude="bbb_models/**/*.egg-info" \
  -czf - boltzgen bbb_models \
  | vast_ssh "${INSTANCE_ID}" "tar -xzf - -C ${REMOTE_ROOT}/packages"

vast_copy_to "${INSTANCE_ID}" "${BOLTZGEN_DESIGN}/targets/gsk3b/gsk3b_peptide_design.yaml" "/workspace/campaign/gsk3b_peptide_design.yaml"
vast_copy_to "${INSTANCE_ID}" "${BOLTZGEN_DESIGN}/targets/gsk3b/gsk3b.cif" "/workspace/campaign/gsk3b.cif"

if [[ -f "${GUIDANCE_LOCAL}" ]]; then
  echo "=== Upload guidance payload ==="
  vast_copy_to "${INSTANCE_ID}" "${GUIDANCE_LOCAL}" "${REMOTE_CAMPAIGN}/guidance_feats.json"
fi

if [[ -n "${BBB_CKPT_LOCAL:-}" ]]; then
  [[ -f "${BBB_CKPT_LOCAL}" ]] || {
    echo "BBB_CKPT_LOCAL does not exist: ${BBB_CKPT_LOCAL}" >&2
    exit 1
  }
  echo "=== Upload BBB geo checkpoint ==="
  vast_copy_to "${INSTANCE_ID}" "${BBB_CKPT_LOCAL}" "${REMOTE_BBB_CKPT}"
fi

echo "=== Install editable packages on remote ==="
vast_ssh "${INSTANCE_ID}" bash -s <<'EOF'
set -euo pipefail
REMOTE_ROOT="${REMOTE_ROOT}"
VENV="/workspace/.venv/guided"

pick_python() {
  for candidate in /opt/conda/bin/python python3.12 python3.11 python3; do
    if command -v "${candidate}" >/dev/null 2>&1 \
      && "${candidate}" -c "import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)"; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

clean_uploaded_tree() {
  # macOS ._*/egg-info artifacts confuse importlib.metadata during pip install.
  find "${REMOTE_ROOT}/packages/bbb_models" "${REMOTE_ROOT}/packages/boltzgen" \
    \( -name '._*' -o -name '*.egg-info' \) -print -delete 2>/dev/null || true
}

BASE_PY="$(pick_python || true)"
[[ -n "${BASE_PY}" ]] || { echo "Need Python >= 3.11 on remote" >&2; exit 1; }
echo "Base Python: $("${BASE_PY}" -V) ($("${BASE_PY}" -c 'import sys; print(sys.executable)'))"

clean_uploaded_tree

venv_args=()
if "${BASE_PY}" -c "import torch" 2>/dev/null; then
  # Keep image CUDA torch (conda) without reinstalling a multi-GB wheel.
  venv_args+=(--system-site-packages)
fi
if [[ -x "${VENV}/bin/python" ]] && ! "${VENV}/bin/python" -c "import torch" 2>/dev/null; then
  echo "Recreating guided venv (CUDA torch not visible in existing venv)"
  rm -rf "${VENV}"
fi
if [[ ! -x "${VENV}/bin/python" ]]; then
  "${BASE_PY}" -m venv "${venv_args[@]}" "${VENV}"
fi
PY="${VENV}/bin/python"
"${PY}" -m pip install -q -U pip setuptools wheel

export PATH="${VENV}/bin:${PATH}"
echo "${PY}" > /workspace/.guided_python

pip_install_editable() {
  "${PY}" -m pip install -q "$@"
}
pip_install_editable -e "${REMOTE_ROOT}/packages/bbb_models"
if "${PY}" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10 else 1)" 2>/dev/null; then
  pip_install_editable -e "${REMOTE_ROOT}/packages/boltzgen"
  echo "Blackwell GPU (B200/B100): --use_kernels false (cu12 cuequivariance + bundled triton are unreliable on sm_100)" >&2
  echo "false" > /workspace/.guided_use_kernels
elif pip_install_editable -e "${REMOTE_ROOT}/packages/boltzgen[cuequivariance]" \
  && "${PY}" -c "import cuequivariance_torch" 2>/dev/null; then
  echo "auto" > /workspace/.guided_use_kernels
else
  echo "WARNING: cuequivariance not available; campaign will use --use_kernels false" >&2
  echo "false" > /workspace/.guided_use_kernels
fi
"${PY}" - <<'PYCODE'
import torch
import boltzgen
import bbb_geo
print("boltzgen import ok")
print("bbb_geo import ok")
print("CUDA:", torch.cuda.is_available(), "count:", torch.cuda.device_count())
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    print("GPU:", torch.cuda.get_device_name(0), "capability:", cap)
PYCODE
EOF

echo ""
echo "Setup complete."
echo "If you uploaded a checkpoint, remote path is: ${REMOTE_BBB_CKPT}"
echo "Next: bash ${INFRA_VAST}/boltzgen_design/run_guided_campaign.sh ${INSTANCE_ID}"
