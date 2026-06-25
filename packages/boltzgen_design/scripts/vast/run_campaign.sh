#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=_vast_env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_vast_env.sh"
# Run BoltzGen campaign on a Vast instance (PyPI boltzgen + uploaded kinase).
#
# Prereq: launch.sh already uploaded gsk3b.cif + gsk3b_peptide_design.yaml
#
# Usage:
#   bash boltzgen_design/scripts/vast/run_campaign.sh
#   bash boltzgen_design/scripts/vast/run_campaign.sh <INSTANCE_ID>
#   SMOKE=1 bash boltzgen_design/scripts/vast/run_campaign.sh
#   REUSE=1 bash boltzgen_design/scripts/vast/run_campaign.sh   # resume
#
# Env: SMOKE=1 NUM_DESIGNS=200 REUSE=1 SKIP_CHECK=1
set -euo pipefail

STATE_FILE="${VAST_DIR}/.last_instance_id"
OUTPUT_STATE="${VAST_DIR}/.last_output_basename"

REMOTE_CAMPAIGN="/workspace/campaign"
DESIGN_SPEC="${REMOTE_CAMPAIGN}/gsk3b_peptide_design.yaml"

INSTANCE_ID="${1:-${INSTANCE_ID:-}}"
if [[ -z "${INSTANCE_ID}" && -f "${STATE_FILE}" ]]; then
  INSTANCE_ID="$(cat "${STATE_FILE}")"
fi
[[ -n "${INSTANCE_ID}" ]] || {
  echo "Usage: $0 [instance_id]  (or run launch.sh first)" >&2
  exit 1
}

command -v vastai >/dev/null 2>&1 || { echo "pip install vastai" >&2; exit 1; }
ensure_vast_ssh_key

if [[ "${SMOKE:-0}" == "1" ]]; then
  OUTPUT_BASENAME="gsk3b_smoke"
  NUM_DESIGNS="${NUM_DESIGNS:-10}"
else
  OUTPUT_BASENAME="gsk3b_baseline"
  NUM_DESIGNS="${NUM_DESIGNS:-300}"
fi
REMOTE_OUTPUT="/workspace/output/${OUTPUT_BASENAME}"
echo "${OUTPUT_BASENAME}" > "${OUTPUT_STATE}"

REUSE_FLAG=""
[[ "${REUSE:-0}" == "1" ]] && REUSE_FLAG="--reuse"

# PyPI boltzgen requires Python >= 3.11 (not 3.8). Bootstrap conda if the image is too old.
REMOTE_SETUP='
ensure_boltzgen_python() {
  local py=""
  for candidate in /opt/conda/bin/python python3.12 python3.11 python3; do
    if command -v "${candidate}" >/dev/null 2>&1 \
      && "${candidate}" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"; then
      py="${candidate}"
      break
    fi
  done
  if [[ -z "${py}" ]]; then
    echo "No Python >= 3.11 found — installing Miniconda env bg (python 3.12)..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /workspace/miniconda
    /workspace/miniconda/bin/conda create -n bg python=3.12 -y
    py=/workspace/miniconda/envs/bg/bin/python
  fi
  echo "boltzgen Python: $(${py} -V)"
  export BOLTZGEN_PY="${py}"
  export PATH="$(dirname "${py}"):${PATH}"
}
ensure_boltzgen_python
'

REMOTE_DISK='
ensure_disk_space() {
  local min_mb="${1:-12000}"
  rm -rf /root/.cache/pip /tmp/pip-* 2>/dev/null || true
  python3 -m pip cache purge 2>/dev/null || true
  rm -rf "${HF_HOME}/hub/models--boltzgen--boltzgen-1/blobs/"*.incomplete 2>/dev/null || true
  local free_mb
  free_mb=$(df -BM /workspace | awk "NR==2 {gsub(/M/,\"\",\$4); print \$4}")
  echo "Free on /workspace: ${free_mb} MB (need >= ${min_mb} MB for BoltzGen weights)"
  if [[ "${free_mb}" -lt "${min_mb}" ]]; then
    echo "ERROR: Not enough disk. BoltzGen needs ~12 GB for model weights + outputs." >&2
    echo "  df -h /workspace" >&2
    echo "  Free space: apt-get clean; pip cache purge; rm -rf /workspace/miniconda /workspace/output/*" >&2
    echo "  Or recreate instance: DISK_GB=150 bash boltzgen_design/scripts/vast/launch.sh" >&2
    exit 1
  fi
}
'

REMOTE_SETUP="${REMOTE_SETUP}${REMOTE_DISK}"

REMOTE_CMD="set -euo pipefail
${REMOTE_SETUP}
export HF_HOME=/workspace/.cache/huggingface
mkdir -p \${HF_HOME} ${REMOTE_OUTPUT}
ensure_disk_space 12000
test -f ${REMOTE_CAMPAIGN}/gsk3b.cif || { echo 'Missing gsk3b.cif — run launch.sh first' >&2; exit 1; }
\${BOLTZGEN_PY} -m pip install -q boltzgen
\${BOLTZGEN_PY} -c \"import torch; assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0))\"
boltzgen --version"

if [[ "${SKIP_CHECK:-0}" != "1" ]]; then
  REMOTE_CMD+="
boltzgen check ${DESIGN_SPEC} --output ${REMOTE_OUTPUT}/checked"
fi

REMOTE_CMD+="
boltzgen run ${DESIGN_SPEC} --output ${REMOTE_OUTPUT} \\
  --protocol peptide-anything \\
  --num_designs ${NUM_DESIGNS} \\
  --design_checkpoints huggingface:boltzgen/boltzgen-1:boltzgen1_adherence.ckpt \\
  --inverse_fold_num_sequences 4 \\
  --refolding_rmsd_threshold ${REFOLDING_RMSD_THRESHOLD:-3.5} \\
  --config filtering filter_bindingsite=true peptide_type=cyclic filter_cysteine=true refolding_rmsd_threshold=${REFOLDING_RMSD_THRESHOLD:-3.5} \\
  --config analysis liability_modality=peptide liability_peptide_type=cyclic \\
  --steps design inverse_folding folding analysis filtering ${REUSE_FLAG}"

echo "=== Campaign on instance ${INSTANCE_ID} ==="
echo "Output: ${REMOTE_OUTPUT}"
echo "Designs: ${NUM_DESIGNS}"
echo "SSH: $(vast_ssh_url "${INSTANCE_ID}")"
vast_ssh "${INSTANCE_ID}" bash -s <<EOF
${REMOTE_CMD}
EOF

echo ""
echo "Done."
echo "Download: bash boltzgen_design/scripts/vast/sync_results.sh ${INSTANCE_ID} ${OUTPUT_BASENAME}"
