#!/usr/bin/env bash
# Shared BoltzGen campaign helpers (source after _common.sh).

campaign_init() {
  INSTANCE_ID="$(resolve_instance_id "${1:-}")"
  REMOTE_CAMPAIGN="${REMOTE_CAMPAIGN:-/workspace/campaign}"
  DESIGN_SPEC="${DESIGN_SPEC:-${REMOTE_CAMPAIGN}/gsk3b_peptide_design.yaml}"
  REUSE_FLAG=""
  [[ "${REUSE:-0}" == "1" ]] && REUSE_FLAG="--reuse"
  require_vast_session
}

campaign_set_output() {
  local smoke_name="${1:?smoke basename required}"
  local full_name="${2:?full basename required}"
  if [[ "${SMOKE:-0}" == "1" ]]; then
    OUTPUT_BASENAME="${OUTPUT_BASENAME:-${smoke_name}}"
    NUM_DESIGNS="${NUM_DESIGNS:-10}"
  else
    OUTPUT_BASENAME="${OUTPUT_BASENAME:-${full_name}}"
    NUM_DESIGNS="${NUM_DESIGNS:-300}"
  fi
  REMOTE_OUTPUT="/workspace/output/${OUTPUT_BASENAME}"
  save_output_basename "${OUTPUT_BASENAME}"
}

read -r -d '' CAMPAIGN_REMOTE_PYTHON_SETUP <<'EOF' || true
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
EOF

read -r -d '' CAMPAIGN_REMOTE_DISK_CHECK <<'EOF' || true
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
    exit 1
  fi
}
EOF

campaign_remote_run_core() {
  cat <<EOF
  --protocol peptide-anything \\
  --num_designs ${NUM_DESIGNS} \\
  --design_checkpoints huggingface:boltzgen/boltzgen-1:boltzgen1_adherence.ckpt \\
  --inverse_fold_num_sequences 4 \\
  --refolding_rmsd_threshold ${REFOLDING_RMSD_THRESHOLD:-3.5}
EOF
}

campaign_remote_run_finish() {
  cat <<EOF
  --config filtering filter_bindingsite=true peptide_type=cyclic filter_cysteine=true refolding_rmsd_threshold=${REFOLDING_RMSD_THRESHOLD:-3.5} \\
  --config analysis liability_modality=peptide liability_peptide_type=cyclic \\
  --steps design inverse_folding folding analysis filtering ${REUSE_FLAG}
EOF
}

campaign_remote_guidance_flags() {
  cat <<EOF
  --config design guidance.feats_json=${GUIDANCE_FEATS_JSON} \\
  --config design guidance.bbb_ckpt=${BBB_CKPT}
EOF
}
