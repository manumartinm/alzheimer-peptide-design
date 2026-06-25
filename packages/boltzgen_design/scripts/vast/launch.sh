#!/usr/bin/env bash
# Rent an A100 on Vast.ai and upload the GSK3β target (2 files only).
#
# Usage:
#   bash packages/boltzgen_design/scripts/vast/launch.sh
set -euo pipefail
# shellcheck source=_vast_env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_vast_env.sh"

STATE_FILE="${VAST_DIR}/.last_instance_id"

REMOTE_CAMPAIGN="/workspace/campaign"
TARGET_DIR="${BOLTZGEN_DESIGN}/targets/gsk3b"

DISK_GB="${DISK_GB:-150}"
# BoltzGen HF weights (~2 GB/checkpoint × several) + outputs need headroom; 150 GB is safe.
# PyPI boltzgen requires Python >= 3.11 + CUDA torch. This image ships Python 3.11.
IMAGE="${IMAGE:-pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime}"
# Empty MAX_DPH = no price cap. Vast needs exact model names (A100_PCIE, not A100).
MAX_DPH="${MAX_DPH:-}"
RELIABILITY="${RELIABILITY:-}"
MIN_GPU_RAM="${MIN_GPU_RAM:-40}"
INSTANCE_ID="${INSTANCE_ID:-42432531}"

A100_VARIANTS=(A100_SXM4 A100_PCIE A100_SXM)

CIF="${TARGET_DIR}/gsk3b.cif"
YAML="${TARGET_DIR}/gsk3b_peptide_design.yaml"

vastai show instance "${INSTANCE_ID}"
echo "SSH: $(vastai ssh-url "${INSTANCE_ID}" 2>/dev/null || true)"

echo "=== Uploading kinase (2 files) ==="
vast_ensure_remote_dirs "${INSTANCE_ID}"
vast_copy_to "${INSTANCE_ID}" "${CIF}" "${REMOTE_CAMPAIGN}/gsk3b.cif"
vast_copy_to "${INSTANCE_ID}" "${YAML}" "${REMOTE_CAMPAIGN}/gsk3b_peptide_design.yaml"

echo ""
echo "Instance ready: ${INSTANCE_ID}"
echo "SSH:    $(vast_ssh_url "${INSTANCE_ID}")"
echo "Next:   bash boltzgen_design/scripts/vast/run_campaign.sh"
echo "        (or SMOKE=1 for a 10-design test run)"
echo "Destroy: vastai destroy instance ${INSTANCE_ID}"
