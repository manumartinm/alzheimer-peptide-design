#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=_vast_env.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_vast_env.sh"

DISK_GB="${DISK_GB:-80}"
IMAGE="${IMAGE:-pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime}"
GPU_QUERY="${GPU_QUERY:-gpu_ram >= 16 rented=False}"
MAX_DPH="${MAX_DPH:-}"
RELIABILITY="${RELIABILITY:-}"

require_vast_cli
ensure_vast_ssh_key

query="${GPU_QUERY}"
[[ -n "${MAX_DPH}" ]] && query+=" dph < ${MAX_DPH}"
[[ -n "${RELIABILITY}" ]] && query+=" reliability > ${RELIABILITY}"

if [[ -z "${OFFER_ID:-}" ]]; then
  echo "Searching offers with query: ${query}"
  vastai search offers "${query}" | head -20
  read -r -p "Offer ID: " OFFER_ID
fi
[[ -n "${OFFER_ID:-}" ]] || { echo "Offer ID required" >&2; exit 1; }

echo "Creating instance from offer ${OFFER_ID}..."
create_raw="$(vastai create instance "${OFFER_ID}" \
  --image "${IMAGE}" \
  --disk "${DISK_GB}" \
  --ssh \
  --direct \
  --onstart-cmd "mkdir -p ${REMOTE_ROOT} /workspace/output /workspace/.cache/huggingface" \
  --raw 2>&1)" || true
echo "${create_raw}"

instance_id="$(python3 - <<'PY' "${create_raw}"
import json, re, sys
raw = sys.argv[1]
try:
    d = json.loads(raw)
    for k in ("new_contract", "instance_id", "id"):
        if d.get(k):
            print(d[k]); raise SystemExit
except json.JSONDecodeError:
    pass
m = re.search(r"(?:new_contract|instance)[\"'\s:]*(\d+)", raw)
print(m.group(1) if m else "")
PY
)"
[[ -n "${instance_id}" ]] || { echo "Could not parse instance id" >&2; exit 1; }

save_instance_id "${instance_id}"
echo "Waiting for instance ${instance_id} to become running..."
for _ in $(seq 1 60); do
  status="$(vastai show instance "${instance_id}" --raw 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin).get('actual_status',''))" 2>/dev/null || true)"
  [[ "${status}" == "running" ]] && break
  sleep 10
done

echo "Instance ready: ${instance_id}"
echo "SSH: $(vast_ssh_url "${instance_id}")"
echo "Next:"
echo "  bash ${BBB_MODELS}/scripts/vast/upload_workspace.sh ${instance_id}"
