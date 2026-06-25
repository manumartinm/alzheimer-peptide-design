#!/usr/bin/env bash
# Canonical Vast.ai helpers for alzheimer-peptide-design (sourced, not executed).

set -euo pipefail

VAST_SSH_IDENTITY="${VAST_SSH_IDENTITY:-${HOME}/.ssh/id_ed25519}"
VAST_SSH_USER="${VAST_SSH_USER:-root}"

_find_repo_root() {
  local dir="${1:?start directory required}"
  while [[ "${dir}" != "/" ]]; do
    if [[ -f "${dir}/pyproject.toml" ]] && grep -q 'alzheimer-peptide-design' "${dir}/pyproject.toml" 2>/dev/null; then
      printf '%s\n' "${dir}"
      return 0
    fi
    dir="$(dirname "${dir}")"
  done
  return 1
}

if [[ -z "${REPO_ROOT:-}" ]]; then
  _VAST_CALLER_DIR="$(cd "$(dirname "${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}")" && pwd)"
  REPO_ROOT="$(_find_repo_root "${_VAST_CALLER_DIR}")" || {
    echo "Could not locate repo root (expected pyproject.toml with alzheimer-peptide-design)." >&2
    exit 1
  }
fi

PACKAGES_ROOT="${REPO_ROOT}/packages"
BBB_MODELS="${PACKAGES_ROOT}/bbb_models"
DATASET="${PACKAGES_ROOT}/dataset"
BOLTZGEN="${PACKAGES_ROOT}/boltzgen"
BOLTZGEN_DESIGN="${PACKAGES_ROOT}/boltzgen_design"
REMOTE_ROOT="${REMOTE_ROOT:-/workspace/alzheimer-peptide-design}"

if [[ -n "${VAST_DIR:-}" ]]; then
  STATE_FILE="${VAST_DIR}/.last_instance_id"
fi

require_vast_cli() {
  command -v vastai >/dev/null 2>&1 || { echo "Install vastai: pip install vastai" >&2; exit 1; }
  vastai show user >/dev/null 2>&1 || { echo "Configure API key: vastai set api-key YOUR_KEY" >&2; exit 1; }
}

require_ssh_identity() {
  [[ -f "${VAST_SSH_IDENTITY}" ]] || {
    echo "SSH private key not found: ${VAST_SSH_IDENTITY}" >&2
    echo "Set VAST_SSH_IDENTITY or create: ssh-keygen -t ed25519" >&2
    exit 1
  }
}

ensure_vast_ssh_key() {
  require_ssh_identity
  local pub="${VAST_SSH_IDENTITY}.pub"
  [[ -f "${pub}" ]] || { echo "Missing public key: ${pub}" >&2; return 1; }

  local keys broken_id key_line
  keys="$(vastai show ssh-keys --raw 2>/dev/null || echo "[]")"
  key_line="$(python3 - <<'PY' "${keys}"
import json, sys
keys = json.loads(sys.argv[1] or "[]")
for k in keys:
    pk = str(k.get("public_key", ""))
    if pk.startswith("/") or pk.endswith(".pub"):
        print(f"broken {k.get('id')}")
        break
PY
)" || true

  if [[ -n "${key_line}" ]]; then
    broken_id="${key_line#broken }"
    echo "Fixing broken Vast SSH key registration (id=${broken_id})..."
    vastai delete ssh-key "${broken_id}" >/dev/null 2>&1 || true
    keys="[]"
  fi

  if [[ "${keys}" == "[]" || -z "${keys}" ]]; then
    echo "Registering SSH key with Vast account..."
    vastai create ssh-key "$(cat "${pub}")"
  fi
}

resolve_instance_id() {
  local id="${1:-}"
  if [[ -z "${id}" && -n "${STATE_FILE:-}" && -f "${STATE_FILE}" ]]; then
    id="$(tr -d '[:space:]' < "${STATE_FILE}")"
  fi
  [[ -n "${id}" ]] || { echo "Missing instance id (pass as first argument)." >&2; return 1; }
  printf "%s\n" "${id}"
}

vast_ssh_endpoint() {
  local instance_id="${1:?instance_id required}"
  local endpoint host port ssh_url raw
  raw="$(vastai show instance "${instance_id}" --raw 2>/dev/null || true)"
  endpoint="$(python3 - <<'PY' "${raw}"
import json, sys
raw = sys.argv[1].strip()
if not raw:
    raise SystemExit(1)
d = json.loads(raw)
host = d.get("ssh_host")
port = d.get("ssh_port")
if host and port:
    print(f"{host} {port}")
PY
)" || true
  if [[ -n "${endpoint}" ]]; then
    printf "%s\n" "${endpoint}"
    return 0
  fi
  ssh_url="$(vastai ssh-url "${instance_id}" 2>/dev/null || true)"
  [[ -n "${ssh_url}" ]] || { echo "Could not resolve ssh endpoint for ${instance_id}" >&2; return 1; }
  host="$(python3 - <<'PY' "${ssh_url}"
import re, sys
u = sys.argv[1].strip()
m = re.match(r"ssh://[^@]+@([^:]+):(\d+)$", u)
if not m:
    raise SystemExit(1)
print(m.group(1))
PY
)"
  port="$(python3 - <<'PY' "${ssh_url}"
import re, sys
u = sys.argv[1].strip()
m = re.match(r"ssh://[^@]+@([^:]+):(\d+)$", u)
if not m:
    raise SystemExit(1)
print(m.group(2))
PY
)"
  printf "%s\n" "${host}" "${port}"
}

vast_ssh_url() {
  local instance_id="${1:?instance_id required}"
  local host port
  read -r host port < <(vast_ssh_endpoint "${instance_id}")
  printf 'ssh://%s@%s:%s\n' "${VAST_SSH_USER}" "${host}" "${port}"
}

vast_ssh_cmd() {
  local instance_id="${1:?instance_id required}"
  local host port
  read -r host port < <(vast_ssh_endpoint "${instance_id}")
  printf 'ssh -i %q -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -p %s %s@%s' \
    "${VAST_SSH_IDENTITY}" "${port}" "${VAST_SSH_USER}" "${host}"
}

vast_ssh() {
  local instance_id="${1:?instance_id required}"
  shift
  local host port
  read -r host port < <(vast_ssh_endpoint "${instance_id}")
  require_ssh_identity
  ssh \
    -i "${VAST_SSH_IDENTITY}" \
    -o IdentitiesOnly=yes \
    -o StrictHostKeyChecking=accept-new \
    -o ConnectTimeout=60 \
    -p "${port}" \
    "${VAST_SSH_USER}@${host}" \
    "$@"
}

vast_copy_to() {
  local instance_id="${1:?instance_id required}"
  local local_path="${2:?local path required}"
  local remote_path="${3:?remote path required}"
  local host port
  read -r host port < <(vast_ssh_endpoint "${instance_id}")
  require_ssh_identity
  scp \
    -i "${VAST_SSH_IDENTITY}" \
    -o IdentitiesOnly=yes \
    -o StrictHostKeyChecking=accept-new \
    -P "${port}" \
    -r "${local_path}" \
    "${VAST_SSH_USER}@${host}:${remote_path}"
}

vast_copy_from() {
  local instance_id="${1:?instance_id required}"
  local remote_path="${2:?remote path required}"
  local local_path="${3:?local path required}"
  local host port
  read -r host port < <(vast_ssh_endpoint "${instance_id}")
  require_ssh_identity
  mkdir -p "${local_path}"
  scp \
    -i "${VAST_SSH_IDENTITY}" \
    -o IdentitiesOnly=yes \
    -o StrictHostKeyChecking=accept-new \
    -P "${port}" \
    -r \
    "${VAST_SSH_USER}@${host}:${remote_path}" \
    "${local_path}/"
}

save_instance_id() {
  local instance_id="${1:?instance_id required}"
  [[ -n "${STATE_FILE:-}" ]] || { echo "STATE_FILE not set" >&2; return 1; }
  printf "%s\n" "${instance_id}" > "${STATE_FILE}"
}

attach_instance_ssh_key() {
  local instance_id="${1:?instance_id required}"
  local pub="${VAST_SSH_IDENTITY}.pub"
  require_ssh_identity
  [[ -f "${pub}" ]] || { echo "Missing SSH public key: ${pub}" >&2; return 1; }
  vastai attach ssh "${instance_id}" "$(cat "${pub}")" || true
}

vast_ensure_remote_dirs() {
  local instance_id="${1:?instance_id required}"
  vast_ssh "${instance_id}" "mkdir -p /workspace/campaign /workspace/.cache/huggingface /workspace/output ${REMOTE_ROOT}"
}
