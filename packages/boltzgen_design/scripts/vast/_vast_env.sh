#!/usr/bin/env bash
# Source canonical Vast helpers from a package scripts/vast/ directory.
VAST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../../../infra/vast/_common.sh
source "${VAST_DIR}/../../../../infra/vast/_common.sh"
