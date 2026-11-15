#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" "$@"
printf '\n=== Delete-only (%s rounds) ===\n' "$ROUNDS"
run_updates delete update "$FULL_INDEX" 1
