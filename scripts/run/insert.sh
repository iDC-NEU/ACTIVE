#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" "$@"
printf '\n=== Insert-only (%s rounds) ===\n' "$ROUNDS"
# Load the full base even though the saved graph initially contains only 50%.
run_updates insert insert "$INSERT_INDEX" 0
