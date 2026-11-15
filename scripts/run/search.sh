#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" "$@"
validate_queries
require_files "$PREPARED_DIR/static_gt.ivecs"
validate_gt "$PREPARED_DIR/static_gt.ivecs"
require_binary search
require_index "$FULL_INDEX"
printf '\n=== Static search (full base) ===\n'
run_command "$BIN_DIR/search" "$BASE_IMG" "$BASE_TEXT" "$FULL_INDEX" \
    "$STATIC_L" "$K" "$ALPHA" "$QUERY_IMG" "$QUERY_TEXT" \
    "$PREPARED_DIR/static_gt.ivecs" "$THREADS" 1 1 "$SEARCH_TYPE"
