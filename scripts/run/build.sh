#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" "$@"
require_files "$BASE_IMG" "$BASE_TEXT" \
    "$PREPARED_DIR/base_for_insert/base_img_emb.fvecs" \
    "$PREPARED_DIR/base_for_insert/base_text_emb.fvecs"
require_binary build_index
[[ "$FULL_INDEX" != "$INSERT_INDEX" ]] || fail 'Full and insertion indexes must use different paths'
build_or_reuse() {
    local label="$1" img="$2" text="$3" graph="$4"
    if [[ -e "$graph" || -L "$graph" ]]; then
        [[ -f "$graph" && -s "$graph" ]] || fail "Existing index is empty or not a regular file: $graph"
        printf '\n=== Reuse %s ===\nIndex: %s\n' "$label" "$graph"
        return
    fi
    printf '\n=== Build %s ===\n' "$label"
    if [[ "$DRY_RUN" == 0 ]]; then mkdir -p -- "$(dirname -- "$graph")"; fi
    run_command "$BIN_DIR/build_index" "$img" "$text" "$graph" \
        "$EF_BUILD" "$R" "$THREADS" 1 1
    require_index "$graph"
}
build_or_reuse 'full-base index' "$BASE_IMG" "$BASE_TEXT" "$FULL_INDEX"
build_or_reuse 'initial insertion index (first 50% of base)' \
    "$PREPARED_DIR/base_for_insert/base_img_emb.fvecs" \
    "$PREPARED_DIR/base_for_insert/base_text_emb.fvecs" "$INSERT_INDEX"
