#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" "$@"
finish_run() {
    local status=$?
    trap - EXIT
    if [[ "$DRY_RUN" == 0 ]]; then
        if ! "$PYTHON" -B "$RUN_SCRIPT_DIR/summarize.py" "$LOG_DIR" \
            --rounds "$ROUNDS" --run-exit-code "$status"; then
            if [[ "$status" == 0 ]]; then status=1; fi
        fi
    fi
    exit "$status"
}
trap finish_run EXIT
# Check all inputs before starting the expensive graph construction.
validate_queries
require_files "$PREPARED_DIR/static_gt.ivecs" \
    "$PREPARED_DIR/base_for_insert/base_img_emb.fvecs" \
    "$PREPARED_DIR/base_for_insert/base_text_emb.fvecs"
validate_gt "$PREPARED_DIR/static_gt.ivecs"
collect_updates delete
collect_updates insert
for binary in build_index search update insert; do require_binary "$binary"; done
for step in build search delete insert; do
    # All four targets were compiled above; do not compile again in child steps.
    AUTO_BUILD=0 bash "$RUN_SCRIPT_DIR/$step.sh"
done
