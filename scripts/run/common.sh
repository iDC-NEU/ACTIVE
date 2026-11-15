#!/usr/bin/env bash
# Shared paths and parameters; override defaults with environment variables.
set -euo pipefail

RUN_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${ACTIVE_ROOT:-$(cd -- "$RUN_SCRIPT_DIR/../.." && pwd)}"
DATASET_DIR="${DATASET_DIR:-$PROJECT_ROOT/datasets/oi_mini}"
DRY_RUN="${DRY_RUN:-0}"
dataset_arg_seen=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --help|-h)
            printf 'Usage: bash %s [DATASET_DIR] [--dry-run]\n' "$(basename -- "$0")"
            exit 0 ;;
        -*) printf 'Unknown option: %s\n' "$arg" >&2; exit 1 ;;
        *)
            if [[ "$dataset_arg_seen" == 1 ]]; then
                printf 'Only one dataset directory is accepted.\n' >&2; exit 1
            fi
            DATASET_DIR="$arg"; dataset_arg_seen=1 ;;
    esac
done
DATASET_DIR="$(cd -- "$DATASET_DIR" && pwd)"
PREPARED_DIR="${PREPARED_DIR:-$DATASET_DIR/prepared}"
# Keep generated indexes and build outputs inside the project.
RUN_DIR="${RUN_DIR:-$RUN_SCRIPT_DIR/artifacts/$(basename -- "$DATASET_DIR")}"
# Save logs and summaries together under this runner directory.
LOG_DIR="${LOG_DIR:-$RUN_SCRIPT_DIR/log/$(basename -- "$DATASET_DIR")/$(date +%Y%m%d-%H%M%S)-$$}"
if [[ -z "${BUILD_DIR:-}" && -z "${BIN_DIR:-}" && -x "$PROJECT_ROOT/build/test/build_index" ]]; then
    BUILD_DIR="$PROJECT_ROOT/build"
fi
if [[ -z "${BUILD_DIR:-}" && "${BIN_DIR:-}" == */test ]]; then
    BUILD_DIR="${BIN_DIR%/test}"
fi
BUILD_DIR="${BUILD_DIR:-$RUN_DIR/build}"
BIN_DIR="${BIN_DIR:-$BUILD_DIR/test}"
FULL_INDEX="${FULL_INDEX:-$RUN_DIR/full.index}"
INSERT_INDEX="${INSERT_INDEX:-$RUN_DIR/insert_initial.index}"
# Paper Section 6.1: construction and update use different candidate pools.
EF_BUILD="${EF_BUILD:-200}"
EF_UPDATE="${EF_UPDATE:-100}"
R="${R:-40}"
THREADS="${THREADS:-32}"
AUTO_BUILD="${AUTO_BUILD:-1}"
BUILD_JOBS="${BUILD_JOBS:-$THREADS}"
CMAKE="${CMAKE:-cmake}"
L="${L:-100}"
STATIC_L="${STATIC_L:-200}"
K="${K:-10}"
ROUNDS="${ROUNDS:-10}"
ANGLE="${ANGLE:-10}"
SEARCH_TYPE="${SEARCH_TYPE:-0}"
PYTHON="${PYTHON:-python3}"
BASE_IMG="$DATASET_DIR/base_img_emb.fvecs"
BASE_TEXT="$DATASET_DIR/base_text_emb.fvecs"
QUERY_IMG="$DATASET_DIR/query_img_emb.fvecs"
QUERY_TEXT="$DATASET_DIR/query_text_emb.fvecs"
ALPHA="$PREPARED_DIR/query_alpha.fvecs"

fail() { printf 'Error: %s\n' "$*" >&2; exit 1; }
for name in EF_BUILD EF_UPDATE R THREADS BUILD_JOBS L STATIC_L K ROUNDS; do
    value="${!name}"
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail "$name must be a positive integer"
done
[[ "$ANGLE" =~ ^[0-9]+$ ]] || fail 'ANGLE must be a nonnegative integer'
[[ "$SEARCH_TYPE" == 0 || "$SEARCH_TYPE" == 1 ]] || fail 'SEARCH_TYPE must be 0 or 1'
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || fail 'DRY_RUN must be 0 or 1'
[[ "$AUTO_BUILD" == 0 || "$AUTO_BUILD" == 1 ]] || fail 'AUTO_BUILD must be 0 or 1'
(( L >= K )) || fail 'L must be at least K'
(( STATIC_L >= K )) || fail 'STATIC_L must be at least K'

require_files() {
    local file
    for file in "$@"; do [[ -s "$file" ]] || fail "Missing or empty file: $file"; done
}
require_binary() {
    if [[ "$AUTO_BUILD" == 1 && "${compiled_this_invocation:-0}" == 0 ]]; then
        [[ "$BIN_DIR" == "$BUILD_DIR/test" ]] || fail 'Automatic compilation requires BIN_DIR=BUILD_DIR/test; use BUILD_DIR to select the build directory, or AUTO_BUILD=0 for external binaries'
        if [[ "$DRY_RUN" == 0 ]]; then
            command -v "$CMAKE" >/dev/null 2>&1 || fail 'cmake is required for automatic compilation'
        fi
        printf '\n=== Compile ACTIVE executables (incremental build) ===\n'
        run_command "$CMAKE" -S "$PROJECT_ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
        run_command "$CMAKE" --build "$BUILD_DIR" --parallel "$BUILD_JOBS" \
            --target build_index search update insert
        compiled_this_invocation=1
    fi
    if [[ "$DRY_RUN" == 0 && ! -x "$BIN_DIR/$1" ]]; then
        fail "Missing executable: $BIN_DIR/$1. Check the build output or set BUILD_DIR to the correct build directory."
    fi
}
require_index() {
    if [[ "$DRY_RUN" == 0 ]]; then require_files "$1"; fi
}
run_command() {
    printf '+'; printf ' %q' "$@"; printf '\n'
    if [[ "$DRY_RUN" == 0 ]]; then "$@"; fi
}
validate_queries() {
    require_files "$BASE_IMG" "$BASE_TEXT" "$QUERY_IMG" "$QUERY_TEXT" "$ALPHA"
}
validate_gt() {
    "$PYTHON" -B -c '
import struct, sys
from pathlib import Path
k = int(sys.argv[1])
for name in sys.argv[2:]:
    p = Path(name)
    with p.open("rb") as f:
        dim, = struct.unpack("<i", f.read(4))
    if not 0 < k <= dim:
        raise SystemExit(f"K={k} exceeds GT width {dim}: {p}")
' "$K" "$@"
}
collect_updates() {
    local mode="$1" r tag
    trace_files=(); gt_files=()
    for ((r=0; r<ROUNDS; r++)); do
        printf -v tag '%02d' "$r"
        trace_files+=("$PREPARED_DIR/$mode/traces/round_${tag}_trace.bin")
    done
    for ((r=0; r<=ROUNDS; r++)); do
        printf -v tag '%02d' "$r"
        gt_files+=("$PREPARED_DIR/$mode/gt/round_${tag}.ivecs")
    done
    require_files "${trace_files[@]}" "${gt_files[@]}"
    validate_gt "${gt_files[@]}"
    # The C++ entry points serialize these paths internally as comma-separated lists.
    local file
    for file in "${trace_files[@]}" "${gt_files[@]}"; do
        [[ "$file" != *,* ]] || fail 'Trace/GT paths must not contain commas'
    done
}
run_updates() {
    local mode="$1" binary="$2" graph="$3" id_flag="$4"
    validate_queries
    collect_updates "$mode"
    require_binary "$binary"
    require_index "$graph"
    run_command "$BIN_DIR/$binary" "$graph" "$BASE_IMG" "$BASE_TEXT" \
        "${trace_files[@]}" "${gt_files[@]}" \
        "$EF_UPDATE" "$R" "$THREADS" "$id_flag" "$L" "$K" "$ANGLE" \
        "$ALPHA" "$QUERY_IMG" "$QUERY_TEXT"
}

export ACTIVE_ROOT="$PROJECT_ROOT" DATASET_DIR PREPARED_DIR RUN_DIR BUILD_DIR BIN_DIR
export AUTO_BUILD BUILD_JOBS CMAKE
export FULL_INDEX INSERT_INDEX EF_BUILD EF_UPDATE R THREADS L STATIC_L K ROUNDS ANGLE SEARCH_TYPE PYTHON DRY_RUN
export LOG_DIR

# Re-enter each entry script through a synchronous tee pipeline. The parent
# run_all log captures everything, while child stages also get their own logs.
# Waiting for tee ensures a stage log is complete before summarization starts.
entry_name="$(basename -- "$0" .sh)"
if [[ "$DRY_RUN" == 0 && "${ACTIVE_LOGGED_ENTRY:-}" != "$entry_name" ]]; then
    mkdir -p -- "$LOG_DIR"
    log_file="$LOG_DIR/$entry_name.log"
    [[ ! -e "$log_file" ]] || fail "Log already exists: $log_file. Choose a fresh LOG_DIR."
    printf 'Log directory: %s\n' "$LOG_DIR"
    set +e
    ACTIVE_LOGGED_ENTRY="$entry_name" bash "$0" "$@" 2>&1 | tee "$log_file"
    log_status=("${PIPESTATUS[@]}")
    set -e
    if [[ "${log_status[0]}" != 0 ]]; then exit "${log_status[0]}"; fi
    exit "${log_status[1]}"
fi
