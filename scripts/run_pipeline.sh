#!/usr/bin/env bash
# Place beside data_tools/ and run/ in the repository's scripts/ directory.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/run/common.sh" "$@"

UPDATE_RATE="${UPDATE_RATE:-0.01}"
SEED="${SEED:-42}"
ALPHA_MIN="${ALPHA_MIN:-0}"
ALPHA_MAX="${ALPHA_MAX:-1}"
prepare_command=("$PYTHON" -B "$SCRIPT_DIR/data_tools/prepare.py" "$DATASET_DIR"
    --output "$PREPARED_DIR" --k "$K" --threads "$THREADS" --rounds "$ROUNDS"
    --update-rate "$UPDATE_RATE" --seed "$SEED"
    --alpha-min "$ALPHA_MIN" --alpha-max "$ALPHA_MAX")

require_files "$BASE_IMG" "$BASE_TEXT" "$QUERY_IMG" "$QUERY_TEXT"
if [[ "$DRY_RUN" == 1 ]]; then
    printf '\n=== Prepare data (or validate and reuse existing preparation) ===\n'
    run_command "${prepare_command[@]}"
    run_command bash "$RUN_SCRIPT_DIR/run_all.sh" "$DATASET_DIR" --dry-run
    exit 0
fi

if [[ -e "$PREPARED_DIR" ]]; then
    printf '\n=== Validate and reuse prepared data ===\n'
    # Match the inputs and requested settings rather than silently trusting a
    # directory from an older experiment. Verification never changes data.
    "$PYTHON" -B - "$DATASET_DIR" "$PREPARED_DIR" "$K" "$ROUNDS" \
        "$UPDATE_RATE" "$SEED" "$ALPHA_MIN" "$ALPHA_MAX" <<'PY'
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys

dataset, output = Path(sys.argv[1]), Path(sys.argv[2])
k, rounds, rate, seed, lo, hi = sys.argv[3:]

def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()

try:
    manifest = json.loads((output / "manifest.json").read_text())
    checks = (
        (manifest["schema_version"] == 1, "manifest schema"),
        (manifest["k"] >= int(k), "GT width K"),
        (manifest["rounds"] >= int(rounds), "number of update rounds"),
        (Decimal(manifest["update_rate"]) == Decimal(rate), "update rate"),
        (manifest["alpha"]["seed"] == int(seed), "alpha seed"),
        (Decimal(str(manifest["alpha"]["min"])) == Decimal(lo), "alpha minimum"),
        (Decimal(str(manifest["alpha"]["max"])) == Decimal(hi), "alpha maximum"),
        (manifest["distance"] == "alpha * L2(img) + (1-alpha) * L2(text)", "distance formula"),
    )
    for valid, label in checks:
        if not valid:
            raise ValueError(f"Prepared data does not match requested {label}")
    for name in ("base_img_emb.fvecs", "base_text_emb.fvecs",
                 "query_img_emb.fvecs", "query_text_emb.fvecs"):
        if digest(dataset / name) != manifest["inputs"][name]["sha256"]:
            raise ValueError(f"Source data has changed: {name}")
    if not manifest["outputs"]:
        raise ValueError("No generated outputs listed in manifest")
    for name, metadata in manifest["outputs"].items():
        path = output / name
        if path.stat().st_size != metadata["bytes"] or digest(path) != metadata["sha256"]:
            raise ValueError(f"Prepared output has changed: {name}")
except (OSError, KeyError, TypeError, ValueError, ArithmeticError) as exc:
    raise SystemExit(f"Error: {exc}. Choose a fresh PREPARED_DIR; existing data was not modified.")
print(f"Prepared data verified: {output}")
PY
else
    printf '\n=== Prepare alpha, static GT, insertion base, traces and dynamic GT ===\n'
    run_command "${prepare_command[@]}" 2>&1 | tee "$LOG_DIR/prepare.log"
fi

printf '\n=== Run build, static search, deletion and insertion ===\n'
run_command bash "$RUN_SCRIPT_DIR/run_all.sh" "$DATASET_DIR"
