# Run ACTIVE

Run from the repository root after data preparation:

```bash
bash scripts/run/run_all.sh [DATASET_DIR] [--dry-run]
```

Or run stages separately:

```bash
bash scripts/run/build.sh    # Build or reuse full-base and insertion indexes
bash scripts/run/search.sh   # Static search
bash scripts/run/delete.sh   # Deletion and evaluation
bash scripts/run/insert.sh   # Insertion and evaluation
```

Parameters (environment variables):

- `DATASET_DIR`: default `datasets/oi_mini`.
- `PREPARED_DIR`: default `DATASET_DIR/prepared`.
- `EF_BUILD=200`, `EF_UPDATE=100`: construction and update candidate pool sizes.
- `R=40`: maximum neighbors per node.
- `THREADS=32`: worker threads.
- `ANGLE=10`: update angle threshold in degrees.
- `K=10`: nearest neighbors returned.
- `L=100`: search candidate pool during updates.
- `ROUNDS=10`: update rounds.
- `STATIC_L=200`: static search L limit.
- `SEARCH_TYPE=0`: sweep from K to the static search L limit. Set to 1 for fixed L.
- `RUN_DIR`: default `scripts/run/artifacts/oi_mini` for indexes and new build outputs.
- `LOG_DIR`: default `scripts/run/log/oi_mini/<timestamp>-<pid>` for logs and summaries.
- `BUILD_JOBS=32`: compilation parallelism.
- `AUTO_BUILD=0`: skip automatic compilation.
- `--dry-run`: preview commands without executing them.

Example:

```bash
THREADS=32 ANGLE=10 EF_BUILD=200 EF_UPDATE=100 R=40 \
  bash scripts/run/run_all.sh datasets/oi_mini
```

Compilation and summaries run automatically. Existing indexes are reused.
Outputs include `search.csv` (recall, latency in ms/query), `delete.csv` and
`insert.csv` (OPS/s and recall before/after each update), plus `summary.txt/json`.
To summarize saved logs again: `python3 scripts/run/summarize.py LOG_DIR`.
