# Full pipeline

Run from the repository root. The pipeline prepares the data, compiles the
programs, then runs index construction, static search, deletion, insertion
and log summarization.

```bash
bash scripts/run_pipeline.sh [DATASET_DIR] [--dry-run]
```

Input files: `base_img_emb.fvecs`, `base_text_emb.fvecs`,
`query_img_emb.fvecs` and `query_text_emb.fvecs`.

Parameters (environment variables):

- `DATASET_DIR`: dataset directory. Default `datasets/oi_mini`.
- `THREADS=32`, `K=10`: worker threads and number of nearest neighbors.
- `ROUNDS=10`, `UPDATE_RATE=0.01`: update rounds and fraction of the full base per round.
- `SEED=42`, `ALPHA_MIN=0`, `ALPHA_MAX=1`: alpha random seed and sampling range.
- `PREPARED_DIR`: prepared data directory. Default `DATASET_DIR/prepared`.
- `RUN_DIR`, `LOG_DIR`: index/build and log directories. See [run/README.md](run/README.md).
- `--dry-run`: preview commands without executing them.

Example:

```bash
THREADS=32 K=10 ROUNDS=10 UPDATE_RATE=0.01 \
  bash scripts/run_pipeline.sh datasets/oi_mini
```

Matching prepared data and existing indexes are reused. Results stay inside the project.
