# Dataset preparation

This script requires Python 3.8+ and a C++17 compiler. Run from the repository root:

```bash
python3 scripts/data_tools/prepare.py DATASET_DIR [options]
```

The script generates alpha, static GT, the first half of the base for insertion, and separate
insertion/deletion traces and dynamic GT from the four base/query `.fvecs` files.

Parameters:

- `--output`: output directory. Default `DATASET_DIR/prepared`. Must not exist.
- `--k 10`: number of nearest neighbors in each GT record.
- `--threads N`: GT workers. Defaults to at most 8 CPU threads.
- `--rounds 10`, `--update-rate 0.01`: update rounds and fraction of the full base per round.
- `--seed 42`, `--alpha-min 0`, `--alpha-max 1`: alpha random seed and sampling range.

Example:

```bash
python3 scripts/data_tools/prepare.py datasets/oi_mini \
  --threads 32 --k 10 --rounds 10 --update-rate 0.01 \
  --seed 42 --alpha-min 0 --alpha-max 1
```
