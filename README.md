# ACTIVE

ACTIVE is a dynamic Pareto-frontier-based unified index for dual-vector queries,
supporting index construction, search, insertion and deletion.
The paper and appendix are available on [arXiv](https://arxiv.org/abs/2610.35751).

## Prerequisites

- Python 3.8+ and Bash.
- A compiler with C++20 support, CMake, Boost, TBB and OpenMP.
- An x86 CPU with AVX-512/FMA support for the current distance implementation.

Install software dependencies on Ubuntu/Debian:

```bash
sudo apt install build-essential cmake python3 libboost-all-dev libtbb-dev
```

## Run Scripts

The reproduction scripts are in `scripts/`. A sample of the OpenImage (OI)
dataset with 10,000 base vectors is in `datasets/oi_mini/`.
`run_pipeline.sh` runs **data preprocessing, index construction, search and updates**.

Run the pipeline from the repository root:

```bash
bash scripts/run_pipeline.sh datasets/oi_mini
```

The pipeline has two stages.

1. Data preprocessing reads `base_img_emb.fvecs`, `base_text_emb.fvecs`,
   `query_img_emb.fvecs` and `query_text_emb.fvecs` from the dataset directory.
   It generates query alpha values, static ground truth, the first half of the base for
   insertion, and separate insertion/deletion traces and dynamic ground truth.
   Outputs are saved to the dataset's `prepared/` directory. Defaults are top-10
   ground truth and 10 rounds of 1% updates for each operation.
2. Experiments run through `scripts/run/run_all.sh`. It automatically compiles
   the executables, builds or reuses the full-base and insertion indexes,
   runs static search, and evaluates deletion and insertion separately. Logs and
   summaries report search recall and latency, plus each update round's OPS/s and recall.

The default thread count is 32. Indexes and new build outputs are stored in
`scripts/run/artifacts/oi_mini/`. Logs and summaries are stored in
`scripts/run/log/oi_mini/`. Existing matching prepared data and indexes are reused.
See [scripts/README.md](scripts/README.md) for parameters and examples.

## Project Structure

```text
ACTIVE/
├── include/              # Index and component headers
├── src/                  # Library implementation
├── test/                 # Build, search and update executables
├── datasets/             # Base/query vectors and prepared data
├── scripts/
│   ├── run_pipeline.sh   # Preprocessing and complete experiment pipeline
│   ├── data_tools/       # Alpha, trace and ground-truth preparation
│   ├── run/              # Experiment runners, logs and summaries
│   └── README.md         # Pipeline usage and parameters
├── CMakeLists.txt        # Build configuration
└── ACTIVE-FULL.pdf       # Paper PDF
```
