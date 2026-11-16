# ACTIVE

ACTIVE is a dynamic, Pareto-frontier-based unified index for dual-vector
queries. It supports index construction, approximate nearest-neighbor search,
insertion, and deletion in a single index.

The paper and appendix are available on
[arXiv](https://arxiv.org/abs/2610.35751).

## Features

- Unified indexing and search for objects with two vector attributes.
- Dynamic insertion and deletion support.
- End-to-end scripts for data preparation, index construction, search, and
  update evaluation.
- A small OpenImage sample for quickly validating the full pipeline.

## Requirements

- Python 3.8 or later and Bash.
- A C++20-compatible compiler, CMake, Boost, TBB, and OpenMP.
- An x86 processor with AVX-512 and FMA support for the current distance
  implementation.

On Ubuntu or Debian, install the required system packages with:

```bash
sudo apt update
sudo apt install build-essential cmake python3 libboost-all-dev libtbb-dev
```

## Quick Start

Clone the repository and enter the project directory:

```bash
git clone https://github.com/iDC-NEU/ACTIVE.git
cd ACTIVE
```

The repository includes a sample of the OpenImage (OI) dataset containing
10,000 base vectors. Run the complete pipeline from the repository root:

```bash
bash scripts/run_pipeline.sh datasets/oi_mini
```


## Input Data

Each dataset directory must contain the following four files:

```text
base_img_emb.fvecs
base_text_emb.fvecs
query_img_emb.fvecs
query_text_emb.fvecs
```

## Pipeline

The end-to-end pipeline consists of two stages:

1. **Data preparation.** The pipeline generates query weights, static ground
   truth, an insertion base, insertion and deletion traces, and dynamic ground
   truth. Prepared files are written to `<dataset>/prepared/`. By default, the
   pipeline computes top-10 ground truth and creates 10 rounds of 1% updates
   for each update operation.
2. **Experiments.** The pipeline compiles the executables, builds or reuses the
   full-base and insertion indexes, runs static search, and evaluates deletion
   and insertion independently.

By default, indexes and build outputs are stored under
`scripts/run/artifacts/<dataset>/`, while logs and summaries are stored under
`scripts/run/log/<dataset>/`. Compatible prepared data and indexes are reused.

The generated reports include:

- `search.csv`: recall and latency in milliseconds per query.
- `delete.csv`: deletion throughput and recall before and after each round.
- `insert.csv`: insertion throughput and recall before and after each round.
- `summary.txt` and `summary.json`: consolidated experiment results.

See [scripts/README.md](scripts/README.md) for the complete pipeline interface,
[scripts/data_tools/README.md](scripts/data_tools/README.md) for data preparation
options, and [scripts/run/README.md](scripts/run/README.md) for individual
experiment stages.

## Configuration

The scripts are configured through environment variables. Common options
include:

| Variable | Default | Description |
| --- | ---: | --- |
| `THREADS` | `32` | Number of worker threads. |
| `K` | `10` | Number of nearest neighbors to return. |
| `ROUNDS` | `10` | Number of insertion or deletion rounds. |
| `UPDATE_RATE` | `0.01` | Fraction of the full base updated per round. |
| `EF_BUILD` | `200` | Candidate-pool size used during construction. |
| `EF_UPDATE` | `100` | Candidate-pool size used during updates. |
| `R` | `40` | Maximum number of neighbors per node. |
| `ANGLE` | `10` | Update angle threshold in degrees. |

For example:

```bash
THREADS=32 K=10 ROUNDS=10 UPDATE_RATE=0.01 \
  bash scripts/run_pipeline.sh datasets/oi_mini
```

## Project Structure

```text
ACTIVE/
├── include/                  # Index and component headers
├── src/                      # Library implementation and CMake configuration
├── test/                     # Index build, search, insertion, and deletion tools
├── datasets/
│   └── oi_mini/              # OpenImage sample base and query vectors
├── scripts/
│   ├── data_tools/           # Data preparation and exact ground-truth tools
│   ├── run/                  # Build, search, update, and summary scripts
│   ├── run_pipeline.sh       # End-to-end reproduction pipeline
│   └── README.md             # Pipeline options and examples
├── CMakeLists.txt            # Top-level build configuration
└── README.md                 # Project documentation
```

## Datasets

We evaluate ACTIVE on five publicly available, real-world dual-vector
datasets. Here, $D_x$ and $D_y$ denote the dimensions of the $x$- and
$y$-attribute vectors, respectively. Each dataset contains 1,000 query
vectors, except Caselaw, which contains 88.3K.

The OI, HM, and CM dataset files are available on
[Google Drive](https://drive.google.com/drive/folders/1K_IybJB1A94_8_ziF8rmVf4EtuMySbzr?usp=drive_link).
Please contact us if you need any of the other datasets used in the paper.

| Dataset | $D_x$ | $D_y$ | Vectors | Contents |
| --- | ---: | ---: | ---: | --- |
| OpenImage (OI) [1] | 768 | 768 | 507,444 | Text-Image |
| Howto100M (HM) [1] | 1,024 | 768 | 1,238,875 | Text-Video |
| CC3M (CM) [1] | 768 | 768 | 3,131,153 | Text-Image |
| LAION (LN) [2] | 512 | 512 | 10,642,155 | Text-Image |
| Caselaw (CW) [3] | 1,536 | 2 | 7,288,481 | Text-Coordinates |

## References

[1] Ziqi Yin, Jianyang Gao, Pasquale Balsebre, Gao Cong, and Cheng Long.
"DEG: Efficient Hybrid Vector Search Using the Dynamic Edge Navigation
Graph." *Proceedings of the ACM on Management of Data*, 3(1), Article 29,
2025. [https://doi.org/10.1145/3709679](https://doi.org/10.1145/3709679)

[2] Christoph Schuhmann, Romain Beaumont, Cade W. Gordon, Ross Wightman,
Theo Coombes, Aarush Katta, Clayton Mullis, Patrick Schramowski, Srivatsa R.
Kundurthy, Katherine Crowson, et al. "LAION-5B: An Open Large-Scale Dataset
for Training Next Generation Image-Text Models." 2022.
[https://arxiv.org/abs/2210.08402](https://arxiv.org/abs/2210.08402)

[3] Caselaw dataset preparation.
[https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/dataset_preparation/caselaw_dataset.md](https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/dataset_preparation/caselaw_dataset.md)

## Citation

If you use ACTIVE in your research, please cite the
[ACTIVE paper](https://arxiv.org/abs/2610.35751).
