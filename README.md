# ACTIVE

ACTIVE: A graph-based index for approximate nearest neighbor search with dual-distance (embedding + spatial) constraints. Supports efficient build, search, insert, and streaming update operations.

## Prerequisites

- **C++17** or later
- **CMake** >= 2.8
- **Boost** (for `dynamic_bitset`)
- **TBB** (Intel Threading Building Blocks)
- **OpenMP**

Install dependencies on Ubuntu/Debian:

```bash
sudo apt install build-essential cmake libboost-all-dev libtbb-dev
```

## Compilation

```bash
cd ACTIVE
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

The compiled executables will be located under `build/test/`.

To compile individual tools under `tools/` (data preparation utilities):

```bash
mkdir -p tools/exe
g++ tools/gen_alpha.cpp -o tools/exe/gen_alpha
g++ tools/process_base.cpp -o tools/exe/process_base
g++ tools/gen_trace.cpp -o tools/exe/gen_trace
g++ tools/gen_gt.cpp -o tools/exe/gen_gt -fopenmp
```

## Four Functionalities

### 1. Build Index

Build a DEG (Distance-Enhanced Graph) index from base embedding and location vectors.

| Item | Path |
|------|------|
| Source | [`test/build_index.cpp`](test/build_index.cpp) |
| Executable | `build/test/build_index` |

**Usage:**

```
./build_index <vec_base_emb> <vec_base_loc> <graph_index> <ef(L)> <max_edges(R)> <threads> <max_emb_dis> <max_loc_dis>
```

### 2. Search

Search the index for k-nearest neighbors given a query set with alpha-weighted distance.

| Item | Path |
|------|------|
| Source | [`test/search.cpp`](test/search.cpp) |
| Executable | `build/test/search` |

**Usage:**

```
./search <vec_base_emb> <vec_base_loc> <graph_index> <L> <K> <alpha_file> <query_emb> <query_loc> <query_gt> <threads> <max_emb_dis> <max_loc_dis> <search_type>
```

### 3. Insert

Insert new data points into an existing DEG index.

| Item | Path |
|------|------|
| Source | [`test/insert.cpp`](test/insert.cpp) |
| Executable | `build/test/insert` |

**Usage:**

```
./insert <graph> <base_emb> <base_loc> <trace1> [trace2 ...] <gt1> [gt2 ...] <ef> <max_edges> <threads> <id_flag> <delete_mode> <L> <K> <angle> <query_alpha> <query_emb> <query_loc>
```

### 4. Update (Single-Round)

Perform a single-round update (insert / delete / both) on the index, with search evaluation between rounds.

| Item | Path |
|------|------|
| Source | [`test/update.cpp`](test/update.cpp) |
| Executable | `build/test/update` |

**Usage:**

```
./update <graph_path> <base_emb> <base_loc> <trace_path> <ef(L)> <max_edges(R)> <threads> <id_flag> [L K query_alpha query_emb query_loc query_gt] <delete_mode>
```

---

## Data Preparation Scripts

Scripts for generating datasets, traces, and ground truth:

| Script | Description |
|--------|-------------|
| [`tools/process_dataset.sh`](tools/process_dataset.sh) | Full pipeline: gen_alpha, process_base, gen_trace, gen_gt |
| [`tools/gen_trace_gt.sh`](tools/gen_trace_gt.sh) | Generate update traces and ground truth for streaming benchmarks |
| [`tools/et_base.sh`](tools/et_base.sh) | Dataset preprocessing with sliding-window ground truth generation |

## Data Format

The index works with `.fvecs` (float vectors) and `.ivecs` (integer vectors) binary formats:
- Each file starts with a 4-byte dimension header
- Each vector is stored as `(dim, values...)` where `dim` is a 4-byte integer followed by `dim` elements

## Project Structure

```
├── include/          # Header files (index, component, builder, etc.)
├── src/              # Library source files
├── test/             # Executable entry points (build, search, insert, update)
├── tools/            # Data preparation utilities
├── CMakeLists.txt    # Top-level build configuration
└── ACTIVE-FULL.pdf   # Full paper
```
