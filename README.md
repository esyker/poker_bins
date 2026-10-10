# Poker Bins

Builds offline Texas hold'em card-abstraction maps commonly called buckets so a solver can share a
strategy across similar hands, significantly reducing computing time.

## Algorithm

Suit-equivalent hands share a canonical index through hand-isomorphism. The
pipeline builds maps from the river backwards:

| Street | Feature | Default bins |
| --- | --- | ---: |
| River | Exact showdown equity against a uniformly random legal opponent hand | 2,000 |
| Turn | Distribution over river bins after each of the 46 possible river cards | 5,000 |
| Flop | Distribution over turn bins after each of the 47 possible turn cards | 5,000 |
| Preflop | Identity at 169 bins; otherwise a distribution over flop bins for all 19,600 legal flops | 169 |

OMPEval evaluates river hands against all 990 legal opponent hands; equity is
wins plus half of ties, divided by 990. Faiss K-means groups equities and
next-street probability histograms using squared Euclidean distance, with 20
iterations and seed 1234 by default (both configurable). Histograms capture possible future outcomes rather
than only current hand strength.

## Install

For a native Ubuntu installation, from the repository root:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ca-certificates libfaiss-dev libopenblas-dev liblapack-dev
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel --target build_offline
```

Only the vendored indexer's two C files and OMPEval's `HandEvaluator.cpp` are compiled.
A custom Faiss installation can be selected with
`-Dfaiss_DIR=/path/to/lib/cmake/faiss`. No GPU or Python is required.

For Docker, the Linux image includes the compiler, CMake, CPU Faiss, OpenBLAS,
LAPACK, OpenMP support, hand-isomorphism, and OMPEval:

```bash
docker build -t poker-bins .
```

The Dockerfile uses Ubuntu 24.04 and installs dependencies before compiling the
single executable. Build with Linux containers. The image has not been built or
run as part of this change.

## How to run

The following is a list of parameters that can be passed to the command line tool:

| Option | Default | Allowed values |
| --- | ---: | --- |
| `--river-bins N` | 2000 | 1..65536 |
| `--turn-bins N` | 5000 | 1..65536 |
| `--flop-bins N` | 5000 | 1..65536 |
| `--preflop-bins N` | 169 | 1..169; 169 preserves the identity map without clustering; 1..168 clusters distributions over flop bins; >169 is rejected because there are only 169 canonical starting hands |
| `--points-per-centroid N` | 64 | Nonnegative; turn/flop/clustered preflop prepared training points per cluster; 0 prepares all hands |
| `--river-points-per-centroid N` | 64 | Nonnegative; river prepared training points per cluster; 0 prepares all hands |
| `--kmeans-iters N` | 20 | Positive K-means iteration count, used on each clustered street |
| `--seed N` | 1234 | 0..INT_MAX; seed for reservoir sampling and K-means initialization |
| `--rows-per-chunk N` | 8192 | Positive batch size for assigning hands to bins after K-means training. Smaller batches use less temporary RAM; larger batches require fewer search calls. |
| `--help`, `-h` | | Print usage without generating maps |

Considerations:

- All-hands training on turn/flop can require enormous RAM.
- Use a fixed seed for comparisons; library versions and thread counts may still affect numerical results.
- The transition assignment buffer requires roughly
`rows_per_chunk * next_street_bins * 4` bytes (about 156 MiB at 8192 rows and
5000 bins), plus maps and training data. Smaller chunks reduce that buffer;
they do not cap total memory. All numeric options must fit a signed 32-bit
integer. See `src/bucketing/build_offline.h` for guidance on every property.
- Output is fixed at `/tmp/poker_solver_buckets/`: `river_map.bin`, `turn_map.bin`,
`flop_map.bin`, and `preflop_map.bin`. Every run recomputes all streets in that
order and overwrites their maps. A volume or bind mount
preserves the files after the container exits.

### Native run

Save maps to `/tmp/poker_solver_buckets/` on the host, using defaults or custom options:

```bash
./build/build_offline
./build/build_offline --river-bins 1000 --turn-bins 2500 --flop-bins 2500 --preflop-bins 80 --points-per-centroid 128 --river-points-per-centroid 128 --kmeans-iters 30 --seed 1234 --rows-per-chunk 4096
```

Use `./build/build_offline --help` to check the CLI without generation.

### Docker run

Use `docker run --rm poker-bins --help` to check the CLI without generation.

Run all streets with default counts, keeping output in a Docker volume:

```bash
docker run --rm -v poker-bins-maps:/tmp/poker_solver_buckets poker-bins
```

The optional flag `--rm` deletes the container after it stops.
`-v poker-bins-maps:/tmp/poker_solver_buckets` connects a persistent Docker volume to the directory where the program writes its maps.

Or choose counts, including clustered preflop, and training samples:

```bash
docker run --rm -v poker-bins-maps:/tmp/poker_solver_buckets poker-bins --river-bins 1000 --turn-bins 2500 --flop-bins 2500 --preflop-bins 80 --points-per-centroid 128 --river-points-per-centroid 128 --kmeans-iters 30 --seed 1234 --rows-per-chunk 4096
```

To write directly to a host folder, replace the volume argument with
`--mount type=bind,source=/absolute/path/to/maps,target=/tmp/poker_solver_buckets`
(the host folder must already exist). Add `-e OMP_NUM_THREADS=8` before the image
name to choose the OpenMP thread count.

## Complexity

For each clustered street, let `N` be its number of canonical hands, `K` its
bin count, `D` its feature dimension (1 for river, next-street bin count
otherwise), `S` its prepared training sample size, `B` the rows per chunk,
and `I` the K-means iteration count. With `P` points per centroid,
`S = min(N, P * K)`; `P = 0` uses all `N` hands.

| Work or storage | Cost |
| --- | --- |
| Prepared training data | `4 * S * D` bytes, plus `8 * S` bytes for sampled hand IDs |
| Assignment buffers | About `4 * B * D + 16 * B` bytes on transition streets; river reuses its equity array |
| River equities | `4 * N` bytes, retained through river assignment |
| Maps in RAM and on disk | `2 * N` bytes per street; earlier maps remain in RAM while later streets are built |
| K-means training | Up to roughly `O(I * S * K * D)` work; Faiss may internally subsample the prepared data |
| Assigning all hands to bins | `O(N * K * D)` distance-comparison work after training |

With default settings, the flop training matrix alone uses about **6.0 GiB**
(`320,000 * 5,000 * 4` bytes), and its assignment histogram buffer uses
**156 MiB**. These coexist with maps, sampled IDs, centroids, and library
allocations, so they are not total RAM estimates. Smaller chunks only reduce
assignment buffers; fewer training samples reduce the training matrix.

Feature generation also costs time: river evaluates all 990 legal opponent
hands per canonical hand; each transition histogram visits 46 river cards,
47 turn cards, or 19,600 flops for clustered preflop. Transition histograms
are generated for training samples and again for all hands during assignment.
Reducing bins or samples does not skip exhaustive river equity calculation or
full-map assignment. Default preflop uses a 169-entry identity map and skips
training and histogram generation.

## Dependencies

| Dependency | Purpose | Source and storage |
| --- | --- | --- |
| **OMPEval** | Evaluates poker hands | [zekyll/OMPEval](https://github.com/zekyll/OMPEval/tree/4aec210ff75b0851af0ee170b35a7899e1a4fe8f), stored in `src/third_party/OMPEval` |
| **hand-isomorphism** | Canonical indexing of suit-equivalent hands | [kdub0/hand-isomorphism](https://github.com/kdub0/hand-isomorphism/tree/dabcee4a84c1d62ee6ded9b6ff02ece6823fcc0f), stored in `src/third_party/hand-isomorphism` |
| **Faiss** | K-means clustering and nearest-centroid assignment | [facebookresearch/faiss](https://github.com/facebookresearch/faiss); Ubuntu package `libfaiss-dev` |
| **OpenMP** | Runs calculation loops across CPU threads | GCC compiler/runtime support, detected by CMake |
| **OpenBLAS / LAPACK** | Numerical dependencies of Faiss | Ubuntu packages `libopenblas-dev` and `liblapack-dev` |
