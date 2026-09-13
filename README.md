# Poker Bins

Builds offline Texas hold'em card-abstraction maps so a solver can share a
strategy across similar hands. Extracted from `poker_solver/cpp/src/nlhe_game`,
with configurable bucket counts and one entry point that builds all
streets. Changes are local to this repository.

## Purpose and algorithm

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

By default, training features are prepared for a reservoir sample of 64 points
per cluster. Every canonical hand is still assigned to its nearest centroid.
Transition assignment streams configurable chunks of 8,192 hands by default
to avoid retaining the full histogram matrix. River equities and completed maps
remain in memory.

Each map is a headerless, native-endian `uint16_t` array: canonical hand ID is
the array index, and bucket ID is the value. Consumers must use matching bucket
counts and the same indexer/card grouping (`{2}`, `{2,3}`, `{2,4}`, `{2,5}`).
Cluster labels are not strength rankings. Custom counts require configuring the
consumer accordingly; `poker_solver` still has its original fixed counts.

The source consists of `src/bucketing/build_offline.cpp` (pipeline),
`src/bucketing/build_offline.h` (options and output path),
`src/bucketing/hand_isomorphism.h` (C++ adapter), and
`src/cli/build_offline.cpp` (argument parsing).

## Dependencies

| Dependency | Purpose | Needed? | Source and storage |
| --- | --- | --- | --- |
| **OMPEval** | Evaluates poker hands | Yes | [zekyll/OMPEval](https://github.com/zekyll/OMPEval/tree/4aec210ff75b0851af0ee170b35a7899e1a4fe8f), stored in `src/third_party/OMPEval` |
| **hand-isomorphism** | Canonical indexing of suit-equivalent hands | Yes | [kdub0/hand-isomorphism](https://github.com/kdub0/hand-isomorphism/tree/dabcee4a84c1d62ee6ded9b6ff02ece6823fcc0f), stored in `src/third_party/hand-isomorphism` |
| **Faiss** | K-means clustering and nearest-centroid assignment | Yes | [facebookresearch/faiss](https://github.com/facebookresearch/faiss); Ubuntu package `libfaiss-dev` |
| **OpenMP** | Runs calculation loops across CPU threads | For existing parallel execution | GCC compiler/runtime support, detected by CMake |
| **OpenBLAS / LAPACK** | Numerical dependencies of Faiss | For the selected Faiss build | Ubuntu packages `libopenblas-dev` and `liblapack-dev` |

The two vendored libraries contain only required, unmodified upstream files and
their original `LICENSE.txt` files. OMPEval was developed by Timo A.;
hand-isomorphism was developed by Kevin Waugh. Their pinned revisions are:

- OMPEval: `4aec210ff75b0851af0ee170b35a7899e1a4fe8f`.
- hand-isomorphism: `dabcee4a84c1d62ee6ded9b6ff02ece6823fcc0f`.

These are ordinary source files stored with this repository, not Git submodules.
CMake compiles them directly from `src/third_party`; it does not download them.
They remain available if upstream disappears or becomes private. System
dependencies still come from Ubuntu repositories and are not pinned to exact
package versions. To update a vendored library, replace its required files and
license from a chosen upstream commit and update the revision recorded here.

## Install

The Linux Docker image includes the compiler, CMake, CPU Faiss, OpenBLAS,
LAPACK, OpenMP support, hand-isomorphism, and OMPEval. From the repository root:

```bash
docker build -t poker-bins .
```

The Dockerfile uses Ubuntu 24.04 and installs dependencies before compiling the
single executable. Build with Linux containers. The image has not been built or
run as part of this change.

For a native Ubuntu installation instead:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ca-certificates libfaiss-dev libopenblas-dev liblapack-dev
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel --target build_offline
```

Only the vendored indexer's two C files and OMPEval's `HandEvaluator.cpp` are compiled.
A custom Faiss installation can be selected with
`-Dfaiss_DIR=/path/to/lib/cmake/faiss`. No GPU or Python is required.

## How to run

Run all streets with default counts, keeping output in a Docker volume:

```bash
docker run --rm -v poker-bins-maps:/tmp/poker_solver_buckets poker-bins
```

Or choose counts, including clustered preflop, and training samples:

```bash
docker run --rm -v poker-bins-maps:/tmp/poker_solver_buckets poker-bins --river-bins 1000 --turn-bins 2500 --flop-bins 2500 --preflop-bins 80 --points-per-centroid 128 --river-points-per-centroid 128 --kmeans-iters 30 --seed 1234 --rows-per-chunk 4096
```

To write directly to a host folder, replace the volume argument with
`--mount type=bind,source=/absolute/path/to/maps,target=/tmp/poker_solver_buckets`
(the host folder must already exist). Add `-e OMP_NUM_THREADS=8` before the image
name to choose the OpenMP thread count.

Native equivalents:

```bash
./build/build_offline
./build/build_offline --river-bins 1000 --turn-bins 2500 --flop-bins 2500 --preflop-bins 80 --points-per-centroid 128 --river-points-per-centroid 128 --kmeans-iters 30 --seed 1234 --rows-per-chunk 4096
```

| Option | Default | Allowed values |
| --- | ---: | --- |
| `--river-bins N` | 2000 | 1..65536 |
| `--turn-bins N` | 5000 | 1..65536 |
| `--flop-bins N` | 5000 | 1..65536 |
| `--preflop-bins N` | 169 | 1..169; 169 preserves the identity map, fewer clusters flop-bin probabilities |
| `--points-per-centroid N` | 64 | Nonnegative; turn/flop/clustered preflop prepared training points per cluster |
| `--river-points-per-centroid N` | 64 | Nonnegative; river prepared training points per cluster |
| `--kmeans-iters N` | 20 | Positive K-means iteration count |
| `--seed N` | 1234 | 0..INT_MAX; reservoir sampling and K-means seed |
| `--rows-per-chunk N` | 8192 | Positive assignment batch size |
| `--help`, `-h` | | Print usage without generating maps |

Preflop uses the existing identity map at 169 bins. For 1..168 bins, it clusters
all or sampled canonical starting hands using their probabilities of reaching
each flop bin. The probabilities count every legal unordered flop equally,
including suit multiplicities; assignment still covers all 169 starting hands.
More than 169 preflop clusters is rejected because there are only 169 inputs.

For either sample option, `0` prepares every hand on the affected streets.
Faiss retains its default internal subsampling. Try 64..256 points per centroid
and 20..50 K-means iterations initially. All-hands training on turn/flop can
require enormous RAM; increasing samples never skips exhaustive river equities.
Use a fixed seed for comparisons; library versions and thread counts may still
affect numerical results. Default preflop identity ignores clustering settings.

Try 1024..8192 rows per chunk. The transition assignment buffer requires roughly
`rows_per_chunk * next_street_bins * 4` bytes (about 156 MiB at 8192 rows and
5000 bins), plus maps and training data. Smaller chunks reduce that buffer;
they do not cap total memory. All numeric options must fit a signed 32-bit
integer. See `src/bucketing/build_offline.h` for guidance on every property.

Output is fixed at `/tmp/poker_solver_buckets/`: `river_map.bin`, `turn_map.bin`,
`flop_map.bin`, and `preflop_map.bin`. Every run recomputes all streets in that
order and overwrites their maps. There is no resume mode. A volume or bind mount
preserves the files after the container exits.

Generation covers every canonical hand, including exhaustive river equities.
Lower bin counts or smaller training samples do not turn it into a quick test.
Use `docker run --rm poker-bins --help` to check the CLI without generation.
