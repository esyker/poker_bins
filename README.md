# Poker Histogram Bins

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
cmake --build build --parallel --target build_bins
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
| `--points-per-centroid N` | 64 | Nonnegative; turn/flop/clustered preflop training sample size per centroid; 0 passes all hands to Faiss, which may subsample |
| `--river-points-per-centroid N` | 64 | Nonnegative; river training sample size per centroid; 0 passes all hands to Faiss, which may subsample |
| `--kmeans-iters N` | 20 | Positive K-means iteration count, used on each clustered street |
| `--seed N` | 1234 | 0..INT_MAX; seed for reservoir sampling and K-means initialization |
| `--assignment-batch-size N` | 8192 | Positive batch size for assigning hands to bins after K-means training. Smaller batches use less temporary RAM; larger batches require fewer search calls. |
| `--help`, `-h` | | Print usage without generating maps |

Considerations:

- All-hands training on turn/flop can require enormous RAM.
- Use a fixed seed for comparisons; library versions and thread counts may still affect numerical results.
- The transition assignment buffer requires roughly
`assignment_batch_size * next_street_bins * 4` bytes (about 156 MiB at 8192 rows and
5000 bins), plus maps and training data. Smaller chunks reduce that buffer;
they do not cap total memory. All numeric options must fit a signed 32-bit
integer. See `src/bucketing/build_bins.h` for guidance on every property.
- Output is fixed at `/tmp/poker_solver_buckets/`: `river_map.bin`, `turn_map.bin`,
`flop_map.bin`, and `preflop_map.bin`. Every run recomputes all streets in that
order and overwrites their maps. A volume or bind mount
preserves the files after the container exits.

### Native run

Save maps to `/tmp/poker_solver_buckets/` on the host, using defaults or custom options:

```bash
./build/build_bins
./build/build_bins --river-bins 1000 --turn-bins 2500 --flop-bins 2500 --preflop-bins 80 --points-per-centroid 128 --river-points-per-centroid 128 --kmeans-iters 30 --seed 1234 --assignment-batch-size 4096
```

Use `./build/build_bins --help` to check the CLI without generation.

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
docker run --rm -v poker-bins-maps:/tmp/poker_solver_buckets poker-bins --river-bins 1000 --turn-bins 2500 --flop-bins 2500 --preflop-bins 80 --points-per-centroid 128 --river-points-per-centroid 128 --kmeans-iters 30 --seed 1234 --assignment-batch-size 4096
```

To write directly to a host folder, replace the volume argument with
`--mount type=bind,source=/absolute/path/to/maps,target=/tmp/poker_solver_buckets`
(the host folder must already exist). Add `-e OMP_NUM_THREADS=8` before the image
name to choose the OpenMP thread count.

## Building Bins Complexity

### Feature Generation

Feature generation computes the equity values or probability histograms that
K-means uses as input.

The transition-street approach is called **histogram binning** because each
hand is represented by a histogram of its possible next-street outcomes:
count how many legal deals lead to each next-street bin, then divide by the
total number of deals to obtain probabilities. For example, if 10 of a turn
hand's 46 possible river cards lead to river bin 7, entry 7 of its histogram
is `10 / 46`. K-means then groups hands with similar histograms into the same
current-street bin. Thus, the histogram describes future outcomes, and binning
groups hands with similar outcome distributions. River starts the process
by clustering single equity values instead of histograms.

Streets are completed in order: **river → turn → flop
→ preflop**, so each earlier street can look up the next street's finished bin
map. With default settings:

- **River:** before K-means for River, compute equity for **all 123,156,254 distinct
  river hands**, evaluating **990 legal opponent hands per hand**. Train
  K-means on a sample of **128,000** of these equity values. After training,
  assign all **123,156,254** hands to their nearest centroid using the stored
  equities; equity is not recomputed.
- **Turn:** before K-means for Turn, build histograms for **320,000 sampled turn
  hands**. For each hand, enumerate **46 possible river cards**, look up
  each resulting hand's river bin, and convert the counts into probabilities.
  Train K-means on these histograms. After training, generate histograms for
  **all 13,960,050 turn hands**, again visiting **46 river cards per hand**,
  and assign each histogram to its nearest centroid.
- **Flop:** before K-means, build histograms for **320,000 sampled flop
  hands**, visiting **47 possible turn cards per hand** and looking up the
  resulting turn bins. Train K-means on these histograms. After training,
  generate histograms for **all 1,286,792 flop hands**, again visiting
  **47 turn cards per hand**, and assign each to its nearest centroid.
- **Preflop:** by default, directly map the **169 distinct starting hands**
  to 169 bins; no feature generation or K-means is needed. If fewer bins are
  requested, before K-means build histograms for `S = min(169, P * K)`
  sampled hands (`S = 169` for `P = 0`), enumerating **19,600 possible flops
  per hand** and looking up their flop bins. After training, regenerate
  histograms for **all 169 hands**, again visiting **19,600 flops per hand**,
  and assign each to its nearest centroid.

For turn, flop, and clustered preflop, sampled hands have their histograms
computed twice: once before training and again during full-map assignment.
The current implementation uses two separate passes:

1. **Training:** compute and store histograms for the sampled hands, then run
   K-means to learn the centroids.
2. **Assignment:** iterate through every hand, compute its histogram, and
   assign it to the nearest learned centroid, including hands already sampled.

Histograms are not regenerated on each K-means iteration; training reuses
the input vectors. Reducing the training sample reduces the work before
K-means on these streets, but every hand still needs assignment afterward.
Reducing bins or samples does not skip the exhaustive river equity calculation.

### Kmeans Clustering

For each street that uses clustering, the costs below use these quantities:

- `N` — **Fixed:** number of distinct hands after grouping suit-equivalent
  hands together. Determined by the street and poker rules.
  Counts: **169 preflop**, **1,286,792 flop**, **13,960,050 turn**,
  and **123,156,254 river**.
- `K` — **Hyperparameter:** number of bins (clusters) to group hands into.
  Set with `--river-bins`, `--turn-bins`, `--flop-bins`, or `--preflop-bins`.
  Smaller `K` reduces computation because each hand is compared with fewer
  clusters; larger `K` increases it and can also increase the training sample
  size and the previous street's feature dimension.
  However, more bins can improve abstraction accuracy by preserving finer
  differences between hands, at the cost of more computation and memory;
  fewer bins are cheaper but merge more distinct hands together.
  Defaults: **2,000 river**, **5,000 turn**, **5,000 flop**, and **169 preflop**.
- `D` — **Derived:** length of the feature vector that K-means compares for
  each hand. On river, this vector contains just the hand's showdown equity.
  On earlier streets, it contains the probability of reaching each bin on
  the next street after dealing the remaining card(s). For example, with
  2,000 river bins, each turn hand has a vector of 2,000 probabilities.
  With default bin counts: **river `D = 1`**, **turn `D = 2,000`** (river bins),
  and **flop `D = 5,000`** (turn bins). **Clustered preflop `D = 5,000`**
  (flop bins), but default preflop uses an identity map and skips K-means,
  so no feature vector is needed. Changing the next street's bin count
  changes `D` for the current street.
- `P` — **Hyperparameter:** training sample size per centroid for K-means training.
  Set with `--river-points-per-centroid` for river or `--points-per-centroid`
  for other streets. Our code randomly samples up to `P * K` hands and passes
  their feature vectors to Faiss to learn the `K` centroids. This is a total
  sample size, not a guarantee of `P` hands in each resulting cluster.
  Smaller positive `P` means fewer sampled vectors, less training data memory,
  and usually fewer distance calculations during K-means; larger `P` increases
  the sample until all `N` hands are included. `0` disables our sampling,
  but Faiss can still subsample internally.
  Defaults: **`--river-points-per-centroid 64`** and
  **`--points-per-centroid 64`**. With default bin counts, this selects
  **128,000 river hands** and **320,000 hands each for turn and flop**;
  default preflop skips clustering.
  **Faiss defaults to `max_points_per_centroid = 256`**, which our code does
  not override. For `1 <= P <= 256`, Faiss uses the entire sample we pass it.
  For `P > 256`, it randomly reduces that sample to `256 * K` vectors if
  more than that many are available. With `P = 0`, the same cap applies to
  the full set of `N` hands. The CLI value controls our input sample, but
  does not override Faiss's training cap: increasing `P` above 256 can
  increase feature-generation work and input memory without increasing
  the number of vectors used to learn centroids.
  For example, with `K = 5,000` and enough hands, `P = 512` passes
  2,560,000 vectors to Faiss, but it trains on only 1,280,000.
  This cap is confirmed in the [Faiss 1.7.2 source](https://github.com/facebookresearch/faiss/blob/v1.7.2/faiss/Clustering.cpp),
  matching the [Ubuntu 22.04 package version](https://packages.ubuntu.com/ca/jammy/libfaiss-dev)
  used by the current Dockerfile. Native builds use the installed Faiss
  version; CMake does not pin one.
- `S` — **Derived:** number of sampled hand feature vectors passed to Faiss:
  `S = min(N, P * K)` when `P > 0`, or `S = N` when `P = 0`.
  Faiss's default `max_points_per_centroid` is 256, which our code leaves
  unchanged: if `S > 256 * K`, Faiss trains on a smaller sample of `256 * K`
  vectors. We still allocate the full `S`-vector input matrix.
  With default settings: **river `S = 128,000`**, **turn `S = 320,000`**,
  and **flop `S = 320,000`**. **Preflop has no training sample by default**
  because its 169-bin identity map skips clustering.
- `I` — **Hyperparameter:** number of K-means training iterations,
  set with `--kmeans-iters`.
  Smaller `I` reduces training work because K-means makes fewer passes;
  larger `I` increases training work roughly linearly, without increasing
  the training data size. Fewer iterations may produce poorer clusters.
  Default: **`--kmeans-iters 20`** for every clustered street. Our code sets
  Faiss's `niter` directly, overriding its default of 25; there is no separate
  iteration cap in our code. Each iteration assigns training vectors to
  centroids and updates the centroids. More iterations can improve clustering,
  but improvements may diminish; they do not increase the sample size or
  override Faiss's sampling cap. Iteration cost also grows with the effective
  training sample size, `K`, and `D`. Default preflop skips training.

| Computation | Time complexity | What it does |
| --- | --- | --- |
| K-means training | Roughly `O(I * min(S, 256 * K) * K * D)` | Learns `K` centroids over `I` iterations, using Faiss's capped training sample |
| Assignment after training | `O(N * K * D)` | Compares every hand's feature vector with the learned centroids to choose its bin |

## Memory Complexity

Memory usage comes from the training matrix, assignment buffers, river equity
array, bin maps, and centroid vectors. The training matrix scales with `S * D`,
while assignment processes hands in batches to limit its temporary buffer to
`B * D` feature values. The table below gives the storage required by each
allocation.

- `B` — **Assignment batch size:** maximum number of hands whose histograms
  are held in the temporary assignment buffer at once, after K-means training.
  Set with **`--assignment-batch-size 8192`** by default. Smaller values use
  less buffer memory; this does not change overall time complexity or cap
  total RAM. River reuses its stored equities instead of buffering histograms.

| Stored data | Memory | Used for |
| --- | --- | --- |
| Training vectors and sampled hand IDs | `4 * S * D + 8 * S` bytes | Holds the sample passed to Faiss for training |
| Assignment buffers | `4 * B * D + 12 * B` bytes on transition streets; `12 * B` on river | Holds one batch of histograms, bin IDs, and distances after training |
| River equities | `4 * N` bytes | Stores all river equities for reuse in training and assignment |
| Bin map per street | `2 * N` bytes in RAM and the same on disk | Stores each hand's bin ID; completed maps stay in RAM while earlier streets are built |
| Centroid vectors | `4 * K * D` bytes per copy | Stores the learned cluster centers; Faiss can hold additional copies |

The formulas above describe individual memory allocations. For example, on
flop the defaults are `S = 320,000` training vectors, `D = 5,000` features per
vector, and `B = 8,192` hands per assignment batch. With 4-byte floating-point
values, the training matrix occupies `4 * S * D` bytes (about **6.0 GiB**),
while the assignment histogram buffer occupies `4 * B * D` bytes
(about **156 MiB**).

Reducing `P` reduces `S` and therefore training matrix memory; reducing `B`
reduces assignment buffer memory. Reducing the next street's bin count reduces
`D` and therefore both allocations. These figures exclude bin maps, sampled
hand IDs, centroids, and Faiss's additional allocations, so neither represents
total process RAM usage.


## Dependencies

| Dependency | Purpose | Source and storage |
| --- | --- | --- |
| **OMPEval** | Evaluates poker hands | [zekyll/OMPEval](https://github.com/zekyll/OMPEval/tree/4aec210ff75b0851af0ee170b35a7899e1a4fe8f), stored in `src/third_party/OMPEval` |
| **hand-isomorphism** | Canonical indexing of suit-equivalent hands | [kdub0/hand-isomorphism](https://github.com/kdub0/hand-isomorphism/tree/dabcee4a84c1d62ee6ded9b6ff02ece6823fcc0f), stored in `src/third_party/hand-isomorphism` |
| **Faiss** | K-means clustering and nearest-centroid assignment | [facebookresearch/faiss](https://github.com/facebookresearch/faiss); Ubuntu package `libfaiss-dev` |
| **OpenMP** | Runs calculation loops across CPU threads | GCC compiler/runtime support, detected by CMake |
| **OpenBLAS / LAPACK** | Numerical dependencies of Faiss | Ubuntu packages `libopenblas-dev` and `liblapack-dev` |
