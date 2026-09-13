#pragma once

namespace bucketing {

inline constexpr const char* kAbstractionDir = "/tmp/poker_solver_buckets";

struct Options {
    // River equity clusters, 1..65536 (uint16 map IDs). Thousands are typical;
    // more bins increase the dimension and memory of turn histograms.
    int river_bins = 2000;
    // Turn transition clusters, 1..65536. Thousands are typical; increasing this
    // also increases flop histogram memory and clustering cost.
    int turn_bins = 5000;
    // Flop transition clusters, 1..65536. Thousands are typical; increasing this
    // also increases the dimension of clustered preflop histograms.
    int flop_bins = 5000;
    // 169 preserves every canonical starting hand without clustering. Values
    // 1..168 cluster distributions over flop bins; >169 cannot split 169 inputs
    // into more nonempty clusters and is rejected. Try 50..100 for compression.
    int preflop_bins = 169;
    // Prepared training points per cluster for turn/flop/clustered preflop.
    // 64..256 is a practical starting range; tiny samples degrade clustering.
    // 0 prepares every hand and can require enormous RAM on turn/flop. Faiss
    // retains its own default internal sample cap for both sampling options.
    int points_per_centroid = 64;
    // Prepared river training points per cluster. 64..256 is a useful starting
    // range; 0 prepares all river equities. Does not reduce equity enumeration.
    int river_points_per_centroid = 64;
    // Positive K-means iteration count for every clustered street. 20..50 is a
    // reasonable starting range; more iterations cost more and may not help.
    int kmeans_iters = 20;
    // Nonnegative seed (0..INT_MAX) for reservoir sampling and Faiss K-means.
    // Any value is valid; hold it fixed for comparisons. Thread/library changes
    // can still affect numerical results.
    int seed = 1234;
    // Positive number of hands per assignment batch. 1024..8192 is a useful
    // starting range. Histogram buffer uses roughly rows * next-street bins *
    // sizeof(float) bytes; huge batches waste RAM, tiny batches add overhead.
    // This does not cap training matrices, river equities, or resident maps.
    int rows_per_chunk = 8192;
};

// Validate before allocating or generating maps. Throws std::invalid_argument.
void validate_options(const Options& options);
// Build all four uint16 maps, using identity preflop only when preflop_bins=169.
void build_offline(const Options& options);

}  // namespace bucketing
