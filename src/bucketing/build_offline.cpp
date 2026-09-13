// Offline abstraction: river equities, turn/flop transition histograms, optional preflop clustering.
#include "bucketing/build_offline.h"

#include "bucketing/hand_isomorphism.h"
#include "omp/HandEvaluator.h"

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <initializer_list>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// hand-isomorphism and OMPEval both encode a card as 4*rank + suit over 0..51,
// so a hand-isomorphism card index is directly a valid OMPEval card.
omp::Hand make_hand(const std::uint8_t* cards, int n) {
    omp::Hand h = omp::Hand::empty();
    for (int i = 0; i < n; ++i) h += omp::Hand(cards[i]);
    return h;
}

void init_indexer(hand_indexer_t* idx, std::initializer_list<std::uint8_t> cpr) {
    std::vector<std::uint8_t> rounds(cpr);
    if (!hand_indexer_init(static_cast<uint_fast32_t>(rounds.size()),
                           rounds.data(), idx)) {
        throw std::runtime_error("hand_indexer_init failed");
    }
}

template <class T>
void write_bin(const std::string& path, const T* data, std::int64_t count) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot open for writing: " + path);
    if (std::fwrite(data, sizeof(T), static_cast<std::size_t>(count), f) !=
        static_cast<std::size_t>(count)) {
        std::fclose(f);
        throw std::runtime_error("short write: " + path);
    }
    std::fclose(f);
}

// --- the sampling function ------------------------------------------------
//
// Choose which hand ids to train K-means on. Returns `sample_size` distinct ids
// drawn uniformly from [0, n) via reservoir sampling, or all of [0, n) when
// sample_size is 0 or >= n (i.e. sampling disabled).
std::vector<std::int64_t> sample_ids(std::int64_t n, std::int64_t sample_size, int seed) {
    if (sample_size <= 0 || sample_size >= n) {
        std::vector<std::int64_t> all(static_cast<std::size_t>(n));
        std::iota(all.begin(), all.end(), std::int64_t{0});
        return all;
    }
    std::vector<std::int64_t> res(static_cast<std::size_t>(sample_size));
    std::mt19937_64 rng(seed);
    for (std::int64_t i = 0; i < n; ++i) {
        if (i < sample_size) {
            res[static_cast<std::size_t>(i)] = i;  // fill the reservoir
        } else {
            std::uniform_int_distribution<std::int64_t> pick(0, i);
            const std::int64_t j = pick(rng);
            if (j < sample_size) res[static_cast<std::size_t>(j)] = i;
        }
    }
    return res;
}

// Run K-means on an in-memory sample and return the k*dim centroids.
std::vector<float> train_centroids(int dim, int k, std::int64_t n,
                                   const float* x, const bucketing::Options& options) {
    faiss::ClusteringParameters cp;
    cp.niter = options.kmeans_iters;
    cp.seed = options.seed;
    faiss::Clustering clustering(dim, k, cp);
    faiss::IndexFlatL2 index(dim);
    clustering.train(n, x, index);
    return clustering.centroids;
}

// --- the hand -> bucket function ------------------------------------------
//
// Compute a transition histogram: enumerate legal next-street deals, look
// up which bucket the resulting next-street hand falls in, and tally.
void fill_histogram(const hand_indexer_t& self, const hand_indexer_t& next,
                    const std::vector<std::uint16_t>& prev_map, int this_cards,
                    int dim, std::int64_t id, float* out) {
    std::fill(out, out + dim, 0.0f);
    std::uint8_t cards[7];  // hole cards and board (<= 7)
    hand_unindex(&self, this_cards == 2 ? 0 : 1, static_cast<hand_index_t>(id), cards);
    bool used[52] = {false};
    for (int i = 0; i < this_cards; ++i) used[cards[i]] = true;

    // Preflop deals an unordered three-card flop: C(50,3) = 19600 equally
    // likely boards. Count actual deals, not canonical boards (their suit
    // multiplicities differ), then normalize to a probability distribution.
    if (this_cards == 2) {
        for (int a = 0; a < 52; ++a) {
            if (used[a]) continue;
            cards[2] = static_cast<std::uint8_t>(a);
            for (int b = a + 1; b < 52; ++b) {
                if (used[b]) continue;
                cards[3] = static_cast<std::uint8_t>(b);
                for (int c = b + 1; c < 52; ++c) {
                    if (used[c]) continue;
                    cards[4] = static_cast<std::uint8_t>(c);
                    const auto next_id = hand_index_last(&next, cards);
                    out[prev_map[static_cast<std::size_t>(next_id)]] += 1.0f;
                }
            }
        }
        for (int bin = 0; bin < dim; ++bin) out[bin] /= 19600.0f;
        return;
    }
    const float weight = 1.0f / static_cast<float>(52 - this_cards);
    for (int c = 0; c < 52; ++c) {
        if (used[c]) continue;
        cards[this_cards] = static_cast<std::uint8_t>(c);
        const hand_index_t next_id = hand_index_last(&next, cards);
        out[prev_map[static_cast<std::size_t>(next_id)]] += weight;
    }
}

std::int64_t sample_size_for(int points_per_centroid, int k) {
    return points_per_centroid <= 0 ? 0
                                     : static_cast<std::int64_t>(points_per_centroid) * k;
}

// ---------------------------------------------------------------------------
// River: showdown equity vs a uniform random opponent, clustered (1-D).
// ---------------------------------------------------------------------------
std::vector<std::uint16_t> build_river(const std::string& dir, const bucketing::Options& options) {
    const int bins = options.river_bins;
    hand_indexer_t river;
    init_indexer(&river, {2, 5});
    const std::int64_t n = static_cast<std::int64_t>(hand_indexer_size(&river, 1));

    // Equity must be computed for every hand (each one needs a bucket), so it is
    // held in RAM and reused for both training and assignment.
    std::vector<float> equity(static_cast<std::size_t>(n));
#pragma omp parallel
    {
        omp::HandEvaluator eval;
#pragma omp for schedule(dynamic, 4096)
        for (std::int64_t id = 0; id < n; ++id) {
            std::uint8_t cards[7];
            hand_unindex(&river, 1, static_cast<hand_index_t>(id), cards);
            const std::uint16_t my_rank = eval.evaluate(make_hand(cards, 7));

            bool used[52] = {false};
            for (int i = 0; i < 7; ++i) used[cards[i]] = true;
            const omp::Hand board = make_hand(cards + 2, 5);

            int wins = 0, ties = 0, total = 0;
            for (int a = 0; a < 52; ++a) {
                if (used[a]) continue;
                for (int b = a + 1; b < 52; ++b) {
                    if (used[b]) continue;
                    const std::uint16_t opp_rank =
                        eval.evaluate(board + omp::Hand(a) + omp::Hand(b));
                    if (my_rank > opp_rank) ++wins;
                    else if (my_rank == opp_rank) ++ties;
                    ++total;
                }
            }
            equity[static_cast<std::size_t>(id)] =
                (wins + 0.5f * ties) / static_cast<float>(total);
        }
    }

    // Prepare the configured river equity training sample.
    const std::vector<std::int64_t> ids =
        sample_ids(n, sample_size_for(options.river_points_per_centroid, bins), options.seed);
    std::printf("[river] %lld hands, training on %zu\n",
                static_cast<long long>(n), ids.size());
    std::vector<float> samples(ids.size());
    for (std::size_t r = 0; r < ids.size(); ++r)
        samples[r] = equity[static_cast<std::size_t>(ids[r])];
    const std::vector<float> centroids =
        train_centroids(1, bins, static_cast<std::int64_t>(ids.size()),
                        samples.data(), options);

    // Assign every hand to its nearest centroid (equity is already contiguous).
    faiss::IndexFlatL2 index(1);
    index.add(bins, centroids.data());
    std::vector<std::uint16_t> map(static_cast<std::size_t>(n));
    std::vector<faiss::idx_t> labels(static_cast<std::size_t>(options.rows_per_chunk));
    std::vector<float> dist(static_cast<std::size_t>(options.rows_per_chunk));
    for (std::int64_t start = 0; start < n; start += options.rows_per_chunk) {
        const std::int64_t rows = std::min<std::int64_t>(options.rows_per_chunk, n - start);
        index.search(rows, &equity[static_cast<std::size_t>(start)], 1,
                     dist.data(), labels.data());
        for (std::int64_t r = 0; r < rows; ++r)
            map[static_cast<std::size_t>(start + r)] =
                static_cast<std::uint16_t>(labels[static_cast<std::size_t>(r)]);
    }

    write_bin(dir + "/river_map.bin", map.data(), n);
    std::printf("[river] wrote %s/river_map.bin\n", dir.c_str());
    return map;
}

// ---------------------------------------------------------------------------
// Transition streets (turn, flop, and clustered preflop): cluster the next-street histograms.
//   dim = number of buckets on the next street (the histogram dimension)
//   k   = number of buckets to produce on this street
// ---------------------------------------------------------------------------
std::vector<std::uint16_t> build_transition(
    const std::string& dir, std::initializer_list<std::uint8_t> this_cpr,
    int this_cards, std::initializer_list<std::uint8_t> next_cpr,
    const std::vector<std::uint16_t>& prev_map, int dim, int k,
    const std::string& out_file, const bucketing::Options& options) {
    hand_indexer_t self;
    hand_indexer_t next;
    init_indexer(&self, this_cpr);
    init_indexer(&next, next_cpr);
    const std::int64_t n = static_cast<std::int64_t>(hand_indexer_size(&self, this_cards == 2 ? 0 : 1));

    // Stage 1: train centroids on a sample of histograms.
    const std::vector<std::int64_t> ids =
        sample_ids(n, sample_size_for(options.points_per_centroid, k), options.seed);
    std::printf("[%s] %lld hands, training on %zu\n", out_file.c_str(),
                static_cast<long long>(n), ids.size());
    std::vector<float> samples(ids.size() * static_cast<std::size_t>(dim));
#pragma omp parallel for schedule(dynamic, 256)
    for (std::int64_t r = 0; r < static_cast<std::int64_t>(ids.size()); ++r)
        fill_histogram(self, next, prev_map, this_cards, dim,
                       ids[static_cast<std::size_t>(r)],
                       &samples[static_cast<std::size_t>(r) * dim]);
    const std::vector<float> centroids = train_centroids(
        dim, k, static_cast<std::int64_t>(ids.size()), samples.data(), options);

    // Stage 2: stream over every hand, recompute its histogram, assign a bucket.
    faiss::IndexFlatL2 index(dim);
    index.add(k, centroids.data());
    std::vector<std::uint16_t> map(static_cast<std::size_t>(n));
    std::vector<float> buf(static_cast<std::size_t>(options.rows_per_chunk) * dim);
    std::vector<faiss::idx_t> labels(static_cast<std::size_t>(options.rows_per_chunk));
    std::vector<float> dist(static_cast<std::size_t>(options.rows_per_chunk));
    for (std::int64_t start = 0; start < n; start += options.rows_per_chunk) {
        const std::int64_t rows = std::min<std::int64_t>(options.rows_per_chunk, n - start);
#pragma omp parallel for schedule(dynamic, 256)
        for (std::int64_t r = 0; r < rows; ++r)
            fill_histogram(self, next, prev_map, this_cards, dim, start + r,
                           &buf[static_cast<std::size_t>(r) * dim]);
        index.search(rows, buf.data(), 1, dist.data(), labels.data());
        for (std::int64_t r = 0; r < rows; ++r)
            map[static_cast<std::size_t>(start + r)] =
                static_cast<std::uint16_t>(labels[static_cast<std::size_t>(r)]);
        std::printf("\r[%s] assigned %lld / %lld", out_file.c_str(),
                    static_cast<long long>(start + rows),
                    static_cast<long long>(n));
        std::fflush(stdout);
    }

    write_bin(dir + "/" + out_file, map.data(), n);
    std::printf("\n[%s] wrote %s/%s\n", out_file.c_str(), dir.c_str(),
                out_file.c_str());
    return map;
}

// ---------------------------------------------------------------------------
// Default preflop: no abstraction. There are only 169 canonical 2-card hands, so each
// maps to its own bucket (an identity map). No clustering, so points_per_centroid
// does not apply here.
// ---------------------------------------------------------------------------
void build_preflop(const std::string& dir) {
    hand_indexer_t preflop;
    init_indexer(&preflop, {2});
    // A 1-round indexer reports its size at round 0 (= 169 canonical hands).
    const std::int64_t n = static_cast<std::int64_t>(hand_indexer_size(&preflop, 0));

    std::vector<std::uint16_t> map(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i)
        map[static_cast<std::size_t>(i)] = static_cast<std::uint16_t>(i);

    write_bin(dir + "/preflop_map.bin", map.data(), n);
    std::printf("[preflop] wrote %s/preflop_map.bin (%lld identity buckets)\n",
                dir.c_str(), static_cast<long long>(n));
}

}  // namespace

namespace bucketing {

void validate_options(const Options& options) {
    for (int bins : {options.river_bins, options.turn_bins, options.flop_bins}) {
        if (bins < 1 || bins > 65536)
            throw std::invalid_argument("postflop bins must be in 1..65536");
    }
    if (options.preflop_bins < 1 || options.preflop_bins > 169)
        throw std::invalid_argument("preflop bins must be in 1..169");
    if (options.points_per_centroid < 0 || options.river_points_per_centroid < 0)
        throw std::invalid_argument("points per centroid must be nonnegative");
    if (options.kmeans_iters < 1 || options.rows_per_chunk < 1)
        throw std::invalid_argument("iterations and rows per chunk must be positive");
    if (options.seed < 0) throw std::invalid_argument("seed must be nonnegative");
}

void build_offline(const Options& options) {
    validate_options(options);
    const std::string dir = kAbstractionDir;
    std::filesystem::create_directories(dir);
    const auto river = build_river(dir, options);
    // Indexer lists separate hole cards from the public board; board counts
    // are cumulative, not cards dealt on that street. The following integer
    // is the total known cards (hole + board), used to exclude blocked cards.
    // Turn: {2,4} = 2 hole + 4 board cards, 6 known. Deal one of the 46
    // remaining cards to reach river {2,5}; histogram bins are river buckets.
    // histogram bins are river buckets.
    const auto turn = build_transition(dir, {2, 4}, 6, {2, 5}, river,
        options.river_bins, options.turn_bins, "turn_map.bin", options);
    // Flop: {2,3} = 2 hole + 3 board cards, 5 known. Deal one of the 47
    // remaining cards to reach turn {2,4}; histogram bins are turn buckets.
    // histogram bins are turn buckets.
    const auto flop = build_transition(dir, {2, 3}, 5, {2, 4}, turn,
        options.turn_bins, options.flop_bins, "flop_map.bin", options);
    if (options.preflop_bins == 169) {
        // Preserve all 169 suit-equivalent starting-hand classes directly.
        build_preflop(dir);
    } else {
        // Preflop: {2} = hole cards only, 2 known. Deal an unordered 3-card
        // flop from the 50 remaining cards: C(50,3) = 19,600 possible boards.
        // The destination is {2,3}; histogram bins are flop buckets.
        build_transition(dir, {2}, 2, {2, 3}, flop,
            options.flop_bins, options.preflop_bins, "preflop_map.bin", options);
    }
}

}  // namespace bucketing
