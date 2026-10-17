// Offline abstraction: river equities, turn/flop transition histograms, optional preflop clustering.
#include "bucketing/build_bins.h"

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

// Combine the first n card IDs in cards into an OMPEval hand for evaluation
// or for combining hole cards with a board. The caller supplies distinct,
// valid card IDs. hand-isomorphism and OMPEval encode cards as 4*rank + suit over 0..51,
// so a hand-isomorphism card index is directly a valid OMPEval card.
omp::Hand make_hand(const std::uint8_t* cards, int n) {
    omp::Hand h = omp::Hand::empty();
    for (int i = 0; i < n; ++i) h += omp::Hand(cards[i]);
    return h;
}

// Initialize idx to index hands up to suit equivalence. cpr specifies the
// number of cards in each indexing group: {2, 4}, for example, separates two
// hole cards from an unordered four-card turn board. These groups do not
// preserve the order in which board cards were dealt. Throw if initialization
// fails; after success, the caller must release idx with hand_indexer_free.
void init_indexer(hand_indexer_t* idx, std::initializer_list<std::uint8_t> cpr) {
    std::vector<std::uint8_t> rounds(cpr);
    if (!hand_indexer_init(static_cast<uint_fast32_t>(rounds.size()),
                           rounds.data(), idx)) {
        throw std::runtime_error("hand_indexer_init failed");
    }
}

// Write count contiguous elements of type T to path, creating or overwriting
// the file. The output is raw native binary data (count * sizeof(T) bytes),
// without a header or byte-order conversion. Throw if opening fails or fwrite
// reports fewer elements than requested; close the file before a short-write
// exception. The final fclose return value is not checked.
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
                    const std::vector<std::uint16_t>& next_street_map, int known_card_count,
                    int dim, std::int64_t id, float* out) {
    std::fill(out, out + dim, 0.0f);
    std::uint8_t cards[7];  // hole cards and board (<= 7)
    hand_unindex(&self, known_card_count == 2 ? 0 : 1, static_cast<hand_index_t>(id), cards);
    bool used[52] = {false};
    for (int i = 0; i < known_card_count; ++i) used[cards[i]] = true;

    // Preflop deals an unordered three-card flop: C(50,3) = 19600 equally
    // likely boards. Count actual deals, not canonical boards (their suit
    // multiplicities differ), then normalize to a probability distribution.
    if (known_card_count == 2) {
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
                    out[next_street_map[static_cast<std::size_t>(next_id)]] += 1.0f;
                }
            }
        }
        for (int bin = 0; bin < dim; ++bin) out[bin] /= 19600.0f;
        return;
    }
    const float weight = 1.0f / static_cast<float>(52 - known_card_count);
    for (int c = 0; c < 52; ++c) {
        if (used[c]) continue;
        cards[known_card_count] = static_cast<std::uint8_t>(c);
        const hand_index_t next_id = hand_index_last(&next, cards);
        out[next_street_map[static_cast<std::size_t>(next_id)]] += weight;
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
    // Faiss labels are int64_t; its idx_t alias moved between versions.
    std::vector<std::int64_t> labels(static_cast<std::size_t>(options.assignment_batch_size));
    std::vector<float> dist(static_cast<std::size_t>(options.assignment_batch_size));
    for (std::int64_t start = 0; start < n; start += options.assignment_batch_size) {
        const std::int64_t rows = std::min<std::int64_t>(options.assignment_batch_size, n - start);
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
// Train K-means on sampled histograms, then assign every current-street hand
// to its nearest centroid. Write and return the resulting hand-ID-to-bin map.
// The card-count lists define which combinations to enumerate. For turn,
// {2,4} enumerates two-hole-card/four-board-card combinations, grouping suit
// equivalents together. For each such hand, fill_histogram tries every legal
// river card and uses the {2,5} indexer to look up the resulting river bin.
//
// Parameters:
//   dir (const std::string&)
//     Output directory; write_bin writes the map to dir + "/" + out_file.
//   this_cpr (std::initializer_list<std::uint8_t>)
//     Counts defining the current street's combinations: {2} preflop,
//     {2,3} flop, {2,4} turn. Each list separates hole cards from the unordered
//     board; "cpr" is the indexer API's abbreviation for cards per round.
//     init_indexer builds self for these combination sizes. hand_indexer_size
//     gives the number of suit-distinct combinations, so training can sample
//     their IDs and assignment can iterate over all of them. For each ID,
//     fill_histogram calls hand_unindex to recover a representative set of
//     hole/board cards; these determine which future cards can legally be dealt.
//   next_cpr (std::initializer_list<std::uint8_t>)
//     Counts defining the next street's combinations: {2,3} flop, {2,4} turn,
//     {2,5} river. For example, {2,5} indexes two-hole-card/five-board-card hands.
//     init_indexer builds next for these combination sizes. After adding each
//     legal future deal to the current hand, fill_histogram uses hand_index_last
//     to convert the resulting cards into a suit-equivalent hand ID. This ID
//     locates the hand's bin in next_street_map, so that outcome contributes
//     to the correct histogram entry. next_cpr must match the map's combination
//     sizes so the generated IDs address the correct map entries.
//   next_street_map (const std::vector<std::uint16_t>&)
//     Next street's bin map in poker order: river bins when building turn, turn
//     bins when building flop, or flop bins when building preflop. Needed to
//     look up the bin reached by each legal future deal; these bin counts form
//     the current hand's histogram. The indexer supplies hand IDs, while this
//     map supplies their learned bin IDs. Bin IDs must be in [0, dim).
//   dim (int)
//     Number of bins represented in next_street_map, hence the number of
//     entries in each histogram (D). The map and dim refer to the SAME street:
//     building flop passes turn and options.turn_bins respectively. "Next"
//     always follows poker order (preflop -> flop -> turn -> river); maps are
//     built in reverse order so the next street's map is already available.
//     k is the number of bins to create for the current street.
//   k (int)
//     Current-street bin count (K). sample_size_for uses it to calculate the
//     training sample size; train_centroids learns k centroids, which are
//     added to the Faiss assignment index.
//   out_file (const std::string&)
//     Output filename within dir, e.g. "turn_map.bin"; also used in logs.
//   options (const bucketing::Options&)
//     Supplies points_per_centroid to sample_size_for, seed to sample_ids,
//     iteration count and seed to train_centroids, and assignment_batch_size
//     to the assignment buffers and loop. Street bin counts are supplied
//     separately through dim and k; river_points_per_centroid is unused here.
// Returns (std::vector<std::uint16_t>): current-street hand ID -> learned bin ID.
// ---------------------------------------------------------------------------
std::vector<std::uint16_t> build_transition(
    const std::string& dir, std::initializer_list<std::uint8_t> this_cpr,
    std::initializer_list<std::uint8_t> next_cpr,
    const std::vector<std::uint16_t>& next_street_map, int dim, int k,
    const std::string& out_file, const bucketing::Options& options) {
    hand_indexer_t self;
    hand_indexer_t next;
    init_indexer(&self, this_cpr);
    init_indexer(&next, next_cpr);
    // Derive the known-card count from the combination sizes, e.g. {2,4} -> 6.
    // fill_histogram uses it to exclude known cards when enumerating legal deals.
    const int known_card_count = std::accumulate(this_cpr.begin(), this_cpr.end(), 0);
    const std::int64_t n = static_cast<std::int64_t>(hand_indexer_size(&self, known_card_count == 2 ? 0 : 1));

    // Stage 1: train centroids on a sample of histograms.
    const std::vector<std::int64_t> ids =
        sample_ids(n, sample_size_for(options.points_per_centroid, k), options.seed);
    std::printf("[%s] %lld hands, training on %zu\n", out_file.c_str(),
                static_cast<long long>(n), ids.size());
    std::vector<float> samples(ids.size() * static_cast<std::size_t>(dim));
#pragma omp parallel for schedule(dynamic, 256)
    for (std::int64_t r = 0; r < static_cast<std::int64_t>(ids.size()); ++r)
        fill_histogram(self, next, next_street_map, known_card_count, dim,
                       ids[static_cast<std::size_t>(r)],
                       &samples[static_cast<std::size_t>(r) * dim]);
    const std::vector<float> centroids = train_centroids(
        dim, k, static_cast<std::int64_t>(ids.size()), samples.data(), options);

    // Stage 2: stream over every hand, recompute its histogram, assign a bucket.
    faiss::IndexFlatL2 index(dim);
    index.add(k, centroids.data());
    std::vector<std::uint16_t> map(static_cast<std::size_t>(n));
    std::vector<float> buf(static_cast<std::size_t>(options.assignment_batch_size) * dim);
    std::vector<std::int64_t> labels(static_cast<std::size_t>(options.assignment_batch_size));
    std::vector<float> dist(static_cast<std::size_t>(options.assignment_batch_size));
    for (std::int64_t start = 0; start < n; start += options.assignment_batch_size) {
        const std::int64_t rows = std::min<std::int64_t>(options.assignment_batch_size, n - start);
#pragma omp parallel for schedule(dynamic, 256)
        for (std::int64_t r = 0; r < rows; ++r)
            fill_histogram(self, next, next_street_map, known_card_count, dim, start + r,
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
    if (options.kmeans_iters < 1 || options.assignment_batch_size < 1)
        throw std::invalid_argument("iterations and assignment batch size must be positive");
    if (options.seed < 0) throw std::invalid_argument("seed must be nonnegative");
}

void build_bins(const Options& options) {
    validate_options(options);
    const std::string dir = kAbstractionDir;
    std::filesystem::create_directories(dir);
    const auto river = build_river(dir, options);
    // Turn: enumerate 2-hole/4-board combinations. Each histogram counts the
    // river bins reached by the 46 legal river cards, using the {2,5} indexer.
    const auto turn = build_transition(dir, {2, 4}, {2, 5}, river,
        options.river_bins, options.turn_bins, "turn_map.bin", options);
    // Flop: enumerate 2-hole/3-board combinations. Each histogram counts the
    // turn bins reached by the 47 legal turn cards, using the {2,4} indexer.
    const auto flop = build_transition(dir, {2, 3}, {2, 4}, turn,
        options.turn_bins, options.flop_bins, "flop_map.bin", options);
    if (options.preflop_bins == 169) {
        // Preserve all 169 suit-equivalent starting-hand classes directly.
        build_preflop(dir);
    } else {
        // Preflop: {2} = hole cards only, 2 known. Deal an unordered 3-card
        // flop from the 50 remaining cards: C(50,3) = 19,600 possible boards.
        // The destination is {2,3}; histogram bins are flop buckets.
        build_transition(dir, {2}, {2, 3}, flop,
            options.flop_bins, options.preflop_bins, "preflop_map.bin", options);
    }
}

}  // namespace bucketing
