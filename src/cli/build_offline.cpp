#include "bucketing/build_offline.h"

#include <charconv>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
struct Flag {
    const char* name;
    int bucketing::Options::*member;
    const char* description;
};
const Flag flags[] = {
    {"--river-bins", &bucketing::Options::river_bins, "1..65536; river equity clusters"},
    {"--turn-bins", &bucketing::Options::turn_bins, "1..65536; river-bin histogram clusters"},
    {"--flop-bins", &bucketing::Options::flop_bins, "1..65536; turn-bin histogram clusters"},
    {"--preflop-bins", &bucketing::Options::preflop_bins, "1..169; 169 = identity, fewer = flop-bin histogram clusters"},
    {"--points-per-centroid", &bucketing::Options::points_per_centroid, "turn/flop/clustered preflop sample; 0 = all hands"},
    {"--river-points-per-centroid", &bucketing::Options::river_points_per_centroid, "river sample; 0 = all hands"},
    {"--kmeans-iters", &bucketing::Options::kmeans_iters, "positive K-means iteration count"},
    {"--seed", &bucketing::Options::seed, "0..INT_MAX; sampling and K-means seed"},
    {"--rows-per-chunk", &bucketing::Options::rows_per_chunk, "positive assignment batch size; trades RAM for throughput"},
};
}  // namespace

int main(int argc, char** argv) {
    bucketing::Options options;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string_view flag = argv[i];
            if (flag == "--help" || flag == "-h") {
                const bucketing::Options defaults;
                std::printf("Usage: build_offline [options]\nBuilds all four streets.\n");
                for (const auto& entry : flags)
                    std::printf("  %-28s N  default %d; %s\n", entry.name,
                                defaults.*(entry.member), entry.description);
                std::printf("  --help, -h                     show this help\nOutput: %s\n",
                            bucketing::kAbstractionDir);
                return 0;
            }
            int* target = nullptr;
            for (const auto& entry : flags)
                if (flag == entry.name) target = &(options.*(entry.member));
            if (!target) throw std::invalid_argument("unknown option: " + std::string(flag));
            if (++i == argc) throw std::invalid_argument("missing value for " + std::string(flag));
            const std::string_view value = argv[i];
            const auto result = std::from_chars(value.data(), value.data() + value.size(), *target);
            if (result.ec != std::errc{} || result.ptr != value.data() + value.size())
                throw std::invalid_argument("invalid integer for " + std::string(flag));
        }
        bucketing::validate_options(options);
    } catch (const std::invalid_argument& e) {
        std::fprintf(stderr, "error: %s (see --help)\n", e.what());
        return 2;
    }
    try {
        bucketing::build_offline(options);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
