FROM ubuntu:22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ca-certificates \
        libfaiss-dev libopenblas-dev liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY CMakeLists.txt ./
COPY src ./src
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --parallel --target build_offline

# Run the build_offline binary when the container starts with the default arguments
# ENTRYPOINT ["/app/build/build_offline"]

# Run the build_offline binary when the container starts with other arguments
# ENTRYPOINT ["/app/build/build_offline", "--river-bins", "1000", "--turn-bins", "2500", "--flop-bins", "2500", "--preflop-bins", "80", "--points-per-centroid", "128", "--river-points-per-centroid", "128", "--kmeans-iters", "30", "--seed", "1234", "--rows-per-chunk", "4096"]
