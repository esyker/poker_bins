FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ca-certificates \
        libfaiss-dev libopenblas-dev liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY CMakeLists.txt ./
COPY src ./src
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --parallel --target build_offline

ENTRYPOINT ["/app/build/build_offline"]
