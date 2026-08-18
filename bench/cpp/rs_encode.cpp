// Reed-Solomon erasure encoder — exact algorithm port from src/reed_solomon.rs.
// GF(256), primitive polynomial 0x11D.
// Cauchy generator matrix: coeffs[i*d+j] = 1 / (i ^ (p + j)), where the two
// index sets are disjoint so no denominator is zero. This mirrors
// MatrixKind::Cauchy on the Rust side; see that type's docs for why every
// square submatrix of it is invertible, which the earlier alpha^(i*j)
// construction did not guarantee.
// encode_into: parity[i][k] ^= mul(coeffs[i][j], data[j][k])
//
// Build: g++ -O3 -march=native -std=c++17 -o rs_encode bench/cpp/rs_encode.cpp
// Usage: ./rs_encode [bench/results dir]
// Output: bench/results/cpp.json, bench/results/cpp.checksum

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <chrono>
#include <cstdio>
#include <cassert>
#include <filesystem>

static constexpr int DATA_SHARDS   = 10;
static constexpr int PARITY_SHARDS = 4;
static constexpr uint32_t ITERS    = 500;
static constexpr uint32_t WARMUP   = 50;

// ---- GF(256) tables --------------------------------------------------------

static uint8_t gf_exp[512];
static uint8_t gf_log[256];

static void build_tables() {
    uint8_t x = 1;
    for (int i = 0; i < 255; ++i) {
        gf_exp[i] = x;
        gf_log[x] = (uint8_t)i;
        bool hi = (x & 0x80) != 0;
        x = (uint8_t)(x << 1);
        if (hi) x ^= 0x1D;
    }
    for (int i = 255; i < 512; ++i)
        gf_exp[i] = gf_exp[i - 255];
}

static inline uint8_t gf_mul(uint8_t a, uint8_t b) {
    if (a == 0 || b == 0) return 0;
    return gf_exp[(int)gf_log[a] + (int)gf_log[b]];
}

static inline uint8_t pow_alpha(int power) {
    return gf_exp[((power % 255) + 255) % 255];
}

// Multiplicative inverse in GF(256): alpha^(255 - log a).
static inline uint8_t gf_inv(uint8_t a) {
    return gf_exp[255 - (int)gf_log[a]];
}

// ---- RS encoder ------------------------------------------------------------

struct ReedSolomon {
    int d, p;
    uint8_t coeffs[PARITY_SHARDS * DATA_SHARDS];

    ReedSolomon(int data_shards, int parity_shards)
        : d(data_shards), p(parity_shards) {
        for (int i = 0; i < p; ++i)
            for (int j = 0; j < d; ++j)
                coeffs[i * d + j] = gf_inv((uint8_t)(i ^ (p + j)));
    }

    void encode_into(const uint8_t* const* data, size_t shard_len,
                     uint8_t* parity_out) const {
        // zero parity
        memset(parity_out, 0, (size_t)p * shard_len);

        for (int j = 0; j < d; ++j) {
            const uint8_t* dj = data[j];
            for (int i = 0; i < p; ++i) {
                uint8_t coef = coeffs[i * d + j];
                if (coef == 0) continue;
                uint8_t* row = parity_out + (size_t)i * shard_len;
                if (coef == 1) {
                    for (size_t k = 0; k < shard_len; ++k)
                        row[k] ^= dj[k];
                } else {
                    for (size_t k = 0; k < shard_len; ++k)
                        row[k] ^= gf_mul(coef, dj[k]);
                }
            }
        }
    }
};

// ---- timing ----------------------------------------------------------------

using Clock = std::chrono::steady_clock;

template<typename F>
double time_ns(F&& fn) {
    // warm-up
    for (uint32_t i = 0; i < WARMUP; ++i) fn();
    auto t0 = Clock::now();
    for (uint32_t i = 0; i < ITERS; ++i) fn();
    auto t1 = Clock::now();
    double total_ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    return total_ns / ITERS;
}

static double mib_per_s(size_t payload_bytes, double ns_per_iter) {
    return (payload_bytes / ns_per_iter) * 1e9 / (1024.0 * 1024.0);
}

// ---- checksum helper -------------------------------------------------------

static std::string to_hex(const uint8_t* data, size_t len) {
    static const char hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(len * 2);
    for (size_t i = 0; i < len; ++i) {
        out.push_back(hex[data[i] >> 4]);
        out.push_back(hex[data[i] & 0xf]);
    }
    return out;
}

// ---- main ------------------------------------------------------------------

int main(int argc, char** argv) {
    build_tables();

    const char* out_dir = (argc > 1) ? argv[1] : "bench/results";
    std::filesystem::create_directories(out_dir);

    static constexpr size_t SHARD_LENS[] = {256, 1024, 4096, 16384};
    static constexpr int N_LENS = 4;
    static constexpr size_t CHECKSUM_SHARD_LEN = 1024;

    ReedSolomon rs(DATA_SHARDS, PARITY_SHARDS);

    std::string json_records;
    std::string checksum_hex;

    for (int li = 0; li < N_LENS; ++li) {
        size_t shard_len = SHARD_LENS[li];
        size_t payload_bytes = (size_t)DATA_SHARDS * shard_len;

        // Allocate data shards: shard j filled with byte j.
        std::vector<std::vector<uint8_t>> owned(DATA_SHARDS);
        std::vector<const uint8_t*> ptrs(DATA_SHARDS);
        for (int j = 0; j < DATA_SHARDS; ++j) {
            owned[j].assign(shard_len, (uint8_t)j);
            ptrs[j] = owned[j].data();
        }
        std::vector<uint8_t> parity((size_t)PARITY_SHARDS * shard_len);

        double ns = time_ns([&]() {
            rs.encode_into(ptrs.data(), shard_len, parity.data());
        });
        double mib = mib_per_s(payload_bytes, ns);

        char buf[512];
        snprintf(buf, sizeof(buf),
            "  {\"lang\":\"cpp\",\"impl\":\"encode_into\","
            "\"shard_len\":%zu,\"data_shards\":%d,\"parity_shards\":%d,"
            "\"payload_bytes\":%zu,\"ns_per_iter\":%.1f,\"mib_per_s\":%.1f}",
            shard_len, DATA_SHARDS, PARITY_SHARDS, payload_bytes, ns, mib);

        if (!json_records.empty()) json_records += ",\n";
        json_records += buf;

        // Capture checksum at shard_len == CHECKSUM_SHARD_LEN using seed data.
        if (shard_len == CHECKSUM_SHARD_LEN) {
            // seed: shard j has all bytes = (j*3+1) & 0xFF (matches Rust exporter)
            std::vector<std::vector<uint8_t>> seed(DATA_SHARDS);
            std::vector<const uint8_t*> sptrs(DATA_SHARDS);
            for (int j = 0; j < DATA_SHARDS; ++j) {
                uint8_t fill = (uint8_t)(((uint8_t)j * 3u) + 1u);
                seed[j].assign(CHECKSUM_SHARD_LEN, fill);
                sptrs[j] = seed[j].data();
            }
            std::vector<uint8_t> cparity((size_t)PARITY_SHARDS * CHECKSUM_SHARD_LEN);
            rs.encode_into(sptrs.data(), CHECKSUM_SHARD_LEN, cparity.data());
            checksum_hex = to_hex(cparity.data(), cparity.size());
        }
    }

    // Write JSON
    std::string json_path = std::string(out_dir) + "/cpp.json";
    FILE* jf = fopen(json_path.c_str(), "w");
    assert(jf);
    fprintf(jf, "[\n%s\n]\n", json_records.c_str());
    fclose(jf);
    printf("Wrote %s\n", json_path.c_str());

    // Write checksum
    std::string cs_path = std::string(out_dir) + "/cpp.checksum";
    FILE* cf = fopen(cs_path.c_str(), "w");
    assert(cf);
    fprintf(cf, "%s", checksum_hex.c_str());
    fclose(cf);
    printf("Wrote %s\n", cs_path.c_str());

    // Print summary
    printf("\nC++ RS encode throughput (MiB/s):\n");
    printf("%-32s %8s %8s %8s %8s\n", "impl", "256B", "1KiB", "4KiB", "16KiB");
    // Parse from records is awkward; re-run quickly.
    double mibs[N_LENS];
    for (int li = 0; li < N_LENS; ++li) {
        size_t shard_len = SHARD_LENS[li];
        size_t payload_bytes = (size_t)DATA_SHARDS * shard_len;
        std::vector<std::vector<uint8_t>> owned(DATA_SHARDS);
        std::vector<const uint8_t*> ptrs(DATA_SHARDS);
        for (int j = 0; j < DATA_SHARDS; ++j) {
            owned[j].assign(shard_len, (uint8_t)j);
            ptrs[j] = owned[j].data();
        }
        std::vector<uint8_t> parity((size_t)PARITY_SHARDS * shard_len);
        double ns = time_ns([&]() {
            rs.encode_into(ptrs.data(), shard_len, parity.data());
        });
        mibs[li] = mib_per_s(payload_bytes, ns);
    }
    printf("%-32s %8.1f %8.1f %8.1f %8.1f\n",
           "encode_into", mibs[0], mibs[1], mibs[2], mibs[3]);

    return 0;
}
