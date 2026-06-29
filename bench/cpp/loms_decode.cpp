// LOMS (Layered Offset Min-Sum) LDPC decoder benchmark — BG1 Z=384
// Exact scalar algorithm port from src/qc_ldpc.rs.
//
// 5G NR QC-LDPC BG1: 46 rows, 68 cols, 316 non-null entries.
// Lifting size Z=384 (iLS=1 in 3GPP Table 5.3.2-1, set {3,6,12,24,48,96,192,384}).
// N = 68×384 = 26112 variable nodes, E = 316 block-edges.
//
// Build:  g++ -O3 -march=native -std=c++17 -o loms_decode loms_decode.cpp
// Usage:  ./loms_decode
// Output: bench/results/ldpc_cpp.json

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cassert>
#include <vector>
#include <array>
#include <algorithm>
#include <chrono>
#include <string>
#include <filesystem>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static constexpr int    BG1_ROWS     = 46;
static constexpr int    BG1_COLS     = 68;
static constexpr int    Z            = 384;
static constexpr int    N_VAR        = BG1_COLS * Z;          // 26112
static constexpr int    TOTAL_EDGES  = 316;
static constexpr int    DECODE_ITERS = 10;
static constexpr int    BENCH_REPS   = 200;
static constexpr float  BETA         = 0.25f;
static constexpr int    ILS          = 1;   // iLS for Z=384

// ---------------------------------------------------------------------------
// Minimal JSON parser — reads data/bg_tables.json
//
// We only need to extract the BG1 entries array: [{r,c,v:[...]}, ...].
// No external deps; hand-rolled integer scanner is sufficient.
// ---------------------------------------------------------------------------

struct BgEntry {
    int r, c;
    int v[8];
};

static bool skip_ws(const char*& p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') ++p;
    return *p != '\0';
}

static bool expect(const char*& p, char ch) {
    skip_ws(p);
    if (*p != ch) return false;
    ++p;
    return true;
}

// Scan a decimal integer (possibly negative).
static bool scan_int(const char*& p, int& out) {
    skip_ws(p);
    bool neg = false;
    if (*p == '-') { neg = true; ++p; }
    if (*p < '0' || *p > '9') return false;
    long long v = 0;
    while (*p >= '0' && *p <= '9') v = v * 10 + (*p++ - '0');
    out = (int)(neg ? -v : v);
    return true;
}

// Advance p past the next occurrence of `needle` in the buffer, or return false.
static bool find_str(const char*& p, const char* needle) {
    const char* found = strstr(p, needle);
    if (!found) return false;
    p = found + strlen(needle);
    return true;
}

// Parse the "bg1" entries array from the JSON blob.
static std::vector<BgEntry> parse_bg1_entries(const char* buf) {
    std::vector<BgEntry> result;
    result.reserve(320);

    const char* p = buf;
    // Navigate to "bg1"
    if (!find_str(p, "\"bg1\"")) {
        fprintf(stderr, "ERROR: 'bg1' key not found in JSON\n");
        return result;
    }
    // Navigate to "entries"
    if (!find_str(p, "\"entries\"")) {
        fprintf(stderr, "ERROR: 'entries' key not found under bg1\n");
        return result;
    }
    // Skip ':' separator between key and value
    if (!expect(p, ':')) {
        fprintf(stderr, "ERROR: expected ':' after 'entries' key\n");
        return result;
    }
    // Expect opening [
    if (!expect(p, '[')) {
        fprintf(stderr, "ERROR: expected '[' after entries:\n");
        return result;
    }

    // Parse each object {r:int, c:int, v:[...]}
    while (true) {
        skip_ws(p);
        if (*p == ']') break;
        if (*p == ',') { ++p; continue; }
        if (*p != '{') { ++p; continue; }
        ++p; // consume '{'

        BgEntry e{};
        bool got_r = false, got_c = false, got_v = false;

        // Parse key-value pairs until '}'
        while (true) {
            skip_ws(p);
            if (*p == '}') { ++p; break; }
            if (*p == ',') { ++p; continue; }

            // Expect a quoted key
            if (*p != '"') { ++p; continue; }
            ++p; // opening quote
            char key = *p;
            // advance past key name and closing quote
            while (*p && *p != '"') ++p;
            if (*p == '"') ++p;

            if (!expect(p, ':')) continue;

            if (key == 'r') {
                scan_int(p, e.r);
                got_r = true;
            } else if (key == 'c') {
                scan_int(p, e.c);
                got_c = true;
            } else if (key == 'v') {
                // Parse integer array [i0, i1, ... i7]
                if (!expect(p, '[')) continue;
                int idx = 0;
                while (idx < 8) {
                    skip_ws(p);
                    if (*p == ']') break;
                    if (*p == ',') { ++p; continue; }
                    int val = 0;
                    if (scan_int(p, val)) e.v[idx++] = val;
                    else ++p;
                }
                if (*p == ']') ++p;
                got_v = true;
            } else {
                // Unknown key — skip its value (either number or string)
                skip_ws(p);
                if (*p == '"') {
                    ++p;
                    while (*p && *p != '"') ++p;
                    if (*p) ++p;
                } else {
                    while (*p && *p != ',' && *p != '}') ++p;
                }
            }
        }

        if (got_r && got_c && got_v) {
            result.push_back(e);
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Decoder layout — mirrors QcLdpcParams::new in Rust
// ---------------------------------------------------------------------------

struct DecoderLayout {
    int layer_offsets[BG1_ROWS + 1]; // prefix sums of row degrees
    int submatrix_cols[TOTAL_EDGES];
    int submatrix_shifts[TOTAL_EDGES];
    int max_layer_degree;
};

static DecoderLayout build_layout(const std::vector<BgEntry>& entries) {
    DecoderLayout L{};

    // Count row degrees
    int row_degrees[BG1_ROWS] = {};
    for (const auto& e : entries) {
        assert(e.r >= 0 && e.r < BG1_ROWS);
        row_degrees[e.r]++;
    }

    // Prefix sums → layer_offsets
    L.layer_offsets[0] = 0;
    for (int r = 0; r < BG1_ROWS; ++r) {
        L.layer_offsets[r + 1] = L.layer_offsets[r] + row_degrees[r];
    }
    assert(L.layer_offsets[BG1_ROWS] == TOTAL_EDGES);

    // Fill submatrix_cols / submatrix_shifts
    int row_fill[BG1_ROWS] = {};
    for (const auto& e : entries) {
        int pos = L.layer_offsets[e.r] + row_fill[e.r];
        L.submatrix_cols[pos]   = e.c;
        // iLS=1: use v[1] % Z for actual cyclic shift
        L.submatrix_shifts[pos] = e.v[ILS] % Z;
        row_fill[e.r]++;
    }

    L.max_layer_degree = 0;
    for (int r = 0; r < BG1_ROWS; ++r) {
        if (row_degrees[r] > L.max_layer_degree)
            L.max_layer_degree = row_degrees[r];
    }

    return L;
}

// ---------------------------------------------------------------------------
// Scalar LOMS kernel — exact port of process_z_position_scalar from Rust
// ---------------------------------------------------------------------------

static inline void process_z_position_scalar(
    int          row_degree,
    int          z_idx,
    float*       q_row,       // [row_degree * Z]
    float*       edge_r,      // [TOTAL_EDGES * Z]
    int          layer_begin,
    const int*   submatrix_cols,
    const int*   submatrix_shifts,
    float*       llr)         // [N_VAR]
{
    // Pass 1: find min1, min2, sign_prod
    float min1      = INFINITY;
    float min2      = INFINITY;
    int   min1_edge = -1;
    float sign_prod = 1.0f;

    for (int edge = 0; edge < row_degree; ++edge) {
        float q     = q_row[edge * Z + z_idx];
        float abs_q = fabsf(q);
        float sign  = (q < 0.0f) ? -1.0f : 1.0f;
        sign_prod  *= sign;

        if (abs_q <= min1) {
            min2      = min1;
            min1      = abs_q;
            min1_edge = edge;
        } else if (abs_q < min2) {
            min2 = abs_q;
        }
    }

    if (std::isinf(min1)) min1 = 0.0f;
    if (std::isinf(min2)) min2 = min1;

    // Pass 2: update edge_r and llr
    for (int edge = 0; edge < row_degree; ++edge) {
        float q            = q_row[edge * Z + z_idx];
        float sign         = (q < 0.0f) ? -1.0f : 1.0f;
        float min_excl     = (edge == min1_edge) ? min2 : min1;
        float check_value  = min_excl - BETA;
        if (check_value < 0.0f) check_value = 0.0f;
        float new_r = sign_prod * sign * check_value;

        int global_edge_pos = (layer_begin + edge) * Z + z_idx;
        float old_r         = edge_r[global_edge_pos];
        edge_r[global_edge_pos] = new_r;

        int col_block = submatrix_cols[layer_begin + edge];
        int shift     = submatrix_shifts[layer_begin + edge];
        int var_idx   = col_block * Z + ((z_idx + shift) % Z);
        llr[var_idx] += new_r - old_r;
    }
}

// ---------------------------------------------------------------------------
// Full decode — 10 layered iterations
// ---------------------------------------------------------------------------

static void decode_loms(
    float*               llr,       // [N_VAR], modified in-place
    float*               edge_r,    // [TOTAL_EDGES * Z], zeroed before decode
    float*               q_row,     // scratch [max_layer_degree * Z]
    const DecoderLayout& L)
{
    // Zero edge_r once before iterations
    memset(edge_r, 0, sizeof(float) * TOTAL_EDGES * Z);

    for (int iter = 0; iter < DECODE_ITERS; ++iter) {
        for (int layer = 0; layer < BG1_ROWS; ++layer) {
            int layer_begin = L.layer_offsets[layer];
            int layer_end   = L.layer_offsets[layer + 1];
            int row_degree  = layer_end - layer_begin;

            // Build v→c messages for this layer into q_row
            for (int edge = 0; edge < row_degree; ++edge) {
                int col_block = L.submatrix_cols[layer_begin + edge];
                int shift     = L.submatrix_shifts[layer_begin + edge];
                int base_edge = (layer_begin + edge) * Z;
                int var_base  = col_block * Z;

                for (int z = 0; z < Z; ++z) {
                    int var_idx = var_base + ((z + shift) % Z);
                    q_row[edge * Z + z] = llr[var_idx] - edge_r[base_edge + z];
                }
            }

            // Process each z-position with the scalar min-sum kernel
            for (int z_idx = 0; z_idx < Z; ++z_idx) {
                process_z_position_scalar(
                    row_degree, z_idx, q_row, edge_r,
                    layer_begin, L.submatrix_cols, L.submatrix_shifts, llr);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// LLR initialiser — alternating +0.5 / -0.5 (matches Rust bench)
// ---------------------------------------------------------------------------

static void init_llr(float* llr) {
    for (int i = 0; i < N_VAR; ++i) {
        llr[i] = (i & 1) ? -0.5f : 0.5f;
    }
}

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

using Clock = std::chrono::steady_clock;

// Returns median ns/iter over BENCH_REPS timed calls.
// Each call resets LLR + edge_r to ensure we measure a fresh decode.
static double bench_median_ns(
    float*               llr,
    float*               edge_r,
    float*               q_row,
    const DecoderLayout& L)
{
    std::vector<double> samples(BENCH_REPS);

    for (int rep = 0; rep < BENCH_REPS; ++rep) {
        // Reset LLR before each timed call
        init_llr(llr);

        auto t0 = Clock::now();
        decode_loms(llr, edge_r, q_row, L);
        auto t1 = Clock::now();

        samples[rep] = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    }

    std::sort(samples.begin(), samples.end());
    return samples[BENCH_REPS / 2];
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    const char* repo_root = (argc > 1) ? argv[1] : ".";

    // Build path to data/bg_tables.json relative to repo_root
    std::string json_path = std::string(repo_root) + "/data/bg_tables.json";

    // Read the JSON file into memory
    FILE* f = fopen(json_path.c_str(), "r");
    if (!f) {
        // Try relative path directly (when run from repo root)
        json_path = "data/bg_tables.json";
        f = fopen(json_path.c_str(), "r");
    }
    if (!f) {
        fprintf(stderr, "ERROR: cannot open bg_tables.json (tried '%s' and 'data/bg_tables.json')\n",
                (std::string(repo_root) + "/data/bg_tables.json").c_str());
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    rewind(f);
    std::vector<char> buf(fsize + 1);
    size_t _nread = fread(buf.data(), 1, fsize, f);
    (void)_nread;
    fclose(f);
    buf[fsize] = '\0';

    // Parse BG1 entries
    std::vector<BgEntry> entries = parse_bg1_entries(buf.data());
    if ((int)entries.size() != TOTAL_EDGES) {
        fprintf(stderr, "ERROR: expected %d BG1 entries, got %d\n",
                TOTAL_EDGES, (int)entries.size());
        return 1;
    }

    // Build decoder layout
    DecoderLayout L = build_layout(entries);

    printf("BG1 Z=%d: %d variable nodes, %d edges, max_layer_degree=%d\n",
           Z, N_VAR, TOTAL_EDGES, L.max_layer_degree);

    // Allocate working buffers (heap, allocated once)
    std::vector<float> llr(N_VAR);
    std::vector<float> edge_r(TOTAL_EDGES * Z, 0.0f);
    std::vector<float> q_row((size_t)L.max_layer_degree * Z);

    // Benchmark
    printf("Running %d reps of %d-iteration decode...\n", BENCH_REPS, DECODE_ITERS);

    double median_ns    = bench_median_ns(llr.data(), edge_r.data(), q_row.data(), L);
    double melem_per_s  = ((double)N_VAR * DECODE_ITERS) / (median_ns * 1e-9) / 1e6;

    printf("Median ns/iter : %.1f\n", median_ns);
    printf("Melem/s        : %.2f\n", melem_per_s);

    // Write output JSON
    std::string out_dir = std::string(repo_root) + "/bench/results";
    std::filesystem::create_directories(out_dir);

    std::string out_path = out_dir + "/ldpc_cpp.json";
    FILE* jf = fopen(out_path.c_str(), "w");
    if (!jf) {
        fprintf(stderr, "ERROR: cannot write %s\n", out_path.c_str());
        return 1;
    }
    fprintf(jf,
        "[\n"
        "  {\"lang\":\"cpp\",\"impl\":\"loms_scalar\","
        "\"shard_len\":0,\"data_shards\":0,\"parity_shards\":0,"
        "\"payload_bytes\":%d,\"ns_per_iter\":%.1f,\"mib_per_s\":0,"
        "\"melem_per_s\":%.2f,"
        "\"n_variable_nodes\":%d,\"n_iters\":%d}\n"
        "]\n",
        N_VAR, median_ns, melem_per_s, N_VAR, DECODE_ITERS);
    fclose(jf);
    printf("Wrote %s\n", out_path.c_str());

    return 0;
}
