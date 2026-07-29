# syndrome Cross-Language Benchmark Suite

Reproducible Reed-Solomon encode throughput comparison: Rust vs same-algorithm C++ vs Python.

## Quick start

```bash
# From repo root:
bash bench/run_all.sh
```

This single command:

1. Runs the **Rust** exporter (`cargo run --release --bin bench_export`) — writes `bench/results/rust.json` + `rust.checksum`.
2. Builds and runs the **C++** driver (`g++ -O3 -march=native -std=c++17`) — writes `bench/results/cpp.json` + `cpp.checksum`.
3. Runs the **Python** driver (`python3 bench/python/rs_encode.py`) — writes `bench/results/python.json` + `python_same_algo.checksum`.
4. Writes `bench/results/meta.json` with host/compiler info.
5. **Checksum gate**: diffs `rust.checksum` vs `cpp.checksum` vs `python_same_algo.checksum` and **fails loudly** if they differ (algorithm bug).
6. Prints a summary table.

## View the dashboard

```bash
cd bench/dashboard
python -m http.server
# open http://localhost:8000
```

Two charts are displayed:
- Grouped column: throughput (MiB/s) per implementation at each shard size.
- Line chart: throughput vs shard size (log-scale x-axis).

## Algorithm

All four implementations use the **identical** algorithm:

- GF(256), primitive polynomial `0x11D`.
- Encoding matrix: `coeffs[i*d+j] = α^((i*j) mod 255)`.
- `encode_into`: zero parity; for each data shard `j`, each parity row `i`, `parity[i][k] ^= mul(coeffs[i][j], data[j][k])`.
- Bench config: `data_shards=10, parity_shards=4`, `shard_len ∈ {256, 1024, 4096, 16384}`.

`python_reedsolo` is a separate bar — it uses generator-polynomial RS (different algorithm, different output) and is included as an ecosystem reference only.

## Prerequisites

| Tool      | Minimum |
|-----------|---------|
| Rust      | stable (cargo + rustc) |
| g++       | any C++17-capable version |
| Python    | 3.10+ |
| reedsolo  | `pip install reedsolo` (auto-installed by `run_all.sh` if missing) |

## License note

The dashboard (`bench/dashboard/`) uses [Highcharts](https://www.highcharts.com/) via CDN under its
**non-commercial** license. The attribution/credits label is intentionally kept visible. Anyone
forking this project for **commercial** use must obtain a separate Highcharts license or replace
the charting library (e.g., with Apache-licensed ECharts).

## LDPC convergence GIF

The animated GIF at the top of `README.md` is generated entirely from real
decoder output — no hand-authored error pattern or convergence curve.

```bash
# From repo root:
cargo run --release --bin ldpc_convergence_export
python3 bench/dashboard/gen_convergence_gif.py
```

1. `cargo run --release --bin ldpc_convergence_export` encodes a real 802.11
   Wi-Fi LDPC codeword (`Z=27`, rate 1/2, `N=648`, via
   [`wifi_ldpc_tables::wifi_ldpc_encoder`](../src/wifi_ldpc_tables.rs)),
   transmits it through a simulated BPSK/AWGN channel
   ([`channel_sim::AwgnChannel`](../src/channel_sim.rs)), and decodes it with
   [`QcLdpcDecoder::decode_layered_offset_min_sum_traced`](../src/qc_ldpc.rs)
   — a small library addition that runs the *exact same* layered offset
   min-sum kernel as the production decode path, but also invokes a
   per-iteration observer closure so the hard-decision bits can be snapshotted
   after every completed pass, from one continuous decode call (calling the
   untraced decoder once per iteration count would reset its extrinsic
   message buffer each time and not reproduce the real trajectory). The
   program searches a small grid of `(Eb/N0, seed)` combinations for one real
   trial whose initial corruption and iteration count make a legible
   animation, then writes the full per-iteration trace to
   `bench/results/ldpc_convergence.json`.
2. `bench/dashboard/gen_convergence_gif.py` reads that JSON and renders it to
   `bench/dashboard/exports/ldpc_convergence.gif` — a bit grid (correct vs.
   wrong, colored) beside a bit-error-count-vs-iteration line, one frame per
   real decode iteration recorded in the JSON. The only liberty taken past
   the raw numbers is holding the final (zero-error) frame for a couple of
   extra, identical repeats so the GIF loop reads clearly.

| File | Contents |
|------|----------|
| `bench/results/ldpc_convergence.json` | Per-iteration hard-decision trace from one real encode→AWGN→LOMS-decode cycle |
| `bench/dashboard/exports/ldpc_convergence.gif` | Rendered animation embedded in `README.md` |

## Output files

| File | Contents |
|------|----------|
| `bench/results/rust.json` | Rust timings, one record per (impl, shard_len) |
| `bench/results/cpp.json` | C++ timings |
| `bench/results/python.json` | Python same_algo + reedsolo timings |
| `bench/results/meta.json` | Host/compiler metadata |
| `bench/results/rust.checksum` | Hex parity bytes for correctness gate |
| `bench/results/cpp.checksum` | Same, from C++ |
| `bench/results/python_same_algo.checksum` | Same, from Python |
| `bench/results/ldpc_rust.json` | LDPC decode, Rust, AVX2 selected at runtime |
| `bench/results/ldpc_cpp.json` | LDPC decode, C++ scalar `-O3 -march=native` |
| `bench/results/ldpc_pipeline_rust.json` | SPSC pipeline frame rate — see the caveat below |

### Caveat: `ldpc_pipeline_bench` does not measure decode throughput

`src/bin/ldpc_pipeline_bench.rs` fills every frame's LLR buffer with the
constant `+5.0`, which is an error-free codeword at high confidence. The
syndrome check therefore passes on the first pass and the decoder returns after
one iteration instead of ten. Its `melem_per_s` is nevertheless computed as
`n_variable_nodes × 10 / ns_per_frame`, so the figure counts nine iterations of
work that never ran and is roughly an order of magnitude above what the same
frames would cost with real channel noise.

The number is consequently not comparable to `ldpc_rust.json` and is not quoted
in the README. What the binary does measure honestly is the ring-buffer and
worker-handoff machinery — frames per second through the SPSC pipeline with a
negligible payload. Making it a throughput benchmark means feeding
AWGN-corrupted codewords (as `src/bin/ldpc_convergence_export.rs` does) and
dividing by the iterations actually consumed.
