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

## Reed-Solomon erasure-recovery GIF

The animation in README §5.1 is likewise generated entirely from real
encoder/decoder output — no hand-drawn "before/after" image.

```bash
# From repo root:
cargo run --release --bin rs_erasure_export
python3 bench/dashboard/gen_rs_erasure_gif.py
```

1. `cargo run --release --bin rs_erasure_export` generates an 80×80 grayscale
   test image from a deterministic arithmetic pattern (concentric rings —
   not a loaded external asset, so the payload is fully specified by the
   code that produces it), encodes it as RS(10,4) with the same
   [`ReedSolomon::encode_with_avx2`](../src/reed_solomon.rs) kernel measured
   in §5.1, erases 4 of the 10 data shards (shard indices 1/3/5/7 — the
   maximum this code tolerates), and reconstructs them with the real
   [`ReedSolomon::decode`](../src/reed_solomon.rs). The exporter asserts the
   reconstruction is byte-for-byte identical to the original **before**
   writing `bench/results/rs_erasure.json` — a mismatch panics the program
   rather than exporting an unverified result.
2. `bench/dashboard/gen_rs_erasure_gif.py` reads that JSON and renders the
   three real phases (original / corrupted / recovered) to
   `bench/dashboard/exports/rs_erasure.gif`, with a shard-status strip
   (green = present or recovered, red = erased, blue = parity) beneath the
   image. The only liberties taken past the raw bytes are which colors mean
   "erased" vs. "present" and how long each phase is held on screen — GIF
   optimization then merges the held duplicate frames and combines their
   durations, so the file's stored frame count is smaller than the number of
   phases rendered; the script re-opens its own output to report the true
   count rather than assuming its pre-save frame list.

| File | Contents |
|------|----------|
| `bench/results/rs_erasure.json` | Original/corrupted/recovered image bytes from one real encode→erase→decode cycle |
| `bench/dashboard/exports/rs_erasure.gif` | Rendered animation embedded in `README.md` |

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
| `bench/results/ldpc_pipeline_rust.json` | SPSC pipeline throughput on real AWGN-corrupted frames |

### What `ldpc_pipeline_bench` measures

`src/bin/ldpc_pipeline_bench.rs` runs the SPSC pipeline (`LdpcPipeline`) end
to end on genuinely noisy frames: a real BG1 Z=384 codeword is produced with
`QcLdpcEncoder`, corrupted once through `AwgnChannel` at a fixed `Eb/N0` and
PRNG seed, and that same noisy LLR vector is submitted as every frame's
input. Because each frame is real noisy data rather than an error-free
codeword, the decoder performs genuine layered passes instead of exiting
after one.

The decoder's actual per-frame iteration count is read back via
[`LdpcFrame::iterations_used`](../src/ldpc_pipeline.rs) and recorded as
`mean_iters_per_frame` in the JSON; `melem_per_s` is derived from that real
count (`n_variable_nodes × total_iterations_used / elapsed_ns`), not from the
configured iteration budget. `frames_per_s` and `ns_per_frame` report the
ring-buffer/worker-handoff throughput directly, timed the same way.

The benchmark sweeps worker counts `{1, 2, 4, 8}` explicitly via
`LdpcPipeline::with_workers`, so `ldpc_pipeline_rust.json` is a JSON
**object**, not a bare array: `worker_sweep` holds one record per worker
count (each carrying `n_workers` and `speedup_vs_1_worker`), and
`overhead_isolation` holds the same-process pipeline-overhead measurement
described below.

**Why an explicit sweep, and why `with_workers` rather than `new`.** An
earlier version called `LdpcPipeline::new` once and printed a hardcoded
`"1 worker thread"` banner. `new` sizes its worker pool from
`std::thread::available_parallelism()` (clamped to 8), so on any multi-core
host that banner was false, and every Melem/s figure the benchmark ever
produced was really an N-worker aggregate mislabeled as a single-worker
number. `with_workers` makes the count explicit and the sweep reproducible
across machines with different core counts, instead of silently depending on
`nproc`.

**Why the 1-worker figure still didn't match `ldpc_bench_export`'s
single-threaded number, and what actually explains the gap.** Fixing the
worker-count mislabeling above was not the whole story: even a genuine
1-worker pipeline measurement came in well below `ldpc_bench_export`'s
figure, and the first attempt to explain that gap (an earlier version of this
file, and of README.md §5.2.1) attributed it to "pipeline overhead" without
measuring the claim — which turned out to be wrong. Two real, separate
effects were conflated:

1. **Workload difference (the dominant one).** `ldpc_bench_export` decodes a
   synthetic, never-converging LLR pattern for a fixed 10-iteration budget.
   The decoder's per-iteration early-exit syndrome check
   (`check_syndrome_f32`, scalar, short-circuiting) fails on the first parity
   row against that input, essentially every iteration — nearly free. This
   file decodes a real AWGN-corrupted codeword that converges in ~5
   iterations, and the *successful* check that ends the decode has to scan
   the entire parity matrix — work costing roughly as much as one more AVX2
   decode iteration, which the `Melem/s` metric (`n × iterations_used /
   time`) does not count as an iteration. Confirmed by capping the real
   codeword's iteration budget below its convergence point (5), so no early
   exit fires: its per-iteration cost then matches the synthetic pattern's
   within ~1%.
2. **Genuine pipeline overhead — smaller, and only measurable safely
   same-process.** This host's wall-clock throughput drifted by tens of
   percent between separate process invocations during this investigation
   (confirmed by running the identical binary repeatedly), which is larger
   than the effect being isolated. Comparing this file's 1-worker number to a
   *different binary's* number, run minutes apart, cannot distinguish real
   overhead from drift — which is exactly the mistake the original "~25%,
   genuine overhead" claim made. The benchmark now alternates a plain decode
   loop (`measure_tight_loop`) against the 1-worker pipeline on identical
   input, `OVERHEAD_ROUNDS = 5` times, within one process, alternating which
   side runs first each round, before the worker-count sweep runs.

   A dozen such same-process runs, across several work sessions, gave: 0.4%,
   1.9%, 4.5%, 5.5%, 6.0%, 6.4%, 6.8%, 7.6%, 7.6%, 8.6%, 8.8%, 9.9%, 12.0%,
   13.4%, 16.7%, 18.1%, 21.8% (yes, more numbers than the round count — the
   point is the spread, not the count). This machine is a
   shared, virtualized host whose background load is outside this
   benchmark's control, and the isolated-overhead figure is genuinely
   sensitive to it — small enough in absolute terms (single-digit to
   low-double-digit Melem/s) that host noise can dominate it on a busy
   machine. **No single percentage is asserted as "the" pipeline overhead
   here.** What is robust across every one of those runs, on every occasion
   this was checked, is the *qualitative* finding: it is nowhere near the
   ~25% the original same-workload-blind comparison suggested, and workload
   difference (point 1) is the larger effect. `bench/results/ldpc_pipeline_rust.json`'s
   `overhead_isolation.pipeline_overhead_pct` always carries the number from
   the run that actually produced the committed JSON — read that field for
   the current figure rather than trusting a number frozen into prose.

That investigation also found and fixed a genuine small bug: the pipeline
worker was zeroing its ~474 KiB extrinsic (`edge_r`) buffer before every
decode call (`ldpc_pipeline.rs`), redundant with
`decode_layered_offset_min_sum` already zeroing it internally at the top of
every call. Removing it lowered the redundant-memset cost that had been
inflating the overhead figure by a few points, but it does not explain the
run-to-run swings above — those are host contention, not the decoder.
