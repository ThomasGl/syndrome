# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-07-26 — first public release

### Fixed
- **All 8 panics discovered by the robustness suite are eliminated.** The
  severe one: `compute_segmentation` asserted $B' \bmod C = 0$, which panicked
  for a large fraction of multi-code-block transport-block sizes and broke
  `DlSchEncoder`/`DlSchDecoder` for most large TBs. Segmentation now follows
  TS 38.212 §5.2.2 filler-bit semantics ($K' = \lceil B'/C \rceil$, slack
  zero-padded like filler bits), proven by end-to-end round-trips over a
  sweep of awkward TB sizes. Also fixed: `usize` overflow near the TB-size
  ceiling (explicit `MAX_TB_SIZE_BITS` bound), unbounded Viterbi constraint
  length (`new`/`with_generators` now return `Result`, `k ≤ 16`),
  division-by-zero in `rate_match`/`rate_dematch_llr` for `qm = 0` or
  `z = 0` (also reachable through `HarqBuffer`), and `PolarDecoder::new`
  accepting `list_size = 0`.
- Reed-Solomon benchmark chart: the three Rust encode variants were drawn in
  the same hue and indistinguishable; series colors are now CVD-validated
  distinct per (language, implementation) and the throughput axis is
  log-scaled so the pure-Python baselines remain visible.

### Added
- **Real IEEE 802.11 Wi-Fi LDPC encode/decode** (`src/wifi_ldpc_tables.rs`):
  all 12 real 802.11 Annex R/F parity-check shift matrices ($Z \in \{27, 54,
  81\} \times R \in \{1/2, 2/3, 3/4, 5/6\}$), cross-validated against the
  IEEE 802.11n-2009 standard text itself (Annex R, Tables R.1–R.3) plus two
  independent open-source transcriptions. `wifi_ldpc_encoder`/
  `wifi_ldpc_decoder` (and `WifiLdpcParams::build_encoder`/
  `build_decoder`) build a real, working `QcLdpcEncoder`/`QcLdpcDecoder`
  for any of the 12 combinations — the same LOMS kernel used for 5G NR,
  proven by `tests/wifi_ldpc_integration.rs` (encode → AWGN → decode
  round-trips, all 12 combinations, at high and 1-bit-error SNR). Previously
  `src/wifi.rs` only derived LDPC *parameters* (N/K/M, row/col block
  counts); it did not contain the actual matrices and could not encode or
  decode a real 802.11 codeword. `QcLdpcParams::from_raw_edges` (and the
  matching `QcLdpcEncoder`/`QcLdpcDecoder::from_raw_edges`) generalizes
  construction to accept any flat `(row, col, shift)` edge list, bypassing
  the 3GPP-specific `ils_for_z`/modulo-scaling path entirely, without
  touching the existing BG1/BG2 construction or its tests. **Out of scope
  for this pass** (real 802.11 mechanics beyond matrix lookup, documented in
  the `wifi`/`wifi_ldpc_tables` module docs): shortening (payloads smaller
  than $K$) and puncturing/rate-matching (the per-MCS coded-bit selection).
- `tests/reference_vectors.rs` — 14 known-answer conformance tests pinning
  codecs to external ground truth (reveng CRC catalogue, CCSDS Reed-Solomon
  conventions, published Hamming/Golay/BCH generator and parity-check
  matrices).
- `tests/robustness.rs` — 30 adversarial-input tests over every public API;
  each formerly-discovered panic is retained as a typed-error regression
  test.
- `fuzz/` — six libFuzzer targets covering the decoder entry points; they
  build on stable without instrumentation so CI can smoke-test them.
- `RateMatchCache` — memoized TS 38.212 §5.4.2 bit-selection/interleave
  index tables keyed on $(BG, Z, rv, Q_m, F, E)$, with `rate_match_into` /
  `rate_dematch_llr_into` allocation-free variants.

### Changed
- **Error handling unified on `FecError`** (now implementing
  `std::error::Error`): all public APIs that previously returned bare
  `&'static str` errors or panicked on degenerate input return
  `Result<_, FecError>`. Breaking signatures: `ViterbiDecoder::new` and
  `with_generators` return `Result`; Reed-Solomon `encode_*` return
  `Result` and reject zero-data-shard geometry; `SegmentationParams` gained
  a `b` field.
- Cargo.toml: added the `repository` field; dropped the `no-std` category
  (the crate currently requires `std` — the category will return only when a
  real `#![no_std]` core build exists).

### Performance
- **Polar CA-SCL decode ~1.63× faster** (N=1024, K=512, L=8: 2.51 ms →
  1.54 ms) — per-fork deep clones of the path buffers replaced by a fixed
  ping-pong arena of $2L$ slots reused via `clone_from`; zero heap
  allocation per information bit. Bit-identical to the retained reference
  implementation across seeded noisy frames.
- **Transport-block encode ~3.3× faster** (20 kbit TB, BG1, C=3: 152 µs →
  46.5 µs) — the rate-matching selection walk and interleave permutation are
  computed once per key and shared across all $C$ code blocks instead of
  re-derived per block.
- **Reed-Solomon AVX2 encode ~2.5× faster on short shards** (~8% at 4 KiB) —
  the 256 lo/hi nibble decompositions for the VPSHUFB kernel are precomputed
  once (8 KiB) instead of re-derived per (coefficient, row) on every call.
- Steady-state decode calls in Viterbi, Polar SC, and `DlSchDecoder` no
  longer heap-allocate: metric/traceback/scratch buffers moved into the
  codec structs (1–6% gains, honestly near noise — the point is the
  zero-allocation invariant, matching the Turbo and pipeline convention).
- **QC-LDPC encode: 2.75 Mbit/s → ~1.66 Gbit/s (~590×)** — replaced the dense
  generator-matrix multiply with the standard sparse structured encoding: the
  double-diagonal core-parity solve is derived programmatically from the 3GPP
  base-graph tables at construction (no hardcoded row/shift folklore), with
  back-substitution for p2–p4 and direct identity-extension rows. Verified by
  bit-identical equivalence with the retained dense reference and by syndrome
  checks across BG1/BG2 × 7 lifting sizes; a dense fallback exists but never
  triggers for any valid (BG, Z). Encoder construction drops from O(M²N) to
  O(E).

Decoder vectorization pass across all cores (single-thread, x86-64 AVX2;
every optimized path is runtime-detected, keeps its scalar implementation as
a tested reference, and is proven output-equivalent on seeded random inputs):
- **Reed-Solomon erasure decode: 972 Mbit/s → ~64 Gbit/s (66×)** — removed
  8 192 hidden per-call heap allocations and routed reconstruction through
  the same AVX2 VPSHUFB kernel as encode.
- **Polar SC decode: 5.9 → 35 Mbit/s (6×), SCL ~2.6×** — eliminated ~3 000
  recursion allocations per call; partial-sum re-encoding reduced from
  O(N log²N) to O(N log N) via GF(2) linearity; branch-free f/g kernels.
- **Viterbi soft decode: 5.1 → 30 Mbit/s (5.9×)** — AVX2 ACS processing all
  64 trellis states per step with shuffle-deinterleaved butterfly gathers.
- **Turbo decode: 3.1 → 13.4 Mbit/s (4.3×)** — AVX2 BCJR with all 8 states
  per register, bit-identical to scalar by design (sign-exact ±1 arithmetic).
- **BCH: encode 2.1×, decode 3.3×** — weight-proportional syndrome tables,
  per-β Chien multiply tables, byte-wise LFSR (≈70 KiB of tables per code).
- **CRC family: 3.8×** — bit-serial LFSR replaced by a 256-entry byte table
  (16-entry nibble table for CRC-6), equivalent by construction.

### Added
- **LTE Turbo codes** (`src/turbo.rs`): rate-1/3 PCCC per TS 36.212 — two 8-state
  RSC constituent encoders, QPP interleaver (8 supported K, pairs from Table
  5.1.3-3), iterative max-log-MAP decoder with extrinsic scaling and early exit.
- **BCH codes** (`src/bch.rs`): binary BCH(255,k,t≤10) over GF(2^8) — generator
  derived programmatically from cyclotomic cosets, systematic LFSR encoder,
  Berlekamp–Massey + Chien decode, shortened-code support; allocation-free
  encode/decode.
- **Extended Golay(24,12,8)** (`src/golay.rs`): syndrome-table decoder correcting
  all ≤3-bit errors in O(1); construction verified against the textbook weight
  enumerator over all 4096 codewords.
- **`examples/` directory**: 7 runnable, heavily-commented teaching examples
  forming a graded learning path from Hamming(7,4) to the full 5G NR
  transport-block chain and the lock-free pipeline.
- **All-algorithm benchmark exporter** (`src/bin/algo_bench_export.rs`): measures
  encode/decode information-bit throughput for all nine FEC cores and exports
  `bench/results/algos.json`; new cross-algorithm comparison chart in
  `bench/dashboard/gen_charts.py`.
- README: algorithm-portfolio table (history, complexity, industry usage, key
  papers), measured cross-algorithm benchmark table, embedded benchmark charts,
  examples ladder, algorithm↔textbook learning map, and references [12]–[20]
  (Arıkan, Tal & Vardy, Berrou, BCH, Golay, Viterbi, Hamming, TS 36.212,
  Lin & Costello).
- MIT `LICENSE`.
- GitHub Actions CI (`.github/workflows/ci.yml`): test suite, doctests, Clippy
  (`-D warnings`), rustfmt, MSRV check, and a `cargo audit` security scan.
- `CONTRIBUTING.md` and this `CHANGELOG.md`.
- `rust-toolchain.toml` pinning the `stable` channel with `rustfmt`/`clippy`.
- Crate metadata in `Cargo.toml` (`license`, `description`, `authors`,
  `keywords`, `categories`, `rust-version = 1.85`) and a documented
  `[lints.clippy]` policy.

### Fixed
- Resolved all 35 rustdoc warnings (unqualified intra-doc links now
  type-qualified, links to private items demoted to plain code spans, and
  bracketed math indices rewritten as LaTeX subscripts so they no longer
  parse as broken links); `cargo doc` is now warning-free alongside rustc
  and clippy.
- **Polar SC/SCL decoder correctness**: the successive-cancellation g-function
  consumed raw decoded bits instead of the re-encoded partial sums, silently
  producing wrong codewords for any message of Hamming weight ≥ 2 (the old
  all-zero-message tests masked this). Also fixed `frozen_mask` for N > 256,
  which previously exceeded the embedded reliability table and mis-froze
  positions (now uses a polarization-weight ordering fallback). Decoding is now
  proven by exhaustive round-trips over every message at N=8/16 and seeded
  sweeps up to N=1024, plus measured 20/20 AWGN recovery at 3 dB.
- `QcLdpcEncoder::clone` no longer re-derives the base graph from a bit-count
  ratio heuristic; the encoder now stores `(bg, z)` and exposes
  `base_graph()`/`lifting_size()`.
- BER-waterfall chart: BER and BLER now share a single log axis (the previous
  dual-axis rendering was misleading), and the Shannon-limit annotation uses
  the correct real-AWGN bound $(2^{2R}-1)/(2R) \approx -0.6$ dB for R=0.324
  (previously off by the factor-2 in the denominator).

### Changed
- README reframed as a multi-standard FEC library with status badges; corrected
  the documented MSRV to 1.85 (required by the Rust 2024 edition).
- Renamed `iLS_for_z` / `iLS_lookup` to snake_case (`ils_for_z` / `ils_lookup`)
  to match the project naming conventions.
- Normalized formatting across the codebase (`cargo fmt`).
- Resolved all Clippy warnings so `cargo clippy --all-targets -- -D warnings`
  passes cleanly, without altering numeric kernels.

## Pre-release development history

Everything below predates the first crates.io publish and is folded into
0.1.0; it is kept separate only to record the order in which the library
was built.

### Added
- 5G NR TS 38.212 transport-block chain: CRC (24A/B/C, 16/11/6), code-block
  segmentation + base-graph selection, QC-LDPC encode/decode, rate matching with
  redundancy versions, and HARQ soft combining (`DlSchEncoder`/`DlSchDecoder`).
- QC-LDPC Layered Offset Min-Sum decoder for BG1/BG2 with AVX2/NEON paths and
  syndrome-check early termination, driven by real 3GPP TS 38.212 lifting tables
  (`data/bg_tables.json`).
- FEC cores: Reed-Solomon GF(256) erasure codec (AVX2 VPSHUFB path), Viterbi
  (hard + soft), Polar SC / CA-SCL, and Hamming(7,4).
- Wi-Fi 6/7 (802.11ax/be) and 6G / IMT-2030 research profiles.
- BPSK AWGN channel simulator and media-reconstruction integration tests.
- Lock-free SPSC ring buffer, multi-worker LDPC pipeline, and thread affinity.
- Reproducible cross-language benchmark suite with a byte-identical checksum gate
  and a Highcharts dashboard.
