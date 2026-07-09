# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

## [0.1.0] — initial

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
