# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- MIT `LICENSE`.
- GitHub Actions CI (`.github/workflows/ci.yml`): test suite, doctests, Clippy
  (`-D warnings`), rustfmt, MSRV check, and a `cargo audit` security scan.
- `CONTRIBUTING.md` and this `CHANGELOG.md`.
- `rust-toolchain.toml` pinning the `stable` channel with `rustfmt`/`clippy`.
- Crate metadata in `Cargo.toml` (`license`, `description`, `authors`,
  `keywords`, `categories`, `rust-version = 1.85`) and a documented
  `[lints.clippy]` policy.

### Changed
- README reframed as a multi-standard FEC library with status badges; corrected
  the documented MSRV to 1.85 (required by the Rust 2024 edition).
- Renamed `iLS_for_z` / `iLS_lookup` to snake_case (`ils_for_z` / `ils_lookup`)
  to match the project naming conventions.
- Normalized formatting across the codebase (`cargo fmt`).

### Fixed
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
