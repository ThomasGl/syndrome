# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] — 2026-07-29

### Added

- `QcLdpcDecoder::decode_layered_offset_min_sum_scalar` — the same layered
  offset min-sum decode with the scalar kernel forced on every architecture,
  bypassing the runtime AVX2 probe and the compile-gated NEON path. It exists
  so the scalar fallback can be benchmarked against the vectorized kernels on
  one machine; a unit test asserts the two produce identical hard-decision
  output and identical iteration counts on the same input. Kernel selection is
  resolved once per call, before the iteration and layer loops, so the
  runtime-dispatched entry points are unchanged.
- `LdpcFrame::iterations_used` — the number of layered passes the decoder
  actually consumed for that frame, which may be fewer than the configured
  budget when the syndrome check terminates early.

- `QcLdpcDecoder::decode_layered_offset_min_sum_traced` — the same layered
  offset min-sum decode, with a callback invoked once per completed layered
  pass so a caller can record a real per-iteration convergence trace from a
  single continuous run. Re-invoking the untraced decoder once per iteration
  count cannot reproduce the trajectory, because the extrinsic message buffer
  is reset at the start of every call. The callback is a monomorphized
  generic, so the untraced entry point compiles to the same code as before.

## [0.1.1] — 2026-07-26

### Fixed

- LaTeX in the API documentation now renders on docs.rs. rustdoc has no math
  support of its own, so the `$...$` expressions throughout the module and
  function docs were being served as literal text. A KaTeX header is now
  injected via `--html-in-header`, declared under `[package.metadata.docs.rs]`.
  Documentation-only; no code, API, or behaviour changed.

## [0.1.0] — 2026-07-26

First public release.

### Added

- **Nine FEC cores**, each with encode and decode: Hamming(7,4), extended
  Golay(24,12,8), BCH(255,k,t≤10) over GF(2⁸) with Berlekamp–Massey and Chien
  search, Reed-Solomon erasure coding over GF(256), rate-1/2 K=7 Viterbi
  (hard and soft decision), LTE rate-1/3 Turbo (iterative max-log-MAP),
  5G NR QC-LDPC (layered offset min-sum), Polar SC and CA-SCL, and the
  CRC-24A/B/C, CRC-16/11/6 family.
- **Complete 3GPP TS 38.212 transport-block chain**: CRC attachment, code-block
  segmentation with base-graph selection, QC-LDPC encode/decode over BG1 and
  BG2, rate matching with redundancy versions, HARQ soft combining, and the
  `DlSchEncoder` / `DlSchDecoder` end-to-end pair.
- **IEEE 802.11 Wi-Fi 6/7 LDPC**: all twelve 802.11 Annex R/F parity-check
  shift matrices ($Z \in \{27, 54, 81\}$, $R \in \{1/2, 2/3, 3/4, 5/6\}$),
  decoded through the same LOMS kernel as 5G NR. Shortening and
  puncturing/rate-matching are out of scope for this release and documented as
  such in the module docs.
- **SIMD kernels**: AVX2 paths for the LDPC, Reed-Solomon, Viterbi and Turbo
  inner loops, selected at runtime via `is_x86_feature_detected!`; NEON paths
  on AArch64. Each is proven output-equivalent to its scalar reference by
  seeded randomized tests, and the scalar implementations remain available.
- **Lock-free concurrency**: a cache-line-padded single-producer
  single-consumer ring buffer, a multi-worker LDPC pipeline sized from
  `available_parallelism()`, and optional per-core thread affinity behind the
  `affinity` feature.
- **Typed error handling**: fallible public APIs return `Result<_, FecError>`,
  with `FecError` implementing `std::error::Error`.
- **Test suite**: 293 tests on x86-64 (294 on AArch64), including 14
  known-answer conformance vectors pinned to external ground truth — the
  reveng CRC catalogue, CCSDS Reed-Solomon conventions, and published
  Hamming/Golay/BCH generator and parity-check matrices — plus 31 adversarial
  input tests, 73 doctests, and six libFuzzer targets over the decoder entry
  points.
- **Seven runnable teaching examples**, from a single corrected bit flip to a
  full 5G NR transport block surviving an AWGN channel.
- **Reproducible benchmark suite** (`bench/run_all.sh`) comparing Rust, a
  same-algorithm C++ port, and Python, gated on byte-identical output so a
  divergence fails the run rather than producing a misleading number.

Minimum supported Rust version: 1.85 (Rust 2024 edition).
