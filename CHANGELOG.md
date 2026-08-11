# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] — 2026-08-09

### Added

- **Wi-Fi 802.11 LDPC shortening and puncturing** (`src/wifi_rate_matching.rs`):
  `encode_shortened`/`decode_shortened` accept a payload smaller than a
  codeword's $K$ (padded with known-zero bits that are never transmitted)
  and a transmitted length smaller than the post-shortening budget (parity
  bits dropped from the tail), for any of the 12 real 802.11 $(Z, R)$
  matrices. The decoder reconstructs the full-length LLR buffer before
  decoding: shortened positions get a high-confidence known-zero LLR,
  punctured positions get an erasure ($LLR = 0$). Previously only a full,
  unshortened, unpunctured codeword ($K$ info bits exactly filling $N$
  coded bits) could be encoded or decoded.
  `tests/wifi_shortening_puncturing_integration.rs` verifies the encode →
  AWGN → decode round-trip, with genuine shortening and puncturing applied,
  across all 12 combinations. Still not implemented, and now the documented
  scope boundary: multi-codeword segmentation (a payload larger than one
  codeword's $K$), and the 802.11 PPDU-level formula (§19.5.3.2) that
  derives the available coded-bit count from an MCS, bandwidth, and PSDU
  length — callers supply that length directly instead.
- **Bluetooth FEC profiles** (`src/bluetooth.rs`): the complete set of FEC
  schemes in the Bluetooth Core Specification (unchanged since 5.0; verified
  against 5.0 through 6.2) — the LE Coded PHY convolutional code ($K=4$,
  rate 1/2, $G_0 = 1+x+x^2+x^3$, $G_1 = 1+x^2+x^3$) built on the existing
  Viterbi engine with hard and soft decoding, the $S=8$ pattern
  mapper/soft-demapper, BR/EDR FEC 1/3 (3× repetition, majority decode),
  and BR/EDR FEC 2/3 (the (15,10) shortened Hamming code,
  $g(D)=D^5+D^4+D^2+1$, single-error correction with double-error
  detection). Unit tests reproduce the specification's own sample data
  bit-exactly: the Vol 6 Part C reference packet (every FEC output bit for
  both CI values and the $S=8$ symbol stream) and all ten (15,10)
  generator rows from the Vol 2 FEC sample data, cross-checked against
  libbtbb's independent table. Out of scope, documented: packet assembly,
  whitening, and the Bluetooth CRC/HEC family.
- **Public `bits` module** (`src/bits.rs`): MSB-first `bytes_to_bits` /
  `bits_to_bytes` (caller-buffer and `_vec` forms, validating that every
  element is 0/1) and `hard_decision`, the crate-wide LLR sign rule
  ($L < 0 \Rightarrow 1$). Nearly every API in the crate speaks
  one-bit-per-byte, and until now every user had to write these conversions
  themselves.
- **`LdpcWorkspace`** (`src/qc_ldpc.rs`): an owning bundle of all four LDPC
  decode buffers, built by `QcLdpcDecoder::workspace()`. New
  `decode_with_workspace` / `decode_5g_with_workspace` /
  `wifi_rate_matching::decode_shortened_with_workspace` entry points replace
  the three sizing calls and four `vec![...]` lines previously required
  before a first decode (the Wi-Fi shortened decode drops from 8 parameters
  to 5). The raw slice entry points are unchanged for callers who want exact
  allocation control; each decode remains allocation-free either way, and
  the workspace path is tested bit- and iteration-identical to the raw path.
- **`DlSchConfig`** (`src/transport_block.rs`): named-field configuration for
  the DL-SCH pair, with `DlSchEncoder::from_config` /
  `DlSchDecoder::from_config`, so the six positional numerics of
  `DlSchDecoder::new` can no longer be transposed silently and the encoder
  and decoder can be built from literally the same value.
- **`syndrome::VERSION`**: the crate version captured at compile time, so
  downstream wrappers (e.g. the Python binding) can report which core they
  were compiled against instead of their own version.

### Fixed

- **A build without `data/bg_tables.json` silently produced fake 5G base
  graphs.** `build.rs` emitted a 2-entry placeholder BG1/BG2 when the table
  file was missing — compiling cleanly into a crate whose "base graphs" were
  two-edge toys, with no warning anywhere. The file ships in both git and
  the crates.io tarball, so its absence always means a broken checkout; it
  is now a hard build error naming the regeneration tool. The `bg1`/`bg2`
  keys and their `rows`/`cols` fields are also required now instead of
  being silently skipped or defaulted.

## [0.2.0] — 2026-08-04

### Changed

- **MSRV raised from 1.85 to 1.97** (latest stable at time of release). A
  minor-version bump rather than a patch, because raising the MSRV is a real
  breaking change for anyone still on an older toolchain — Cargo's default
  caret matching (`^0.1.x`) would otherwise have handed this to existing
  dependents automatically. `rust-toolchain.toml` already tracked `channel =
  "stable"` generically, so no pin needed updating there; only the declared
  floor moved.
- Two workarounds that existed solely to reconcile the old MSRV against
  current stable are gone, not just relaxed: three AVX2 helpers in
  `src/turbo.rs` no longer need an inner `unsafe` block plus
  `#[allow(unused_unsafe)]` (Rust 1.87 made `core::arch` intrinsic calls safe
  from within an equally `#[target_feature]`-gated function; MSRV 1.97 is
  past that), and the `manual_is_multiple_of` clippy allow is gone —
  clippy's MSRV-aware lints now correctly suggest `.is_multiple_of()` in
  `rate_matching.rs`, `transport_block.rs`, and a test in `viterbi.rs`
  (`is_multiple_of` stabilized in 1.87, also now behind the floor), and two
  `if`-let-chains simplify nested `if let Some(...) { if ... { ... } }`
  patterns in `reed_solomon.rs`, `ldpc_pipeline.rs`, and
  `src/bin/ldpc_pipeline_bench.rs`. None of this changes behavior; every
  site was verified equivalent by the existing test suite passing unchanged.

## [0.1.3] — 2026-07-31

### Fixed

- **`AwgnChannel::new` seeded xorshift64 with `seed | 1`, silently colliding
  every `(even, even + 1)` seed pair onto identical noise.** `0` and `1`
  produced the same sequence, as did `2`/`3`, `100`/`101`, and so on for
  every consecutive pair — two channels built with "different" seeds could
  transmit byte-identical noise. The seed is now mixed through SplitMix64
  before becoming the xorshift state, which is both collision-free (the
  mixer is a bijection) and free of the weak diffusion between low-bit-similar
  seeds that a raw `seed | 1` (or a naive `if seed == 0 { 1 }`, which merely
  trades the bug for a new collision against the literal seed `1`) would
  still carry. Found while investigating why a HARQ combining test was
  silently combining a transmission with itself.
- **`DlSchDecoder::decode` dropped the last `2Z` accumulated HARQ LLRs of
  every code block before they reached the LDPC decoder.** The mapping from
  the HARQ circular buffer into the decoder's LLR array computed
  `valid_len = ncb - 2Z`, double-subtracting the puncture width: `ncb`
  (`HarqBuffer`'s circular buffer size) is already `N - 2Z` by construction,
  so the correct value is `ncb` with no further subtraction. The bug never
  panicked or returned an error — it silently zeroed real information the
  receiver had (worth roughly one AVX2 decode iteration's amount of data per
  code block), and specifically weakened incremental-redundancy
  retransmissions at RV 2 and RV 3, whose rate-matching walk reaches into
  exactly the dropped region. A test decoding a real, `AwgnChannel`-corrupted
  codeword confirms an RV0 transmission that fails CRC alone is recovered by
  combining it with an RV3 retransmission, and was verified — by temporarily
  reverting only this fix — to fail under the old code.
- `DlSchDecoder` had no way to report the real (rounded) coded length it
  expects, so a caller could only guess it from the raw `g` constructor
  argument — wrong whenever `g` wasn't already a multiple of
  `Qm * num_code_blocks`. Added `DlSchDecoder::output_bits`, mirroring the
  accessor `DlSchEncoder` already had.

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
