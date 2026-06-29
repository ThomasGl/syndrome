# glezer-rsv

[![CI](https://github.com/thomas-glezer/glezer-rsv/actions/workflows/ci.yml/badge.svg)](https://github.com/thomas-glezer/glezer-rsv/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org/)
[![Tests](https://img.shields.io/badge/tests-135%20passing-brightgreen)](tests/)
[![5G NR](https://img.shields.io/badge/5G%20NR-TS%2038.212-blue)](src/transport_block.rs)
[![Wi-Fi 7](https://img.shields.io/badge/Wi--Fi%207-802.11be-blue)](src/wifi.rs)
[![6G Research](https://img.shields.io/badge/6G-IMT--2030%20research-blueviolet)](src/sixg.rs)

**A high-performance, multi-standard Forward Error Correction library in safe Rust — covering the full 5G NR TS 38.212 transport-block chain, Wi-Fi 6/7 (802.11ax/be), 6G research extensions, Reed-Solomon, Viterbi, and Polar codes — with AVX2/NEON SIMD, lock-free pipelining, and end-to-end media reconstruction tests.**

---

## At a Glance

| What | Detail |
|---|---|
| **Standards** | 3GPP TS 38.212 (5G NR), 802.11ax/be (Wi-Fi 6/7), IMT-2030 (6G research) |
| **Algorithms** | QC-LDPC LOMS, Viterbi ACS, Polar SC/CA-SCL, Reed-Solomon GF(256) |
| **SIMD** | AVX2 (x86-64, runtime-detected), NEON (AArch64, compile-gated) |
| **Concurrency** | Lock-free SPSC ring buffer, multi-worker LDPC pipeline, per-core affinity |
| **Tests** | 135 total — 81 unit · 7 integration · 4 media reconstruction · 43 doctests |
| **Allocations** | Zero heap allocation inside the decode hot-path |
| **Benchmarks** | RS: ~10 GiB/s (AVX2 VPSHUFB), LDPC: ~119 Melem/s · all numbers from running code |

---

## Plain-English Summary *(for recruiters and non-specialists)*

> **The one-sentence version:** this library implements the error-correction algorithms that live inside every 5G phone chip, Wi-Fi router, and satellite terminal — in safe Rust, matching hand-optimised C++ speed.

### What problem does it solve?

Every radio link, storage device, and network connection corrupts data.  Noise adds errors; packets get dropped; storage cells flip bits over time.  **Forward Error Correction (FEC)** adds mathematical redundancy so the receiver can reconstruct the original without asking for a retransmission.  It is the invisible backbone of every wireless standard, streaming platform, and storage system built after 2000.

Think of it like a smarter parity bit: a 5G base station transmits 26,000-bit blocks with enough redundancy that the phone can recover the original even when ~30% of the received bits are wrong — all within ~2 milliseconds on the baseband chip.

### Why does this implementation stand out?

| Challenge | What this library does |
|---|---|
| **Mathematical fidelity** | Full 3GPP TS 38.212 chain: CRC, segmentation, rate matching, HARQ soft combining — not just the LDPC kernel |
| **Multi-standard** | One codebase covers 5G NR, Wi-Fi 6/7, and 6G research extensions |
| **Speed** | AVX2 SIMD path beats the C++ scalar reference; NEON path for ARM |
| **Memory discipline** | Zero heap allocation inside the decode inner loop; flat layout; cache-aligned |
| **End-to-end proof** | Audio and video payload reconstruction tests assert perfect bit recovery through simulated AWGN noise |
| **Concurrency** | Lock-free SPSC pipeline — no OS mutex on the decode hot path |

---

## Table of Contents

1. [Module Overview](#1-module-overview)
2. [Quickstart](#2-quickstart)
3. [API Examples](#3-api-examples)
4. [Test Suite](#4-test-suite)
5. [Benchmarks](#5-benchmarks)
6. [Engineering Background](#6-engineering-background)
7. [Design Notes](#7-design-notes)
8. [Real-World Applications](#8-real-world-applications)
9. [References](#9-references)
10. [Learning Path](#10-learning-path)
11. [Similar Projects](#11-similar-projects)

---

## 1. Module Overview

```
glezer_rsv/
├── src/
│   ├── lib.rs              — Crate root; re-exports all public API
│   ├── error.rs            — FecError enum (InvalidParam, CrcMismatch, …)
│   │
│   │   ── 5G NR TS 38.212 transport-block chain ──────────────────────
│   ├── crc.rs              — CRC-24A/B/C/16/11/6 (§5.1 generator polynomials)
│   ├── segmentation.rs     — Code block segmentation + BG selection (§5.2.2)
│   ├── qc_ldpc.rs          — LOMS QC-LDPC encoder + decoder, BG1/BG2, SIMD
│   ├── rate_matching.rs    — Bit selection, interleaving, RV starting offsets (§5.4.2)
│   ├── harq.rs             — Soft-combining LLR accumulator across HARQ rounds
│   ├── transport_block.rs  — DlSchEncoder / DlSchDecoder (full TB chain)
│   ├── quantize.rs         — f32 → i8 LLR quantization (scale + clamp, −127..127)
│   │
│   │   ── Multi-standard FEC cores ──────────────────────────────────
│   ├── viterbi.rs          — Rate-1/2 K=7 Viterbi (hard Hamming ACS + soft max-log-MAP)
│   ├── polar.rs            — Polar codes: SC + CA-SCL decode, 3GPP reliability seq
│   ├── reed_solomon.rs     — GF(256) Vandermonde erasure RS encoder/decoder
│   ├── hamming.rs          — Hamming(7,4) encode/decode
│   │
│   │   ── Wi-Fi 6 / 7 and 6G ─────────────────────────────────────────
│   ├── wifi.rs             — 802.11ax/be MCS tables, LDPC parameter selection
│   ├── sixg.rs             — IMT-2030 profiles, AMC ladder, 4096-QAM, 8 Mbit TB
│   │
│   │   ── Channel simulation ───────────────────────────────────────────
│   ├── channel_sim.rs      — BPSK AWGN simulator, Box-Muller, xorshift64 PRNG
│   │
│   │   ── Infrastructure ───────────────────────────────────────────────
│   ├── spsc_queue.rs       — Lock-free SPSC ring buffer (AtomicUsize, ~1.1 ns)
│   ├── ldpc_pipeline.rs    — Multi-worker LDPC pipeline, per-core affinity
│   ├── affinity.rs         — Thread-to-core pinning (optional `affinity` feature)
│   ├── simd_avx2.rs        — AVX2 inner-loop kernels (x86_64, runtime-detected)
│   └── simd_neon.rs        — NEON inner-loop kernels (aarch64, compile-gated)
├── tests/
│   ├── ldpc_integration.rs     — Encode→decode round-trips + 1-bit error correction
│   └── media_reconstruction.rs — Audio/video AWGN simulation + perfect reconstruction
├── bench/
│   ├── cpp/                — Same-algorithm C++ ports (checksum-gated)
│   ├── python/             — Python same-algo + reedsolo baselines
│   ├── dashboard/          — Highcharts visualisation (non-commercial)
│   └── run_all.sh          — Orchestration + byte-identical checksum gate
├── data/
│   └── bg_tables.json      — BG1/BG2 extracted from 3GPP TS 38.212
└── tools/
    └── gen_bg_tables.py    — Parses the 3GPP DOCX spec (38212-gf0.zip)
```

---

## 2. Quickstart

**Prerequisites:** Rust stable ≥ 1.85 (Rust 2024 edition)

```bash
# Build
cargo build --release

# Unit and integration tests
cargo test

# End-to-end media reconstruction (prints BER waterfall tables)
cargo test --test media_reconstruction -- --nocapture

# Doctests
cargo test --doc

# Full cross-language benchmark suite (RS + LDPC, requires GCC + Python 3.9+)
bash bench/run_all.sh

# Highcharts throughput dashboard
cd bench/dashboard && python -m http.server   # open http://localhost:8000
```

---

## 3. API Examples

### 5G NR transport-block encode + decode

```rust
use glezer_rsv::transport_block::{DlSchEncoder, DlSchDecoder};
use glezer_rsv::channel_sim::AwgnChannel;

// Encode a 1 000-byte payload through the full 5G NR chain
let payload = vec![0x42u8; 1000];
let encoder  = DlSchEncoder::new();
let (coded, tb_params) = encoder.encode(&payload)?;

// Simulate AWGN at 4 dB Eb/N0 (rate-1/3 BG1 channel)
let mut ch = AwgnChannel::new(4.0, 1.0 / 3.0, /*seed=*/42);
let llr    = ch.transmit(&coded);

// Decode — returns original bytes on CRC pass
let mut decoder = DlSchDecoder::new();
let recovered   = decoder.decode(&llr, &tb_params)?;
assert_eq!(recovered, payload);
```

### QC-LDPC encoder + decoder (low-level)

```rust
use glezer_rsv::{BaseGraph, QcLdpcDecoder, QcLdpcEncoder};

let enc = QcLdpcEncoder::new(BaseGraph::Bg1, 384)?;
let dec = QcLdpcDecoder::with_lifting_size(BaseGraph::Bg1, 384, 0.25)?;

let k = enc.info_bit_count();   // 22 × 384 = 8 448 info bits
let n = dec.variable_node_count(); // 68 × 384 = 26 112

let info = vec![0u8; k];
let mut codeword = vec![0u8; n];
enc.encode(&info, &mut codeword)?;

// Decode with soft LLRs (no heap allocation in the inner loop)
let mut llr     = codeword.iter().map(|&b| if b == 0 { 5.0f32 } else { -5.0 }).collect::<Vec<_>>();
let mut edge_r  = vec![0.0f32; dec.required_edge_buffer()];
let mut scratch = vec![0.0f32; dec.required_layer_buffer()];
let mut hard    = vec![0u8; n];
let iters_used  = dec.decode_layered_offset_min_sum(
    &mut llr, &mut edge_r, &mut scratch, &mut hard, /*max_iters=*/10,
)?;
println!("converged in {iters_used} iterations (syndrome-check early exit)");
```

### Reed-Solomon packet-loss recovery

```rust
use glezer_rsv::ReedSolomon;

// RS(10, 4): recover any 4 lost packets from 14 transmitted
let mut rs = ReedSolomon::new(10, 4);
rs.precompute_mul_tables();

let data: Vec<Vec<u8>> = (0..10).map(|_| vec![0xABu8; 1024]).collect();
let refs: Vec<&[u8]>   = data.iter().map(|v| v.as_slice()).collect();
let mut parity = vec![0u8; 4 * 1024];
rs.encode_with_tables_chunked(&refs, &mut parity); // ~10 GiB/s on AVX2 hardware
```

### Viterbi (rate-1/2, K=7)

```rust
use glezer_rsv::viterbi::ViterbiDecoder;

let dec = ViterbiDecoder::new(7); // constraint length 7, G=(0o133, 0o171)
let info_bits = vec![1u8, 0, 1, 1, 0, 0, 1];
let coded     = dec.encode(&info_bits);   // adds 6 zero-tail bits → 28 coded bits

// Hard-decision decode
let decoded = dec.decode_hard(&coded);
assert_eq!(decoded, info_bits);

// Soft-decision decode from channel LLRs
let llr: Vec<f32> = coded.iter().map(|&b| if b == 0 { 3.0 } else { -3.0 }).collect();
let decoded_soft  = dec.decode_soft(&llr);
assert_eq!(decoded_soft, info_bits);
```

### Lock-free LDPC pipeline (multi-worker)

```rust
use glezer_rsv::{BaseGraph, QcLdpcDecoder, LdpcPipeline};

let decoder  = QcLdpcDecoder::with_lifting_size(BaseGraph::Bg1, 384, 0.25).unwrap();
let mut pipe = LdpcPipeline::with_workers(decoder, /*max_iters=*/10, /*threads=*/4);

let mut frame = pipe.acquire().expect("16-slot pool");
frame.llr_mut().iter_mut().for_each(|v| *v = 5.0);
pipe.submit(frame);

loop {
    if let Some(result) = pipe.try_recv() {
        let _bits = result.hard(); // zero-copy read
        pipe.release(result);
        break;
    }
    std::hint::spin_loop();
}
```

---

## 4. Test Suite

### 4.1 Component coverage (135 tests total)

| Category | Count | Location |
|---|---|---|
| Unit tests | 81 | embedded in `src/*.rs` |
| LDPC integration (encode→decode round-trips) | 7 | `tests/ldpc_integration.rs` |
| End-to-end media reconstruction | 4 | `tests/media_reconstruction.rs` |
| Doctests | 43 | `///` examples in all public API |

Run all:
```bash
cargo test          # unit + integration + doctests
cargo test --test media_reconstruction -- --nocapture   # BER waterfall output
```

### 4.2 End-to-end media reconstruction

Each test encodes a real payload, simulates BPSK AWGN at multiple SNR levels, decodes, and asserts CRC pass + bit-exact reconstruction.

| Test | Payload | Protocol | Threshold | Result |
|---|---|---|---|---|
| `audio_frame_5g_nr_reconstruction` | 100 bytes (Opus audio frame) | 5G NR BG2, Z=88, R=1/2 | 5 dB Eb/No | ✓ perfect |
| `video_nalu_5g_nr_reconstruction` | 1 000 bytes (H.265 NAL unit) | 5G NR BG1, Z=384, R=1/3 | 4 dB Eb/No | ✓ perfect |
| `wifi6_frame_reconstruction` | 63 bytes (Wi-Fi data frame) | Wi-Fi 6 MCS7 proxy, R=5/6 | 7 dB Eb/No | ✓ perfect |
| `sixg_embb_ultra_reliable` | 250 bytes (6G eMBB block) | 6G BG1, Z=96, R=0.89 | 12 dB Eb/No | ✓ perfect |

Sample output (`--nocapture`):
```
[audio_frame] Eb/No waterfall:
  Eb/No = 1.0 dB → BER = 0.1823 (no convergence)
  Eb/No = 3.0 dB → BER = 0.0412
  Eb/No = 5.0 dB → BER = 0.0000  ✓  CRC PASS — bit-exact reconstruction
```

---

## 5. Benchmarks

All numbers are produced by `bench/run_all.sh` — never hand-written. The RS results are protected by a byte-identical checksum gate that fails loudly if any implementation diverges.

### 5.1 Reed-Solomon encode throughput

10 data shards × N bytes, 4 parity shards.

| Implementation | 256 B | 1 KiB | 4 KiB | 16 KiB |
|---|---|---|---|---|
| **Rust `encode_with_avx2` (VPSHUFB)** | **~5 GiB/s** | **~10 GiB/s** | **~8.5 GiB/s** | **~10 GiB/s** |
| Rust `encode_with_tables_chunked` | ~740 MiB/s | ~810 | ~770 | ~740 |
| Rust `encode_into` | ~590 MiB/s | ~660 | ~690 | ~570 |
| C++ same-algorithm `-O3 -march=native` | within ~5% of scalar Rust | | | |
| Python same-algorithm | ~2.7 MiB/s | ~2.5 | ~2.4 | ~2.4 |

The AVX2 path uses VPSHUFB nibble-decomposition: `GF_mul(c, x) = lo_tbl[x & 0xF] ^ hi_tbl[x >> 4]`, processing 32 bytes/cycle.  Rust and C++ produce **byte-identical parity output** (checksum-gated).

### 5.2 QC-LDPC LOMS decode throughput

BG1, Z=384 ($N = 26{,}112$ variable nodes), 10 iterations, median of 200 reps.
Metric: $N \times \text{iters} / t_{\text{call}}$ (variable-node-iterations/s).

| Implementation | Throughput | Wall-clock / call |
|---|---|---|
| **Rust AVX2 (runtime-detected)** | **~119 Melem/s** | **~2.2 ms** |
| Rust multi-worker (4 threads, AVX2) | ~108 Melem/s × workers | ~2.4 ms/frame |
| Rust scalar | ~65 Melem/s | ~4.0 ms |
| C++ scalar `-O3 -march=native` | ~66 Melem/s | ~3.9 ms |

AVX2 speed-up breakdown (3 optimisations that together match C++ on the scalar path):
1. **Loop inversion** — Z-inner loop (384 independent iterations) gives LLVM a vectorisable axis.
2. **Conditional subtract** — replaces `% Z` (integer divide) with `if s >= Z { s - Z } else { s }`.
3. **Sign bit XOR** — `bits & 0x8000_0000` accumulation replaces float multiply.

### 5.3 BER Waterfall (5G NR BG1)

Simulation: BPSK AWGN, BG1 Z=384, $R \approx 0.324$, $\beta = 0.25$, 10 LOMS iterations.
Shannon limit for this rate: $E_b/N_0 \approx -1.1\ \text{dB}$.

| Eb/No (dB) | BER | BLER | Frames simulated |
|---|---|---|---|
| −1.0 | ~0.25 | 1.00 | 50 |
| 0.0 | ~0.21 | 1.00 | 50 |
| 0.5 | ~0.13 | 1.00 | 50 |
| 1.0 | ~7.6×10⁻⁴ | 0.18 | 283 |
| 1.5 | < 10⁻⁷ | < 2×10⁻³ | 500 (0 errors) |

The waterfall region (0.5 → 1.5 dB) represents **~6 dB coding gain** over uncoded BPSK — the property that makes LDPC essential to every modern wireless standard.

### 5.4 SPSC ring

```
push + pop: ~1.1 ns  (AtomicUsize head/tail, no syscall)
```

To reproduce all benchmarks:
```bash
cargo run --release --bin bench_export    # RS timing → bench/results/rust.json
bash bench/run_all.sh                     # all languages + checksum gate
cargo bench --bench fec_bench            # Criterion HTML report
```

---

## 6. Engineering Background

### 6.1 Channel Capacity and the Role of FEC

Shannon's noisy-channel coding theorem (1948) [1] establishes that, for a channel with capacity $C$ bits per channel use, there exist codes of rate $R < C$ that achieve arbitrarily small error probability as block length grows.

For the AWGN channel with bandwidth $W$ and signal-to-noise ratio $E_b/N_0$:

$$C = W \log_2\!\left(1 + \frac{E_b}{N_0} \cdot R\right)$$

Modern FEC codes (LDPC, Turbo, Polar) operate within a fraction of a dB of the Shannon limit.  5G NR mandates LDPC for all data channels because it offers the highest throughput at practical block lengths and parallelises naturally across SIMD units.

### 6.2 QC-LDPC and the 5G NR Base Graphs

An LDPC code is defined by a sparse parity-check matrix $H \in \{0,1\}^{M \times N}$.  The code $\mathcal{C}$ consists of all binary vectors $\mathbf{c}$ satisfying $H \mathbf{c} = \mathbf{0} \pmod{2}$.

5G NR uses **Quasi-Cyclic LDPC** [5] in which $H$ is built from $Z \times Z$ circulant sub-matrices — cyclic shifts $\mathbf{I}^{(p)}$ of the identity by $p$ positions.  The full matrix expands from a small **base graph** $H_b$:

$$H = \text{expand}(H_b,\, Z) \in \{0,1\}^{m_b Z \times n_b Z}$$

3GPP TS 38.212 specifies two base graphs [11]:

| Parameter | BG1 | BG2 |
|---|---|---|
| Base rows $m_b$ | 46 | 42 |
| Base columns $n_b$ | 68 | 52 |
| Non-null entries | 316 | 197 |
| Info columns $k_b$ | 22 | 10 |
| Max code rate | 8/9 | 2/3 |
| Lifted $N$ at $Z=384$ | 26 112 | — |
| Lifted $N$ at $Z=128$ | — | 6 656 |

BG1 is used for large, high-rate transport blocks; BG2 for smaller or lower-rate blocks.

### 6.3 Layered Offset Min-Sum (LOMS)

Layered (turbo-scheduled) decoding [9] processes one row-block at a time, immediately feeding updated LLRs into subsequent layers.  This halves the required iteration count versus flooding belief propagation.

Per-layer update equations:

$$Q_{mn}^{(t)} = L_n^{(t-1)} - R_{mn}^{(t-1)}$$

$$R_{mn}^{(t)} = \left(\prod_{n'} \text{sign}\,Q_{mn'}^{(t)}\right) \cdot \max\!\left(\min_{n'} \left|Q_{mn'}^{(t)}\right| - \beta,\; 0\right)$$

$$L_n^{(t)} = Q_{mn}^{(t)} + R_{mn}^{(t)}$$

This implementation uses $\beta = 0.25$, a standard operating point for 5G NR BG1 [8].  After each layer iteration, a **syndrome check** ($H \mathbf{c} = \mathbf{0}$) enables early termination without exhausting all iterations.

### 6.4 Algorithm-to-code mapping

| Mathematical object | Rust symbol | Notes |
|---|---|---|
| $H_b$ BG1 entries | `BG1_ENTRIES` (build-time const) | `[(u8,u8,[i16;8]); 316]` |
| $p_{ij}(Z)$ cyclic shift | `entry_col_shift()` | `v[iLS] % Z` |
| $L_n$ channel LLR buffer | `llr: &mut [f32]` | Flat, len $= n_b \cdot Z$ |
| $R_{mn}$ extrinsic buffer | `edge_r: &mut [f32]` | Flat, len $= E \cdot Z$ |
| Per-layer $Q_{mn}$ scratch | `layer_scratch: &mut [f32]` | Len $= d_m^{\max} \cdot Z$, reused per layer |
| LOMS offset $\beta$ | `offset_beta: f32` | 0.25 default |
| LOMS inner loop | `process_layer_all_z()` | Z-inner; vectorised by LLVM |

### 6.5 Wi-Fi 6 / 7 (802.11ax/be) LDPC parameters

802.11ax/be use the same LOMS algorithm with lifting sizes $Z \in \{27, 54, 81\}$ and $N = 24Z$ always. The `wifi` module provides the full MCS parameter table:

| Standard | MCS range | Modulation | Max rate |
|---|---|---|---|
| Wi-Fi 6 / 6E (802.11ax) | MCS 0–11 | BPSK → 1024-QAM | 5/6 |
| Wi-Fi 7 (802.11be) | MCS 0–13 | BPSK → **4096-QAM** | 5/6 |

### 6.6 6G NR Research Module (IMT-2030)

The `sixg` module captures confirmed 3GPP / ITU-R research directions for IMT-2030:

| Feature | Detail |
|---|---|
| Extended TB size | 8 Mbit research target vs 5G's 1.28 Mbit |
| Modulation depth | Up to 4096-QAM |
| Service profiles | eMBB, URLLC, mMTC, Integrated Sensing, AI/ML-assisted, Semantic |
| AMC ladder | SNR-threshold modulation selection from BPSK to 4096-QAM |

> **Scope note:** 3GPP Release 18–19 covers "5G Advanced." True 6G (IMT-2030) standardization begins ~2025–2028. Speculative features are clearly marked in the module.

### 6.7 Practical motivation: coding gain and media delivery

**LDPC vs repetition code (rate 1/3, $P_b = 10^{-5}$ target):**

| Code | $E_b/N_0$ required | Gap to Shannon limit |
|---|---|---|
| Uncoded BPSK | ≈ 9.8 dB | — |
| Repetition(3) — trivial | ≈ 9.0 dB | ≈ 10.1 dB |
| **LDPC BG1 — this library** | **≈ 0.7 dB** | **≈ 1.8 dB** |
| Shannon limit (rate 1/3) | −1.1 dB | 0 dB |

The **~8.3 dB coding gain** over the repetition code corresponds to 6.7× less transmit power for the same reliability.

**RS packet-loss recovery (RS(10, 4) on a 2 Mbps video stream at 1% packet loss):**

- No FEC: visible glitch every **0.5 s**.
- With RS(10, 4): expected failure once every **70 hours** — at 40% bandwidth overhead.

---

## 7. Design Notes

**Zero-allocation hot path.** The decode inner loops make no heap allocations.  All buffers (`llr`, `edge_r`, `layer_scratch`) are owned by the caller and reused across calls.  The 4.5 KiB per-layer scratch (`min1`, `min2`, `sign_xor`) is stack-allocated inside `decode_layered_offset_min_sum`.

**Flat memory layout.** The extrinsic buffer is indexed as `edge_r[global_edge * Z + z_pos]` — shape `(E, Z)`, row-major.  This keeps all $Z$ z-positions for one edge contiguous, enabling a future AVX2 kernel to load them as a single SIMD register.

**Syndrome-check early termination.** After each full layer sweep, `check_syndrome_f32` XORs hard decisions across all parity equations.  A clean syndrome exits before `max_iters`; `decode_layered_offset_min_sum` returns the number of iterations actually used.

**SPSC ring.** `SpscRing<T, N>` uses `AtomicUsize` head/tail with `Ordering::Acquire`/`Ordering::Release` pairs — the standard wait-free SPSC pattern.  No OS primitives; safe from pinned producer/consumer threads with no syscall overhead.

**Thread affinity.** Worker threads call `crate::affinity::pin_to_core(wi)` on startup; the call is silently ignored if the `affinity` feature or OS support is absent.  When affinity binds each decoder to a physical core, the working set stays in that core's L1/L2 cache, eliminating cross-socket LLR buffer migration.

**AVX2 kernel.** `src/simd_avx2.rs` is wired via `is_x86_feature_detected!("avx2")` at runtime — no compile-time feature flag needed.
- Pass 1: 8-wide `_mm256_blendv_ps` branchless min1/min2 update; `_mm256_xor_si256` sign accumulation.
- Pass 2: `_mm256_cmp_ps` + `_mm256_blendv_ps` exclusive-min select; `_mm256_or_ps` sign application.

---

## 8. Real-World Applications

Between them, the codecs in this library appear in virtually every digital communication and storage system built after 2000.

### 8.1 — 5G NR smartphone downlink

Every phone call or data session on a 5G network. Your phone's baseband chip (e.g. Qualcomm X65 modem) receives OFDM symbols, equalises the channel, then feeds soft LLR values directly into a QC-LDPC LOMS decoder — exactly what `QcLdpcDecoder::decode_layered_offset_min_sum` implements.

**Code path:** [src/qc_ldpc.rs](src/qc_ldpc.rs) → [src/simd_avx2.rs](src/simd_avx2.rs)

### 8.2 — Starlink satellite internet

Starlink uses DVB-S2X, which mandates QC-LDPC codes.  At ~500 Mbit/s downlink, the decoder must handle one frame every few milliseconds.  Multi-worker pipelines like `LdpcPipeline::with_workers` are how production hardware achieves this: frames dispatch across decoder cores so throughput is sustained even as individual decode calls take ~2 ms.

**Code path:** [src/ldpc_pipeline.rs](src/ldpc_pipeline.rs)

### 8.3 — QR code scanning

QR codes embed RS erasure codes over GF(256) with the same primitive polynomial (0x11D) as this library.  A QR code at error-correction level "H" recovers even when 30% of printed modules are obscured.

**Code path:** [src/reed_solomon.rs](src/reed_solomon.rs)

### 8.4 — Cloud storage (AWS S3, Ceph, HDFS)

AWS S3 Standard uses approximately RS(14, 4) — 14 data + 4 parity fragments across availability zones, tolerating any 4 simultaneous failures at 28% storage overhead versus 200% for 3× replication.

**Code path:** `ReedSolomon::new(data_shards, parity_shards)` → `encode_with_avx2` → `decode_erasure`

### 8.5 — Deep-space probe communication (Mars rovers, Voyager)

One-way light-travel time makes ARQ impossible; FEC is the only option.  CCSDS has mandated RS outer codes since the 1980s.  Every scientific image from Mars reaches Earth because of the same GF(256) arithmetic in [src/reed_solomon.rs](src/reed_solomon.rs).

### 8.6 — HD video streaming over RTP/UDP

Live broadcast (SMPTE 2022-1/2, RIST), video conferencing (Zoom, WebRTC), and studio audio (AES67 / Ravenna) all apply RS packet-level FEC.  With RS(10, 4) at 1% packet loss, visible glitches drop from every 0.5 s to once every 70 hours (§6.7).

### 8.7 — 400G Ethernet and submarine cables

400G Ethernet (IEEE 802.3bs) mandates RS(544, 514) over GF(1024).  The AVX2 `gf256_muladd_avx2` VPSHUFB kernel is the same computational pattern used in line-rate hardware accelerators — the maths scales to larger fields.

### 8.8 — Wi-Fi 6 / 802.11ax indoor access points

802.11ax requires LDPC at all MCS indices above 5.  At MCS 11 (1024-QAM, rate 5/6), one 80 MHz spatial stream carries ~1.2 Gbps.  The LDPC encoder/decoder is the throughput bottleneck — which is why Wi-Fi chipmakers (Broadcom, Qualcomm Atheros, Intel) deploy AVX2/NEON kernels essentially identical to [src/simd_avx2.rs](src/simd_avx2.rs).

### 8.9 — Digital Audio Broadcasting (DAB+)

DAB+ uses a Viterbi-decoded convolutional inner code + RS(120, 110) outer code.  The RS outer decode corrects residual errors that survive the Viterbi step — exactly the concatenated FEC pattern that preceded LDPC in 2G/3G systems.

**Code paths:** [src/reed_solomon.rs](src/reed_solomon.rs) (outer), [src/viterbi.rs](src/viterbi.rs) (inner)

### 8.10 — Blu-ray and M-DISC archival storage

Blu-ray uses RS Product-Code (RS-PC) for burst error correction.  M-DISC archives use RS variants over GF(256).  RS is exceptionally suited to burst errors: a 100-byte scratch counts as only one RS symbol error if it falls within one RS block.

---

## 9. References

[1] C. E. Shannon, "A Mathematical Theory of Communication," *Bell System Technical Journal*, vol. 27, pp. 379–423, July–Oct. 1948.

[2] R. G. Gallager, *Low-Density Parity-Check Codes*, MIT Press, 1963.

[3] D. J. C. MacKay and R. M. Neal, "Near Shannon Limit Performance of Low Density Parity Check Codes," *Electronics Letters*, vol. 32, no. 18, pp. 1645–1646, Aug. 1996.

[4] R. M. Tanner, "A Recursive Approach to Low Complexity Codes," *IEEE Trans. Inf. Theory*, vol. 27, no. 5, pp. 533–547, Sept. 1981.

[5] M. P. C. Fossorier, "Quasi-Cyclic Low-Density Parity-Check Codes From Circulant Permutation Matrices," *IEEE Trans. Inf. Theory*, vol. 50, no. 8, pp. 1788–1793, Aug. 2004.

[6] N. Wiberg, *Codes and Decoding on General Graphs*, PhD thesis, Linköping University, 1996.

[7] J. Chen, A. Dholakia, E. Eleftheriou, M. P. C. Fossorier, and X.-Y. Hu, "Reduced-Complexity Decoding of LDPC Codes," *IEEE Trans. Commun.*, vol. 53, no. 8, pp. 1288–1299, Aug. 2005.

[8] T. Richardson and S. Urbanke, *Modern Coding Theory*, Cambridge University Press, 2008.

[9] D. E. Hocevar, "A Reduced Complexity Decoder Architecture via Layered Decoding of LDPC Codes," *Proc. IEEE SIPS*, pp. 107–112, Oct. 2004.

[10] I. S. Reed and G. Solomon, "Polynomial Codes over Certain Finite Fields," *J. SIAM*, vol. 8, no. 2, pp. 300–304, 1960.

[11] 3GPP, "NR; Multiplexing and channel coding," TS 38.212 V16.15.0, Dec. 2025.

---

## 10. Learning Path

### Suggested 3-month roadmap (EEE BSc background)

```
Month 1 — Theory
  Week 1-2  MacKay Ch. 1 (probability) + Ch. 16 (factor graphs) + Ch. 47 (LDPC)
            [free at inference.org.uk]
  Week 3    Work through a 4×8 LDPC decode by hand. Write it in Python.
  Week 4    Ryan & Lin Ch. 5 (LDPC) + Ch. 13 (QC-LDPC)
            Read 3GPP TS 38.212 §5.3.2 (base graphs) — just the tables.

Month 2 — Implementation
  Week 5    Read src/qc_ldpc.rs top-to-bottom; trace one call with Z=2.
  Week 6    Implement GF(256) multiply from scratch in 50 lines of Rust.
  Week 7    Read src/simd_avx2.rs alongside the Intel Intrinsics Guide.
  Week 8    Modify the scalar fallback to print intermediate LLRs; verify
            against the sum-product equations from Month 1.

Month 3 — Performance and systems
  Week 9    Read Agner Fog "Optimizing Software in C++" (free) — Ch. 7 & 13.
  Week 10   Implement your own SPSC ring in 50 lines; run the same tests.
  Week 11   BER simulation: add uncoded BPSK to the waterfall; observe coding gain.
  Week 12   Read the AFF3CT paper (Cassagne et al. 2019); compare to ldpc_pipeline.rs.
```

### Your BSc connections to this codebase

| What you know | How it connects here |
|---|---|
| **Hamming(7,4)** | LDPC is the same idea: $H\mathbf{c} = 0$. Hamming decodes algebraically; LDPC decodes iteratively. |
| **BER curves** | The BER waterfall in §5.3 *is* that curve for a practical code near the Shannon limit. |
| **QAM decision regions** | The LLR input to the LDPC decoder is $\Lambda = \ln P(\text{bit}=0|y)/P(\text{bit}=1|y)$ — the soft probability at each decision boundary. |
| **FFT for OFDM** | 5G NR places LDPC *after* the OFDM FFT demodulator. This library is the last step in the chain. |
| **Quantisation / ADC** | Hardware LDPC decoders quantise LLRs to 6–8 bits; `src/quantize.rs` is that step. |

---

## 11. Similar Projects

| Project | Language | Domain | Notes |
|---|---|---|---|
| [AFF3CT](https://github.com/aff3ct/aff3ct) | C++17 | LDPC, Turbo, Polar, RS | Full FEC framework; AVX2/AVX-512; primary C++ reference. Paper: Cassagne et al., *SoftwareX* 2019. |
| [OpenAirInterface](https://gitlab.eurecom.fr/oai/openairinterface5g) | C | 5G NR PHY L1 | Open-source gNB/UE; LDPC from 3GPP BG1/BG2. |
| [srsRAN Project](https://github.com/srsran/srsRAN_Project) | C++17 | 5G NR PHY | Production-quality open-source gNB; LDPC with AVX2 paths. |
| [rav1e](https://github.com/xiph/rav1e) | Rust | AV1 video codec | Demonstrates Rust competing with C++ on DSP kernels. |
| [ldpc-codes](https://crates.io/crates/ldpc-codes) | Rust | Generic LDPC | Pure-Rust LDPC; different design goals (no 5G BG, no SIMD). |

---

*The Highcharts dashboard (`bench/dashboard/`) is used under the [Highcharts non-commercial license](https://www.highcharts.com/blog/products/highcharts/). Credits attribution is preserved in all rendered charts.*

*Licensed under the [MIT License](LICENSE) — © 2025 Thomas Glezer.*
