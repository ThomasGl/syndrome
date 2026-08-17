//! Equivalence of the fixed-point QC-LDPC kernels.
//!
//! The crate has two implementations of the fixed-point layered offset
//! min-sum update: a portable scalar reference
//! ([`QcLdpcDecoder::decode_layered_offset_min_sum_i8_scalar`]) and an AVX2
//! kernel that processes 32 z-positions per 256-bit register
//! ([`QcLdpcDecoder::decode_layered_offset_min_sum_i8`] on an AVX2 host).
//! CLAUDE.md requires the scalar path to stay a tested reference the
//! vectorized path is proven equal to, and here that requirement is
//! unusually strong: every operation in the fixed-point path is integer, so
//! the two kernels must agree **bit-for-bit** — not merely to within a
//! tolerance, the way two floating-point kernels free to reassociate
//! additions would.
//!
//! # What each test targets
//!
//! The vectorized kernel has three places a lane-boundary mistake can hide,
//! and the lifting sizes below are chosen to hit all of them:
//!
//! * Passes 1 and 2 step 32 z-positions at a time and finish with a **scalar
//!   tail** of $Z \bmod 32$ elements; the Q-build steps 16 at a time, so it
//!   has its own tail of $Z \bmod 16$.
//! * The cyclic shift splits pass 2 and the Q-build into two contiguous
//!   **runs**, each chunked and tailed separately, so the run boundary lands
//!   mid-chunk for most shifts even when $Z$ divides evenly.
//! * $Z \bmod 16 \neq 0$ additionally makes the Q-build's saturating pack
//!   land on a partial store.
//!
//! One consequence is easy to get wrong and worth stating: at $Z < 32$ the
//! pass-1 and pass-2 vector bodies never execute at all — `Z & !31` is zero,
//! so the whole layer runs through the scalar tails and the test proves
//! nothing about the vector code. Covering the vector body therefore needs
//! $Z \ge 32$, and covering body *and* tail together needs $Z \ge 32$ with
//! $Z \bmod 32 \neq 0$. The sizes used here are 7 and 13 (tails only), 44,
//! 52, 60 and 88 ($Z \bmod 32 \in \{12, 20, 28, 24\}$ and $Z \bmod 16 \in
//! \{12, 4, 12, 8\}$: body plus both tails), 96, 128 and 384 (exact
//! multiples, every chunk full), plus the 802.11 matrices at $Z \in \{27,
//! 54, 81\}$, whose shift distribution and row degrees differ completely
//! from the 3GPP graphs.
//!
//! That distinction was found by mutation rather than by reading: seeding a
//! deliberate off-by-one into the AVX2 magnitude computation left the
//! small-$Z$ cases green.

use syndrome::channel_sim::AwgnChannel;
use syndrome::qc_ldpc::{BaseGraph, QcLdpcDecoder, QcLdpcEncoder};
use syndrome::quantize::{APP_CLAMP_I8, QuantParams, quantize_llr_i16};
use syndrome::wifi_ldpc_tables::{wifi_ldpc_decoder, wifi_ldpc_encoder};

/// Iteration budget. High enough that the decoder runs many layers and the
/// posterior genuinely accumulates, so a lane mix-up cannot cancel out.
const ITERS: usize = 12;

/// Deterministic bit source, so a failure is reproducible from the seed
/// alone.
fn random_bits(seed: u64, n: usize) -> Vec<u8> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            // The low bit of a raw xorshift word is weakly mixed; take a
            // middle bit instead.
            ((s >> 27) & 1) as u8
        })
        .collect()
}

/// Result of one fixed-point decode: everything the two kernels must agree on.
struct DecodeState {
    iters: usize,
    app: Vec<i16>,
    edge_r: Vec<i8>,
    hard: Vec<u8>,
}

/// Run both fixed-point kernels on identical input and return their states.
///
/// The channel LLRs are generated once and quantized once, so the only
/// difference between the two runs is which kernel executes.
fn decode_both(
    enc: &QcLdpcEncoder,
    dec: &QcLdpcDecoder,
    quant: QuantParams,
    ebno_db: f32,
    seed: u64,
) -> (DecodeState, DecodeState) {
    let k = enc.info_bit_count();
    let n = enc.codeword_bit_count();
    let rate = k as f32 / n as f32;

    let info = random_bits(seed ^ 0xA5A5_A5A5, k);
    let mut codeword = vec![0u8; n];
    enc.encode(&info, &mut codeword).unwrap();

    let mut ch = AwgnChannel::new(ebno_db, rate, seed);
    let llr = ch.transmit(&codeword);
    let mut app0 = vec![0i16; n];
    quantize_llr_i16(&llr, &mut app0, quant.scale);

    let run = |scalar: bool| {
        let mut app = app0.clone();
        let mut edge_r = vec![0i8; dec.required_edge_buffer()];
        let mut scratch = vec![0i8; dec.required_layer_buffer()];
        let mut hard = vec![0u8; n];
        let iters = if scalar {
            dec.decode_layered_offset_min_sum_i8_scalar(
                &mut app,
                &mut edge_r,
                &mut scratch,
                &mut hard,
                ITERS,
                quant,
            )
        } else {
            dec.decode_layered_offset_min_sum_i8(
                &mut app,
                &mut edge_r,
                &mut scratch,
                &mut hard,
                ITERS,
                quant,
            )
        }
        .unwrap();
        DecodeState {
            iters,
            app,
            edge_r,
            hard,
        }
    };

    (run(true), run(false))
}

/// Assert that two kernel runs are indistinguishable, reporting the first
/// divergent index so a lane-boundary bug is immediately locatable.
fn assert_identical(label: &str, scalar: &DecodeState, auto: &DecodeState) {
    assert_eq!(
        scalar.iters, auto.iters,
        "{label}: iteration counts differ (scalar {}, dispatched {}) — the two \
         kernels disagreed on the early-termination syndrome check",
        scalar.iters, auto.iters
    );
    if let Some(i) = (0..scalar.app.len()).find(|&i| scalar.app[i] != auto.app[i]) {
        panic!(
            "{label}: posterior differs at variable {i} (scalar {}, dispatched {}); \
             {} of {} positions differ",
            scalar.app[i],
            auto.app[i],
            (0..scalar.app.len())
                .filter(|&j| scalar.app[j] != auto.app[j])
                .count(),
            scalar.app.len(),
        );
    }
    if let Some(i) = (0..scalar.edge_r.len()).find(|&i| scalar.edge_r[i] != auto.edge_r[i]) {
        panic!(
            "{label}: check-to-variable message differs at edge slot {i} \
             (scalar {}, dispatched {})",
            scalar.edge_r[i], auto.edge_r[i],
        );
    }
    assert_eq!(scalar.hard, auto.hard, "{label}: hard decisions differ");
}

/// Bit-for-bit agreement across both 3GPP base graphs, a spread of lifting
/// sizes that exercises every tail path, and SNRs from "the decoder is
/// struggling" to "it converges in two iterations".
#[test]
fn scalar_and_dispatched_kernels_agree_on_3gpp_graphs() {
    let quant = QuantParams::default();
    for (bg, z) in [
        (BaseGraph::Bg1, 7usize),
        (BaseGraph::Bg1, 44),
        (BaseGraph::Bg1, 96),
        (BaseGraph::Bg2, 13),
        (BaseGraph::Bg2, 52),
        (BaseGraph::Bg2, 128),
    ] {
        let enc = QcLdpcEncoder::new(bg, z).unwrap();
        let dec = QcLdpcDecoder::with_lifting_size(bg, z, 0.5).unwrap();
        for &ebno in &[0.0f32, 1.5, 4.0] {
            for seed in 0..4u64 {
                let s = 0x0001_0000u64 ^ (z as u64) << 8 ^ seed;
                let (scalar, auto) = decode_both(&enc, &dec, quant, ebno, s);
                assert_identical(
                    &format!("{bg:?} Z={z} Eb/N0={ebno} seed={s:#x}"),
                    &scalar,
                    &auto,
                );
            }
        }
    }
}

/// The same agreement at the largest production lifting size, where every
/// vector chunk is full and the posterior has the most room to drift apart
/// if the two kernels ever disagreed on saturation.
#[test]
fn scalar_and_dispatched_kernels_agree_at_z384() {
    let quant = QuantParams::default();
    let enc = QcLdpcEncoder::new(BaseGraph::Bg1, 384).unwrap();
    let dec = QcLdpcDecoder::with_lifting_size(BaseGraph::Bg1, 384, 0.5).unwrap();
    for seed in 0..2u64 {
        let (scalar, auto) = decode_both(&enc, &dec, quant, 1.0, 0x0384_0000 ^ seed);
        assert_identical(&format!("BG1 Z=384 seed={seed}"), &scalar, &auto);
    }
}

/// The 802.11 matrices reach the decoder through
/// [`syndrome::qc_ldpc::QcLdpcParams::from_raw_edges`] and have a different
/// shift distribution and row-degree profile from the 3GPP graphs, so they
/// place the pass-2 run boundary in different places.
#[test]
fn scalar_and_dispatched_kernels_agree_on_wifi_matrices() {
    let quant = QuantParams::default();
    for (z, rn, rd) in [(27usize, 1usize, 2usize), (54, 2, 3), (81, 5, 6)] {
        let enc = wifi_ldpc_encoder(z, rn, rd).unwrap();
        let dec = wifi_ldpc_decoder(z, rn, rd, 0.5).unwrap();
        for &ebno in &[1.0f32, 3.0] {
            let s = 0x01F1_0000u64 ^ (z as u64) << 4 ^ rd as u64;
            let (scalar, auto) = decode_both(&enc, &dec, quant, ebno, s);
            assert_identical(
                &format!("Wi-Fi Z={z} R={rn}/{rd} Eb/N0={ebno}"),
                &scalar,
                &auto,
            );
        }
    }
}

/// A narrow posterior accumulator drives the clamp constantly, which is the
/// one code path where the AVX2 kernel's `_mm256_adds_epi16` plus min/max
/// clamp could diverge from the scalar `saturating_add(..).clamp(..)`.
#[test]
fn scalar_and_dispatched_kernels_agree_with_a_clamped_posterior() {
    let quant = QuantParams::default().with_app_clamp(APP_CLAMP_I8);
    for (bg, z) in [(BaseGraph::Bg1, 44usize), (BaseGraph::Bg2, 88)] {
        let enc = QcLdpcEncoder::new(bg, z).unwrap();
        let dec = QcLdpcDecoder::with_lifting_size(bg, z, 0.5).unwrap();
        for seed in 0..3u64 {
            let (scalar, auto) = decode_both(&enc, &dec, quant, 1.0, 0x0C1A_0000 ^ seed);
            assert_identical(&format!("{bg:?} Z={z} clamped seed={seed}"), &scalar, &auto);
        }
    }
}

/// $\beta = 0$ removes the offset correction, so the check magnitude is no
/// longer floored by the `max(_, 0)` and every lane takes the untrimmed
/// value. It is the boundary case for the AVX2 kernel's
/// `max_epi8(sub_epi8(min_excl, beta), 0)`.
#[test]
fn scalar_and_dispatched_kernels_agree_with_zero_offset() {
    let quant = QuantParams::default();
    let enc = QcLdpcEncoder::new(BaseGraph::Bg1, 60).unwrap();
    let dec = QcLdpcDecoder::with_lifting_size(BaseGraph::Bg1, 60, 0.0).unwrap();
    let (scalar, auto) = decode_both(&enc, &dec, quant, 1.0, 0xBE7A_0000);
    assert_identical("BG1 Z=60 beta=0", &scalar, &auto);
}

/// A large $\beta$ pins most check magnitudes at zero after the offset
/// subtraction, exercising the opposite side of the same clamp.
#[test]
fn scalar_and_dispatched_kernels_agree_with_saturating_offset() {
    let quant = QuantParams::default();
    let enc = QcLdpcEncoder::new(BaseGraph::Bg2, 104).unwrap();
    // beta = 16.0 at scale 8 quantizes to beta_q = 127, the maximum.
    let dec = QcLdpcDecoder::with_lifting_size(BaseGraph::Bg2, 104, 16.0).unwrap();
    let (scalar, auto) = decode_both(&enc, &dec, quant, 2.0, 0xBE7B_0000);
    assert_identical("BG2 Z=104 beta_q=127", &scalar, &auto);
}

/// The 5G wrapper adds filler-bit and puncture initialisation on top of the
/// core loop; both kernels must see the same initial posterior and stay in
/// agreement through it.
#[test]
fn scalar_and_dispatched_kernels_agree_through_the_5g_wrapper() {
    let quant = QuantParams::default();
    let z = 60usize;
    let dec = QcLdpcDecoder::with_lifting_size(BaseGraph::Bg1, z, 0.5).unwrap();
    let n = dec.variable_node_count();
    let llr: Vec<f32> = (0..n)
        .map(|i| if i % 5 == 0 { -0.75 } else { 1.25 })
        .collect();
    let mut app0 = vec![0i16; n];
    quantize_llr_i16(&llr, &mut app0, quant.scale);

    let n_filler = 3 * z / 2;
    let run = |scalar: bool| {
        let mut app = app0.clone();
        let mut edge_r = vec![0i8; dec.required_edge_buffer()];
        let mut scratch = vec![0i8; dec.required_layer_buffer()];
        let mut hard = vec![0u8; n];
        if scalar {
            // Reproduce decode_5g_i8's initialisation, then force the scalar
            // kernel: there is no scalar-forcing 5G entry point, and adding
            // one purely for this test would widen the public API.
            let k_b = 22; // BG1 systematic block columns
            let k = k_b * z;
            for slot in &mut app[k - n_filler..k] {
                *slot = quant.app_clamp;
            }
            for slot in &mut app[..2 * z] {
                *slot = 0;
            }
            dec.decode_layered_offset_min_sum_i8_scalar(
                &mut app,
                &mut edge_r,
                &mut scratch,
                &mut hard,
                ITERS,
                quant,
            )
        } else {
            dec.decode_5g_i8(
                &mut app,
                n_filler,
                &mut edge_r,
                &mut scratch,
                &mut hard,
                ITERS,
                quant,
            )
        }
        .unwrap();
        DecodeState {
            iters: 0,
            app,
            edge_r,
            hard,
        }
    };
    let scalar = run(true);
    let auto = run(false);
    assert_identical("BG1 Z=60 decode_5g_i8", &scalar, &auto);
}

/// Sanity floor: at a high SNR the fixed-point decoder must recover the
/// transmitted information bits exactly. Kernel equivalence alone would be
/// satisfied by two identically-broken kernels, so this pins the path to
/// actually decoding.
#[test]
fn fixed_point_path_recovers_the_codeword_at_high_snr() {
    let quant = QuantParams::default();
    for (bg, z) in [(BaseGraph::Bg1, 96usize), (BaseGraph::Bg2, 128)] {
        let enc = QcLdpcEncoder::new(bg, z).unwrap();
        let dec = QcLdpcDecoder::with_lifting_size(bg, z, 0.5).unwrap();
        let k = enc.info_bit_count();
        let n = enc.codeword_bit_count();
        let rate = k as f32 / n as f32;

        let info = random_bits(0x5EED, k);
        let mut codeword = vec![0u8; n];
        enc.encode(&info, &mut codeword).unwrap();
        let mut ch = AwgnChannel::new(6.0, rate, 0x600D_0000);
        let llr = ch.transmit(&codeword);
        let mut app = vec![0i16; n];
        quantize_llr_i16(&llr, &mut app, quant.scale);

        let mut edge_r = vec![0i8; dec.required_edge_buffer()];
        let mut scratch = vec![0i8; dec.required_layer_buffer()];
        let mut hard = vec![0u8; n];
        dec.decode_layered_offset_min_sum_i8(
            &mut app,
            &mut edge_r,
            &mut scratch,
            &mut hard,
            ITERS,
            quant,
        )
        .unwrap();
        assert_eq!(
            &hard[..k],
            &info[..],
            "{bg:?} Z={z}: fixed-point decode did not recover the information bits \
             at Eb/N0 = 6 dB"
        );
    }
}
