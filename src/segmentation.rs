//! Code block segmentation for 3GPP TS 38.212 §5.2.2.
//!
//! After CRC-24A is attached to the transport block, this module determines:
//!
//! 1. Which base graph (BG1 or BG2) to use (§7.2.2).
//! 2. How many code blocks $C$ to segment into, and the per-block info size $K'$.
//! 3. The 3GPP lifting size $Z$ (minimum valid $Z$ such that $K_b \cdot Z \ge K'$).
//! 4. The number of filler bits (known-zero, treated as $+\infty$ LLR on decode).
//!
//! # Mathematical Summary
//!
//! Let $A$ = TB size in bits.
//!
//! $B = A + 24$ (TB + CRC-24A length, $L=24$).
//!
//! $K_{cb}$ = maximum code block size = 8448 (BG1) or 3840 (BG2).
//!
//! $$C = \begin{cases} 1 & \text{if } B \le K_{cb} \\\\ \lceil B / (K_{cb} - 24) \rceil & \text{otherwise} \end{cases}$$
//!
//! $L = 24$ if $C > 1$, else $L = 0$ (CRC-24B is only added when segmenting).
//!
//! $B' = B + C \cdot L$ (total bits including per-CB CRCs).
//!
//! ## $K'$ and the non-divisibility case
//!
//! In general $C$ does **not** divide $B'$ evenly. The literal 3GPP text
//! handles this by splitting the $C$ code blocks into two groups: $C_+ =
//! B' \bmod C$ blocks of size $K'\_+ = \lceil B'/C \rceil$ and the remaining
//! $C - C\_+$ blocks of size $K'\_- = \lfloor B'/C \rfloor$, so that the sizes
//! sum to exactly $B'$.
//!
//! This module deliberately uses a **simpler, documented approximation**:
//! every code block gets the *same* size,
//! $$K' = \lceil B' / C \rceil,$$
//! and any bits beyond the real payload (at most $C-1$ bits, concentrated in
//! the tail of the last code block) are zero-padded by [`segment`] exactly
//! like ordinary filler bits — the LDPC decoder already treats them as
//! `+∞`-LLR knowns, so the padding is invisible to the decoded payload. This
//! trades a handful of wasted bits (bounded by $C-1 < K_{cb}$, i.e.
//! negligible next to $B'$) for a single-size code block path, matching the
//! polarization-weight approximation the polar decoder uses for the same
//! kind of "exact table vs. documented closed-form substitute" trade-off
//! (see `crate::polar::frozen_mask`'s doc comment). Crucially, **every**
//! `a` now produces a valid segmentation — the previous implementation's
//! `assert_eq!(b_prime % c, 0, ...)` panicked whenever $B'$ did not divide
//! $C$ evenly, which is the majority of multi-code-block transport block
//! sizes.
//!
//! Choose minimum valid 3GPP $Z$ from Table 5.3.2-1 such that $K_b \cdot Z \ge K'$,
//! where $K_b = 22$ (BG1) or $\lbrace 10, 9, 8, 6\rbrace$ (BG2, depending on $B$).
//!
//! $K = K_b \cdot Z$ = full systematic length (includes filler bits).
//!
//! Filler bits = $K - K'$ (appended as zeros before encode, as $+\infty$ LLRs before decode).
//!
//! # Bounds
//!
//! `a` (transport block size) is rejected outright above
//! [`MAX_TB_SIZE_BITS`] (the largest transport block size reachable under
//! the 3GPP-defined maximum data rate, TS 38.214 Annex). This keeps every
//! subsequent arithmetic expression (`a + 24`, `b + c * 24`, …) far below
//! `usize::MAX`, so no individual step needs a checked/overflowing variant.

use crate::alloc_prelude::*;
use crate::error::FecError;
use crate::qc_ldpc::BaseGraph;

// ---------------------------------------------------------------------------
// 3GPP lifting size table (TS 38.212 Table 5.3.2-1)
// ---------------------------------------------------------------------------

/// All valid 3GPP lifting sizes, organised into the 8 sets (iLS index 0..=7).
/// The sets are listed in Table 5.3.2-1; we flatten them into a sorted list
/// and search linearly.
const VALID_Z: &[usize] = &[
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32, 36, 40, 44,
    48, 52, 56, 60, 64, 72, 80, 88, 96, 104, 112, 120, 128, 144, 160, 176, 192, 208, 224, 240, 256,
    288, 320, 352, 384,
];

/// Largest transport block size (in bits) accepted by [`compute_segmentation`].
///
/// This is the maximum DL-SCH/UL-SCH transport block size reachable under
/// the 3GPP-defined maximum data rate (TS 38.214 Annex, max PRBs / max MIMO
/// layers / 256-QAM / numerology 3), so any caller-supplied `a` above this
/// is unambiguously out of spec. Rejecting it up front means every
/// subsequent addition/multiplication in [`compute_segmentation`] operates
/// on values bounded well below `usize::MAX`, so plain (non-checked)
/// arithmetic is safe throughout the rest of the function.
pub const MAX_TB_SIZE_BITS: usize = 1_277_992;

/// Find the smallest valid 3GPP lifting size $Z$ such that $k_b \cdot Z \ge k\_prime$.
///
/// Returns `Err` if no valid $Z$ exists (i.e. $K'$ is too large even for $Z=384$).
fn min_valid_z(k_prime: usize, k_b: usize) -> Result<usize, FecError> {
    for &z in VALID_Z {
        if k_b * z >= k_prime {
            return Ok(z);
        }
    }
    Err(FecError::InvalidParam(
        "K' too large: no valid 3GPP lifting size exists",
    ))
}

// ---------------------------------------------------------------------------
// Base graph selection (TS 38.212 §7.2.2)
// ---------------------------------------------------------------------------

/// Select the QC-LDPC base graph for a transport block of `a` info bits at
/// code rate `r`.
///
/// Rules (TS 38.212 §7.2.2):
/// - BG2 if $A \le 292$, or ($A \le 3824$ and $R \le 0.67$), or $R \le 0.25$.
/// - BG1 otherwise.
///
/// # Arguments
///
/// * `a` - Transport block size in bits (before CRC).
/// * `r` - Target code rate (bits/channel_bit, e.g. `22.0/68.0` for BG1 R≈0.32).
///
/// # Returns
///
/// [`BaseGraph::Bg1`] or [`BaseGraph::Bg2`].
pub fn select_base_graph(a: usize, r: f32) -> BaseGraph {
    if a <= 292 || (a <= 3824 && r <= 0.67) || r <= 0.25 {
        BaseGraph::Bg2
    } else {
        BaseGraph::Bg1
    }
}

// ---------------------------------------------------------------------------
// Segmentation parameters
// ---------------------------------------------------------------------------

/// All parameters derived by the 3GPP code block segmentation procedure
/// (TS 38.212 §5.2.2).
///
/// These drive both the LDPC encoder (for building the circulant graph at
/// lifting size $Z$) and the rate matcher (which reads the full code block
/// size $N = n_b \cdot Z$).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentationParams {
    /// Selected base graph.
    pub bg: BaseGraph,
    /// 3GPP lifting size $Z$ (TS 38.212 Table 5.3.2-1).
    pub z: usize,
    /// Number of code blocks $C$ (1 if no segmentation).
    pub c: usize,
    /// $B = A + 24$ — real TB + CRC-24A bit length that [`segment`] must
    /// read from `tb_with_crc` (i.e. *before* per-CB CRC-24B and any
    /// $K' = \lceil B'/C \rceil$ rounding padding). Not necessarily equal to
    /// `k_prime * c` (see the module-level doc's "K' and the non-divisibility
    /// case" section) — this field is what `segment` uses instead of
    /// re-deriving a (possibly larger, due to rounding) source length from
    /// `k_prime`.
    pub b: usize,
    /// Info bits per code block $K'$ (includes CRC-24B if $C > 1$).  Equal to
    /// $\lceil B'/C \rceil$ where $B' = B + C \cdot 24$; see the module-level
    /// doc for why this is a uniform per-block size rather than the literal
    /// spec's two-size split.
    pub k_prime: usize,
    /// Full systematic length $K = K_b \cdot Z$ (includes filler bits).
    pub k: usize,
    /// $N = n_b \cdot Z$ — full encoded length per code block.
    pub n: usize,
    /// Number of filler bits ($K - K'$) appended at positions $K'..K$.
    pub n_filler: usize,
    /// Whether a CRC-24B is appended to each code block ($C > 1$).
    pub has_cb_crc: bool,
    /// $K_b$ used in lifting size selection.
    pub k_b: usize,
}

/// $K_b$ for lifting-size selection, per 3GPP TS 38.212 §5.2.2.
///
/// $K_b$ is the number of systematic base-graph columns the lifting size has
/// to cover, and it is *not* the base graph's full systematic column count —
/// for BG2 the spec steps it down for short transport blocks, so a small block
/// gets a larger $Z$ than its information length alone would suggest. The
/// difference between this and the encoder's fixed column count becomes filler
/// bits.
///
/// | Base graph | $B$ | $K_b$ |
/// |---|---|---|
/// | BG1 | any | 22 |
/// | BG2 | $> 640$ | 10 |
/// | BG2 | $> 560$ | 9 |
/// | BG2 | $> 192$ | 8 |
/// | BG2 | otherwise | 6 |
///
/// Note every BG2 comparison is strict: at $B = 640$ exactly, $K_b$ is 9, not
/// 10. This is a transcribed spec table feeding a value that shifts $Z$, $K$
/// and the filler count, so it is a separate function purely so the ladder can
/// be tested at each boundary from both sides — `cargo mutants` showed that
/// through [`compute_segmentation`] alone, every one of those comparisons
/// could be loosened to `>=` without failing a test, because $K_b$ is not
/// among the fields [`SegmentationParams`] reports.
///
/// # Arguments
///
/// * `bg` - Selected base graph.
/// * `b`  - $B = A + 24$, the transport block plus its CRC-24A.
///
/// # Returns
///
/// $K_b \in \lbrace 6, 8, 9, 10, 22 \rbrace$.
fn lifting_selection_k_b(bg: BaseGraph, b: usize) -> usize {
    match bg {
        BaseGraph::Bg1 => 22,
        BaseGraph::Bg2 => {
            if b > 640 {
                10
            } else if b > 560 {
                9
            } else if b > 192 {
                8
            } else {
                6
            }
        }
    }
}

/// Compute segmentation parameters for a transport block.
///
/// # Arguments
///
/// * `a`            - Transport block size in bits (before CRC-24A attachment).
/// * `target_rate`  - Target code rate, used only for base graph selection.
///
/// # Returns
///
/// [`SegmentationParams`] on success.
///
/// # Errors
///
/// Returns [`FecError::InvalidParam`] if `a` is 0, `a` exceeds
/// [`MAX_TB_SIZE_BITS`], or the resulting $K'$ has no valid 3GPP lifting
/// size.
///
/// # Examples
///
/// ```
/// use syndrome::segmentation::compute_segmentation;
/// use syndrome::qc_ldpc::BaseGraph;
///
/// // Small TB → BG2, no segmentation.
/// let p = compute_segmentation(100, 0.5).unwrap();
/// assert_eq!(p.c, 1);
/// assert_eq!(p.bg, BaseGraph::Bg2);
///
/// // Large TB → BG1, multiple code blocks.
/// let p = compute_segmentation(10000, 0.5).unwrap();
/// assert_eq!(p.bg, BaseGraph::Bg1);
/// assert!(p.c > 1);
///
/// // Awkward TB size where B' does not divide C evenly — used to panic,
/// // now returns a valid, self-consistent segmentation (see the
/// // module-level doc's "K' and the non-divisibility case" section).
/// let p = compute_segmentation(8425, 0.5).unwrap();
/// assert!(p.k_prime * p.c >= p.b + p.c * 24);
/// ```
pub fn compute_segmentation(a: usize, target_rate: f32) -> Result<SegmentationParams, FecError> {
    if a == 0 {
        return Err(FecError::InvalidParam("transport block size must be > 0"));
    }
    if a > MAX_TB_SIZE_BITS {
        return Err(FecError::InvalidParam(
            "transport block size exceeds the 3GPP maximum (1,277,992 bits)",
        ));
    }

    let bg = select_base_graph(a, target_rate);
    let k_cb = match bg {
        BaseGraph::Bg1 => 8448,
        BaseGraph::Bg2 => 3840,
    };
    let n_b = match bg {
        BaseGraph::Bg1 => 66,
        BaseGraph::Bg2 => 50,
    };

    // B = TB + CRC-24A. Safe: `a <= MAX_TB_SIZE_BITS`, checked above.
    let b = a + 24;

    let (c, l) = if b <= k_cb {
        (1usize, 0usize)
    } else {
        let c = b.div_ceil(k_cb - 24); // ceil(B / (Kcb-24))
        (c, 24usize)
    };

    let b_prime = b + c * l;
    // K' = ceil(B'/C): a uniform per-block size (see the module-level doc's
    // "K' and the non-divisibility case" section for why this is used
    // instead of requiring exact divisibility, and how the resulting slack
    // is zero-padded by `segment`).
    let k_prime = b_prime.div_ceil(c);

    let k_b = lifting_selection_k_b(bg, b);

    let z = min_valid_z(k_prime, k_b)?;

    // k_b_encoder is the full systematic column count of the base graph (always fixed).
    // BG1 has 22 systematic columns (68 total - 46 check = 22).
    // BG2 has 10 systematic columns (52 total - 42 check = 10).
    // The LDPC encoder always uses k_b_encoder * Z systematic bits; the delta
    // (k_b_encoder - k_b_seg) * Z bits are additional filler zeros.
    let k_b_encoder: usize = match bg {
        BaseGraph::Bg1 => 22,
        BaseGraph::Bg2 => 10,
    };
    let k = k_b_encoder * z; // full systematic length seen by the encoder
    let n = n_b * z;
    let n_filler = k - k_prime; // encoder-perspective filler count

    Ok(SegmentationParams {
        bg,
        z,
        c,
        b,
        k_prime,
        k,
        n,
        n_filler,
        has_cb_crc: l > 0,
        k_b: k_b_encoder,
    })
}

// ---------------------------------------------------------------------------
// Segmentation (bit-level)
// ---------------------------------------------------------------------------

/// Segment a transport block (already including CRC-24A) into per-code-block
/// bit strings, each with CRC-24B attached and filler bits appended.
///
/// The returned vectors have length `k_prime` (info bits) per block.  Filler
/// bits ($K - K'$ zeros) must be prepended/appended by the caller before
/// passing to the LDPC encoder — they are **not** included here because they
/// are implicit (always zero).
///
/// # Arguments
///
/// * `tb_with_crc` - Full transport block bits **including** the CRC-24A
///   parity bits appended at the end.  Length must be `a + 24`.
/// * `p`           - [`SegmentationParams`] from [`compute_segmentation`].
///
/// # Returns
///
/// `Vec<Vec<u8>>` — one `Vec<u8>` per code block, each of length `k_prime`.
/// The last `24` bits of each vector (when `p.has_cb_crc` is true) are the
/// CRC-24B parity.
///
/// # Errors
///
/// Returns [`FecError::BufferTooSmall`] if `tb_with_crc` is shorter than
/// `a + 24` bits.
pub fn segment(tb_with_crc: &[u8], p: &SegmentationParams) -> Result<Vec<Vec<u8>>, FecError> {
    // `p.b` is the real TB+CRC-24A bit length that must be sliced out of
    // `tb_with_crc` — NOT re-derived from `k_prime * c`, since with the
    // uniform K' = ceil(B'/C) rounding (see the module-level doc), `k_prime *
    // c` can be a few bits larger than the real B'/B (the slack is made up by
    // zero-padding below, exactly like ordinary filler bits).
    if tb_with_crc.len() < p.b {
        return Err(FecError::BufferTooSmall {
            required: p.b,
            provided: tb_with_crc.len(),
        });
    }

    use crate::crc::{Crc24, CrcKind};
    let cb_crc = Crc24::new(CrcKind::Crc24B);

    // Bits per CB before CRC-24B.
    let payload_per_cb = if p.has_cb_crc {
        p.k_prime - 24
    } else {
        p.k_prime
    };

    let mut blocks: Vec<Vec<u8>> = Vec::with_capacity(p.c);
    for ci in 0..p.c {
        let start = ci * payload_per_cb;
        // Clip to `p.b`, not `tb_with_crc.len()`: any bits beyond the real
        // payload (whether because this is the short tail of the last CB, or
        // because `payload_per_cb * c` slightly overshoots `p.b` due to the
        // K' rounding) must be zero-padded, not read from the caller's buffer
        // (which may be longer than `p.b`, e.g. exact-multiple-of-8 byte
        // buffers).
        let end = (start + payload_per_cb).min(p.b);
        let src: &[u8] = if start < end {
            &tb_with_crc[start..end]
        } else {
            &[]
        };
        let mut cb = src.to_vec();
        // Zero-pad the remainder: covers both the traditional "last CB is
        // short" case and the slack introduced by K' = ceil(B'/C) rounding.
        cb.resize(payload_per_cb, 0);
        if p.has_cb_crc {
            cb_crc.attach(&mut cb);
        }
        blocks.push(cb);
    }

    Ok(blocks)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Every $K_b$ threshold from TS 38.212 §5.2.2, tested on both sides.
    ///
    /// $K_b$ feeds lifting-size selection, so getting a comparison wrong
    /// shifts $Z$, $K$ and the filler count for a *band* of transport block
    /// sizes while leaving every size outside that band correct. `cargo
    /// mutants` found that all four BG2 boundary comparisons could be loosened
    /// to `>=` (or `==`) without failing anything: the round-trip tests sample
    /// transport block sizes that are never *at* a threshold, which is
    /// precisely where a transcription slip in a spec table lands.
    ///
    /// Every comparison in the ladder is strict, so each threshold value
    /// itself belongs to the band *below* it.
    #[test]
    fn kb_thresholds_match_the_spec_on_both_sides() {
        // (B, expected Kb) — each threshold and the value one bit above it.
        const BG2_CASES: [(usize, usize); 9] = [
            (0, 6),
            (192, 6),  // at the threshold: `>` means 192 is still 6
            (193, 8),  // one above
            (560, 8),  // at the second threshold
            (561, 9),  // one above
            (640, 9),  // at the third threshold
            (641, 10), // one above
            (3824, 10),
            (usize::MAX, 10),
        ];
        for (b, expected) in BG2_CASES {
            assert_eq!(
                lifting_selection_k_b(BaseGraph::Bg2, b),
                expected,
                "BG2 at B = {b}: TS 38.212 §5.2.2 gives Kb = {expected}"
            );
        }

        // BG1 has no ladder: 22 at every B, including where BG2's thresholds
        // sit, so a stray BG2 comparison leaking into the BG1 arm shows here.
        for b in [0usize, 192, 193, 560, 641, 8448, usize::MAX] {
            assert_eq!(lifting_selection_k_b(BaseGraph::Bg1, b), 22);
        }
    }

    #[test]
    fn bg_selection_boundary_values() {
        // A ≤ 292 → BG2 regardless of rate.
        assert_eq!(select_base_graph(292, 0.9), BaseGraph::Bg2);
        // A = 293, R = 0.5 ≤ 0.67, A ≤ 3824 → BG2.
        assert_eq!(select_base_graph(293, 0.5), BaseGraph::Bg2);
        // A = 293, R = 0.68 → BG1 (A > 292, R > 0.67, A > 3824 condition false).
        assert_eq!(select_base_graph(293, 0.68), BaseGraph::Bg1);
        // A > 3824, R = 0.5 → BG1 (neither BG2 condition applies).
        assert_eq!(select_base_graph(4000, 0.5), BaseGraph::Bg1);
        // R ≤ 0.25 → BG2 always.
        assert_eq!(select_base_graph(9999, 0.25), BaseGraph::Bg2);
    }

    #[test]
    fn single_block_no_segmentation() {
        // A = 100 → B=124 ≤ 3840, BG2, C=1.
        let p = compute_segmentation(100, 0.5).unwrap();
        assert_eq!(p.c, 1);
        assert!(!p.has_cb_crc);
        assert_eq!(p.n_filler, p.k - p.k_prime);
    }

    #[test]
    fn large_tb_bg1_segments() {
        // A = 10000 → BG1, B=10024 > 8448 → C > 1.
        let p = compute_segmentation(10000, 0.5).unwrap();
        assert_eq!(p.bg, BaseGraph::Bg1);
        assert!(p.c > 1);
        assert!(p.has_cb_crc);
        // k_prime * c must be >= b_prime = b + c*24 (equality only when B'
        // happens to divide C evenly; see the module doc's "K' and the
        // non-divisibility case" section — K' = ceil(B'/C) may overshoot by
        // up to C-1 bits, made up by zero-padding in `segment`).
        assert_eq!(p.b, 10000 + 24);
        let b_prime = p.b + p.c * 24;
        assert!(p.k_prime * p.c >= b_prime);
        assert!(p.k_prime * p.c - b_prime < p.c);
    }

    /// FINDING 9 regression guard (was `finding_segmentation_bg_c_not_dividing_bprime_panics`
    /// in tests/robustness.rs, `#[should_panic]`): `compute_segmentation`
    /// used to panic via `assert_eq!(b_prime % c, 0, ...)` for any transport
    /// block size where B' does not divide C evenly — the majority of
    /// multi-code-block sizes. It must now return a valid, self-consistent
    /// segmentation instead.
    #[test]
    fn segmentation_bg1_c2_non_divisible_bprime_no_longer_panics() {
        let p = compute_segmentation(8425, 0.5).unwrap();
        assert_eq!(p.bg, BaseGraph::Bg1);
        assert_eq!(p.c, 2);
        let b_prime = p.b + p.c * 24;
        // Sum of per-block payloads must cover the real B' (with at most
        // C-1 bits of rounding slack, zero-padded).
        assert!(p.k_prime * p.c >= b_prime);
        assert!(p.k_prime * p.c - b_prime < p.c);
        // Filler bit count is internally consistent (K = Kb*Z >= K').
        assert_eq!(p.n_filler, p.k - p.k_prime);
        assert!(p.k >= p.k_prime);
    }

    /// Sweep of "awkward" transport block sizes (deliberately including
    /// values that hit every residue of `B' mod C`, not just the lucky
    /// evenly-divisible ones) verifying `compute_segmentation` never panics
    /// and always produces a self-consistent, valid segmentation.
    #[test]
    fn segmentation_sweep_awkward_sizes_self_consistent() {
        for a in [
            1usize,
            100,
            292,
            293,
            3800,
            3824,
            3825,
            4000,
            8424,
            8425,
            8426,
            8447,
            8448,
            8449,
            10000,
            10001,
            12345,
            16896,
            16897,
            20000,
            50000,
            100_000,
            500_000,
            1_000_000,
            MAX_TB_SIZE_BITS,
        ] {
            for &rate in &[0.25f32, 0.5, 0.67, 0.9] {
                let p = compute_segmentation(a, rate).unwrap_or_else(|e| {
                    panic!("compute_segmentation({a}, {rate}) unexpectedly rejected: {e:?}")
                });
                // L = 24 only when segmenting (C > 1 / has_cb_crc); L = 0 for
                // a single code block.
                let l = if p.has_cb_crc { 24 } else { 0 };
                let b_prime = p.b + p.c * l;
                assert!(
                    p.k_prime * p.c >= b_prime,
                    "a={a} rate={rate}: k_prime*c must cover B'"
                );
                assert!(
                    p.k_prime * p.c - b_prime < p.c,
                    "a={a} rate={rate}: rounding slack must stay below C bits"
                );
                assert!(p.k >= p.k_prime, "a={a} rate={rate}: K must be >= K'");
                assert_eq!(p.n_filler, p.k - p.k_prime);
                assert_eq!(
                    p.n,
                    match p.bg {
                        BaseGraph::Bg1 => 66,
                        BaseGraph::Bg2 => 50,
                    } * p.z
                );
            }
        }
    }

    #[test]
    fn compute_segmentation_rejects_absurd_tb_size() {
        // FINDING 6/2 regression guard (was
        // `finding_segmentation_huge_tb_size_overflows`, `#[should_panic]`):
        // `b = a + 24` used to overflow for `a` near `usize::MAX`. Now
        // rejected outright via the explicit `MAX_TB_SIZE_BITS` bound.
        assert!(compute_segmentation(usize::MAX - 5, 0.5).is_err());
        assert!(compute_segmentation(usize::MAX, 0.5).is_err());
        assert!(compute_segmentation(MAX_TB_SIZE_BITS + 1, 0.5).is_err());
        assert!(compute_segmentation(MAX_TB_SIZE_BITS, 0.5).is_ok());
    }

    /// End-to-end round-trip through `segment()` at the exact TB size from
    /// finding 9's reproducer (8425 bits, rate 0.5, BG1, C=2, non-divisible
    /// B'): the real payload must be recovered byte-for-byte from the
    /// segmented code blocks, proving the `p.b`-based slicing in `segment`
    /// (not the old, now-larger-than-real `k_prime * c` derivation) is
    /// correct.
    #[test]
    fn segment_round_trip_awkward_tb_size_8425() {
        use crate::crc::{Crc24, CrcKind};
        let a = 8425usize;
        let tb_crc = Crc24::new(CrcKind::Crc24A);
        let mut tb: Vec<u8> = (0..a).map(|i| ((i * 7 + 3) % 5 < 2) as u8).collect();
        tb_crc.attach(&mut tb); // tb.len() == a + 24 == p.b
        let p = compute_segmentation(a, 0.5).unwrap();
        assert_eq!(tb.len(), p.b);

        let blocks = segment(&tb, &p).unwrap();
        assert_eq!(blocks.len(), p.c);

        // Reconstruct the payload (info bits only, CRC-24B stripped) and
        // compare against the real TB+CRC-24A bits it was sliced from.
        let payload_per_cb = if p.has_cb_crc {
            p.k_prime - 24
        } else {
            p.k_prime
        };
        let mut recovered: Vec<u8> = Vec::with_capacity(payload_per_cb * p.c);
        for block in &blocks {
            assert_eq!(block.len(), p.k_prime);
            recovered.extend_from_slice(&block[..payload_per_cb]);
        }
        // The first `p.b` recovered bits must exactly match the source TB;
        // anything beyond that is documented zero-padding.
        assert_eq!(&recovered[..p.b], &tb[..]);
        assert!(recovered[p.b..].iter().all(|&b| b == 0));
    }

    #[test]
    fn k_contains_k_prime_plus_filler() {
        let p = compute_segmentation(500, 0.5).unwrap();
        assert_eq!(p.k, p.k_prime + p.n_filler);
    }

    #[test]
    fn n_is_nb_times_z() {
        let p_bg1 = compute_segmentation(5000, 0.8).unwrap();
        assert_eq!(p_bg1.bg, BaseGraph::Bg1);
        assert_eq!(p_bg1.n, 66 * p_bg1.z);

        let p_bg2 = compute_segmentation(200, 0.5).unwrap();
        assert_eq!(p_bg2.bg, BaseGraph::Bg2);
        assert_eq!(p_bg2.n, 50 * p_bg2.z);
    }

    #[test]
    fn segment_single_block() {
        let a = 100usize;
        use crate::crc::{Crc24, CrcKind};
        let tb_crc = Crc24::new(CrcKind::Crc24A);
        let mut tb: Vec<u8> = (0..a as u8).map(|i| i & 1).collect();
        tb_crc.attach(&mut tb);
        let p = compute_segmentation(a, 0.5).unwrap();
        let blocks = segment(&tb, &p).unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].len(), p.k_prime);
    }
}
