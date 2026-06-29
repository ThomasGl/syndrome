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
//! $$C = \begin{cases} 1 & \text{if } B \le K_{cb} \\ \lceil B / (K_{cb} - 24) \rceil & \text{otherwise} \end{cases}$$
//!
//! $L = 24$ if $C > 1$, else $L = 0$ (CRC-24B is only added when segmenting).
//!
//! $B' = B + C \cdot L$ (total bits including per-CB CRCs).
//!
//! $K' = B' / C$ (info bits per code block, always integer by construction).
//!
//! Choose minimum valid 3GPP $Z$ from Table 5.3.2-1 such that $K_b \cdot Z \ge K'$,
//! where $K_b = 22$ (BG1) or $\{10, 9, 8, 6\}$ (BG2, depending on $B$).
//!
//! $K = K_b \cdot Z$ = full systematic length (includes filler bits).
//!
//! Filler bits = $K - K'$ (appended as zeros before encode, as $+\infty$ LLRs before decode).

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
    /// Info bits per code block $K'$ (includes CRC-24B if $C > 1$).
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
/// Returns [`FecError::InvalidParam`] if `a` is 0 or the resulting $K'$ has
/// no valid 3GPP lifting size.
///
/// # Examples
///
/// ```
/// use glezer_rsv::segmentation::compute_segmentation;
/// use glezer_rsv::qc_ldpc::BaseGraph;
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
/// ```
pub fn compute_segmentation(a: usize, target_rate: f32) -> Result<SegmentationParams, FecError> {
    if a == 0 {
        return Err(FecError::InvalidParam("transport block size must be > 0"));
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

    // B = TB + CRC-24A.
    let b = a + 24;

    let (c, l) = if b <= k_cb {
        (1usize, 0usize)
    } else {
        let c = (b + (k_cb - 24) - 1) / (k_cb - 24); // ceil(B / (Kcb-24))
        (c, 24usize)
    };

    let b_prime = b + c * l;
    // K' = B'/C, must be integer.
    assert_eq!(b_prime % c, 0, "B' must be divisible by C");
    let k_prime = b_prime / c;

    // Kb depends on BG and (for BG2) on B (§5.2.2):
    //   BG1: Kb = 22
    //   BG2: Kb = 10 if B > 640; 9 if B > 560; 8 if B > 192; else 6
    let k_b: usize = match bg {
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
    };

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
    // b_prime worth of systematic bits spread across C code blocks.
    let b_prime = p.k_prime * p.c;
    // Source is B = b_prime - C*24 (remove per-CB CRCs we will add back)
    // but the TB already came with CRC-24A, total B bits.
    let b = b_prime - if p.has_cb_crc { p.c * 24 } else { 0 };
    if tb_with_crc.len() < b {
        return Err(FecError::BufferTooSmall {
            required: b,
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
        let end = start + payload_per_cb;
        let src = &tb_with_crc[start..end.min(tb_with_crc.len())];
        let mut cb = src.to_vec();
        // Pad with zeros if the last CB is short (only possible when B not
        // divisible by payload_per_cb — spec pads with zeros).
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
        // k_prime * c must equal b_prime = b + c*24.
        let b = 10000 + 24;
        let b_prime = b + p.c * 24;
        assert_eq!(p.k_prime * p.c, b_prime);
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
