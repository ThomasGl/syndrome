//! Polar codes for 5G NR control channels (TS 38.212 §5.3.1).
//!
//! 5G NR uses polar codes for:
//! - PDCCH (Downlink Control Information)  — CRC-24C or CRC-11
//! - PBCH  (Physical Broadcast Channel)    — CRC-24C
//! - PUCCH/PUSCH UCI                       — CRC-6 or CRC-11
//!
//! # Encoding (§5.3.1)
//!
//! The generator matrix is $G_N = F^{\otimes n}$ where $F = \begin{bmatrix}1 &
//! 0 \\\\ 1 & 1\end{bmatrix}$ and $n = \log_2 N$.  The encoding operation is:
//!
//! $$x = u \cdot G_N, \quad u_i = 0 \text{ for frozen bits}$$
//!
//! Efficiently computed via the butterfly (bit-reversal permuted) polar
//! transform in $O(N \log N)$.
//!
//! # Decoding
//!
//! ## Successive Cancellation (SC) — $O(N \log N)$
//!
//! Recursive computation of $f$ and $g$ LLR combining rules:
//! $$f(a, b) = \text{sgn}(a) \cdot \text{sgn}(b) \cdot \min(|a|, |b|)$$
//! $$g(a, b, \hat{u}) = b + (1 - 2\hat{u}) \cdot a$$
//!
//! The encoder recursion is $x_1 = (u_1 \oplus u_2) \cdot G_{N/2}$,
//! $x_2 = u_2 \cdot G_{N/2}$, so by GF(2) linearity
//! $x_1 \oplus x_2 = u_1 \cdot G_{N/2}$: the left ($f$) subtree's decoded
//! output is $\hat u_1$ *re-encoded* through the sub-code, not $\hat u_1$
//! itself. The $g$ rule needs the actual value combined into $x_1$, so
//! every internal decode-tree node must re-run `polar_transform` over its
//! left child's decoded output to recover this **partial sum** before
//! computing the right child's LLRs — using the raw decoded bits directly
//! silently reconstructs the wrong codeword for any information pattern
//! where a sub-block's decoded output differs from its own re-encoding
//! (invisible for the all-zero and single-flag vectors, since re-encoding
//! low-weight vectors is close to a no-op, but wrong in general).
//!
//! ## Successive Cancellation List (SCL) — $O(L \cdot N \log N)$
//!
//! Maintains $L$ candidate paths, walking the same $f$/$g$/partial-sum
//! decode tree as SC but forking every path into a `0` and `1` candidate at
//! each information-bit leaf, then pruning back to $L$ survivors by path
//! metric.
//! CRC-aided SCL (CA-SCL, the 5G NR configuration) selects the path that
//! passes the CRC from the surviving list.
//!
//! ### SCL decode-tree shape
//!
//! SC and SCL walk the *same* binary recursion tree over `llr`; SCL just
//! carries a whole list of candidate paths through it instead of one. Each
//! internal node splits its `length`-wide LLR block into a left half (fed
//! through `f_kernel`) and a right half (fed through `g_kernel`, once the
//! left half's hard decision is known); a leaf is a single bit — frozen
//! (forced to 0) or an information bit (forks the path list in two):
//!
//! ```text
//!                 [ length=N, level=0 ]
//!                /                      \
//!         f: left half              g: right half (needs left's
//!         (length=N/2)               partial sum first)
//!              ...                          ...
//!               \                          /
//!            [ length=1 ]              [ length=1 ]
//!          frozen: fix 0           info: fork into {0,1},
//!          (no fork)                sort by path metric,
//!                                   keep best `list_size`
//! ```
//!
//! # Input contract: LLRs must be finite
//!
//! [`PolarDecoder::decode_sc`] and [`PolarDecoder::decode_scl`] both reject
//! any `llr` slice containing `NaN` or `±infinity` with
//! [`FecError::InvalidParam`] before doing any path-metric arithmetic. A
//! `NaN` LLR is not meaningful soft information (there is no sensible "how
//! confident is the channel" reading for it), and letting one reach the
//! path-metric sort would either panic (`f32::partial_cmp` returns `None`
//! for `NaN`) or, if silently tolerated via `total_cmp`, produce a
//! decode result with no relationship to the actual transmitted codeword --
//! worse than an error, because it looks like a normal answer.

use crate::crc::{Crc24, CrcKind};
use crate::error::FecError;

// ---------------------------------------------------------------------------
// Frozen bit reliability sequence (TS 38.212 §5.3.1, Table 5.3.1.2-1)
// ---------------------------------------------------------------------------

// The complete 5G NR reliability sequence Q_Nmax for N_max = 1024, i.e. the
// full 1024-entry Table 5.3.1.2-1 of TS 38.212.
// Bit index Q[i] is the i-th most reliable position in a rate-1 polar code of length N_max.
// For shorter codes, the frozen set is the complement of the K most reliable positions ∩ {0..N-1}
// (equivalently: filter this sequence to entries < N, preserving order -- per TS 38.212 §5.3.1.2,
// that filtered sequence *is* Q_0^{N-1}, the reliability ordering for length N).
// Indices are in the **channel-order** (before bit-reversal permutation).
//
// Cross-validated against two independent, credible sources, byte-for-byte identical across
// all 1024 entries:
//   - robmaunder/polar-3gpp-matlab, `components/get_3GPP_sequence_pattern.m`
//     (https://github.com/robmaunder/polar-3gpp-matlab)
//   - OpenAirInterface5G, `openair1/PHY/CODING/nrPolar_tools/nr_polar_sequence_pattern.c`,
//     the `Q_0_Nminus1_10` table (https://github.com/OPENAIRINTERFACE/openairinterface5g)
const RELIABILITY_SEQ: &[u16] = &[
    0, 1, 2, 4, 8, 16, 32, 3, 5, 64, 9, 6, 17, 10, 18, 128, 12, 33, 65, 20, 256, 34, 24, 36, 7,
    129, 66, 512, 11, 40, 68, 130, 19, 13, 48, 14, 72, 257, 21, 132, 35, 258, 26, 513, 80, 37, 25,
    22, 136, 260, 264, 38, 514, 96, 67, 41, 144, 28, 69, 42, 516, 49, 74, 272, 160, 520, 288, 528,
    192, 544, 70, 44, 131, 81, 50, 73, 15, 320, 133, 52, 23, 134, 384, 76, 137, 82, 56, 27, 97, 39,
    259, 84, 138, 145, 261, 29, 43, 98, 515, 88, 140, 30, 146, 71, 262, 265, 161, 576, 45, 100,
    640, 51, 148, 46, 75, 266, 273, 517, 104, 162, 53, 193, 152, 77, 164, 768, 268, 274, 518, 54,
    83, 57, 521, 112, 135, 78, 289, 194, 85, 276, 522, 58, 168, 139, 99, 86, 60, 280, 89, 290, 529,
    524, 196, 141, 101, 147, 176, 142, 530, 321, 31, 200, 90, 545, 292, 322, 532, 263, 149, 102,
    105, 304, 296, 163, 92, 47, 267, 385, 546, 324, 208, 386, 150, 153, 165, 106, 55, 328, 536,
    577, 548, 113, 154, 79, 269, 108, 578, 224, 166, 519, 552, 195, 270, 641, 523, 275, 580, 291,
    59, 169, 560, 114, 277, 156, 87, 197, 116, 170, 61, 531, 525, 642, 281, 278, 526, 177, 293,
    388, 91, 584, 769, 198, 172, 120, 201, 336, 62, 282, 143, 103, 178, 294, 93, 644, 202, 592,
    323, 392, 297, 770, 107, 180, 151, 209, 284, 648, 94, 204, 298, 400, 608, 352, 325, 533, 155,
    210, 305, 547, 300, 109, 184, 534, 537, 115, 167, 225, 326, 306, 772, 157, 656, 329, 110, 117,
    212, 171, 776, 330, 226, 549, 538, 387, 308, 216, 416, 271, 279, 158, 337, 550, 672, 118, 332,
    579, 540, 389, 173, 121, 553, 199, 784, 179, 228, 338, 312, 704, 390, 174, 554, 581, 393, 283,
    122, 448, 353, 561, 203, 63, 340, 394, 527, 582, 556, 181, 295, 285, 232, 124, 205, 182, 643,
    562, 286, 585, 299, 354, 211, 401, 185, 396, 344, 586, 645, 593, 535, 240, 206, 95, 327, 564,
    800, 402, 356, 307, 301, 417, 213, 568, 832, 588, 186, 646, 404, 227, 896, 594, 418, 302, 649,
    771, 360, 539, 111, 331, 214, 309, 188, 449, 217, 408, 609, 596, 551, 650, 229, 159, 420, 310,
    541, 773, 610, 657, 333, 119, 600, 339, 218, 368, 652, 230, 391, 313, 450, 542, 334, 233, 555,
    774, 175, 123, 658, 612, 341, 777, 220, 314, 424, 395, 673, 583, 355, 287, 183, 234, 125, 557,
    660, 616, 342, 316, 241, 778, 563, 345, 452, 397, 403, 207, 674, 558, 785, 432, 357, 187, 236,
    664, 624, 587, 780, 705, 126, 242, 565, 398, 346, 456, 358, 405, 303, 569, 244, 595, 189, 566,
    676, 361, 706, 589, 215, 786, 647, 348, 419, 406, 464, 680, 801, 362, 590, 409, 570, 788, 597,
    572, 219, 311, 708, 598, 601, 651, 421, 792, 802, 611, 602, 410, 231, 688, 653, 248, 369, 190,
    364, 654, 659, 335, 480, 315, 221, 370, 613, 422, 425, 451, 614, 543, 235, 412, 343, 372, 775,
    317, 222, 426, 453, 237, 559, 833, 804, 712, 834, 661, 808, 779, 617, 604, 433, 720, 816, 836,
    347, 897, 243, 662, 454, 318, 675, 618, 898, 781, 376, 428, 665, 736, 567, 840, 625, 238, 359,
    457, 399, 787, 591, 678, 434, 677, 349, 245, 458, 666, 620, 363, 127, 191, 782, 407, 436, 626,
    571, 465, 681, 246, 707, 350, 599, 668, 790, 460, 249, 682, 573, 411, 803, 789, 709, 365, 440,
    628, 689, 374, 423, 466, 793, 250, 371, 481, 574, 413, 603, 366, 468, 655, 900, 805, 615, 684,
    710, 429, 794, 252, 373, 605, 848, 690, 713, 632, 482, 806, 427, 904, 414, 223, 663, 692, 835,
    619, 472, 455, 796, 809, 714, 721, 837, 716, 864, 810, 606, 912, 722, 696, 377, 435, 817, 319,
    621, 812, 484, 430, 838, 667, 488, 239, 378, 459, 622, 627, 437, 380, 818, 461, 496, 669, 679,
    724, 841, 629, 351, 467, 438, 737, 251, 462, 442, 441, 469, 247, 683, 842, 738, 899, 670, 783,
    849, 820, 728, 928, 791, 367, 901, 630, 685, 844, 633, 711, 253, 691, 824, 902, 686, 740, 850,
    375, 444, 470, 483, 415, 485, 905, 795, 473, 634, 744, 852, 960, 865, 693, 797, 906, 715, 807,
    474, 636, 694, 254, 717, 575, 913, 798, 811, 379, 697, 431, 607, 489, 866, 723, 486, 908, 718,
    813, 476, 856, 839, 725, 698, 914, 752, 868, 819, 814, 439, 929, 490, 623, 671, 739, 916, 463,
    843, 381, 497, 930, 821, 726, 961, 872, 492, 631, 729, 700, 443, 741, 845, 920, 382, 822, 851,
    730, 498, 880, 742, 445, 471, 635, 932, 687, 903, 825, 500, 846, 745, 826, 732, 446, 962, 936,
    475, 853, 867, 637, 907, 487, 695, 746, 828, 753, 854, 857, 504, 799, 255, 964, 909, 719, 477,
    915, 638, 748, 944, 869, 491, 699, 754, 858, 478, 968, 383, 910, 815, 976, 870, 917, 727, 493,
    873, 701, 931, 756, 860, 499, 731, 823, 922, 874, 918, 502, 933, 743, 760, 881, 494, 702, 921,
    501, 876, 847, 992, 447, 733, 827, 934, 882, 937, 963, 747, 505, 855, 924, 734, 829, 965, 938,
    884, 506, 749, 945, 966, 755, 859, 940, 830, 911, 871, 639, 888, 479, 946, 750, 969, 508, 861,
    757, 970, 919, 875, 862, 758, 948, 977, 923, 972, 761, 877, 952, 495, 703, 935, 978, 883, 762,
    503, 925, 878, 735, 993, 885, 939, 994, 980, 926, 764, 941, 967, 886, 831, 947, 507, 889, 984,
    751, 942, 996, 971, 890, 509, 949, 973, 1000, 892, 950, 863, 759, 1008, 510, 979, 953, 763,
    974, 954, 879, 981, 982, 927, 995, 765, 956, 887, 985, 997, 986, 943, 891, 998, 766, 511, 988,
    1001, 951, 1002, 893, 975, 894, 1009, 955, 1004, 1010, 957, 983, 958, 987, 1012, 999, 1016,
    767, 989, 1003, 990, 1005, 959, 1011, 1013, 895, 1006, 1014, 1017, 1018, 991, 1020, 1007, 1015,
    1019, 1021, 1022, 1023,
];

/// Largest `N` for which `RELIABILITY_SEQ` has full coverage. This spans the
/// entire 3GPP-defined range (`Q_Nmax` is only defined up to `Nmax = 1024`),
/// so `frozen_mask`'s polarization-weight fallback (see its doc comment) is
/// never exercised for any block length this crate documents supporting.
const RELIABILITY_TABLE_MAX_N: usize = 1024;

/// Compute the frozen bit mask for a polar code of length `n_polar` and
/// `k_info` information bits.
///
/// Returns a `Vec<bool>` of length `n_polar` where `true` = frozen bit.
///
/// `RELIABILITY_SEQ` embeds the complete 3GPP `Q_Nmax` sequence (Table
/// 5.3.1.2-1 of TS 38.212, all 1024 entries), so every block length this
/// module supports -- including PBCH (`N=512`) and PDCCH up to the 5G NR
/// maximum `N=1024` -- is served directly from the real table, not an
/// approximation. Given `N`, the construction filters `RELIABILITY_SEQ` to
/// the entries `< N` (which, per TS 38.212 §5.3.1.2, *is* `Q_0^{N-1}`, the
/// reliability ordering for length `N`, preserving relative order) and takes
/// the `K` most reliable of those as the information set.
///
/// For `N` above the 3GPP-defined `Nmax = 1024` -- outside the range the
/// standard's `Q_Nmax` sequence covers, and larger than any 5G NR polar
/// code this module documents supporting -- `frozen_mask` falls back to a
/// **polarization-weight (PW)** heuristic:
///
/// $$W(i) = \sum_{b : \text{bit } b \text{ of } i \text{ is set}} 2^{0.25 \thinspace b}$$
///
/// ranking indices by ascending $W$ (ties broken by index). This is a
/// standard closed-form approximation to the true (Bhattacharyya-parameter)
/// reliability ordering, used only in that out-of-spec regime where no
/// standardized table exists to consult.
fn frozen_mask(n_polar: usize, k_info: usize) -> Vec<bool> {
    debug_assert!(n_polar.is_power_of_two());
    debug_assert!(k_info <= n_polar);
    let mut is_frozen = vec![true; n_polar];

    if n_polar <= RELIABILITY_TABLE_MAX_N {
        // Collect candidate positions from the reliability sequence that
        // fall within [0, n_polar).
        let candidates: Vec<u16> = RELIABILITY_SEQ
            .iter()
            .copied()
            .filter(|&idx| (idx as usize) < n_polar)
            .collect();
        // Take the K most reliable positions as information bits.
        // candidates is in reliability order (last entry = most reliable).
        // We need the K highest-reliability, i.e. the LAST K entries.
        let info_count = k_info.min(candidates.len());
        let info_start = candidates.len().saturating_sub(info_count);
        for &pos in &candidates[info_start..] {
            is_frozen[pos as usize] = false;
        }
    } else {
        const BETA: f64 = 0.25;
        let nbits = n_polar.trailing_zeros();
        let pw_weight = |idx: u32| -> f64 {
            (0..nbits)
                .filter(|b| (idx >> b) & 1 == 1)
                .map(|b| 2f64.powf(BETA * b as f64))
                .sum()
        };
        let mut order: Vec<u32> = (0..n_polar as u32).collect();
        order.sort_by(|&a, &b| {
            pw_weight(a)
                .partial_cmp(&pw_weight(b))
                .expect("PW weights are always finite")
                .then(a.cmp(&b))
        });
        let info_count = k_info.min(order.len());
        let info_start = order.len().saturating_sub(info_count);
        for &pos in &order[info_start..] {
            is_frozen[pos as usize] = false;
        }
    }

    is_frozen
}

// ---------------------------------------------------------------------------
// Polar transform (encoder)
// ---------------------------------------------------------------------------

/// Apply the polar encoding transform $x = u \cdot G_N$ via the butterfly.
///
/// The transform is performed in-place.  Input `u` must have length $N$ (a
/// power of 2).  After the call, `u` holds the encoded codeword $x$ in
/// **natural order** (not bit-reversed; the spec applies bit-reversal as part
/// of the sub-block interleaver, handled separately).
fn polar_transform(u: &mut [u8]) {
    let n = u.len();
    debug_assert!(n.is_power_of_two());
    let mut step = 1usize;
    while step < n {
        let mut i = 0;
        while i < n {
            for j in 0..step {
                u[i + j] ^= u[i + j + step];
            }
            i += 2 * step;
        }
        step *= 2;
    }
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/// Polar encoder for 5G NR control channels.
///
/// The encoder inserts frozen bits (value 0) at the positions determined by
/// the 3GPP reliability sequence, applies the polar transform $G_N$, and
/// (optionally) performs rate-matching (puncturing/shortening) to produce
/// exactly `e_bits` coded bits.
///
/// # Examples
///
/// ```
/// use syndrome::polar::PolarEncoder;
///
/// // K=4 info bits, N=8 polar code.
/// let enc = PolarEncoder::new(8, 4).unwrap();
/// let info = vec![1u8, 0, 1, 1];
/// let mut codeword = vec![0u8; 8];
/// enc.encode(&info, &mut codeword).unwrap();
/// assert_eq!(codeword.len(), 8);
/// ```
pub struct PolarEncoder {
    n: usize,
    k: usize,
    is_frozen: Vec<bool>,
}

impl PolarEncoder {
    /// Create a polar encoder with block length `n` and `k` info bits.
    ///
    /// # Arguments
    ///
    /// * `n` - Code block length (must be a power of 2, max 1024 for 5G NR).
    /// * `k` - Number of information bits ($k < n$).
    ///
    /// # Errors
    ///
    /// Returns [`FecError::InvalidParam`] if `n` is not a power of 2 or
    /// `k >= n`.
    pub fn new(n: usize, k: usize) -> Result<Self, FecError> {
        if !n.is_power_of_two() || n == 0 {
            return Err(FecError::InvalidParam(
                "polar N must be a positive power of 2",
            ));
        }
        if k >= n {
            return Err(FecError::InvalidParam("k must be < n"));
        }
        let is_frozen = frozen_mask(n, k);
        Ok(Self { n, k, is_frozen })
    }

    /// Block length $N$.
    pub fn n(&self) -> usize {
        self.n
    }
    /// Information bits per block $K$.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Encode `info_bits` into `codeword`.
    ///
    /// # Arguments
    ///
    /// * `info_bits` - Slice of exactly `k` bits (values 0 or 1).
    /// * `codeword`  - Output slice of exactly `n` bits.
    ///
    /// # Errors
    ///
    /// Returns [`FecError::BufferTooSmall`] if lengths don't match.
    pub fn encode(&self, info_bits: &[u8], codeword: &mut [u8]) -> Result<(), FecError> {
        if info_bits.len() != self.k {
            return Err(FecError::BufferTooSmall {
                required: self.k,
                provided: info_bits.len(),
            });
        }
        if codeword.len() != self.n {
            return Err(FecError::BufferTooSmall {
                required: self.n,
                provided: codeword.len(),
            });
        }
        // Fill u: info at non-frozen positions, 0 at frozen positions.
        let mut u = vec![0u8; self.n];
        let mut info_idx = 0;
        for i in 0..self.n {
            if !self.is_frozen[i] {
                u[i] = info_bits[info_idx] & 1;
                info_idx += 1;
            }
        }
        polar_transform(&mut u);
        codeword.copy_from_slice(&u);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Successive Cancellation decoder
// ---------------------------------------------------------------------------

/// Branch-free $f$ combine kernel over equal-length contiguous slices:
///   $f(a, b) = \text{sgn}(a)\text{sgn}(b)\min(|a|, |b|)$ (min-sum approximation).
///
/// Computed with bit tricks instead of comparison branches: the sign of the
/// result is the XOR of `a`'s and `b`'s sign bits (`to_bits() ^ to_bits()`
/// masked to bit 31), and the magnitude is `min(|a|, |b|)`, both of which
/// LLVM lowers to branch-free instructions (`andps`/`minps` and friends), so
/// a plain loop over slices auto-vectorizes into packed SIMD ops without any
/// architecture-specific code.
///
/// # Arguments
///
/// * `a`, `b` - Equal-length LLR slices (the two halves of a decode-tree
///   node's input LLRs).
/// * `out` - Output slice, same length as `a`/`b`.
#[inline]
fn f_kernel(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    const SIGN_MASK: u32 = 0x8000_0000;
    for ((&av, &bv), ov) in a.iter().zip(b.iter()).zip(out.iter_mut()) {
        let sign = (av.to_bits() ^ bv.to_bits()) & SIGN_MASK;
        let abs_min = av.abs().min(bv.abs());
        *ov = f32::from_bits(abs_min.to_bits() | sign);
    }
}

/// Branch-free $g$ combine kernel over equal-length contiguous slices:
///   $g(a, b, \hat{u}) = b + (1 - 2\hat{u}) \cdot a$
///
/// `u_hat` is 0/1, so $(1 - 2\hat u)$ is exactly $\pm 1$: instead of a
/// multiply, `a`'s sign bit is XOR-ed with `u_hat` shifted into bit 31
/// (flipping `a`'s sign iff `u_hat = 1`), then added to `b` -- a bitwise op
/// plus a float add, both branch-free and auto-vectorizable.
///
/// # Arguments
///
/// * `a`, `b` - Equal-length LLR slices (the two halves of a decode-tree
///   node's input LLRs).
/// * `u_hat` - Equal-length hard-decision slice (the partial sum, see
///   [`sc_decode_recursive`]'s doc comment).
/// * `out` - Output slice, same length as `a`/`b`/`u_hat`.
#[inline]
fn g_kernel(a: &[f32], b: &[f32], u_hat: &[u8], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), u_hat.len());
    debug_assert_eq!(a.len(), out.len());
    for (((&av, &bv), &u), ov) in a.iter().zip(b.iter()).zip(u_hat.iter()).zip(out.iter_mut()) {
        let flip = (u as u32) << 31;
        let signed_a = f32::from_bits(av.to_bits() ^ flip);
        *ov = bv + signed_a;
    }
}

/// XOR-combine two adjacent, equal-length halves of a partial-sum ("beta")
/// buffer in place: `block[i] ^= block[half + i]` for `i in 0..half`, then
/// `block[half..]` is left untouched.
///
/// This is exactly the *last* butterfly step of [`polar_transform`] for a
/// block of this size, reusing the two halves' own (already fully
/// transformed, by the recursive invariant documented on
/// [`sc_decode_recursive`]) partial sums instead of re-deriving them from
/// scratch. A tight XOR loop over `u8` slices auto-vectorizes into packed
/// byte XORs.
#[inline]
fn beta_combine(block: &mut [u8]) {
    let half = block.len() / 2;
    let (left, right) = block.split_at_mut(half);
    for (l, &r) in left.iter_mut().zip(right.iter()) {
        *l ^= r;
    }
}

/// Preallocated scratch for [`PolarDecoder::decode_sc`]: a single flat LLR
/// buffer covering every recursion level below the root, the flat
/// partial-sum ("beta") byte array, and the decoded-bit output buffer --
/// all built **once, at [`PolarDecoder::new`] time** (this decoder's `n`
/// and recursion depth never change afterward) instead of once per decode
/// call. The original implementation allocated three `Vec`s (`left_llr`,
/// `right_llr`, `partial_sum`) at *every internal recursion node*; profiling
/// `decode_sc` at $(N,K)=(1024,512)$ showed that allocation churn alone
/// accounted for roughly half of total decode time (see the module's perf
/// notes), dwarfing the actual $f$/$g$/partial-sum arithmetic. Hoisting the
/// single per-call allocation (`ScScratch::new` below) up further, to
/// construction time, removes it entirely from the decode call path.
struct ScScratch {
    /// Flat LLR scratch for recursion levels `1..=levels` (the root's own
    /// level-0 LLRs are the caller-supplied `llr` slice, held separately).
    /// Laid out as `levels` consecutive blocks of `n` `f32`s each; level
    /// `L`'s block is peeled off the front of the slice as recursion
    /// descends (see `sc_decode_recursive`), so no level's block is ever
    /// aliased by another level's mutable borrow.
    llr: Vec<f32>,
    /// Flat, in-place bottom-up partial-sum ("beta") byte array, `n` bytes,
    /// addressed directly by absolute bit position (not peeled).
    beta: Vec<u8>,
    /// Decoded-bit output buffer, `n` bytes (info bits are extracted from
    /// this into the caller's `out` slice once decoding finishes).
    decoded: Vec<u8>,
}

impl ScScratch {
    fn new(n: usize, levels: usize) -> Self {
        Self {
            llr: vec![0.0f32; n * levels],
            beta: vec![0u8; n],
            decoded: vec![0u8; n],
        }
    }
}

/// Recursive SC decode over LLR array `llr` starting at `bit_start` for
/// `length` bits.  Decoded bits are written into `decoded[bit_start..]`.
///
/// `level_scratch` holds the flat LLR storage for *this node's children and
/// deeper* (`n` `f32`s per remaining level); it is peeled one level off the
/// front at each recursive step via `split_at_mut`, so every level's block
/// is a disjoint region of the one buffer allocated in [`ScScratch::new`] --
/// no allocation happens inside the recursion itself.
///
/// # Partial-sum propagation
///
/// The encoder recursion is $x_1 = (u_1 \oplus u_2) \cdot G_{N/2}$,
/// $x_2 = u_2 \cdot G_{N/2}$ (see `polar_transform`), so by GF(2) linearity
/// $x_1 \oplus x_2 = (u_1 \cdot G_{N/2})$: it is $u_1$ *re-encoded* through
/// the sub-code, not $u_1$ itself. That means `f_kernel`'s output, and hence
/// the left recursion's decoded result, lives in the *encoded* domain of
/// $u_1$ -- so the correct "hard decision" to feed into `g_kernel` for the
/// right branch is the **partial sum**: $\hat u_1$ re-encoded through
/// $G_{N/2}$, not the raw decoded bits themselves.
///
/// ## Bottom-up combine (avoiding $O(N \log^2 N)$)
///
/// Naively recomputing that re-encode via a fresh call to `polar_transform`
/// at *every* internal node costs $O(m \log m)$ at a node covering $m$
/// bits, and summed over the whole tree that is $O(N \log^2 N)$ total --
/// asymptotically worse than the $O(N \log N)$ $f$/$g$ steps, and it
/// dominated measured runtime (see `ScScratch`'s doc). Because
/// `polar_transform` is GF(2)-*linear* (pure XOR, no data dependence beyond
/// bit values), $\text{transform}(a \oplus b) = \text{transform}(a) \oplus
/// \text{transform}(b)$. Applying that identity to the encoder recursion
/// means: if `beta[bit_start..+half]` and `beta[bit_start+half..+length]`
/// already hold the fully re-encoded ("beta") values for the left and right
/// children *by the time both children's recursive calls return* (an
/// invariant maintained inductively -- true trivially at a leaf, where
/// $G_1$ is the identity, and preserved by the `beta_combine` call at the
/// end of this function), then this node's own beta is simply
/// `beta_L ^ beta_R` for the first half and `beta_R` unchanged for the
/// second -- one `O(half)` XOR pass (`beta_combine`), not a fresh
/// `O(half log half)` transform. Total cost over the whole tree drops back
/// to $O(N \log N)$. Skipping this combine (or re-introducing the naive
/// re-encode) is invisible for the all-zero and single-flag info vectors,
/// since re-encoding low-weight vectors is close to a no-op, but corrupts
/// most patterns of weight $\geq 2$ -- exactly the case the exhaustive
/// tests below exist to catch.
fn sc_decode_recursive(
    llr: &[f32],
    decoded: &mut [u8],
    beta: &mut [u8],
    is_frozen: &[bool],
    bit_start: usize,
    length: usize,
    level_scratch: &mut [f32],
    n: usize,
) {
    if length == 1 {
        let bit_pos = bit_start;
        let bit = if is_frozen[bit_pos] {
            0
        } else {
            (llr[0] < 0.0) as u8
        };
        decoded[bit_pos] = bit;
        beta[bit_pos] = bit; // G_1 is the identity: beta == decoded at a leaf.
        return;
    }
    let half = length / 2;
    let (this_level, deeper) = level_scratch.split_at_mut(n);

    // f-branch: left child's LLRs, written into this level's block.
    let (a, b) = llr.split_at(half);
    f_kernel(a, b, &mut this_level[bit_start..bit_start + half]);

    sc_decode_recursive(
        &this_level[bit_start..bit_start + half],
        decoded,
        beta,
        is_frozen,
        bit_start,
        half,
        deeper,
        n,
    );

    // g-branch: right child's LLRs, using the partial sum (not the raw
    // decoded left bits) as the hard-decision input. beta[bit_start..+half]
    // already holds the left subtree's fully re-encoded output (see doc
    // comment above), so no re-transform is needed here.
    let (a, b) = llr.split_at(half);
    let partial_sum = &beta[bit_start..bit_start + half];
    g_kernel(
        a,
        b,
        partial_sum,
        &mut this_level[bit_start + half..bit_start + length],
    );

    sc_decode_recursive(
        &this_level[bit_start + half..bit_start + length],
        decoded,
        beta,
        is_frozen,
        bit_start + half,
        half,
        deeper,
        n,
    );

    // This node's own partial sum (needed by its parent, if any): combine
    // the two children's already-transformed halves in place.
    beta_combine(&mut beta[bit_start..bit_start + length]);
}

// ---------------------------------------------------------------------------
// SCL decoder (list size L)
// ---------------------------------------------------------------------------

/// Per-level offsets into [`ScPath::llr_flat`], shared read-only across the
/// whole path list for one `decode_scl` call (computed once, not per path).
///
/// Level `lvl` covers `n >> lvl` elements, addressed **locally** (0-based,
/// relative to whichever node is currently occupying that level -- not by
/// the node's absolute `bit_start`): the decode tree is walked depth-first,
/// so at most one node per level is "live" at any instant, and a level's
/// block is safely reused (overwritten) for the next sibling once the
/// previous occupant's subtree has returned. This keeps the flat buffer at
/// `sum_{l=0}^{levels} (n >> l) = 2n - 1` elements total (matching the
/// original per-level `Vec<Vec<f32>>`'s total memory) instead of `n *
/// (levels + 1)` (which local-vs-absolute confusion would otherwise cost --
/// a 1024-length code has 11 levels, so that difference is the whole
/// ballgame for how much a path-fork clone has to copy).
fn scl_level_offsets(n: usize, levels: usize) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(levels + 1);
    let mut acc = 0usize;
    for lvl in 0..=levels {
        offsets.push(acc);
        acc += n >> lvl;
    }
    offsets
}

/// One path in the Successive Cancellation List decoder.
///
/// `llr_flat` holds this path's LLR arrays for *every* recursion level in
/// one flat buffer, laid out via [`scl_level_offsets`] (level `lvl`'s block
/// is `llr_flat[offsets[lvl]..offsets[lvl] + (n >> lvl)]`, addressed
/// locally -- see that function's doc for why). This is a flat SoA layout
/// rather than a `Vec<Vec<f32>>` per CLAUDE.md's flat-memory-layout
/// guidance, and it lets a fork clone the whole LLR history with one
/// contiguous copy instead of `levels + 1` separate small-`Vec` copies.
/// Keeping every level (rather than peeling levels off as
/// `sc_decode_recursive` does) is what lets a forked path carry its LLR
/// history through further recursion for free: cloning a path deep-copies
/// `llr_flat` verbatim, so every survivor keeps its own correct view of the
/// LLRs it will need once it reaches this level's `g` branch, no matter how
/// much list forking/pruning happened underneath in the meantime.
///
/// `beta` is this path's flat partial-sum byte array (see
/// [`sc_decode_recursive`]'s doc comment for the bottom-up XOR-combine
/// scheme it implements), addressed directly by absolute bit position (it
/// is *not* reused across siblings -- ancestors need a subtree's beta long
/// after that subtree has returned).
struct ScPath {
    decoded: Vec<u8>,
    beta: Vec<u8>,
    path_metric: f32,
    llr_flat: Vec<f32>,
}

impl ScPath {
    fn new(n: usize, initial_llr: &[f32], level_offsets: &[usize]) -> Self {
        let levels = level_offsets.len() - 1;
        let total = level_offsets[levels] + (n >> levels);
        let mut llr_flat = vec![0.0f32; total];
        llr_flat[..n].copy_from_slice(initial_llr);
        Self {
            decoded: vec![0u8; n],
            beta: vec![0u8; n],
            path_metric: 0.0,
            llr_flat,
        }
    }

    /// This path's LLR slice for the node currently occupying recursion
    /// `level`, of `length` elements (addressed locally -- see
    /// [`scl_level_offsets`]).
    #[inline]
    fn llr_at(&self, level_offsets: &[usize], level: usize, length: usize) -> &[f32] {
        let base = level_offsets[level];
        &self.llr_flat[base..base + length]
    }
}

/// Manual `Clone` impl so that `clone_from` can be overridden (the derived
/// impl only provides `clone`, and its inherited default `clone_from` is
/// just `*self = source.clone()` -- a fresh allocation, defeating the whole
/// point of [`SclArena`]).
impl Clone for ScPath {
    fn clone(&self) -> Self {
        Self {
            decoded: self.decoded.clone(),
            beta: self.beta.clone(),
            path_metric: self.path_metric,
            llr_flat: self.llr_flat.clone(),
        }
    }

    /// Overwrite `self` with `source`'s contents **reusing `self`'s existing
    /// heap allocations**. `Vec<T>::clone_from` is specialized (see its
    /// std docs) to `clone_from_slice` into the existing buffer instead of
    /// allocating when lengths already match -- which they always do here,
    /// since every [`ScPath`] built for one `decode_scl` call shares the
    /// same `n`/level layout. This is the operation [`SclArena`] forks
    /// through: no `Vec::new`/`Vec::with_capacity` runs on this path.
    fn clone_from(&mut self, source: &Self) {
        self.decoded.clone_from(&source.decoded);
        self.beta.clone_from(&source.beta);
        self.path_metric = source.path_metric;
        self.llr_flat.clone_from(&source.llr_flat);
    }
}

/// Fixed-capacity, allocation-free arena for [`ScPath`] forking, replacing
/// the original scheme of forking every surviving path via two full
/// `path.clone()` calls per info bit (3 heap allocations apiece --
/// `decoded`, `beta`, `llr_flat` -- for `O(list_size * k_info)` total
/// allocation/deallocation churn per `decode_scl` call; see
/// `scl_decode_recursive_reference`, kept for the equivalence test, for that
/// original code).
///
/// This holds **two** fixed-size `Vec<ScPath>` buffers ("ping-pong"
/// buffers), each with `2 * list_size` slots built once via [`ScPath::new`]
/// so every slot's `decoded`/`beta`/`llr_flat` `Vec`s are already correctly
/// sized. Forking at an information-bit leaf writes the (up to) `2 *
/// list_size` candidate paths into the *inactive* buffer's slots via
/// [`ScPath::clone_from`] (reusing each destination slot's existing
/// allocation) rather than [`Clone::clone`] (which would allocate fresh
/// `Vec`s), sorts by path metric, keeps the best `list_size`, and flips
/// which buffer is active. Only the logical path count (`len`) and the
/// active-buffer flag change per fork -- no heap (de)allocation happens
/// anywhere after [`SclArena::new`] returns.
struct SclArena {
    buf_a: Vec<ScPath>,
    buf_b: Vec<ScPath>,
    active_is_a: bool,
    len: usize,
}

impl SclArena {
    /// Build the arena for one `decode_scl` call: one live path (the root),
    /// backed by `2 * list_size`-slot buffers on both sides so any sequence
    /// of forks (each at most doubling, then pruned back to `list_size`)
    /// always has room in the currently-inactive buffer.
    fn new(n: usize, initial_llr: &[f32], level_offsets: &[usize], list_size: usize) -> Self {
        let cap = 2 * list_size;
        let seed = ScPath::new(n, initial_llr, level_offsets);
        let buf_a: Vec<ScPath> = (0..cap).map(|_| seed.clone()).collect();
        let buf_b: Vec<ScPath> = (0..cap).map(|_| seed.clone()).collect();
        Self {
            buf_a,
            buf_b,
            active_is_a: true,
            len: 1,
        }
    }

    /// Currently-live path list (read-only).
    #[inline]
    fn active(&self) -> &[ScPath] {
        if self.active_is_a {
            &self.buf_a[..self.len]
        } else {
            &self.buf_b[..self.len]
        }
    }

    /// Currently-live path list (mutable) -- used by the non-forking
    /// (frozen-bit leaf, $f$-branch, $g$-branch, beta-combine) steps, which
    /// mutate paths in place without changing how many there are.
    #[inline]
    fn active_mut(&mut self) -> &mut [ScPath] {
        if self.active_is_a {
            &mut self.buf_a[..self.len]
        } else {
            &mut self.buf_b[..self.len]
        }
    }

    /// Fork every live path into a `bit=0` and `bit=1` candidate at
    /// `bit_pos`, written into the inactive buffer via `clone_from` (no
    /// allocation), then sort by path metric and keep the best
    /// `list_size` survivors. Flips the active buffer.
    fn fork_at_leaf(
        &mut self,
        bit_pos: usize,
        level_offsets: &[usize],
        level: usize,
        list_size: usize,
    ) {
        let old_len = self.len;
        let new_len = 2 * old_len; // <= 2*list_size == capacity, since old_len <= list_size always.
        if self.active_is_a {
            let (src, dst) = (&self.buf_a, &mut self.buf_b);
            Self::fork_into(&src[..old_len], dst, level_offsets, level, bit_pos);
        } else {
            let (src, dst) = (&self.buf_b, &mut self.buf_a);
            Self::fork_into(&src[..old_len], dst, level_offsets, level, bit_pos);
        }
        let dst = if self.active_is_a {
            &mut self.buf_b
        } else {
            &mut self.buf_a
        };
        dst[..new_len].sort_by(|a, b| a.path_metric.partial_cmp(&b.path_metric).unwrap());
        self.len = new_len.min(list_size);
        self.active_is_a = !self.active_is_a;
    }

    /// Write `src[i]`'s two forked candidates into `dst[2*i]`/`dst[2*i+1]`
    /// via `clone_from`, in the same relative order the original
    /// clone-per-fork implementation pushed them in (bit=0 then bit=1 per
    /// source path, in source order) so the subsequent stable sort produces
    /// a bit-identical survivor list.
    fn fork_into(
        src: &[ScPath],
        dst: &mut [ScPath],
        level_offsets: &[usize],
        level: usize,
        bit_pos: usize,
    ) {
        for (i, path) in src.iter().enumerate() {
            let bit_llr = path.llr_at(level_offsets, level, 1)[0];

            dst[2 * i].clone_from(path);
            dst[2 * i].decoded[bit_pos] = 0;
            dst[2 * i].beta[bit_pos] = 0;
            dst[2 * i].path_metric += 0.0_f32.max(-bit_llr);

            dst[2 * i + 1].clone_from(path);
            dst[2 * i + 1].decoded[bit_pos] = 1;
            dst[2 * i + 1].beta[bit_pos] = 1;
            dst[2 * i + 1].path_metric += 0.0_f32.max(bit_llr);
        }
    }
}

/// Recursive CA-SCL decode over the whole path list at once.
///
/// Mirrors `sc_decode_recursive`'s $f$/$g$/partial-sum structure (see its
/// doc comment for the derivation of why the left sub-block must be
/// re-encoded through $G_{N/2}$ before it can feed `g_kernel`, and for the
/// $O(N \log N)$ bottom-up XOR-combine that avoids re-running
/// `polar_transform` at every node), generalised to a list of candidate
/// paths:
///
/// - `path.llr_at(level_offsets, level, length)` must already hold path
///   `p`'s own LLR array for *this* node, addressed **locally** (0-based;
///   see [`scl_level_offsets`]) -- the root call gets it from
///   `ScPath::new`; every other call is populated by its parent immediately
///   before recursing.
/// - At a leaf: frozen bits force `decoded[bit_start] = 0`; info bits fork
///   every current path into `0` and `1` candidates, then the whole list is
///   sorted by path metric and truncated to `list_size`. `beta[bit_start]`
///   (absolute-addressed, unlike the LLR levels) is set to the same hard
///   decision -- a leaf's partial sum is itself, since $G_1$ is the
///   identity.
/// - At an internal node: $f$-LLRs are computed per path into level
///   `level + 1`'s block and the left subtree recurses -- which may
///   fork/reorder/prune `paths`. Crucially, every surviving path (whatever
///   its lineage) still carries its own untouched level-`level` block
///   (clones copy `llr_flat` verbatim, and nothing below level `level + 1`
///   ever writes to level `level`), so the $g$-LLRs for the right subtree
///   can be computed correctly regardless of how the list changed
///   underneath. By the same inductive invariant as `sc_decode_recursive`,
///   `path.beta[bit_start..+half]` holds the left subtree's fully
///   re-encoded partial sum as soon as the left recursive call returns, so
///   no re-transform is needed before the $g$-branch. After the right
///   recursive call returns, each surviving path's own partial sum is
///   folded together with one `beta_combine` XOR pass, maintaining the
///   invariant for this node's parent.
fn scl_decode_recursive(
    is_frozen: &[bool],
    bit_start: usize,
    length: usize,
    level: usize,
    list_size: usize,
    level_offsets: &[usize],
    arena: &mut SclArena,
) {
    if length == 1 {
        let bit_pos = bit_start;
        if is_frozen[bit_pos] {
            for path in arena.active_mut() {
                let bit_llr = path.llr_at(level_offsets, level, 1)[0];
                path.decoded[bit_pos] = 0;
                path.beta[bit_pos] = 0;
                path.path_metric += 0.0_f32.max(-bit_llr); // ln(1 + e^-|llr|) ≈ 0
            }
        } else {
            // Info bit: fork every path into bit=0 and bit=1 candidates
            // (see `SclArena::fork_at_leaf` -- no heap allocation here).
            arena.fork_at_leaf(bit_pos, level_offsets, level, list_size);
        }
        return;
    }

    let half = length / 2;
    let this_base = level_offsets[level];
    let next_base = level_offsets[level + 1];

    // f-branch, computed per path (a path may already have forked away
    // from its siblings deeper in an earlier subtree). Local addressing:
    // this level's block starts at `this_base`, spans `length`; the next
    // level's block starts at `next_base`, spans `half` (reused below for
    // the g-branch once the left recursion has fully consumed it).
    for path in arena.active_mut() {
        let (head, tail) = path.llr_flat.split_at_mut(next_base);
        let src = &head[this_base..this_base + length];
        let (a, b) = src.split_at(half);
        f_kernel(a, b, &mut tail[..half]);
    }

    scl_decode_recursive(
        is_frozen,
        bit_start,
        half,
        level + 1,
        list_size,
        level_offsets,
        arena,
    );

    // g-branch: needs each surviving path's own partial sum from the left
    // subtree -- see doc comment above. `path.beta[bit_start..+half]` is
    // already correct by the recursive invariant, and level `level`'s LLR
    // block is untouched by the recursion above (which only ever writes to
    // level `level + 1` and deeper), so it's still valid here for every
    // surviving (possibly forked) path.
    for path in arena.active_mut() {
        let (head, tail) = path.llr_flat.split_at_mut(next_base);
        let src = &head[this_base..this_base + length];
        let (a, b) = src.split_at(half);
        let partial_sum = &path.beta[bit_start..bit_start + half];
        g_kernel(a, b, partial_sum, &mut tail[..half]);
    }

    scl_decode_recursive(
        is_frozen,
        bit_start + half,
        half,
        level + 1,
        list_size,
        level_offsets,
        arena,
    );

    // This node's own partial sum (needed by its parent, if any): combine
    // the two children's already-transformed halves in place, per path.
    for path in arena.active_mut() {
        beta_combine(&mut path.beta[bit_start..bit_start + length]);
    }
}

/// Reference (pre-optimization) CA-SCL recursion, kept only for the
/// equivalence test [`tests::scl_arena_matches_reference_random_noisy`].
/// Structurally identical to [`scl_decode_recursive`] except that path
/// forking uses a plain `Vec<ScPath>` and two `path.clone()` calls per
/// surviving path per info bit -- the O(list_size * k_info) heap
/// allocation/deallocation churn [`SclArena`] was written to eliminate. Not
/// on the hot path; not built into non-test binaries.
#[cfg(test)]
fn scl_decode_recursive_reference(
    is_frozen: &[bool],
    bit_start: usize,
    length: usize,
    level: usize,
    list_size: usize,
    level_offsets: &[usize],
    paths: &mut Vec<ScPath>,
) {
    if length == 1 {
        let bit_pos = bit_start;
        if is_frozen[bit_pos] {
            for path in paths.iter_mut() {
                let bit_llr = path.llr_at(level_offsets, level, 1)[0];
                path.decoded[bit_pos] = 0;
                path.beta[bit_pos] = 0;
                path.path_metric += 0.0_f32.max(-bit_llr); // ln(1 + e^-|llr|) ≈ 0
            }
        } else {
            // Info bit: fork every path into bit=0 and bit=1 candidates.
            let mut forked: Vec<ScPath> = Vec::with_capacity(paths.len() * 2);
            for path in paths.iter() {
                let bit_llr = path.llr_at(level_offsets, level, 1)[0];

                let mut p0 = path.clone();
                p0.decoded[bit_pos] = 0;
                p0.beta[bit_pos] = 0;
                p0.path_metric += 0.0_f32.max(-bit_llr);
                forked.push(p0);

                let mut p1 = path.clone();
                p1.decoded[bit_pos] = 1;
                p1.beta[bit_pos] = 1;
                p1.path_metric += 0.0_f32.max(bit_llr);
                forked.push(p1);
            }
            forked.sort_by(|a, b| a.path_metric.partial_cmp(&b.path_metric).unwrap());
            forked.truncate(list_size);
            *paths = forked;
        }
        return;
    }

    let half = length / 2;
    let this_base = level_offsets[level];
    let next_base = level_offsets[level + 1];

    for path in paths.iter_mut() {
        let (head, tail) = path.llr_flat.split_at_mut(next_base);
        let src = &head[this_base..this_base + length];
        let (a, b) = src.split_at(half);
        f_kernel(a, b, &mut tail[..half]);
    }

    scl_decode_recursive_reference(
        is_frozen,
        bit_start,
        half,
        level + 1,
        list_size,
        level_offsets,
        paths,
    );

    for path in paths.iter_mut() {
        let (head, tail) = path.llr_flat.split_at_mut(next_base);
        let src = &head[this_base..this_base + length];
        let (a, b) = src.split_at(half);
        let partial_sum = &path.beta[bit_start..bit_start + half];
        g_kernel(a, b, partial_sum, &mut tail[..half]);
    }

    scl_decode_recursive_reference(
        is_frozen,
        bit_start + half,
        half,
        level + 1,
        list_size,
        level_offsets,
        paths,
    );

    for path in paths.iter_mut() {
        beta_combine(&mut path.beta[bit_start..bit_start + length]);
    }
}

// ---------------------------------------------------------------------------
// Public decoder
// ---------------------------------------------------------------------------

/// Polar decoder supporting both Successive Cancellation (SC) and
/// CRC-aided Successive Cancellation List (CA-SCL) decoding.
///
/// # Examples
///
/// ```
/// use syndrome::channel_sim::AwgnChannel;
/// use syndrome::polar::{PolarEncoder, PolarDecoder};
///
/// let n = 32usize;
/// let k = 16usize;
/// let enc = PolarEncoder::new(n, k).unwrap();
/// let dec = PolarDecoder::new(n, k, 1, None).unwrap(); // SC (list=1, no CRC)
///
/// // A non-trivial (mixed 0/1) information vector -- SC decode must
/// // reconstruct it exactly, not just the all-zero pattern.
/// let info = vec![1u8, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1];
/// let mut codeword = vec![0u8; n];
/// enc.encode(&info, &mut codeword).unwrap();
///
/// // Noiseless channel.
/// let channel = AwgnChannel::new(10.0, k as f32 / n as f32, 1);
/// let llr = channel.transmit_noiseless(&codeword);
/// let mut out = vec![0u8; k];
/// dec.decode_sc(&llr, &mut out).unwrap();
/// assert_eq!(out, info);
/// ```
pub struct PolarDecoder {
    n: usize,
    k: usize,
    list_size: usize,
    is_frozen: Vec<bool>,
    crc: Option<Crc24>,
    /// Preallocated [`decode_sc`](Self::decode_sc) scratch (see
    /// [`ScScratch`]'s doc comment). Behind a `RefCell` because `decode_sc`
    /// takes `&self` (preserving its public signature); borrowing is
    /// uncontended (single-threaded, non-reentrant use per call).
    sc_scratch: std::cell::RefCell<ScScratch>,
}

impl PolarDecoder {
    /// Create a polar decoder.
    ///
    /// # Arguments
    ///
    /// * `n`         - Code block length (power of 2).
    /// * `k`         - Information bits.
    /// * `list_size` - SCL list size $L$ (1 = plain SC).
    /// * `crc_kind`  - Optional CRC kind for CA-SCL (e.g. [`CrcKind::Crc11`]
    ///   for DCI, [`CrcKind::Crc6`] for small UCI).
    ///
    /// # Errors
    ///
    /// Returns [`FecError::InvalidParam`] if `n` is not a power of 2, `k >=
    /// n`, or `list_size == 0`.
    pub fn new(
        n: usize,
        k: usize,
        list_size: usize,
        crc_kind: Option<CrcKind>,
    ) -> Result<Self, FecError> {
        if !n.is_power_of_two() || n == 0 {
            return Err(FecError::InvalidParam(
                "polar N must be a positive power of 2",
            ));
        }
        if k >= n {
            return Err(FecError::InvalidParam("k must be < n"));
        }
        if list_size == 0 {
            return Err(FecError::InvalidParam(
                "list_size must be >= 1 (0 would leave the SCL path list empty)",
            ));
        }
        let is_frozen = frozen_mask(n, k);
        let crc = crc_kind.map(Crc24::new);
        let levels = n.trailing_zeros() as usize;
        let sc_scratch = std::cell::RefCell::new(ScScratch::new(n, levels));
        Ok(Self {
            n,
            k,
            list_size,
            is_frozen,
            crc,
            sc_scratch,
        })
    }

    /// Decode using plain Successive Cancellation.
    ///
    /// # Arguments
    ///
    /// * `llr` - Channel LLRs (positive = likely 0, negative = likely 1),
    ///   length must equal `n`.
    /// * `out` - Output buffer of length `k` (info bits only).
    ///
    /// # Errors
    ///
    /// Returns [`FecError::BufferTooSmall`] on length mismatch, or
    /// [`FecError::InvalidParam`] if any `llr` value is `NaN` or `±infinity`
    /// (see [`Self::decode_scl`]'s doc comment for why non-finite LLRs are
    /// rejected outright rather than silently tolerated).
    pub fn decode_sc(&self, llr: &[f32], out: &mut [u8]) -> Result<(), FecError> {
        if llr.len() != self.n {
            return Err(FecError::BufferTooSmall {
                required: self.n,
                provided: llr.len(),
            });
        }
        if out.len() != self.k {
            return Err(FecError::BufferTooSmall {
                required: self.k,
                provided: out.len(),
            });
        }
        if llr.iter().any(|v| !v.is_finite()) {
            return Err(FecError::InvalidParam(
                "polar decode_sc: llr contains NaN or infinite value",
            ));
        }

        let mut scratch_ref = self.sc_scratch.borrow_mut();
        let scratch = &mut *scratch_ref;
        sc_decode_recursive(
            llr,
            &mut scratch.decoded,
            &mut scratch.beta,
            &self.is_frozen,
            0,
            self.n,
            &mut scratch.llr,
            self.n,
        );

        // Extract info bits.
        let mut info_idx = 0;
        for i in 0..self.n {
            if !self.is_frozen[i] {
                out[info_idx] = scratch.decoded[i];
                info_idx += 1;
            }
        }

        Ok(())
    }

    /// Decode using CRC-aided Successive Cancellation List (CA-SCL).
    ///
    /// Maintains `list_size` paths.  At each information bit position, the
    /// path is split into two candidates (bit=0 and bit=1).  The list is
    /// pruned to `list_size` survivors by path metric.  At the end, the
    /// path passing the CRC (if configured) is selected; otherwise the
    /// path with the best metric.
    ///
    /// # Arguments
    ///
    /// * `llr` - Channel LLRs, length `n`.
    /// * `out` - Info bit output, length `k`.
    ///
    /// # Errors
    ///
    /// Returns [`FecError::BufferTooSmall`] on size mismatch, or
    /// [`FecError::InvalidParam`] if any `llr` value is `NaN` or `±infinity`.
    /// `f32::partial_cmp` returns `None` for a `NaN` operand, which the path
    /// metric sort below relies on being `Some` (via `.unwrap()`); rather
    /// than silently reordering with `total_cmp` or letting the corruption
    /// propagate deep into path-metric arithmetic, a non-finite LLR is
    /// rejected here, at the boundary, since it is not meaningful soft
    /// information and decoding through it would not produce a meaningful
    /// answer, just a well-formed-looking wrong one.
    pub fn decode_scl(&self, llr: &[f32], out: &mut [u8]) -> Result<(), FecError> {
        if llr.len() != self.n {
            return Err(FecError::BufferTooSmall {
                required: self.n,
                provided: llr.len(),
            });
        }
        if out.len() != self.k {
            return Err(FecError::BufferTooSmall {
                required: self.k,
                provided: out.len(),
            });
        }
        if llr.iter().any(|v| !v.is_finite()) {
            return Err(FecError::InvalidParam(
                "polar decode_scl: llr contains NaN or infinite value",
            ));
        }

        // For list_size=1, fall back to the efficient SC path.
        if self.list_size == 1 {
            return self.decode_sc(llr, out);
        }

        let levels = self.n.trailing_zeros() as usize;
        let level_offsets = scl_level_offsets(self.n, levels);
        // Build the fork arena once for this call: one live path (the
        // root), backed by allocation-free ping-pong buffers for every
        // subsequent fork (see `SclArena`'s doc comment).
        let mut arena = SclArena::new(self.n, llr, &level_offsets, self.list_size);

        // Walk the decode tree once, forking/pruning the whole path list at
        // each information-bit leaf (see `scl_decode_recursive`'s doc for why
        // this must follow the same f/g/combine recursion as `decode_sc`
        // rather than a flat left-to-right bit scan).
        scl_decode_recursive(
            &self.is_frozen,
            0,
            self.n,
            0,
            self.list_size,
            &level_offsets,
            &mut arena,
        );

        let paths = arena.active();
        // Select the best CRC-passing path, or the best path if none pass.
        let best = if let Some(ref crc_eng) = self.crc {
            // Collect info bits for each path and CRC-check.
            let crc_pass = paths.iter().find(|p| {
                let mut info = Vec::with_capacity(self.k);
                for i in 0..self.n {
                    if !self.is_frozen[i] {
                        info.push(p.decoded[i]);
                    }
                }
                crc_eng.check(&info)
            });
            crc_pass.unwrap_or(&paths[0])
        } else {
            &paths[0]
        };

        // Extract info bits from the best path.
        let mut info_idx = 0;
        for i in 0..self.n {
            if !self.is_frozen[i] {
                out[info_idx] = best.decoded[i];
                info_idx += 1;
            }
        }

        Ok(())
    }

    /// Reference (pre-optimization) CA-SCL decode: identical to
    /// [`Self::decode_scl`] except it forks paths via
    /// [`scl_decode_recursive_reference`]'s plain `Vec<ScPath>` +
    /// `path.clone()` scheme instead of [`SclArena`]. Kept only to give
    /// [`tests::scl_arena_matches_reference_random_noisy`] an independent
    /// ground truth for bit-identity; not part of the public API.
    ///
    /// # Errors
    ///
    /// Returns [`FecError::BufferTooSmall`] on size mismatch (mirrors
    /// [`Self::decode_scl`]).
    #[cfg(test)]
    fn decode_scl_reference(&self, llr: &[f32], out: &mut [u8]) -> Result<(), FecError> {
        if llr.len() != self.n {
            return Err(FecError::BufferTooSmall {
                required: self.n,
                provided: llr.len(),
            });
        }
        if out.len() != self.k {
            return Err(FecError::BufferTooSmall {
                required: self.k,
                provided: out.len(),
            });
        }

        if self.list_size == 1 {
            return self.decode_sc(llr, out);
        }

        let levels = self.n.trailing_zeros() as usize;
        let level_offsets = scl_level_offsets(self.n, levels);
        let mut paths: Vec<ScPath> = vec![ScPath::new(self.n, llr, &level_offsets)];

        scl_decode_recursive_reference(
            &self.is_frozen,
            0,
            self.n,
            0,
            self.list_size,
            &level_offsets,
            &mut paths,
        );

        let best = if let Some(ref crc_eng) = self.crc {
            let crc_pass = paths.iter().find(|p| {
                let mut info = Vec::with_capacity(self.k);
                for i in 0..self.n {
                    if !self.is_frozen[i] {
                        info.push(p.decoded[i]);
                    }
                }
                crc_eng.check(&info)
            });
            crc_pass.unwrap_or(&paths[0])
        } else {
            &paths[0]
        };

        let mut info_idx = 0;
        for i in 0..self.n {
            if !self.is_frozen[i] {
                out[info_idx] = best.decoded[i];
                info_idx += 1;
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel_sim::AwgnChannel;

    fn noiseless_llr(codeword: &[u8], scale: f32) -> Vec<f32> {
        codeword
            .iter()
            .map(|&b| if b == 0 { scale } else { -scale })
            .collect()
    }

    use crate::test_util::Xorshift64;

    fn random_bits(rng: &mut Xorshift64, len: usize) -> Vec<u8> {
        (0..len).map(|_| rng.next_bool() as u8).collect()
    }

    #[test]
    fn encode_all_zeros_info_gives_all_zeros() {
        let enc = PolarEncoder::new(8, 4).unwrap();
        let info = vec![0u8; 4];
        let mut cw = vec![0u8; 8];
        enc.encode(&info, &mut cw).unwrap();
        // All-zero info → all-zero codeword (since frozen bits = 0 too).
        assert!(cw.iter().all(|&b| b == 0));
    }

    #[test]
    fn invalid_n_rejected() {
        assert!(PolarEncoder::new(7, 4).is_err());
        assert!(PolarEncoder::new(0, 0).is_err());
    }

    /// FINDING 7 regression guard (was
    /// `finding_polar_scl_list_size_zero_panics` in tests/robustness.rs,
    /// `#[should_panic]`): `PolarDecoder::new` used to accept `list_size ==
    /// 0`, and `decode_scl` would then index the (empty, after
    /// `truncate(0)`) path list out of bounds. `list_size == 0` is now
    /// rejected at construction.
    #[test]
    fn list_size_zero_rejected() {
        assert!(PolarDecoder::new(8, 4, 0, None).is_err());
        assert!(PolarDecoder::new(8, 4, 1, None).is_ok());
    }

    #[test]
    fn frozen_mask_has_correct_info_count() {
        let n = 32usize;
        let k = 12usize;
        let mask = frozen_mask(n, k);
        let info_count = mask.iter().filter(|&&f| !f).count();
        assert_eq!(info_count, k);
    }

    /// Exact-recovery-rate check at `N=512` and `N=1024` (PBCH and the 5G NR
    /// maximum PDCCH size): compares `frozen_mask`'s selected information
    /// set against the ground truth recomputed independently from
    /// `RELIABILITY_SEQ` (filter to positions `< N`, keep the `K` most
    /// reliable). For a construction driven by the real table this is 100%
    /// recovery by definition -- what the test actually establishes is that
    /// `frozen_mask` is wired to *use* `RELIABILITY_SEQ` end-to-end at these
    /// lengths rather than silently still falling through to the
    /// polarization-weight heuristic, which the negative control below
    /// rules out by confirming PW would have picked a different set.
    #[test]
    fn frozen_mask_n512_n1024_exact_recovery_from_real_table() {
        for &(n, k) in &[(512usize, 256usize), (1024usize, 512usize)] {
            let mask = frozen_mask(n, k);
            let info_count = mask.iter().filter(|&&f| !f).count();
            assert_eq!(info_count, k, "info bit count mismatch at N={n}, K={k}");

            // Ground truth: per TS 38.212 §5.3.1.2, filtering Q_Nmax to
            // entries < N (order preserved) yields Q_0^{N-1}; the K most
            // reliable of those (the last K) are the expected info set.
            let candidates: Vec<u16> = RELIABILITY_SEQ
                .iter()
                .copied()
                .filter(|&idx| (idx as usize) < n)
                .collect();
            assert_eq!(
                candidates.len(),
                n,
                "RELIABILITY_SEQ doesn't cover all {n} positions"
            );
            let mut expected_info: Vec<u16> = candidates[candidates.len() - k..].to_vec();
            expected_info.sort_unstable();

            let mut actual_info: Vec<u16> = (0..n as u16).filter(|&i| !mask[i as usize]).collect();
            actual_info.sort_unstable();

            assert_eq!(
                actual_info, expected_info,
                "frozen_mask's selection at N={n}, K={k} does not exactly \
                 match the real 3GPP table's K best positions -- exact \
                 recovery rate should be 100%, not partial"
            );

            // Negative control: independently recompute the PW fallback
            // ordering -- what frozen_mask uses only above the 3GPP table's
            // N=1024 -- and confirm it picks a genuinely *different* info
            // set from the table's at this N. Without
            // this, the assertion above would pass even if frozen_mask had
            // silently kept using PW here, because PW's own doc comment
            // already claims (unverifiably) to approximate the real table
            // reasonably well -- this rules that out concretely.
            const BETA: f64 = 0.25;
            let nbits = n.trailing_zeros();
            let pw_weight = |idx: u32| -> f64 {
                (0..nbits)
                    .filter(|b| (idx >> b) & 1 == 1)
                    .map(|b| 2f64.powf(BETA * b as f64))
                    .sum()
            };
            let mut pw_order: Vec<u32> = (0..n as u32).collect();
            pw_order.sort_by(|&a, &b| {
                pw_weight(a)
                    .partial_cmp(&pw_weight(b))
                    .unwrap()
                    .then(a.cmp(&b))
            });
            let mut pw_info: Vec<u16> = pw_order[pw_order.len() - k..]
                .iter()
                .map(|&x| x as u16)
                .collect();
            pw_info.sort_unstable();

            assert_ne!(
                actual_info, pw_info,
                "frozen_mask's output at N={n}, K={k} is identical to the \
                 PW heuristic's, which would make this test unable to tell \
                 apart 'using the real table' from 'still using PW'"
            );
        }
    }

    /// Exhaustive round-trip: every one of the $2^K$ possible information
    /// vectors must decode back exactly over a noiseless channel. This is
    /// the test that catches the missing partial-sum combine -- an all-zero
    /// (or any single-flag) message alone cannot, since the combine step is
    /// a no-op whenever the right half of every subtree is zero.
    #[test]
    fn sc_decode_exhaustive_n8_k4() {
        let n = 8usize;
        let k = 4usize;
        let enc = PolarEncoder::new(n, k).unwrap();
        let dec = PolarDecoder::new(n, k, 1, None).unwrap();
        for msg in 0u32..(1 << k) {
            let info: Vec<u8> = (0..k).map(|i| ((msg >> i) & 1) as u8).collect();
            let mut cw = vec![0u8; n];
            enc.encode(&info, &mut cw).unwrap();
            let llr = noiseless_llr(&cw, 10.0);
            let mut out = vec![0u8; k];
            dec.decode_sc(&llr, &mut out).unwrap();
            assert_eq!(
                out, info,
                "SC decode failed for info={info:?} (n={n}, k={k})"
            );
        }
    }

    #[test]
    fn sc_decode_exhaustive_n16_k8() {
        let n = 16usize;
        let k = 8usize;
        let enc = PolarEncoder::new(n, k).unwrap();
        let dec = PolarDecoder::new(n, k, 1, None).unwrap();
        for msg in 0u32..(1 << k) {
            let info: Vec<u8> = (0..k).map(|i| ((msg >> i) & 1) as u8).collect();
            let mut cw = vec![0u8; n];
            enc.encode(&info, &mut cw).unwrap();
            let llr = noiseless_llr(&cw, 10.0);
            let mut out = vec![0u8; k];
            dec.decode_sc(&llr, &mut out).unwrap();
            assert_eq!(
                out, info,
                "SC decode failed for info={info:?} (n={n}, k={k})"
            );
        }
    }

    /// Same exhaustive sweep through the list decoder (list=8, no CRC): SCL
    /// must be at least as capable as plain SC for every message.
    #[test]
    fn scl_decode_exhaustive_n8_k4_list8() {
        let n = 8usize;
        let k = 4usize;
        let enc = PolarEncoder::new(n, k).unwrap();
        let dec = PolarDecoder::new(n, k, 8, None).unwrap();
        for msg in 0u32..(1 << k) {
            let info: Vec<u8> = (0..k).map(|i| ((msg >> i) & 1) as u8).collect();
            let mut cw = vec![0u8; n];
            enc.encode(&info, &mut cw).unwrap();
            let llr = noiseless_llr(&cw, 10.0);
            let mut out = vec![0u8; k];
            dec.decode_scl(&llr, &mut out).unwrap();
            assert_eq!(
                out, info,
                "SCL decode failed for info={info:?} (n={n}, k={k})"
            );
        }
    }

    fn sc_random_noiseless_round_trip(n: usize, k: usize, trials: usize, seed: u64) {
        let enc = PolarEncoder::new(n, k).unwrap();
        let dec = PolarDecoder::new(n, k, 1, None).unwrap();
        let mut rng = Xorshift64::new(seed);
        for trial in 0..trials {
            let info = random_bits(&mut rng, k);
            let mut cw = vec![0u8; n];
            enc.encode(&info, &mut cw).unwrap();
            let llr = noiseless_llr(&cw, 10.0);
            let mut out = vec![0u8; k];
            dec.decode_sc(&llr, &mut out).unwrap();
            assert_eq!(
                out, info,
                "SC random round-trip failed at n={n}, k={k}, trial={trial}"
            );
        }
    }

    #[test]
    fn sc_decode_random_noiseless_n64_k32() {
        sc_random_noiseless_round_trip(64, 32, 100, 0xC0FF_EE01_u64);
    }

    #[test]
    fn sc_decode_random_noiseless_n256_k128() {
        sc_random_noiseless_round_trip(256, 128, 100, 0xC0FF_EE02_u64);
    }

    #[test]
    fn sc_decode_random_noiseless_n512_k256() {
        sc_random_noiseless_round_trip(512, 256, 100, 0xC0FF_EE04_u64);
    }

    #[test]
    fn sc_decode_random_noiseless_n1024_k512() {
        sc_random_noiseless_round_trip(1024, 512, 100, 0xC0FF_EE03_u64);
    }

    /// CA-SCL (list=8, CRC-24A) over random payloads, noiseless channel.
    #[test]
    fn scl_ca_decode_random_noiseless_with_crc24() {
        let n = 128usize;
        let crc_len = 24usize;
        let info_len = 40usize;
        let k_with_crc = info_len + crc_len;
        let enc = PolarEncoder::new(n, k_with_crc).unwrap();
        let dec = PolarDecoder::new(n, k_with_crc, 8, Some(CrcKind::Crc24A)).unwrap();
        let crc_eng = Crc24::new(CrcKind::Crc24A);
        let mut rng = Xorshift64::new(0xC0FF_EE10_u64);
        for trial in 0..100 {
            let mut info = random_bits(&mut rng, info_len);
            crc_eng.attach(&mut info);
            assert_eq!(info.len(), k_with_crc);
            let mut cw = vec![0u8; n];
            enc.encode(&info, &mut cw).unwrap();
            let llr = noiseless_llr(&cw, 10.0);
            let mut out = vec![0u8; k_with_crc];
            dec.decode_scl(&llr, &mut out).unwrap();
            assert_eq!(
                out, info,
                "CA-SCL random round-trip failed at trial={trial}"
            );
        }
    }

    /// AWGN, N=1024/K=512 (rate 1/2) at 3 dB Eb/N0: plain SC vs. CA-SCL
    /// (list=8, CRC-24A) over 20 fixed-seed trials each. Both use overall
    /// K=512 (CA-SCL spends 24 of those on the CRC, 488 on the message) so
    /// the two decoders are compared at the same coded rate.
    #[test]
    fn awgn_n1024_k512_3db_sc_vs_ca_scl() {
        let n = 1024usize;
        let k = 512usize;
        let crc_len = 24usize;
        let info_len = k - crc_len;
        let ebno_db = 3.0f32;
        let trials = 20usize;
        let rate = k as f32 / n as f32;

        let enc_sc = PolarEncoder::new(n, k).unwrap();
        let dec_sc = PolarDecoder::new(n, k, 1, None).unwrap();
        let mut rng_sc = Xorshift64::new(0x5EED_5C00_u64);
        let mut channel_sc = AwgnChannel::new(ebno_db, rate, 0x5EED_C4A0_u64);
        let mut sc_successes = 0usize;
        for _ in 0..trials {
            let info = random_bits(&mut rng_sc, k);
            let mut cw = vec![0u8; n];
            enc_sc.encode(&info, &mut cw).unwrap();
            let llr = channel_sc.transmit(&cw);
            let mut out = vec![0u8; k];
            dec_sc.decode_sc(&llr, &mut out).unwrap();
            if out == info {
                sc_successes += 1;
            }
        }

        let enc_ca = PolarEncoder::new(n, k).unwrap();
        let dec_ca = PolarDecoder::new(n, k, 8, Some(CrcKind::Crc24A)).unwrap();
        let crc_eng = Crc24::new(CrcKind::Crc24A);
        let mut rng_ca = Xorshift64::new(0x5EED_5C01_u64);
        let mut channel_ca = AwgnChannel::new(ebno_db, rate, 0x5EED_C4A1_u64);
        let mut ca_successes = 0usize;
        for _ in 0..trials {
            let mut info = random_bits(&mut rng_ca, info_len);
            crc_eng.attach(&mut info);
            let mut cw = vec![0u8; n];
            enc_ca.encode(&info, &mut cw).unwrap();
            let llr = channel_ca.transmit(&cw);
            let mut out = vec![0u8; k];
            dec_ca.decode_scl(&llr, &mut out).unwrap();
            if out == info {
                ca_successes += 1;
            }
        }

        // Measured on this seed/SNR: SC 20/20, CA-SCL 20/20, with frozen-bit
        // selection at N=1024 now driven directly by the real 3GPP `Q_Nmax`
        // table (see `frozen_mask`'s doc comment) rather than the PW
        // heuristic. Assert with a small margin below the observed counts
        // to tolerate float-rounding differences across platforms, rather
        // than the exact measured values.
        assert!(
            sc_successes >= 15,
            "SC success rate too low: {sc_successes}/{trials}"
        );
        assert!(
            ca_successes >= 15,
            "CA-SCL success rate too low: {ca_successes}/{trials}"
        );
    }

    /// Task-1 equivalence guard: `decode_scl` (arena-based forking, no
    /// per-fork heap allocation -- see [`SclArena`]) must produce
    /// byte-identical output to `decode_scl_reference` (the retained
    /// pre-optimization `Vec<ScPath>` + `path.clone()` implementation, see
    /// [`scl_decode_recursive_reference`]) across many random noisy frames,
    /// several (N, K, list_size, CRC) configurations, both with and without
    /// a CRC winner actually present in the list (low SNR exercises the
    /// "no path passes CRC, fall back to best metric" branch too).
    #[test]
    fn scl_arena_matches_reference_random_noisy() {
        struct Cfg {
            n: usize,
            k: usize,
            list_size: usize,
            crc: Option<CrcKind>,
            ebno_db: f32,
            seed: u64,
        }
        let configs = [
            Cfg {
                n: 64,
                k: 32,
                list_size: 4,
                crc: None,
                ebno_db: 2.0,
                seed: 0xA11C_E000,
            },
            Cfg {
                n: 128,
                k: 64,
                list_size: 8,
                crc: Some(CrcKind::Crc24A),
                ebno_db: 1.0,
                seed: 0xA11C_E001,
            },
            Cfg {
                n: 256,
                k: 128,
                list_size: 16,
                crc: Some(CrcKind::Crc11),
                ebno_db: 0.0,
                seed: 0xA11C_E002,
            },
            Cfg {
                n: 1024,
                k: 512,
                list_size: 8,
                crc: Some(CrcKind::Crc24A),
                ebno_db: (-1.0),
                seed: 0xA11C_E003,
            },
        ];

        for cfg in configs {
            let rate = cfg.k as f32 / cfg.n as f32;
            let enc = PolarEncoder::new(cfg.n, cfg.k).unwrap();
            let dec = PolarDecoder::new(cfg.n, cfg.k, cfg.list_size, cfg.crc).unwrap();
            let mut rng = Xorshift64::new(cfg.seed);
            let mut channel = AwgnChannel::new(cfg.ebno_db, rate, cfg.seed ^ 0x5EED);

            for trial in 0..40 {
                let info = random_bits(&mut rng, cfg.k);
                let mut cw = vec![0u8; cfg.n];
                enc.encode(&info, &mut cw).unwrap();
                let llr = channel.transmit(&cw);

                let mut out_arena = vec![0u8; cfg.k];
                let mut out_reference = vec![0u8; cfg.k];
                dec.decode_scl(&llr, &mut out_arena).unwrap();
                dec.decode_scl_reference(&llr, &mut out_reference).unwrap();

                assert_eq!(
                    out_arena, out_reference,
                    "arena vs reference SCL decode mismatch: n={}, k={}, list_size={}, trial={trial}",
                    cfg.n, cfg.k, cfg.list_size
                );
            }
        }
    }
}
