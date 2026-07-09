//! Viterbi convolutional decoder — rate-1/2, K=7 (3GPP/NASA standard).
//!
//! Implements the classic Viterbi algorithm with Add-Compare-Select (ACS) trellis
//! search for binary convolutional codes of rate 1/2.  The standard K=7 code
//! (generators $G_0 = 0o133$, $G_1 = 0o171$) is used by many 3GPP profiles
//! (LTE PDCCH tail-biting variant, Turbo component encoder, etc.).
//!
//! # Algorithm
//!
//! For a constraint length $K$ and rate $1/2$ code, the encoder maintains a
//! $(K-1)$-bit shift register.  At each step, one input bit and the register
//! produce two output bits via
//! $$
//!   c_0 = \bigoplus_{i=0}^{K-1} u_{n-i} \cdot g_0^{(i)}, \quad
//!   c_1 = \bigoplus_{i=0}^{K-1} u_{n-i} \cdot g_1^{(i)}
//! $$
//! where $g_0, g_1$ are the generator polynomials in binary.
//!
//! The decoder recovers the maximum-likelihood input sequence by searching all
//! $2^{K-1}$ states with an ACS forward pass followed by a traceback.
//!
//! # Zero termination
//!
//! The encoder is assumed to start and end in state 0 (zero-terminated frame):
//! $(K-1)$ tail zeros are appended to the information bits before encoding.
//! The decoder exploits this: the traceback always starts from state 0 at the
//! end of the trellis.
//!
//! # Examples
//!
//! ```
//! use glezer_rsv::viterbi::ViterbiDecoder;
//!
//! let dec = ViterbiDecoder::new(7);
//! let info: Vec<u8> = vec![1, 0, 1, 1, 0, 0, 1];
//! let coded = dec.encode(&info);
//! let decoded = dec.decode_hard(&coded);
//! assert_eq!(decoded, info);
//! ```

// ---------------------------------------------------------------------------
// Trellis transition table (built once at construction)
// ---------------------------------------------------------------------------

/// Precomputed trellis: for each (state, input_bit) pair stores the next
/// state and both output bits.  Indexed as `table[state * 2 + input_bit]`.
struct TrellisTable {
    n_states: usize,
    next_state: Vec<u8>,
    out0: Vec<u8>,
    out1: Vec<u8>,
    /// Butterfly-reorganised branch tables used by the AVX2 ACS kernel.
    bfly: ButterflyTables,
}

/// Branch-output tables reorganised into the "butterfly" layout consumed by
/// the vectorised ACS kernel.
///
/// # Butterfly structure
///
/// For a rate-1/2 code the next-state function is
/// $$
///   \mathrm{ns}(s, b) = (b \ll (K-2)) \mathrel{|} (s \gg 1),
/// $$
/// i.e. the two states $2j$ and $2j+1$ (same $j = s \gg 1$) are the *only*
/// predecessors of next-states $j$ (via input $b=0$) and $\mathrm{half}+j$
/// (via input $b=1$).  Grouping the trellis by $j = 0 \dots \mathrm{half}-1$
/// turns the per-bit ACS update into `half` independent 2-way compare-selects,
/// which vectorise cleanly across 8-wide SIMD lanes when `half` is a multiple
/// of 8 (true for K=7, where `half = 32`).
///
/// Layout: `out0_even[b * half + j]` is the `out0` value for source state
/// $2j$ with input `b`; `out0_odd`/`out1_odd` are the same for source state
/// $2j+1$.  Built for any even `n_states`, but only consumed by the AVX2
/// kernel when `n_states == 64` (K=7); other constraint lengths use the
/// scalar fallback and never read these tables.
struct ButterflyTables {
    /// `n_states / 2` (0 when `n_states == 1`, i.e. K=1).
    half: usize,
    out0_even: Vec<i32>,
    out1_even: Vec<i32>,
    out0_odd: Vec<i32>,
    out1_odd: Vec<i32>,
}

impl ButterflyTables {
    /// Reorganise the state-major `out0`/`out1` tables into the butterfly
    /// (predecessor-pair-major) layout described on [`ButterflyTables`].
    fn build(n_states: usize, out0: &[u8], out1: &[u8]) -> Self {
        let half = n_states / 2;
        let mut out0_even = vec![0i32; 2 * half];
        let mut out1_even = vec![0i32; 2 * half];
        let mut out0_odd = vec![0i32; 2 * half];
        let mut out1_odd = vec![0i32; 2 * half];
        for j in 0..half {
            let s_even = 2 * j;
            let s_odd = 2 * j + 1;
            for b in 0..2usize {
                out0_even[b * half + j] = out0[s_even * 2 + b] as i32;
                out1_even[b * half + j] = out1[s_even * 2 + b] as i32;
                out0_odd[b * half + j] = out0[s_odd * 2 + b] as i32;
                out1_odd[b * half + j] = out1[s_odd * 2 + b] as i32;
            }
        }
        ButterflyTables {
            half,
            out0_even,
            out1_even,
            out0_odd,
            out1_odd,
        }
    }
}

impl TrellisTable {
    /// Build for constraint length `k` and 32-bit generator polynomials `g0`/`g1`.
    ///
    /// For K=7 (standard): `g0=0o133`, `g1=0o171`.
    fn build(k: usize, g0: u32, g1: u32) -> Self {
        // n_states = 2^(K-1); state stores the last K-1 input bits.
        let n_states = if k >= 1 { 1usize << (k - 1) } else { 1 };
        let mask = if k < 32 { (1u32 << k) - 1 } else { u32::MAX };
        let mut next_state = vec![0u8; n_states * 2];
        let mut out0 = vec![0u8; n_states * 2];
        let mut out1 = vec![0u8; n_states * 2];

        for s in 0..n_states {
            for b in 0..2usize {
                // full shift register: new bit `b` enters the MSB.
                let full = (((b << (k - 1)) | s) as u32) & mask;
                // Next state drops the oldest bit (LSB of full after shift).
                let ns = (full >> 1) as u8;
                let o0 = ((full & g0).count_ones() & 1) as u8;
                let o1 = ((full & g1).count_ones() & 1) as u8;
                next_state[s * 2 + b] = ns;
                out0[s * 2 + b] = o0;
                out1[s * 2 + b] = o1;
            }
        }
        let bfly = ButterflyTables::build(n_states, &out0, &out1);
        TrellisTable {
            n_states,
            next_state,
            out0,
            out1,
            bfly,
        }
    }
}

// ---------------------------------------------------------------------------
// AVX2 add-compare-select kernel (x86_64, K=7 / 64-state specific)
// ---------------------------------------------------------------------------
//
// This module vectorises the butterfly ACS recursion described on
// [`ButterflyTables`]. Each call processes one trellis step (one received
// bit pair) for all 64 states, expressed as 4 groups of 8 lanes over the
// `half = 32` predecessor pairs.
//
// # Deinterleave trick
//
// The predecessor metrics `cur_met[2j]` / `cur_met[2j+1]` are adjacent pairs
// in the natural (state-indexed) metric array, so reading 8 lanes worth of
// `j` requires deinterleaving 16 consecutive `cur_met` entries into their
// even- and odd-indexed halves. `_mm256_permutevar8x32_{ps,epi32}` extracts
// the even (or odd) elements from each 8-wide half independently (fast,
// single-cycle-throughput shuffle — no AVX2 gather instruction needed), and
// `_mm256_insertf128_{ps,si256}` stitches the two 4-lane results back into
// one 8-lane vector. Destination writes need no such trick: the two
// butterfly outputs (`b=0` group and `b=1` group) land on the *contiguous*
// natural state ranges `0..32` and `32..64` respectively.
#[cfg(target_arch = "x86_64")]
mod avx2_acs {
    use super::ButterflyTables;
    use core::arch::x86_64::*;

    /// Number of butterfly pairs processed for K=7 (`half = 32`, 4 lanes of 8).
    const N_CHUNKS: usize = 4;

    /// Deinterleave 16 consecutive `f32`s starting at `src[base..base+16]`
    /// into their even-indexed and odd-indexed 8-wide halves.
    ///
    /// Returns `(even, odd)` where `even` lane `i` = `src[base + 2*i]` and
    /// `odd` lane `i` = `src[base + 2*i + 1]`, for `i` in `0..8`.
    ///
    /// # Safety
    ///
    /// Caller must guarantee AVX2 support and `src.len() >= base + 16`.
    #[target_feature(enable = "avx2")]
    unsafe fn deinterleave16_ps(src: &[f32], base: usize) -> (__m256, __m256) {
        unsafe {
            let idx_even = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);
            let idx_odd = _mm256_set_epi32(7, 5, 3, 1, 7, 5, 3, 1);
            let r0 = _mm256_loadu_ps(src.as_ptr().add(base));
            let r1 = _mm256_loadu_ps(src.as_ptr().add(base + 8));
            let e0 = _mm256_permutevar8x32_ps(r0, idx_even);
            let e1 = _mm256_permutevar8x32_ps(r1, idx_even);
            let o0 = _mm256_permutevar8x32_ps(r0, idx_odd);
            let o1 = _mm256_permutevar8x32_ps(r1, idx_odd);
            let even = _mm256_insertf128_ps(e0, _mm256_castps256_ps128(e1), 1);
            let odd = _mm256_insertf128_ps(o0, _mm256_castps256_ps128(o1), 1);
            (even, odd)
        }
    }

    /// Integer counterpart of [`deinterleave16_ps`] for the hard-decision path.
    ///
    /// # Safety
    ///
    /// Caller must guarantee AVX2 support and `src.len() >= base + 16`.
    #[target_feature(enable = "avx2")]
    unsafe fn deinterleave16_epi32(src: &[i32], base: usize) -> (__m256i, __m256i) {
        unsafe {
            let idx_even = _mm256_set_epi32(6, 4, 2, 0, 6, 4, 2, 0);
            let idx_odd = _mm256_set_epi32(7, 5, 3, 1, 7, 5, 3, 1);
            let r0 = _mm256_loadu_si256(src.as_ptr().add(base) as *const __m256i);
            let r1 = _mm256_loadu_si256(src.as_ptr().add(base + 8) as *const __m256i);
            let e0 = _mm256_permutevar8x32_epi32(r0, idx_even);
            let e1 = _mm256_permutevar8x32_epi32(r1, idx_even);
            let o0 = _mm256_permutevar8x32_epi32(r0, idx_odd);
            let o1 = _mm256_permutevar8x32_epi32(r1, idx_odd);
            let even = _mm256_insertf128_si256(e0, _mm256_castsi256_si128(e1), 1);
            let odd = _mm256_insertf128_si256(o0, _mm256_castsi256_si128(o1), 1);
            (even, odd)
        }
    }

    /// One AVX2 trellis step of the soft-decision (max-log-MAP) ACS recursion.
    ///
    /// `cur_met`/`nxt_met` are length-64 metric arrays (natural state order).
    /// `traceback_row` is the length-64 traceback row for this step (previous
    /// state that won each next-state). `sign*_{even,odd}` are the
    /// precomputed $\pm 1$ branch-sign tables (length `2*half`, see
    /// [`ButterflyTables`] layout) built once per `decode_soft` call from the
    /// integer `out0`/`out1` tables. `l0`/`l1` are this step's received LLRs.
    ///
    /// # Safety
    ///
    /// Caller must guarantee AVX2 support at runtime and `bfly.half == 32`
    /// (K=7 / 64 states); all slice arguments must have the lengths
    /// documented above.
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn acs_step_soft(
        cur_met: &[f32],
        nxt_met: &mut [f32],
        traceback_row: &mut [u8],
        bfly: &ButterflyTables,
        sign0_even: &[f32],
        sign1_even: &[f32],
        sign0_odd: &[f32],
        sign1_odd: &[f32],
        l0: f32,
        l1: f32,
    ) {
        unsafe {
            debug_assert_eq!(bfly.half, N_CHUNKS * 8);
            let half = bfly.half;
            let l0v = _mm256_set1_ps(l0);
            let l1v = _mm256_set1_ps(l1);
            let one = _mm256_set1_epi32(1);
            let iota = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);

            for c in 0..N_CHUNKS {
                let base16 = c * 16;
                let (even_v, odd_v) = deinterleave16_ps(cur_met, base16);
                let j_base = _mm256_add_epi32(iota, _mm256_set1_epi32((c * 8) as i32));
                let state_base = _mm256_slli_epi32(j_base, 1); // 2*j

                for b in 0..2usize {
                    let off = b * half + c * 8;
                    let s0e = _mm256_loadu_ps(sign0_even[off..].as_ptr());
                    let s1e = _mm256_loadu_ps(sign1_even[off..].as_ptr());
                    let s0o = _mm256_loadu_ps(sign0_odd[off..].as_ptr());
                    let s1o = _mm256_loadu_ps(sign1_odd[off..].as_ptr());

                    let bm_even = _mm256_add_ps(_mm256_mul_ps(s0e, l0v), _mm256_mul_ps(s1e, l1v));
                    let bm_odd = _mm256_add_ps(_mm256_mul_ps(s0o, l0v), _mm256_mul_ps(s1o, l1v));

                    let cand_even = _mm256_add_ps(even_v, bm_even);
                    let cand_odd = _mm256_add_ps(odd_v, bm_odd);

                    // Strict '>' matches the scalar tie-break: the even (2j)
                    // predecessor always arrives first, so ties keep it.
                    let mask = _mm256_cmp_ps(cand_odd, cand_even, _CMP_GT_OQ);
                    let winner = _mm256_blendv_ps(cand_even, cand_odd, mask);

                    let dest = b * half + c * 8;
                    _mm256_storeu_ps(nxt_met[dest..].as_mut_ptr(), winner);

                    let mask_i = _mm256_castps_si256(mask);
                    let winner_bit = _mm256_and_si256(mask_i, one);
                    let pred_state = _mm256_add_epi32(state_base, winner_bit);

                    let mut tmp = [0i32; 8];
                    _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, pred_state);
                    for i in 0..8 {
                        traceback_row[dest + i] = tmp[i] as u8;
                    }
                }
            }
        }
    }

    /// One AVX2 trellis step of the hard-decision (Hamming-metric) ACS
    /// recursion. Mirrors [`acs_step_soft`] with `i32` metrics: branch
    /// metrics are `(r ^ out)` summed (values in `{0, 1, 2}`), and the
    /// merge picks the *smaller* candidate (min, not max), again breaking
    /// ties toward the even predecessor to match the scalar reference.
    ///
    /// # Safety
    ///
    /// Same preconditions as [`acs_step_soft`].
    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn acs_step_hard(
        cur_met: &[i32],
        nxt_met: &mut [i32],
        traceback_row: &mut [u8],
        bfly: &ButterflyTables,
        r0: i32,
        r1: i32,
    ) {
        unsafe {
            debug_assert_eq!(bfly.half, N_CHUNKS * 8);
            let half = bfly.half;
            let r0v = _mm256_set1_epi32(r0);
            let r1v = _mm256_set1_epi32(r1);
            let one = _mm256_set1_epi32(1);
            let iota = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);

            for c in 0..N_CHUNKS {
                let base16 = c * 16;
                let (even_v, odd_v) = deinterleave16_epi32(cur_met, base16);
                let j_base = _mm256_add_epi32(iota, _mm256_set1_epi32((c * 8) as i32));
                let state_base = _mm256_slli_epi32(j_base, 1); // 2*j

                for b in 0..2usize {
                    let off = b * half + c * 8;
                    let o0e = _mm256_loadu_si256(bfly.out0_even[off..].as_ptr() as *const __m256i);
                    let o1e = _mm256_loadu_si256(bfly.out1_even[off..].as_ptr() as *const __m256i);
                    let o0o = _mm256_loadu_si256(bfly.out0_odd[off..].as_ptr() as *const __m256i);
                    let o1o = _mm256_loadu_si256(bfly.out1_odd[off..].as_ptr() as *const __m256i);

                    let bm_even =
                        _mm256_add_epi32(_mm256_xor_si256(r0v, o0e), _mm256_xor_si256(r1v, o1e));
                    let bm_odd =
                        _mm256_add_epi32(_mm256_xor_si256(r0v, o0o), _mm256_xor_si256(r1v, o1o));

                    let cand_even = _mm256_add_epi32(even_v, bm_even);
                    let cand_odd = _mm256_add_epi32(odd_v, bm_odd);

                    // odd wins only if strictly smaller (min-metric, tie -> even).
                    let mask = _mm256_cmpgt_epi32(cand_even, cand_odd);
                    let winner = _mm256_blendv_epi8(cand_even, cand_odd, mask);

                    let dest = b * half + c * 8;
                    _mm256_storeu_si256(nxt_met[dest..].as_mut_ptr() as *mut __m256i, winner);

                    let winner_bit = _mm256_and_si256(mask, one);
                    let pred_state = _mm256_add_epi32(state_base, winner_bit);

                    let mut tmp = [0i32; 8];
                    _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, pred_state);
                    for i in 0..8 {
                        traceback_row[dest + i] = tmp[i] as u8;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Generator polynomial defaults
// ---------------------------------------------------------------------------

/// Select standard generator polynomials for a given constraint length.
///
/// | K | G0 (octal) | G1 (octal) | Standard |
/// |---|-----------|-----------|----------|
/// | 3 | 7 | 5 | CCSDS R=1/2 K=3 |
/// | 5 | 23 | 35 | CCSDS R=1/2 K=5 |
/// | 7 | 133 | 171 | NASA/3GPP R=1/2 K=7 |
/// | 9 | 561 | 753 | CCSDS R=1/2 K=9 |
fn default_generators(k: usize) -> (u32, u32) {
    match k {
        1 => (0b1, 0b1),
        2 => (0b11, 0b10),
        3 => (0o7, 0o5),
        4 => (0o17, 0o13),
        5 => (0o23, 0o35),
        6 => (0o53, 0o75),
        7 => (0o133, 0o171),
        8 => (0o247, 0o371),
        9 => (0o561, 0o753),
        _ => {
            let g0 = ((1u64 << k) - 1) as u32;
            let g1 = ((1u64 << k) - 1) as u32 >> 1 | 1;
            (g0, g1)
        }
    }
}

// ---------------------------------------------------------------------------
// Public decoder struct
// ---------------------------------------------------------------------------

/// Rate-1/2 Viterbi convolutional decoder.
///
/// Decodes zero-terminated frames encoded with the matching convolutional
/// encoder.  Supports both hard-decision (Hamming-metric) and soft-decision
/// (max-log-MAP branch metric) modes.
pub struct ViterbiDecoder {
    /// Constraint length $K$.  The encoder register holds $K-1$ bits.
    pub constraint_length: usize,
    trellis: TrellisTable,
}

impl ViterbiDecoder {
    /// Create a decoder for a rate-1/2, constraint-length `k` code using the
    /// standard generator polynomials for that `k` (see `default_generators`).
    ///
    /// # Arguments
    ///
    /// * `k` - Constraint length (≥ 1).  K=7 uses generators 0o133/0o171.
    ///
    /// # Examples
    ///
    /// ```
    /// use glezer_rsv::viterbi::ViterbiDecoder;
    /// let dec = ViterbiDecoder::new(7);
    /// assert_eq!(dec.constraint_length, 7);
    /// ```
    pub fn new(k: usize) -> Self {
        let k = k.max(1);
        let (g0, g1) = default_generators(k);
        Self {
            constraint_length: k,
            trellis: TrellisTable::build(k, g0, g1),
        }
    }

    /// Create a decoder with explicit generator polynomials.
    ///
    /// # Arguments
    ///
    /// * `k`  - Constraint length (≥ 1).
    /// * `g0` - First generator polynomial (K-bit integer, MSB = current input).
    /// * `g1` - Second generator polynomial.
    pub fn with_generators(k: usize, g0: u32, g1: u32) -> Self {
        let k = k.max(1);
        Self {
            constraint_length: k,
            trellis: TrellisTable::build(k, g0, g1),
        }
    }

    /// Encode `info` bits as a zero-terminated rate-1/2 convolutional stream.
    ///
    /// Appends $K-1$ tail zeros after `info` to force the encoder to end in
    /// state 0, then outputs pairs `(c0, c1)` for each input bit.
    ///
    /// # Returns
    ///
    /// A `Vec<u8>` of length `2 * (info.len() + K - 1)`, values in `{0, 1}`.
    ///
    /// # Examples
    ///
    /// ```
    /// use glezer_rsv::viterbi::ViterbiDecoder;
    /// let dec = ViterbiDecoder::new(7);
    /// let coded = dec.encode(&[1, 0, 1]);
    /// assert_eq!(coded.len(), 2 * (3 + 6));  // 3 info + 6 tail
    /// ```
    pub fn encode(&self, info: &[u8]) -> Vec<u8> {
        let k = self.constraint_length;
        let total = info.len() + k.saturating_sub(1);
        let mut coded = Vec::with_capacity(total * 2);
        let mut state = 0usize;
        for i in 0..total {
            let b = if i < info.len() {
                (info[i] & 1) as usize
            } else {
                0
            };
            let o0 = self.trellis.out0[state * 2 + b];
            let o1 = self.trellis.out1[state * 2 + b];
            coded.push(o0);
            coded.push(o1);
            state = self.trellis.next_state[state * 2 + b] as usize;
        }
        coded
    }

    /// Hard-decision Viterbi decode.
    ///
    /// `coded` must contain flat pairs `[c0, c1, c0, c1, …]` of 0/1 values
    /// (length must be a multiple of 2).  The frame must be zero-terminated:
    /// the encoder appended $K-1$ tail zeros, so `coded.len()/2 ≥ K-1`.
    ///
    /// # Returns
    ///
    /// Decoded information bits (length = `coded.len()/2 - (K-1)`).
    ///
    /// # Examples
    ///
    /// ```
    /// use glezer_rsv::viterbi::ViterbiDecoder;
    /// let dec = ViterbiDecoder::new(7);
    /// let info = vec![1u8, 1, 0, 1, 0, 0, 1, 1];
    /// let coded = dec.encode(&info);
    /// assert_eq!(dec.decode_hard(&coded), info);
    /// ```
    pub fn decode_hard(&self, coded: &[u8]) -> Vec<u8> {
        #[cfg(target_arch = "x86_64")]
        if self.trellis.n_states == 64 && is_x86_feature_detected!("avx2") {
            return self.decode_hard_avx2(coded);
        }
        self.decode_hard_scalar(coded)
    }

    /// Scalar reference implementation of hard-decision decode (see
    /// [`decode_hard`](Self::decode_hard)). Kept `pub(crate)` so tests can
    /// compare it directly against [`decode_hard_avx2`](Self::decode_hard_avx2).
    pub(crate) fn decode_hard_scalar(&self, coded: &[u8]) -> Vec<u8> {
        let k = self.constraint_length;
        let n_states = self.trellis.n_states;
        let t_max = coded.len() / 2;
        let tail = k.saturating_sub(1);
        let n_info = t_max.saturating_sub(tail);
        if n_info == 0 {
            return vec![];
        }

        const INF: u32 = u32::MAX / 2;
        let mut cur_met = vec![INF; n_states];
        let mut nxt_met = vec![INF; n_states];
        cur_met[0] = 0;

        // traceback[t * n_states + s] = previous state that led to s at step t.
        let mut traceback = vec![0u8; n_states * t_max];

        for t in 0..t_max {
            let r0 = coded[2 * t] & 1;
            let r1 = coded[2 * t + 1] & 1;
            nxt_met.iter_mut().for_each(|m| *m = INF);
            for s in 0..n_states {
                if cur_met[s] == INF {
                    continue;
                }
                for b in 0..2usize {
                    let ns = self.trellis.next_state[s * 2 + b] as usize;
                    let o0 = self.trellis.out0[s * 2 + b];
                    let o1 = self.trellis.out1[s * 2 + b];
                    let bm = ((r0 ^ o0) + (r1 ^ o1)) as u32;
                    let new = cur_met[s].saturating_add(bm);
                    if new < nxt_met[ns] {
                        nxt_met[ns] = new;
                        traceback[t * n_states + ns] = s as u8;
                    }
                }
            }
            core::mem::swap(&mut cur_met, &mut nxt_met);
        }

        self.traceback_from_zero(t_max, n_info, &traceback)
    }

    /// AVX2-vectorised hard-decision decode for the K=7 (64-state) trellis.
    ///
    /// Uses `i32` Hamming path metrics (branch metrics are in `{0, 1, 2}`
    /// per step, so `i32` gives enormous headroom against overflow for any
    /// realistic frame length — a `u16` metric would need explicit
    /// saturation logic to stay safe on long frames, which AVX2 cannot do
    /// for `epi32`-width lanes without extra instructions, so plain
    /// non-saturating `i32` addition was chosen for simplicity and safety
    /// margin). See [`avx2_acs::acs_step_hard`] and [`ButterflyTables`] for
    /// the vectorisation strategy.
    ///
    /// # Safety requirement (caller-enforced)
    ///
    /// Only called from [`decode_hard`](Self::decode_hard) after checking
    /// `n_states == 64` and `is_x86_feature_detected!("avx2")`; exposed as
    /// `pub(crate)` for the equivalence tests, which perform the same check.
    #[cfg(target_arch = "x86_64")]
    pub(crate) fn decode_hard_avx2(&self, coded: &[u8]) -> Vec<u8> {
        debug_assert_eq!(self.trellis.n_states, 64, "AVX2 ACS path requires K=7");
        let k = self.constraint_length;
        let n_states = self.trellis.n_states;
        let t_max = coded.len() / 2;
        let tail = k.saturating_sub(1);
        let n_info = t_max.saturating_sub(tail);
        if n_info == 0 {
            return vec![];
        }

        // Generous headroom sentinel; see doc comment above for why `i32`
        // (not saturating, not `u16`) is safe here.
        const INF: i32 = i32::MAX / 4;
        let mut cur_met = vec![INF; n_states];
        let mut nxt_met = vec![INF; n_states];
        cur_met[0] = 0;

        let mut traceback = vec![0u8; n_states * t_max];
        let bfly = &self.trellis.bfly;

        for t in 0..t_max {
            let r0 = (coded[2 * t] & 1) as i32;
            let r1 = (coded[2 * t + 1] & 1) as i32;
            let row = &mut traceback[t * n_states..(t + 1) * n_states];
            // SAFETY: caller guarantees AVX2 support (checked in
            // `decode_hard`/test callers); `bfly.half == 32` for n_states==64.
            unsafe {
                avx2_acs::acs_step_hard(&cur_met, &mut nxt_met, row, bfly, r0, r1);
            }
            core::mem::swap(&mut cur_met, &mut nxt_met);
        }

        self.traceback_from_zero(t_max, n_info, &traceback)
    }

    /// Soft-decision Viterbi decode (max-log-MAP branch metric).
    ///
    /// `llr` contains paired LLR values `[L0, L1, L0, L1, …]` where
    /// $L_i > 0$ means bit $i$ is likely 0, $L_i < 0$ means likely 1.
    /// (Standard log-likelihood ratio convention.)
    ///
    /// The branch metric is $\sum_j (1 - 2\,c_j)\,L_j$ (maximized), which
    /// is the max-log-MAP approximation.
    ///
    /// # Returns
    ///
    /// Decoded information bits (length = `llr.len()/2 - (K-1)`).
    ///
    /// # Examples
    ///
    /// ```
    /// use glezer_rsv::viterbi::ViterbiDecoder;
    /// let dec = ViterbiDecoder::new(7);
    /// let info = vec![0u8; 8];
    /// let coded = dec.encode(&info);
    /// // Convert to soft LLR (error-free channel: 0→+10.0, 1→-10.0)
    /// let llr: Vec<f32> = coded.iter().map(|&b| if b == 0 { 10.0 } else { -10.0 }).collect();
    /// assert_eq!(dec.decode_soft(&llr), info);
    /// ```
    pub fn decode_soft(&self, llr: &[f32]) -> Vec<u8> {
        #[cfg(target_arch = "x86_64")]
        if self.trellis.n_states == 64 && is_x86_feature_detected!("avx2") {
            return self.decode_soft_avx2(llr);
        }
        self.decode_soft_scalar(llr)
    }

    /// Scalar reference implementation of soft-decision decode (see
    /// [`decode_soft`](Self::decode_soft)). Kept `pub(crate)` so tests can
    /// compare it directly against [`decode_soft_avx2`](Self::decode_soft_avx2).
    pub(crate) fn decode_soft_scalar(&self, llr: &[f32]) -> Vec<u8> {
        let k = self.constraint_length;
        let n_states = self.trellis.n_states;
        let t_max = llr.len() / 2;
        let tail = k.saturating_sub(1);
        let n_info = t_max.saturating_sub(tail);
        if n_info == 0 {
            return vec![];
        }

        let mut cur_met = vec![f32::NEG_INFINITY; n_states];
        let mut nxt_met = vec![f32::NEG_INFINITY; n_states];
        cur_met[0] = 0.0;

        let mut traceback = vec![0u8; n_states * t_max];

        for t in 0..t_max {
            let l0 = llr[2 * t];
            let l1 = llr[2 * t + 1];
            nxt_met.iter_mut().for_each(|m| *m = f32::NEG_INFINITY);
            for s in 0..n_states {
                if cur_met[s] == f32::NEG_INFINITY {
                    continue;
                }
                for b in 0..2usize {
                    let ns = self.trellis.next_state[s * 2 + b] as usize;
                    let o0 = self.trellis.out0[s * 2 + b] as f32;
                    let o1 = self.trellis.out1[s * 2 + b] as f32;
                    // Max-log-MAP: (1-2*c)*LLR, positively correlated with correctness.
                    let bm = (1.0 - 2.0 * o0) * l0 + (1.0 - 2.0 * o1) * l1;
                    let new = cur_met[s] + bm;
                    if new > nxt_met[ns] {
                        nxt_met[ns] = new;
                        traceback[t * n_states + ns] = s as u8;
                    }
                }
            }
            core::mem::swap(&mut cur_met, &mut nxt_met);
        }

        self.traceback_from_zero(t_max, n_info, &traceback)
    }

    /// AVX2-vectorised soft-decision (max-log-MAP) decode for the K=7
    /// (64-state) trellis.
    ///
    /// The branch metric $\mathrm{bm} = (1-2c_0)L_0 + (1-2c_1)L_1$ is fused
    /// directly into the vectorised ACS step (via precomputed $\pm 1$ sign
    /// tables built once per call below) rather than computed in a separate
    /// pass — profiling showed the scalar branch-metric arithmetic is not a
    /// separable cost, it is interleaved with the compare-select on every
    /// trellis edge, so fusing it avoids an extra memory round-trip.
    ///
    /// # Safety requirement (caller-enforced)
    ///
    /// Only called from [`decode_soft`](Self::decode_soft) after checking
    /// `n_states == 64` and `is_x86_feature_detected!("avx2")`; exposed as
    /// `pub(crate)` for the equivalence tests, which perform the same check.
    #[cfg(target_arch = "x86_64")]
    pub(crate) fn decode_soft_avx2(&self, llr: &[f32]) -> Vec<u8> {
        debug_assert_eq!(self.trellis.n_states, 64, "AVX2 ACS path requires K=7");
        let k = self.constraint_length;
        let n_states = self.trellis.n_states;
        let t_max = llr.len() / 2;
        let tail = k.saturating_sub(1);
        let n_info = t_max.saturating_sub(tail);
        if n_info == 0 {
            return vec![];
        }

        let bfly = &self.trellis.bfly;
        // Precompute the +-1 branch-sign tables once per call (not once per
        // bit): bm = sign0*l0 + sign1*l1 replaces the per-edge (1-2*out)*L
        // scalar formula with a fused multiply-add over constant vectors.
        let to_sign =
            |out: &[i32]| -> Vec<f32> { out.iter().map(|&o| 1.0 - 2.0 * o as f32).collect() };
        let sign0_even = to_sign(&bfly.out0_even);
        let sign1_even = to_sign(&bfly.out1_even);
        let sign0_odd = to_sign(&bfly.out0_odd);
        let sign1_odd = to_sign(&bfly.out1_odd);

        let mut cur_met = vec![f32::NEG_INFINITY; n_states];
        let mut nxt_met = vec![f32::NEG_INFINITY; n_states];
        cur_met[0] = 0.0;

        let mut traceback = vec![0u8; n_states * t_max];

        for t in 0..t_max {
            let l0 = llr[2 * t];
            let l1 = llr[2 * t + 1];
            let row = &mut traceback[t * n_states..(t + 1) * n_states];
            // SAFETY: caller guarantees AVX2 support (checked in
            // `decode_soft`/test callers); `bfly.half == 32` for n_states==64.
            unsafe {
                avx2_acs::acs_step_soft(
                    &cur_met,
                    &mut nxt_met,
                    row,
                    bfly,
                    &sign0_even,
                    &sign1_even,
                    &sign0_odd,
                    &sign1_odd,
                    l0,
                    l1,
                );
            }
            core::mem::swap(&mut cur_met, &mut nxt_met);
        }

        self.traceback_from_zero(t_max, n_info, &traceback)
    }

    /// Decode hard-decision coded bits (alias for [`ViterbiDecoder::decode_hard`]).
    ///
    /// Kept for backward compatibility with the original stub API.
    pub fn decode(&self, bits: &[u8]) -> Vec<u8> {
        self.decode_hard(bits)
    }

    // ---- private -----------------------------------------------------------

    /// Walk the traceback table from state 0 backward to recover decoded bits.
    ///
    /// `t_max` trellis steps; `n_info` is how many decoded bits to keep
    /// (steps 0..n_info); tail steps (n_info..t_max) are discarded.
    fn traceback_from_zero(&self, t_max: usize, n_info: usize, traceback: &[u8]) -> Vec<u8> {
        let n_states = self.trellis.n_states;
        let mut decoded = vec![0u8; n_info];
        let mut s = 0usize; // zero-terminated: start traceback at state 0

        for t in (0..t_max).rev() {
            let ps = traceback[t * n_states + s] as usize;
            // Recover input bit: check which branch (b=0 or b=1) from `ps` leads to `s`.
            let b = if self.trellis.next_state[ps * 2] as usize == s {
                0u8
            } else {
                1u8
            };
            if t < n_info {
                decoded[t] = b;
            }
            s = ps;
        }
        decoded
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn create_decoder() {
        let d = ViterbiDecoder::new(7);
        assert_eq!(d.constraint_length, 7);
    }

    proptest! {
        #[test]
        fn decode_empty_returns_empty(k in 1usize..10usize) {
            let d = ViterbiDecoder::new(k);
            let out = d.decode(&[]);
            prop_assert!(out.is_empty());
        }
    }

    #[test]
    fn encode_decode_hard_roundtrip_k7() {
        let dec = ViterbiDecoder::new(7);
        let info: Vec<u8> = vec![1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1];
        let coded = dec.encode(&info);
        // Encoded length = 2 * (n_info + K - 1)
        assert_eq!(coded.len(), 2 * (info.len() + 6));
        let decoded = dec.decode_hard(&coded);
        assert_eq!(decoded, info, "hard decode must exactly recover info bits");
    }

    #[test]
    fn encode_decode_soft_roundtrip_k7() {
        let dec = ViterbiDecoder::new(7);
        let info: Vec<u8> = vec![1, 0, 1, 1, 0, 0, 1];
        let coded = dec.encode(&info);
        // Error-free soft channel: 0-bit → +8.0 LLR, 1-bit → -8.0 LLR.
        let llr: Vec<f32> = coded
            .iter()
            .map(|&b| if b == 0 { 8.0 } else { -8.0 })
            .collect();
        let decoded = dec.decode_soft(&llr);
        assert_eq!(decoded, info, "soft decode must exactly recover info bits");
    }

    #[test]
    fn single_bit_error_corrected_k7() {
        // Introduce a single coded-bit error; K=7 code can correct it.
        let dec = ViterbiDecoder::new(7);
        let info: Vec<u8> = vec![1, 0, 1, 1, 0, 1, 0, 1, 1, 0];
        let mut coded = dec.encode(&info);
        coded[4] ^= 1; // flip one bit
        let decoded = dec.decode_hard(&coded);
        assert_eq!(
            decoded, info,
            "one-bit error must be corrected by K=7 decoder"
        );
    }

    #[test]
    fn all_zeros_encodes_to_all_zeros() {
        // All-zero input → all-zero codeword for any linear code.
        let dec = ViterbiDecoder::new(7);
        let coded = dec.encode(&[0u8; 8]);
        assert!(coded.iter().all(|&b| b == 0));
    }

    #[test]
    fn with_generators_k7_standard() {
        // Explicit generators must produce same result as default K=7.
        let dec1 = ViterbiDecoder::new(7);
        let dec2 = ViterbiDecoder::with_generators(7, 0o133, 0o171);
        let info: Vec<u8> = vec![1, 0, 0, 1, 1, 0, 1];
        assert_eq!(dec1.encode(&info), dec2.encode(&info));
        let coded = dec1.encode(&info);
        assert_eq!(dec1.decode_hard(&coded), dec2.decode_hard(&coded));
    }

    proptest! {
        #[test]
        fn encode_decode_roundtrip_arbitrary_bits(
            bits in prop::collection::vec(0u8..=1u8, 1..=32)
        ) {
            let dec = ViterbiDecoder::new(7);
            let coded = dec.encode(&bits);
            let decoded = dec.decode_hard(&coded);
            prop_assert_eq!(decoded, bits);
        }
    }

    // -----------------------------------------------------------------------
    // AVX2 vs. scalar equivalence (K=7 / 64-state ACS kernel)
    // -----------------------------------------------------------------------
    //
    // These tests only exercise the AVX2 path on hosts that actually have
    // AVX2 at runtime (checked explicitly, never assumed from
    // `target_feature` at compile time) so they degrade gracefully to a
    // no-op on non-AVX2 x86_64 hosts and are entirely absent on other
    // architectures.

    /// Small deterministic PRNG (splitmix64) so the equivalence tests are
    /// reproducible without adding a dependency on `rand`.
    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn decode_soft_avx2_matches_scalar_on_random_noisy_llrs() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("skipping: host has no AVX2");
            return;
        }
        let dec = ViterbiDecoder::new(7);
        let mut seed = 0xC0FF_EE00_1234_5678u64;
        for trial in 0..50 {
            let len = 100 + (splitmix64(&mut seed) as usize % 1901); // 100..=2000
            let info: Vec<u8> = (0..len)
                .map(|_| (splitmix64(&mut seed) & 1) as u8)
                .collect();
            let coded = dec.encode(&info);
            // Noisy soft channel: correct-sign LLR plus continuous noise in
            // [-1.2, 1.2), occasionally flipping the sign outright so both
            // decoders see genuinely ambiguous (tie-adjacent) metrics.
            let llr: Vec<f32> = coded
                .iter()
                .map(|&b| {
                    let base = if b == 0 { 3.0 } else { -3.0 };
                    let noise = (splitmix64(&mut seed) % 2401) as f32 / 1000.0 - 1.2;
                    base + noise
                })
                .collect();
            let scalar = dec.decode_soft_scalar(&llr);
            let avx2 = dec.decode_soft_avx2(&llr);
            assert_eq!(
                scalar, avx2,
                "trial {trial} (len={len}) soft decode mismatch between scalar and AVX2"
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn decode_hard_avx2_matches_scalar_on_random_bit_flips() {
        if !is_x86_feature_detected!("avx2") {
            eprintln!("skipping: host has no AVX2");
            return;
        }
        let dec = ViterbiDecoder::new(7);
        let mut seed = 0x1357_9BDF_2468_ACE0u64;
        for trial in 0..50 {
            let len = 100 + (splitmix64(&mut seed) as usize % 1901); // 100..=2000
            let info: Vec<u8> = (0..len)
                .map(|_| (splitmix64(&mut seed) & 1) as u8)
                .collect();
            let mut coded = dec.encode(&info);
            // Flip roughly 5% of coded bits at pseudo-random positions.
            for bit in coded.iter_mut() {
                if splitmix64(&mut seed) % 20 == 0 {
                    *bit ^= 1;
                }
            }
            let scalar = dec.decode_hard_scalar(&coded);
            let avx2 = dec.decode_hard_avx2(&coded);
            assert_eq!(
                scalar, avx2,
                "trial {trial} (len={len}) hard decode mismatch between scalar and AVX2"
            );
        }
    }

    #[test]
    fn decode_soft_moderate_noise_roundtrip_still_passes() {
        // Round-trip correctness at moderate noise (not just error-free or
        // single-bit-flip): a fixed, reproducible noisy LLR channel that
        // stays well within the K=7 code's correction capability.
        let dec = ViterbiDecoder::new(7);
        let mut seed = 0xFEED_FACE_C0DE_BEEFu64;
        let info: Vec<u8> = (0..500)
            .map(|_| (splitmix64(&mut seed) & 1) as u8)
            .collect();
        let coded = dec.encode(&info);
        let llr: Vec<f32> = coded
            .iter()
            .map(|&b| {
                let base = if b == 0 { 4.0 } else { -4.0 };
                let noise = (splitmix64(&mut seed) % 3001) as f32 / 1000.0 - 1.5;
                base + noise
            })
            .collect();
        let decoded = dec.decode_soft(&llr);
        assert_eq!(
            decoded, info,
            "moderate-noise soft decode must still recover info bits"
        );
    }
}
