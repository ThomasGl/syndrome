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
        TrellisTable {
            n_states,
            next_state,
            out0,
            out1,
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
    /// standard generator polynomials for that `k` (see [`default_generators`]).
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

    /// Decode hard-decision coded bits (alias for [`decode_hard`]).
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
}
