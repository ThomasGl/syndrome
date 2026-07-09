//! LTE-style rate-1/3 parallel-concatenated convolutional (Turbo) code.
//!
//! Implements the 3GPP TS 36.212 §5.1.3.2 Turbo encoder (two identical 8-state
//! recursive systematic convolutional (RSC) constituent encoders joined by a
//! quadratic permutation polynomial (QPP) interleaver) together with an
//! iterative max-log-MAP (BCJR) decoder.
//!
//! # Constituent code
//!
//! Each constituent encoder implements the transfer function
//! $$ G(D) = \left[1, \frac{g_1(D)}{g_0(D)}\right], \quad
//!    g_0(D) = 1 + D^2 + D^3 \;(\text{octal } 13), \quad
//!    g_1(D) = 1 + D + D^3 \;(\text{octal } 15). $$
//! The 3-bit shift register $s = (s_0, s_1, s_2) = (d_{k-1}, d_{k-2}, d_{k-3})$
//! gives 8 trellis states.  At each step, given input bit $x_k$:
//! $$ d_k = x_k \oplus s_1 \oplus s_2 \quad (\text{feedback, from } g_0), $$
//! $$ z_k = d_k \oplus s_0 \oplus s_2 \quad (\text{parity, from } g_1), $$
//! and the register shifts: $s_2 \leftarrow s_1,\; s_1 \leftarrow s_0,\; s_0 \leftarrow d_k$.
//!
//! Encoder 1 processes the $K$ information bits in natural order; encoder 2
//! processes the same bits permuted through the QPP interleaver
//! $$ \Pi(i) = (f_1 \cdot i + f_2 \cdot i^2) \bmod K. $$
//!
//! # Bit-stream layout
//!
//! `encode` writes `3*K + 12` bits: for $k = 0 \ldots K-1$ the triplet
//! `[x_k, z_k, z'_k]` (systematic, encoder-1 parity, encoder-2 parity), followed
//! by 12 trellis-termination bits — three `[x, z]` pairs from encoder 1's own
//! forced tail, then three `[x', z']` pairs from encoder 2's own forced tail.
//! This layout is internally consistent between `encode` and `decode` and
//! matches the general TS 36.212 structure (systematic + two parity streams,
//! 12 termination bits); it does **not** attempt to reproduce the exact
//! wire-level tail bit multiplexing of §5.1.3.2.2, nor the downstream
//! sub-block interleaving / rate matching of §5.1.4, which are out of scope
//! for a standalone constituent-code module.
//!
//! # QPP interleaver parameters
//!
//! The $(f_1, f_2)$ pairs below are exactly 3GPP TS 36.212 Table 5.1.3-3,
//! cross-checked against the independent open-source reproduction of that
//! table at
//! <https://github.com/robmaunder/turbo-3gpp-matlab/blob/master/internal_interleaver.m>
//! (Robert G. Maunder, University of Southampton — a line-for-line lookup
//! table matching the standard), and independently corroborated for the
//! $K=6144$ row (an easily-checked outlier value) via public patent and
//! research-paper citations. Only the sizes verified this way are supported;
//! no interleaver parameters were invented.
//!
//! | $K$  | $f_1$ | $f_2$ |
//! |------|-------|-------|
//! | 40   | 3     | 10    |
//! | 104  | 7     | 26    |
//! | 256  | 15    | 32    |
//! | 512  | 31    | 64    |
//! | 1024 | 31    | 64    |
//! | 2048 | 31    | 64    |
//! | 4096 | 31    | 64    |
//! | 6144 | 263   | 480   |
//!
//! # LLR sign convention
//!
//! Matches [`crate::channel_sim::AwgnChannel`] and [`crate::qc_ldpc`]: a
//! **positive** LLR favours bit `0`, a **negative** LLR favours bit `1`.
//!
//! # Decoder
//!
//! The two constituent SISO decoders run the max-log-MAP approximation of
//! BCJR (forward $\alpha$, backward $\beta$, branch metric $\gamma$) and
//! exchange extrinsic information through the same QPP interleaver, scaled by
//! the empirical factor $0.75$ to compensate the systematic optimism of the
//! max-log approximation (Vogt & Finger, "Extrinsic information scaling for
//! turbo codes," 2000). All trellis buffers ($\alpha$, $\beta$, extrinsic and
//! interleaved LLR arrays) are preallocated once in [`TurboDecoder::new`] and
//! reused across calls to [`TurboDecoder::decode`]; the per-iteration loop
//! performs no heap allocation. Each half-iteration is
//! $O(K \cdot 8)$, so `decode` runs in $O(K \cdot 8 \cdot \text{iters})$ time.

use crate::error::FecError;

/// Extrinsic-information damping factor for max-log-MAP turbo decoding.
///
/// The max-log approximation of BCJR systematically overestimates the
/// reliability of extrinsic information; scaling it by a constant close to
/// but below 1 recovers most of the loss relative to true log-MAP. See
/// J. Vogt and A. Finger, "Improving the max-log-MAP turbo decoder," IEE
/// Electronics Letters, 2000.
const EXTRINSIC_SCALE: f32 = 0.75;

/// A very large finite "negative infinity" stand-in used for pruned trellis
/// paths. Using a large finite value instead of `f32::NEG_INFINITY` avoids
/// producing `NaN` from `(-inf) - (-inf)` in degenerate corner cases.
const NEG_INF: f32 = -1.0e30;

// ---------------------------------------------------------------------------
// QPP interleaver parameters (3GPP TS 36.212 Table 5.1.3-3, verified subset)
// ---------------------------------------------------------------------------

/// Supported `(K, f1, f2)` rows, verified against the sources cited in the
/// module documentation. Only these block lengths are supported.
const QPP_TABLE: &[(usize, usize, usize)] = &[
    (40, 3, 10),
    (104, 7, 26),
    (256, 15, 32),
    (512, 31, 64),
    (1024, 31, 64),
    (2048, 31, 64),
    (4096, 31, 64),
    (6144, 263, 480),
];

/// Look up the QPP parameters `(f1, f2)` for a supported block length `k`.
fn qpp_params(k: usize) -> Option<(usize, usize)> {
    QPP_TABLE
        .iter()
        .find(|&&(kk, _, _)| kk == k)
        .map(|&(_, f1, f2)| (f1, f2))
}

/// Build the QPP interleaver permutation $\Pi(i) = (f_1 i + f_2 i^2) \bmod K$
/// for $i = 0 \ldots K-1$.
///
/// Computed once at construction time; the returned `Vec<usize>` is owned by
/// the caller's encoder/decoder struct and reused for the lifetime of that
/// struct (never reallocated per-encode/decode).
fn build_qpp_interleaver(k: usize, f1: usize, f2: usize) -> Vec<usize> {
    let k64 = k as u64;
    let f1 = f1 as u64;
    let f2 = f2 as u64;
    (0..k as u64)
        .map(|i| ((f1 * i + f2 * i * i) % k64) as usize)
        .collect()
}

// ---------------------------------------------------------------------------
// 8-state RSC constituent trellis (built once, at compile time)
// ---------------------------------------------------------------------------

/// Build the 8-state RSC trellis transition tables for $g_0 = 13_8$,
/// $g_1 = 15_8$.
///
/// Returns `(next_state, out_z)`, each indexed as `table[state * 2 + input_bit]`.
/// The systematic output equals the input bit itself and needs no table.
const fn build_rsc_trellis() -> ([u8; 16], [u8; 16]) {
    let mut next_state = [0u8; 16];
    let mut out_z = [0u8; 16];
    let mut s = 0usize;
    while s < 8 {
        let s0 = (s & 1) as u8;
        let s1 = ((s >> 1) & 1) as u8;
        let s2 = ((s >> 2) & 1) as u8;
        let mut b = 0usize;
        while b < 2 {
            let bb = b as u8;
            let d = bb ^ s1 ^ s2; // feedback, from g0 = 1 + D^2 + D^3
            let z = d ^ s0 ^ s2; // parity, from g1 = 1 + D + D^3
            let ns = (d as usize) | ((s0 as usize) << 1) | ((s1 as usize) << 2);
            next_state[s * 2 + b] = ns as u8;
            out_z[s * 2 + b] = z;
            b += 1;
        }
        s += 1;
    }
    (next_state, out_z)
}

/// Compile-time constant RSC trellis shared by the encoder and decoder.
const RSC_TRELLIS: ([u8; 16], [u8; 16]) = build_rsc_trellis();

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/// LTE-style rate-1/3 parallel-concatenated (Turbo) convolutional encoder.
///
/// See the module documentation for the constituent code, bit-stream layout,
/// and QPP interleaver parameters.
pub struct TurboEncoder {
    /// Number of information bits $K$ per code block.
    k: usize,
    /// QPP interleaver permutation, length `k`.
    pi: Vec<usize>,
}

impl TurboEncoder {
    /// Create an encoder for `k` information bits.
    ///
    /// # Arguments
    ///
    /// * `k` - Number of information bits per block. Must be one of the
    ///   3GPP TS 36.212 QPP sizes supported by this module (see module docs).
    ///
    /// # Errors
    ///
    /// Returns [`FecError::InvalidParam`] if `k` is not a supported block length.
    ///
    /// # Examples
    ///
    /// ```
    /// use glezer_rsv::TurboEncoder;
    /// let enc = TurboEncoder::new(40).unwrap();
    /// assert_eq!(enc.output_len(), 3 * 40 + 12);
    /// ```
    pub fn new(k: usize) -> Result<Self, FecError> {
        let (f1, f2) =
            qpp_params(k).ok_or(FecError::InvalidParam("unsupported turbo block length K"))?;
        let pi = build_qpp_interleaver(k, f1, f2);
        Ok(Self { k, pi })
    }

    /// Number of information bits this encoder was constructed for.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Total number of coded output bits: $3K + 12$.
    ///
    /// # Returns
    ///
    /// The required length of the `out` slice passed to [`Self::encode`].
    ///
    /// # Examples
    ///
    /// ```
    /// use glezer_rsv::TurboEncoder;
    /// let enc = TurboEncoder::new(104).unwrap();
    /// assert_eq!(enc.output_len(), 3 * 104 + 12);
    /// ```
    pub fn output_len(&self) -> usize {
        3 * self.k + 12
    }

    /// Encode `info` (length `k`, values `{0, 1}`) into the rate-1/3 Turbo
    /// codeword `out` (length [`Self::output_len`]).
    ///
    /// # Arguments
    ///
    /// * `info` - Information bits, length exactly `k`.
    /// * `out`  - Output buffer, length at least [`Self::output_len`]; only
    ///   the first `output_len()` entries are written.
    ///
    /// # Errors
    ///
    /// * [`FecError::InvalidParam`] if `info.len() != k`.
    /// * [`FecError::BufferTooSmall`] if `out` is shorter than `output_len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use glezer_rsv::TurboEncoder;
    /// let enc = TurboEncoder::new(40).unwrap();
    /// let info = vec![1u8, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0,
    ///                  1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0];
    /// let mut out = vec![0u8; enc.output_len()];
    /// enc.encode(&info, &mut out).unwrap();
    /// assert_eq!(out.len(), enc.output_len());
    /// ```
    pub fn encode(&self, info: &[u8], out: &mut [u8]) -> Result<(), FecError> {
        let k = self.k;
        if info.len() != k {
            return Err(FecError::InvalidParam("info length must equal K"));
        }
        let required = self.output_len();
        if out.len() < required {
            return Err(FecError::BufferTooSmall {
                required,
                provided: out.len(),
            });
        }
        let (next_state, out_z) = RSC_TRELLIS;

        // Encoder 1: natural order.
        let mut state1 = 0usize;
        for t in 0..k {
            let b = (info[t] & 1) as usize;
            let idx = state1 * 2 + b;
            out[3 * t] = b as u8;
            out[3 * t + 1] = out_z[idx];
            state1 = next_state[idx] as usize;
        }

        // Encoder 2: QPP-interleaved order. Only its parity is transmitted
        // in the body (its systematic output duplicates natural-order info).
        let mut state2 = 0usize;
        for t in 0..k {
            let b = (info[self.pi[t]] & 1) as usize;
            let idx = state2 * 2 + b;
            out[3 * t + 2] = out_z[idx];
            state2 = next_state[idx] as usize;
        }

        // Tail: 3 forced termination steps per encoder, driving each
        // register back to state 0 (feedback bit forced to zero).
        let tail1_base = 3 * k;
        for i in 0..3usize {
            let s1 = (state1 >> 1) & 1;
            let s2 = (state1 >> 2) & 1;
            let b = s1 ^ s2;
            let idx = state1 * 2 + b;
            out[tail1_base + 2 * i] = b as u8;
            out[tail1_base + 2 * i + 1] = out_z[idx];
            state1 = next_state[idx] as usize;
        }
        let tail2_base = tail1_base + 6;
        for i in 0..3usize {
            let s1 = (state2 >> 1) & 1;
            let s2 = (state2 >> 2) & 1;
            let b = s1 ^ s2;
            let idx = state2 * 2 + b;
            out[tail2_base + 2 * i] = b as u8;
            out[tail2_base + 2 * i + 1] = out_z[idx];
            state2 = next_state[idx] as usize;
        }

        debug_assert_eq!(state1, 0, "encoder 1 must terminate in state 0");
        debug_assert_eq!(state2, 0, "encoder 2 must terminate in state 0");
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

/// Iterative max-log-MAP (BCJR) Turbo decoder.
///
/// All scratch buffers (forward/backward trellis metrics, extrinsic and
/// interleaved LLR arrays, hard-decision buffers) are allocated once in
/// [`Self::new`] and reused by every call to [`Self::decode`]; the
/// per-iteration loop performs no heap allocation. See the module
/// documentation for the LLR sign convention, extrinsic scaling, and
/// complexity.
pub struct TurboDecoder {
    k: usize,
    pi: Vec<usize>,

    // Parsed channel LLRs (filled once per `decode` call).
    ch_sys: Vec<f32>,    // natural-order systematic LLR, length k
    ch_par1: Vec<f32>,   // encoder-1 parity LLR, length k
    ch_par2: Vec<f32>,   // encoder-2 parity LLR, length k
    ch_sys_il: Vec<f32>, // ch_sys gathered through pi, length k
    tail_sys1: [f32; 3],
    tail_par1: [f32; 3],
    tail_sys2: [f32; 3],
    tail_par2: [f32; 3],

    // Per-iteration exchange buffers.
    apriori1: Vec<f32>,       // natural order, length k
    apriori2: Vec<f32>,       // interleaved order, length k
    extrinsic1: Vec<f32>,     // natural order, length k
    extrinsic2_nat: Vec<f32>, // decoder-2 extrinsic mapped back to natural order
    llr_total1: Vec<f32>,     // decoder-1 a posteriori LLR, length k
    llr_total2: Vec<f32>,     // decoder-2 a posteriori LLR (interleaved order), length k
    hard1: Vec<u8>,
    hard2_nat: Vec<u8>,

    // BCJR trellis metrics, shared scratch reused by both constituent
    // decoders (sized for k info steps + 3 termination steps).
    alpha: Vec<f32>,
    beta: Vec<f32>,
}

impl TurboDecoder {
    /// Create a decoder for `k` information bits.
    ///
    /// # Arguments
    ///
    /// * `k` - Number of information bits per block; must match the value
    ///   used by the corresponding [`TurboEncoder`] and be one of the
    ///   supported QPP sizes (see module docs).
    ///
    /// # Errors
    ///
    /// Returns [`FecError::InvalidParam`] if `k` is not a supported block length.
    ///
    /// # Examples
    ///
    /// ```
    /// use glezer_rsv::TurboDecoder;
    /// let dec = TurboDecoder::new(40).unwrap();
    /// ```
    pub fn new(k: usize) -> Result<Self, FecError> {
        let (f1, f2) =
            qpp_params(k).ok_or(FecError::InvalidParam("unsupported turbo block length K"))?;
        let pi = build_qpp_interleaver(k, f1, f2);
        let n_alpha_beta = (k + 4) * 8;
        Ok(Self {
            k,
            pi,
            ch_sys: vec![0.0; k],
            ch_par1: vec![0.0; k],
            ch_par2: vec![0.0; k],
            ch_sys_il: vec![0.0; k],
            tail_sys1: [0.0; 3],
            tail_par1: [0.0; 3],
            tail_sys2: [0.0; 3],
            tail_par2: [0.0; 3],
            apriori1: vec![0.0; k],
            apriori2: vec![0.0; k],
            extrinsic1: vec![0.0; k],
            extrinsic2_nat: vec![0.0; k],
            llr_total1: vec![0.0; k],
            llr_total2: vec![0.0; k],
            hard1: vec![0u8; k],
            hard2_nat: vec![0u8; k],
            alpha: vec![0.0; n_alpha_beta],
            beta: vec![0.0; n_alpha_beta],
        })
    }

    /// Number of information bits this decoder was constructed for.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Iteratively decode a rate-1/3 Turbo codeword.
    ///
    /// `llr` holds `3*k + 12` channel log-likelihood ratios in the same
    /// layout produced by [`TurboEncoder::encode`] (systematic/parity-1/
    /// parity-2 triplets followed by 12 termination-bit LLRs). A positive
    /// LLR favours bit `0`; a negative LLR favours bit `1` (see module docs).
    ///
    /// Runs up to `max_iters` full iterations (each consisting of two SISO
    /// half-iterations, one per constituent decoder), exiting early as soon
    /// as both half-iterations agree on every hard decision.
    ///
    /// # Arguments
    ///
    /// * `llr`        - Channel LLRs, length at least `3*k + 12`.
    /// * `out`        - Decoded information bits, length at least `k`.
    /// * `max_iters`  - Maximum number of full iterations to run (must be $\ge 1$).
    ///
    /// # Returns
    ///
    /// The number of full iterations actually performed (`Ok(iters)`,
    /// `1 <= iters <= max_iters`).
    ///
    /// # Errors
    ///
    /// * [`FecError::InvalidParam`] if `max_iters == 0`.
    /// * [`FecError::BufferTooSmall`] if `llr` or `out` is too short.
    ///
    /// # Examples
    ///
    /// ```
    /// use glezer_rsv::{TurboDecoder, TurboEncoder};
    ///
    /// let k = 40;
    /// let enc = TurboEncoder::new(k).unwrap();
    /// let mut dec = TurboDecoder::new(k).unwrap();
    /// let info: Vec<u8> = (0..k).map(|i| (i % 3 == 0) as u8).collect();
    /// let mut coded = vec![0u8; enc.output_len()];
    /// enc.encode(&info, &mut coded).unwrap();
    ///
    /// // Noiseless channel: map bit 0 -> +10.0, bit 1 -> -10.0.
    /// let llr: Vec<f32> = coded.iter().map(|&b| if b == 0 { 10.0 } else { -10.0 }).collect();
    /// let mut out = vec![0u8; k];
    /// let iters = dec.decode(&llr, &mut out, 8).unwrap();
    /// assert_eq!(out, info);
    /// assert!(iters <= 8);
    /// ```
    pub fn decode(
        &mut self,
        llr: &[f32],
        out: &mut [u8],
        max_iters: usize,
    ) -> Result<usize, FecError> {
        let k = self.k;
        if max_iters == 0 {
            return Err(FecError::InvalidParam("max_iters must be >= 1"));
        }
        let required = 3 * k + 12;
        if llr.len() < required {
            return Err(FecError::BufferTooSmall {
                required,
                provided: llr.len(),
            });
        }
        if out.len() < k {
            return Err(FecError::BufferTooSmall {
                required: k,
                provided: out.len(),
            });
        }

        for t in 0..k {
            self.ch_sys[t] = llr[3 * t];
            self.ch_par1[t] = llr[3 * t + 1];
            self.ch_par2[t] = llr[3 * t + 2];
        }
        let tail1_base = 3 * k;
        for i in 0..3usize {
            self.tail_sys1[i] = llr[tail1_base + 2 * i];
            self.tail_par1[i] = llr[tail1_base + 2 * i + 1];
        }
        let tail2_base = tail1_base + 6;
        for i in 0..3usize {
            self.tail_sys2[i] = llr[tail2_base + 2 * i];
            self.tail_par2[i] = llr[tail2_base + 2 * i + 1];
        }
        for i in 0..k {
            self.ch_sys_il[i] = self.ch_sys[self.pi[i]];
        }
        self.apriori1.fill(0.0);

        let mut iters_used = 0usize;
        for _iter in 0..max_iters {
            // --- constituent decoder 1: natural order ---
            siso_max_log_map(
                k,
                &self.ch_sys,
                &self.ch_par1,
                &self.tail_sys1,
                &self.tail_par1,
                &self.apriori1,
                &mut self.alpha,
                &mut self.beta,
                &mut self.llr_total1,
            );
            for t in 0..k {
                self.extrinsic1[t] =
                    EXTRINSIC_SCALE * (self.llr_total1[t] - self.ch_sys[t] - self.apriori1[t]);
                self.hard1[t] = (self.llr_total1[t] < 0.0) as u8;
            }

            // --- constituent decoder 2: QPP-interleaved order ---
            for i in 0..k {
                self.apriori2[i] = self.extrinsic1[self.pi[i]];
            }
            siso_max_log_map(
                k,
                &self.ch_sys_il,
                &self.ch_par2,
                &self.tail_sys2,
                &self.tail_par2,
                &self.apriori2,
                &mut self.alpha,
                &mut self.beta,
                &mut self.llr_total2,
            );
            for i in 0..k {
                let e2 =
                    EXTRINSIC_SCALE * (self.llr_total2[i] - self.ch_sys_il[i] - self.apriori2[i]);
                let nat = self.pi[i];
                self.extrinsic2_nat[nat] = e2;
                self.hard2_nat[nat] = (self.llr_total2[i] < 0.0) as u8;
            }
            self.apriori1.copy_from_slice(&self.extrinsic2_nat);

            iters_used += 1;
            if self.hard1 == self.hard2_nat {
                break;
            }
        }

        out[..k].copy_from_slice(&self.hard1);
        Ok(iters_used)
    }
}

/// One max-log-MAP SISO (BCJR) pass over a single 8-state RSC constituent
/// trellis of `k` information positions followed by 3 termination positions.
///
/// Runs in $O(k \cdot 8)$ time using only the caller-provided buffers: no
/// heap allocation occurs inside this function.
///
/// * `ch_sys`, `ch_par` - Channel LLRs for the `k` information positions
///   (systematic and this encoder's parity, respectively), in this
///   decoder's own bit order.
/// * `tail_sys`, `tail_par` - Channel LLRs for this encoder's own 3
///   termination positions.
/// * `apriori` - A priori (extrinsic-from-the-other-decoder) LLR for the `k`
///   information positions; termination positions have no a priori term.
/// * `alpha`, `beta` - Scratch trellis metrics, length `(k + 4) * 8`,
///   indexed `[time * 8 + state]`.
/// * `llr_out` - Receives the total a posteriori LLR for the `k` information
///   positions.
#[allow(clippy::too_many_arguments)]
fn siso_max_log_map(
    k: usize,
    ch_sys: &[f32],
    ch_par: &[f32],
    tail_sys: &[f32; 3],
    tail_par: &[f32; 3],
    apriori: &[f32],
    alpha: &mut [f32],
    beta: &mut [f32],
    llr_out: &mut [f32],
) {
    let (next_state, out_z) = RSC_TRELLIS;
    let n_ext = k + 3;

    // Branch metric gamma for a transition with input bit `b` and parity
    // output `z`, at trellis time `t`.
    let gamma_at = |t: usize, b: usize, z: u8| -> f32 {
        let (lsys, lpar, la) = if t >= k {
            (tail_sys[t - k], tail_par[t - k], 0.0)
        } else {
            (ch_sys[t], ch_par[t], apriori[t])
        };
        let sys_bp = 1.0 - 2.0 * (b as f32);
        let z_bp = 1.0 - 2.0 * (z as f32);
        0.5 * sys_bp * (lsys + la) + 0.5 * z_bp * lpar
    };

    // Forward pass: alpha[0] = log(1) at state 0.
    for s in 0..8 {
        alpha[s] = if s == 0 { 0.0 } else { NEG_INF };
    }
    for t in 0..n_ext {
        let is_tail = t >= k;
        let base_out = (t + 1) * 8;
        for ns in 0..8 {
            alpha[base_out + ns] = NEG_INF;
        }
        let base_in = t * 8;
        for s in 0..8usize {
            let a = alpha[base_in + s];
            if a <= NEG_INF {
                continue;
            }
            let s1 = (s >> 1) & 1;
            let s2 = (s >> 2) & 1;
            let (lo, hi) = if is_tail {
                let b = s1 ^ s2;
                (b, b)
            } else {
                (0usize, 1usize)
            };
            for b in lo..=hi {
                let idx = s * 2 + b;
                let ns = next_state[idx] as usize;
                let z = out_z[idx];
                let cand = a + gamma_at(t, b, z);
                if cand > alpha[base_out + ns] {
                    alpha[base_out + ns] = cand;
                }
            }
        }
    }

    // Backward pass: beta[n_ext] = log(1) at state 0 (forced termination).
    {
        let base = n_ext * 8;
        for s in 0..8 {
            beta[base + s] = if s == 0 { 0.0 } else { NEG_INF };
        }
    }
    for t in (0..n_ext).rev() {
        let is_tail = t >= k;
        let base_cur = t * 8;
        for s in 0..8 {
            beta[base_cur + s] = NEG_INF;
        }
        let base_next = (t + 1) * 8;
        for s in 0..8usize {
            let s1 = (s >> 1) & 1;
            let s2 = (s >> 2) & 1;
            let (lo, hi) = if is_tail {
                let b = s1 ^ s2;
                (b, b)
            } else {
                (0usize, 1usize)
            };
            for b in lo..=hi {
                let idx = s * 2 + b;
                let ns = next_state[idx] as usize;
                let z = out_z[idx];
                let cand = gamma_at(t, b, z) + beta[base_next + ns];
                if cand > beta[base_cur + s] {
                    beta[base_cur + s] = cand;
                }
            }
        }
    }

    // Total a posteriori LLR for each information position.
    for t in 0..k {
        let base_in = t * 8;
        let base_next = (t + 1) * 8;
        let mut best0 = NEG_INF;
        let mut best1 = NEG_INF;
        for s in 0..8usize {
            for b in 0..2usize {
                let idx = s * 2 + b;
                let ns = next_state[idx] as usize;
                let z = out_z[idx];
                let val = alpha[base_in + s] + gamma_at(t, b, z) + beta[base_next + ns];
                if b == 0 {
                    if val > best0 {
                        best0 = val;
                    }
                } else if val > best1 {
                    best1 = val;
                }
            }
        }
        llr_out[t] = best0 - best1;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel_sim::AwgnChannel;

    /// Every supported (K, f1, f2) row must yield a valid permutation of
    /// `0..K` (each output value appears exactly once).
    #[test]
    fn qpp_tables_are_valid_permutations() {
        for &(k, f1, f2) in QPP_TABLE {
            let pi = build_qpp_interleaver(k, f1, f2);
            assert_eq!(pi.len(), k);
            let mut sorted = pi.clone();
            sorted.sort_unstable();
            let expected: Vec<usize> = (0..k).collect();
            assert_eq!(
                sorted, expected,
                "K={k} f1={f1} f2={f2} is not a valid permutation"
            );
        }
    }

    /// The RSC constituent trellis must terminate both encoders in state 0
    /// for arbitrary information bits (verified by replaying the same
    /// trellis walk `encode` uses, directly against the private trellis
    /// table).
    #[test]
    fn tail_termination_reaches_state_zero() {
        let k = 104;
        let (f1, f2) = qpp_params(k).unwrap();
        let pi = build_qpp_interleaver(k, f1, f2);
        let (next_state, out_z) = RSC_TRELLIS;
        let info: Vec<u8> = (0..k).map(|i| ((i * 7 + 3) % 5 == 0) as u8).collect();

        let mut state1 = 0usize;
        for t in 0..k {
            let b = (info[t] & 1) as usize;
            state1 = next_state[state1 * 2 + b] as usize;
        }
        for _ in 0..3 {
            let s1 = (state1 >> 1) & 1;
            let s2 = (state1 >> 2) & 1;
            let b = s1 ^ s2;
            state1 = next_state[state1 * 2 + b] as usize;
        }
        assert_eq!(state1, 0, "encoder 1 must terminate in state 0");

        let mut state2 = 0usize;
        for t in 0..k {
            let b = (info[pi[t]] & 1) as usize;
            state2 = next_state[state2 * 2 + b] as usize;
        }
        for _ in 0..3 {
            let s1 = (state2 >> 1) & 1;
            let s2 = (state2 >> 2) & 1;
            let b = s1 ^ s2;
            state2 = next_state[state2 * 2 + b] as usize;
        }
        assert_eq!(state2, 0, "encoder 2 must terminate in state 0");

        // out_z is exercised implicitly by build_rsc_trellis's own
        // consistency (used identically by encode/decode); touch it so the
        // table is not considered dead code under any feature combination.
        let _ = out_z[0];
    }

    /// Noiseless round-trip: perfect LLRs must decode bit-exactly for a
    /// small (K=40) and a large (K=1024) block length.
    #[test]
    fn noiseless_roundtrip_k40_and_k1024() {
        for &k in &[40usize, 1024] {
            let enc = TurboEncoder::new(k).unwrap();
            let mut dec = TurboDecoder::new(k).unwrap();
            let info: Vec<u8> = (0..k).map(|i| ((i * 13 + 5) % 7 < 3) as u8).collect();
            let mut coded = vec![0u8; enc.output_len()];
            enc.encode(&info, &mut coded).unwrap();

            let llr: Vec<f32> = coded
                .iter()
                .map(|&b| if b == 0 { 12.0 } else { -12.0 })
                .collect();
            let mut out = vec![0u8; k];
            let iters = dec.decode(&llr, &mut out, 8).unwrap();
            assert_eq!(out, info, "K={k} noiseless round-trip must be bit-exact");
            assert!((1..=8).contains(&iters));
        }
    }

    /// AWGN round-trip at a comfortable Eb/N0 for rate-1/3, K=1024: must
    /// reconstruct exactly, and must stay fast in debug builds.
    #[test]
    fn awgn_roundtrip_k1024_exact() {
        let k = 1024;
        let enc = TurboEncoder::new(k).unwrap();
        let mut dec = TurboDecoder::new(k).unwrap();
        let info: Vec<u8> = (0..k).map(|i| ((i * 97 + 11) % 5 < 2) as u8).collect();
        let mut coded = vec![0u8; enc.output_len()];
        enc.encode(&info, &mut coded).unwrap();

        let mut ch = AwgnChannel::new(3.0, 1.0 / 3.0, 42);
        let llr = ch.transmit(&coded);

        let mut out = vec![0u8; k];
        let iters = dec.decode(&llr, &mut out, 12).unwrap();
        assert_eq!(out, info, "AWGN round-trip at 3 dB Eb/N0 must be bit-exact");
        assert!((1..=12).contains(&iters));
    }

    /// At high SNR, the decoder should converge (and early-exit) in fewer
    /// than the maximum allowed iterations.
    #[test]
    fn early_exit_at_high_snr() {
        let k = 256;
        let enc = TurboEncoder::new(k).unwrap();
        let mut dec = TurboDecoder::new(k).unwrap();
        let info: Vec<u8> = (0..k).map(|i| ((i * 31 + 1) % 4 == 0) as u8).collect();
        let mut coded = vec![0u8; enc.output_len()];
        enc.encode(&info, &mut coded).unwrap();

        let mut ch = AwgnChannel::new(8.0, 1.0 / 3.0, 7);
        let llr = ch.transmit(&coded);

        let max_iters = 16;
        let mut out = vec![0u8; k];
        let iters = dec.decode(&llr, &mut out, max_iters).unwrap();
        assert_eq!(out, info);
        assert!(
            iters < max_iters,
            "expected early exit at high SNR, used {iters}/{max_iters} iterations"
        );
    }

    /// Rejects an unsupported block length.
    #[test]
    fn unsupported_k_rejected() {
        assert!(TurboEncoder::new(41).is_err());
        assert!(TurboDecoder::new(41).is_err());
    }

    /// Rejects mismatched buffer lengths.
    #[test]
    fn encode_rejects_bad_lengths() {
        let enc = TurboEncoder::new(40).unwrap();
        let info = vec![0u8; 40];
        let mut short_out = vec![0u8; 4];
        assert!(enc.encode(&info, &mut short_out).is_err());
        let wrong_info = vec![0u8; 39];
        let mut out = vec![0u8; enc.output_len()];
        assert!(enc.encode(&wrong_info, &mut out).is_err());
    }

    /// Rejects `max_iters == 0` and undersized decode buffers.
    #[test]
    fn decode_rejects_bad_arguments() {
        let k = 40;
        let enc = TurboEncoder::new(k).unwrap();
        let mut dec = TurboDecoder::new(k).unwrap();
        let info = vec![0u8; k];
        let mut coded = vec![0u8; enc.output_len()];
        enc.encode(&info, &mut coded).unwrap();
        let llr: Vec<f32> = coded
            .iter()
            .map(|&b| if b == 0 { 10.0 } else { -10.0 })
            .collect();

        let mut out = vec![0u8; k];
        assert!(dec.decode(&llr, &mut out, 0).is_err());

        let mut short_llr = llr.clone();
        short_llr.truncate(3);
        assert!(dec.decode(&short_llr, &mut out, 4).is_err());

        let mut short_out = vec![0u8; 2];
        assert!(dec.decode(&llr, &mut short_out, 4).is_err());
    }
}
