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
    /// use syndrome::TurboEncoder;
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
    /// use syndrome::TurboEncoder;
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
    /// use syndrome::TurboEncoder;
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

    // Precomputed per-position branch-metric halves `a = 0.5*(sys+apriori)`
    // and `p = 0.5*par`, length `k + 3` (info positions + 3 termination
    // positions). Filled once per `siso_max_log_map` call and then reused,
    // unmodified, by the forward, backward, *and* LLR-extraction passes (and
    // by both the scalar and AVX2 kernels) instead of recomputing the same
    // arithmetic from `ch_sys`/`ch_par`/`apriori` on every trellis edge.
    gamma_a: Vec<f32>,
    gamma_p: Vec<f32>,
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
    /// use syndrome::TurboDecoder;
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
            gamma_a: vec![0.0; k + 3],
            gamma_p: vec![0.0; k + 3],
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
    /// use syndrome::{TurboDecoder, TurboEncoder};
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
        self.decode_with_backend(llr, out, max_iters, SisoBackend::Auto)
    }

    /// Identical to [`Self::decode`] but lets the caller force a specific
    /// SISO backend instead of the runtime auto-detection `decode` uses.
    ///
    /// This exists solely so the test suite can assert bit-for-bit (or
    /// near-bit-for-bit) equivalence between the scalar reference kernel and
    /// the AVX2 kernel on identical inputs; it is not part of the public API.
    #[cfg_attr(not(test), allow(dead_code))]
    fn decode_with_backend(
        &mut self,
        llr: &[f32],
        out: &mut [u8],
        max_iters: usize,
        backend: SisoBackend,
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
                &mut self.gamma_a,
                &mut self.gamma_p,
                &mut self.llr_total1,
                backend,
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
                &mut self.gamma_a,
                &mut self.gamma_p,
                &mut self.llr_total2,
                backend,
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

/// Which SISO kernel implementation to run.
///
/// `Auto` (used by the public [`TurboDecoder::decode`] through
/// [`TurboDecoder::decode_with_backend`]) picks the AVX2 kernel on x86_64
/// when the running CPU supports it, or the NEON kernel unconditionally on
/// AArch64 (NEON is mandatory there), falling back to the scalar reference
/// otherwise. `Scalar`, `Avx2`, and `Neon` force one specific backend; they
/// exist only so the test suite can assert equivalence between the kernels
/// on identical inputs and are never selected by production code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(not(test), allow(dead_code))]
enum SisoBackend {
    Auto,
    Scalar,
    Avx2,
    Neon,
}

/// Renormalization period (in trellis steps) for the forward/backward metric
/// recursions: every `NORM_PERIOD`-th step, the 8 state metrics of the row
/// just written are shifted by their common maximum (see
/// [`normalize_row`]). See that function's docs for why this is exact.
const NORM_PERIOD: usize = 64;

/// Subtract the row maximum from all 8 lanes of a just-written alpha/beta
/// row, in place.
///
/// # Numerical safety
///
/// Max-log-MAP forms its output ([`llr_out`](siso_max_log_map_scalar)) as a
/// *difference* `best0 - best1` of two path metrics, and each path metric is
/// built by repeatedly adding branch metrics ($\gamma$, unaffected by this
/// function) to a chain of $\alpha$ or $\beta$ values. If every one of the 8
/// states at trellis step $t$ is shifted by the *same* constant $C_t$ (which
/// is exactly what this function does — it subtracts one scalar from all 8
/// lanes), that constant propagates additively and identically through every
/// subsequent `max` in the recursion (because
/// $\max(x_1 - C, x_2 - C) = \max(x_1, x_2) - C$), and both `best0` and
/// `best1` in the final LLR difference draw from the *same* (possibly
/// shifted) $\alpha[t]$ row and the *same* $\beta[t+1]$ row. So any
/// accumulated shift cancels exactly in `best0 - best1`, regardless of how
/// the forward and backward renormalization schedules happen to align. This
/// is the standard periodic-renormalization trick used to bound metric
/// growth in log-domain BCJR/Viterbi decoders; here it guards against
/// metric magnitude growth over long blocks (large $K$) or many extrinsic
/// exchange iterations, at negligible cost (one 8-wide max reduction and one
/// subtraction every `NORM_PERIOD` steps, not every step).
///
/// The guard `m > NEG_INF * 0.5` skips normalization on a (never actually
/// reached, but defensively handled) row where every state is still the
/// pruned-path sentinel [`NEG_INF`]: subtracting `NEG_INF` from `NEG_INF`
/// would otherwise yield `0.0` and make an unreachable state look reachable.
#[inline(always)]
fn normalize_row(row: &mut [f32]) {
    let m = row.iter().copied().fold(NEG_INF, f32::max);
    if m > NEG_INF * 0.5 {
        for v in row.iter_mut() {
            *v -= m;
        }
    }
}

/// Branch metric $\gamma(b, z)$ given the precomputed halves
/// `a = 0.5*(sys+apriori)` and `p = 0.5*par` for the current trellis step
/// (see [`TurboDecoder::gamma_a`]/`gamma_p`).
///
/// Expands to exactly `0.5*sys_bp*(lsys+la) + 0.5*z_bp*lpar` (the original,
/// direct formula) for every `(b, z)` combination: only four distinct
/// branch-metric values exist per trellis step ($\pm a \pm p$), so instead
/// of recomputing that product/sum from scratch on every one of the (up to)
/// 16 edges touched per step, every call site precomputes `a`/`p` once and
/// this function just picks the right sign combination.
#[inline(always)]
fn gamma_ab(a: f32, p: f32, b: usize, z: u8) -> f32 {
    match (b, z) {
        (0, 0) => a + p,
        (0, _) => a - p,
        (_, 0) => -a + p,
        _ => -a - p,
    }
}

/// One max-log-MAP SISO (BCJR) pass over a single 8-state RSC constituent
/// trellis of `k` information positions followed by 3 termination positions.
///
/// Runs in $O(k \cdot 8)$ time using only the caller-provided buffers: no
/// heap allocation occurs inside this function. Dispatches to the AVX2
/// kernel ([`avx2_kernel::siso_max_log_map_avx2`]) when `backend` allows it
/// and the CPU supports AVX2, otherwise runs the scalar reference
/// ([`siso_max_log_map_scalar`]).
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
/// * `gamma_a`, `gamma_p` - Scratch, length `k + 3`; filled here from
///   `ch_sys`/`ch_par`/`apriori`/`tail_sys`/`tail_par` and then reused by
///   every pass (see [`gamma_ab`]).
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
    gamma_a: &mut [f32],
    gamma_p: &mut [f32],
    llr_out: &mut [f32],
    backend: SisoBackend,
) {
    // Precompute the two branch-metric halves once per trellis position;
    // shared by the forward, backward, and LLR-extraction passes below (and
    // by both the scalar and AVX2 kernels), instead of recomputing
    // `lsys + la` / `lpar` and the sign-multiplications on every trellis
    // edge (up to 16 edges per step) as the original per-edge closure did.
    for t in 0..k {
        gamma_a[t] = 0.5 * (ch_sys[t] + apriori[t]);
        gamma_p[t] = 0.5 * ch_par[t];
    }
    for i in 0..3 {
        gamma_a[k + i] = 0.5 * tail_sys[i];
        gamma_p[k + i] = 0.5 * tail_par[i];
    }

    match backend {
        SisoBackend::Scalar => {
            siso_max_log_map_scalar(k, gamma_a, gamma_p, alpha, beta, llr_out);
        }
        SisoBackend::Avx2 => {
            #[cfg(target_arch = "x86_64")]
            {
                assert!(
                    is_x86_feature_detected!("avx2"),
                    "SisoBackend::Avx2 forced but this CPU has no AVX2 support"
                );
                // SAFETY: AVX2 support asserted immediately above.
                unsafe {
                    avx2_kernel::siso_max_log_map_avx2(k, gamma_a, gamma_p, alpha, beta, llr_out);
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                siso_max_log_map_scalar(k, gamma_a, gamma_p, alpha, beta, llr_out);
            }
        }
        SisoBackend::Neon => {
            #[cfg(target_arch = "aarch64")]
            {
                // SAFETY: NEON is mandatory on AArch64.
                unsafe {
                    neon_kernel::siso_max_log_map_neon(k, gamma_a, gamma_p, alpha, beta, llr_out);
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                siso_max_log_map_scalar(k, gamma_a, gamma_p, alpha, beta, llr_out);
            }
        }
        SisoBackend::Auto => {
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") {
                    // SAFETY: AVX2 support checked immediately above.
                    unsafe {
                        avx2_kernel::siso_max_log_map_avx2(
                            k, gamma_a, gamma_p, alpha, beta, llr_out,
                        );
                    }
                    return;
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                // NEON is mandatory on AArch64; no runtime check needed.
                // SAFETY: see above.
                unsafe {
                    neon_kernel::siso_max_log_map_neon(k, gamma_a, gamma_p, alpha, beta, llr_out);
                }
                return;
            }
            #[allow(unreachable_code)]
            siso_max_log_map_scalar(k, gamma_a, gamma_p, alpha, beta, llr_out);
        }
    }
}

/// Scalar reference SISO kernel (kept as ground truth / non-x86_64
/// fallback). See [`siso_max_log_map`] for the shared contract; `gamma_a`
/// and `gamma_p` (length `k + 3`) must already be filled by the caller.
fn siso_max_log_map_scalar(
    k: usize,
    gamma_a: &[f32],
    gamma_p: &[f32],
    alpha: &mut [f32],
    beta: &mut [f32],
    llr_out: &mut [f32],
) {
    let (next_state, out_z) = RSC_TRELLIS;
    let n_ext = k + 3;

    // Forward pass: alpha[0] = log(1) at state 0. Only the `k` info-step
    // iterations are run: the 3 termination rows (alpha[k+1..=n_ext]) are
    // *never read* by anything (the LLR-extraction loop below only ever
    // indexes alpha[0..k]), so computing them would be pure dead work; this
    // holds for both the scalar and AVX2 kernels.
    for s in 0..8 {
        alpha[s] = if s == 0 { 0.0 } else { NEG_INF };
    }
    for t in 0..k {
        let a = gamma_a[t];
        let p = gamma_p[t];
        let base_out = (t + 1) * 8;
        for ns in 0..8 {
            alpha[base_out + ns] = NEG_INF;
        }
        let base_in = t * 8;
        for s in 0..8usize {
            let av = alpha[base_in + s];
            if av <= NEG_INF {
                continue;
            }
            for b in 0..2usize {
                let idx = s * 2 + b;
                let ns = next_state[idx] as usize;
                let z = out_z[idx];
                let cand = av + gamma_ab(a, p, b, z);
                if cand > alpha[base_out + ns] {
                    alpha[base_out + ns] = cand;
                }
            }
        }
        if (t + 1) % NORM_PERIOD == 0 {
            normalize_row(&mut alpha[base_out..base_out + 8]);
        }
    }

    // Backward pass: beta[n_ext] = log(1) at state 0 (forced termination).
    // Unlike alpha's tail rows, beta's tail rows *are* needed (they bootstrap
    // beta[k], which the LLR-extraction loop reads), so the full sweep runs.
    {
        let base = n_ext * 8;
        for s in 0..8 {
            beta[base + s] = if s == 0 { 0.0 } else { NEG_INF };
        }
    }
    for t in (0..n_ext).rev() {
        let is_tail = t >= k;
        let a = gamma_a[t];
        let p = gamma_p[t];
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
                let cand = gamma_ab(a, p, b, z) + beta[base_next + ns];
                if cand > beta[base_cur + s] {
                    beta[base_cur + s] = cand;
                }
            }
        }
        if t % NORM_PERIOD == 0 {
            normalize_row(&mut beta[base_cur..base_cur + 8]);
        }
    }

    // Total a posteriori LLR for each information position.
    for t in 0..k {
        let a = gamma_a[t];
        let p = gamma_p[t];
        let base_in = t * 8;
        let base_next = (t + 1) * 8;
        let mut best0 = NEG_INF;
        let mut best1 = NEG_INF;
        for s in 0..8usize {
            for b in 0..2usize {
                let idx = s * 2 + b;
                let ns = next_state[idx] as usize;
                let z = out_z[idx];
                let val = alpha[base_in + s] + gamma_ab(a, p, b, z) + beta[base_next + ns];
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
// AVX2 vectorized SISO kernel (x86_64 only)
// ---------------------------------------------------------------------------

/// AVX2-accelerated SISO kernel: all 8 trellis states for one time step fit
/// in a single 256-bit `f32x8` register, so each recursion step is one
/// gather (fixed permutation), one FMA-based branch-metric compute, one
/// `max` (or, for the 3 forced termination steps, one blend), and one store.
///
/// See the module-level trellis derivation in the source for how the fixed
/// permutation/sign tables below were obtained from [`RSC_TRELLIS`].
#[cfg(target_arch = "x86_64")]
mod avx2_kernel {
    use super::{NEG_INF, NORM_PERIOD, RSC_TRELLIS};
    use std::arch::x86_64::*;

    /// Fixed lookup tables for the AVX2 recursions, derived once (at compile
    /// time) from [`RSC_TRELLIS`] so there is a single source of truth for
    /// the trellis topology between the scalar and AVX2 kernels.
    ///
    /// For the `b=0`/`b=1` transition out of (or into) any state, the
    /// parity output `z` for `b=1` is always the complement of the `z` for
    /// `b=0` at the same lane (a structural property of this RSC trellis:
    /// flipping the input bit flips both the feedback and parity bits).
    /// Consequently `gamma(t, 1, ·) == -gamma(t, 0, ·)` at every lane, so
    /// only one sign table is needed per recursion (`alpha_sign0`,
    /// `beta_sign0`) and the `b=1` branch metric is a single sign-bit XOR of
    /// the `b=0` one — see [`siso_max_log_map_avx2`].
    struct Tables {
        /// `pred_b0[ns]` = source state whose `b=0` edge lands on `ns`.
        pred_b0: [i32; 8],
        /// `pred_b1[ns]` = source state whose `b=1` edge lands on `ns`.
        pred_b1: [i32; 8],
        /// `succ0[s]` = destination state of the `b=0` edge out of `s`.
        succ0: [i32; 8],
        /// `succ1[s]` = destination state of the `b=1` edge out of `s`.
        succ1: [i32; 8],
        /// `alpha_sign0[ns] = 1 - 2*z` for the `b=0` edge landing on `ns`.
        alpha_sign0: [f32; 8],
        /// `beta_sign0[s] = 1 - 2*z` for the `b=0` edge leaving `s`.
        beta_sign0: [f32; 8],
        /// All-ones (as i32 bit pattern) where the forced termination input
        /// bit for source state `s` is `0`, else all-zero. Used to `blendv`
        /// between the two branch candidates during the 3 forced
        /// termination steps of the backward recursion (only one of the two
        /// candidates is legal per source state during termination).
        beta_tail_mask0: [i32; 8],
    }

    const fn build_tables() -> Tables {
        let (next_state, out_z) = RSC_TRELLIS;
        let mut pred_b0 = [0i32; 8];
        let mut pred_b1 = [0i32; 8];
        let mut succ0 = [0i32; 8];
        let mut succ1 = [0i32; 8];
        let mut alpha_sign0 = [0f32; 8];
        let mut beta_sign0 = [0f32; 8];
        let mut beta_tail_mask0 = [0i32; 8];
        let mut s = 0usize;
        while s < 8 {
            let ns0 = next_state[s * 2] as usize;
            let z0 = out_z[s * 2];
            let ns1 = next_state[s * 2 + 1] as usize;
            succ0[s] = ns0 as i32;
            succ1[s] = ns1 as i32;
            let sign0 = 1.0 - 2.0 * (z0 as f32);
            beta_sign0[s] = sign0;
            pred_b0[ns0] = s as i32;
            pred_b1[ns1] = s as i32;
            alpha_sign0[ns0] = sign0;

            let s1 = (s >> 1) & 1;
            let s2 = (s >> 2) & 1;
            let forced_b = s1 ^ s2;
            beta_tail_mask0[s] = if forced_b == 0 { -1i32 } else { 0i32 };
            s += 1;
        }
        Tables {
            pred_b0,
            pred_b1,
            succ0,
            succ1,
            alpha_sign0,
            beta_sign0,
            beta_tail_mask0,
        }
    }

    const TABLES: Tables = build_tables();

    #[target_feature(enable = "avx2")]
    unsafe fn load_idx(t: &[i32; 8]) -> __m256i {
        _mm256_setr_epi32(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7])
    }

    #[target_feature(enable = "avx2")]
    unsafe fn load_signs(t: &[f32; 8]) -> __m256 {
        _mm256_setr_ps(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7])
    }

    /// Horizontal max of the 8 lanes of `v`. Order-independent (max is
    /// associative/commutative for the non-NaN finite values used here), so
    /// this yields the exact same value as the scalar kernel's sequential
    /// `>`-comparison reduction over the same 8 numbers.
    #[target_feature(enable = "avx2")]
    unsafe fn hmax(v: __m256) -> f32 {
        let hi = _mm256_extractf128_ps(v, 1);
        let lo = _mm256_castps256_ps128(v);
        let m1 = _mm_max_ps(hi, lo);
        let m2 = _mm_max_ps(m1, _mm_movehl_ps(m1, m1));
        let m3 = _mm_max_ss(m2, _mm_shuffle_ps(m2, m2, 1));
        _mm_cvtss_f32(m3)
    }

    /// Subtract the row max from all 8 lanes of `row` (AVX2 counterpart of
    /// [`super::normalize_row`]; see that function's docs for the numerical
    /// safety argument, which applies identically here).
    #[target_feature(enable = "avx2")]
    unsafe fn normalize_row_avx2(row: __m256) -> __m256 {
        unsafe {
            let m = hmax(row);
            if m > NEG_INF * 0.5 {
                _mm256_sub_ps(row, _mm256_set1_ps(m))
            } else {
                row
            }
        }
    }

    /// AVX2 SISO kernel. See [`super::siso_max_log_map`] for the shared
    /// contract; `gamma_a`/`gamma_p` (length `k + 3`) must already be filled
    /// by the caller.
    ///
    /// # Safety
    ///
    /// Caller must verify AVX2 support at runtime (e.g. via
    /// `is_x86_feature_detected!("avx2")`) before calling this function.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn siso_max_log_map_avx2(
        k: usize,
        gamma_a: &[f32],
        gamma_p: &[f32],
        alpha: &mut [f32],
        beta: &mut [f32],
        llr_out: &mut [f32],
    ) {
        // SAFETY: every intrinsic below requires AVX2, which the caller
        // guarantees; all slice indices are in-bounds given the length
        // contracts documented on `siso_max_log_map`.
        unsafe {
            let n_ext = k + 3;
            let sign_bit = _mm256_set1_ps(-0.0f32);

            let pred_b0_v = load_idx(&TABLES.pred_b0);
            let pred_b1_v = load_idx(&TABLES.pred_b1);
            let succ0_v = load_idx(&TABLES.succ0);
            let succ1_v = load_idx(&TABLES.succ1);
            let alpha_sign0_v = load_signs(&TABLES.alpha_sign0);
            let beta_sign0_v = load_signs(&TABLES.beta_sign0);
            let tail_mask0 = _mm256_castsi256_ps(load_idx(&TABLES.beta_tail_mask0));

            // ── Forward (alpha) recursion: info steps only (t = 0..k). ────
            // Tail rows are dead code — see the scalar kernel's comment.
            _mm256_storeu_ps(
                alpha.as_mut_ptr(),
                _mm256_setr_ps(
                    0.0, NEG_INF, NEG_INF, NEG_INF, NEG_INF, NEG_INF, NEG_INF, NEG_INF,
                ),
            );
            for t in 0..k {
                let a_v = _mm256_set1_ps(gamma_a[t]);
                let p_v = _mm256_set1_ps(gamma_p[t]);
                // gamma(0, ·) = a + p*sign0 (one lane per destination state);
                // gamma(1, ·) = -gamma(0, ·) exactly (sign-bit flip only —
                // see the `Tables` docs for why this holds per lane).
                // Not an FMA: `alpha_sign0_v` is always exactly +-1.0, so
                // `p_v * alpha_sign0_v` is an *exact* sign flip (no
                // rounding); a plain mul+add therefore has the same single
                // rounding step (on the add) as a fused multiply-add would,
                // giving bit-identical results without requiring the
                // separate "fma" target feature beyond plain AVX2.
                let gamma0 = _mm256_add_ps(_mm256_mul_ps(p_v, alpha_sign0_v), a_v);
                let gamma1 = _mm256_xor_ps(gamma0, sign_bit);

                let row = _mm256_loadu_ps(alpha[t * 8..].as_ptr());
                let cand0 = _mm256_add_ps(_mm256_permutevar8x32_ps(row, pred_b0_v), gamma0);
                let cand1 = _mm256_add_ps(_mm256_permutevar8x32_ps(row, pred_b1_v), gamma1);
                let mut new_row = _mm256_max_ps(cand0, cand1);
                if (t + 1) % NORM_PERIOD == 0 {
                    new_row = normalize_row_avx2(new_row);
                }
                _mm256_storeu_ps(alpha[(t + 1) * 8..].as_mut_ptr(), new_row);
            }

            // ── Backward (beta) recursion: full sweep, n_ext-1 downto 0. ──
            {
                let base = n_ext * 8;
                _mm256_storeu_ps(
                    beta[base..].as_mut_ptr(),
                    _mm256_setr_ps(
                        0.0, NEG_INF, NEG_INF, NEG_INF, NEG_INF, NEG_INF, NEG_INF, NEG_INF,
                    ),
                );
            }
            for t in (0..n_ext).rev() {
                let is_tail = t >= k;
                let a_v = _mm256_set1_ps(gamma_a[t]);
                let p_v = _mm256_set1_ps(gamma_p[t]);
                // See the forward-pass comment above: exact +-1.0 sign
                // multiply, so plain mul+add is bit-identical to an FMA here.
                let gamma0 = _mm256_add_ps(_mm256_mul_ps(p_v, beta_sign0_v), a_v);
                let gamma1 = _mm256_xor_ps(gamma0, sign_bit);

                let next_row = _mm256_loadu_ps(beta[(t + 1) * 8..].as_ptr());
                let cand0 = _mm256_add_ps(gamma0, _mm256_permutevar8x32_ps(next_row, succ0_v));
                let cand1 = _mm256_add_ps(gamma1, _mm256_permutevar8x32_ps(next_row, succ1_v));
                let mut new_row = if is_tail {
                    // Forced termination: exactly one of the two candidates
                    // is legal per source-state lane; select it directly
                    // instead of maxing (the illegal candidate must not
                    // participate at all).
                    _mm256_blendv_ps(cand1, cand0, tail_mask0)
                } else {
                    _mm256_max_ps(cand0, cand1)
                };
                if t % NORM_PERIOD == 0 {
                    new_row = normalize_row_avx2(new_row);
                }
                _mm256_storeu_ps(beta[t * 8..].as_mut_ptr(), new_row);
            }

            // ── LLR extraction: two masked max reductions per step. ───────
            for t in 0..k {
                let a_v = _mm256_set1_ps(gamma_a[t]);
                let p_v = _mm256_set1_ps(gamma_p[t]);
                // See the forward-pass comment above: exact +-1.0 sign
                // multiply, so plain mul+add is bit-identical to an FMA here.
                let gamma0 = _mm256_add_ps(_mm256_mul_ps(p_v, beta_sign0_v), a_v);
                let gamma1 = _mm256_xor_ps(gamma0, sign_bit);

                let alpha_row = _mm256_loadu_ps(alpha[t * 8..].as_ptr());
                let beta_next = _mm256_loadu_ps(beta[(t + 1) * 8..].as_ptr());
                let val0 = _mm256_add_ps(
                    _mm256_add_ps(alpha_row, gamma0),
                    _mm256_permutevar8x32_ps(beta_next, succ0_v),
                );
                let val1 = _mm256_add_ps(
                    _mm256_add_ps(alpha_row, gamma1),
                    _mm256_permutevar8x32_ps(beta_next, succ1_v),
                );
                let best0 = hmax(val0);
                let best1 = hmax(val1);
                llr_out[t] = best0 - best1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// NEON vectorized SISO kernel (AArch64 only)
// ---------------------------------------------------------------------------
//
// NEON counterpart of [`avx2_kernel`]: the 8 trellis states for one time
// step are split across two `float32x4_t` registers instead of AVX2's
// single `f32x8`. NEON is mandatory on AArch64, so no runtime detection is
// required (unlike the AVX2 kernel, which is gated behind
// `is_x86_feature_detected!`).
//
// CRITICAL: mirrors the AVX2 kernel's deliberate avoidance of FMA
// (`vfmaq_f32`) to stay bit-identical with the scalar reference — the sign
// multiplier (`alpha_sign0`/`beta_sign0`) is always exactly `+-1.0`, so a
// plain `vmulq_f32` + `vaddq_f32` rounds identically to the scalar `+`/`-`
// formula (single rounding step, same as an FMA would give, without
// requiring FMA).
#[cfg(target_arch = "aarch64")]
mod neon_kernel {
    use super::{NEG_INF, NORM_PERIOD, RSC_TRELLIS};
    use core::arch::aarch64::*;

    /// Fixed lookup tables for the NEON recursions, derived once (at compile
    /// time) from [`RSC_TRELLIS`] — the same topology [`super::avx2_kernel::Tables`]
    /// derives, just laid out for NEON's 2x4-lane split.
    ///
    /// AVX2 gathers a permuted row with a single `_mm256_permutevar8x32_*`
    /// (native 8-wide permute). NEON has no direct 8-wide `f32` permute, so
    /// the 8-state row is instead treated as a 32-byte table (two
    /// `float32x4_t` registers bit-cast to `uint8x16_t`) and gathered with
    /// `vqtbl2q_u8` (a 2-register, 32-entry byte-table lookup): each output
    /// lane's state index is expanded here, at compile time, into its 4
    /// constituent byte offsets (`4*state + 0..=3`), split into the low
    /// 4 output lanes (`_lo`) and high 4 output lanes (`_hi`) — see
    /// [`gather8`].
    struct Tables {
        pred_b0_idx_lo: [u8; 16],
        pred_b0_idx_hi: [u8; 16],
        pred_b1_idx_lo: [u8; 16],
        pred_b1_idx_hi: [u8; 16],
        succ0_idx_lo: [u8; 16],
        succ0_idx_hi: [u8; 16],
        succ1_idx_lo: [u8; 16],
        succ1_idx_hi: [u8; 16],
        alpha_sign0_lo: [f32; 4],
        alpha_sign0_hi: [f32; 4],
        beta_sign0_lo: [f32; 4],
        beta_sign0_hi: [f32; 4],
        /// All-ones (`u32` bit pattern, as `i32`) where the forced
        /// termination input bit for source state `s` is `0`, else
        /// all-zero; split lo/hi as above. See
        /// [`super::avx2_kernel::Tables::beta_tail_mask0`].
        beta_tail_mask0_lo: [i32; 4],
        beta_tail_mask0_hi: [i32; 4],
    }

    /// Expand an 8-entry state-index table into `vqtbl2q_u8` byte indices:
    /// `out[lane*4 + k] = 4*states[lane] + k` for `k` in `0..4`, split into
    /// the low 4 lanes (`lo`) and high 4 lanes (`hi`).
    const fn expand_byte_idx(states: &[i32; 8]) -> ([u8; 16], [u8; 16]) {
        let mut lo = [0u8; 16];
        let mut hi = [0u8; 16];
        let mut lane = 0usize;
        while lane < 4 {
            let base = (states[lane] as usize) * 4;
            let mut k = 0usize;
            while k < 4 {
                lo[lane * 4 + k] = (base + k) as u8;
                k += 1;
            }
            lane += 1;
        }
        while lane < 8 {
            let base = (states[lane] as usize) * 4;
            let mut k = 0usize;
            while k < 4 {
                hi[(lane - 4) * 4 + k] = (base + k) as u8;
                k += 1;
            }
            lane += 1;
        }
        (lo, hi)
    }

    const fn split4(v: &[f32; 8]) -> ([f32; 4], [f32; 4]) {
        ([v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]])
    }

    const fn split4i(v: &[i32; 8]) -> ([i32; 4], [i32; 4]) {
        ([v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]])
    }

    const fn build_tables() -> Tables {
        let (next_state, out_z) = RSC_TRELLIS;
        let mut pred_b0 = [0i32; 8];
        let mut pred_b1 = [0i32; 8];
        let mut succ0 = [0i32; 8];
        let mut succ1 = [0i32; 8];
        let mut alpha_sign0 = [0f32; 8];
        let mut beta_sign0 = [0f32; 8];
        let mut beta_tail_mask0 = [0i32; 8];
        let mut s = 0usize;
        while s < 8 {
            let ns0 = next_state[s * 2] as usize;
            let z0 = out_z[s * 2];
            let ns1 = next_state[s * 2 + 1] as usize;
            succ0[s] = ns0 as i32;
            succ1[s] = ns1 as i32;
            let sign0 = 1.0 - 2.0 * (z0 as f32);
            beta_sign0[s] = sign0;
            pred_b0[ns0] = s as i32;
            pred_b1[ns1] = s as i32;
            alpha_sign0[ns0] = sign0;

            let s1 = (s >> 1) & 1;
            let s2 = (s >> 2) & 1;
            let forced_b = s1 ^ s2;
            beta_tail_mask0[s] = if forced_b == 0 { -1i32 } else { 0i32 };
            s += 1;
        }

        let (pred_b0_idx_lo, pred_b0_idx_hi) = expand_byte_idx(&pred_b0);
        let (pred_b1_idx_lo, pred_b1_idx_hi) = expand_byte_idx(&pred_b1);
        let (succ0_idx_lo, succ0_idx_hi) = expand_byte_idx(&succ0);
        let (succ1_idx_lo, succ1_idx_hi) = expand_byte_idx(&succ1);
        let (alpha_sign0_lo, alpha_sign0_hi) = split4(&alpha_sign0);
        let (beta_sign0_lo, beta_sign0_hi) = split4(&beta_sign0);
        let (beta_tail_mask0_lo, beta_tail_mask0_hi) = split4i(&beta_tail_mask0);

        Tables {
            pred_b0_idx_lo,
            pred_b0_idx_hi,
            pred_b1_idx_lo,
            pred_b1_idx_hi,
            succ0_idx_lo,
            succ0_idx_hi,
            succ1_idx_lo,
            succ1_idx_hi,
            alpha_sign0_lo,
            alpha_sign0_hi,
            beta_sign0_lo,
            beta_sign0_hi,
            beta_tail_mask0_lo,
            beta_tail_mask0_hi,
        }
    }

    const TABLES: Tables = build_tables();

    /// Gather 8 lanes of `f32` (split across `row_lo`/`row_hi`) into a new
    /// `(lo, hi)` pair using two 16-byte `vqtbl2q_u8` byte-index tables. See
    /// [`Tables`]'s doc comment for why a byte-table lookup replaces AVX2's
    /// native 8-wide permute.
    ///
    /// # Safety
    ///
    /// NEON is guaranteed on AArch64; no runtime detection is required.
    #[inline(always)]
    unsafe fn gather8(
        row_lo: float32x4_t,
        row_hi: float32x4_t,
        idx_lo: uint8x16_t,
        idx_hi: uint8x16_t,
    ) -> (float32x4_t, float32x4_t) {
        unsafe {
            let table = uint8x16x2_t(vreinterpretq_u8_f32(row_lo), vreinterpretq_u8_f32(row_hi));
            let out_lo = vqtbl2q_u8(table, idx_lo);
            let out_hi = vqtbl2q_u8(table, idx_hi);
            (vreinterpretq_f32_u8(out_lo), vreinterpretq_f32_u8(out_hi))
        }
    }

    /// Horizontal max of 8 lanes split across two `float32x4_t` registers.
    /// Order-independent for the non-NaN finite values used here (same
    /// argument as [`super::avx2_kernel::hmax`]).
    ///
    /// # Safety
    ///
    /// NEON is guaranteed on AArch64; no runtime detection is required.
    #[inline(always)]
    unsafe fn hmax8(lo: float32x4_t, hi: float32x4_t) -> f32 {
        unsafe { vmaxvq_f32(vmaxq_f32(lo, hi)) }
    }

    /// Subtract the row max from all 8 lanes, split across `(lo, hi)`
    /// (NEON counterpart of [`super::normalize_row`]; see that function's
    /// docs for the numerical safety argument, which applies identically
    /// here).
    ///
    /// # Safety
    ///
    /// NEON is guaranteed on AArch64; no runtime detection is required.
    #[inline(always)]
    unsafe fn normalize_row_neon(lo: float32x4_t, hi: float32x4_t) -> (float32x4_t, float32x4_t) {
        unsafe {
            let m = hmax8(lo, hi);
            if m > NEG_INF * 0.5 {
                let mv = vdupq_n_f32(m);
                (vsubq_f32(lo, mv), vsubq_f32(hi, mv))
            } else {
                (lo, hi)
            }
        }
    }

    /// NEON SISO kernel. See [`super::siso_max_log_map`] for the shared
    /// contract; `gamma_a`/`gamma_p` (length `k + 3`) must already be filled
    /// by the caller.
    ///
    /// # Safety
    ///
    /// NEON is guaranteed on AArch64; no runtime detection is required.
    pub(super) unsafe fn siso_max_log_map_neon(
        k: usize,
        gamma_a: &[f32],
        gamma_p: &[f32],
        alpha: &mut [f32],
        beta: &mut [f32],
        llr_out: &mut [f32],
    ) {
        // SAFETY: every intrinsic below is unconditionally available on
        // AArch64 (NEON is mandatory); all slice indices are in-bounds
        // given the length contracts documented on `siso_max_log_map`.
        unsafe {
            let n_ext = k + 3;
            let sign_bit = vdupq_n_u32(0x8000_0000u32);

            let pred_b0_lo = vld1q_u8(TABLES.pred_b0_idx_lo.as_ptr());
            let pred_b0_hi = vld1q_u8(TABLES.pred_b0_idx_hi.as_ptr());
            let pred_b1_lo = vld1q_u8(TABLES.pred_b1_idx_lo.as_ptr());
            let pred_b1_hi = vld1q_u8(TABLES.pred_b1_idx_hi.as_ptr());
            let succ0_lo = vld1q_u8(TABLES.succ0_idx_lo.as_ptr());
            let succ0_hi = vld1q_u8(TABLES.succ0_idx_hi.as_ptr());
            let succ1_lo = vld1q_u8(TABLES.succ1_idx_lo.as_ptr());
            let succ1_hi = vld1q_u8(TABLES.succ1_idx_hi.as_ptr());
            let alpha_sign0_lo = vld1q_f32(TABLES.alpha_sign0_lo.as_ptr());
            let alpha_sign0_hi = vld1q_f32(TABLES.alpha_sign0_hi.as_ptr());
            let beta_sign0_lo = vld1q_f32(TABLES.beta_sign0_lo.as_ptr());
            let beta_sign0_hi = vld1q_f32(TABLES.beta_sign0_hi.as_ptr());
            let tail_mask0_lo =
                vreinterpretq_u32_s32(vld1q_s32(TABLES.beta_tail_mask0_lo.as_ptr()));
            let tail_mask0_hi =
                vreinterpretq_u32_s32(vld1q_s32(TABLES.beta_tail_mask0_hi.as_ptr()));

            // ── Forward (alpha) recursion: info steps only (t = 0..k). ────
            // Tail rows are dead code — see the scalar kernel's comment.
            {
                let init_lo_arr = [0.0f32, NEG_INF, NEG_INF, NEG_INF];
                vst1q_f32(alpha.as_mut_ptr(), vld1q_f32(init_lo_arr.as_ptr()));
                vst1q_f32(alpha[4..].as_mut_ptr(), vdupq_n_f32(NEG_INF));
            }
            for t in 0..k {
                let a_v = vdupq_n_f32(gamma_a[t]);
                let p_v = vdupq_n_f32(gamma_p[t]);
                // Not an FMA — see this module's doc comment for why plain
                // mul+add is bit-identical to an FMA here.
                let gamma0_lo = vaddq_f32(vmulq_f32(p_v, alpha_sign0_lo), a_v);
                let gamma0_hi = vaddq_f32(vmulq_f32(p_v, alpha_sign0_hi), a_v);
                let gamma1_lo =
                    vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(gamma0_lo), sign_bit));
                let gamma1_hi =
                    vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(gamma0_hi), sign_bit));

                let row_lo = vld1q_f32(alpha[t * 8..].as_ptr());
                let row_hi = vld1q_f32(alpha[t * 8 + 4..].as_ptr());
                let (g0_lo, g0_hi) = gather8(row_lo, row_hi, pred_b0_lo, pred_b0_hi);
                let (g1_lo, g1_hi) = gather8(row_lo, row_hi, pred_b1_lo, pred_b1_hi);

                let cand0_lo = vaddq_f32(g0_lo, gamma0_lo);
                let cand0_hi = vaddq_f32(g0_hi, gamma0_hi);
                let cand1_lo = vaddq_f32(g1_lo, gamma1_lo);
                let cand1_hi = vaddq_f32(g1_hi, gamma1_hi);

                let mut new_lo = vmaxq_f32(cand0_lo, cand1_lo);
                let mut new_hi = vmaxq_f32(cand0_hi, cand1_hi);
                if (t + 1) % NORM_PERIOD == 0 {
                    (new_lo, new_hi) = normalize_row_neon(new_lo, new_hi);
                }
                vst1q_f32(alpha[(t + 1) * 8..].as_mut_ptr(), new_lo);
                vst1q_f32(alpha[(t + 1) * 8 + 4..].as_mut_ptr(), new_hi);
            }

            // ── Backward (beta) recursion: full sweep, n_ext-1 downto 0. ──
            {
                let base = n_ext * 8;
                let init_lo_arr = [0.0f32, NEG_INF, NEG_INF, NEG_INF];
                vst1q_f32(beta[base..].as_mut_ptr(), vld1q_f32(init_lo_arr.as_ptr()));
                vst1q_f32(beta[base + 4..].as_mut_ptr(), vdupq_n_f32(NEG_INF));
            }
            for t in (0..n_ext).rev() {
                let is_tail = t >= k;
                let a_v = vdupq_n_f32(gamma_a[t]);
                let p_v = vdupq_n_f32(gamma_p[t]);
                // Not an FMA — see this module's doc comment.
                let gamma0_lo = vaddq_f32(vmulq_f32(p_v, beta_sign0_lo), a_v);
                let gamma0_hi = vaddq_f32(vmulq_f32(p_v, beta_sign0_hi), a_v);
                let gamma1_lo =
                    vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(gamma0_lo), sign_bit));
                let gamma1_hi =
                    vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(gamma0_hi), sign_bit));

                let next_lo = vld1q_f32(beta[(t + 1) * 8..].as_ptr());
                let next_hi = vld1q_f32(beta[(t + 1) * 8 + 4..].as_ptr());
                let (s0_lo, s0_hi) = gather8(next_lo, next_hi, succ0_lo, succ0_hi);
                let (s1_lo, s1_hi) = gather8(next_lo, next_hi, succ1_lo, succ1_hi);

                let cand0_lo = vaddq_f32(gamma0_lo, s0_lo);
                let cand0_hi = vaddq_f32(gamma0_hi, s0_hi);
                let cand1_lo = vaddq_f32(gamma1_lo, s1_lo);
                let cand1_hi = vaddq_f32(gamma1_hi, s1_hi);

                let (mut new_lo, mut new_hi) = if is_tail {
                    // Forced termination: exactly one of the two candidates
                    // is legal per source-state lane; select it directly
                    // instead of maxing (the illegal candidate must not
                    // participate at all).
                    (
                        vbslq_f32(tail_mask0_lo, cand0_lo, cand1_lo),
                        vbslq_f32(tail_mask0_hi, cand0_hi, cand1_hi),
                    )
                } else {
                    (vmaxq_f32(cand0_lo, cand1_lo), vmaxq_f32(cand0_hi, cand1_hi))
                };
                if t % NORM_PERIOD == 0 {
                    (new_lo, new_hi) = normalize_row_neon(new_lo, new_hi);
                }
                vst1q_f32(beta[t * 8..].as_mut_ptr(), new_lo);
                vst1q_f32(beta[t * 8 + 4..].as_mut_ptr(), new_hi);
            }

            // ── LLR extraction: two masked max reductions per step. ───────
            for t in 0..k {
                let a_v = vdupq_n_f32(gamma_a[t]);
                let p_v = vdupq_n_f32(gamma_p[t]);
                // Not an FMA — see this module's doc comment.
                let gamma0_lo = vaddq_f32(vmulq_f32(p_v, beta_sign0_lo), a_v);
                let gamma0_hi = vaddq_f32(vmulq_f32(p_v, beta_sign0_hi), a_v);
                let gamma1_lo =
                    vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(gamma0_lo), sign_bit));
                let gamma1_hi =
                    vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(gamma0_hi), sign_bit));

                let alpha_lo = vld1q_f32(alpha[t * 8..].as_ptr());
                let alpha_hi = vld1q_f32(alpha[t * 8 + 4..].as_ptr());
                let beta_next_lo = vld1q_f32(beta[(t + 1) * 8..].as_ptr());
                let beta_next_hi = vld1q_f32(beta[(t + 1) * 8 + 4..].as_ptr());
                let (s0_lo, s0_hi) = gather8(beta_next_lo, beta_next_hi, succ0_lo, succ0_hi);
                let (s1_lo, s1_hi) = gather8(beta_next_lo, beta_next_hi, succ1_lo, succ1_hi);

                let val0_lo = vaddq_f32(vaddq_f32(alpha_lo, gamma0_lo), s0_lo);
                let val0_hi = vaddq_f32(vaddq_f32(alpha_hi, gamma0_hi), s0_hi);
                let val1_lo = vaddq_f32(vaddq_f32(alpha_lo, gamma1_lo), s1_lo);
                let val1_hi = vaddq_f32(vaddq_f32(alpha_hi, gamma1_hi), s1_hi);

                let best0 = hmax8(val0_lo, val0_hi);
                let best1 = hmax8(val1_lo, val1_hi);
                llr_out[t] = best0 - best1;
            }
        }
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

    /// Whether this CPU can actually run the NEON SISO kernel; the
    /// equivalence tests below are skipped (not failed) when it can't, so
    /// the suite stays green on non-AArch64 CI runners. NEON is mandatory
    /// on AArch64 (no runtime feature check needed, unlike AVX2).
    fn neon_available() -> bool {
        cfg!(target_arch = "aarch64")
    }

    /// Whether this CPU can actually run the AVX2 SISO kernel; the
    /// equivalence tests below are skipped (not failed) when it can't, so
    /// the suite stays green on non-x86_64 CI runners or old x86_64 CPUs.
    fn avx2_available() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            is_x86_feature_detected!("avx2")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    /// Small deterministic xorshift PRNG for generating varied seeded test
    /// data without pulling in a `rand` dependency (matches the style of
    /// `random_bits` in `src/bin/algo_bench_export.rs`).
    fn xorshift_bits(len: usize, mut seed: u64) -> Vec<u8> {
        if seed == 0 {
            seed = 0x9E37_79B9_7F4A_7C15;
        }
        (0..len)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed >> 63) as u8
            })
            .collect()
    }

    /// AVX2 vs scalar SISO-kernel equivalence.
    ///
    /// The AVX2 restructuring in [`avx2_kernel`] was designed to be
    /// bit-identical to the scalar reference for the forward/backward/LLR
    /// passes themselves (see the derivation comments on `Tables` and on the
    /// `gamma0`/`gamma1` computations: the sign multiplier is always exactly
    /// `+-1.0`, so `mul+add` rounds identically to the scalar `+`/`-`
    /// formula, and `max`/horizontal-max are order-independent for the
    /// non-NaN values used here). The periodic renormalization
    /// ([`NORM_PERIOD`]) is applied identically by both backends, so it does
    /// not break this equivalence either.
    ///
    /// This test still checks both hard-decision equality *and* a numeric
    /// tolerance on the total a posteriori LLR (rather than relying solely
    /// on the bit-identical design argument), across 100 seeded AWGN trials
    /// spanning 1.0-2.5 dB Eb/N0, for every supported block length that
    /// realistically appears in the mandate: K = 40 (short), 1024 (typical
    /// LTE), and 6144 (max LTE Turbo block).
    #[test]
    fn simd_scalar_equivalence_100_trials() {
        if !avx2_available() {
            eprintln!("skipping simd_scalar_equivalence_100_trials: no AVX2 on this CPU");
            return;
        }
        const TRIALS: u64 = 100;
        const MAX_ITERS: usize = 8;

        for &k in &[40usize, 1024, 6144] {
            let enc = TurboEncoder::new(k).unwrap();
            let mut dec_scalar = TurboDecoder::new(k).unwrap();
            let mut dec_avx2 = TurboDecoder::new(k).unwrap();

            for trial in 0..TRIALS {
                let snr_db = 1.0f32 + 1.5f32 * (trial as f32) / (TRIALS as f32 - 1.0);
                let info = xorshift_bits(k, 0xD1B5_4A32 ^ (k as u64) << 32 ^ trial);
                let mut coded = vec![0u8; enc.output_len()];
                enc.encode(&info, &mut coded).unwrap();

                let mut ch = AwgnChannel::new(snr_db, 1.0 / 3.0, 10_000 + trial);
                let llr = ch.transmit(&coded);

                let mut out_scalar = vec![0u8; k];
                let mut out_avx2 = vec![0u8; k];
                let iters_scalar = dec_scalar
                    .decode_with_backend(&llr, &mut out_scalar, MAX_ITERS, SisoBackend::Scalar)
                    .unwrap();
                let iters_avx2 = dec_avx2
                    .decode_with_backend(&llr, &mut out_avx2, MAX_ITERS, SisoBackend::Avx2)
                    .unwrap();

                assert_eq!(
                    iters_scalar, iters_avx2,
                    "K={k} trial={trial} snr={snr_db:.3}dB: iteration counts differ \
                     (scalar={iters_scalar}, avx2={iters_avx2})"
                );
                assert_eq!(
                    out_scalar, out_avx2,
                    "K={k} trial={trial} snr={snr_db:.3}dB: hard decisions differ between backends"
                );

                let max_abs_diff = dec_scalar
                    .llr_total1
                    .iter()
                    .zip(dec_avx2.llr_total1.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_abs_diff < 1e-3,
                    "K={k} trial={trial} snr={snr_db:.3}dB: max abs extrinsic/LLR diff \
                     {max_abs_diff} >= 1e-3"
                );
            }
        }
    }

    /// Forcing the AVX2 backend on noiseless input (perfect LLRs) must
    /// still round-trip exactly, for every supported block length — this
    /// exercises the AVX2 kernel's termination/tail handling directly
    /// (rather than only through `Auto` dispatch) across all QPP sizes, not
    /// just the three checked by the main equivalence test.
    #[test]
    fn avx2_noiseless_roundtrip_all_sizes() {
        if !avx2_available() {
            eprintln!("skipping avx2_noiseless_roundtrip_all_sizes: no AVX2 on this CPU");
            return;
        }
        for &(k, _, _) in QPP_TABLE {
            let enc = TurboEncoder::new(k).unwrap();
            let mut dec = TurboDecoder::new(k).unwrap();
            let info = xorshift_bits(k, 0x1234_5678 ^ (k as u64));
            let mut coded = vec![0u8; enc.output_len()];
            enc.encode(&info, &mut coded).unwrap();

            let llr: Vec<f32> = coded
                .iter()
                .map(|&b| if b == 0 { 12.0 } else { -12.0 })
                .collect();
            let mut out = vec![0u8; k];
            dec.decode_with_backend(&llr, &mut out, 8, SisoBackend::Avx2)
                .unwrap();
            assert_eq!(
                out, info,
                "K={k}: AVX2-forced noiseless round-trip must be bit-exact"
            );
        }
    }

    // -----------------------------------------------------------------------
    // NEON vs. scalar equivalence (AArch64 SISO kernel)
    // -----------------------------------------------------------------------
    //
    // NEON is mandatory on AArch64 (no runtime detection to skip), so these
    // tests run unconditionally there and are entirely absent on other
    // architectures — mirrors the AVX2 equivalence tests above.

    /// NEON counterpart of [`simd_scalar_equivalence_100_trials`]: the NEON
    /// restructuring in [`neon_kernel`] is designed to be bit-identical to
    /// the scalar reference for the same reasons documented on that test
    /// (fixed compile-time gather tables, exact +-1.0 sign multipliers so
    /// mul+add rounds identically to an FMA, order-independent max/hmax).
    #[test]
    fn simd_scalar_equivalence_100_trials_neon() {
        if !neon_available() {
            eprintln!("skipping simd_scalar_equivalence_100_trials_neon: not AArch64");
            return;
        }
        const TRIALS: u64 = 100;
        const MAX_ITERS: usize = 8;

        for &k in &[40usize, 1024, 6144] {
            let enc = TurboEncoder::new(k).unwrap();
            let mut dec_scalar = TurboDecoder::new(k).unwrap();
            let mut dec_neon = TurboDecoder::new(k).unwrap();

            for trial in 0..TRIALS {
                let snr_db = 1.0f32 + 1.5f32 * (trial as f32) / (TRIALS as f32 - 1.0);
                let info = xorshift_bits(k, 0xD1B5_4A32 ^ (k as u64) << 32 ^ trial);
                let mut coded = vec![0u8; enc.output_len()];
                enc.encode(&info, &mut coded).unwrap();

                let mut ch = AwgnChannel::new(snr_db, 1.0 / 3.0, 10_000 + trial);
                let llr = ch.transmit(&coded);

                let mut out_scalar = vec![0u8; k];
                let mut out_neon = vec![0u8; k];
                let iters_scalar = dec_scalar
                    .decode_with_backend(&llr, &mut out_scalar, MAX_ITERS, SisoBackend::Scalar)
                    .unwrap();
                let iters_neon = dec_neon
                    .decode_with_backend(&llr, &mut out_neon, MAX_ITERS, SisoBackend::Neon)
                    .unwrap();

                assert_eq!(
                    iters_scalar, iters_neon,
                    "K={k} trial={trial} snr={snr_db:.3}dB: iteration counts differ \
                     (scalar={iters_scalar}, neon={iters_neon})"
                );
                assert_eq!(
                    out_scalar, out_neon,
                    "K={k} trial={trial} snr={snr_db:.3}dB: hard decisions differ between backends"
                );

                let max_abs_diff = dec_scalar
                    .llr_total1
                    .iter()
                    .zip(dec_neon.llr_total1.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_abs_diff < 1e-3,
                    "K={k} trial={trial} snr={snr_db:.3}dB: max abs extrinsic/LLR diff \
                     {max_abs_diff} >= 1e-3"
                );
            }
        }
    }

    /// NEON counterpart of [`avx2_noiseless_roundtrip_all_sizes`]: forcing
    /// the NEON backend on noiseless input (perfect LLRs) must still
    /// round-trip exactly, for every supported block length.
    #[test]
    fn neon_noiseless_roundtrip_all_sizes() {
        if !neon_available() {
            eprintln!("skipping neon_noiseless_roundtrip_all_sizes: not AArch64");
            return;
        }
        for &(k, _, _) in QPP_TABLE {
            let enc = TurboEncoder::new(k).unwrap();
            let mut dec = TurboDecoder::new(k).unwrap();
            let info = xorshift_bits(k, 0x1234_5678 ^ (k as u64));
            let mut coded = vec![0u8; enc.output_len()];
            enc.encode(&info, &mut coded).unwrap();

            let llr: Vec<f32> = coded
                .iter()
                .map(|&b| if b == 0 { 12.0 } else { -12.0 })
                .collect();
            let mut out = vec![0u8; k];
            dec.decode_with_backend(&llr, &mut out, 8, SisoBackend::Neon)
                .unwrap();
            assert_eq!(
                out, info,
                "K={k}: NEON-forced noiseless round-trip must be bit-exact"
            );
        }
    }

    /// Informational, not a correctness check: prints scalar-vs-AVX2
    /// decode throughput attribution at K=1024/8 iters/1.5 dB (same shape as
    /// `src/bin/algo_bench_export.rs`'s Turbo case). `#[ignore]`d so it
    /// doesn't run as part of the normal test gate; run explicitly with
    /// `cargo test --release turbo::tests::bench_backend_attribution -- \
    /// --ignored --nocapture`.
    #[test]
    #[ignore]
    fn bench_backend_attribution() {
        if !avx2_available() {
            eprintln!("no AVX2 on this CPU; only the scalar backend can be measured");
        }
        let k = 1024usize;
        let enc = TurboEncoder::new(k).unwrap();
        let info = xorshift_bits(k, 19);
        let mut coded = vec![0u8; enc.output_len()];
        enc.encode(&info, &mut coded).unwrap();
        let mut ch = AwgnChannel::new(1.5, 1.0 / 3.0, 21);
        let llr = ch.transmit(&coded);

        let backends: &[(&str, SisoBackend)] = &[
            ("scalar", SisoBackend::Scalar),
            ("avx2", SisoBackend::Avx2),
            ("auto", SisoBackend::Auto),
        ];
        for &(name, backend) in backends {
            if backend == SisoBackend::Avx2 && !avx2_available() {
                continue;
            }
            let mut dec = TurboDecoder::new(k).unwrap();
            let mut out = vec![0u8; k];
            for _ in 0..3 {
                dec.decode_with_backend(&llr, &mut out, 8, backend).unwrap();
            }
            let iters = 30;
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                dec.decode_with_backend(&llr, &mut out, 8, backend).unwrap();
            }
            let ns = t0.elapsed().as_nanos() as f64 / iters as f64;
            let mbit_s = k as f64 / ns * 1e3;
            eprintln!("backend={name:<6} ns/call={ns:>10.1} mbit/s={mbit_s:>8.3}");
        }
    }
}
