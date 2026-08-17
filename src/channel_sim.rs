//! BPSK AWGN channel simulator for deterministic FEC testing.
//!
//! Provides a self-contained additive white Gaussian noise (AWGN) channel model
//! using BPSK modulation. All randomness comes from a deterministic xorshift64 PRNG
//! so that simulation results are reproducible across runs without any external crate.
//!
//! # Signal model
//!
//! A BPSK symbol is mapped as:
//! $$x = \begin{cases} +1 & \text{if bit} = 0 \\ -1 & \text{if bit} = 1 \end{cases}$$
//!
//! After passing through an AWGN channel with noise standard deviation $\sigma$:
//! $$r = x + n, \quad n \sim \mathcal{N}(0, \sigma^2)$$
//!
//! The soft log-likelihood ratio (LLR) assuming equal priors is:
//! $$\text{LLR} = \frac{2r}{\sigma^2}$$
//!
//! The noise standard deviation is derived from the $E_b/N_0$ (energy per bit to
//! noise spectral density ratio) and the code rate $R$:
//! $$\sigma = \sqrt{\frac{1}{2 R \cdot 10^{E_b/N_0 \,[\text{dB}] / 10}}}$$

/// BPSK AWGN channel simulator with a deterministic xorshift64 PRNG.
///
/// Converts coded bits to BPSK symbols, adds Gaussian noise calibrated to a
/// target $E_b/N_0$, and returns soft LLR values ready for an LDPC or Viterbi
/// decoder. Because the PRNG is seeded deterministically, the same seed and
/// the same input always produce the same output — ideal for regression tests.
///
/// # Examples
///
/// ```
/// use syndrome::channel_sim::AwgnChannel;
///
/// let mut ch = AwgnChannel::new(5.0, 0.5, 42);
/// let bits: Vec<u8> = vec![0, 1, 0, 0, 1];
/// let llrs = ch.transmit(&bits);
/// assert_eq!(llrs.len(), bits.len());
/// ```
/// Mix a `u64` seed into a well-distributed, (for all practical purposes)
/// non-zero xorshift64 initial state.
///
/// This is the standard SplitMix64 output mixer, the usual way to turn a
/// small, low-entropy seed (`0`, `1`, `2`, ...) into a state suitable for a
/// generator like xorshift64 that has no mixing step of its own on
/// construction. Two reasons a raw seed isn't good enough on its own:
///
/// 1. Xorshift64's all-zero state is a fixed point (it never leaves it), so
///    `seed = 0` must be avoided.
/// 2. Xorshift64's *state transition* is a lightweight bit permutation with
///    weak diffusion between seeds that differ only in low bits — feeding
///    consecutive small integers straight in as state risks correlated
///    early output, not just an exact collision.
///
/// Hashing addresses both at once, and does so for every seed rather than
/// for the particular values someone thought to check. The tempting cheaper
/// fixes do not: forcing the low bit with `seed | 1` maps every
/// `(even, even + 1)` pair onto one state, so two "different" seeds yield
/// byte-identical noise; remapping only `0 => 1` collides that seed with the
/// literal seed `1`. Both leave the same class of bug intact for some other
/// pair of inputs, which is why this uses a hash instead.
fn splitmix64(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    if z == 0 { 1 } else { z } // astronomically unlikely, but keep the invariant absolute
}

/// Advance a xorshift64 state and return one uniform $U[0, 1)$ sample.
///
/// Uses the standard xorshift64 shift triplet $(13, 7, 17)$ and the IEEE-754
/// mantissa trick to convert the 64-bit word to a float in $[1, 2)$, then
/// subtracts 1 to land in $[0, 1)$.
///
/// Shared by [`AwgnChannel`] and [`RayleighBlockChannel`] so both draw from
/// an identical generator rather than two near-copies that could drift apart.
fn next_uniform(state: &mut u64) -> f32 {
    // Xorshift64 — period $2^{64} - 1$.
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;

    // Mantissa trick: set the exponent bits to represent [1, 2), cast to
    // f32, then subtract 1.0 to land in [0, 1).
    // We use the upper 23 bits (mantissa width of f32).
    let mantissa_bits = (x >> 41) as u32; // top 23 bits of the 64-bit word
    let float_bits = 0x3F80_0000_u32 | mantissa_bits; // exponent = 127 → [1, 2)
    f32::from_bits(float_bits) - 1.0
}

/// Draw one standard-normal $\mathcal{N}(0,1)$ sample via Box-Muller.
///
/// Box-Muller transform: given $u_1, u_2 \sim U(0,1)$,
/// $$z = \sqrt{-2 \ln u_1} \cdot \cos(2\pi u_2) \sim \mathcal{N}(0,1)$$
///
/// $u_1$ is clamped to $[10^{-38}, 1)$ to avoid $\ln(0)$. Only the cosine
/// branch is used; the sine branch's sample is discarded rather than cached,
/// which costs a factor of two in generator calls and changes nothing about
/// the distribution.
fn next_gaussian(state: &mut u64) -> f32 {
    let u1 = next_uniform(state).max(1e-38_f32); // clamp away from zero
    let u2 = next_uniform(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * core::f32::consts::PI * u2).cos()
}

pub struct AwgnChannel {
    /// Internal PRNG state (xorshift64; guaranteed non-zero).
    state: u64,
    /// Noise standard deviation $\sigma$.
    sigma: f32,
}

impl AwgnChannel {
    /// Construct a new AWGN channel.
    ///
    /// # Arguments
    ///
    /// * `ebno_db`    — $E_b/N_0$ in decibels.
    /// * `code_rate`  — Code rate $R \in (0, 1]$ (e.g. `0.5` for rate-1/2).
    /// * `seed`       — PRNG seed. Every `u64` value, including `0`, is
    ///   accepted and produces a distinct, well-mixed noise sequence: the
    ///   seed is hashed (SplitMix64) into the xorshift state rather than
    ///   used directly, so small integer seeds don't collide or correlate.
    ///
    /// # Returns
    ///
    /// An `AwgnChannel` whose noise power is calibrated to the requested SNR.
    ///
    /// # Examples
    ///
    /// ```
    /// use syndrome::channel_sim::AwgnChannel;
    ///
    /// // Rate-1/2 code at 3 dB Eb/N0.
    /// let ch = AwgnChannel::new(3.0, 0.5, 1);
    /// ```
    pub fn new(ebno_db: f32, code_rate: f32, seed: u64) -> Self {
        // $\sigma = \sqrt{\frac{1}{2 R \cdot 10^{E_b/N_0/10}}}$
        let ebno_linear = 10.0_f32.powf(ebno_db / 10.0);
        let sigma = (1.0 / (2.0 * code_rate * ebno_linear)).sqrt();
        Self {
            state: splitmix64(seed),
            sigma,
        }
    }

    /// The noise standard deviation $\sigma$ this channel was calibrated to.
    ///
    /// Exposed so a caller can recover the received symbol from an LLR
    /// ($r = \text{LLR} \cdot \sigma^2 / 2$) — which is what the statistical
    /// validation tests use to measure the realized noise distribution — and
    /// so a simulation can report the $\sigma$ actually in force rather than
    /// re-deriving it from $E_b/N_0$ and hoping the formulas agree.
    ///
    /// # Examples
    ///
    /// ```
    /// use syndrome::channel_sim::AwgnChannel;
    ///
    /// // sigma = sqrt(1 / (2 * R * 10^(EbN0/10))); at R=0.5 and 0 dB this is 1.
    /// let ch = AwgnChannel::new(0.0, 0.5, 1);
    /// assert!((ch.sigma() - 1.0).abs() < 1e-6);
    /// ```
    pub fn sigma(&self) -> f32 {
        self.sigma
    }

    /// Draw one standard-normal $\mathcal{N}(0,1)$ sample.
    fn gaussian(&mut self) -> f32 {
        next_gaussian(&mut self.state)
    }

    /// Modulate coded bits through a BPSK AWGN channel and return soft LLRs.
    ///
    /// Each bit is BPSK-mapped ($0 \mapsto +1$, $1 \mapsto -1$), corrupted by
    /// $\mathcal{N}(0, \sigma^2)$ noise, and converted to a log-likelihood ratio:
    /// $$\text{LLR}_i = \frac{2 r_i}{\sigma^2}$$
    ///
    /// A positive LLR favours bit 0; a negative LLR favours bit 1.
    ///
    /// # Arguments
    ///
    /// * `coded_bits` — Slice of `0`/`1` encoded bits (e.g. output of a rate-matcher).
    ///
    /// # Returns
    ///
    /// A `Vec<f32>` of soft LLR values with the same length as `coded_bits`.
    ///
    /// # Examples
    ///
    /// ```
    /// use syndrome::channel_sim::AwgnChannel;
    ///
    /// let mut ch = AwgnChannel::new(10.0, 0.5, 7);
    /// let bits = vec![0u8; 64];
    /// let llrs = ch.transmit(&bits);
    /// assert_eq!(llrs.len(), 64);
    /// // At 10 dB Eb/No the average LLR is strongly positive.
    /// let mean_llr: f32 = llrs.iter().sum::<f32>() / llrs.len() as f32;
    /// assert!(mean_llr > 0.0, "mean LLR must be positive for all-zero input");
    /// ```
    pub fn transmit(&mut self, coded_bits: &[u8]) -> Vec<f32> {
        let sigma_sq = self.sigma * self.sigma;
        let scale = 2.0 / sigma_sq;
        coded_bits
            .iter()
            .map(|&b| {
                let x = if b == 0 { 1.0_f32 } else { -1.0_f32 };
                let r = x + self.gaussian() * self.sigma;
                r * scale // LLR = 2r / σ²
            })
            .collect()
    }

    /// Transmit with zero noise (useful for unit tests that verify LLR sign only).
    ///
    /// Returns $\text{LLR}_i = 2 x_i / \sigma^2$ with no noise added.  The PRNG
    /// state is **not** advanced.
    ///
    /// # Arguments
    ///
    /// * `coded_bits` — Slice of `0`/`1` bits.
    ///
    /// # Returns
    ///
    /// LLR vector derived from noise-free received symbols.
    ///
    /// # Examples
    ///
    /// ```
    /// use syndrome::channel_sim::AwgnChannel;
    ///
    /// let ch = AwgnChannel::new(0.0, 0.5, 1);
    /// let bits = vec![0u8, 1u8, 0u8];
    /// let llrs = ch.transmit_noiseless(&bits);
    /// assert!(llrs[0] > 0.0);
    /// assert!(llrs[1] < 0.0);
    /// ```
    pub fn transmit_noiseless(&self, coded_bits: &[u8]) -> Vec<f32> {
        let sigma_sq = self.sigma * self.sigma;
        let scale = 2.0 / sigma_sq;
        coded_bits
            .iter()
            .map(|&b| {
                let x = if b == 0 { 1.0_f32 } else { -1.0_f32 };
                x * scale
            })
            .collect()
    }

    /// Compute the bit error rate (BER) between two equal-length byte slices.
    ///
    /// $$\text{BER} = \frac{\text{number of differing bits}}{|\text{decoded}|}$$
    ///
    /// # Arguments
    ///
    /// * `decoded`  — Decoder output bits (0/1 per element).
    /// * `original` — Ground-truth transmitted bits.
    ///
    /// # Returns
    ///
    /// BER as a `f64` in $[0.0, 1.0]$.  Returns `0.0` if the slice is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use syndrome::channel_sim::AwgnChannel;
    ///
    /// let a = vec![0u8, 0, 1, 1];
    /// let b = vec![0u8, 1, 1, 0];
    /// let ber = AwgnChannel::bit_error_rate(&a, &b);
    /// assert!((ber - 0.5).abs() < 1e-9);
    /// ```
    pub fn bit_error_rate(decoded: &[u8], original: &[u8]) -> f64 {
        let n = decoded.len().min(original.len());
        if n == 0 {
            return 0.0;
        }
        Self::count_errors(decoded, original) as f64 / n as f64
    }

    /// Count the number of positions where `decoded` and `original` differ.
    ///
    /// # Arguments
    ///
    /// * `decoded`  — Decoder output bits.
    /// * `original` — Ground-truth bits.
    ///
    /// # Returns
    ///
    /// Number of bit errors as `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use syndrome::channel_sim::AwgnChannel;
    ///
    /// let a = vec![0u8, 1, 0, 1];
    /// let b = vec![0u8, 0, 0, 1];
    /// assert_eq!(AwgnChannel::count_errors(&a, &b), 1);
    /// ```
    pub fn count_errors(decoded: &[u8], original: &[u8]) -> usize {
        decoded
            .iter()
            .zip(original.iter())
            .filter(|&(&d, &o)| d != o)
            .count()
    }
}

/// BPSK channel with Rayleigh block fading and perfect receiver CSI.
///
/// AWGN is the friendliest channel a code will ever see: every symbol
/// arrives with the same amplitude, so errors come only from noise. A
/// mobile channel does not behave that way. Multipath propagation makes the
/// received amplitude itself random, and a code that looks strong under AWGN
/// can perform very differently once deep fades enter the picture — which is
/// why 3GPP evaluates codes over fading channels, not just AWGN.
///
/// # Signal model
///
/// Symbols are grouped into blocks of `block_len`. Each block draws one
/// fading amplitude $h$, held constant across the whole block ("block
/// fading" — the coherence time is one block):
///
/// $$r_i = h \cdot x_i + n_i, \qquad n_i \sim \mathcal{N}(0, \sigma^2)$$
///
/// $h$ is Rayleigh-distributed, obtained as the magnitude of a circularly
/// symmetric complex Gaussian $h_c \sim \mathcal{CN}(0, 1)$:
///
/// $$h = \sqrt{g_1^2 + g_2^2}, \qquad g_1, g_2 \sim \mathcal{N}(0, 1/2)$$
///
/// The $1/2$ per component is what normalizes the channel to unit average
/// power, $E[h^2] = 1$, so that a given $E_b/N_0$ means the same average
/// received energy as it does on the AWGN channel and the two are
/// comparable. Without it every fading result would be shifted by a constant
/// no reader could detect.
///
/// # Receiver knowledge
///
/// The receiver is assumed to know $h$ exactly (perfect channel state
/// information). Under that assumption the LLR is
///
/// $$\text{LLR}_i = \frac{2 h r_i}{\sigma^2},$$
///
/// i.e. the AWGN LLR scaled by the realized gain, which automatically
/// down-weights symbols from a faded block — exactly the soft information an
/// iterative decoder needs. Channel estimation error is **not** modelled;
/// results from this channel are therefore an optimistic bound on what a
/// real receiver achieves, and should be read that way.
///
/// # Examples
///
/// ```
/// use syndrome::channel_sim::RayleighBlockChannel;
///
/// let mut ch = RayleighBlockChannel::new(5.0, 0.5, 16, 42);
/// let bits = vec![0u8; 64];
/// let llrs = ch.transmit(&bits);
/// assert_eq!(llrs.len(), 64);
/// ```
pub struct RayleighBlockChannel {
    /// Internal PRNG state (xorshift64; guaranteed non-zero).
    state: u64,
    /// Noise standard deviation $\sigma$.
    sigma: f32,
    /// Number of consecutive symbols sharing one fading amplitude.
    block_len: usize,
}

impl RayleighBlockChannel {
    /// Construct a Rayleigh block-fading channel.
    ///
    /// # Arguments
    ///
    /// * `ebno_db` — average $E_b/N_0$ in decibels. "Average" because the
    ///   instantaneous ratio varies with $h$; the normalization $E[h^2] = 1$
    ///   makes this the mean.
    /// * `code_rate` — code rate $R \in (0, 1]$.
    /// * `block_len` — symbols per fading block. `1` gives fully
    ///   independent (fast) fading; a value at or above the codeword length
    ///   gives one fade for the whole codeword (quasi-static).
    /// * `seed` — PRNG seed, hashed exactly as [`AwgnChannel::new`] does.
    ///
    /// # Panics
    ///
    /// Panics if `block_len` is zero — a fading block must contain at least
    /// one symbol, and silently substituting 1 would hide a caller's bug.
    ///
    /// # Examples
    ///
    /// ```
    /// use syndrome::channel_sim::RayleighBlockChannel;
    ///
    /// let ch = RayleighBlockChannel::new(3.0, 0.5, 32, 7);
    /// assert_eq!(ch.block_len(), 32);
    /// ```
    pub fn new(ebno_db: f32, code_rate: f32, block_len: usize, seed: u64) -> Self {
        assert!(block_len > 0, "block_len must be at least 1 symbol");
        let ebno_linear = 10.0_f32.powf(ebno_db / 10.0);
        let sigma = (1.0 / (2.0 * code_rate * ebno_linear)).sqrt();
        Self {
            state: splitmix64(seed),
            sigma,
            block_len,
        }
    }

    /// The noise standard deviation $\sigma$ this channel was calibrated to.
    ///
    /// # Examples
    ///
    /// ```
    /// use syndrome::channel_sim::RayleighBlockChannel;
    ///
    /// let ch = RayleighBlockChannel::new(0.0, 0.5, 8, 1);
    /// assert!((ch.sigma() - 1.0).abs() < 1e-6);
    /// ```
    pub fn sigma(&self) -> f32 {
        self.sigma
    }

    /// Symbols per fading block.
    ///
    /// # Examples
    ///
    /// ```
    /// use syndrome::channel_sim::RayleighBlockChannel;
    ///
    /// assert_eq!(RayleighBlockChannel::new(0.0, 0.5, 64, 1).block_len(), 64);
    /// ```
    pub fn block_len(&self) -> usize {
        self.block_len
    }

    /// Draw one Rayleigh fading amplitude with $E[h^2] = 1$.
    fn fading_gain(&mut self) -> f32 {
        // Two i.i.d. N(0, 1/2) components: N(0,1) scaled by sqrt(1/2).
        const COMPONENT_SCALE: f32 = core::f32::consts::FRAC_1_SQRT_2;
        let g1 = next_gaussian(&mut self.state) * COMPONENT_SCALE;
        let g2 = next_gaussian(&mut self.state) * COMPONENT_SCALE;
        (g1 * g1 + g2 * g2).sqrt()
    }

    /// Modulate coded bits through the fading channel and return soft LLRs.
    ///
    /// # Arguments
    ///
    /// * `coded_bits` — slice of `0`/`1` encoded bits.
    ///
    /// # Returns
    ///
    /// A `Vec<f32>` of LLRs the same length as `coded_bits`. Positive
    /// favours bit 0.
    ///
    /// # Examples
    ///
    /// ```
    /// use syndrome::channel_sim::RayleighBlockChannel;
    ///
    /// let mut ch = RayleighBlockChannel::new(20.0, 0.5, 8, 3);
    /// let llrs = ch.transmit(&vec![0u8; 32]);
    /// // At 20 dB most symbols survive even a fade, so the mean is positive.
    /// assert!(llrs.iter().sum::<f32>() > 0.0);
    /// ```
    pub fn transmit(&mut self, coded_bits: &[u8]) -> Vec<f32> {
        self.transmit_with_gains(coded_bits).0
    }

    /// Modulate coded bits and also return the per-block fading amplitudes.
    ///
    /// Same as [`Self::transmit`], but additionally hands back the realized
    /// $h$ for each block. Useful for a receiver study that wants to compare
    /// against genie-aided knowledge, and for the tests that check the gain
    /// distribution and its block-constancy directly rather than inferring
    /// them from LLRs.
    ///
    /// # Arguments
    ///
    /// * `coded_bits` — slice of `0`/`1` encoded bits.
    ///
    /// # Returns
    ///
    /// `(llrs, gains)` where `llrs.len() == coded_bits.len()` and
    /// `gains.len() == ceil(coded_bits.len() / block_len)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use syndrome::channel_sim::RayleighBlockChannel;
    ///
    /// let mut ch = RayleighBlockChannel::new(5.0, 0.5, 10, 11);
    /// let (llrs, gains) = ch.transmit_with_gains(&vec![0u8; 25]);
    /// assert_eq!(llrs.len(), 25);
    /// assert_eq!(gains.len(), 3); // 10 + 10 + 5
    /// assert!(gains.iter().all(|&g| g >= 0.0));
    /// ```
    pub fn transmit_with_gains(&mut self, coded_bits: &[u8]) -> (Vec<f32>, Vec<f32>) {
        let sigma_sq = self.sigma * self.sigma;
        let n_blocks = coded_bits.len().div_ceil(self.block_len);
        let mut llrs = Vec::with_capacity(coded_bits.len());
        let mut gains = Vec::with_capacity(n_blocks);

        for chunk in coded_bits.chunks(self.block_len) {
            let h = self.fading_gain();
            gains.push(h);
            // LLR = 2 h r / sigma^2 with r = h x + n (perfect CSI).
            let scale = 2.0 * h / sigma_sq;
            for &b in chunk {
                let x = if b == 0 { 1.0_f32 } else { -1.0_f32 };
                let r = h * x + next_gaussian(&mut self.state) * self.sigma;
                llrs.push(r * scale);
            }
        }
        (llrs, gains)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Statistical validation of the channel model itself
    //
    // The tests above this block check plumbing: determinism, LLR signs,
    // BER ordering. None of them would notice if the noise had the wrong
    // variance, a non-zero mean, or a non-Gaussian shape -- and any of those
    // would silently shift every BER curve the crate produces while leaving
    // every existing test green. A wrong sigma is the worst case: it is
    // invisible in the output and it moves published results by a fixed
    // number of decibels. These tests check the distribution directly.
    // =========================================================================

    /// Standard normal CDF $\Phi(x)$, via the Abramowitz & Stegun 7.1.26
    /// rational approximation to $\mathrm{erf}$ (absolute error
    /// $< 1.5 \times 10^{-7}$, far below what a chi-square test over a few
    /// thousand samples can resolve).
    ///
    /// Rust's standard library has no error function, and pulling a
    /// dependency in for one test-only helper would violate the
    /// keep-`Cargo.toml`-light rule. Correctness of this helper is pinned by
    /// [`normal_cdf_matches_published_values`] before any other test relies
    /// on it.
    fn normal_cdf(x: f64) -> f64 {
        0.5 * (1.0 + erf(x / core::f64::consts::SQRT_2))
    }

    /// Abramowitz & Stegun 7.1.26 approximation to $\mathrm{erf}(x)$.
    fn erf(x: f64) -> f64 {
        const P: f64 = 0.327_591_1;
        const A: [f64; 5] = [
            0.254_829_592,
            -0.284_496_736,
            1.421_413_741,
            -1.453_152_027,
            1.061_405_429,
        ];
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();
        let t = 1.0 / (1.0 + P * x);
        // Horner evaluation of the degree-5 polynomial in t (no constant term).
        let poly = t * (A[0] + t * (A[1] + t * (A[2] + t * (A[3] + t * A[4]))));
        sign * (1.0 - poly * (-x * x).exp())
    }

    /// Pin the test-only `normal_cdf` helper against published values of
    /// $\Phi$ before any distributional test leans on it. Without this, a
    /// bug in the helper would show up as a bug in the channel.
    #[test]
    fn normal_cdf_matches_published_values() {
        // Standard-normal CDF values from any statistical table.
        for &(x, expected) in &[
            (0.0_f64, 0.500_000_0_f64),
            (1.0, 0.841_344_7),
            (-1.0, 0.158_655_3),
            // 1.959964 and 2.575829 are by definition the 97.5% and 99.5%
            // quantiles, so their CDF values are exactly 0.975 and 0.995.
            (1.959_964, 0.975_000_0),
            (-1.959_964, 0.025_000_0),
            (2.575_829, 0.995_000_0),
            (3.0, 0.998_650_1),
            (-3.0, 0.001_349_9),
        ] {
            let got = normal_cdf(x);
            assert!(
                (got - expected).abs() < 2e-7,
                "Phi({x}) = {got}, expected {expected}"
            );
        }
    }

    /// Recover the realized noise samples from a channel run.
    ///
    /// The channel returns $\text{LLR} = 2r/\sigma^2$ with $r = x + n$, so
    /// with the transmitted bits known the noise is
    /// $n = \text{LLR}\,\sigma^2/2 - x$. Recovering the noise rather than
    /// inspecting LLRs directly is what lets these tests check the *noise*
    /// distribution against its specification.
    fn recover_noise(ch: &AwgnChannel, bits: &[u8], llrs: &[f32]) -> Vec<f64> {
        let sigma_sq = (ch.sigma() as f64) * (ch.sigma() as f64);
        bits.iter()
            .zip(llrs.iter())
            .map(|(&b, &l)| {
                let x = if b == 0 { 1.0_f64 } else { -1.0_f64 };
                (l as f64) * sigma_sq / 2.0 - x
            })
            .collect()
    }

    /// The realized noise must have the mean and variance the $E_b/N_0$
    /// calibration promises. This is the test that would catch a wrong
    /// $\sigma$ formula — the failure mode that shifts every BER curve by a
    /// constant number of dB while leaving all other tests passing.
    #[test]
    fn noise_mean_and_variance_match_the_ebn0_calibration() {
        const N: usize = 200_000;
        for &(ebno_db, rate) in &[
            (0.0_f32, 0.5_f32),
            (3.0, 0.5),
            (6.0, 1.0 / 3.0),
            (-2.0, 0.75),
        ] {
            let mut ch = AwgnChannel::new(ebno_db, rate, 0x5EED);
            let sigma = ch.sigma() as f64;
            // Alternating bits so any polarity-dependent bug shows up.
            let bits: Vec<u8> = (0..N).map(|i| (i % 2) as u8).collect();
            let llrs = ch.transmit(&bits);
            let noise = recover_noise(&ch, &bits, &llrs);

            let mean = noise.iter().sum::<f64>() / N as f64;
            let var = noise.iter().map(|n| (n - mean) * (n - mean)).sum::<f64>() / (N - 1) as f64;

            // Standard error of the mean is sigma/sqrt(N); 5 sigma is a
            // comfortable band that still catches any real bias.
            let se_mean = sigma / (N as f64).sqrt();
            assert!(
                mean.abs() < 5.0 * se_mean,
                "Eb/N0={ebno_db} R={rate}: noise mean {mean:.6} exceeds 5 standard errors ({:.6})",
                5.0 * se_mean,
            );

            // Variance of the sample variance for a Gaussian is
            // 2*sigma^4/(N-1), so the relative standard error is
            // sqrt(2/(N-1)) ~ 0.32% here. Allow 2%.
            let expected_var = sigma * sigma;
            let rel_err = (var - expected_var).abs() / expected_var;
            assert!(
                rel_err < 0.02,
                "Eb/N0={ebno_db} R={rate}: noise variance {var:.6} vs expected {expected_var:.6} \
                 (relative error {rel_err:.4})",
            );
        }
    }

    /// The $E_b/N_0$ recovered from the realized noise must match the value
    /// requested, closing the loop on the SNR calibration end to end:
    /// $\sigma^2 = 1/(2 R \cdot 10^{E_b/N_0/10})$ inverted gives
    /// $E_b/N_0 = 10\log_{10}\left(1/(2 R \sigma^2)\right)$.
    #[test]
    fn measured_snr_matches_requested_snr() {
        const N: usize = 200_000;
        for &(ebno_db, rate) in &[
            (0.0_f32, 0.5_f32),
            (2.0, 0.5),
            (5.0, 1.0 / 3.0),
            (8.0, 0.75),
        ] {
            let mut ch = AwgnChannel::new(ebno_db, rate, 0xD1CE);
            let bits: Vec<u8> = (0..N).map(|i| (i % 3 == 0) as u8).collect();
            let llrs = ch.transmit(&bits);
            let noise = recover_noise(&ch, &bits, &llrs);

            let mean = noise.iter().sum::<f64>() / N as f64;
            let var = noise.iter().map(|n| (n - mean) * (n - mean)).sum::<f64>() / (N - 1) as f64;
            let measured_db = 10.0 * (1.0 / (2.0 * rate as f64 * var)).log10();

            assert!(
                (measured_db - ebno_db as f64).abs() < 0.1,
                "requested {ebno_db} dB, measured {measured_db:.4} dB (sigma^2 = {var:.6})",
            );
        }
    }

    /// Chi-square goodness-of-fit test: the standardized noise must actually
    /// be Gaussian, not merely have the right first two moments. A generator
    /// bug that produced, say, a triangular or truncated distribution would
    /// pass the mean/variance test above and fail here.
    ///
    /// Bins are fixed edges from $-3.5$ to $+3.5$ in steps of $0.5$ plus two
    /// open tails (16 bins, 15 degrees of freedom). The statistic is
    /// compared against the $\chi^2_{15}$ upper 0.1% critical value, 37.70 —
    /// a deliberately loose threshold, since the seed is fixed and the test
    /// must not flake, while still rejecting any grossly wrong shape.
    #[test]
    fn noise_passes_a_chi_square_normality_test() {
        const N: usize = 200_000;
        let mut ch = AwgnChannel::new(3.0, 0.5, 0xC0FFEE);
        let sigma = ch.sigma() as f64;
        let bits: Vec<u8> = (0..N).map(|i| (i % 2) as u8).collect();
        let llrs = ch.transmit(&bits);
        let noise = recover_noise(&ch, &bits, &llrs);

        // Fixed bin edges in units of sigma.
        let mut edges = Vec::new();
        let mut e = -3.5_f64;
        while e <= 3.5 + 1e-9 {
            edges.push(e);
            e += 0.5;
        }
        let n_bins = edges.len() + 1; // two open tails
        let mut observed = vec![0usize; n_bins];
        for &x in &noise {
            let z = x / sigma;
            let mut bin = 0usize;
            while bin < edges.len() && z >= edges[bin] {
                bin += 1;
            }
            observed[bin] += 1;
        }

        // Expected counts from the standard normal CDF.
        let mut expected = vec![0.0_f64; n_bins];
        expected[0] = normal_cdf(edges[0]) * N as f64;
        for i in 1..edges.len() {
            expected[i] = (normal_cdf(edges[i]) - normal_cdf(edges[i - 1])) * N as f64;
        }
        expected[n_bins - 1] = (1.0 - normal_cdf(edges[edges.len() - 1])) * N as f64;

        let chi2: f64 = observed
            .iter()
            .zip(expected.iter())
            .map(|(&o, &e)| {
                let d = o as f64 - e;
                d * d / e
            })
            .sum();

        // Guard the test's own premise: the chi-square approximation needs
        // every expected count to be reasonably large.
        for (i, &e) in expected.iter().enumerate() {
            assert!(
                e >= 5.0,
                "bin {i} expected count {e:.2} too small for a chi-square test"
            );
        }

        let df = n_bins - 1;
        assert_eq!(df, 15, "critical value below is tabulated for 15 df");
        const CHI2_15_DF_UPPER_0_001: f64 = 37.697;
        assert!(
            chi2 < CHI2_15_DF_UPPER_0_001,
            "chi-square statistic {chi2:.3} exceeds the 0.1% critical value \
             {CHI2_15_DF_UPPER_0_001} for {df} df -- noise is not Gaussian",
        );
    }

    /// Box-Muller here uses only the cosine branch and discards the sine
    /// branch. That is a valid (if wasteful) generator, but only if
    /// consecutive samples stay independent — a cached-branch implementation
    /// with a bug, or a shared-state slip, would show up as correlation.
    /// Checked via the lag-1 autocorrelation of the realized noise.
    #[test]
    fn consecutive_noise_samples_are_uncorrelated() {
        const N: usize = 200_000;
        let mut ch = AwgnChannel::new(4.0, 0.5, 0xA5A5_1234);
        let bits = vec![0u8; N];
        let llrs = ch.transmit(&bits);
        let noise = recover_noise(&ch, &bits, &llrs);

        let mean = noise.iter().sum::<f64>() / N as f64;
        let var = noise.iter().map(|n| (n - mean) * (n - mean)).sum::<f64>() / N as f64;
        let cov: f64 = noise
            .windows(2)
            .map(|w| (w[0] - mean) * (w[1] - mean))
            .sum::<f64>()
            / (N - 1) as f64;
        let rho = cov / var;

        // For independent samples the lag-1 autocorrelation has standard
        // error ~1/sqrt(N) = 0.0022; 5 sigma is 0.011.
        assert!(
            rho.abs() < 5.0 / (N as f64).sqrt(),
            "lag-1 autocorrelation {rho:.5} too large — samples are not independent",
        );
    }

    // =========================================================================
    // Rayleigh block fading
    // =========================================================================

    /// The fading amplitude must be normalized to unit average power,
    /// $E[h^2] = 1$, and match the Rayleigh distribution's other moments:
    /// $E[h] = \sqrt{\pi}/2 \approx 0.8862$ and $E[h^4] = 2$. Without the
    /// normalization every fading result would sit at a different effective
    /// SNR than the one requested, and no reader could tell.
    #[test]
    fn rayleigh_gains_are_unit_power_and_correctly_distributed() {
        const N_BLOCKS: usize = 200_000;
        let mut ch = RayleighBlockChannel::new(5.0, 0.5, 1, 0xFADE);
        let bits = vec![0u8; N_BLOCKS];
        let (_, gains) = ch.transmit_with_gains(&bits);
        assert_eq!(gains.len(), N_BLOCKS);

        let n = N_BLOCKS as f64;
        let m1 = gains.iter().map(|&g| g as f64).sum::<f64>() / n;
        let m2 = gains.iter().map(|&g| (g as f64).powi(2)).sum::<f64>() / n;
        let m4 = gains.iter().map(|&g| (g as f64).powi(4)).sum::<f64>() / n;

        let expected_m1 = core::f64::consts::PI.sqrt() / 2.0;
        assert!(
            (m1 - expected_m1).abs() < 0.01,
            "E[h] = {m1:.5}, expected {expected_m1:.5}"
        );
        assert!((m2 - 1.0).abs() < 0.01, "E[h^2] = {m2:.5}, expected 1.0");
        assert!((m4 - 2.0).abs() < 0.05, "E[h^4] = {m4:.5}, expected 2.0");
    }

    /// The defining property of *block* fading: one amplitude per block,
    /// applied to every symbol in it, redrawn between blocks.
    ///
    /// Checked by residual variance rather than by comparing each LLR to a
    /// noiseless prediction. The naive comparison does not work: the LLR is
    /// $2h(h + n)/\sigma^2$, so its *relative* deviation from $2h^2/\sigma^2$
    /// is $n/h$, which is unbounded in a deep fade — a correct
    /// implementation would fail such a test whenever $h$ happened to be
    /// small. Instead, invert the model to recover each symbol's noise,
    /// $n_i = \text{LLR}_i \sigma^2 / (2 h_b) - h_b$, and check the pooled
    /// residuals against $\mathcal{N}(0, \sigma^2)$.
    ///
    /// This is what makes the test discriminating: if the gain were redrawn
    /// per *symbol* instead of per block, the residual would be
    /// $(h_i - h_b) + n_i$ with variance $\operatorname{Var}(h) + \sigma^2 =
    /// (1 - \pi/4) + \sigma^2 \approx 0.215 + \sigma^2$, roughly triple the
    /// correct value at the SNR used here.
    #[test]
    fn fading_amplitude_is_constant_within_a_block() {
        let block_len = 16usize;
        let n_blocks = 2_000usize;
        let mut ch = RayleighBlockChannel::new(10.0, 0.5, block_len, 0xB10C);
        let bits = vec![0u8; block_len * n_blocks];
        let (llrs, gains) = ch.transmit_with_gains(&bits);
        assert_eq!(gains.len(), n_blocks);

        let sigma_sq = (ch.sigma() as f64) * (ch.sigma() as f64);
        let mut residuals = Vec::with_capacity(llrs.len());
        for (b, chunk) in llrs.chunks(block_len).enumerate() {
            let h = gains[b] as f64;
            for &l in chunk {
                // r_i = LLR_i * sigma^2 / (2h); for bit 0, r_i = h + n_i.
                let r = (l as f64) * sigma_sq / (2.0 * h);
                residuals.push(r - h);
            }
        }

        let n = residuals.len() as f64;
        let mean = residuals.iter().sum::<f64>() / n;
        let var = residuals
            .iter()
            .map(|x| (x - mean) * (x - mean))
            .sum::<f64>()
            / (n - 1.0);

        assert!(
            mean.abs() < 5.0 * (sigma_sq / n).sqrt(),
            "residual mean {mean:.5} is biased — the per-block gain is not what was applied",
        );
        let rel_err = (var - sigma_sq).abs() / sigma_sq;
        assert!(
            rel_err < 0.05,
            "residual variance {var:.5} vs expected sigma^2 {sigma_sq:.5} \
             (relative error {rel_err:.4}) — the gain is not constant across each block",
        );

        // And genuinely varies between blocks (not one gain reused).
        let distinct = gains
            .iter()
            .filter(|&&g| (g - gains[0]).abs() > 1e-6)
            .count();
        assert!(
            distinct > n_blocks / 2,
            "only {distinct} of {n_blocks} blocks differ from the first — gains are not redrawn",
        );
    }

    /// Both channels must reproduce the textbook uncoded-BPSK error
    /// probabilities, and Rayleigh must be substantially worse than AWGN at
    /// the same average $E_b/N_0$ — deep fades cost more than the good blocks
    /// give back.
    ///
    /// Matching a closed form is far stronger evidence than an
    /// is-it-worse ratio: a fading model that merely added extra noise would
    /// also look "worse", but only a correctly distributed one lands on
    /// these curves. With $r = hx + n$ and hard decisions, the error
    /// probability given $h$ is $Q(h/\sigma)$, so:
    ///
    /// * AWGN ($h \equiv 1$): $\text{BER} = Q(1/\sigma)$.
    /// * Rayleigh: averaging $Q(\sqrt{\gamma})$ over an exponentially
    ///   distributed $\gamma = h^2/\sigma^2$ with mean $1/\sigma^2$ gives the
    ///   standard result
    ///   $\text{BER} = \tfrac{1}{2}\left(1 - \sqrt{\bar\gamma_b/(1+\bar\gamma_b)}\right)$
    ///   with $\bar\gamma_b = 1/(2\sigma^2)$.
    ///
    /// Both are written in terms of $\sigma$ alone, so they test the channel
    /// against theory without re-deriving $\sigma$ from $E_b/N_0$ — that
    /// link is already covered by [`measured_snr_matches_requested_snr`].
    #[test]
    fn uncoded_ber_matches_closed_form_on_both_channels() {
        const N: usize = 400_000;
        let ebno = 6.0_f32;
        let rate = 0.5_f32;
        let bits: Vec<u8> = (0..N).map(|i| (i % 2) as u8).collect();

        let mut awgn = AwgnChannel::new(ebno, rate, 0x1111);
        let sigma = awgn.sigma() as f64;
        let awgn_hard: Vec<u8> = awgn
            .transmit(&bits)
            .iter()
            .map(|&l| u8::from(l < 0.0))
            .collect();
        let awgn_ber = AwgnChannel::bit_error_rate(&awgn_hard, &bits);
        let awgn_theory = 1.0 - normal_cdf(1.0 / sigma);
        assert!(
            (awgn_ber - awgn_theory).abs() / awgn_theory < 0.05,
            "AWGN BER {awgn_ber:.5} differs from Q(1/sigma) = {awgn_theory:.5} by more than 5%",
        );

        // block_len = 1: independent fading per symbol, which is the
        // condition the closed form averages over.
        let mut ray = RayleighBlockChannel::new(ebno, rate, 1, 0x2222);
        let ray_sigma = ray.sigma() as f64;
        let ray_hard: Vec<u8> = ray
            .transmit(&bits)
            .iter()
            .map(|&l| u8::from(l < 0.0))
            .collect();
        let ray_ber = AwgnChannel::bit_error_rate(&ray_hard, &bits);
        let gamma_b = 1.0 / (2.0 * ray_sigma * ray_sigma);
        let ray_theory = 0.5 * (1.0 - (gamma_b / (1.0 + gamma_b)).sqrt());
        assert!(
            (ray_ber - ray_theory).abs() / ray_theory < 0.05,
            "Rayleigh BER {ray_ber:.5} differs from the closed form {ray_theory:.5} by more than 5%",
        );

        assert!(
            ray_ber > awgn_ber * 3.0,
            "Rayleigh BER {ray_ber:.5} should be far worse than AWGN {awgn_ber:.5} \
             at the same average Eb/N0",
        );
    }

    /// Same seed, same output — the reproducibility guarantee the whole
    /// deterministic-simulation approach rests on.
    #[test]
    fn rayleigh_is_deterministic_for_a_given_seed() {
        let bits: Vec<u8> = (0..500).map(|i| (i % 5 == 0) as u8).collect();
        let a = RayleighBlockChannel::new(4.0, 0.5, 12, 31337).transmit(&bits);
        let b = RayleighBlockChannel::new(4.0, 0.5, 12, 31337).transmit(&bits);
        assert_eq!(a, b);
        let c = RayleighBlockChannel::new(4.0, 0.5, 12, 31338).transmit(&bits);
        assert_ne!(a, c, "different seeds must give different fading");
    }

    /// A block length that does not divide the payload must still cover
    /// every symbol, with the final short block getting its own gain.
    #[test]
    fn rayleigh_handles_a_ragged_final_block() {
        let mut ch = RayleighBlockChannel::new(5.0, 0.5, 10, 77);
        let (llrs, gains) = ch.transmit_with_gains(&[0u8; 25]);
        assert_eq!(llrs.len(), 25);
        assert_eq!(gains.len(), 3);
    }

    /// A zero-length block is a caller bug, not something to paper over.
    #[test]
    #[should_panic(expected = "block_len must be at least 1")]
    fn rayleigh_rejects_zero_block_len() {
        let _ = RayleighBlockChannel::new(5.0, 0.5, 0, 1);
    }

    /// Verify that with zero sigma (infinite SNR) all-zero bits produce positive LLRs.
    ///
    /// We use `transmit_noiseless` so that there is no random perturbation at all.
    /// Every BPSK symbol is $+1$, so every LLR must be strictly positive.
    #[test]
    fn all_zero_bits_no_noise_all_positive() {
        // Build a channel at a very high SNR so sigma is tiny but non-zero, then
        // use the noiseless path to be exact.
        let ch = AwgnChannel::new(30.0, 0.5, 42);
        let bits = vec![0u8; 1000];
        let llrs = ch.transmit_noiseless(&bits);
        assert_eq!(llrs.len(), 1000);
        for (i, &l) in llrs.iter().enumerate() {
            assert!(
                l > 0.0,
                "LLR at index {i} was {l} — expected positive for bit 0"
            );
        }
    }

    /// Verify that BER decreases as SNR increases from 0 dB to 10 dB.
    ///
    /// Uses a 1000-bit alternating pattern so that both BPSK constellation
    /// points are exercised equally.  The same PRNG seed is used for both
    /// channels so that the noise realisations are comparable.
    #[test]
    fn ber_decreases_with_snr() {
        let n = 1000;
        // Alternating 0/1 pattern exercises both symbol polarities.
        let original: Vec<u8> = (0..n).map(|i| (i % 2) as u8).collect();

        let code_rate = 0.5_f32;

        // Low SNR: 0 dB Eb/N0
        let mut ch_low = AwgnChannel::new(0.0, code_rate, 1234);
        let llrs_low = ch_low.transmit(&original);
        let decoded_low: Vec<u8> = llrs_low
            .iter()
            .map(|&l| if l >= 0.0 { 0 } else { 1 })
            .collect();
        let ber_low = AwgnChannel::bit_error_rate(&decoded_low, &original);

        // High SNR: 10 dB Eb/N0
        let mut ch_high = AwgnChannel::new(10.0, code_rate, 1234);
        let llrs_high = ch_high.transmit(&original);
        let decoded_high: Vec<u8> = llrs_high
            .iter()
            .map(|&l| if l >= 0.0 { 0 } else { 1 })
            .collect();
        let ber_high = AwgnChannel::bit_error_rate(&decoded_high, &original);

        assert!(
            ber_high < ber_low,
            "Expected BER at 10 dB ({ber_high:.4}) < BER at 0 dB ({ber_low:.4})"
        );

        // Sanity: at 10 dB the uncoded BER should be well below 10 %.
        assert!(ber_high < 0.10, "BER at 10 dB too high: {ber_high:.4}");
    }

    /// Confirm that `count_errors` returns 0 on identical slices.
    #[test]
    fn count_errors_zero_on_identical() {
        let bits: Vec<u8> = (0..64).map(|i| (i % 2) as u8).collect();
        assert_eq!(AwgnChannel::count_errors(&bits, &bits), 0);
    }

    /// Confirm that `bit_error_rate` returns 1.0 on fully inverted slices.
    #[test]
    fn ber_one_on_fully_inverted() {
        let original = vec![0u8; 100];
        let decoded = vec![1u8; 100];
        let ber = AwgnChannel::bit_error_rate(&decoded, &original);
        assert!((ber - 1.0).abs() < 1e-12);
    }

    /// Confirm xorshift advances the PRNG state (transmitting the same bits
    /// twice with the same seed produces identical LLR vectors).
    #[test]
    fn deterministic_output_same_seed() {
        let bits = vec![0u8, 1, 0, 1, 0, 0, 1, 1];
        let mut ch1 = AwgnChannel::new(5.0, 0.5, 99);
        let mut ch2 = AwgnChannel::new(5.0, 0.5, 99);
        let llrs1 = ch1.transmit(&bits);
        let llrs2 = ch2.transmit(&bits);
        assert_eq!(llrs1, llrs2);
    }

    /// Adjacent seeds must give independent noise. This is the property the
    /// SplitMix64 seed hash exists to guarantee: the cheaper `seed | 1`
    /// remapping collapses every (even, even+1) pair onto one xorshift state
    /// and therefore produces byte-identical noise, which would silently
    /// defeat anything relying on two "different" seeds giving independent
    /// realizations (e.g. HARQ retransmission testing). Checked over
    /// consecutive pairs, including the seed-0 case, and over non-adjacent
    /// seeds for good measure.
    #[test]
    fn adjacent_seeds_are_not_identical() {
        let bits: Vec<u8> = (0..256).map(|i| (i % 3 == 0) as u8).collect();
        for (a, b) in [(0u64, 1u64), (2, 3), (4, 5), (100, 101), (1_000, 1_001)] {
            let llrs_a = AwgnChannel::new(3.0, 0.5, a).transmit(&bits);
            let llrs_b = AwgnChannel::new(3.0, 0.5, b).transmit(&bits);
            assert_ne!(
                llrs_a, llrs_b,
                "seeds {a} and {b} produced identical noise; the seed-remapping regression is back"
            );
        }
    }
}
