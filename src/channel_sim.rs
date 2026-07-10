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
    /// * `seed`       — PRNG seed; the value `0` is mapped to `1` to prevent
    ///   the degenerate all-zero xorshift state.
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
            state: seed | 1, // prevent zero state
            sigma,
        }
    }

    /// Draw one uniform $U[0, 1)$ sample via xorshift64.
    ///
    /// Uses the standard xorshift64 shift triplet $(13, 7, 17)$ and the
    /// IEEE-754 mantissa trick to convert the 64-bit integer to a float in
    /// $[1, 2)$, then subtracts 1 to get $[0, 1)$.
    fn xorshift(&mut self) -> f32 {
        // Xorshift64 — period $2^{64} - 1$.
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;

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
    /// $u_1$ is clamped to $[10^{-38}, 1)$ to avoid $\ln(0)$.
    fn gaussian(&mut self) -> f32 {
        let u1 = self.xorshift().max(1e-38_f32); // clamp away from zero
        let u2 = self.xorshift();
        (-2.0 * u1.ln()).sqrt() * (2.0 * core::f32::consts::PI * u2).cos()
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
