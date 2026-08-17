//! LLR quantization for the fixed-point LDPC decode path.
//!
//! Log-likelihood ratios are soft values, and a min-sum decoder mostly
//! compares them rather than doing arithmetic on them, so they do not need
//! floating point. Carrying them as signed bytes lets the check-node update
//! run 32 lanes per 256-bit register instead of 8. This module defines the
//! fixed-point format ([`QuantParams`]) and the conversions into and out of
//! it; the decoder that consumes them is
//! [`QcLdpcDecoder::decode_layered_offset_min_sum_i8`](crate::qc_ldpc::QcLdpcDecoder::decode_layered_offset_min_sum_i8),
//! which has a portable scalar kernel everywhere and an AVX2 kernel on
//! x86-64. There is no NEON kernel for it: on AArch64 the fixed-point path
//! runs its scalar reference, and only the `f32` path is vectorized there.
//!
//! # Fixed-point format
//!
//! An LLR is stored as $\hat{L} = \operatorname{clamp}(\lfloor L \cdot s
//! \rceil, -127, 127)$, where $s$ is [`QuantParams::scale`]. The value
//! $-128$ is never produced: `i8::MIN.abs()` overflows and
//! `_mm256_abs_epi8` returns $-128$ unchanged, so a single $-128$ anywhere
//! would make the scalar and AVX2 kernels disagree.
//!
//! Messages are `i8`, but the a-posteriori accumulator is `i16`, and the two
//! widths are a measured choice rather than a convenience — see below.
//!
//! # What the format costs, measured
//!
//! Quantization is lossy, so the number that matters is how much extra
//! $E_b/N_0$ the fixed-point decoder needs to reach the same block error
//! rate as the `f32` one. `tests/ldpc_int8_quantization_loss.rs` measures it
//! on this crate's own decoders over the BPSK AWGN channel of
//! [`crate::channel_sim`], with $s = 8$, $\beta = 0.5$ and 10 iterations:
//!
//! | Code | $E_b/N_0$ | Shift | 95% CI |
//! |---|---|---|---|
//! | BG1, $Z = 128$ | 0.80 dB | +0.0031 dB | [+0.0005, +0.0057] |
//! | BG1, $Z = 384$ | 0.75 dB | +0.0052 dB | [+0.0035, +0.0070] |
//! | BG2, $Z = 128$ | 0.60 dB | +0.0096 dB | [+0.0066, +0.0126] |
//! | BG2, $Z = 384$ | 0.60 dB | +0.0067 dB | [+0.0044, +0.0089] |
//!
//! The loss is real — every interval excludes zero, so the fixed-point path
//! is genuinely behind — and it is between 0.003 and 0.010 dB, with every
//! upper bound below 0.013 dB.
//!
//! Resolving a hundredth of a dB is possible because the measurement is
//! **paired**: each trial hands the same received vector to both decoders,
//! so the channel's variance cancels and only the trials on which the two
//! disagree carry information. The dB figure is a block-error-rate ratio
//! converted through the waterfall's locally measured slope; the test module
//! documents the derivation and the assumption behind it.
//!
//! Two caveats belong with the number. It is measured in the waterfall on
//! BPSK AWGN at the operating points above — a different rate, modulation or
//! fading model needs its own run, which is what the study in that file is
//! for. And it is a *decoder-input* quantization loss: the LLRs handed to
//! [`quantize_llr_i16`] are the exact ones the `f32` path receives, so
//! nothing here accounts for a receiver that also quantizes upstream.
//!
//! # Why the posterior is wider than the messages
//!
//! A message magnitude is bounded by the smallest incoming magnitude in its
//! layer, so [`MSG_MAX`] is a real ceiling and extra bits buy nothing. A
//! posterior is a *sum* — the channel LLR plus one message per incident
//! edge, up to 31 terms for BG1 column 0 — and behaves completely
//! differently. Clamping it to the message range does not cost a constant
//! factor; it produces an error floor. Measured on BG1 $Z = 128$ at 0.8 dB,
//! [`APP_CLAMP_I8`] roughly doubles the block error rate (0.165 to 0.312)
//! and raises the bit error rate by about eighty times, because a clamped
//! posterior turns decodes that were converging into decodes that stick.
//!
//! The same sweep shows how little width is actually needed: every clamp
//! from 255 upward gives bit-identical results at this operating point, so
//! one bit beyond the messages already leaves the accumulator effectively
//! unclamped. [`APP_CLAMP_WIDE`] is the default anyway, for two reasons —
//! `i16` is the natural type for a value that needs nine bits, and
//! [`QcLdpcDecoder::decode_5g_i8`](crate::qc_ldpc::QcLdpcDecoder::decode_5g_i8)
//! pins filler bits at the clamp, which has to stay above the $127 \cdot 30$
//! that all incident check messages together could subtract.
//!
//! # Choosing the scale
//!
//! The scale is the one parameter that has to be matched to the channel: too
//! small and the LLR distribution is coarsely resolved, too large and its
//! tails clip. The sweep in the same test file finds a broad plateau — on
//! both base graphs every $s$ from 8 to 24 is indistinguishable, while $s =
//! 2$ and $s = 32$ are resolvably worse. [`DEFAULT_SCALE`] sits at the lower
//! edge of that plateau deliberately: the optimum falls as the operating
//! $E_b/N_0$ rises, because a better channel widens the LLR distribution and
//! makes clipping the binding constraint, and $s = 8$ is inside the plateau
//! at both ends of the range measured. Re-run the sweep for your own
//! operating point rather than assuming one constant transfers, exactly as
//! `tests/ldpc_offset_beta_sweep.rs` advises for $\beta$.
//!
//! # Examples
//!
//! The whole fixed-point path, from channel LLRs to hard decisions:
//!
//! ```
//! use syndrome::qc_ldpc::{BaseGraph, QcLdpcDecoder};
//! use syndrome::quantize::{QuantParams, quantize_llr_i16};
//!
//! let quant = QuantParams::default();
//! let dec = QcLdpcDecoder::with_lifting_size(BaseGraph::Bg1, 32, 0.5).unwrap();
//! let n = dec.variable_node_count();
//!
//! // A confident all-zero channel, as f32 LLRs.
//! let llr = vec![4.0f32; n];
//!
//! // Quantize into the decoder's i16 posterior buffer, then decode.
//! let mut app = vec![0i16; n];
//! quantize_llr_i16(&llr, &mut app, quant.scale);
//! let mut edge_r = vec![0i8; dec.required_edge_buffer()];
//! let mut scratch = vec![0i8; dec.required_layer_buffer()];
//! let mut hard = vec![0u8; n];
//! dec.decode_layered_offset_min_sum_i8(
//!     &mut app, &mut edge_r, &mut scratch, &mut hard, 10, quant,
//! ).unwrap();
//! assert!(hard.iter().all(|&b| b == 0));
//! ```
//!
//! The plain `i8` conversions, for a caller holding messages rather than
//! posteriors:
//!
//! ```
//! use syndrome::quantize::{quantize_llr, dequantize_llr};
//!
//! let f = [1.5f32, -3.0, 0.0, 200.0];
//! let mut q = [0i8; 4];
//! quantize_llr(&f, &mut q, 8.0);
//! let mut out = [0.0f32; 4];
//! dequantize_llr(&q, &mut out, 8.0);
//! // Large values saturate at the message ceiling.
//! assert_eq!(q[3], 127);
//! ```

/// Quantise f32 LLRs to i8 with saturation.
///
/// $\hat{L}_i = \text{clamp}(\lfloor L_i \cdot \text{scale} \rceil, -127, 127)$
///
/// The value −128 is never produced; see [`MSG_MAX`] for why it is excluded.
///
/// # Arguments
///
/// * `llr`   - Input f32 LLR slice.
/// * `out`   - Output i8 slice, must be the same length.
/// * `scale` - Scale factor (inverse of the LLR step size).
pub fn quantize_llr(llr: &[f32], out: &mut [i8], scale: f32) {
    debug_assert_eq!(llr.len(), out.len());
    for (l, o) in llr.iter().zip(out.iter_mut()) {
        let v = (l * scale).round() as i32;
        *o = v.clamp(-127, 127) as i8;
    }
}

/// Dequantise i8 LLRs back to f32 (for diagnostics / BER simulation).
///
/// # Arguments
///
/// * `q`     - Input i8 LLR slice.
/// * `out`   - Output f32 slice, must be the same length.
/// * `scale` - Same scale factor used in [`quantize_llr`].
pub fn dequantize_llr(q: &[i8], out: &mut [f32], scale: f32) {
    debug_assert_eq!(q.len(), out.len());
    let inv = 1.0 / scale;
    for (qi, o) in q.iter().zip(out.iter_mut()) {
        *o = (*qi as f32) * inv;
    }
}

// ---------------------------------------------------------------------------
// Fixed-point decoder parameters
// ---------------------------------------------------------------------------

/// Default LLR scale factor $s$ used by the fixed-point LOMS path.
///
/// At $s = 8$ one quantization step is $1/8$ of an LLR unit and the
/// representable range is $\pm 15.875$, which covers a BPSK working-point
/// LLR distribution at the $E_b/N_0$ values where a rate-1/3 to rate-8/9
/// 5G code operates without spending resolution on magnitudes that never
/// occur.
pub const DEFAULT_SCALE: f32 = 8.0;

/// Largest magnitude any quantized message may take.
///
/// $-128$ is deliberately excluded so that negating or taking the absolute
/// value of a message is always representable — `i8::MIN.abs()` overflows,
/// and `_mm256_abs_epi8` returns $-128$ unchanged, so a single $-128$
/// anywhere would make the scalar and AVX2 kernels disagree.
pub const MSG_MAX: i8 = 127;

/// A-posteriori clamp that emulates an 8-bit posterior accumulator: the
/// running belief is held to the same $\pm 127$ range as the messages.
pub const APP_CLAMP_I8: i16 = MSG_MAX as i16;

/// A-posteriori clamp wide enough that the posterior never clips.
///
/// The posterior of a variable node is its channel LLR plus one
/// check-to-variable message per incident edge, each bounded by
/// [`MSG_MAX`], so $\lvert \text{APP} \rvert \le 127 \cdot (1 + d_v)$ where
/// $d_v$ is the variable-node degree. The largest column degree in either
/// 5G NR base graph is 30 (BG1 column 0), giving a worst case of 3937 —
/// comfortably inside `i16`, so at this setting the clamp is unreachable
/// and the accumulator behaves as an exact 16-bit sum.
pub const APP_CLAMP_WIDE: i16 = i16::MAX;

/// Fixed-point format for the i8 LOMS decode path.
///
/// Two numbers fix the format: the scale $s$ shared by
/// [`quantize_llr`]/[`dequantize_llr`], and the clamp applied to the
/// a-posteriori accumulator after every layer update. They are separate
/// because the messages and the posterior have genuinely different dynamic
/// ranges — see [`APP_CLAMP_WIDE`] — and collapsing them to one width
/// costs error-rate performance that
/// `tests/ldpc_int8_quantization_loss.rs` measures directly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantParams {
    /// Scale factor $s$: an LLR of $L$ is stored as $\mathrm{round}(L \cdot
    /// s)$ saturated to $\pm$[`MSG_MAX`].
    pub scale: f32,
    /// Symmetric clamp on the a-posteriori accumulator. [`APP_CLAMP_WIDE`]
    /// leaves it effectively unclamped; [`APP_CLAMP_I8`] reproduces an 8-bit
    /// posterior.
    pub app_clamp: i16,
}

impl Default for QuantParams {
    /// [`DEFAULT_SCALE`] with a non-clipping posterior ([`APP_CLAMP_WIDE`]).
    fn default() -> Self {
        Self {
            scale: DEFAULT_SCALE,
            app_clamp: APP_CLAMP_WIDE,
        }
    }
}

impl QuantParams {
    /// Replace the scale factor.
    ///
    /// # Examples
    ///
    /// ```
    /// use syndrome::quantize::QuantParams;
    ///
    /// assert_eq!(QuantParams::default().with_scale(4.0).scale, 4.0);
    /// ```
    #[must_use]
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// Replace the a-posteriori clamp.
    ///
    /// # Examples
    ///
    /// ```
    /// use syndrome::quantize::{APP_CLAMP_I8, QuantParams};
    ///
    /// let p = QuantParams::default().with_app_clamp(APP_CLAMP_I8);
    /// assert_eq!(p.app_clamp, 127);
    /// ```
    #[must_use]
    pub fn with_app_clamp(mut self, app_clamp: i16) -> Self {
        self.app_clamp = app_clamp;
        self
    }

    /// Quantize the offset correction $\beta$ into the message format.
    ///
    /// The check-node update subtracts $\beta$ from a *message magnitude*, so
    /// it must be expressed in the same units: $\beta_q = \mathrm{round}(\beta
    /// \cdot s)$. At the crate's default $\beta = 0.5$ and $s = 8$ this is
    /// exactly 4, so the fixed-point decoder applies the same correction the
    /// `f32` decoder does with no rounding error at all.
    ///
    /// Negative or non-finite $\beta$ is clamped into $[0,$ [`MSG_MAX`]$]$;
    /// a negative offset would *add* confidence, which is never the intent.
    ///
    /// # Arguments
    ///
    /// * `offset_beta` — the `f32` offset $\beta$.
    ///
    /// # Returns
    ///
    /// $\beta_q$ in $[0,$ [`MSG_MAX`]$]$.
    ///
    /// # Examples
    ///
    /// ```
    /// use syndrome::quantize::QuantParams;
    ///
    /// // The crate's default beta = 0.5 at scale 8 quantizes exactly.
    /// assert_eq!(QuantParams::default().beta_q(0.5), 4);
    /// assert_eq!(QuantParams::default().beta_q(0.0), 0);
    /// ```
    #[must_use]
    pub fn beta_q(&self, offset_beta: f32) -> i8 {
        let scaled = offset_beta * self.scale;
        if !scaled.is_finite() {
            return MSG_MAX;
        }
        scaled.round().clamp(0.0, MSG_MAX as f32) as i8
    }
}

/// Quantize f32 LLRs into an `i16` a-posteriori buffer.
///
/// The values produced are in the *message* range $\pm$[`MSG_MAX`] exactly as
/// [`quantize_llr`] would produce, but stored `i16`-wide because that is the
/// buffer the fixed-point decoder accumulates its posterior in
/// ([`crate::qc_ldpc::QcLdpcDecoder::decode_layered_offset_min_sum_i8`]).
/// This is the normal way to turn a channel LLR vector into that decoder's
/// input.
///
/// # Arguments
///
/// * `llr`   - Input f32 LLR slice.
/// * `out`   - Output i16 slice, must be the same length.
/// * `scale` - Scale factor $s$.
///
/// # Examples
///
/// ```
/// use syndrome::quantize::{quantize_llr_i16, DEFAULT_SCALE};
///
/// let mut app = [0i16; 3];
/// quantize_llr_i16(&[1.0, -0.25, 99.0], &mut app, DEFAULT_SCALE);
/// assert_eq!(app, [8, -2, 127]);
/// ```
pub fn quantize_llr_i16(llr: &[f32], out: &mut [i16], scale: f32) {
    debug_assert_eq!(llr.len(), out.len());
    for (l, o) in llr.iter().zip(out.iter_mut()) {
        let v = l * scale;
        *o = if v.is_nan() {
            0
        } else {
            (v.round() as i32).clamp(-(MSG_MAX as i32), MSG_MAX as i32) as i16
        };
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_clamps_to_127() {
        let mut q = [0i8; 1];
        quantize_llr(&[1e9f32], &mut q, 8.0);
        assert_eq!(q[0], 127);
    }

    #[test]
    fn negative_clamps_to_neg127() {
        let mut q = [0i8; 1];
        quantize_llr(&[-1e9f32], &mut q, 8.0);
        assert_eq!(q[0], -127);
    }

    #[test]
    fn minus128_never_produced() {
        let mut q = [0i8; 1];
        quantize_llr(&[f32::NEG_INFINITY], &mut q, 8.0);
        assert_ne!(q[0], i8::MIN);
    }

    #[test]
    fn beta_quantizes_exactly_at_the_default_scale() {
        // beta = 0.5 at s = 8 is an exact multiple of the step, so the
        // fixed-point decoder applies the same offset the f32 one does with
        // no rounding error at all. This is why the crate's chosen beta and
        // its chosen scale are worth keeping compatible.
        let p = QuantParams::default();
        assert_eq!(p.beta_q(0.5), 4);
        assert_eq!(p.beta_q(0.125), 1);
        assert_eq!(p.beta_q(0.0), 0);
    }

    #[test]
    fn beta_is_clamped_into_the_message_range() {
        let p = QuantParams::default();
        // A negative offset would *add* confidence, which is never intended.
        assert_eq!(p.beta_q(-1.0), 0);
        // Anything past the message ceiling saturates rather than wrapping.
        assert_eq!(p.beta_q(1e6), MSG_MAX);
        assert_eq!(p.beta_q(f32::INFINITY), MSG_MAX);
        assert_eq!(p.beta_q(f32::NAN), MSG_MAX);
    }

    #[test]
    fn i16_quantization_saturates_at_the_message_ceiling() {
        // The posterior buffer is i16-wide, but the values that enter it are
        // channel LLRs and must start inside the *message* range.
        let mut out = [0i16; 5];
        quantize_llr_i16(
            &[0.0, 1.0, -1.0, 1e9, f32::NEG_INFINITY],
            &mut out,
            DEFAULT_SCALE,
        );
        assert_eq!(out, [0, 8, -8, 127, -127]);
    }

    #[test]
    fn i16_quantization_maps_nan_to_zero() {
        // NaN carries no information about the bit, and zero is the LLR that
        // says exactly that. It also matches `quantize_llr`, whose saturating
        // float-to-int cast sends NaN to 0.
        let mut wide = [99i16; 1];
        quantize_llr_i16(&[f32::NAN], &mut wide, DEFAULT_SCALE);
        assert_eq!(wide[0], 0);
        let mut narrow = [99i8; 1];
        quantize_llr(&[f32::NAN], &mut narrow, DEFAULT_SCALE);
        assert_eq!(narrow[0], 0);
    }

    #[test]
    fn wide_clamp_exceeds_the_largest_reachable_posterior() {
        // The posterior is the channel LLR plus one message per incident
        // edge. BG1 column 0 has degree 30, the largest in either 5G base
        // graph, so nothing can exceed 127 * 31 -- and APP_CLAMP_WIDE has to
        // sit above that for the clamp to be unreachable, which is what makes
        // it a faithful 16-bit accumulator rather than a narrow one in
        // disguise.
        const MAX_COLUMN_DEGREE: i32 = 30;
        let reachable = MSG_MAX as i32 * (1 + MAX_COLUMN_DEGREE);
        assert_eq!(reachable, 3937);
        assert!(
            i32::from(APP_CLAMP_WIDE) > reachable,
            "APP_CLAMP_WIDE = {APP_CLAMP_WIDE} does not exceed the reachable posterior {reachable}"
        );
        assert_eq!(APP_CLAMP_I8, MSG_MAX as i16);
    }

    #[test]
    fn roundtrip_within_range() {
        let scale = 8.0f32;
        let vals = [-10.0f32, -1.0, 0.0, 0.5, 1.0, 10.0];
        let mut q = [0i8; 6];
        quantize_llr(&vals, &mut q, scale);
        let mut out = [0.0f32; 6];
        dequantize_llr(&q, &mut out, scale);
        for (&orig, &dq) in vals.iter().zip(out.iter()) {
            let err = (orig - dq).abs();
            assert!(
                err <= 1.0 / scale + 1e-5,
                "quantization error {err} exceeds 1 LSB for {orig}"
            );
        }
    }
}
