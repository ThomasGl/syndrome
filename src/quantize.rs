//! LLR quantization for fixed-point LDPC decoding (Phase 2).
//!
//! Industrial LDPC decoders quantise soft log-likelihood ratios to 8-bit
//! integers to exploit 32-wide integer SIMD (vs 8-wide f32).  This module
//! provides the scalar quantization/dequantization layer.  The vectorized
//! i8 LOMS kernels that would consume it are planned but not yet
//! implemented — the shipping LDPC decode paths (scalar, AVX2, NEON)
//! operate on f32 LLRs.
//!
//! # Fixed-Point Format
//!
//! Each LLR is stored as a signed byte: $\hat{L} = \text{clamp}(L \cdot s,
//! -127, 127)$ where $s$ is a scale factor chosen to maximise the dynamic
//! range for the expected LLR distribution.  The value $-128$ is reserved as
//! a sentinel (unused) so that `i8::abs()` never overflows.
//!
//! Typical scale: $s = 8.0$ for 5G BPSK channels at operating SNR, chosen to
//! keep the dynamic range of a working-point LLR distribution inside the
//! `[-127, 127]` clamp without wasting precision on values that never occur.
//!
//! This crate has not yet measured its own quantization loss, because the
//! vectorized i8 LOMS kernel this module exists to feed is not implemented
//! (see above) — there is no decode path here to measure. For general
//! min-sum LDPC decoders, published results report well under 0.1 dB loss
//! at typical operating error rates for well-tuned 6-8 bit quantization
//! (e.g. the NSF-hosted survey "Modified Min-Sum Algorithm for Quantized
//! LDPC Decoders", <https://par.nsf.gov/servlets/purl/10156560>), but that
//! figure describes other implementations, not this one, until this crate
//! has a kernel and a BER/BLER sweep (via `bin/ber_sim.rs`) to back its own
//! number.
//!
//! # Examples
//!
//! ```
//! use syndrome::quantize::{quantize_llr, dequantize_llr};
//!
//! let f = [1.5f32, -3.0, 0.0, 200.0];
//! let mut q = [0i8; 4];
//! quantize_llr(&f, &mut q, 8.0);
//! let mut out = [0.0f32; 4];
//! dequantize_llr(&q, &mut out, 8.0);
//! // Large values clamp to ±127.
//! assert_eq!(q[3], 127);
//! ```

/// Quantise f32 LLRs to i8 with saturation.
///
/// $\hat{L}_i = \text{clamp}(\lfloor L_i \cdot \text{scale} \rceil, -127, 127)$
///
/// The value −128 is never produced (reserved as sentinel).
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
