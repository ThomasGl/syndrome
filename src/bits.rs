//! Bit-buffer utilities shared by every codec in the crate.
//!
//! Almost every API in this library represents binary data as **one bit per
//! byte**: a `&[u8]` whose elements are all `0` or `1`, most-significant bit
//! of the original byte stream first. Real applications hold packed bytes, so
//! every user needs the same two conversions and the same LLR hard-decision
//! rule. This module makes them public instead of leaving each caller to
//! reimplement them.
//!
//! # Conventions
//!
//! - **Bit order is MSB-first**, matching the rest of the crate: byte `0xB4 =
//!   0b1011_0100` unpacks to `[1, 0, 1, 1, 0, 1, 0, 0]`. (The one exception
//!   in the crate is [`crate::golay`], whose published generator is
//!   LSB-first-indexed; its module docs cover that.)
//! - **LLR sign follows the crate-wide rule**: a positive log-likelihood
//!   ratio $L = \ln\frac{P(b=0)}{P(b=1)}$ favours bit 0, a negative one
//!   favours bit 1, so the hard decision is $\hat{b} = \mathbb{1}\lbrace L < 0\rbrace$.
//!   An exact zero (an erasure) decides 0, consistent with
//!   `f32::is_sign_negative` being false for `+0.0`.

use crate::alloc_prelude::*;
use crate::error::FecError;

/// Unpack packed bytes into one-bit-per-byte form, MSB first, into a
/// caller-provided buffer.
///
/// # Arguments
///
/// * `bytes` - Packed input bytes.
/// * `bits` - Output buffer; must hold exactly `8 * bytes.len()` elements.
///   Each element is written as `0` or `1`.
///
/// # Errors
///
/// Returns [`FecError::BufferTooSmall`] if `bits` is not exactly
/// `8 * bytes.len()` long. (An oversized buffer is rejected too: a partial
/// write with stale tail bytes is a silent-corruption hazard, not a
/// convenience.)
///
/// # Examples
///
/// ```
/// use syndrome::bits::bytes_to_bits;
///
/// let mut bits = [0u8; 8];
/// bytes_to_bits(&[0b1011_0100], &mut bits).unwrap();
/// assert_eq!(bits, [1, 0, 1, 1, 0, 1, 0, 0]);
/// ```
pub fn bytes_to_bits(bytes: &[u8], bits: &mut [u8]) -> Result<(), FecError> {
    let required = bytes.len() * 8;
    if bits.len() != required {
        return Err(FecError::BufferTooSmall {
            required,
            provided: bits.len(),
        });
    }
    for (byte_idx, &byte) in bytes.iter().enumerate() {
        let out = &mut bits[byte_idx * 8..byte_idx * 8 + 8];
        for (bit_idx, slot) in out.iter_mut().enumerate() {
            *slot = (byte >> (7 - bit_idx)) & 1;
        }
    }
    Ok(())
}

/// Unpack packed bytes into a freshly allocated one-bit-per-byte `Vec`,
/// MSB first.
///
/// Allocating convenience over [`bytes_to_bits`]; use the buffer-taking form
/// in loops that must not allocate.
///
/// # Arguments
///
/// * `bytes` - Packed input bytes.
///
/// # Returns
///
/// A `Vec<u8>` of length `8 * bytes.len()` whose elements are all `0` or `1`.
///
/// # Examples
///
/// ```
/// use syndrome::bits::bytes_to_bits_vec;
///
/// assert_eq!(bytes_to_bits_vec(&[0xF0]), vec![1, 1, 1, 1, 0, 0, 0, 0]);
/// ```
#[must_use]
pub fn bytes_to_bits_vec(bytes: &[u8]) -> Vec<u8> {
    let mut bits = vec![0u8; bytes.len() * 8];
    // Infallible: the buffer is sized exactly above.
    let _ = bytes_to_bits(bytes, &mut bits);
    bits
}

/// Pack a one-bit-per-byte buffer into bytes, MSB first, into a
/// caller-provided buffer.
///
/// # Arguments
///
/// * `bits` - Input bits, one per byte, each element `0` or `1`. The length
///   must be a multiple of 8.
/// * `bytes` - Output buffer; must hold exactly `bits.len() / 8` elements.
///
/// # Errors
///
/// * [`FecError::InvalidParam`] if `bits.len()` is not a multiple of 8, or
///   if any element of `bits` is neither `0` nor `1`. A stray `0xFF` in a
///   bit buffer is always an upstream bug, and masking it away here would
///   hide it.
/// * [`FecError::BufferTooSmall`] if `bytes` is not exactly `bits.len() / 8`
///   long.
///
/// # Examples
///
/// ```
/// use syndrome::bits::bits_to_bytes;
///
/// let mut bytes = [0u8; 1];
/// bits_to_bytes(&[1, 0, 1, 1, 0, 1, 0, 0], &mut bytes).unwrap();
/// assert_eq!(bytes, [0b1011_0100]);
/// ```
pub fn bits_to_bytes(bits: &[u8], bytes: &mut [u8]) -> Result<(), FecError> {
    if !bits.len().is_multiple_of(8) {
        return Err(FecError::InvalidParam(
            "bits length must be a multiple of 8",
        ));
    }
    let required = bits.len() / 8;
    if bytes.len() != required {
        return Err(FecError::BufferTooSmall {
            required,
            provided: bytes.len(),
        });
    }
    if bits.iter().any(|&b| b > 1) {
        return Err(FecError::InvalidParam(
            "bit buffer contains a value other than 0 or 1",
        ));
    }
    for (byte_idx, chunk) in bits.chunks_exact(8).enumerate() {
        let mut byte = 0u8;
        for &bit in chunk {
            byte = (byte << 1) | bit;
        }
        bytes[byte_idx] = byte;
    }
    Ok(())
}

/// Pack a one-bit-per-byte buffer into a freshly allocated byte `Vec`,
/// MSB first.
///
/// Allocating convenience over [`bits_to_bytes`]; use the buffer-taking form
/// in loops that must not allocate.
///
/// # Arguments
///
/// * `bits` - Input bits, one per byte, each element `0` or `1`. The length
///   must be a multiple of 8.
///
/// # Returns
///
/// A `Vec<u8>` of length `bits.len() / 8`.
///
/// # Errors
///
/// Same conditions as [`bits_to_bytes`]: a length that is not a multiple of
/// 8, or an element that is neither `0` nor `1`.
///
/// # Examples
///
/// ```
/// use syndrome::bits::{bits_to_bytes_vec, bytes_to_bits_vec};
///
/// let original = b"syndrome";
/// let bits = bytes_to_bits_vec(original);
/// assert_eq!(bits_to_bytes_vec(&bits).unwrap(), original);
/// ```
pub fn bits_to_bytes_vec(bits: &[u8]) -> Result<Vec<u8>, FecError> {
    let mut bytes = vec![0u8; bits.len() / 8];
    bits_to_bytes(bits, &mut bytes)?;
    Ok(bytes)
}

/// Hard-decide a buffer of LLRs into bits, using the crate-wide sign
/// convention ($L \geq 0 \Rightarrow 0$, $L < 0 \Rightarrow 1$).
///
/// # Arguments
///
/// * `llr` - Log-likelihood ratios, one per bit.
/// * `bits` - Output buffer; must hold exactly `llr.len()` elements. Each is
///   written as `0` or `1`.
///
/// # Errors
///
/// Returns [`FecError::BufferTooSmall`] if `bits.len() != llr.len()`.
///
/// # Examples
///
/// ```
/// use syndrome::bits::hard_decision;
///
/// let mut bits = [0u8; 4];
/// hard_decision(&[3.2, -0.5, 0.0, -7.1], &mut bits).unwrap();
/// assert_eq!(bits, [0, 1, 0, 1]);
/// ```
pub fn hard_decision(llr: &[f32], bits: &mut [u8]) -> Result<(), FecError> {
    if bits.len() != llr.len() {
        return Err(FecError::BufferTooSmall {
            required: llr.len(),
            provided: bits.len(),
        });
    }
    for (slot, &l) in bits.iter_mut().zip(llr.iter()) {
        *slot = (l < 0.0) as u8;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_every_byte_value() {
        let bytes: Vec<u8> = (0..=255).collect();
        let bits = bytes_to_bits_vec(&bytes);
        assert_eq!(bits.len(), 256 * 8);
        assert!(bits.iter().all(|&b| b <= 1));
        assert_eq!(bits_to_bytes_vec(&bits).unwrap(), bytes);
    }

    #[test]
    fn msb_first_order_is_exact() {
        let mut bits = [0u8; 8];
        bytes_to_bits(&[0x01], &mut bits).unwrap();
        assert_eq!(bits, [0, 0, 0, 0, 0, 0, 0, 1]);
        bytes_to_bits(&[0x80], &mut bits).unwrap();
        assert_eq!(bits, [1, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn unpack_rejects_wrong_size_buffer_both_directions() {
        let mut too_small = [0u8; 7];
        let mut too_large = [0u8; 9];
        assert!(matches!(
            bytes_to_bits(&[0xAA], &mut too_small),
            Err(FecError::BufferTooSmall {
                required: 8,
                provided: 7
            })
        ));
        assert!(matches!(
            bytes_to_bits(&[0xAA], &mut too_large),
            Err(FecError::BufferTooSmall {
                required: 8,
                provided: 9
            })
        ));
    }

    #[test]
    fn pack_rejects_non_multiple_of_8() {
        let mut bytes = [0u8; 1];
        assert!(matches!(
            bits_to_bytes(&[1, 0, 1], &mut bytes),
            Err(FecError::InvalidParam(_))
        ));
    }

    #[test]
    fn pack_rejects_non_bit_values() {
        let mut bytes = [0u8; 1];
        assert!(matches!(
            bits_to_bytes(&[1, 0, 1, 2, 0, 0, 0, 0], &mut bytes),
            Err(FecError::InvalidParam(_))
        ));
        assert!(bits_to_bytes_vec(&[0xFF; 8]).is_err());
    }

    #[test]
    fn pack_rejects_wrong_output_size() {
        let mut bytes = [0u8; 2];
        assert!(matches!(
            bits_to_bytes(&[0u8; 8], &mut bytes),
            Err(FecError::BufferTooSmall {
                required: 1,
                provided: 2
            })
        ));
    }

    #[test]
    fn hard_decision_sign_convention() {
        let mut bits = [9u8; 5];
        hard_decision(&[1.0, -1.0, 0.0, -0.0, f32::MIN_POSITIVE], &mut bits).unwrap();
        // +0.0 and -0.0 both compare `< 0.0` as false, so an erasure decides 0.
        assert_eq!(bits, [0, 1, 0, 0, 0]);
    }

    #[test]
    fn hard_decision_rejects_mismatched_lengths() {
        let mut bits = [0u8; 3];
        assert!(matches!(
            hard_decision(&[1.0, -1.0], &mut bits),
            Err(FecError::BufferTooSmall { .. })
        ));
    }

    #[test]
    fn empty_inputs_are_valid() {
        let mut none: [u8; 0] = [];
        bytes_to_bits(&[], &mut none).unwrap();
        bits_to_bytes(&[], &mut none).unwrap();
        hard_decision(&[], &mut none).unwrap();
        assert!(bytes_to_bits_vec(&[]).is_empty());
        assert!(bits_to_bytes_vec(&[]).unwrap().is_empty());
    }
}
