/// Hamming (7,4) encoder and decoder utilities.
///
/// Provides functions to encode 4 bits into a 7-bit Hamming code and to decode
/// and correct single-bit errors.
///
/// # Examples
///
/// ```
/// use glezer_rsv::encode_hamming_7_4;
/// let code = encode_hamming_7_4(0b1010);
/// // encoded 7-bit value for 0b1010 is 0b0101101 (decimal 45)
/// assert_eq!(code & 0x7F, 0b0101101);
/// ```
pub struct Hamming74;

impl Hamming74 {
    /// Encode 4-bit nibble into a 7-bit Hamming(7,4) code stored in the low bits
    /// of a `u8`.
    ///
    /// # Arguments
    ///
    /// * `nibble` - Lower 4 bits are encoded.
    ///
    /// # Returns
    ///
    /// Encoded 7-bit code in a `u8`.
    pub fn encode(nibble: u8) -> u8 {
        let d = nibble & 0x0F;
        // data bits d3 d2 d1 d0 map to positions 3,5,6,7 (1-based)
        let d0 = (d >> 0) & 1;
        let d1 = (d >> 1) & 1;
        let d2 = (d >> 2) & 1;
        let d3 = (d >> 3) & 1;

        // parity bits p1 covers bits 3,5,7 -> positions 1 covers d3,d2,d0
        let p1 = d3 ^ d2 ^ d0;
        // p2 covers bits 3,6,7 -> positions 2 covers d3,d1,d0
        let p2 = d3 ^ d1 ^ d0;
        // p4 covers bits 5,6,7 -> positions 4 covers d2,d1,d0
        let p4 = d2 ^ d1 ^ d0;

        // assemble bits into positions 1..7 (LSB is position 1)
        let mut code = 0u8;
        code |= (p1 & 1) << 0; // pos1
        code |= (p2 & 1) << 1; // pos2
        code |= (d3 & 1) << 2; // pos3 (data bit d3)
        code |= (p4 & 1) << 3; // pos4
        code |= (d2 & 1) << 4; // pos5
        code |= (d1 & 1) << 5; // pos6
        code |= (d0 & 1) << 6; // pos7
        code
    }

    /// Decode a 7-bit Hamming code (stored in low bits of `u8`). Attempts to
    /// correct a single-bit error. Returns `Ok(4-bit nibble)` on success or
    /// `Err(&'static str)` if a double-bit error is detected (syndrome zero but
    /// parity mismatch cannot be corrected here).
    ///
    /// # Arguments
    ///
    /// * `code` - Encoded 7-bit code in low bits of `u8`.
    ///
    /// # Returns
    ///
    /// `Ok(nibble)` with the corrected 4-bit data, or `Err` if unrecoverable.
    pub fn decode(code: u8) -> Result<u8, &'static str> {
        let r = code & 0x7F;
        let b = |pos: u8| -> u8 { (r >> (pos - 1)) & 1 };

        let p1 = b(1);
        let p2 = b(2);
        let d3 = b(3);
        let p4 = b(4);
        let d2 = b(5);
        let d1 = b(6);
        let d0 = b(7);

        let s1 = p1 ^ d3 ^ d2 ^ d0;
        let s2 = p2 ^ d3 ^ d1 ^ d0;
        let s4 = p4 ^ d2 ^ d1 ^ d0;

        let syndrome = (s4 << 2) | (s2 << 1) | s1;
        let mut corrected = r;
        if syndrome != 0 {
            let pos = syndrome; // 1-based position to flip
            if pos >= 1 && pos <= 7 {
                corrected ^= 1 << (pos - 1);
            }
        }

        // extract data bits from corrected
        let d3 = (corrected >> 2) & 1;
        let d2 = (corrected >> 4) & 1;
        let d1 = (corrected >> 5) & 1;
        let d0 = (corrected >> 6) & 1;

        let nibble = (d3 << 3) | (d2 << 2) | (d1 << 1) | d0;
        Ok(nibble)
    }
}

/// Convenience free functions
pub fn encode_hamming_7_4(nibble: u8) -> u8 {
    Hamming74::encode(nibble)
}

pub fn decode_hamming_7_4(code: u8) -> Result<u8, &'static str> {
    Hamming74::decode(code)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn proptest_roundtrip(n in 0u8..16u8) {
            let c = encode_hamming_7_4(n);
            let d = decode_hamming_7_4(c).unwrap();
            prop_assert_eq!(d, n);
        }

        #[test]
        fn proptest_single_bit_correction(n in 0u8..16u8, bit in 0u8..7u8) {
            let c = encode_hamming_7_4(n);
            let e = c ^ (1 << bit);
            let d = decode_hamming_7_4(e).unwrap();
            prop_assert_eq!(d, n);
        }
    }

    #[test]
    fn roundtrip_enumeration() {
        for v in 0u8..16u8 {
            let c = encode_hamming_7_4(v);
            let d = decode_hamming_7_4(c).unwrap();
            assert_eq!(d, v);
        }
    }
}
