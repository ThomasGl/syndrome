//! Hamming(7,4) single-error-correcting code.
//!
//! Encodes 4 data bits into a 7-bit codeword by adding 3 parity bits, each
//! parity bit checking a different, overlapping subset of the 7 positions.
//! To decode, the same 3 checks are recomputed on the received word: if all
//! three agree, nothing is wrong; if some disagree, the exact *pattern* of
//! which checks failed identifies the single bit position to flip. This is
//! the smallest nontrivial linear error-correcting code and the standard
//! first example of syndrome decoding.
//!
//! # Codeword layout
//!
//! Bit positions are 1-indexed (the classical presentation of the code);
//! [`Hamming74::encode`] packs them into a `u8` LSB-first, so position $j$
//! lives at bit `j - 1`:
//!
//! ```text
//! position:   1    2    3    4    5    6    7
//!            [p1] [p2] [d3] [p4] [d2] [d1] [d0]
//! ```
//!
//! `p1`, `p2`, `p4` are parity bits; `d3 d2 d1 d0` is the 4-bit input nibble
//! (`d3` = most significant bit of the nibble).
//!
//! # Parity-check matrix
//!
//! $$
//! H = \begin{pmatrix}
//! 1 & 0 & 1 & 0 & 1 & 0 & 1 \\
//! 0 & 1 & 1 & 0 & 0 & 1 & 1 \\
//! 0 & 0 & 0 & 1 & 1 & 1 & 1
//! \end{pmatrix}
//! $$
//!
//! Row $i$ (for $i = 0, 1, 2$, corresponding to parity bits $p_1, p_2, p_4$)
//! covers exactly the positions whose binary representation has bit $i$ set.
//! Equivalently: column $j$ of $H$, read top-to-bottom as a 3-bit binary
//! number, equals $j$ itself. That single fact is the entire decoding
//! algorithm — see below. `H_ROWS` (private, see its own doc comment) is
//! this matrix's rows, packed the same
//! way [`Hamming74::encode`] packs a codeword.
//!
//! # Syndrome decoding
//!
//! For a received word $r \in \mathbb{F}_2^7$, the syndrome is
//! $$ s = H\,r^{\mathsf T} \in \mathbb{F}_2^3. $$
//! If $r$ is a valid codeword (no error), $s = 0$: every parity check
//! agrees. If exactly one bit at position $j$ was flipped, $r = c \oplus
//! e_j$ for the true codeword $c$, so
//! $$ s = H c^{\mathsf T} \oplus H e_j^{\mathsf T} = 0 \oplus (\text{column } j \text{ of } H) = j $$
//! (using $Hc^\mathsf{T} = 0$ for any codeword, and that column $j$ of $H$
//! *is* $j$ as a 3-bit integer, per the previous section). So a nonzero
//! syndrome's numeric *value* directly names the 1-indexed bit position to
//! flip — no lookup table needed, just three parity computations and an
//! index. The private `syndrome` function computes $s$ directly from `H_ROWS`; a
//! `#[cfg(test)]` test in this module checks that identity exhaustively
//! (against an independent re-derivation of the classic $s_1, s_2, s_4$
//! formula) for all 128 possible received 7-bit words, and a second test
//! checks the "syndrome value = position to flip" property directly for
//! every codeword and every single-bit error.
//!
//! # Examples
//!
//! ```
//! use syndrome::encode_hamming_7_4;
//! let code = encode_hamming_7_4(0b1010);
//! // encoded 7-bit value for 0b1010 is 0b0101101 (decimal 45)
//! assert_eq!(code & 0x7F, 0b0101101);
//! ```

use crate::error::FecError;

/// Parity-check matrix $H$ for Hamming(7,4) (see module docs), one row per
/// parity-check equation ($p_1$, $p_2$, $p_4$ respectively). Each row is
/// packed as a 7-bit mask over codeword positions $1..=7$: bit $k$ (value
/// $2^k$) represents position $k + 1$, the same LSB-first convention
/// [`Hamming74::encode`] uses for its output `u8`.
///
/// Spelled out (most-significant bit = position 7, down to least-significant
/// = position 1):
/// * `H_ROWS[0]` ($p_1$'s row) = `0b1010101`: covers positions 1, 3, 5, 7.
/// * `H_ROWS[1]` ($p_2$'s row) = `0b1100110`: covers positions 2, 3, 6, 7.
/// * `H_ROWS[2]` ($p_4$'s row) = `0b1111000`: covers positions 4, 5, 6, 7.
const H_ROWS: [u8; 3] = [0b1010101, 0b1100110, 0b1111000];

/// Compute the 3-bit syndrome $s = H\,r^{\mathsf T}$ (over $\mathrm{GF}(2)$)
/// of a received 7-bit word `r` (low 7 bits significant; same bit layout as
/// [`Hamming74::encode`]'s output — see module docs).
///
/// Branch-free: each syndrome bit is the parity of `r` ANDed with the
/// matching row of [`H_ROWS`] (`count_ones() & 1`). Bit `k` of the result is
/// the check for parity bit $2^k$, so the packed value $s = s_1 + 2 s_2 + 4
/// s_4$ is — for a received word with at most one bit error — exactly the
/// 1-indexed position of the flipped bit, or `0` if there is no error (see
/// module docs for the derivation).
#[inline]
fn syndrome(r: u8) -> u8 {
    let r = r & 0x7F;
    let mut s = 0u8;
    for (k, &row) in H_ROWS.iter().enumerate() {
        s |= ((r & row).count_ones() as u8 & 1) << k;
    }
    s
}

/// Hamming(7,4) single-error-correcting encoder and decoder.
///
/// See the [module-level documentation](self) for the codeword layout,
/// parity-check matrix $H$, and the syndrome decoding rule.
pub struct Hamming74;

impl Hamming74 {
    /// Encode 4-bit nibble into a 7-bit Hamming(7,4) code stored in the low bits
    /// of a `u8`.
    ///
    /// Computes the three parity bits ($p_1$, $p_2$, $p_4$) so that the
    /// resulting 7-bit word satisfies $Hc^{\mathsf T} = 0$ for the matrix
    /// $H$ documented on the module — see there for the exact bit layout.
    ///
    /// # Arguments
    ///
    /// * `nibble` - Lower 4 bits are encoded.
    ///
    /// # Returns
    ///
    /// Encoded 7-bit code in a `u8`.
    ///
    /// # Examples
    ///
    /// ```
    /// use syndrome::Hamming74;
    /// let code = Hamming74::encode(0b1010);
    /// assert_eq!(code & 0x7F, 0b0101101);
    /// ```
    pub fn encode(nibble: u8) -> u8 {
        let d = nibble & 0x0F;
        // data bits d3 d2 d1 d0 map to positions 3,5,6,7 (1-based)
        let d0 = d & 1;
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
        code |= p1 & 1; // pos1
        code |= (p2 & 1) << 1; // pos2
        code |= (d3 & 1) << 2; // pos3 (data bit d3)
        code |= (p4 & 1) << 3; // pos4
        code |= (d2 & 1) << 4; // pos5
        code |= (d1 & 1) << 5; // pos6
        code |= (d0 & 1) << 6; // pos7
        code
    }

    /// Decode a 7-bit Hamming code (stored in low bits of `u8`). Attempts to
    /// correct a single-bit error.
    ///
    /// Computes the syndrome $s = H r^{\mathsf T}$ via the private `syndrome`
    /// helper; if
    /// $s \ne 0$, its value is the 1-indexed position of the single bit to
    /// flip (see module docs for why).
    ///
    /// # Arguments
    ///
    /// * `code` - Encoded 7-bit code in low bits of `u8`.
    ///
    /// # Returns
    ///
    /// `Ok(nibble)` with the corrected 4-bit data.
    ///
    /// # Errors
    ///
    /// This single-error-correcting decode is total over all 7-bit inputs
    /// (Hamming(7,4)'s syndrome always maps to a correctable single-bit
    /// position or to zero) and currently never fails, but returns
    /// [`FecError`] rather than a bare `u8` to match the crate-wide error
    /// convention and leave room for future stricter validation.
    ///
    /// # Examples
    ///
    /// ```
    /// use syndrome::Hamming74;
    /// let code = Hamming74::encode(0b0110);
    /// let corrected = code ^ 0b0000_0100; // flip bit at position 3
    /// assert_eq!(Hamming74::decode(corrected).unwrap(), 0b0110);
    /// ```
    pub fn decode(code: u8) -> Result<u8, FecError> {
        let r = code & 0x7F;
        let s = syndrome(r);
        let mut corrected = r;
        if s != 0 {
            let pos = s; // 1-based position to flip
            if (1..=7).contains(&pos) {
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

/// Free-function form of [`Hamming74::encode`]; encodes a 4-bit nibble into
/// a 7-bit Hamming(7,4) code.
///
/// # Arguments
///
/// * `nibble` - Lower 4 bits are encoded.
///
/// # Returns
///
/// Encoded 7-bit code in a `u8`.
///
/// # Examples
///
/// ```
/// use syndrome::encode_hamming_7_4;
/// assert_eq!(encode_hamming_7_4(0b1010) & 0x7F, 0b0101101);
/// ```
pub fn encode_hamming_7_4(nibble: u8) -> u8 {
    Hamming74::encode(nibble)
}

/// Free-function form of [`Hamming74::decode`]; decodes a 7-bit Hamming
/// code, correcting a single-bit error if present.
///
/// # Arguments
///
/// * `code` - Encoded 7-bit code in low bits of `u8`.
///
/// # Returns
///
/// `Ok(nibble)` with the corrected 4-bit data.
///
/// # Errors
///
/// See [`Hamming74::decode`]: total over all 7-bit inputs, never fails.
///
/// # Examples
///
/// ```
/// use syndrome::{decode_hamming_7_4, encode_hamming_7_4};
/// let code = encode_hamming_7_4(0b1111);
/// assert_eq!(decode_hamming_7_4(code).unwrap(), 0b1111);
/// ```
pub fn decode_hamming_7_4(code: u8) -> Result<u8, FecError> {
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

    /// Closes the exact gap a code reviewer flagged: the module docs assert
    /// `syndrome == H * received`, and this test verifies that identity
    /// exhaustively (all 128 possible 7-bit received words) against an
    /// independent re-derivation of the classic `s1`/`s2`/`s4` parity
    /// formula, so the matrix-vector claim is checked against the actual
    /// decoder, not just asserted in prose.
    #[test]
    fn syndrome_matches_h_matrix_exhaustively() {
        for r in 0u8..128u8 {
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
            let expected = (s4 << 2) | (s2 << 1) | s1;

            assert_eq!(
                syndrome(r),
                expected,
                "syndrome(H, r={r:#09b}) must match the classic s1/s2/s4 formula"
            );
        }
    }

    /// H_ROWS must exactly match the parity-check matrix documented on the
    /// module (each row's covered positions).
    #[test]
    fn h_rows_match_documented_matrix() {
        assert_eq!(H_ROWS[0], 0b1010101, "p1 row must cover positions 1,3,5,7");
        assert_eq!(H_ROWS[1], 0b1100110, "p2 row must cover positions 2,3,6,7");
        assert_eq!(H_ROWS[2], 0b1111000, "p4 row must cover positions 4,5,6,7");
    }

    /// The syndrome's *value* must equal the 1-indexed position of a single
    /// bit error, for every codeword and every position — the key property
    /// that makes this decoder a table-free arithmetic step (see module
    /// docs).
    #[test]
    fn nonzero_syndrome_value_equals_error_position() {
        for n in 0u8..16u8 {
            let c = encode_hamming_7_4(n);
            for pos in 1u8..=7u8 {
                let flipped = c ^ (1 << (pos - 1));
                assert_eq!(
                    syndrome(flipped & 0x7F),
                    pos,
                    "flipping position {pos} of codeword for nibble {n:#06b} \
                     must yield syndrome value {pos}"
                );
            }
        }
    }
}
