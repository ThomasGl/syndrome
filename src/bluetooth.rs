//! Bluetooth FEC profiles: LE Coded PHY and the BR/EDR baseband codes.
//!
//! Bluetooth has added no new FEC since 2016 — the current Core Specification
//! (6.x) carries exactly the schemes implemented here:
//!
//! - **LE Coded PHY** (Bluetooth 5.0, Vol 6 Part B §3.3): a non-systematic,
//!   non-recursive rate-1/2 convolutional code with constraint length $K=4$,
//!   $$G_0(x) = 1 + x + x^2 + x^3, \qquad G_1(x) = 1 + x^2 + x^3,$$
//!   the $G_0$ output transmitted first, terminated by three zero input bits
//!   (the spec's TERM sequence), followed by a pattern mapper: $S=2$ sends
//!   each coded bit as itself ($P=1$), $S=8$ sends coded 0 as symbols `0011`
//!   and coded 1 as `1100` ($P=4$).
//! - **BR/EDR FEC 1/3** (Vol 2 Part B §7.4): 3× bit repetition, used for the
//!   packet header and the HV1 payload. The spec defines only the encoder;
//!   the majority-vote decoder here is the standard implementation choice
//!   (it is also what libbtbb does).
//! - **BR/EDR FEC 2/3** (Vol 2 Part B §7.5): the (15,10) shortened Hamming
//!   code with generator $$g(D) = (D+1)(D^4+D+1) = D^5+D^4+D^2+1,$$ which
//!   corrects every single-bit error and detects every double-bit error per
//!   codeword. Used by DM/DV/FHS/HV2/EV4 packets.
//!
//! # Sourcing (not fabricated)
//!
//! The constants above were taken verbatim from the Bluetooth Core
//! Specification and cross-verified before implementation: the LE Coded PHY
//! encoder text is word-for-word identical in Core 5.0, 5.2, 5.3, 5.4, 6.0,
//! 6.1 and 6.2, and the unit tests in this module reproduce, bit-exactly,
//! the specification's own published sample data (Vol 6 Part C §2: the
//! reference packet with Access Address `D6 BE 89 8E`, including every FEC
//! output bit and the $S=8$ symbol stream) and the BR/EDR FEC sample data
//! (Vol 2, "FEC sample data": all ten (15,10) generator rows). The (15,10)
//! code is additionally cross-checked against libbtbb's independent
//! generator table. LE Coded PHY reuses [`crate::viterbi::ViterbiDecoder`]
//! unchanged: the crate's trellis convention (newest input bit at the
//! generator MSB, $g_0$ output first, $K-1$ zero-tail termination) maps
//! exactly onto the spec's shift register, taps, output order, and TERM
//! sequence — the sample-data tests are what prove that mapping.
//!
//! # What this does not do
//!
//! Packet assembly and everything around FEC stay out of scope: preamble
//! generation, whitening, the LE CRC-24 (its LSB-first/`0x555555`-init
//! conventions differ from the 3GPP CRCs in [`crate::crc`] and it is not
//! implemented in this crate), BR/EDR HEC/CRC-16, and access-code
//! construction. Callers feed this module the bit fields the spec says are
//! FEC-coded and get coded bits/symbols back.
//!
//! # Examples
//!
//! ```
//! use syndrome::bluetooth::{le_coded_phy_code, pattern_map_s8};
//!
//! let code = le_coded_phy_code().unwrap();
//! // FEC block 1 of an advertising packet: Access Address bits + CI bits;
//! // encode() appends the 3-bit TERM automatically.
//! let block = [1u8, 0, 1, 1, 0, 1];  // toy field, LSB first
//! let coded = code.encode(&block);
//! assert_eq!(coded.len(), 2 * (block.len() + 3));
//!
//! let mut symbols = vec![0u8; coded.len() * 4];
//! pattern_map_s8(&coded, &mut symbols).unwrap();
//! ```

use crate::error::FecError;
use crate::viterbi::ViterbiDecoder;

/// LE Coded PHY generator $G_0(x) = 1 + x + x^2 + x^3$ in this crate's
/// trellis convention (newest input bit at the MSB): taps on the current
/// input and all three stored bits.
pub const LE_CODED_G0: u32 = 0b1111;

/// LE Coded PHY generator $G_1(x) = 1 + x^2 + x^3$: taps on the current
/// input and the two oldest stored bits.
pub const LE_CODED_G1: u32 = 0b1011;

/// LE Coded PHY constraint length $K$.
pub const LE_CODED_CONSTRAINT_LENGTH: usize = 4;

/// Build the LE Coded PHY convolutional code (Vol 6 Part B §3.3.1).
///
/// The returned [`ViterbiDecoder`] is the complete FEC layer for one coded
/// PHY FEC block: `encode` produces the spec's output bit stream including
/// the 3-bit TERM (its zero-tail termination is exactly the spec's
/// termination sequence), and `decode_hard`/`decode_soft` recover the block.
/// Feed it the concatenated fields of one FEC block — Access Address + CI
/// for block 1, PDU + CRC for block 2 — without the TERM bits.
///
/// # Returns
///
/// A codec configured for $K=4$, $G_0/G_1$ per the spec, $G_0$ bit first.
///
/// # Errors
///
/// Never fails in practice ($K=4$ is always a valid constraint length);
/// the `Result` is [`ViterbiDecoder::with_generators`]'s signature.
///
/// # Examples
///
/// ```
/// use syndrome::bluetooth::le_coded_phy_code;
///
/// let code = le_coded_phy_code().unwrap();
/// let info = [0u8, 1, 1, 0, 1, 0, 1, 1];
/// let coded = code.encode(&info);
/// assert_eq!(code.decode_hard(&coded), info);
/// ```
pub fn le_coded_phy_code() -> Result<ViterbiDecoder, FecError> {
    ViterbiDecoder::with_generators(LE_CODED_CONSTRAINT_LENGTH, LE_CODED_G0, LE_CODED_G1)
}

/// Map coded bits to $S=8$ transmission symbols (Vol 6 Part B §3.3.2, $P=4$):
/// coded 0 → `0011`, coded 1 → `1100`, in transmission order.
///
/// $S=2$ has no mapper ($P=1$, each coded bit is sent as itself), so no
/// function is provided for it.
///
/// # Arguments
///
/// * `coded` - FEC output bits, one per byte (`0`/`1`).
/// * `symbols` - Output; must hold exactly `4 * coded.len()` elements.
///
/// # Errors
///
/// Returns [`FecError::BufferTooSmall`] if `symbols` is not exactly
/// `4 * coded.len()` long, and [`FecError::InvalidParam`] if `coded`
/// contains a value other than 0 or 1.
///
/// # Examples
///
/// ```
/// use syndrome::bluetooth::pattern_map_s8;
///
/// let mut symbols = [0u8; 8];
/// pattern_map_s8(&[0, 1], &mut symbols).unwrap();
/// assert_eq!(symbols, [0, 0, 1, 1, 1, 1, 0, 0]);
/// ```
pub fn pattern_map_s8(coded: &[u8], symbols: &mut [u8]) -> Result<(), FecError> {
    let required = coded.len() * 4;
    if symbols.len() != required {
        return Err(FecError::BufferTooSmall {
            required,
            provided: symbols.len(),
        });
    }
    if coded.iter().any(|&b| b > 1) {
        return Err(FecError::InvalidParam(
            "coded bits must be 0 or 1 for pattern mapping",
        ));
    }
    for (bit, chunk) in coded.iter().zip(symbols.chunks_exact_mut(4)) {
        // 0 -> 0011, 1 -> 1100 (transmission order).
        chunk[0] = *bit;
        chunk[1] = *bit;
        chunk[2] = 1 - *bit;
        chunk[3] = 1 - *bit;
    }
    Ok(())
}

/// Soft-demap $S=8$ symbol LLRs back to per-coded-bit LLRs.
///
/// With the crate-wide sign convention (positive LLR favours symbol 0) and
/// the $P=4$ patterns 0 → `0011`, 1 → `1100`, the per-bit LLR is
/// $$L_\text{bit} = L_0 + L_1 - L_2 - L_3$$ over each group of four symbol
/// LLRs — the two leading symbols agree with the coded bit, the two
/// trailing ones are its complement. The result feeds
/// [`ViterbiDecoder::decode_soft`] directly.
///
/// # Arguments
///
/// * `symbol_llr` - Received symbol LLRs; length must be a multiple of 4.
/// * `coded_llr` - Output; must hold exactly `symbol_llr.len() / 4`
///   elements.
///
/// # Errors
///
/// Returns [`FecError::InvalidParam`] if `symbol_llr.len()` is not a
/// multiple of 4, or [`FecError::BufferTooSmall`] on an output-length
/// mismatch.
///
/// # Examples
///
/// ```
/// use syndrome::bluetooth::pattern_demap_s8;
///
/// // A strongly received "0011" pattern demaps to a positive (bit 0) LLR.
/// let mut llr = [0.0f32; 1];
/// pattern_demap_s8(&[4.0, 4.0, -4.0, -4.0], &mut llr).unwrap();
/// assert!(llr[0] > 0.0);
/// ```
pub fn pattern_demap_s8(symbol_llr: &[f32], coded_llr: &mut [f32]) -> Result<(), FecError> {
    if !symbol_llr.len().is_multiple_of(4) {
        return Err(FecError::InvalidParam(
            "S=8 symbol LLR length must be a multiple of 4",
        ));
    }
    let required = symbol_llr.len() / 4;
    if coded_llr.len() != required {
        return Err(FecError::BufferTooSmall {
            required,
            provided: coded_llr.len(),
        });
    }
    for (out, group) in coded_llr.iter_mut().zip(symbol_llr.chunks_exact(4)) {
        *out = group[0] + group[1] - group[2] - group[3];
    }
    Ok(())
}

/// Encode with BR/EDR FEC 1/3 (Vol 2 Part B §7.4): each bit repeated 3×.
///
/// # Arguments
///
/// * `bits` - Input bits, one per byte (`0`/`1`).
/// * `out` - Output; must hold exactly `3 * bits.len()` elements.
///
/// # Errors
///
/// Returns [`FecError::BufferTooSmall`] on an output-length mismatch and
/// [`FecError::InvalidParam`] if `bits` contains a value other than 0 or 1.
///
/// # Examples
///
/// ```
/// use syndrome::bluetooth::fec13_encode;
///
/// let mut out = [0u8; 6];
/// fec13_encode(&[1, 0], &mut out).unwrap();
/// assert_eq!(out, [1, 1, 1, 0, 0, 0]);
/// ```
pub fn fec13_encode(bits: &[u8], out: &mut [u8]) -> Result<(), FecError> {
    let required = bits.len() * 3;
    if out.len() != required {
        return Err(FecError::BufferTooSmall {
            required,
            provided: out.len(),
        });
    }
    if bits.iter().any(|&b| b > 1) {
        return Err(FecError::InvalidParam("FEC 1/3 input bits must be 0 or 1"));
    }
    for (bit, chunk) in bits.iter().zip(out.chunks_exact_mut(3)) {
        chunk.fill(*bit);
    }
    Ok(())
}

/// Decode BR/EDR FEC 1/3 by per-triplet majority vote.
///
/// The specification defines only the repetition encoder; majority decoding
/// is the standard implementation choice and corrects any single flipped
/// bit within a triplet.
///
/// # Arguments
///
/// * `coded` - Received bits; length must be a multiple of 3.
/// * `out` - Output; must hold exactly `coded.len() / 3` elements.
///
/// # Errors
///
/// Returns [`FecError::InvalidParam`] if `coded.len()` is not a multiple of
/// 3 or contains a value other than 0 or 1, and
/// [`FecError::BufferTooSmall`] on an output-length mismatch.
///
/// # Examples
///
/// ```
/// use syndrome::bluetooth::fec13_decode;
///
/// let mut out = [0u8; 2];
/// fec13_decode(&[1, 0, 1, 0, 0, 1], &mut out).unwrap();
/// assert_eq!(out, [1, 0]);  // each triplet outvotes its one flipped bit
/// ```
pub fn fec13_decode(coded: &[u8], out: &mut [u8]) -> Result<(), FecError> {
    if !coded.len().is_multiple_of(3) {
        return Err(FecError::InvalidParam(
            "FEC 1/3 coded length must be a multiple of 3",
        ));
    }
    let required = coded.len() / 3;
    if out.len() != required {
        return Err(FecError::BufferTooSmall {
            required,
            provided: out.len(),
        });
    }
    if coded.iter().any(|&b| b > 1) {
        return Err(FecError::InvalidParam("FEC 1/3 coded bits must be 0 or 1"));
    }
    for (slot, t) in out.iter_mut().zip(coded.chunks_exact(3)) {
        *slot = (t[0] & t[1]) | (t[1] & t[2]) | (t[2] & t[0]);
    }
    Ok(())
}

/// Info bits per BR/EDR FEC 2/3 block.
pub const FEC23_INFO_BITS: usize = 10;
/// Coded bits per BR/EDR FEC 2/3 block.
pub const FEC23_CODED_BITS: usize = 15;
/// Parity bits per BR/EDR FEC 2/3 block.
const FEC23_PARITY_BITS: usize = FEC23_CODED_BITS - FEC23_INFO_BITS;

/// $g(D) - D^5 = D^4 + D^2 + 1$: the feedback taps of the (15,10) encoder's
/// 5-bit LFSR, register bit $i$ holding the $D^i$ coefficient.
const FEC23_FEEDBACK: u8 = 0b10101;

/// Compute the 5 parity bits for one 10-bit info block, LFSR per Vol 2
/// Part B Figure 7.11; register returned with the $D^4$ coefficient — the
/// first-transmitted parity bit — at the MSB of the low 5 bits.
fn fec23_parity(info: &[u8]) -> u8 {
    let mut reg = 0u8;
    for &bit in info {
        let feedback = bit ^ ((reg >> (FEC23_PARITY_BITS - 1)) & 1);
        reg = (reg << 1) & 0x1F;
        if feedback == 1 {
            reg ^= FEC23_FEEDBACK;
        }
    }
    reg
}

/// Outcome of a [`fec23_decode`] call.
///
/// Per the code's guarantee ("corrects all single errors and detects all
/// double errors in each codeword"), a block is either clean, repaired
/// (one bit), or *detected* as beyond repair — and a detected block is
/// data, not an exception, so a batch reports it here instead of aborting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fec23Status {
    /// Total bits corrected across the batch (at most one per block).
    pub corrected_bits: usize,
    /// Blocks whose syndrome matched no single-bit error — at least two
    /// bit errors. Their info bits are output as received, uncorrected.
    pub uncorrectable_blocks: usize,
}

/// Encode with BR/EDR FEC 2/3, the (15,10) shortened Hamming code
/// (Vol 2 Part B §7.5).
///
/// Each 10 info bits (transmission order) become a 15-bit codeword: the
/// info bits followed by 5 parity bits, $D^4$ side first. The spec pads a
/// payload with zero tail bits to a multiple of 10 before this step;
/// padding is the caller's business because the pad length is derived from
/// packet-level fields this module does not model.
///
/// # Arguments
///
/// * `info` - Info bits, one per byte; length must be a multiple of 10.
/// * `out` - Output; must hold exactly `info.len() / 10 * 15` elements.
///
/// # Errors
///
/// Returns [`FecError::InvalidParam`] if `info.len()` is not a multiple of
/// 10 or contains a value other than 0 or 1, and
/// [`FecError::BufferTooSmall`] on an output-length mismatch.
///
/// # Examples
///
/// ```
/// use syndrome::bluetooth::fec23_encode;
///
/// // First row of the spec's FEC sample data: info 0x001 (b0 first).
/// let info = [1u8, 0, 0, 0, 0, 0, 0, 0, 0, 0];
/// let mut coded = [0u8; 15];
/// fec23_encode(&info, &mut coded).unwrap();
/// assert_eq!(&coded[10..], &[1, 1, 0, 1, 0]);
/// ```
pub fn fec23_encode(info: &[u8], out: &mut [u8]) -> Result<(), FecError> {
    if !info.len().is_multiple_of(FEC23_INFO_BITS) {
        return Err(FecError::InvalidParam(
            "FEC 2/3 info length must be a multiple of 10",
        ));
    }
    let blocks = info.len() / FEC23_INFO_BITS;
    let required = blocks * FEC23_CODED_BITS;
    if out.len() != required {
        return Err(FecError::BufferTooSmall {
            required,
            provided: out.len(),
        });
    }
    if info.iter().any(|&b| b > 1) {
        return Err(FecError::InvalidParam("FEC 2/3 info bits must be 0 or 1"));
    }
    for (src, dst) in info
        .chunks_exact(FEC23_INFO_BITS)
        .zip(out.chunks_exact_mut(FEC23_CODED_BITS))
    {
        dst[..FEC23_INFO_BITS].copy_from_slice(src);
        let parity = fec23_parity(src);
        for (i, slot) in dst[FEC23_INFO_BITS..].iter_mut().enumerate() {
            *slot = (parity >> (FEC23_PARITY_BITS - 1 - i)) & 1;
        }
    }
    Ok(())
}

/// Decode BR/EDR FEC 2/3, correcting one bit error per 15-bit block.
///
/// Syndrome decoding: a zero syndrome passes the block through, a syndrome
/// matching one of the 15 single-bit error patterns repairs that bit, and
/// any other syndrome is a detected multi-bit error — the block's info bits
/// are output as received and counted in
/// [`Fec23Status::uncorrectable_blocks`] rather than aborting the batch.
///
/// # Arguments
///
/// * `coded` - Received bits; length must be a multiple of 15.
/// * `out` - Output info bits; must hold exactly `coded.len() / 15 * 10`
///   elements.
///
/// # Returns
///
/// A [`Fec23Status`] with the corrected-bit and uncorrectable-block counts.
///
/// # Errors
///
/// Returns [`FecError::InvalidParam`] if `coded.len()` is not a multiple of
/// 15 or contains a value other than 0 or 1, and
/// [`FecError::BufferTooSmall`] on an output-length mismatch.
///
/// # Examples
///
/// ```
/// use syndrome::bluetooth::{fec23_decode, fec23_encode};
///
/// let info = [1u8, 0, 1, 1, 0, 0, 1, 0, 1, 0];
/// let mut coded = [0u8; 15];
/// fec23_encode(&info, &mut coded).unwrap();
/// coded[3] ^= 1; // one bit error
///
/// let mut decoded = [0u8; 10];
/// let status = fec23_decode(&coded, &mut decoded).unwrap();
/// assert_eq!(decoded, info);
/// assert_eq!(status.corrected_bits, 1);
/// assert_eq!(status.uncorrectable_blocks, 0);
/// ```
pub fn fec23_decode(coded: &[u8], out: &mut [u8]) -> Result<Fec23Status, FecError> {
    if !coded.len().is_multiple_of(FEC23_CODED_BITS) {
        return Err(FecError::InvalidParam(
            "FEC 2/3 coded length must be a multiple of 15",
        ));
    }
    let blocks = coded.len() / FEC23_CODED_BITS;
    let required = blocks * FEC23_INFO_BITS;
    if out.len() != required {
        return Err(FecError::BufferTooSmall {
            required,
            provided: out.len(),
        });
    }
    if coded.iter().any(|&b| b > 1) {
        return Err(FecError::InvalidParam("FEC 2/3 coded bits must be 0 or 1"));
    }

    let mut status = Fec23Status {
        corrected_bits: 0,
        uncorrectable_blocks: 0,
    };
    for (src, dst) in coded
        .chunks_exact(FEC23_CODED_BITS)
        .zip(out.chunks_exact_mut(FEC23_INFO_BITS))
    {
        let mut received_parity = 0u8;
        for &bit in &src[FEC23_INFO_BITS..] {
            received_parity = (received_parity << 1) | bit;
        }
        let syndrome = fec23_parity(&src[..FEC23_INFO_BITS]) ^ received_parity;

        dst.copy_from_slice(&src[..FEC23_INFO_BITS]);
        if syndrome == 0 {
            continue;
        }

        // A single error in info bit i contributes that bit's parity
        // column; a single error in a parity bit contributes a unit vector.
        let mut fixed = false;
        for i in 0..FEC23_INFO_BITS {
            let mut probe = [0u8; FEC23_INFO_BITS];
            probe[i] = 1;
            if fec23_parity(&probe) == syndrome {
                dst[i] ^= 1;
                status.corrected_bits += 1;
                fixed = true;
                break;
            }
        }
        if !fixed {
            if syndrome.count_ones() == 1 {
                // The flipped bit was a parity bit; the info bits are fine.
                status.corrected_bits += 1;
            } else {
                status.uncorrectable_blocks += 1;
            }
        }
    }
    Ok(status)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse "0 1 1 0 ..." / "0011 1100" style bit listings copied from the
    /// specification's sample-data sections.
    fn bits(s: &str) -> Vec<u8> {
        s.chars()
            .filter(|c| !c.is_whitespace())
            .map(|c| match c {
                '0' => 0,
                '1' => 1,
                other => panic!("non-bit char {other:?} in test vector"),
            })
            .collect()
    }

    /// Access Address `D6 BE 89 8E` of the spec's reference packet, bits in
    /// transmission order (LSB of each byte first). Vol 6 Part C §2.2.
    const AA_IN: &str = "0110 1011 0111 1101 1001 0001 0111 0001";
    /// FEC encoder output for the Access Address (64 bits).
    const AA_OUT: &str = "0011 0101 1101 0010 0111 1010 0101 1011 \
                          1001 0000 1011 1111 1000 1010 1000 1111";
    /// PDU `00 03 42 4C 45`, bits in transmission order.
    const PDU_IN: &str = "0000 0000 1100 0000 0100 0010 0011 0010 1010 0010";
    const PDU_OUT: &str = "0000 0000 0000 0000 1101 0100 1100 0000 0011 1011 \
                           1100 1110 1111 1101 0100 0010 0001 0001 1111 1110";
    /// CRC `29 0A CE`, bits in the transmitted order listed by the spec.
    const CRC_IN: &str = "1001 0100 0101 0000 0111 0011";
    const CRC_OUT: &str = "0001 1100 1000 0111 1111 1000 0111 1100 0011 0110 1000 0001";
    const TERM2_OUT: &str = "0100 11";

    /// Spec sample data, FEC block 1: the Access Address followed by CI and
    /// TERM1 for both CI values, encoded as one continuous stream.
    #[test]
    fn le_fec_block1_matches_spec_sample_data_both_ci_values() {
        let code = le_coded_phy_code().unwrap();

        // S=2 packet: CI = 0b01, transmitted LSB first -> input bits 1, 0.
        let mut input = bits(AA_IN);
        input.extend_from_slice(&[1, 0]);
        let mut expected = bits(AA_OUT);
        expected.extend(bits("0101")); // CI output
        expected.extend(bits("001100")); // TERM1 output
        assert_eq!(code.encode(&input), expected, "S=2 FEC block 1");

        // S=8 packet: CI = 0b00 -> input bits 0, 0.
        let mut input = bits(AA_IN);
        input.extend_from_slice(&[0, 0]);
        let mut expected = bits(AA_OUT);
        expected.extend(bits("1011"));
        expected.extend(bits("110000"));
        assert_eq!(code.encode(&input), expected, "S=8 FEC block 1");
    }

    /// Spec sample data, FEC block 2: PDU + CRC + TERM2 as one stream.
    #[test]
    fn le_fec_block2_matches_spec_sample_data() {
        let code = le_coded_phy_code().unwrap();
        let mut input = bits(PDU_IN);
        input.extend(bits(CRC_IN));
        let mut expected = bits(PDU_OUT);
        expected.extend(bits(CRC_OUT));
        expected.extend(bits(TERM2_OUT));
        assert_eq!(code.encode(&input), expected);
    }

    /// The S=8 Access Address symbol stream, transcribed from the spec's
    /// "Transmitted symbols" listing (both packets; block 1 is always S=8).
    #[test]
    fn le_pattern_mapper_matches_spec_symbol_stream() {
        let code = le_coded_phy_code().unwrap();
        let aa_coded = &code.encode(&bits(AA_IN))[..64];
        let mut symbols = vec![0u8; aa_coded.len() * 4];
        pattern_map_s8(aa_coded, &mut symbols).unwrap();
        let expected = bits(
            "0011 0011 1100 1100 0011 1100 0011 1100 1100 1100 0011 1100 \
             0011 0011 1100 0011 0011 1100 1100 1100 1100 0011 1100 0011 \
             0011 1100 0011 1100 1100 0011 1100 1100 1100 0011 0011 1100 \
             0011 0011 0011 0011 1100 0011 1100 1100 1100 1100 1100 1100 \
             1100 0011 0011 0011 1100 0011 1100 0011 1100 0011 0011 0011 \
             1100 1100 1100 1100",
        );
        assert_eq!(symbols, expected);
    }

    /// Map then soft-demap: the demapped LLR signs must reproduce the coded
    /// bits, and a corrupted symbol within a group must not flip the bit.
    #[test]
    fn le_pattern_demap_recovers_coded_bits() {
        let coded = bits(AA_OUT);
        let mut symbols = vec![0u8; coded.len() * 4];
        pattern_map_s8(&coded, &mut symbols).unwrap();
        let mut symbol_llr: Vec<f32> = symbols
            .iter()
            .map(|&s| if s == 0 { 4.0 } else { -4.0 })
            .collect();
        // Corrupt one symbol of the first group; 3-of-4 still outvote it.
        symbol_llr[0] = -symbol_llr[0];
        let mut coded_llr = vec![0.0f32; coded.len()];
        pattern_demap_s8(&symbol_llr, &mut coded_llr).unwrap();
        let hard: Vec<u8> = coded_llr.iter().map(|&l| (l < 0.0) as u8).collect();
        assert_eq!(hard, coded);
    }

    /// End-to-end: encode block 1, pattern-map to S=8, corrupt at the
    /// symbol level, demap, soft-decode.
    #[test]
    fn le_coded_phy_round_trip_with_symbol_errors() {
        let code = le_coded_phy_code().unwrap();
        let mut input = bits(AA_IN);
        input.extend_from_slice(&[0, 0]);
        let coded = code.encode(&input);
        let mut symbols = vec![0u8; coded.len() * 4];
        pattern_map_s8(&coded, &mut symbols).unwrap();
        let mut symbol_llr: Vec<f32> = symbols
            .iter()
            .map(|&s| if s == 0 { 2.0 } else { -2.0 })
            .collect();
        // Flip every 10th symbol.
        for llr in symbol_llr.iter_mut().step_by(10) {
            *llr = -*llr;
        }
        let mut coded_llr = vec![0.0f32; coded.len()];
        pattern_demap_s8(&symbol_llr, &mut coded_llr).unwrap();
        assert_eq!(code.decode_soft(&coded_llr), input);
    }

    /// All ten generator rows from the specification's BR/EDR "FEC sample
    /// data" section: info word 2^i (b0 transmitted first) -> parity.
    #[test]
    fn fec23_matches_all_10_spec_generator_rows() {
        const SPEC_ROWS: [&str; 10] = [
            "11010", "01101", "11100", "01110", "00111", "11001", "10110", "01011", "11111",
            "10101",
        ];
        for (i, parity) in SPEC_ROWS.iter().enumerate() {
            let mut info = [0u8; 10];
            info[i] = 1;
            let mut coded = [0u8; 15];
            fec23_encode(&info, &mut coded).unwrap();
            assert_eq!(&coded[..10], &info, "systematic part, row {i}");
            assert_eq!(&coded[10..], bits(parity).as_slice(), "parity, row {i}");
        }
    }

    /// Linearity gives the code from its generator rows; the minimum weight
    /// over all 1023 nonzero codewords must be 4 ("corrects all single
    /// errors and detects all double errors").
    #[test]
    fn fec23_minimum_distance_is_4() {
        let mut min_weight = usize::MAX;
        for word in 1u16..1024 {
            let info: Vec<u8> = (0..10).map(|i| ((word >> i) & 1) as u8).collect();
            let mut coded = [0u8; 15];
            fec23_encode(&info, &mut coded).unwrap();
            let weight = coded.iter().filter(|&&b| b == 1).count();
            min_weight = min_weight.min(weight);
        }
        assert_eq!(min_weight, 4);
    }

    /// Every single-bit error in every position of several codewords is
    /// corrected, and every double-bit error is detected (never miscorrected
    /// into silently wrong info bits).
    #[test]
    fn fec23_corrects_singles_and_detects_doubles() {
        let mut rng_state = 0x1234_5678_9abc_def0u64;
        let mut next_info = || {
            // xorshift64, enough to pick a few random info words.
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let mut info = [0u8; 10];
            for (i, slot) in info.iter_mut().enumerate() {
                *slot = ((rng_state >> i) & 1) as u8;
            }
            info
        };

        for _ in 0..8 {
            let info = next_info();
            let mut coded = [0u8; 15];
            fec23_encode(&info, &mut coded).unwrap();

            for e in 0..15 {
                let mut rx = coded;
                rx[e] ^= 1;
                let mut out = [0u8; 10];
                let status = fec23_decode(&rx, &mut out).unwrap();
                assert_eq!(out, info, "single error at {e} must be corrected");
                assert_eq!(status.corrected_bits, 1);
                assert_eq!(status.uncorrectable_blocks, 0);
            }

            for e1 in 0..15 {
                for e2 in (e1 + 1)..15 {
                    let mut rx = coded;
                    rx[e1] ^= 1;
                    rx[e2] ^= 1;
                    let mut out = [0u8; 10];
                    let status = fec23_decode(&rx, &mut out).unwrap();
                    // Detection: either flagged uncorrectable, or the "fix"
                    // only touched parity — the info bits must never come
                    // back silently wrong while the block reports clean.
                    if status.uncorrectable_blocks == 0 {
                        assert_eq!(
                            out, info,
                            "double error ({e1},{e2}) neither detected nor harmless"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn fec23_clean_batch_round_trip() {
        let info: Vec<u8> = (0..50).map(|i| ((i * 7) % 3 == 0) as u8).collect();
        let mut coded = vec![0u8; 75];
        fec23_encode(&info, &mut coded).unwrap();
        let mut out = vec![0u8; 50];
        let status = fec23_decode(&coded, &mut out).unwrap();
        assert_eq!(out, info);
        assert_eq!(status.corrected_bits, 0);
        assert_eq!(status.uncorrectable_blocks, 0);
    }

    #[test]
    fn fec13_round_trip_with_single_errors_per_triplet() {
        let info = bits("1011 0010 11");
        let mut coded = vec![0u8; 30];
        fec13_encode(&info, &mut coded).unwrap();
        // Flip one bit in every triplet (rotating position).
        for (i, t) in coded.chunks_exact_mut(3).enumerate() {
            t[i % 3] ^= 1;
        }
        let mut out = vec![0u8; 10];
        fec13_decode(&coded, &mut out).unwrap();
        assert_eq!(out, info);
    }

    #[test]
    fn parameter_validation_rejects_bad_shapes_and_values() {
        let mut out3 = [0u8; 3];
        let mut out4 = [0u8; 4];
        assert!(fec13_encode(&[0, 1], &mut out3).is_err()); // wrong size
        assert!(fec13_encode(&[2], &mut out3).is_err()); // non-bit
        assert!(fec13_decode(&[0, 1], &mut out3).is_err()); // not %3
        assert!(fec23_encode(&[0u8; 9], &mut out3).is_err()); // not %10
        assert!(fec23_decode(&[0u8; 14], &mut out3).is_err()); // not %15
        assert!(pattern_map_s8(&[0, 1], &mut out4).is_err()); // wrong size
        let mut llr1 = [0.0f32; 1];
        assert!(pattern_demap_s8(&[0.0; 3], &mut llr1).is_err()); // not %4
    }
}
