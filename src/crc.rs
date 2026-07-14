//! CRC computation for 3GPP TS 38.212 §5.1.
//!
//! Implements byte-wise CRC tables for all generator polynomials used in 5G NR:
//!
//! | Kind   | Length | Use (TS 38.212)                       | Polynomial (MSB-first hex) |
//! |--------|--------|---------------------------------------|----------------------------|
//! | CRC24A | 24     | Transport block (DL-SCH/UL-SCH)       | `0x864CFB`                 |
//! | CRC24B | 24     | Code block (segmented TBs)            | `0x800063`                 |
//! | CRC24C | 24     | UCI (uplink control information)      | `0xB2B117`                 |
//! | CRC16  | 16     | UCI (small payloads)                  | `0x11021`                  |
//! | CRC11  | 11     | DCI / DL control                      | `0xE21`                    |
//! | CRC6   | 6      | UCI (very small payloads)             | `0x61`                     |
//!
//! Bits are processed MSB-first over a bit-string represented as a `&[u8]` of
//! 0/1 values, matching the 3GPP bit-string convention.
//!
//! # How it works, in plain English
//!
//! A CRC is a remainder from long division of the message by a fixed
//! polynomial $g(x)$ over $GF(2)$ (i.e. binary coefficients, addition = XOR).
//! The classic hardware way to compute that remainder is a Linear Feedback
//! Shift Register (LFSR): an `L`-bit register that shifts one input bit in
//! at a time, XORing in the generator polynomial whenever the bit shifted
//! *out* of the top was a `1`:
//!
//! ```text
//!  input bit
//!      │
//!      ▼
//!  ┌───┴────────────────────────────┐
//!  │  reg[L-1]  reg[L-2] ... reg[0] │  ◀── shift left each step
//!  └────┬───────────────────────────┘
//!       │ (bit shifted out, i.e. old reg[L-1] XOR input)
//!       ▼
//!   feedback ──▶ XOR into every tap position where g(x) has a `1` bit
//! ```
//!
//! `bit_serial_step` (a private helper below) is exactly this: one
//! shift-and-conditionally-XOR step.
//! Running it once per input bit (see `compute_bit_serial`, the
//! test-only reference) is correct but slow -- 1 bit of work per bit of
//! input. [`Crc24::compute`] instead precomputes, once per [`Crc24::new`],
//! what 8 (or 4, for `Crc6`) consecutive `bit_serial_step` calls do to the
//! register for every possible input byte (or nibble) -- a 256-entry (or
//! 16-entry) lookup table -- so encoding processes a whole byte per table
//! lookup instead of 8 individual bit shifts. The two paths are equivalent
//! *by construction*, and additionally cross-checked against each other over
//! randomized inputs by `table_matches_bit_serial_reference` below.
//!
//! # Examples
//!
//! ```
//! use syndrome::crc::{Crc24, CrcKind};
//!
//! let crc = Crc24::new(CrcKind::Crc24A);
//! let mut bits: Vec<u8> = vec![1, 0, 1, 1, 0, 0, 1, 0];
//! let remainder = crc.compute(&bits);
//! crc.attach(&mut bits);
//! assert!(crc.check(&bits));
//! ```

/// Identifies which CRC polynomial to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrcKind {
    /// 24-bit CRC-A, generator $g_{CRC24A}$, used on transport blocks.
    Crc24A,
    /// 24-bit CRC-B, generator $g_{CRC24B}$, used on code blocks.
    Crc24B,
    /// 24-bit CRC-C, generator $g_{CRC24C}$, used on UCI.
    Crc24C,
    /// 16-bit CRC, used on small UCI payloads.
    Crc16,
    /// 11-bit CRC, used on DL control (DCI).
    Crc11,
    /// 6-bit CRC, used on very small UCI payloads.
    Crc6,
}

impl CrcKind {
    /// Generator polynomial as a u32 (MSB-first, bit `L` implicit).
    ///
    /// E.g. CRC24A: $g(D) = D^{24} + D^{23} + D^{18} + \ldots + 1$ → `0x864CFB`.
    pub const fn poly(self) -> u32 {
        match self {
            CrcKind::Crc24A => 0x864CFB,
            CrcKind::Crc24B => 0x800063,
            CrcKind::Crc24C => 0xB2B117,
            CrcKind::Crc16 => 0x11021,
            CrcKind::Crc11 => 0xE21,
            CrcKind::Crc6 => 0x61,
        }
    }

    /// Number of CRC parity bits $L$.
    pub const fn length(self) -> usize {
        match self {
            CrcKind::Crc24A | CrcKind::Crc24B | CrcKind::Crc24C => 24,
            CrcKind::Crc16 => 16,
            CrcKind::Crc11 => 11,
            CrcKind::Crc6 => 6,
        }
    }
}

/// Single-bit LFSR step: the mathematical definition of the CRC recurrence.
///
/// Given the current register `reg` (only the low `len` bits are
/// meaningful), folds in one more input `bit` (0 or 1):
///
/// $$ \text{feedback} = (\text{reg} \gg (\text{len}-1)) \oplus \text{bit} $$
/// $$ \text{reg}' = ((\text{reg} \ll 1) \bmod 2^{\text{len}}) \oplus
///    (\text{feedback} \cdot \text{poly}) $$
///
/// Every table entry used by [`Crc24::compute`] is built by iterating this
/// exact function, so the table-driven fast path is equivalent to the
/// bit-serial path *by construction*.
#[inline]
const fn bit_serial_step(reg: u32, bit: u8, len: usize, poly: u32, mask: u32) -> u32 {
    let feedback = ((reg >> (len - 1)) ^ (bit as u32 & 1)) & 1;
    let shifted = (reg << 1) & mask;
    if feedback != 0 {
        shifted ^ poly
    } else {
        shifted
    }
}

/// Reference bit-serial LFSR CRC over a full bit-string (historical
/// implementation, 1 bit per iteration).
///
/// Kept as a private reference so the equivalence tests below can check the
/// table-driven [`Crc24::compute`] against it directly, bit for bit, over
/// randomized inputs. [`bit_serial_step`] (the per-bit primitive this is
/// built from) is also used directly by production code: to derive the
/// tables in [`Crc24::new`], and to process the tail bits that don't fill a
/// whole table chunk in [`Crc24::compute`].
#[cfg(test)]
fn compute_bit_serial(bits: &[u8], len: usize, poly: u32, mask: u32) -> u32 {
    let mut reg = 0u32;
    for &bit in bits {
        reg = bit_serial_step(reg, bit, len, poly, mask);
    }
    reg
}

/// CRC engine built around one of the 3GPP generator polynomials.
///
/// Runs a byte-wise (or, for `Crc6`, nibble-wise) table-driven LFSR so that
/// [`Crc24::compute`] advances 8 (or 4) input bit-entries per step instead
/// of 1. CRC sits inside the 5G transport-block hot path (every TB and code
/// block is CRC-checked), so this table-driven form matters for throughput.
///
/// The tables are derived directly from the bit-serial recurrence
/// (see `bit_serial_step`), so the two paths are equivalent by
/// construction, and are additionally checked against each other by the
/// `table_matches_bit_serial_reference` proptest-style test.
pub struct Crc24 {
    kind: CrcKind,
    /// Byte-wise (8-bit) lookup table, used when `kind.length() >= 8`
    /// (`Crc24A/B/C`, `Crc16`, `Crc11`). Entry `i` is the register value
    /// obtained by feeding the 8 bits of `i` (MSB-first) into
    /// [`bit_serial_step`] starting from a zero register.
    ///
    /// Unused (left zeroed) for `Crc6`.
    table: [u32; 256],
    /// Nibble-wise (4-bit) lookup table, used only for `Crc6`
    /// (`length() == 6`). An 8-bit table would require shifting the
    /// 6-bit register right by `len - 8 = -2` bits, which is not
    /// representable, so `Crc6` processes 4 bits at a time instead using
    /// this 16-entry table (same construction as `table`, but over 4-bit
    /// chunks).
    ///
    /// Unused (left zeroed) for all other kinds.
    nibble_table: [u32; 16],
}

impl Crc24 {
    /// Construct a CRC engine for the given polynomial kind.
    ///
    /// Precomputes the byte-wise (or, for `Crc6`, nibble-wise) lookup table
    /// used by [`Crc24::compute`].
    ///
    /// # Arguments
    ///
    /// * `kind` - Which 3GPP CRC polynomial to use.
    ///
    /// # Examples
    ///
    /// ```
    /// use syndrome::crc::{Crc24, CrcKind};
    /// let crc = Crc24::new(CrcKind::Crc24A);
    /// ```
    pub fn new(kind: CrcKind) -> Self {
        let len = kind.length();
        let mask = (1u32 << len).wrapping_sub(1);
        let poly = kind.poly() & mask;

        let mut table = [0u32; 256];
        let mut nibble_table = [0u32; 16];

        if len >= 8 {
            for (i, entry) in table.iter_mut().enumerate() {
                let mut reg = 0u32;
                for shift in (0..8).rev() {
                    let bit = ((i >> shift) & 1) as u8;
                    reg = bit_serial_step(reg, bit, len, poly, mask);
                }
                *entry = reg;
            }
        } else {
            for (i, entry) in nibble_table.iter_mut().enumerate() {
                let mut reg = 0u32;
                for shift in (0..4).rev() {
                    let bit = ((i >> shift) & 1) as u8;
                    reg = bit_serial_step(reg, bit, len, poly, mask);
                }
                *entry = reg;
            }
        }

        Self {
            kind,
            table,
            nibble_table,
        }
    }

    /// Return the CRC kind this engine was built for.
    pub fn kind(&self) -> CrcKind {
        self.kind
    }

    /// Return the parity length $L$ in bits.
    pub fn length(&self) -> usize {
        self.kind.length()
    }

    /// Compute the CRC remainder over a bit-string `bits` (values 0 or 1,
    /// MSB-first, matching 3GPP §5.1 convention).
    ///
    /// # Arguments
    ///
    /// * `bits` - Slice of `u8` values, each must be 0 or 1.
    ///
    /// # Returns
    ///
    /// The $L$-bit remainder as a `u32` (only the low `L` bits are meaningful).
    ///
    /// # Examples
    ///
    /// ```
    /// use syndrome::crc::{Crc24, CrcKind};
    /// let crc = Crc24::new(CrcKind::Crc24A);
    /// // All-zero input must produce zero remainder.
    /// assert_eq!(crc.compute(&vec![0u8; 24]), 0);
    /// ```
    pub fn compute(&self, bits: &[u8]) -> u32 {
        let len = self.kind.length();
        let mask = (1u32 << len).wrapping_sub(1);
        let poly = self.kind.poly() & mask;
        let mut reg = 0u32;

        if len >= 8 {
            // Byte-wise table step: `reg' = (reg << 8) ^ table[(reg >> (len-8)) ^ byte]`.
            // Equivalent to 8 bit_serial_step calls by construction (see `new`).
            let chunks = bits.len() / 8;
            let processed = chunks * 8;
            for c in 0..chunks {
                let base = c * 8;
                let mut packed = 0u32;
                for k in 0..8 {
                    packed = (packed << 1) | (bits[base + k] as u32 & 1);
                }
                let idx = ((reg >> (len - 8)) ^ packed) & 0xFF;
                reg = ((reg << 8) & mask) ^ self.table[idx as usize];
            }
            // Tail: fewer than 8 bits left, fall back to the bit-serial step.
            for &bit in &bits[processed..] {
                reg = bit_serial_step(reg, bit, len, poly, mask);
            }
        } else {
            // Crc6 (len == 6 < 8): a byte shift would require `reg >> (len-8)`,
            // i.e. a negative shift. Use a 4-bit nibble table instead: same
            // construction, half the chunk width.
            let chunks = bits.len() / 4;
            let processed = chunks * 4;
            for c in 0..chunks {
                let base = c * 4;
                let mut nibble = 0u32;
                for k in 0..4 {
                    nibble = (nibble << 1) | (bits[base + k] as u32 & 1);
                }
                let idx = ((reg >> (len - 4)) ^ nibble) & 0xF;
                reg = ((reg << 4) & mask) ^ self.nibble_table[idx as usize];
            }
            // Tail: fewer than 4 bits left, fall back to the bit-serial step.
            for &bit in &bits[processed..] {
                reg = bit_serial_step(reg, bit, len, poly, mask);
            }
        }

        reg
    }

    /// Append $L$ CRC parity bits to `bits` in-place (MSB-first).
    ///
    /// After calling this, `bits.len()` increases by `L`.
    ///
    /// # Arguments
    ///
    /// * `bits` - Mutable bit-string to append the CRC to.
    ///
    /// # Examples
    ///
    /// ```
    /// use syndrome::crc::{Crc24, CrcKind};
    /// let crc = Crc24::new(CrcKind::Crc24B);
    /// let mut bits = vec![1u8, 0, 1, 0, 1, 1, 0, 1];
    /// crc.attach(&mut bits);
    /// assert!(crc.check(&bits));
    /// ```
    pub fn attach(&self, bits: &mut Vec<u8>) {
        let remainder = self.compute(bits);
        let len = self.kind.length();
        for shift in (0..len).rev() {
            bits.push(((remainder >> shift) & 1) as u8);
        }
    }

    /// Verify the CRC of a bit-string that includes $L$ appended parity bits.
    ///
    /// Returns `true` if the CRC of `bits[..n-L]` matches `bits[n-L..]`.
    ///
    /// # Arguments
    ///
    /// * `bits` - Complete bit-string (payload + CRC parity bits appended).
    ///
    /// # Returns
    ///
    /// `true` if the CRC matches; `false` if any bit flipped.
    pub fn check(&self, bits: &[u8]) -> bool {
        let l = self.kind.length();
        if bits.len() < l {
            return false;
        }
        let payload = &bits[..bits.len() - l];
        let received_crc = bits[bits.len() - l..]
            .iter()
            .fold(0u32, |acc, &b| (acc << 1) | (b as u32 & 1));
        self.compute(payload) == received_crc
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_zeros_produces_zero_remainder() {
        for kind in [CrcKind::Crc24A, CrcKind::Crc24B, CrcKind::Crc6] {
            let crc = Crc24::new(kind);
            assert_eq!(
                crc.compute(&vec![0u8; kind.length()]),
                0,
                "kind={kind:?} all-zeros should give remainder 0"
            );
        }
    }

    #[test]
    fn attach_then_check_roundtrip() {
        for kind in [
            CrcKind::Crc24A,
            CrcKind::Crc24B,
            CrcKind::Crc24C,
            CrcKind::Crc16,
            CrcKind::Crc11,
            CrcKind::Crc6,
        ] {
            let crc = Crc24::new(kind);
            let payload: Vec<u8> = (0u8..64).map(|i| i & 1).collect();
            let mut bits = payload.clone();
            crc.attach(&mut bits);
            assert_eq!(bits.len(), payload.len() + kind.length());
            assert!(crc.check(&bits), "round-trip failed for {kind:?}");
        }
    }

    #[test]
    fn single_bit_flip_detected() {
        let crc = Crc24::new(CrcKind::Crc24A);
        let payload: Vec<u8> = vec![1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0];
        let mut bits = payload.clone();
        crc.attach(&mut bits);
        bits[0] ^= 1; // flip one payload bit
        assert!(!crc.check(&bits));
    }

    #[test]
    fn crc_polynomial_lengths() {
        assert_eq!(CrcKind::Crc24A.length(), 24);
        assert_eq!(CrcKind::Crc16.length(), 16);
        assert_eq!(CrcKind::Crc11.length(), 11);
        assert_eq!(CrcKind::Crc6.length(), 6);
    }

    #[test]
    fn check_returns_false_for_too_short_input() {
        let crc = Crc24::new(CrcKind::Crc24A);
        assert!(!crc.check(&[0u8; 10]));
    }

    use crate::test_util::Xorshift64;

    /// Draw `len` random 0/1 bits from `rng` (see `crate::test_util`).
    fn next_bits(rng: &mut Xorshift64, len: usize) -> Vec<u8> {
        (0..len).map(|_| rng.next_bool() as u8).collect()
    }

    /// Table-driven `compute` must match the bit-serial reference exactly,
    /// for every `CrcKind`, over 200 seeded-random bit strings per kind with
    /// random lengths in `1..=2000` — including lengths not divisible by 8
    /// (exercising the tail loop) and, for `Crc6`, not divisible by 4.
    #[test]
    fn table_matches_bit_serial_reference() {
        for kind in [
            CrcKind::Crc24A,
            CrcKind::Crc24B,
            CrcKind::Crc24C,
            CrcKind::Crc16,
            CrcKind::Crc11,
            CrcKind::Crc6,
        ] {
            let crc = Crc24::new(kind);
            let len = kind.length();
            let mask = (1u32 << len).wrapping_sub(1);
            let poly = kind.poly() & mask;

            // Seed varies per kind so the 6 sweeps don't share a sequence.
            let mut rng = Xorshift64::new(0x9E3779B97F4A7C15 ^ (kind as u64 + 1));
            for _ in 0..200 {
                let n = rng.next_len(2000);
                let bits = next_bits(&mut rng, n);
                let table_driven = crc.compute(&bits);
                let bit_serial = compute_bit_serial(&bits, len, poly, mask);
                assert_eq!(
                    table_driven, bit_serial,
                    "kind={kind:?} len={n} mismatch: table={table_driven:#x} bit_serial={bit_serial:#x}"
                );
            }
        }
    }
}
