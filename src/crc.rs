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
//! # Examples
//!
//! ```
//! use glezer_rsv::crc::{Crc24, CrcKind};
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

/// CRC engine built around one of the 3GPP generator polynomials.
///
/// Uses a bit-serial LFSR shift register so it works correctly for all
/// polynomial lengths (6, 11, 16, 24 bits). This is a setup-path operation
/// (runs on transport blocks / code blocks, not in the LDPC inner loop),
/// so throughput optimisation is not needed.
pub struct Crc24 {
    kind: CrcKind,
}

impl Crc24 {
    /// Construct a CRC engine for the given polynomial kind.
    ///
    /// # Arguments
    ///
    /// * `kind` - Which 3GPP CRC polynomial to use.
    ///
    /// # Examples
    ///
    /// ```
    /// use glezer_rsv::crc::{Crc24, CrcKind};
    /// let crc = Crc24::new(CrcKind::Crc24A);
    /// ```
    pub fn new(kind: CrcKind) -> Self {
        Self { kind }
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
    /// use glezer_rsv::crc::{Crc24, CrcKind};
    /// let crc = Crc24::new(CrcKind::Crc24A);
    /// // All-zero input must produce zero remainder.
    /// assert_eq!(crc.compute(&vec![0u8; 24]), 0);
    /// ```
    pub fn compute(&self, bits: &[u8]) -> u32 {
        let len = self.kind.length();
        let mask = (1u32 << len).wrapping_sub(1);
        let poly = self.kind.poly() & mask;
        let mut reg = 0u32;
        // Bit-serial LFSR: works for any poly length 1..=32.
        for &bit in bits {
            let feedback = ((reg >> (len - 1)) ^ (bit as u32 & 1)) & 1;
            reg = (reg << 1) & mask;
            if feedback != 0 {
                reg ^= poly;
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
    /// use glezer_rsv::crc::{Crc24, CrcKind};
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
}
