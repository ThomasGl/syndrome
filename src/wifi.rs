//! IEEE 802.11ax / 802.11be Wi-Fi 6/6E/7 LDPC parameters, MCS tables, and
//! full-codeword encode/decode.
//!
//! This module provides the LDPC code parameters defined in IEEE 802.11-2020
//! Annex R and the corresponding Wi-Fi 6/6E/7 Modulation and Coding Scheme
//! (MCS) tables. As of this crate's Wi-Fi LDPC integration, it is no longer
//! metadata-only: `crate::wifi_ldpc_tables` supplies the real 802.11 Annex
//! R/F shift matrices, so a genuine 802.11 codeword can be encoded and
//! decoded end-to-end with the same [`crate::qc_ldpc`] LOMS kernel used for
//! 5G NR — see [`crate::wifi_ldpc_tables::wifi_ldpc_encoder`]/
//! [`crate::wifi_ldpc_tables::wifi_ldpc_decoder`].
//!
//! # 802.11 LDPC structure
//!
//! All 802.11ax/be LDPC codes are quasi-cyclic with exactly 24 column blocks and
//! a lifting size $Z \in \{27, 54, 81\}$:
//!
//! $$N = 24 Z, \quad K = \lfloor N \cdot R \rfloor, \quad M = N - K$$
//!
//! where $R \in \{1/2,\, 2/3,\, 3/4,\, 5/6\}$ and the parity row-block count is
//! $M / Z$.
//!
//! The LOMS decoding algorithm is structurally identical to 5G NR QC-LDPC;
//! only the shift matrices differ, and those real matrices (all 12 $(Z, R)$
//! combinations, cross-validated against the IEEE 802.11n-2009 standard text
//! itself) now live in [`crate::wifi_ldpc_tables`].
//!
//! # What is real vs. still out of scope
//!
//! **Real today:** full, unshortened, unpunctured codeword encode/decode
//! ($K$ info bits exactly filling $N = 24Z$ coded bits) for all 12 $(Z, R)$
//! combinations, using the actual IEEE 802.11 Annex R/F shift matrices —
//! see `tests/wifi_ldpc_integration.rs` for encode -> AWGN -> decode
//! round-trips. **Shortening and puncturing** (a payload smaller than $K$,
//! a transmitted length smaller than the post-shortening budget) are also
//! implemented, in [`crate::wifi_rate_matching`] — see that module's docs
//! and `tests/wifi_shortening_puncturing_integration.rs` for the same
//! encode -> AWGN -> decode round-trip with genuine shortening and
//! puncturing applied, across all 12 combinations.
//!
//! **Still out of scope:**
//! * Multi-codeword segmentation — a payload larger than one codeword's
//!   $K$ is rejected, not split across several LDPC codewords (see
//!   [`crate::wifi_rate_matching`]'s module docs).
//! * The 802.11 PPDU-level formula that derives *how many* coded bits are
//!   available for a given MCS/bandwidth/PSDU length (IEEE 802.11-2020
//!   §19.5.3.2) — [`crate::wifi_rate_matching::encode_shortened`] takes the
//!   transmitted length directly rather than deriving it.
//! * The rest of the 802.11 PHY chain (scrambling, interleaving, OFDM
//!   subcarrier mapping) — out of scope for this FEC-only crate.
//!
//! See [`crate::wifi_ldpc_tables`] module docs for the full sourcing note
//! and cross-validation methodology for the 12 matrices.

use crate::error::FecError;

/// Wi-Fi generation identifier.
///
/// Selects the IEEE standard that governs MCS definitions and permitted
/// OFDMA/multi-link operation modes.  All three standards share the same
/// 802.11ax LDPC parity-check matrix structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WifiStandard {
    /// IEEE 802.11ax — Wi-Fi 6 (2.4 GHz and 5 GHz bands).
    WiFi6,
    /// IEEE 802.11ax-6GHz — Wi-Fi 6E (6 GHz band extension).
    WiFi6E,
    /// IEEE 802.11be — Wi-Fi 7 (multi-link operation, 4096-QAM).
    WiFi7,
}

/// 802.11ax/be LDPC code parameters for a single $(Z, R)$ combination.
///
/// All derived fields are exact integer arithmetic from the two primary
/// parameters `z` and `(rate_num, rate_den)`.  No floating-point rounding
/// is performed on the structural dimensions.
///
/// # Field relationships
///
/// $$N = 24 Z, \quad K = N \cdot \frac{\text{rate\_num}}{\text{rate\_den}},
///   \quad M = N - K, \quad \text{row\_blocks} = M / Z$$
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WifiLdpcParams {
    /// Wi-Fi generation for which these parameters are selected.
    pub standard: WifiStandard,
    /// Lifting size $Z \in \{27, 54, 81\}$.
    pub z: usize,
    /// Codeword length $N = 24 Z$.
    pub n: usize,
    /// Information bits $K = N \cdot R$ (integer; floor applied).
    pub k: usize,
    /// Parity bits $M = N - K$.
    pub m: usize,
    /// Number of parity row blocks $= M / Z$.
    ///
    /// Values: rate 1/2 → 12, rate 2/3 → 8, rate 3/4 → 6, rate 5/6 → 4.
    pub row_blocks: usize,
    /// Number of column blocks — always 24 for 802.11 LDPC.
    pub col_blocks: usize,
    /// Numerator of the code rate fraction.
    pub rate_num: usize,
    /// Denominator of the code rate fraction.
    pub rate_den: usize,
}

impl WifiLdpcParams {
    /// Compute the code rate as a floating-point value.
    ///
    /// $$R = \frac{\text{rate\_num}}{\text{rate\_den}}$$
    ///
    /// # Returns
    ///
    /// Code rate $R \in (0, 1)$ as `f32`.
    ///
    /// # Examples
    ///
    /// ```
    /// use syndrome::wifi::{WifiStandard, WifiLdpcParams};
    ///
    /// let p = WifiLdpcParams {
    ///     standard: WifiStandard::WiFi6,
    ///     z: 27, n: 648, k: 324, m: 324,
    ///     row_blocks: 12, col_blocks: 24,
    ///     rate_num: 1, rate_den: 2,
    /// };
    /// assert!((p.rate() - 0.5).abs() < 1e-6);
    /// ```
    pub fn rate(&self) -> f32 {
        self.rate_num as f32 / self.rate_den as f32
    }

    /// Build a ready-to-use [`crate::qc_ldpc::QcLdpcEncoder`] for this
    /// $(Z, R)$ combination, backed by the real IEEE 802.11 Annex R/F shift
    /// matrix (see [`crate::wifi_ldpc_tables`] for sourcing).
    ///
    /// Handles exactly one full, unshortened, unpunctured codeword: `K`
    /// info bits in, `N = 24*z` coded bits out. For a payload smaller than
    /// `K` and/or a transmitted length smaller than `N`, encode through
    /// this encoder via [`crate::wifi_rate_matching::encode_shortened`]
    /// instead — see that module's doc.
    ///
    /// # Errors
    ///
    /// Returns [`FecError::InvalidParam`] if this instance's `(z, rate_num,
    /// rate_den)` is not one of the 12 supported 802.11 LDPC combinations
    /// (this should not happen for a `WifiLdpcParams` obtained from
    /// [`select_wifi_ldpc`] or [`wifi_ldpc_params_for_mcs`], since those
    /// only ever select `z in {27, 54, 81}` and one of the four standard
    /// rates).
    ///
    /// # Examples
    ///
    /// ```
    /// use syndrome::wifi::{select_wifi_ldpc, WifiStandard};
    ///
    /// let p = select_wifi_ldpc(20, WifiStandard::WiFi6, 0.5);
    /// let enc = p.build_encoder().unwrap();
    /// assert_eq!(enc.codeword_bit_count(), p.n);
    /// ```
    pub fn build_encoder(&self) -> Result<crate::qc_ldpc::QcLdpcEncoder, FecError> {
        crate::wifi_ldpc_tables::wifi_ldpc_encoder(self.z, self.rate_num, self.rate_den)
    }

    /// Build a ready-to-use [`crate::qc_ldpc::QcLdpcDecoder`] for this
    /// $(Z, R)$ combination, backed by the real IEEE 802.11 Annex R/F shift
    /// matrix (see [`crate::wifi_ldpc_tables`] for sourcing).
    ///
    /// # Arguments
    ///
    /// * `offset_beta` - Offset correction $\beta$ for the LOMS min-sum
    ///   update (0.25 is a reasonable default; see the 5G NR usage in
    ///   [`crate::qc_ldpc`]).
    ///
    /// # Errors
    ///
    /// Returns [`FecError::InvalidParam`] under the same condition as
    /// [`WifiLdpcParams::build_encoder`].
    ///
    /// # Examples
    ///
    /// ```
    /// use syndrome::wifi::{select_wifi_ldpc, WifiStandard};
    ///
    /// let p = select_wifi_ldpc(20, WifiStandard::WiFi6, 0.5);
    /// let dec = p.build_decoder(0.25).unwrap();
    /// assert_eq!(dec.variable_node_count(), p.n);
    /// ```
    pub fn build_decoder(
        &self,
        offset_beta: f32,
    ) -> Result<crate::qc_ldpc::QcLdpcDecoder, FecError> {
        crate::wifi_ldpc_tables::wifi_ldpc_decoder(
            self.z,
            self.rate_num,
            self.rate_den,
            offset_beta,
        )
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// The four 802.11 LDPC code-rate fractions in ascending order.
const RATES: [(usize, usize); 4] = [(1, 2), (2, 3), (3, 4), (5, 6)];

/// Build a [`WifiLdpcParams`] from primitive components.
fn make_params(
    standard: WifiStandard,
    z: usize,
    rate_num: usize,
    rate_den: usize,
) -> WifiLdpcParams {
    let n = 24 * z;
    // Integer floor: K = N * rate_num / rate_den
    let k = n * rate_num / rate_den;
    let m = n - k;
    let row_blocks = m / z;
    WifiLdpcParams {
        standard,
        z,
        n,
        k,
        m,
        row_blocks,
        col_blocks: 24,
        rate_num,
        rate_den,
    }
}

/// Choose the rate fraction from `RATES` closest to `target_rate`.
fn closest_rate(target_rate: f32) -> (usize, usize) {
    RATES
        .iter()
        .copied()
        .min_by(|&(an, ad), &(bn, bd)| {
            let da = (an as f32 / ad as f32 - target_rate).abs();
            let db = (bn as f32 / bd as f32 - target_rate).abs();
            da.partial_cmp(&db).unwrap_or(core::cmp::Ordering::Equal)
        })
        .unwrap_or((1, 2))
}

// ---------------------------------------------------------------------------
// Public selector
// ---------------------------------------------------------------------------

/// Select the smallest LDPC codeword that can carry `payload_bytes` information
/// bits at the rate closest to `target_rate`.
///
/// The three available lifting sizes $Z \in \{27, 54, 81\}$ yield information
/// capacities (at a given rate $R$):
///
/// $$K(Z, R) = 24 Z \cdot R$$
///
/// The selector picks the smallest $Z$ such that $K(Z, R) \ge 8 \cdot
/// \text{payload\_bytes}$.  If no single codeword is large enough (payload
/// exceeds $K(81, R)$) the largest codeword ($Z = 81$) is returned and the
/// caller is responsible for segmentation.
///
/// # Arguments
///
/// * `payload_bytes` — Number of information bytes to protect.
/// * `standard`      — Target Wi-Fi generation (stored in the result).
/// * `target_rate`   — Desired code rate; the closest 802.11 rate is selected.
///
/// # Returns
///
/// A [`WifiLdpcParams`] struct describing the chosen LDPC configuration.
///
/// # Examples
///
/// ```
/// use syndrome::wifi::{select_wifi_ldpc, WifiStandard};
///
/// let p = select_wifi_ldpc(20, WifiStandard::WiFi6, 0.5);
/// assert_eq!(p.z, 27);        // 20 bytes = 160 bits; K(27,1/2)=324 ≥ 160
/// assert_eq!(p.rate_num, 1);
/// assert_eq!(p.rate_den, 2);
/// ```
pub fn select_wifi_ldpc(
    payload_bytes: usize,
    standard: WifiStandard,
    target_rate: f32,
) -> WifiLdpcParams {
    let (rate_num, rate_den) = closest_rate(target_rate);
    let payload_bits = payload_bytes * 8;

    for &z in &[27_usize, 54, 81] {
        let k = 24 * z * rate_num / rate_den;
        if k >= payload_bits {
            return make_params(standard, z, rate_num, rate_den);
        }
    }
    // Payload is larger than any single codeword; return the maximum size.
    make_params(standard, 81, rate_num, rate_den)
}

// ---------------------------------------------------------------------------
// MCS tables
// ---------------------------------------------------------------------------

/// A single entry in the 802.11ax/be Modulation and Coding Scheme table.
///
/// The throughput in bits per OFDM subcarrier per spatial stream is:
/// $$\eta = \text{bits\_per\_symbol} \times \frac{\text{code\_rate\_num}}{\text{code\_rate\_den}}$$
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WifiMcs {
    /// MCS index (0-based).
    pub mcs_index: u8,
    /// Modulation scheme name (e.g. `"BPSK"`, `"QPSK"`, `"256-QAM"`).
    pub modulation: &'static str,
    /// Number of coded bits per OFDM subcarrier per stream ($\log_2 M$ for $M$-QAM).
    pub bits_per_symbol: u8,
    /// Numerator of the code-rate fraction.
    pub code_rate_num: u8,
    /// Denominator of the code-rate fraction.
    pub code_rate_den: u8,
}

/// Wi-Fi 6 (IEEE 802.11ax) MCS table, indices 0–11.
///
/// Covers BPSK through 1024-QAM as defined in IEEE 802.11ax-2021 Table 27-47.
/// MCS 10 and 11 (1024-QAM) are mandatory for Wi-Fi 6 CERTIFIED devices.
///
/// | MCS | Modulation   | Bits/sym | Rate |
/// |-----|-------------|----------|------|
/// |  0  | BPSK        |    1     | 1/2  |
/// |  1  | QPSK        |    2     | 1/2  |
/// |  2  | QPSK        |    2     | 3/4  |
/// |  3  | 16-QAM      |    4     | 1/2  |
/// |  4  | 16-QAM      |    4     | 3/4  |
/// |  5  | 64-QAM      |    6     | 2/3  |
/// |  6  | 64-QAM      |    6     | 3/4  |
/// |  7  | 64-QAM      |    6     | 5/6  |
/// |  8  | 256-QAM     |    8     | 3/4  |
/// |  9  | 256-QAM     |    8     | 5/6  |
/// | 10  | 1024-QAM    |   10     | 3/4  |
/// | 11  | 1024-QAM    |   10     | 5/6  |
pub const WIFI6_MCS_TABLE: [WifiMcs; 12] = [
    WifiMcs {
        mcs_index: 0,
        modulation: "BPSK",
        bits_per_symbol: 1,
        code_rate_num: 1,
        code_rate_den: 2,
    },
    WifiMcs {
        mcs_index: 1,
        modulation: "QPSK",
        bits_per_symbol: 2,
        code_rate_num: 1,
        code_rate_den: 2,
    },
    WifiMcs {
        mcs_index: 2,
        modulation: "QPSK",
        bits_per_symbol: 2,
        code_rate_num: 3,
        code_rate_den: 4,
    },
    WifiMcs {
        mcs_index: 3,
        modulation: "16-QAM",
        bits_per_symbol: 4,
        code_rate_num: 1,
        code_rate_den: 2,
    },
    WifiMcs {
        mcs_index: 4,
        modulation: "16-QAM",
        bits_per_symbol: 4,
        code_rate_num: 3,
        code_rate_den: 4,
    },
    WifiMcs {
        mcs_index: 5,
        modulation: "64-QAM",
        bits_per_symbol: 6,
        code_rate_num: 2,
        code_rate_den: 3,
    },
    WifiMcs {
        mcs_index: 6,
        modulation: "64-QAM",
        bits_per_symbol: 6,
        code_rate_num: 3,
        code_rate_den: 4,
    },
    WifiMcs {
        mcs_index: 7,
        modulation: "64-QAM",
        bits_per_symbol: 6,
        code_rate_num: 5,
        code_rate_den: 6,
    },
    WifiMcs {
        mcs_index: 8,
        modulation: "256-QAM",
        bits_per_symbol: 8,
        code_rate_num: 3,
        code_rate_den: 4,
    },
    WifiMcs {
        mcs_index: 9,
        modulation: "256-QAM",
        bits_per_symbol: 8,
        code_rate_num: 5,
        code_rate_den: 6,
    },
    WifiMcs {
        mcs_index: 10,
        modulation: "1024-QAM",
        bits_per_symbol: 10,
        code_rate_num: 3,
        code_rate_den: 4,
    },
    WifiMcs {
        mcs_index: 11,
        modulation: "1024-QAM",
        bits_per_symbol: 10,
        code_rate_num: 5,
        code_rate_den: 6,
    },
];

/// Wi-Fi 7 (IEEE 802.11be) MCS table, indices 0–13.
///
/// Extends the Wi-Fi 6 MCS table with MCS 12 and 13, which use 4096-QAM
/// ($\log_2 4096 = 12$ bits per subcarrier).  These require extremely high
/// receive SNR (typically $> 40$ dB) and are used in near-field / wired
/// backhaul scenarios.
///
/// | MCS | Modulation   | Bits/sym | Rate |
/// |-----|-------------|----------|------|
/// |  0–11 | (same as Wi-Fi 6)         |
/// | 12  | 4096-QAM    |   12     | 3/4  |
/// | 13  | 4096-QAM    |   12     | 5/6  |
pub const WIFI7_MCS_TABLE: [WifiMcs; 14] = [
    WifiMcs {
        mcs_index: 0,
        modulation: "BPSK",
        bits_per_symbol: 1,
        code_rate_num: 1,
        code_rate_den: 2,
    },
    WifiMcs {
        mcs_index: 1,
        modulation: "QPSK",
        bits_per_symbol: 2,
        code_rate_num: 1,
        code_rate_den: 2,
    },
    WifiMcs {
        mcs_index: 2,
        modulation: "QPSK",
        bits_per_symbol: 2,
        code_rate_num: 3,
        code_rate_den: 4,
    },
    WifiMcs {
        mcs_index: 3,
        modulation: "16-QAM",
        bits_per_symbol: 4,
        code_rate_num: 1,
        code_rate_den: 2,
    },
    WifiMcs {
        mcs_index: 4,
        modulation: "16-QAM",
        bits_per_symbol: 4,
        code_rate_num: 3,
        code_rate_den: 4,
    },
    WifiMcs {
        mcs_index: 5,
        modulation: "64-QAM",
        bits_per_symbol: 6,
        code_rate_num: 2,
        code_rate_den: 3,
    },
    WifiMcs {
        mcs_index: 6,
        modulation: "64-QAM",
        bits_per_symbol: 6,
        code_rate_num: 3,
        code_rate_den: 4,
    },
    WifiMcs {
        mcs_index: 7,
        modulation: "64-QAM",
        bits_per_symbol: 6,
        code_rate_num: 5,
        code_rate_den: 6,
    },
    WifiMcs {
        mcs_index: 8,
        modulation: "256-QAM",
        bits_per_symbol: 8,
        code_rate_num: 3,
        code_rate_den: 4,
    },
    WifiMcs {
        mcs_index: 9,
        modulation: "256-QAM",
        bits_per_symbol: 8,
        code_rate_num: 5,
        code_rate_den: 6,
    },
    WifiMcs {
        mcs_index: 10,
        modulation: "1024-QAM",
        bits_per_symbol: 10,
        code_rate_num: 3,
        code_rate_den: 4,
    },
    WifiMcs {
        mcs_index: 11,
        modulation: "1024-QAM",
        bits_per_symbol: 10,
        code_rate_num: 5,
        code_rate_den: 6,
    },
    WifiMcs {
        mcs_index: 12,
        modulation: "4096-QAM",
        bits_per_symbol: 12,
        code_rate_num: 3,
        code_rate_den: 4,
    },
    WifiMcs {
        mcs_index: 13,
        modulation: "4096-QAM",
        bits_per_symbol: 12,
        code_rate_num: 5,
        code_rate_den: 6,
    },
];

/// Select LDPC parameters suitable for the given MCS and payload size.
///
/// Uses the MCS code-rate fraction to call [`select_wifi_ldpc`], passing
/// `WiFi6E` as the standard (the LDPC structure is the same for all three
/// generations; the caller should override `.standard` if needed).
///
/// # Arguments
///
/// * `mcs`           — MCS entry from [`WIFI6_MCS_TABLE`] or [`WIFI7_MCS_TABLE`].
/// * `payload_bytes` — Number of information bytes to protect.
///
/// # Returns
///
/// The smallest [`WifiLdpcParams`] whose information capacity $K$ is at least
/// $8 \times \text{payload\_bytes}$ at the MCS code rate.
///
/// # Examples
///
/// ```
/// use syndrome::wifi::{wifi_ldpc_params_for_mcs, WIFI6_MCS_TABLE};
///
/// // MCS 5: 64-QAM 2/3
/// let p = wifi_ldpc_params_for_mcs(&WIFI6_MCS_TABLE[5], 30);
/// assert_eq!(p.rate_num, 2);
/// assert_eq!(p.rate_den, 3);
/// assert!(p.k >= 30 * 8);
/// ```
pub fn wifi_ldpc_params_for_mcs(mcs: &WifiMcs, payload_bytes: usize) -> WifiLdpcParams {
    let target_rate = mcs.code_rate_num as f32 / mcs.code_rate_den as f32;
    // Default to WiFi6E as a neutral standard; callers can rebind .standard.
    select_wifi_ldpc(payload_bytes, WifiStandard::WiFi6E, target_rate)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Confirm that both MCS tables have the correct number of entries and that
    /// `mcs_index` values are contiguous starting from 0.
    #[test]
    fn mcs_table_lengths() {
        assert_eq!(WIFI6_MCS_TABLE.len(), 12);
        assert_eq!(WIFI7_MCS_TABLE.len(), 14);

        for (i, mcs) in WIFI6_MCS_TABLE.iter().enumerate() {
            assert_eq!(
                mcs.mcs_index as usize, i,
                "WiFi6 MCS index mismatch at slot {i}"
            );
        }
        for (i, mcs) in WIFI7_MCS_TABLE.iter().enumerate() {
            assert_eq!(
                mcs.mcs_index as usize, i,
                "WiFi7 MCS index mismatch at slot {i}"
            );
        }
    }

    /// Verify that `WifiLdpcParams::rate()` returns the correct fraction for all
    /// standard 802.11 rates.
    #[test]
    fn rate_fraction_correct() {
        let cases: &[(usize, usize, f32)] = &[
            (1, 2, 0.5),
            (2, 3, 2.0 / 3.0),
            (3, 4, 0.75),
            (5, 6, 5.0 / 6.0),
        ];
        for &(rn, rd, expected) in cases {
            let p = make_params(WifiStandard::WiFi6, 27, rn, rd);
            assert!(
                (p.rate() - expected).abs() < 1e-6,
                "rate({rn}/{rd}) = {} ≠ {expected}",
                p.rate()
            );
        }
    }

    /// Check that `select_wifi_ldpc` always returns a codeword whose $K$ is at
    /// least as large as the payload in bits.
    #[test]
    fn select_params_fits_payload() {
        let test_cases: &[(usize, f32)] =
            &[(10, 0.5), (20, 0.5), (40, 0.5), (80, 0.75), (100, 0.5)];
        for &(bytes, rate) in test_cases {
            let p = select_wifi_ldpc(bytes, WifiStandard::WiFi6, rate);
            assert!(
                p.k >= bytes * 8,
                "payload {bytes} B ({} bits): K={} is too small (Z={}, rate={}/{})",
                bytes * 8,
                p.k,
                p.z,
                p.rate_num,
                p.rate_den
            );
        }
    }

    /// Confirm that Wi-Fi 7 extends Wi-Fi 6 with 4096-QAM at MCS 12 and 13.
    #[test]
    fn wifi7_adds_4096qam() {
        // The first 12 entries must match Wi-Fi 6 exactly.
        for i in 0..12 {
            assert_eq!(
                WIFI7_MCS_TABLE[i], WIFI6_MCS_TABLE[i],
                "MCS {i} differs between Wi-Fi 6 and Wi-Fi 7 tables"
            );
        }
        // MCS 12 and 13 must be 4096-QAM.
        assert_eq!(WIFI7_MCS_TABLE[12].modulation, "4096-QAM");
        assert_eq!(WIFI7_MCS_TABLE[12].bits_per_symbol, 12);
        assert_eq!(WIFI7_MCS_TABLE[12].code_rate_num, 3);
        assert_eq!(WIFI7_MCS_TABLE[12].code_rate_den, 4);

        assert_eq!(WIFI7_MCS_TABLE[13].modulation, "4096-QAM");
        assert_eq!(WIFI7_MCS_TABLE[13].bits_per_symbol, 12);
        assert_eq!(WIFI7_MCS_TABLE[13].code_rate_num, 5);
        assert_eq!(WIFI7_MCS_TABLE[13].code_rate_den, 6);
    }

    /// Verify structural invariants for all $(Z, R)$ combinations:
    /// $N = 24Z$, $M = N - K$, $\text{row\_blocks} = M / Z$, $\text{col\_blocks} = 24$.
    #[test]
    fn structural_invariants_all_combinations() {
        for &z in &[27_usize, 54, 81] {
            for &(rn, rd) in &RATES {
                let p = make_params(WifiStandard::WiFi7, z, rn, rd);
                assert_eq!(p.n, 24 * z, "N invariant failed for Z={z}");
                assert_eq!(
                    p.m,
                    p.n - p.k,
                    "M = N-K invariant failed for Z={z} rate={rn}/{rd}"
                );
                assert_eq!(
                    p.row_blocks,
                    p.m / z,
                    "row_blocks invariant failed for Z={z} rate={rn}/{rd}"
                );
                assert_eq!(p.col_blocks, 24, "col_blocks must always be 24");
            }
        }
    }

    /// Verify that `wifi_ldpc_params_for_mcs` respects the MCS code rate.
    #[test]
    fn ldpc_for_mcs_uses_mcs_rate() {
        // MCS 7: 64-QAM 5/6
        let mcs = &WIFI6_MCS_TABLE[7];
        let p = wifi_ldpc_params_for_mcs(mcs, 10);
        assert_eq!(p.rate_num, 5);
        assert_eq!(p.rate_den, 6);
    }

    /// Boundary: payload exactly equal to K should return the same Z.
    #[test]
    fn select_params_exact_boundary() {
        // K for Z=27, rate=1/2: 24*27*1/2 = 324 bits = 40 bytes (floor 40.5)
        // 324/8 = 40 bytes exactly.
        let k_27_half = (24_usize * 27) / 2; // = 324 bits
        let bytes = k_27_half / 8; // = 40
        let p = select_wifi_ldpc(bytes, WifiStandard::WiFi6, 0.5);
        assert_eq!(p.z, 27, "Should select Z=27 when payload exactly fits");
    }
}
