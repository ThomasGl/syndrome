//! The 12 real IEEE 802.11 LDPC parity-check prototype matrices (shift tables).
//!
//! This module supplies the actual sparse $(row, col, shift)$ edge lists for
//! all 12 802.11 QC-LDPC base graphs — 3 lifting sizes $Z \in \{27, 54, 81\}$
//! times 4 code rates $R \in \{1/2, 2/3, 3/4, 5/6\}$ — so that
//! [`crate::qc_ldpc::QcLdpcEncoder`]/[`crate::qc_ldpc::QcLdpcDecoder`] can be
//! instantiated for real 802.11 codewords via
//! [`crate::qc_ldpc::QcLdpcParams::from_raw_edges`].
//!
//! Unlike the 3GPP BG1/BG2 tables in [`crate::bg_tables`], 802.11 shift
//! values are **literal per-Z absolute shifts** (each of the 3 lifting sizes
//! has its own complete, independent table) — there is no
//! lifting-size-*set*/`ils_for_z`/modulo-scaling step as in 3GPP TS 38.212
//! Table 5.3.2-1. Every shift value stored here is already `< Z` for its row.
//!
//! # Matrix shape
//!
//! Every 802.11 LDPC base graph has exactly 24 column blocks; the row-block
//! count is fixed by the code rate (see [`crate::wifi::WifiLdpcParams`]):
//!
//! | Rate | Row blocks $m_b$ | Info column blocks $k_b = 24 - m_b$ |
//! |------|------|------|
//! | 1/2  | 12   | 12   |
//! | 2/3  | 8    | 16   |
//! | 3/4  | 6    | 18   |
//! | 5/6  | 4    | 20   |
//!
//! # Sourcing and cross-validation
//!
//! All 12 matrices were obtained and cross-checked from **three** independent
//! sources that agree exactly, element-for-element, after normalizing each
//! source's own "no connection" sentinel (`-1`, `-11`, or a bare `-`) to a
//! single representation:
//!
//! 1. **The IEEE 802.11n-2009 standard text itself** — Annex R ("HT LDPC
//!    matrix definitions"), Tables R.1 (Z=27), R.2 (Z=54), and R.3 (Z=81),
//!    parts (a)-(d) for rates 1/2, 2/3, 3/4, 5/6 respectively. This is the
//!    original normative table (802.11-2016/2020 carries Annex R forward
//!    unchanged, renumbering it; the LDPC codes themselves have never
//!    changed across 802.11n/ac/ax/be). The standard text was extracted
//!    programmatically (`pypdf`) from the copy of the amendment bundled in
//!    the `ldpc-matrix-prototypes/802.11n-2009.pdf` file of the
//!    `jeroenoverman/ECC-LDPC-application` GitHub repository (TU Delft
//!    Error Correcting Codes course project) and parsed back into numeric
//!    matrices for a byte-for-byte diff against the other two sources below.
//! 2. <https://github.com/simgunz/802.11n-ldpc> — `protoH/{648,1296,1944}_{12,23,34,56}`,
//!    a standalone MATLAB 802.11n LDPC encoder/decoder implementation whose
//!    `buildH.m` reads these files as "the prototype matrix" directly from
//!    the standard.
//! 3. <https://github.com/jeroenoverman/ECC-LDPC-application> —
//!    `ldpc-matrix-prototypes/n648-z27-r{1_2,2_3,3_4,5_6}.txt`, transcribed
//!    independently from the same standard (Z=27 rates only; this source
//!    does not carry Z=54/Z=81 transcriptions but was cross-checked against
//!    the standard PDF it ships alongside, which does cover all 12).
//!
//! All three sources matched exactly (after sentinel normalization) for
//! every one of the 12 matrices; every shift value here was confirmed
//! `0 <= shift < Z` for its row. **Confidence: high, all 12 matrices**, none
//! were extrapolated or approximated.
//!
//! # Explicitly out of scope (not modeled by this table)
//!
//! These raw matrices only describe the parity-check structure for a
//! **full, unshortened, unpunctured codeword** ($K$ info bits exactly fill
//! $N = 24Z$ coded bits). The following real 802.11 PHY mechanics are
//! deliberately not implemented anywhere in this crate yet:
//!
//! * **Shortening** — zero-padding a payload smaller than $K$ before
//!   encoding and discarding the pad after decoding (802.11-2020 §19.3.11.7.1 /
//!   §36.3.11.4.3 for EHT).
//! * **Puncturing / rate-matching** — the per-MCS coded-bit selection that
//!   maps a full codeword onto the actual number of bits transmitted in an
//!   OFDM symbol allocation.
//! * The rest of the 802.11 PHY chain (scrambling, bit interleaving, OFDM
//!   subcarrier mapping) — entirely out of scope for this FEC-only crate.
//!
//! See [`crate::wifi`] module docs and the crate `CHANGELOG.md` for the
//! same scope note.

use crate::error::FecError;
use crate::qc_ldpc::{QcLdpcDecoder, QcLdpcEncoder};

/// One 802.11 LDPC prototype matrix: a sparse $(row\_block, col\_block,
/// shift)$ edge list plus its shape, for a single $(Z, R)$ combination.
///
/// `shift` values are literal (already `< z`); see the module doc for why
/// this differs from the 3GPP per-iLS scaled tables.
#[derive(Debug, Clone, Copy)]
pub struct WifiLdpcMatrix {
    /// Lifting size $Z \in \{27, 54, 81\}$.
    pub z: usize,
    /// Numerator of the code rate fraction.
    pub rate_num: usize,
    /// Denominator of the code rate fraction.
    pub rate_den: usize,
    /// Number of check-node block rows $m_b$.
    pub row_blocks: usize,
    /// Number of variable-node block columns $n_b$ (always 24).
    pub col_blocks: usize,
    /// Sparse edges: `(row_block, col_block, shift)`, `shift` already `< z`.
    pub entries: &'static [(u8, u8, i16)],
}

#[rustfmt::skip]
const WIFI_Z27_R1_2_ENTRIES: [(u8, u8, i16); 88] = [
    (0, 0, 0), (0, 4, 0), (0, 5, 0), (0, 8, 0), (0, 11, 0), (0, 12, 1),
    (0, 13, 0), (1, 0, 22), (1, 1, 0), (1, 4, 17), (1, 6, 0), (1, 7, 0),
    (1, 8, 12), (1, 13, 0), (1, 14, 0), (2, 0, 6), (2, 2, 0), (2, 4, 10),
    (2, 8, 24), (2, 10, 0), (2, 14, 0), (2, 15, 0), (3, 0, 2), (3, 3, 0),
    (3, 4, 20), (3, 8, 25), (3, 9, 0), (3, 15, 0), (3, 16, 0), (4, 0, 23),
    (4, 4, 3), (4, 8, 0), (4, 10, 9), (4, 11, 11), (4, 16, 0), (4, 17, 0),
    (5, 0, 24), (5, 2, 23), (5, 3, 1), (5, 4, 17), (5, 6, 3), (5, 8, 10),
    (5, 17, 0), (5, 18, 0), (6, 0, 25), (6, 4, 8), (6, 8, 7), (6, 9, 18),
    (6, 12, 0), (6, 18, 0), (6, 19, 0), (7, 0, 13), (7, 1, 24), (7, 4, 0),
    (7, 6, 8), (7, 8, 6), (7, 19, 0), (7, 20, 0), (8, 0, 7), (8, 1, 20),
    (8, 3, 16), (8, 4, 22), (8, 5, 10), (8, 8, 23), (8, 20, 0), (8, 21, 0),
    (9, 0, 11), (9, 4, 19), (9, 8, 13), (9, 10, 3), (9, 11, 17), (9, 21, 0),
    (9, 22, 0), (10, 0, 25), (10, 2, 8), (10, 4, 23), (10, 5, 18), (10, 7, 14),
    (10, 8, 9), (10, 22, 0), (10, 23, 0), (11, 0, 3), (11, 4, 16), (11, 7, 2),
    (11, 8, 25), (11, 9, 5), (11, 12, 1), (11, 23, 0),
];

#[rustfmt::skip]
const WIFI_Z27_R2_3_ENTRIES: [(u8, u8, i16); 88] = [
    (0, 0, 25), (0, 1, 26), (0, 2, 14), (0, 4, 20), (0, 6, 2), (0, 8, 4),
    (0, 11, 8), (0, 13, 16), (0, 15, 18), (0, 16, 1), (0, 17, 0), (1, 0, 10),
    (1, 1, 9), (1, 2, 15), (1, 3, 11), (1, 5, 0), (1, 7, 1), (1, 10, 18),
    (1, 12, 8), (1, 14, 10), (1, 17, 0), (1, 18, 0), (2, 0, 16), (2, 1, 2),
    (2, 2, 20), (2, 3, 26), (2, 4, 21), (2, 6, 6), (2, 8, 1), (2, 9, 26),
    (2, 11, 7), (2, 18, 0), (2, 19, 0), (3, 0, 10), (3, 1, 13), (3, 2, 5),
    (3, 3, 0), (3, 5, 3), (3, 7, 7), (3, 10, 26), (3, 13, 13), (3, 15, 16),
    (3, 19, 0), (3, 20, 0), (4, 0, 23), (4, 1, 14), (4, 2, 24), (4, 4, 12),
    (4, 6, 19), (4, 8, 17), (4, 12, 20), (4, 14, 21), (4, 16, 0), (4, 20, 0),
    (4, 21, 0), (5, 0, 6), (5, 1, 22), (5, 2, 9), (5, 3, 20), (5, 5, 25),
    (5, 7, 17), (5, 9, 8), (5, 11, 14), (5, 13, 18), (5, 21, 0), (5, 22, 0),
    (6, 0, 14), (6, 1, 23), (6, 2, 21), (6, 3, 11), (6, 4, 20), (6, 6, 24),
    (6, 8, 18), (6, 10, 19), (6, 15, 22), (6, 22, 0), (6, 23, 0), (7, 0, 17),
    (7, 1, 11), (7, 2, 11), (7, 3, 20), (7, 5, 21), (7, 7, 26), (7, 9, 3),
    (7, 12, 18), (7, 14, 26), (7, 16, 1), (7, 23, 0),
];

#[rustfmt::skip]
const WIFI_Z27_R3_4_ENTRIES: [(u8, u8, i16); 88] = [
    (0, 0, 16), (0, 1, 17), (0, 2, 22), (0, 3, 24), (0, 4, 9), (0, 5, 3),
    (0, 6, 14), (0, 8, 4), (0, 9, 2), (0, 10, 7), (0, 12, 26), (0, 14, 2),
    (0, 16, 21), (0, 18, 1), (0, 19, 0), (1, 0, 25), (1, 1, 12), (1, 2, 12),
    (1, 3, 3), (1, 4, 3), (1, 5, 26), (1, 6, 6), (1, 7, 21), (1, 9, 15),
    (1, 10, 22), (1, 12, 15), (1, 14, 4), (1, 17, 16), (1, 19, 0), (1, 20, 0),
    (2, 0, 25), (2, 1, 18), (2, 2, 26), (2, 3, 16), (2, 4, 22), (2, 5, 23),
    (2, 6, 9), (2, 8, 0), (2, 10, 4), (2, 12, 4), (2, 14, 8), (2, 15, 23),
    (2, 16, 11), (2, 20, 0), (2, 21, 0), (3, 0, 9), (3, 1, 7), (3, 2, 0),
    (3, 3, 1), (3, 4, 17), (3, 7, 7), (3, 8, 3), (3, 10, 3), (3, 11, 23),
    (3, 13, 16), (3, 16, 21), (3, 18, 0), (3, 21, 0), (3, 22, 0), (4, 0, 24),
    (4, 1, 5), (4, 2, 26), (4, 3, 7), (4, 4, 1), (4, 7, 15), (4, 8, 24),
    (4, 9, 15), (4, 11, 8), (4, 13, 13), (4, 15, 13), (4, 17, 11), (4, 22, 0),
    (4, 23, 0), (5, 0, 2), (5, 1, 2), (5, 2, 19), (5, 3, 14), (5, 4, 24),
    (5, 5, 1), (5, 6, 15), (5, 7, 19), (5, 9, 21), (5, 11, 2), (5, 13, 24),
    (5, 15, 3), (5, 17, 2), (5, 18, 1), (5, 23, 0),
];

#[rustfmt::skip]
const WIFI_Z27_R5_6_ENTRIES: [(u8, u8, i16); 88] = [
    (0, 0, 17), (0, 1, 13), (0, 2, 8), (0, 3, 21), (0, 4, 9), (0, 5, 3),
    (0, 6, 18), (0, 7, 12), (0, 8, 10), (0, 9, 0), (0, 10, 4), (0, 11, 15),
    (0, 12, 19), (0, 13, 2), (0, 14, 5), (0, 15, 10), (0, 16, 26), (0, 17, 19),
    (0, 18, 13), (0, 19, 13), (0, 20, 1), (0, 21, 0), (1, 0, 3), (1, 1, 12),
    (1, 2, 11), (1, 3, 14), (1, 4, 11), (1, 5, 25), (1, 6, 5), (1, 7, 18),
    (1, 8, 0), (1, 9, 9), (1, 10, 2), (1, 11, 26), (1, 12, 26), (1, 13, 10),
    (1, 14, 24), (1, 15, 7), (1, 16, 14), (1, 17, 20), (1, 18, 4), (1, 19, 2),
    (1, 21, 0), (1, 22, 0), (2, 0, 22), (2, 1, 16), (2, 2, 4), (2, 3, 3),
    (2, 4, 10), (2, 5, 21), (2, 6, 12), (2, 7, 5), (2, 8, 21), (2, 9, 14),
    (2, 10, 19), (2, 11, 5), (2, 13, 8), (2, 14, 5), (2, 15, 18), (2, 16, 11),
    (2, 17, 5), (2, 18, 5), (2, 19, 15), (2, 20, 0), (2, 22, 0), (2, 23, 0),
    (3, 0, 7), (3, 1, 7), (3, 2, 14), (3, 3, 14), (3, 4, 4), (3, 5, 16),
    (3, 6, 16), (3, 7, 24), (3, 8, 24), (3, 9, 10), (3, 10, 1), (3, 11, 7),
    (3, 12, 15), (3, 13, 6), (3, 14, 10), (3, 15, 26), (3, 16, 8), (3, 17, 18),
    (3, 18, 21), (3, 19, 14), (3, 20, 1), (3, 23, 0),
];

#[rustfmt::skip]
const WIFI_Z54_R1_2_ENTRIES: [(u8, u8, i16); 86] = [
    (0, 0, 40), (0, 4, 22), (0, 6, 49), (0, 7, 23), (0, 8, 43), (0, 12, 1),
    (0, 13, 0), (1, 0, 50), (1, 1, 1), (1, 4, 48), (1, 5, 35), (1, 8, 13),
    (1, 10, 30), (1, 13, 0), (1, 14, 0), (2, 0, 39), (2, 1, 50), (2, 4, 4),
    (2, 6, 2), (2, 11, 49), (2, 14, 0), (2, 15, 0), (3, 0, 33), (3, 3, 38),
    (3, 4, 37), (3, 7, 4), (3, 8, 1), (3, 15, 0), (3, 16, 0), (4, 0, 45),
    (4, 4, 0), (4, 5, 22), (4, 8, 20), (4, 9, 42), (4, 16, 0), (4, 17, 0),
    (5, 0, 51), (5, 3, 48), (5, 4, 35), (5, 8, 44), (5, 10, 18), (5, 17, 0),
    (5, 18, 0), (6, 0, 47), (6, 1, 11), (6, 5, 17), (6, 8, 51), (6, 12, 0),
    (6, 18, 0), (6, 19, 0), (7, 0, 5), (7, 2, 25), (7, 4, 6), (7, 6, 45),
    (7, 8, 13), (7, 9, 40), (7, 19, 0), (7, 20, 0), (8, 0, 33), (8, 3, 34),
    (8, 4, 24), (8, 8, 23), (8, 11, 46), (8, 20, 0), (8, 21, 0), (9, 0, 1),
    (9, 2, 27), (9, 4, 1), (9, 8, 38), (9, 10, 44), (9, 21, 0), (9, 22, 0),
    (10, 1, 18), (10, 4, 23), (10, 7, 8), (10, 8, 0), (10, 9, 35), (10, 22, 0),
    (10, 23, 0), (11, 0, 49), (11, 2, 17), (11, 4, 30), (11, 8, 34), (11, 11, 19),
    (11, 12, 1), (11, 23, 0),
];

#[rustfmt::skip]
const WIFI_Z54_R2_3_ENTRIES: [(u8, u8, i16); 88] = [
    (0, 0, 39), (0, 1, 31), (0, 2, 22), (0, 3, 43), (0, 5, 40), (0, 6, 4),
    (0, 8, 11), (0, 11, 50), (0, 15, 6), (0, 16, 1), (0, 17, 0), (1, 0, 25),
    (1, 1, 52), (1, 2, 41), (1, 3, 2), (1, 4, 6), (1, 6, 14), (1, 8, 34),
    (1, 12, 24), (1, 14, 37), (1, 17, 0), (1, 18, 0), (2, 0, 43), (2, 1, 31),
    (2, 2, 29), (2, 3, 0), (2, 4, 21), (2, 6, 28), (2, 9, 2), (2, 12, 7),
    (2, 14, 17), (2, 18, 0), (2, 19, 0), (3, 0, 20), (3, 1, 33), (3, 2, 48),
    (3, 4, 4), (3, 5, 13), (3, 7, 26), (3, 10, 22), (3, 13, 46), (3, 14, 42),
    (3, 19, 0), (3, 20, 0), (4, 0, 45), (4, 1, 7), (4, 2, 18), (4, 3, 51),
    (4, 4, 12), (4, 5, 25), (4, 9, 50), (4, 12, 5), (4, 16, 0), (4, 20, 0),
    (4, 21, 0), (5, 0, 35), (5, 1, 40), (5, 2, 32), (5, 3, 16), (5, 4, 5),
    (5, 7, 18), (5, 10, 43), (5, 11, 51), (5, 13, 32), (5, 21, 0), (5, 22, 0),
    (6, 0, 9), (6, 1, 24), (6, 2, 13), (6, 3, 22), (6, 4, 28), (6, 7, 37),
    (6, 10, 25), (6, 13, 52), (6, 15, 13), (6, 22, 0), (6, 23, 0), (7, 0, 32),
    (7, 1, 22), (7, 2, 4), (7, 3, 21), (7, 4, 16), (7, 8, 27), (7, 9, 28),
    (7, 11, 38), (7, 15, 8), (7, 16, 1), (7, 23, 0),
];

#[rustfmt::skip]
const WIFI_Z54_R3_4_ENTRIES: [(u8, u8, i16); 88] = [
    (0, 0, 39), (0, 1, 40), (0, 2, 51), (0, 3, 41), (0, 4, 3), (0, 5, 29),
    (0, 6, 8), (0, 7, 36), (0, 9, 14), (0, 11, 6), (0, 13, 33), (0, 15, 11),
    (0, 17, 4), (0, 18, 1), (0, 19, 0), (1, 0, 48), (1, 1, 21), (1, 2, 47),
    (1, 3, 9), (1, 4, 48), (1, 5, 35), (1, 6, 51), (1, 8, 38), (1, 10, 28),
    (1, 12, 34), (1, 14, 50), (1, 16, 50), (1, 19, 0), (1, 20, 0), (2, 0, 30),
    (2, 1, 39), (2, 2, 28), (2, 3, 42), (2, 4, 50), (2, 5, 39), (2, 6, 5),
    (2, 7, 17), (2, 9, 6), (2, 11, 18), (2, 13, 20), (2, 15, 15), (2, 17, 40),
    (2, 20, 0), (2, 21, 0), (3, 0, 29), (3, 1, 0), (3, 2, 1), (3, 3, 43),
    (3, 4, 36), (3, 5, 30), (3, 6, 47), (3, 8, 49), (3, 10, 47), (3, 12, 3),
    (3, 14, 35), (3, 16, 34), (3, 18, 0), (3, 21, 0), (3, 22, 0), (4, 0, 1),
    (4, 1, 32), (4, 2, 11), (4, 3, 23), (4, 4, 10), (4, 5, 44), (4, 6, 12),
    (4, 7, 7), (4, 9, 48), (4, 11, 4), (4, 13, 9), (4, 15, 17), (4, 17, 16),
    (4, 22, 0), (4, 23, 0), (5, 0, 13), (5, 1, 7), (5, 2, 15), (5, 3, 47),
    (5, 4, 23), (5, 5, 16), (5, 6, 47), (5, 8, 43), (5, 10, 29), (5, 12, 52),
    (5, 14, 2), (5, 16, 53), (5, 18, 1), (5, 23, 0),
];

#[rustfmt::skip]
const WIFI_Z54_R5_6_ENTRIES: [(u8, u8, i16); 85] = [
    (0, 0, 48), (0, 1, 29), (0, 2, 37), (0, 3, 52), (0, 4, 2), (0, 5, 16),
    (0, 6, 6), (0, 7, 14), (0, 8, 53), (0, 9, 31), (0, 10, 34), (0, 11, 5),
    (0, 12, 18), (0, 13, 42), (0, 14, 53), (0, 15, 31), (0, 16, 45), (0, 18, 46),
    (0, 19, 52), (0, 20, 1), (0, 21, 0), (1, 0, 17), (1, 1, 4), (1, 2, 30),
    (1, 3, 7), (1, 4, 43), (1, 5, 11), (1, 6, 24), (1, 7, 6), (1, 8, 14),
    (1, 9, 21), (1, 10, 6), (1, 11, 39), (1, 12, 17), (1, 13, 40), (1, 14, 47),
    (1, 15, 7), (1, 16, 15), (1, 17, 41), (1, 18, 19), (1, 21, 0), (1, 22, 0),
    (2, 0, 7), (2, 1, 2), (2, 2, 51), (2, 3, 31), (2, 4, 46), (2, 5, 23),
    (2, 6, 16), (2, 7, 11), (2, 8, 53), (2, 9, 40), (2, 10, 10), (2, 11, 7),
    (2, 12, 46), (2, 13, 53), (2, 14, 33), (2, 15, 35), (2, 17, 25), (2, 18, 35),
    (2, 19, 38), (2, 20, 0), (2, 22, 0), (2, 23, 0), (3, 0, 19), (3, 1, 48),
    (3, 2, 41), (3, 3, 1), (3, 4, 10), (3, 5, 7), (3, 6, 36), (3, 7, 47),
    (3, 8, 5), (3, 9, 29), (3, 10, 52), (3, 11, 52), (3, 12, 31), (3, 13, 10),
    (3, 14, 26), (3, 15, 6), (3, 16, 3), (3, 17, 2), (3, 19, 51), (3, 20, 1),
    (3, 23, 0),
];

#[rustfmt::skip]
const WIFI_Z81_R1_2_ENTRIES: [(u8, u8, i16); 86] = [
    (0, 0, 57), (0, 4, 50), (0, 6, 11), (0, 8, 50), (0, 10, 79), (0, 12, 1),
    (0, 13, 0), (1, 0, 3), (1, 2, 28), (1, 4, 0), (1, 8, 55), (1, 9, 7),
    (1, 13, 0), (1, 14, 0), (2, 0, 30), (2, 4, 24), (2, 5, 37), (2, 8, 56),
    (2, 9, 14), (2, 14, 0), (2, 15, 0), (3, 0, 62), (3, 1, 53), (3, 4, 53),
    (3, 7, 3), (3, 8, 35), (3, 15, 0), (3, 16, 0), (4, 0, 40), (4, 3, 20),
    (4, 4, 66), (4, 7, 22), (4, 8, 28), (4, 16, 0), (4, 17, 0), (5, 0, 0),
    (5, 4, 8), (5, 6, 42), (5, 8, 50), (5, 11, 8), (5, 17, 0), (5, 18, 0),
    (6, 0, 69), (6, 1, 79), (6, 2, 79), (6, 6, 56), (6, 8, 52), (6, 12, 0),
    (6, 18, 0), (6, 19, 0), (7, 0, 65), (7, 4, 38), (7, 5, 57), (7, 8, 72),
    (7, 10, 27), (7, 19, 0), (7, 20, 0), (8, 0, 64), (8, 4, 14), (8, 5, 52),
    (8, 8, 30), (8, 11, 32), (8, 20, 0), (8, 21, 0), (9, 1, 45), (9, 3, 70),
    (9, 4, 0), (9, 8, 77), (9, 9, 9), (9, 21, 0), (9, 22, 0), (10, 0, 2),
    (10, 1, 56), (10, 3, 57), (10, 4, 35), (10, 10, 12), (10, 22, 0), (10, 23, 0),
    (11, 0, 24), (11, 2, 61), (11, 4, 60), (11, 7, 27), (11, 8, 51), (11, 11, 16),
    (11, 12, 1), (11, 23, 0),
];

#[rustfmt::skip]
const WIFI_Z81_R2_3_ENTRIES: [(u8, u8, i16); 88] = [
    (0, 0, 61), (0, 1, 75), (0, 2, 4), (0, 3, 63), (0, 4, 56), (0, 11, 8),
    (0, 13, 2), (0, 14, 17), (0, 15, 25), (0, 16, 1), (0, 17, 0), (1, 0, 56),
    (1, 1, 74), (1, 2, 77), (1, 3, 20), (1, 7, 64), (1, 8, 24), (1, 9, 4),
    (1, 10, 67), (1, 12, 7), (1, 17, 0), (1, 18, 0), (2, 0, 28), (2, 1, 21),
    (2, 2, 68), (2, 3, 10), (2, 4, 7), (2, 5, 14), (2, 6, 65), (2, 10, 23),
    (2, 14, 75), (2, 18, 0), (2, 19, 0), (3, 0, 48), (3, 1, 38), (3, 2, 43),
    (3, 3, 78), (3, 4, 76), (3, 9, 5), (3, 10, 36), (3, 12, 15), (3, 13, 72),
    (3, 19, 0), (3, 20, 0), (4, 0, 40), (4, 1, 2), (4, 2, 53), (4, 3, 25),
    (4, 5, 52), (4, 6, 62), (4, 8, 20), (4, 11, 44), (4, 16, 0), (4, 20, 0),
    (4, 21, 0), (5, 0, 69), (5, 1, 23), (5, 2, 64), (5, 3, 10), (5, 4, 22),
    (5, 6, 21), (5, 12, 68), (5, 13, 23), (5, 14, 29), (5, 21, 0), (5, 22, 0),
    (6, 0, 12), (6, 1, 0), (6, 2, 68), (6, 3, 20), (6, 4, 55), (6, 5, 61),
    (6, 7, 40), (6, 11, 52), (6, 15, 44), (6, 22, 0), (6, 23, 0), (7, 0, 58),
    (7, 1, 8), (7, 2, 34), (7, 3, 64), (7, 4, 78), (7, 7, 11), (7, 8, 78),
    (7, 9, 24), (7, 15, 58), (7, 16, 1), (7, 23, 0),
];

#[rustfmt::skip]
const WIFI_Z81_R3_4_ENTRIES: [(u8, u8, i16); 85] = [
    (0, 0, 48), (0, 1, 29), (0, 2, 28), (0, 3, 39), (0, 4, 9), (0, 5, 61),
    (0, 9, 63), (0, 10, 45), (0, 11, 80), (0, 15, 37), (0, 16, 32), (0, 17, 22),
    (0, 18, 1), (0, 19, 0), (1, 0, 4), (1, 1, 49), (1, 2, 42), (1, 3, 48),
    (1, 4, 11), (1, 5, 30), (1, 9, 49), (1, 10, 17), (1, 11, 41), (1, 12, 37),
    (1, 13, 15), (1, 15, 54), (1, 19, 0), (1, 20, 0), (2, 0, 35), (2, 1, 76),
    (2, 2, 78), (2, 3, 51), (2, 4, 37), (2, 5, 35), (2, 6, 21), (2, 8, 17),
    (2, 9, 64), (2, 13, 59), (2, 14, 7), (2, 17, 32), (2, 20, 0), (2, 21, 0),
    (3, 0, 9), (3, 1, 65), (3, 2, 44), (3, 3, 9), (3, 4, 54), (3, 5, 56),
    (3, 6, 73), (3, 7, 34), (3, 8, 42), (3, 12, 35), (3, 16, 46), (3, 17, 39),
    (3, 18, 0), (3, 21, 0), (3, 22, 0), (4, 0, 3), (4, 1, 62), (4, 2, 7),
    (4, 3, 80), (4, 4, 68), (4, 5, 26), (4, 7, 80), (4, 8, 55), (4, 10, 36),
    (4, 12, 26), (4, 14, 9), (4, 16, 72), (4, 22, 0), (4, 23, 0), (5, 0, 26),
    (5, 1, 75), (5, 2, 33), (5, 3, 21), (5, 4, 69), (5, 5, 59), (5, 6, 3),
    (5, 7, 38), (5, 11, 35), (5, 13, 62), (5, 14, 36), (5, 15, 26), (5, 18, 1),
    (5, 23, 0),
];

#[rustfmt::skip]
const WIFI_Z81_R5_6_ENTRIES: [(u8, u8, i16); 79] = [
    (0, 0, 13), (0, 1, 48), (0, 2, 80), (0, 3, 66), (0, 4, 4), (0, 5, 74),
    (0, 6, 7), (0, 7, 30), (0, 8, 76), (0, 9, 52), (0, 10, 37), (0, 11, 60),
    (0, 13, 49), (0, 14, 73), (0, 15, 31), (0, 16, 74), (0, 17, 73), (0, 18, 23),
    (0, 20, 1), (0, 21, 0), (1, 0, 69), (1, 1, 63), (1, 2, 74), (1, 3, 56),
    (1, 4, 64), (1, 5, 77), (1, 6, 57), (1, 7, 65), (1, 8, 6), (1, 9, 16),
    (1, 10, 51), (1, 12, 64), (1, 14, 68), (1, 15, 9), (1, 16, 48), (1, 17, 62),
    (1, 18, 54), (1, 19, 27), (1, 21, 0), (1, 22, 0), (2, 0, 51), (2, 1, 15),
    (2, 2, 0), (2, 3, 80), (2, 4, 24), (2, 5, 25), (2, 6, 42), (2, 7, 54),
    (2, 8, 44), (2, 9, 71), (2, 10, 71), (2, 11, 9), (2, 12, 67), (2, 13, 35),
    (2, 15, 58), (2, 17, 29), (2, 19, 53), (2, 20, 0), (2, 22, 0), (2, 23, 0),
    (3, 0, 16), (3, 1, 29), (3, 2, 36), (3, 3, 41), (3, 4, 44), (3, 5, 56),
    (3, 6, 59), (3, 7, 37), (3, 8, 50), (3, 9, 24), (3, 11, 65), (3, 12, 4),
    (3, 13, 65), (3, 14, 52), (3, 16, 4), (3, 18, 73), (3, 19, 52), (3, 20, 1),
    (3, 23, 0),
];

/// All 12 IEEE 802.11 LDPC prototype matrices, indexed by $(Z, R)$.
///
/// Looked up by [`wifi_ldpc_matrix`]; see the module-level sourcing note for
/// provenance and cross-validation.
pub(crate) const WIFI_LDPC_MATRICES: [WifiLdpcMatrix; 12] = [
    WifiLdpcMatrix {
        z: 27,
        rate_num: 1,
        rate_den: 2,
        row_blocks: 12,
        col_blocks: 24,
        entries: &WIFI_Z27_R1_2_ENTRIES,
    },
    WifiLdpcMatrix {
        z: 27,
        rate_num: 2,
        rate_den: 3,
        row_blocks: 8,
        col_blocks: 24,
        entries: &WIFI_Z27_R2_3_ENTRIES,
    },
    WifiLdpcMatrix {
        z: 27,
        rate_num: 3,
        rate_den: 4,
        row_blocks: 6,
        col_blocks: 24,
        entries: &WIFI_Z27_R3_4_ENTRIES,
    },
    WifiLdpcMatrix {
        z: 27,
        rate_num: 5,
        rate_den: 6,
        row_blocks: 4,
        col_blocks: 24,
        entries: &WIFI_Z27_R5_6_ENTRIES,
    },
    WifiLdpcMatrix {
        z: 54,
        rate_num: 1,
        rate_den: 2,
        row_blocks: 12,
        col_blocks: 24,
        entries: &WIFI_Z54_R1_2_ENTRIES,
    },
    WifiLdpcMatrix {
        z: 54,
        rate_num: 2,
        rate_den: 3,
        row_blocks: 8,
        col_blocks: 24,
        entries: &WIFI_Z54_R2_3_ENTRIES,
    },
    WifiLdpcMatrix {
        z: 54,
        rate_num: 3,
        rate_den: 4,
        row_blocks: 6,
        col_blocks: 24,
        entries: &WIFI_Z54_R3_4_ENTRIES,
    },
    WifiLdpcMatrix {
        z: 54,
        rate_num: 5,
        rate_den: 6,
        row_blocks: 4,
        col_blocks: 24,
        entries: &WIFI_Z54_R5_6_ENTRIES,
    },
    WifiLdpcMatrix {
        z: 81,
        rate_num: 1,
        rate_den: 2,
        row_blocks: 12,
        col_blocks: 24,
        entries: &WIFI_Z81_R1_2_ENTRIES,
    },
    WifiLdpcMatrix {
        z: 81,
        rate_num: 2,
        rate_den: 3,
        row_blocks: 8,
        col_blocks: 24,
        entries: &WIFI_Z81_R2_3_ENTRIES,
    },
    WifiLdpcMatrix {
        z: 81,
        rate_num: 3,
        rate_den: 4,
        row_blocks: 6,
        col_blocks: 24,
        entries: &WIFI_Z81_R3_4_ENTRIES,
    },
    WifiLdpcMatrix {
        z: 81,
        rate_num: 5,
        rate_den: 6,
        row_blocks: 4,
        col_blocks: 24,
        entries: &WIFI_Z81_R5_6_ENTRIES,
    },
];

/// Look up the raw 802.11 LDPC prototype matrix for `(z, rate_num,
/// rate_den)`.
///
/// # Arguments
///
/// * `z` - Lifting size; must be one of $\{27, 54, 81\}$.
/// * `rate_num`, `rate_den` - Code rate fraction; must be one of the four
///   802.11 rates $1/2, 2/3, 3/4, 5/6$ (in exactly this reduced form).
///
/// # Returns
///
/// A reference to the matching [`WifiLdpcMatrix`].
///
/// # Errors
///
/// Returns [`FecError::InvalidParam`] if `(z, rate_num, rate_den)` does not
/// match one of the 12 supported combinations.
pub fn wifi_ldpc_matrix(
    z: usize,
    rate_num: usize,
    rate_den: usize,
) -> Result<&'static WifiLdpcMatrix, FecError> {
    WIFI_LDPC_MATRICES
        .iter()
        .find(|m| m.z == z && m.rate_num == rate_num && m.rate_den == rate_den)
        .ok_or(FecError::InvalidParam(
            "z/rate is not one of the 12 supported 802.11 LDPC (Z, R) combinations",
        ))
}

/// Build a ready-to-use [`QcLdpcEncoder`] for the 802.11 LDPC code at
/// `(z, rate_num, rate_den)`, using the real Annex R/F shift matrix.
///
/// The resulting encoder handles exactly one full, unshortened, unpunctured
/// codeword: `K` info bits in, `N = 24*z` coded bits out. See the module
/// doc for what is explicitly out of scope (shortening, puncturing).
///
/// # Errors
///
/// Propagates [`FecError::InvalidParam`] from [`wifi_ldpc_matrix`] or
/// [`crate::qc_ldpc::QcLdpcParams::from_raw_edges`].
///
/// # Examples
///
/// ```
/// use syndrome::wifi_ldpc_tables::wifi_ldpc_encoder;
///
/// let enc = wifi_ldpc_encoder(27, 1, 2).unwrap();
/// assert_eq!(enc.codeword_bit_count(), 24 * 27);
/// ```
pub fn wifi_ldpc_encoder(
    z: usize,
    rate_num: usize,
    rate_den: usize,
) -> Result<QcLdpcEncoder, FecError> {
    let m = wifi_ldpc_matrix(z, rate_num, rate_den)?;
    QcLdpcEncoder::from_raw_edges(m.entries, m.row_blocks, m.col_blocks, m.z)
}

/// Build a ready-to-use [`QcLdpcDecoder`] for the 802.11 LDPC code at
/// `(z, rate_num, rate_den)`, using the real Annex R/F shift matrix.
///
/// # Arguments
///
/// * `z`, `rate_num`, `rate_den` - Selects the (Z, R) combination; see
///   [`wifi_ldpc_matrix`].
/// * `offset_beta` - Offset correction $\beta$ for the LOMS min-sum update.
///
/// # Errors
///
/// Propagates [`FecError::InvalidParam`] from [`wifi_ldpc_matrix`] or
/// [`crate::qc_ldpc::QcLdpcParams::from_raw_edges`].
///
/// # Examples
///
/// ```
/// use syndrome::wifi_ldpc_tables::wifi_ldpc_decoder;
///
/// let dec = wifi_ldpc_decoder(27, 1, 2, 0.25).unwrap();
/// assert_eq!(dec.variable_node_count(), 24 * 27);
/// ```
pub fn wifi_ldpc_decoder(
    z: usize,
    rate_num: usize,
    rate_den: usize,
    offset_beta: f32,
) -> Result<QcLdpcDecoder, FecError> {
    let m = wifi_ldpc_matrix(z, rate_num, rate_den)?;
    QcLdpcDecoder::from_raw_edges(m.entries, m.row_blocks, m.col_blocks, m.z, offset_beta)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// All 12 (Z, rate) combinations must be present exactly once, with the
    /// row/col block counts matching `WifiLdpcParams` (see `src/wifi.rs`).
    #[test]
    fn all_12_combinations_present() {
        let zs = [27usize, 54, 81];
        let rates = [(1usize, 2usize), (2, 3), (3, 4), (5, 6)];
        let expected_rows = [(1, 2, 12), (2, 3, 8), (3, 4, 6), (5, 6, 4)];

        assert_eq!(WIFI_LDPC_MATRICES.len(), 12);
        for &z in &zs {
            for &(rn, rd) in &rates {
                let m = wifi_ldpc_matrix(z, rn, rd).unwrap_or_else(|_| {
                    panic!("missing 802.11 LDPC matrix for Z={z} rate={rn}/{rd}")
                });
                assert_eq!(m.z, z);
                assert_eq!(m.col_blocks, 24);
                let expected = expected_rows
                    .iter()
                    .find(|&&(a, b, _)| a == rn && b == rd)
                    .unwrap()
                    .2;
                assert_eq!(m.row_blocks, expected, "row_blocks for rate {rn}/{rd}");
            }
        }
    }

    /// Every stored shift value must be `< z` for its matrix (802.11 shifts
    /// are literal, not modulo-reduced at lookup time).
    #[test]
    fn all_shifts_within_z() {
        for m in WIFI_LDPC_MATRICES.iter() {
            for &(r, c, s) in m.entries {
                assert!(
                    (r as usize) < m.row_blocks,
                    "row {r} out of range for Z={} rate={}/{}",
                    m.z,
                    m.rate_num,
                    m.rate_den
                );
                assert!(
                    (c as usize) < m.col_blocks,
                    "col {c} out of range for Z={} rate={}/{}",
                    m.z,
                    m.rate_num,
                    m.rate_den
                );
                assert!(
                    (s as usize) < m.z,
                    "shift {s} >= Z={} for rate {}/{}",
                    m.z,
                    m.rate_num,
                    m.rate_den
                );
            }
        }
    }

    /// Building an encoder/decoder pair from every one of the 12 tables
    /// must succeed and report the expected dimensions.
    #[test]
    fn encoder_decoder_dimensions_all_12() {
        for m in WIFI_LDPC_MATRICES.iter() {
            let enc = wifi_ldpc_encoder(m.z, m.rate_num, m.rate_den).unwrap_or_else(|e| {
                panic!(
                    "encoder build failed for Z={} rate={}/{}: {e}",
                    m.z, m.rate_num, m.rate_den
                )
            });
            let dec = wifi_ldpc_decoder(m.z, m.rate_num, m.rate_den, 0.25).unwrap_or_else(|e| {
                panic!(
                    "decoder build failed for Z={} rate={}/{}: {e}",
                    m.z, m.rate_num, m.rate_den
                )
            });
            let n = 24 * m.z;
            let k = (24 - m.row_blocks) * m.z;
            assert_eq!(enc.codeword_bit_count(), n);
            assert_eq!(enc.info_bit_count(), k);
            assert_eq!(dec.variable_node_count(), n);
            assert_eq!(dec.check_node_count(), m.row_blocks * m.z);
        }
    }

    /// Unsupported (z, rate) combinations must be a typed error, never a
    /// panic.
    #[test]
    fn unsupported_combination_is_typed_error() {
        assert!(wifi_ldpc_matrix(27, 1, 3).is_err());
        assert!(wifi_ldpc_matrix(100, 1, 2).is_err());
        assert!(wifi_ldpc_encoder(27, 1, 3).is_err());
        assert!(wifi_ldpc_decoder(27, 1, 3, 0.25).is_err());
    }
}
