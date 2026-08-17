//! Extended binary Golay code $G(24,12,8)$: encoder and a guaranteed
//! 3-error-correcting decoder.
//!
//! # History and motivation
//!
//! The extended Golay code is one of the most celebrated objects in coding
//! theory: a *perfect-adjacent*, highly symmetric `[24,12,8]` code whose
//! automorphism group is (twice) the Mathieu group $M_{24}$. It flew on
//! Voyager 1 and 2 to compress and protect the color imagery returned from
//! Jupiter and Saturn (the shorter, perfect $(23,12,7)$ code is the
//! "punctured" version of the code implemented here). Variants and relatives
//! are still specified today in CCSDS telecommand link protocols, in
//! MIL-STD-188-141 (ALE) for HF radio, and the closely related non-extended
//! $(23,12,7)$ Golay code underlies POCSAG radio paging.
//!
//! # Construction
//!
//! The code is systematic: $G = [I_{12} \mid B]$, with parity-check matrix
//! $H = [B^\mathsf{T} \mid I_{12}]$ (this identity holds for *any* $B$,
//! since $G H^\mathsf{T} = I_{12} B + B I_{12} = B + B = 0$ over
//! $\mathrm{GF}(2)$ regardless of whether $B$ happens to be symmetric).
//!
//! $B$ is the classical $12 \times 12$ "bordered circulant" built from the
//! quadratic residues of $11$: $QR_{11} = \lbrace 1,3,4,5,9\rbrace$. Index rows/columns
//! $0..10$ by $\mathrm{GF}(11)$ and index $11$ by a distinguished "point at
//! infinity". Then
//!
//! $$
//! B_{ij} = \begin{cases}
//!   1 & i = 11 \text{ or } j = 11, \text{ except } B_{11,11} = 0 \\\\
//!   1 & i = j \ne 11 \\\\
//!   1 & i \ne j,\ (i - j) \bmod 11 \in QR_{11} \\\\
//!   0 & \text{otherwise}
//! \end{cases}
//! $$
//!
//! This $B$ is computed programmatically from `QR_11` in `build_b_rows`
//! rather than hardcoded, and is verified by the test suite below to
//! produce a genuine `[24,12,8]` self-dual code (weight enumerator
//! $1 + 759x^8 + 2576x^{12} + 759x^{16} + x^{24}$).
//!
//! Schematically, $B$'s "bordered circulant" shape is an $11\times11$
//! QR-difference core (built by the private `quadratic_residues_mod_11`) framed by an
//! all-ones border row/column (except the corner), matching the case split
//! in `build_b_rows` line for line:
//!
//! ```text
//!          0   1   2  ...  10   ∞
//!        +---------------------+---+
//!    0   |                     | 1 |
//!    1   |   11x11 core:       | 1 |
//!    .   |   B[i][j] = 1 iff   | . |   (border column: all 1s)
//!    .   |   i==j, or          | . |
//!   10   |   (i-j) mod 11 ∈ QR | 1 |
//!        +---------------------+---+
//!    ∞   |  1   1   1  ...  1  | 0 |   (border row: all 1s except corner)
//!        +---------------------+---+
//! ```
//!
//! Note on symmetry: since $-1$ is a quadratic *non*-residue mod $11$
//! ($11 \equiv 3 \pmod 4$), the off-diagonal QR-difference structure is
//! provably a tournament (for every $i \ne j$ exactly one of $B_{ij}$,
//! $B_{ji}$ is $1$), so this particular $B$ is **not** symmetric — the
//! tests verify self-duality directly ($G G^\mathsf{T} = 0$) and use
//! $H = [B^\mathsf{T} \mid I_{12}]$ rather than assuming $B = B^\mathsf{T}$.
//!
//! # Decoding
//!
//! Minimum distance $d = 8$ guarantees correction of
//! $t = \lfloor (d-1)/2 \rfloor = 3$ errors. The code meets the sphere-packing
//! bound with equality up to a factor of $2$ (it is "quasi-perfect" /
//! nearly perfect for $t=3$):
//!
//! $$ \sum_{i=0}^{3} \binom{24}{i} = 1 + 24 + 276 + 2024 = 2325, $$
//!
//! which is exactly half of $2^{12} = 4096$ syndromes; the coset leaders of
//! weight $0..3$ are unique per syndrome (any two would differ by a
//! codeword of weight $\le 6 < 8$, impossible), so decoding is unambiguous
//! for up to $3$ errors. The remaining $4096 - 2325 = 1771$ syndromes
//! correspond to cosets whose minimum-weight representatives all have
//! weight $4$ — and each such coset contains exactly $6$ equally-valid
//! weight-4 leaders ($1771 \times 6 = \binom{24}{4} = 10626$), so a weight-4
//! error is *detectable* (non-correctable syndrome) but never silently
//! decoded into a wrong codeword.
//!
//! # Performance
//!
//! [`GolayCode::new`] precomputes two flat lookup tables once:
//! * `encode_table`: $2^{12}$ entries of `u32` (message → codeword), 16 KiB.
//! * `syndrome_table`: $2^{12}$ entries of `u32` (syndrome → error pattern,
//!   or a sentinel for detected-but-uncorrectable), 16 KiB.
//!
//! After construction, [`GolayCode::encode`] and [`GolayCode::decode`] are
//! $O(1)$: a bit-packing pass over the fixed-size input slice, one table
//! lookup, and (for decode) a 12-iteration branch-free syndrome computation
//! over `u32` popcounts. Neither performs any heap allocation.
//!
//! # Examples
//!
//! ```
//! use syndrome::GolayCode;
//!
//! let golay = GolayCode::new();
//! let info: [u8; 12] = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0];
//! let mut codeword = [0u8; 24];
//! golay.encode(&info, &mut codeword).unwrap();
//!
//! // Introduce 3 bit errors (the maximum guaranteed-correctable count).
//! codeword[0] ^= 1;
//! codeword[5] ^= 1;
//! codeword[19] ^= 1;
//!
//! let mut decoded = [0u8; 12];
//! let corrected = golay.decode(&codeword, &mut decoded).unwrap();
//! assert_eq!(corrected, 3);
//! assert_eq!(decoded, info);
//! ```

use crate::error::FecError;

/// Sentinel stored in `syndrome_table` for syndromes whose coset minimum
/// weight is $\ge 4$: detectable, but not uniquely correctable.
const UNCORRECTABLE: u32 = u32::MAX;

/// Number of information bits, $k = 12$.
pub const GOLAY_K: usize = 12;
/// Number of codeword bits, $n = 24$.
pub const GOLAY_N: usize = 24;
/// Minimum distance, $d = 8$.
pub const GOLAY_D: usize = 8;

/// Quadratic residues of $11$ as a bitset indexed $0..11$: `is_qr[r]` is
/// `true` iff $r$ is a nonzero square mod $11$.
///
/// # Returns
///
/// An 11-entry boolean table with `is_qr[r] == true` for
/// $r \in \lbrace 1,3,4,5,9\rbrace$.
fn quadratic_residues_mod_11() -> [bool; 11] {
    let mut is_qr = [false; 11];
    for i in 1..11usize {
        is_qr[(i * i) % 11] = true;
    }
    is_qr
}

/// Build the bordered $12 \times 12$ matrix $B$ from the quadratic residues
/// of $11$ (see module docs for the exact construction).
///
/// # Returns
///
/// `b[i]` is row $i$ of $B$, packed as the low 12 bits of a `u16`.
fn build_b_rows() -> [u16; 12] {
    let is_qr = quadratic_residues_mod_11();
    let mut b = [0u16; 12];
    for i in 0..11usize {
        for j in 0..11usize {
            let bit = if i == j {
                true
            } else {
                is_qr[(i + 11 - j) % 11]
            };
            if bit {
                b[i] |= 1 << j;
            }
        }
        // Border column: B[i][11] = 1 for i != 11.
        b[i] |= 1 << 11;
    }
    // Border row: B[11][j] = 1 for j != 11, B[11][11] = 0.
    b[11] = (1u16 << 11) - 1;
    b
}

/// Transpose the $12 \times 12$ bit matrix $B$.
///
/// # Arguments
///
/// * `b` - Rows of $B$, low 12 bits each.
///
/// # Returns
///
/// Rows of $B^\mathsf{T}$, low 12 bits each.
fn transpose_b(b: &[u16; 12]) -> [u16; 12] {
    let mut bt = [0u16; 12];
    for i in 0..12usize {
        for k in 0..12usize {
            if (b[i] >> k) & 1 == 1 {
                bt[k] |= 1 << i;
            }
        }
    }
    bt
}

/// Build the 12 rows of the systematic generator matrix $G = [I_{12}\mid B]$
/// as 24-bit codewords (one basis codeword per information bit).
///
/// # Arguments
///
/// * `b` - Rows of $B$, low 12 bits each.
///
/// # Returns
///
/// `g[i]` is row $i$ of $G$ packed into the low 24 bits of a `u32`: bits
/// `0..12` are the identity part, bits `12..24` are row $i$ of $B$.
fn build_g_rows(b: &[u16; 12]) -> [u32; 12] {
    let mut g = [0u32; 12];
    for i in 0..12usize {
        g[i] = (1u32 << i) | ((b[i] as u32) << 12);
    }
    g
}

/// Build the 12 rows of the parity-check matrix $H = [B^\mathsf{T} \mid
/// I_{12}]$ as 24-bit masks.
///
/// # Arguments
///
/// * `bt` - Rows of $B^\mathsf{T}$, low 12 bits each.
///
/// # Returns
///
/// `h[k]` is row $k$ of $H$ packed into the low 24 bits of a `u32`: bits
/// `0..12` are row $k$ of $B^\mathsf{T}$, bits `12..24` are the identity part.
fn build_h_rows(bt: &[u16; 12]) -> [u32; 12] {
    let mut h = [0u32; 12];
    for k in 0..12usize {
        h[k] = (bt[k] as u32) | (1u32 << (12 + k));
    }
    h
}

/// Compute the 12-bit syndrome $s = H e^\mathsf{T}$ of a 24-bit vector $e$
/// (an error pattern, or a full received word).
///
/// Branch-free: each syndrome bit is the parity (`count_ones() & 1`) of the
/// bitwise AND with the corresponding row of $H$.
///
/// # Arguments
///
/// * `h_rows` - Rows of $H$, packed as returned by [`build_h_rows`].
/// * `e` - 24-bit vector (low 24 bits significant).
///
/// # Returns
///
/// The 12-bit syndrome in the low bits of a `u32`.
#[inline]
fn syndrome_of(h_rows: &[u32; 12], e: u32) -> u32 {
    let mut s = 0u32;
    for k in 0..12usize {
        s |= ((e & h_rows[k]).count_ones() & 1) << k;
    }
    s
}

/// Pack a bit-per-byte slice (values `0`/`1`) into the low `n` bits of a
/// `u32`, slice index `0` mapping to bit `0`.
///
/// # Arguments
///
/// * `bits` - Slice of at least `n` bytes, each `0` or `1` (only the LSB of
///   each byte is consulted).
/// * `n` - Number of bits to pack (`<= 32`).
///
/// # Returns
///
/// The packed value in the low `n` bits of a `u32`.
#[inline]
fn bits_to_u32(bits: &[u8], n: usize) -> u32 {
    let mut v = 0u32;
    for i in 0..n {
        v |= ((bits[i] & 1) as u32) << i;
    }
    v
}

/// Unpack the low `n` bits of `v` into a bit-per-byte slice, bit `0` mapping
/// to slice index `0`.
///
/// # Arguments
///
/// * `v` - Value whose low `n` bits are unpacked.
/// * `n` - Number of bits to unpack (`<= 32`).
/// * `out` - Destination slice of at least `n` bytes.
#[inline]
fn u32_to_bits(v: u32, n: usize, out: &mut [u8]) {
    for i in 0..n {
        out[i] = ((v >> i) & 1) as u8;
    }
}

/// Extended binary Golay code $G(24,12,8)$ encoder/decoder.
///
/// Precomputes an encode table and a syndrome-based decode table once in
/// [`GolayCode::new`] ($\approx 16\thinspace \text{KiB} + 16\thinspace \text{KiB}$); afterwards
/// [`encode`](Self::encode) and [`decode`](Self::decode) are $O(1)$
/// table lookups with no heap allocation.
pub struct GolayCode {
    /// `encode_table[msg]` is the 24-bit codeword for the 12-bit message
    /// `msg`, for all $2^{12}$ messages.
    encode_table: Box<[u32; 4096]>,
    /// `syndrome_table[s]` is the weight-`<=3` error pattern with syndrome
    /// `s`, or [`UNCORRECTABLE`] if `s` is only reachable by weight-`>=4`
    /// errors.
    syndrome_table: Box<[u32; 4096]>,
    /// Rows of the parity-check matrix $H = [B^\mathsf{T}\mid I_{12}]$,
    /// used to compute the syndrome of a received word at decode time.
    h_rows: [u32; 12],
}

impl Default for GolayCode {
    fn default() -> Self {
        Self::new()
    }
}

impl GolayCode {
    /// Build the encode and syndrome-decode tables.
    ///
    /// Constructs $B$ from the quadratic residues of $11$, then:
    /// * fills `encode_table` by XOR-combining the 12 basis rows of
    ///   $G = [I_{12}\mid B]$ selected by each message's set bits;
    /// * fills `syndrome_table` by enumerating every error pattern of
    ///   Hamming weight $0..=3$ over 24 bits (`1 + 24 + 276 + 2024 = 2325`
    ///   patterns), computing its syndrome via $H = [B^\mathsf{T}\mid
    ///   I_{12}]$, and recording it as that syndrome's coset leader.
    ///
    /// This is the only place in the module that does more than $O(1)$
    /// work; it runs once and is not on any decode hot path.
    ///
    /// # Returns
    ///
    /// A ready-to-use [`GolayCode`] with both tables populated.
    ///
    /// # Examples
    ///
    /// ```
    /// use syndrome::GolayCode;
    /// let golay = GolayCode::new();
    /// let info = [0u8; 12];
    /// let mut codeword = [0u8; 24];
    /// golay.encode(&info, &mut codeword).unwrap();
    /// assert_eq!(codeword, [0u8; 24]);
    /// ```
    pub fn new() -> Self {
        let b = build_b_rows();
        let bt = transpose_b(&b);
        let g_rows = build_g_rows(&b);
        let h_rows = build_h_rows(&bt);

        let mut encode_table = Box::new([0u32; 4096]);
        for msg in 0..4096u32 {
            let mut cw = 0u32;
            for i in 0..12usize {
                if (msg >> i) & 1 == 1 {
                    cw ^= g_rows[i];
                }
            }
            debug_assert_eq!(cw & 0xFFF, msg, "G must be systematic: cw = [msg | parity]");
            encode_table[msg as usize] = cw;
        }

        let mut syndrome_table = Box::new([UNCORRECTABLE; 4096]);
        let mut record = |e: u32| {
            let s = syndrome_of(&h_rows, e);
            debug_assert_eq!(
                syndrome_table[s as usize], UNCORRECTABLE,
                "weight <= 3 coset leaders must be unique for a distance-8 code"
            );
            syndrome_table[s as usize] = e;
        };

        // Weight 0.
        record(0);
        // Weight 1.
        for b0 in 0..24u32 {
            record(1 << b0);
        }
        // Weight 2.
        for b0 in 0..24u32 {
            for b1 in (b0 + 1)..24u32 {
                record((1 << b0) | (1 << b1));
            }
        }
        // Weight 3.
        for b0 in 0..24u32 {
            for b1 in (b0 + 1)..24u32 {
                for b2 in (b1 + 1)..24u32 {
                    record((1 << b0) | (1 << b1) | (1 << b2));
                }
            }
        }

        GolayCode {
            encode_table,
            syndrome_table,
            h_rows,
        }
    }

    /// Encode a 12-bit message into a 24-bit extended Golay codeword.
    ///
    /// # Arguments
    ///
    /// * `info` - Exactly [`GOLAY_K`] (12) bytes, each `0` or `1`.
    /// * `out` - Exactly [`GOLAY_N`] (24) bytes; overwritten with the
    ///   systematic codeword (`out[0..12] == info`, `out[12..24]` is parity).
    ///
    /// # Errors
    ///
    /// Returns [`FecError::BufferTooSmall`] if `info.len() != `[`GOLAY_K`]
    /// or `out.len() != `[`GOLAY_N`].
    ///
    /// # Examples
    ///
    /// ```
    /// use syndrome::GolayCode;
    /// let golay = GolayCode::new();
    /// let info = [1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0];
    /// let mut codeword = [0u8; 24];
    /// golay.encode(&info, &mut codeword).unwrap();
    /// assert_eq!(&codeword[0..12], &info[..]);
    /// ```
    pub fn encode(&self, info: &[u8], out: &mut [u8]) -> Result<(), FecError> {
        if info.len() != GOLAY_K {
            return Err(FecError::BufferTooSmall {
                required: GOLAY_K,
                provided: info.len(),
            });
        }
        if out.len() != GOLAY_N {
            return Err(FecError::BufferTooSmall {
                required: GOLAY_N,
                provided: out.len(),
            });
        }
        let msg = bits_to_u32(info, GOLAY_K);
        let cw = self.encode_table[msg as usize];
        u32_to_bits(cw, GOLAY_N, out);
        Ok(())
    }

    /// Decode a (possibly corrupted) 24-bit word, correcting up to 3
    /// bit errors.
    ///
    /// # Arguments
    ///
    /// * `received` - Exactly [`GOLAY_N`] (24) bytes, each `0` or `1`.
    /// * `out` - Exactly [`GOLAY_K`] (12) bytes; overwritten with the
    ///   corrected 12-bit message on success.
    ///
    /// # Returns
    ///
    /// `Ok(n)` with the number of bits corrected, `0 <= n <= 3`.
    ///
    /// # Errors
    ///
    /// * [`FecError::BufferTooSmall`] if `received` or `out` has the wrong
    ///   length.
    /// * [`FecError::DecoderNotConverged`] if the syndrome indicates a
    ///   detected but non-correctable error pattern (coset minimum weight
    ///   $\ge 4$); the extended Golay code's weight-4 cosets each contain 6
    ///   equally valid leaders, so no attempt is made to guess one.
    ///
    /// # Examples
    ///
    /// ```
    /// use syndrome::GolayCode;
    /// let golay = GolayCode::new();
    /// let info = [0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0];
    /// let mut codeword = [0u8; 24];
    /// golay.encode(&info, &mut codeword).unwrap();
    /// codeword[3] ^= 1; // single-bit error
    ///
    /// let mut decoded = [0u8; 12];
    /// assert_eq!(golay.decode(&codeword, &mut decoded), Ok(1));
    /// assert_eq!(decoded, info);
    /// ```
    pub fn decode(&self, received: &[u8], out: &mut [u8]) -> Result<usize, FecError> {
        if received.len() != GOLAY_N {
            return Err(FecError::BufferTooSmall {
                required: GOLAY_N,
                provided: received.len(),
            });
        }
        if out.len() != GOLAY_K {
            return Err(FecError::BufferTooSmall {
                required: GOLAY_K,
                provided: out.len(),
            });
        }

        let r = bits_to_u32(received, GOLAY_N);
        let s = syndrome_of(&self.h_rows, r);
        let e = self.syndrome_table[s as usize];
        if e == UNCORRECTABLE {
            return Err(FecError::DecoderNotConverged);
        }

        let corrected = r ^ e;
        u32_to_bits(corrected & 0x0FFF, GOLAY_K, out);
        Ok(e.count_ones() as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny deterministic xorshift32 PRNG so tests are reproducible without
    /// pulling in a `rand` dependency.
    struct XorShift32(u32);
    impl XorShift32 {
        fn next(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            self.0 = x;
            x
        }
    }

    fn full_g_rows() -> [u32; 12] {
        let b = build_b_rows();
        build_g_rows(&b)
    }

    fn full_h_rows() -> [u32; 12] {
        let b = build_b_rows();
        let bt = transpose_b(&b);
        build_h_rows(&bt)
    }

    #[test]
    fn qr_set_is_1_3_4_5_9() {
        let is_qr = quadratic_residues_mod_11();
        let expected = [1, 3, 4, 5, 9];
        for r in 0..11usize {
            assert_eq!(is_qr[r], expected.contains(&r), "residue {r}");
        }
    }

    #[test]
    fn g_h_transpose_is_zero() {
        // G H^T = 0: every basis codeword is orthogonal to every parity row.
        let g = full_g_rows();
        let h = full_h_rows();
        for gi in g {
            for hk in h {
                assert_eq!((gi & hk).count_ones() % 2, 0, "G H^T must vanish");
            }
        }
    }

    #[test]
    fn code_is_self_dual() {
        // G G^T = 0: the code is self-orthogonal, and since dim = n/2 = 12,
        // self-orthogonal + full rank implies self-dual.
        let g = full_g_rows();
        for gi in g {
            for gj in g {
                assert_eq!((gi & gj).count_ones() % 2, 0, "G G^T must vanish");
            }
        }
    }

    #[test]
    fn every_generator_row_has_weight_at_least_8() {
        let g = full_g_rows();
        for (i, gi) in g.iter().enumerate() {
            assert!(
                gi.count_ones() >= 8,
                "row {i} has weight {}",
                gi.count_ones()
            );
        }
    }

    #[test]
    fn minimum_weight_of_all_codewords_is_8() {
        let golay = GolayCode::new();
        let mut min_weight = u32::MAX;
        let mut weight_counts = [0u32; 25];
        for msg in 0..4096u32 {
            let cw = golay.encode_table[msg as usize];
            let w = cw.count_ones();
            weight_counts[w as usize] += 1;
            if msg != 0 {
                min_weight = min_weight.min(w);
            }
        }
        assert_eq!(min_weight, 8, "minimum nonzero codeword weight must be 8");
        // Full Golay weight enumerator: 1 + 759x^8 + 2576x^12 + 759x^16 + x^24.
        assert_eq!(weight_counts[0], 1);
        assert_eq!(weight_counts[8], 759);
        assert_eq!(weight_counts[12], 2576);
        assert_eq!(weight_counts[16], 759);
        assert_eq!(weight_counts[24], 1);
    }

    #[test]
    fn syndrome_table_has_exactly_2325_correctable_entries() {
        let golay = GolayCode::new();
        let correctable = golay
            .syndrome_table
            .iter()
            .filter(|&&e| e != UNCORRECTABLE)
            .count();
        assert_eq!(correctable, 1 + 24 + 276 + 2024);
        assert_eq!(correctable, 2325);
    }

    fn message_bits(msg: u32) -> [u8; 12] {
        let mut bits = [0u8; 12];
        u32_to_bits(msg, 12, &mut bits);
        bits
    }

    #[test]
    fn exhaustive_correction_up_to_weight_3() {
        let golay = GolayCode::new();
        let mut rng = XorShift32(0xC0FF_EE01);

        for _ in 0..32 {
            let msg = rng.next() & 0x0FFF;
            let info = message_bits(msg);
            let mut codeword = [0u8; 24];
            golay.encode(&info, &mut codeword).unwrap();
            let cw_bits = bits_to_u32(&codeword, 24);

            // Weight 1.
            for b0 in 0..24u32 {
                check_one(&golay, cw_bits, 1 << b0, &info, 1);
            }
            // Weight 2.
            for b0 in 0..24u32 {
                for b1 in (b0 + 1)..24u32 {
                    check_one(&golay, cw_bits, (1 << b0) | (1 << b1), &info, 2);
                }
            }
            // Weight 3.
            for b0 in 0..24u32 {
                for b1 in (b0 + 1)..24u32 {
                    for b2 in (b1 + 1)..24u32 {
                        check_one(&golay, cw_bits, (1 << b0) | (1 << b1) | (1 << b2), &info, 3);
                    }
                }
            }
        }
    }

    fn check_one(golay: &GolayCode, cw_bits: u32, err: u32, info: &[u8; 12], expected_weight: u32) {
        let received_bits = cw_bits ^ err;
        let mut received = [0u8; 24];
        u32_to_bits(received_bits, 24, &mut received);
        let mut decoded = [0u8; 12];
        let corrected = golay
            .decode(&received, &mut decoded)
            .unwrap_or_else(|e| panic!("expected correction, got {e:?}"));
        assert_eq!(corrected as u32, expected_weight);
        assert_eq!(&decoded, info);
    }

    #[test]
    fn weight_4_errors_are_detected_not_miscorrected() {
        let golay = GolayCode::new();
        let mut rng = XorShift32(0x5EED_1234);

        let info = [1u8, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
        let mut codeword = [0u8; 24];
        golay.encode(&info, &mut codeword).unwrap();
        let cw_bits = bits_to_u32(&codeword, 24);

        let mut tested = 0;
        while tested < 200 {
            // Draw 4 distinct bit positions in 0..24.
            let mut positions = [0u32; 4];
            let mut count = 0;
            while count < 4 {
                let candidate = rng.next() % 24;
                if !positions[..count].contains(&candidate) {
                    positions[count] = candidate;
                    count += 1;
                }
            }
            let err = positions.iter().fold(0u32, |acc, &p| acc | (1 << p));

            let received_bits = cw_bits ^ err;
            let mut received = [0u8; 24];
            u32_to_bits(received_bits, 24, &mut received);
            let mut decoded = [0u8; 12];
            let result = golay.decode(&received, &mut decoded);
            assert_eq!(
                result,
                Err(FecError::DecoderNotConverged),
                "weight-4 error pattern {err:#08x} must be detected, not miscorrected"
            );
            tested += 1;
        }
    }

    #[test]
    fn encode_decode_roundtrip_zero_errors() {
        let golay = GolayCode::new();
        for msg in [0u32, 1, 0xABC, 0xFFF, 0x555, 0xAAA] {
            let info = message_bits(msg);
            let mut codeword = [0u8; 24];
            golay.encode(&info, &mut codeword).unwrap();
            let mut decoded = [0u8; 12];
            assert_eq!(golay.decode(&codeword, &mut decoded), Ok(0));
            assert_eq!(decoded, info);
        }
    }

    #[test]
    fn encode_rejects_wrong_lengths() {
        let golay = GolayCode::new();
        let info_short = [0u8; 11];
        let mut out = [0u8; 24];
        assert_eq!(
            golay.encode(&info_short, &mut out),
            Err(FecError::BufferTooSmall {
                required: GOLAY_K,
                provided: 11
            })
        );

        let info = [0u8; 12];
        let mut out_short = [0u8; 23];
        assert_eq!(
            golay.encode(&info, &mut out_short),
            Err(FecError::BufferTooSmall {
                required: GOLAY_N,
                provided: 23
            })
        );
    }

    #[test]
    fn decode_reports_buffer_too_small() {
        let golay = GolayCode::new();
        let received = [0u8; 23];
        let mut out = [0u8; 12];
        assert_eq!(
            golay.decode(&received, &mut out),
            Err(FecError::BufferTooSmall {
                required: 24,
                provided: 23
            })
        );

        let received = [0u8; 24];
        let mut out_short = [0u8; 11];
        assert_eq!(
            golay.decode(&received, &mut out_short),
            Err(FecError::BufferTooSmall {
                required: 12,
                provided: 11
            })
        );
    }
}
