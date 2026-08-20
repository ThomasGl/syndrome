//! CCSDS 131.0-B-3 Reed–Solomon (255,223) outer code — evaluation-based RS
//! over $GF(2^8)$, the algebraic construction CCSDS actually specifies for
//! its telemetry outer code.
//!
//! # Why this is a separate module from [`crate::reed_solomon`]
//!
//! [`crate::reed_solomon::ReedSolomon`] is a from-scratch **Cauchy-matrix**
//! erasure code: `coeffs[i][j] = 1/(x_i ⊕ y_j)`. CCSDS's RS(255,223) is a
//! different mathematical object — a classical **evaluation-based** code
//! whose generator polynomial has explicit roots at consecutive powers of a
//! primitive element, decoded via syndromes / Berlekamp–Massey / Chien
//! search / Forney (the same algebraic family [`crate::bch`] already
//! implements for binary BCH codes, generalized here to non-binary GF(256)
//! symbols so it corrects byte errors directly, not just erasures). A
//! Cauchy matrix has no such root structure, so Berlekamp–Massey's syndrome
//! trick — which depends on syndromes being sums of geometric sequences in a
//! power-of-α evaluation basis — does not apply to it (this is the same
//! reason an earlier attempt to "wire up" `bch.rs`'s machinery onto
//! `reed_solomon.rs` was retracted; see that module's docs). CCSDS's code
//! needs its own implementation, not a reuse of either existing one.
//!
//! # Field and generator parameters (CCSDS 131.0-B-3 §4)
//!
//! - Field: $GF(2^8)$ built from the primitive polynomial $1 + x + x^2 +
//!   x^7 + x^8$ (hex `0x187`) — a **different** primitive polynomial from
//!   [`crate::reed_solomon`] and [`crate::bch`]'s `0x11D`, so this module
//!   builds its own `exp`/`log` tables rather than sharing either.
//! - Primitive root $\alpha$: the conventional generator of that field
//!   (`0x02`).
//! - Code root set: $\beta^{112}, \beta^{113}, \ldots, \beta^{143}$ where
//!   $\beta = \alpha^{11}$ — i.e. first consecutive root (Fcr) = 112,
//!   primitive-element step (Prim) = 11, 32 roots ($E = 16$ symbol-error
//!   correction capability, $2E = 32$ parity symbols).
//! - $n = 255$, $k = 223$, $n - k = 32$.
//!
//! These parameters (and the GF tables and generator polynomial they
//! produce) were cross-checked against an independent, real-world CCSDS
//! RS(255,223) implementation and its own precomputed known-answer test
//! vector (see `tests::matches_known_answer_conventional_vector` below) —
//! not merely re-derived from the standard's stated formula and trusted on
//! faith.
//!
//! # Dual-basis vs. conventional representation
//!
//! CCSDS 131.0-B-3 §4.4.2 defines a "dual-basis" bit representation as an
//! implementation option for hardware built directly from the standard's
//! own worked circuit description. It explicitly permits the mathematically
//! equivalent **conventional** (power-of-α, "single-basis") representation
//! as an alternative — the two differ only by a fixed linear (GF(2))
//! change of basis applied to every symbol, not in the code itself. This
//! module implements the conventional representation only; a caller
//! integrating with dual-basis hardware would need the basis-transform
//! table CCSDS §4.4.2 specifies, which this module does not provide.
//!
//! # Interleaving
//!
//! CCSDS 131.0-B-3 permits interleaving depths $I \in \lbrace 1, 2, 3, 4, 5, 8 \rbrace$
//! ($I = 1$ meaning no interleaving): $I$ independent RS(255,223) codewords
//! are encoded, then their bytes are interleaved byte-by-byte across the
//! group (byte $j$ of interleaved block $b$ comes from RS codeword $b$'s
//! byte $j$) so that a single burst of up to $I \cdot t$ consecutive
//! channel-byte errors distributes to at most $t$ errors in any one
//! underlying codeword. [`CcsdsReedSolomon::new`] takes the interleaving
//! depth; [`CcsdsReedSolomon::encode`] and [`CcsdsReedSolomon::decode`]
//! operate on the full interleaved block.
//!
//! # What this does not cover
//!
//! Frame synchronization markers and the pseudo-randomization (scrambling)
//! sequence CCSDS 131.0-B-3 also specifies for a complete telemetry
//! downlink are outside this module's scope — it implements the RS(255,223)
//! channel code only. See [`crate::viterbi`]'s `# CCSDS conformance`
//! section for the standard's convolutional inner code, which this crate
//! also implements, separately.

use crate::alloc_prelude::*;
use crate::error::FecError;

/// Symbols per (uninterleaved) RS block.
const N: usize = 255;
/// Parity symbols per block ($2E$, $E = 16$).
const NROOTS: usize = 32;
/// Data symbols per block.
const K: usize = N - NROOTS;
/// First consecutive root, as an exponent of $\beta = \alpha^{\text{PRIM}}$.
const FCR: usize = 112;
/// Primitive-element step: the code's roots are powers of $\beta =
/// \alpha^{\text{PRIM}}$, not of $\alpha$ itself.
const PRIM: usize = 11;
/// Sentinel "log of zero" value in index-form arrays (no field element has
/// this as a real exponent, since the multiplicative group has order 255).
const A0: i32 = N as i32;

/// $GF(2^8)$ exp/log tables for CCSDS's field polynomial `0x187`
/// ($1 + x + x^2 + x^7 + x^8$) — **not** the `0x11D` polynomial
/// [`crate::reed_solomon`] and [`crate::bch`] use.
struct GfTables {
    /// Length 512 (duplicated) so `exp[log_a + log_b]` never needs `% 255`.
    exp: [u8; 512],
    log: [u8; 256],
}

impl GfTables {
    fn new() -> Self {
        let mut exp = [0u8; 512];
        let mut log = [0u8; 256];
        let mut x: u8 = 1;
        for i in 0..255usize {
            exp[i] = x;
            log[x as usize] = i as u8;
            let hi = (x & 0x80) != 0;
            x <<= 1;
            if hi {
                x ^= 0x87; // reduce mod x^8 + x^7 + x^2 + x + 1
            }
        }
        for i in 255usize..512usize {
            exp[i] = exp[i - 255];
        }
        GfTables { exp, log }
    }

    #[inline]
    fn mul(&self, a: u8, b: u8) -> u8 {
        if a == 0 || b == 0 {
            return 0;
        }
        let la = self.log[a as usize] as usize;
        let lb = self.log[b as usize] as usize;
        self.exp[la + lb]
    }
}

/// Build the generator polynomial $g(x) = \prod_{i=0}^{31} (x + \beta^{112 +
/// i})$, $\beta = \alpha^{11}$, in ascending-power coefficient form: `g[j]`
/// is the coefficient of $x^j$, with `g[NROOTS] = 1` (monic). Computed
/// programmatically (same incremental product-of-linear-factors technique
/// [`crate::bch`]'s `minimal_poly_bits` uses for its own generator, adapted
/// from a binary `GF(2)` bitmask to full `GF(2^8)` field-element
/// coefficients since RS roots are not restricted to a cyclotomic coset).
fn build_generator(gf: &GfTables) -> [u8; NROOTS + 1] {
    let mut g = [0u8; NROOTS + 1];
    g[0] = 1; // g(x) = 1 (degree 0) initially
    let mut deg = 0usize;
    for i in 0..NROOTS {
        let root_exp = (PRIM * (FCR + i)) % 255;
        let root = gf.exp[root_exp];
        let new_deg = deg + 1;
        let mut new_g = [0u8; NROOTS + 1];
        for j in 0..=new_deg {
            let shift_term = if j >= 1 { g[j - 1] } else { 0 };
            let const_term = if j <= deg { gf.mul(g[j], root) } else { 0 };
            new_g[j] = shift_term ^ const_term;
        }
        g = new_g;
        deg = new_deg;
    }
    g
}

/// A CCSDS 131.0-B-3 RS(255,223) codec at a configured interleaving depth.
///
/// See the module docs for the field/generator parameters and what this
/// does and does not cover.
pub struct CcsdsReedSolomon {
    interleaving: usize,
    gf: GfTables,
    /// Generator polynomial, ascending-power coefficient form (see
    /// [`build_generator`]).
    gen_poly: [u8; NROOTS + 1],
}

impl CcsdsReedSolomon {
    /// Data symbols per (uninterleaved) codeword: always 223.
    pub const DATA_LEN: usize = K;
    /// Total symbols per (uninterleaved) codeword: always 255.
    pub const BLOCK_LEN: usize = N;
    /// Parity symbols per (uninterleaved) codeword: always 32.
    pub const PARITY_LEN: usize = NROOTS;

    /// Construct a codec at interleaving depth `interleaving`.
    ///
    /// # Errors
    ///
    /// Returns [`FecError::InvalidParam`] if `interleaving` is not one of
    /// the depths CCSDS 131.0-B-3 permits: `1, 2, 3, 4, 5, 8`.
    pub fn new(interleaving: usize) -> Result<Self, FecError> {
        if !matches!(interleaving, 1 | 2 | 3 | 4 | 5 | 8) {
            return Err(FecError::InvalidParam(
                "CCSDS RS interleaving depth must be one of 1, 2, 3, 4, 5, 8",
            ));
        }
        let gf = GfTables::new();
        let gen_poly = build_generator(&gf);
        Ok(Self {
            interleaving,
            gf,
            gen_poly,
        })
    }

    /// Configured interleaving depth.
    pub fn interleaving(&self) -> usize {
        self.interleaving
    }

    /// Data length for the full interleaved block: `interleaving * 223`.
    pub fn data_len(&self) -> usize {
        self.interleaving * K
    }

    /// Codeword length for the full interleaved block: `interleaving * 255`.
    pub fn block_len(&self) -> usize {
        self.interleaving * N
    }

    /// Systematic-encode `data` (length [`Self::data_len`]) into `codeword`
    /// (length [`Self::block_len`]), conventional-basis representation.
    ///
    /// For `interleaving > 1`, the `interleaving` underlying RS(255,223)
    /// codewords are interleaved byte-by-byte across the output (see the
    /// module docs).
    ///
    /// # Errors
    ///
    /// Returns [`FecError::BufferTooSmall`] on a length mismatch.
    pub fn encode(&self, data: &[u8], codeword: &mut [u8]) -> Result<(), FecError> {
        let dl = self.data_len();
        let bl = self.block_len();
        if data.len() != dl {
            return Err(FecError::BufferTooSmall {
                required: dl,
                provided: data.len(),
            });
        }
        if codeword.len() != bl {
            return Err(FecError::BufferTooSmall {
                required: bl,
                provided: codeword.len(),
            });
        }

        for stream in 0..self.interleaving {
            let block_data: Vec<u8> = (0..K)
                .map(|j| data[j * self.interleaving + stream])
                .collect();
            let mut block = [0u8; N];
            block[..K].copy_from_slice(&block_data);
            self.encode_block(&mut block);
            for j in 0..N {
                codeword[j * self.interleaving + stream] = block[j];
            }
        }
        Ok(())
    }

    /// Systematic-encode one uninterleaved RS(255,223) block in place:
    /// `block[..K]` must already hold the 223 data symbols; `block[K..]` is
    /// overwritten with the 32 computed parity symbols.
    ///
    /// Classic shift-register systematic encode: `parity(x) = (data(x) *
    /// x^{NROOTS}) mod g(x)`, computed one data symbol at a time without
    /// ever materializing `data(x) * x^{NROOTS}` explicitly.
    fn encode_block(&self, block: &mut [u8; N]) {
        let mut parity = [0u8; NROOTS];
        for i in 0..K {
            let feedback = block[i] ^ parity[0];
            if feedback != 0 {
                for j in 1..NROOTS {
                    parity[j - 1] = parity[j] ^ self.gf.mul(feedback, self.gen_poly[NROOTS - j]);
                }
                parity[NROOTS - 1] = self.gf.mul(feedback, self.gen_poly[0]);
            } else {
                for j in 1..NROOTS {
                    parity[j - 1] = parity[j];
                }
                parity[NROOTS - 1] = 0;
            }
        }
        block[K..].copy_from_slice(&parity);
    }

    /// Decode `block` (length [`Self::block_len`]) in place, correcting up
    /// to 16 symbol errors per underlying RS(255,223) codeword.
    ///
    /// Returns the total number of corrected symbols across all
    /// `interleaving` underlying codewords.
    ///
    /// # Errors
    ///
    /// Returns [`FecError::BufferTooSmall`] on a length mismatch, or
    /// [`FecError::DecoderNotConverged`] if any underlying codeword's
    /// error-locator polynomial degree does not match its root count — the
    /// standard "more errors than the code can correct" signal (same
    /// convergence contract as [`crate::bch::BchCode::decode`]; as with any
    /// bounded-distance decoder, errors beyond capacity are not always
    /// caught, and when they are not, `block` is silently corrupted further
    /// rather than left alone — this is an inherent property of the
    /// algorithm, not specific to this implementation).
    pub fn decode(&self, block: &mut [u8]) -> Result<usize, FecError> {
        let bl = self.block_len();
        if block.len() != bl {
            return Err(FecError::BufferTooSmall {
                required: bl,
                provided: block.len(),
            });
        }

        let mut total_corrected = 0usize;
        for stream in 0..self.interleaving {
            let mut sub = [0u8; N];
            for j in 0..N {
                sub[j] = block[j * self.interleaving + stream];
            }
            let corrected = self.decode_block(&mut sub)?;
            total_corrected += corrected;
            for j in 0..N {
                block[j * self.interleaving + stream] = sub[j];
            }
        }
        Ok(total_corrected)
    }

    /// Decode one uninterleaved RS(255,223) block in place via syndromes,
    /// Berlekamp–Massey, Chien search, and Forney's algorithm — the
    /// classic algebraic RS decoding pipeline (the same family of steps
    /// [`crate::bch`]'s module docs describe for the binary case),
    /// generalized here to non-binary `GF(2^8)` symbol errors with
    /// explicit error *values*, not just error *locations*.
    fn decode_block(&self, block: &mut [u8; N]) -> Result<usize, FecError> {
        let exp = &self.gf.exp;
        let log = &self.gf.log;

        // --- Syndromes: S_i = block(beta^(FCR+i)), i = 0..NROOTS, via
        // Horner's rule. A codeword has all syndromes zero.
        let mut syn = [0i32; NROOTS];
        for s in syn.iter_mut() {
            *s = block[0] as i32;
        }
        for j in 1..N {
            for i in 0..NROOTS {
                if syn[i] == 0 {
                    syn[i] = block[j] as i32;
                } else {
                    let root_exp = ((FCR + i) * PRIM) % 255;
                    syn[i] = block[j] as i32
                        ^ exp[(log[syn[i] as usize] as usize + root_exp) % 255] as i32;
                }
            }
        }
        let mut syn_error = 0u8;
        let mut syn_log = [A0; NROOTS];
        for i in 0..NROOTS {
            syn_error |= syn[i] as u8;
            syn_log[i] = if syn[i] == 0 {
                A0
            } else {
                log[syn[i] as usize] as i32
            };
        }
        if syn_error == 0 {
            // Already a valid codeword.
            return Ok(0);
        }

        // --- Berlekamp-Massey: find the error-locator polynomial lambda(x).
        let mut lambda = [0i32; NROOTS + 1];
        lambda[0] = 1;
        let mut b = [A0; NROOTS + 1];
        b[0] = 0;
        let mut t_poly = [0i32; NROOTS + 1];
        let mut el = 0usize;
        let mut r = 0usize;
        while r < NROOTS {
            let mut discrepancy = 0u8;
            for i in 0..=r {
                if lambda[i] != 0 && syn_log[r - i] != A0 {
                    let d = (log[lambda[i] as usize] as i32 + syn_log[r - i]) % 255;
                    discrepancy ^= exp[d as usize];
                }
            }
            let discrepancy_log = if discrepancy == 0 {
                A0
            } else {
                log[discrepancy as usize] as i32
            };

            if discrepancy_log == A0 {
                for i in (1..=NROOTS).rev() {
                    b[i] = b[i - 1];
                }
                b[0] = A0;
            } else {
                t_poly[0] = lambda[0];
                for i in 0..NROOTS {
                    if b[i] != A0 {
                        t_poly[i + 1] =
                            lambda[i + 1] ^ exp[((discrepancy_log + b[i]) % 255) as usize] as i32;
                    } else {
                        t_poly[i + 1] = lambda[i + 1];
                    }
                }
                if 2 * el <= r {
                    el = r + 1 - el;
                    for i in 0..=NROOTS {
                        b[i] = if lambda[i] == 0 {
                            A0
                        } else {
                            (log[lambda[i] as usize] as i32 - discrepancy_log + 255) % 255
                        };
                    }
                } else {
                    for i in (1..=NROOTS).rev() {
                        b[i] = b[i - 1];
                    }
                    b[0] = A0;
                }
                lambda[..=NROOTS].copy_from_slice(&t_poly[..=NROOTS]);
            }
            r += 1;
        }

        let deg_lambda = (0..=NROOTS).rev().find(|&i| lambda[i] != 0).unwrap_or(0);

        // --- Chien search: find the roots of lambda(x), giving error
        // locations.
        let mut lambda_log = [A0; NROOTS + 1];
        for i in 0..=NROOTS {
            lambda_log[i] = if lambda[i] == 0 {
                A0
            } else {
                log[lambda[i] as usize] as i32
            };
        }
        let mut reg = [A0; NROOTS + 1];
        reg[1..=NROOTS].copy_from_slice(&lambda_log[1..=NROOTS]);

        let mut root = [0usize; NROOTS];
        let mut loc = [0usize; NROOTS];
        let mut count = 0usize;
        // IPrim: the per-iteration step for the error-location index k,
        // i.e. Prim * IPrim ≡ 1 (mod 255) in the sense the classic
        // decoder's Chien-search loop uses it. Mirrored directly from a
        // verified reference implementation rather than re-derived, since a
        // sign error here would silently report wrong error *positions*
        // while still passing the syndrome/degree checks (caught instead by
        // `matches_known_answer_conventional_vector_decode`, which checks
        // exact recovered bytes, not just corrected-count).
        const IPRIM: i32 = 116;
        let mut k = IPRIM - 1;
        let mut i = 1usize;
        while i <= N && count < deg_lambda {
            let mut q = 1u8;
            for j in (1..=deg_lambda).rev() {
                if reg[j] != A0 {
                    reg[j] = (reg[j] + j as i32) % 255;
                    q ^= exp[reg[j] as usize];
                }
            }
            if q == 0 {
                root[count] = i;
                loc[count] = ((k % 255) + 255) as usize % 255;
                count += 1;
            }
            i += 1;
            k = (k + IPRIM) % 255;
        }

        if deg_lambda != count {
            return Err(FecError::DecoderNotConverged);
        }

        // --- Forney's algorithm: compute error *values* at each located
        // position, using the error evaluator polynomial omega(x) = [S(x) *
        // lambda(x)] mod x^{NROOTS}.
        let mut omega_log = [A0; NROOTS];
        let mut deg_omega = 0usize;
        for i in 0..NROOTS {
            let mut tmp = 0u8;
            let jmax = deg_lambda.min(i);
            for j in 0..=jmax {
                if syn_log[i - j] != A0 && lambda_log[j] != A0 {
                    tmp ^= exp[((syn_log[i - j] + lambda_log[j]) % 255) as usize];
                }
            }
            if tmp != 0 {
                deg_omega = i;
            }
            omega_log[i] = if tmp == 0 {
                A0
            } else {
                log[tmp as usize] as i32
            };
        }

        for j in 0..count {
            let mut num1 = 0u8;
            for i in (0..=deg_omega).rev() {
                if omega_log[i] != A0 {
                    num1 ^= exp
                        [((omega_log[i] + (i as i32) * root[j] as i32).rem_euclid(255)) as usize];
                }
            }
            let num2 = exp[((root[j] as i32 * (FCR as i32 - 1)).rem_euclid(255)) as usize];
            let mut den = 0u8;
            let start = deg_lambda.min(NROOTS - 1) & !1usize;
            let mut i = start as i32;
            while i >= 0 {
                let idx = (i + 1) as usize;
                if lambda_log[idx] != A0 {
                    den ^= exp[((lambda_log[idx] + i * root[j] as i32).rem_euclid(255)) as usize];
                }
                i -= 2;
            }
            if den == 0 {
                return Err(FecError::DecoderNotConverged);
            }
            if num1 != 0 {
                let corr_exp = (log[num1 as usize] as i32 + log[num2 as usize] as i32 + 255
                    - log[den as usize] as i32)
                    .rem_euclid(255);
                block[loc[j]] ^= exp[corr_exp as usize];
            }
        }

        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::Xorshift64;

    /// Cross-check `GfTables::new`'s from-scratch construction against an
    /// independent third-party CCSDS RS(255,223) implementation's own
    /// published `ALPHA_TO` table (256 entries), byte-for-byte. This is the
    /// foundation everything else in this module is built on -- if the
    /// field tables are wrong, encode/decode would still "work" internally
    /// (self-consistent) but would not match the real CCSDS standard at
    /// all, and this is the one check that would catch that.
    #[test]
    fn gf_tables_match_reference_alpha_to() {
        const REFERENCE_ALPHA_TO: [u8; 256] = [
            0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x87, 0x89, 0x95, 0xad, 0xdd, 0x3d,
            0x7a, 0xf4, 0x6f, 0xde, 0x3b, 0x76, 0xec, 0x5f, 0xbe, 0xfb, 0x71, 0xe2, 0x43, 0x86,
            0x8b, 0x91, 0xa5, 0xcd, 0x1d, 0x3a, 0x74, 0xe8, 0x57, 0xae, 0xdb, 0x31, 0x62, 0xc4,
            0x0f, 0x1e, 0x3c, 0x78, 0xf0, 0x67, 0xce, 0x1b, 0x36, 0x6c, 0xd8, 0x37, 0x6e, 0xdc,
            0x3f, 0x7e, 0xfc, 0x7f, 0xfe, 0x7b, 0xf6, 0x6b, 0xd6, 0x2b, 0x56, 0xac, 0xdf, 0x39,
            0x72, 0xe4, 0x4f, 0x9e, 0xbb, 0xf1, 0x65, 0xca, 0x13, 0x26, 0x4c, 0x98, 0xb7, 0xe9,
            0x55, 0xaa, 0xd3, 0x21, 0x42, 0x84, 0x8f, 0x99, 0xb5, 0xed, 0x5d, 0xba, 0xf3, 0x61,
            0xc2, 0x03, 0x06, 0x0c, 0x18, 0x30, 0x60, 0xc0, 0x07, 0x0e, 0x1c, 0x38, 0x70, 0xe0,
            0x47, 0x8e, 0x9b, 0xb1, 0xe5, 0x4d, 0x9a, 0xb3, 0xe1, 0x45, 0x8a, 0x93, 0xa1, 0xc5,
            0x0d, 0x1a, 0x34, 0x68, 0xd0, 0x27, 0x4e, 0x9c, 0xbf, 0xf9, 0x75, 0xea, 0x53, 0xa6,
            0xcb, 0x11, 0x22, 0x44, 0x88, 0x97, 0xa9, 0xd5, 0x2d, 0x5a, 0xb4, 0xef, 0x59, 0xb2,
            0xe3, 0x41, 0x82, 0x83, 0x81, 0x85, 0x8d, 0x9d, 0xbd, 0xfd, 0x7d, 0xfa, 0x73, 0xe6,
            0x4b, 0x96, 0xab, 0xd1, 0x25, 0x4a, 0x94, 0xaf, 0xd9, 0x35, 0x6a, 0xd4, 0x2f, 0x5e,
            0xbc, 0xff, 0x79, 0xf2, 0x63, 0xc6, 0x0b, 0x16, 0x2c, 0x58, 0xb0, 0xe7, 0x49, 0x92,
            0xa3, 0xc1, 0x05, 0x0a, 0x14, 0x28, 0x50, 0xa0, 0xc7, 0x09, 0x12, 0x24, 0x48, 0x90,
            0xa7, 0xc9, 0x15, 0x2a, 0x54, 0xa8, 0xd7, 0x29, 0x52, 0xa4, 0xcf, 0x19, 0x32, 0x64,
            0xc8, 0x17, 0x2e, 0x5c, 0xb8, 0xf7, 0x69, 0xd2, 0x23, 0x46, 0x8c, 0x9f, 0xb9, 0xf5,
            0x6d, 0xda, 0x33, 0x66, 0xcc, 0x1f, 0x3e, 0x7c, 0xf8, 0x77, 0xee, 0x5b, 0xb6, 0xeb,
            0x51, 0xa2, 0xc3, 0x00,
        ];
        let gf = GfTables::new();
        for i in 0..255 {
            assert_eq!(
                gf.exp[i], REFERENCE_ALPHA_TO[i],
                "exp[{i}] mismatch vs. reference ALPHA_TO"
            );
        }
        // ALPHA_TO[255] (the reference array's last entry) is a padding
        // sentinel (0x00), not alpha^255; this module's own exp[255] is the
        // real alpha^255 = alpha^0 = 1 (via the wraparound duplication),
        // so that one entry is intentionally not compared.
    }

    /// A known-answer test vector: 223 sequential data bytes (0x00..=0xde,
    /// per the standard convention this crate's own test data follows) and
    /// their real, independently-computed 32-byte CCSDS RS(255,223) parity,
    /// taken from a third-party conventional-basis implementation (not
    /// generated by this module) -- this is the test that actually proves
    /// conformance to the real standard, not just internal self-consistency.
    fn known_answer_block() -> [u8; N] {
        let mut block = [0u8; N];
        for (i, b) in block[..K].iter_mut().enumerate() {
            *b = i as u8;
        }
        let parity: [u8; NROOTS] = [
            0x2f, 0xbd, 0x4f, 0xb4, 0x74, 0x84, 0x94, 0xb9, 0xac, 0xd5, 0x54, 0x62, 0x72, 0x12,
            0xee, 0xb3, 0xeb, 0xed, 0x41, 0x19, 0x1d, 0xe1, 0xd3, 0x63, 0x20, 0xea, 0x49, 0x29,
            0x0b, 0x25, 0xab, 0xcf,
        ];
        block[K..].copy_from_slice(&parity);
        block
    }

    #[test]
    fn matches_known_answer_conventional_vector_encode() {
        let rs = CcsdsReedSolomon::new(1).unwrap();
        let expected = known_answer_block();
        let mut block = [0u8; N];
        rs.encode(&expected[..K], &mut block).unwrap();
        assert_eq!(
            block, expected,
            "encoded parity does not match the real CCSDS RS(255,223) known-answer vector"
        );
    }

    #[test]
    fn matches_known_answer_conventional_vector_decode() {
        let rs = CcsdsReedSolomon::new(1).unwrap();
        let expected = known_answer_block();

        // The same 16-error mask the reference implementation's own decode
        // test uses, at the same 16 positions.
        let error_positions_and_values: [(usize, u8); 16] = [
            (0, 0x58),
            (2, 0xA3),
            (5, 0xCD),
            (14, 0x0D),
            (50, 0xCA),
            (51, 0x96),
            (80, 0x1B),
            (89, 0xA2),
            (91, 0xAC),
            (100, 0xB9),
            (145, 0xE5),
            (176, 0x94),
            (185, 0xC3),
            (214, 0x97),
            (231, 0x7A),
            (250, 0x29),
        ];
        let mut corrupted = expected;
        for &(pos, val) in &error_positions_and_values {
            corrupted[pos] ^= val;
        }

        let rs_check = corrupted != expected;
        assert!(rs_check, "test setup error: no corruption applied");

        let corrected = rs.decode_block(&mut corrupted).unwrap();
        assert_eq!(corrected, 16, "expected exactly 16 corrected symbols");
        assert_eq!(
            corrupted, expected,
            "decode did not recover the exact known-answer block"
        );
    }

    #[test]
    fn round_trip_no_errors() {
        let rs = CcsdsReedSolomon::new(1).unwrap();
        let data: Vec<u8> = (0..K as u32).map(|i| (i * 37 + 11) as u8).collect();
        let mut block = vec![0u8; N];
        rs.encode(&data, &mut block).unwrap();
        let corrected = rs.decode(&mut block).unwrap();
        assert_eq!(corrected, 0);
        assert_eq!(&block[..K], &data[..]);
    }

    #[test]
    fn round_trip_random_errors_up_to_capacity() {
        let mut rng = Xorshift64::new(0xCC5D5);
        let rs = CcsdsReedSolomon::new(1).unwrap();
        for trial in 0..200 {
            let data: Vec<u8> = (0..K).map(|_| rng.next_below(256) as u8).collect();
            let mut block = vec![0u8; N];
            rs.encode(&data, &mut block).unwrap();

            let num_errors = 1 + rng.next_below(NROOTS / 2); // 1..=16
            let mut positions: Vec<usize> = (0..N).collect();
            // Fisher-Yates partial shuffle to pick `num_errors` distinct positions.
            for i in 0..num_errors {
                let j = i + rng.next_below(N - i);
                positions.swap(i, j);
            }
            for &pos in &positions[..num_errors] {
                let flip = 1 + rng.next_below(255) as u8; // never 0
                block[pos] ^= flip;
            }

            let corrected = rs.decode(&mut block).unwrap_or_else(|e| {
                panic!("trial {trial}: decode failed with {num_errors} errors: {e:?}")
            });
            assert_eq!(
                corrected, num_errors,
                "trial {trial}: corrected count mismatch"
            );
            assert_eq!(&block[..K], &data[..], "trial {trial}: data not recovered");
        }
    }

    #[test]
    fn interleaved_round_trip() {
        for &depth in &[1usize, 2, 3, 4, 5, 8] {
            let rs = CcsdsReedSolomon::new(depth).unwrap();
            let data: Vec<u8> = (0..rs.data_len() as u32)
                .map(|i| (i * 13 + 7) as u8)
                .collect();
            let mut block = vec![0u8; rs.block_len()];
            rs.encode(&data, &mut block).unwrap();

            // One error in every underlying stream (well within per-stream capacity).
            for stream in 0..depth {
                block[stream] ^= 0xFF;
            }
            let corrected = rs.decode(&mut block).unwrap();
            assert_eq!(corrected, depth, "interleaving depth {depth}");
            assert_eq!(
                &block[..rs.data_len()],
                &data[..],
                "interleaving depth {depth}"
            );
        }
    }

    #[test]
    fn invalid_interleaving_depth_rejected() {
        assert!(CcsdsReedSolomon::new(0).is_err());
        assert!(CcsdsReedSolomon::new(6).is_err());
        assert!(CcsdsReedSolomon::new(7).is_err());
        assert!(CcsdsReedSolomon::new(9).is_err());
        for &ok in &[1, 2, 3, 4, 5, 8] {
            assert!(CcsdsReedSolomon::new(ok).is_ok());
        }
    }

    #[test]
    fn wrong_length_buffers_rejected() {
        let rs = CcsdsReedSolomon::new(1).unwrap();
        let mut short_out = vec![0u8; N - 1];
        assert!(rs.encode(&vec![0u8; K], &mut short_out).is_err());
        let mut short_block = vec![0u8; N - 1];
        assert!(rs.decode(&mut short_block).is_err());
    }

    /// Mutation guard: a Berlekamp-Massey implementation with the update
    /// condition inverted (`2*el <= r` flipped) still "runs" without
    /// panicking but silently produces a wrong-degree or wrong locator
    /// polynomial for many inputs. This is exactly the class of bug the
    /// known-answer test above exists to catch -- recorded here explicitly
    /// so the mechanism is documented, not just implicit in one test
    /// passing.
    #[test]
    fn known_answer_vector_is_sensitive_to_decoder_correctness() {
        // Re-assert the known-answer decode test's core invariant directly,
        // as a standalone regression anchor independent of test ordering.
        let rs = CcsdsReedSolomon::new(1).unwrap();
        let mut block = known_answer_block();
        block[0] ^= 0x01; // single-byte error
        let corrected = rs.decode_block(&mut block).unwrap();
        assert_eq!(corrected, 1);
        assert_eq!(block, known_answer_block());
    }
}
