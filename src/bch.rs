//! Binary BCH codes over $GF(2^8)$ — systematic encode and algebraic decode.
//!
//! Implements primitive, narrow-sense binary Bose–Chaudhuri–Hocquenghem (BCH)
//! codes with block length $n = 255$ built over $GF(2^8)$, for a configurable
//! error-correction capability $t \in \{1, \ldots, 10\}$. The generator
//! polynomial is computed programmatically from cyclotomic cosets of the
//! primitive element $\alpha$ (no hardcoded coset or generator tables), the
//! encoder is a systematic bit-serial LFSR, and the decoder is the classic
//! algebraic pipeline: syndromes, Berlekamp–Massey, Chien search.
//!
//! # Industry motivation
//!
//! Binary BCH$(255, k, t)$ codes are a workhorse outer/inner code across the
//! communications and storage industries:
//!
//! - **DVB-S2** uses a shortened BCH code as the *outer* code of its
//!   BCH + LDPC concatenated FEC chain (the LDPC handles bulk noise, the BCH
//!   outer code mops up the LDPC's residual error floor).
//! - **NAND-flash controllers** use *shortened* BCH codes (see
//!   [`BchCode::shortened_encode`] / [`BchCode::shortened_decode`]) sized to
//!   the ECC budget of a flash page — a full $n=255$ codeword is longer than
//!   most page/sector ECC regions, so only the low-order $k_{\text{short}}$
//!   information positions are ever transmitted; the high-order positions are
//!   implicit, always-zero "virtual" bits.
//! - **CCSDS** (space data systems) recommends BCH codes as part of its
//!   telemetry channel coding options.
//!
//! # Algorithm
//!
//! ## Field arithmetic
//!
//! $GF(2^8)$ is built from the primitive polynomial $p(x) = x^8 + x^4 + x^3 +
//! x^2 + 1$ (hex `0x11D`), with $\alpha = 2$ a primitive root. `exp`/`log`
//! tables (the `exp` table doubled to length 512) give mod-free
//! multiplication: $a \cdot b = \exp(\log a + \log b)$.
//!
//! ## Generator polynomial
//!
//! For design distance $2t+1$, the generator polynomial is
//! $$ g(x) = \mathrm{lcm}\bigl(m_1(x), m_3(x), m_5(x), \ldots, m_{2t-1}(x)\bigr) $$
//! where $m_i(x)$ is the minimal polynomial of $\alpha^i$ over $GF(2)$. Each
//! $m_i(x)$ is the product $\prod_{j \in C_i} (x + \alpha^j)$ over the
//! cyclotomic coset $C_i = \{ i \cdot 2^s \bmod 255 : s \ge 0 \}$, computed
//! with $GF(2^8)$ arithmetic (the result is provably binary since it is
//! fixed by the Frobenius automorphism $x \mapsto x^2$). Cosets that overlap
//! (e.g. sharing a common minimal polynomial) are only multiplied into
//! $g(x)$ once — this is the "dedup of shared factors" the LCM requires,
//! since distinct irreducible factors multiply directly into the LCM. The
//! dimension is $k = n - \deg g(x)$.
//!
//! ## Encoding (systematic)
//!
//! $$ c(x) = x^{n-k} \, i(x) + \bigl( x^{n-k} \, i(x) \bmod g(x) \bigr) $$
//! computed with a single bit-serial LFSR of width $n-k$ (mirrors
//! [`crate::crc::Crc24::compute`], generalized to a `u128` register).
//! Complexity: $O(k)$ LFSR steps over an $(n-k)$-bit register, i.e.
//! $O(n \cdot (n-k))$ bit operations.
//!
//! ## Decoding (bounded-distance algebraic decoding)
//!
//! 1. **Syndromes** $S_1, \ldots, S_{2t}$: $S_j = c(\alpha^j)$ via Horner's
//!    rule. Only odd $S_j$ need a full $O(n)$ Horner evaluation; even-indexed
//!    syndromes follow from the binary-field Frobenius identity
//!    $S_{2i} = S_i^2$ (since $c_j \in \{0,1\}$, so $c_j^2 = c_j$), halving
//!    the number of Horner passes. Cost: $O(n \cdot t)$.
//! 2. **Berlekamp–Massey**: finds the minimal-degree error-locator polynomial
//!    $\Lambda(x)$ satisfying $S_{i} + \sum_{j=1}^{L} \Lambda_j S_{i-j} = 0$
//!    for $i > L$. Cost: $O(t^2)$.
//! 3. **Chien search**: evaluates $\Lambda(\alpha^{-i})$ for every candidate
//!    position; a zero indicates a bit error at that position. Cost:
//!    $O(n \cdot t)$.
//!
//! If Berlekamp–Massey produces a locator of degree $L > t$, or Chien search
//! finds a root count different from $\deg \Lambda$, more errors are present
//! than the code can correct and [`FecError::DecoderNotConverged`] is
//! returned rather than silently miscorrecting (though for a `binary` code
//! with $> t$ errors, a *false* correction to a different valid codeword is
//! an inherent, well-documented risk of bounded-distance decoding — see the
//! `beyond_capacity` tests below).
//!
//! # Bit representation
//!
//! As in [`crate::hamming`] / [`crate::crc`], bits are one-per-byte `u8`
//! values (`0` or `1`), MSB-first: `codeword[0]` is the coefficient of the
//! highest-order term of $c(x)$.
//!
//! # Examples
//!
//! ```
//! use glezer_rsv::BchCode;
//!
//! // t = 2: corrects up to 2 bit errors per 255-bit block.
//! let bch = BchCode::new(2).unwrap();
//! let info = vec![1u8, 0, 1, 1, 0, 0, 1, 0, 1, 1];
//! let mut info_full = vec![0u8; bch.k()];
//! info_full[bch.k() - info.len()..].copy_from_slice(&info);
//!
//! let mut codeword = vec![0u8; bch.n()];
//! bch.encode(&info_full, &mut codeword).unwrap();
//!
//! // Inject 2 bit errors.
//! codeword[3] ^= 1;
//! codeword[200] ^= 1;
//!
//! let corrected = bch.decode(&mut codeword).unwrap();
//! assert_eq!(corrected, 2);
//! assert_eq!(&codeword[..bch.k()], &info_full[..]);
//! ```

use crate::error::FecError;

/// Block length $n$ for BCH codes over $GF(2^8)$.
const N: usize = 255;
/// Maximum supported error-correction capability.
const MAX_T: usize = 10;
/// Maximum syndrome count ($2 \cdot \text{MAX\_T}$).
const MAX_TWO_T: usize = 2 * MAX_T;
/// Maximum cyclotomic coset size in $GF(2^8)$ (divides $8 = \deg GF(2^8)$).
const MAX_COSET: usize = 8;
/// Primitive polynomial $x^8 + x^4 + x^3 + x^2 + 1$, low 8 bits (leading bit
/// implicit, standard reduction trick).
const PRIMITIVE_POLY_LOW: u8 = 0x1D;

/// $GF(2^8)$ multiplication via `exp`/`log` tables (mod-free: `exp` has
/// length 512 so `log[a] + log[b]` never needs a modulo).
#[inline]
fn gf_mul(exp: &[u8; 512], log: &[u8; 256], a: u8, b: u8) -> u8 {
    if a == 0 || b == 0 {
        return 0;
    }
    let la = log[a as usize] as usize;
    let lb = log[b as usize] as usize;
    exp[la + lb]
}

/// $GF(2^8)$ multiplicative inverse via `exp`/`log` tables.
///
/// # Errors
///
/// None (panics via `debug_assert` on `a == 0` in debug builds only; the
/// caller must never invoke this with a zero element).
#[inline]
fn gf_inv(exp: &[u8; 512], log: &[u8; 256], a: u8) -> u8 {
    debug_assert!(a != 0, "GF(256) inverse of zero is undefined");
    let la = log[a as usize] as usize;
    exp[255 - la]
}

/// Cyclotomic coset of `i` modulo `255`: $\{ i \cdot 2^s \bmod 255 \}$.
///
/// Returns a fixed-size buffer and the number of valid entries.
fn cyclotomic_coset_mod255(i: usize) -> ([usize; MAX_COSET], usize) {
    let mut coset = [0usize; MAX_COSET];
    let mut len = 0usize;
    let mut x = i;
    loop {
        coset[len] = x;
        len += 1;
        x = (x * 2) % 255;
        if x == i || len == MAX_COSET {
            break;
        }
    }
    (coset, len)
}

/// Compute the minimal polynomial (over $GF(2)$) of $\alpha^i$ given its
/// cyclotomic coset, as a bitmask (bit $j$ = coefficient of $x^j$) plus its
/// degree.
///
/// The polynomial is built as $\prod_{j \in \text{coset}} (x + \alpha^j)$
/// using $GF(2^8)$ arithmetic for the running coefficients; the final
/// coefficients are guaranteed to be binary (`0`/`1`) because the coset is
/// closed under the Frobenius map $x \mapsto x^2$, which fixes $GF(2)$.
fn minimal_poly_bits(
    exp: &[u8; 512],
    log: &[u8; 256],
    coset: &[usize; MAX_COSET],
    coset_len: usize,
) -> (u16, usize) {
    let mut poly = [0u8; MAX_COSET + 1];
    poly[0] = 1;
    let mut deg = 0usize;
    for idx in 0..coset_len {
        let alpha_e = exp[coset[idx]];
        let new_deg = deg + 1;
        let mut new_poly = [0u8; MAX_COSET + 1];
        for j in 0..=new_deg {
            let shift_term = if j >= 1 { poly[j - 1] } else { 0 };
            let const_term = if j <= deg {
                gf_mul(exp, log, poly[j], alpha_e)
            } else {
                0
            };
            new_poly[j] = shift_term ^ const_term;
        }
        poly = new_poly;
        deg = new_deg;
    }
    let mut bits: u16 = 0;
    for (j, &coeff) in poly.iter().enumerate().take(deg + 1) {
        debug_assert!(
            coeff == 0 || coeff == 1,
            "minimal polynomial must have binary coefficients"
        );
        if coeff == 1 {
            bits |= 1 << j;
        }
    }
    (bits, deg)
}

/// A binary, primitive, narrow-sense BCH$(255, k, t)$ code over $GF(2^8)$.
///
/// Construct with [`BchCode::new`]; `k` is derived from `t` (not chosen
/// directly), matching standard BCH code construction. See the module docs
/// for the full algorithm description.
pub struct BchCode {
    t: usize,
    n: usize,
    k: usize,
    /// $\deg g(x) = n - k$, the number of parity bits.
    nk: usize,
    /// Generator polynomial $g(x)$: bit $j$ = coefficient of $x^j$, for
    /// $j = 0 \ldots nk$ (bit `nk` is always `1`, the monic leading term).
    g_full: u128,
    exp: [u8; 512],
    log: [u8; 256],
}

impl BchCode {
    /// Construct a binary BCH$(255, k, t)$ code for the given error
    /// correction capability `t`.
    ///
    /// The generator polynomial $g(x)$ and dimension $k$ are computed
    /// programmatically from the cyclotomic cosets of $\alpha, \alpha^3,
    /// \ldots, \alpha^{2t-1}$ — no coset or generator tables are hardcoded.
    ///
    /// # Arguments
    ///
    /// * `t` — Designed error-correction capability, `1..=10`.
    ///
    /// # Returns
    ///
    /// A ready-to-use [`BchCode`] with `k = 255 - deg(g)`.
    ///
    /// # Errors
    ///
    /// Returns [`FecError::InvalidParam`] if `t` is `0` or greater than the
    /// supported maximum (`10`).
    ///
    /// # Examples
    ///
    /// ```
    /// use glezer_rsv::BchCode;
    /// let bch = BchCode::new(3).unwrap();
    /// assert_eq!(bch.n(), 255);
    /// assert_eq!(bch.n() - bch.k(), bch.parity_len());
    /// ```
    pub fn new(t: usize) -> Result<Self, FecError> {
        if t == 0 || t > MAX_T {
            return Err(FecError::InvalidParam(
                "t must be in 1..=10 for BCH(255) over GF(2^8)",
            ));
        }

        // Build GF(2^8) exp/log tables (primitive polynomial 0x11D, alpha=2).
        let mut exp = [0u8; 512];
        let mut log = [0u8; 256];
        let mut x: u8 = 1;
        for i in 0..255usize {
            exp[i] = x;
            log[x as usize] = i as u8;
            let hi = (x & 0x80) != 0;
            x <<= 1;
            if hi {
                x ^= PRIMITIVE_POLY_LOW;
            }
        }
        for i in 255usize..512usize {
            exp[i] = exp[i - 255];
        }

        // g(x) = lcm(m_1, m_3, ..., m_{2t-1}), built incrementally with dedup
        // of shared cyclotomic cosets (each irreducible minimal polynomial is
        // multiplied into g(x) exactly once).
        let mut g_full: u128 = 1; // g(x) = 1 (degree 0) initially.
        let mut deg_g: usize = 0;
        let mut seen = [false; 255];
        let mut i = 1usize;
        while i < 2 * t {
            if !seen[i] {
                let (coset, coset_len) = cyclotomic_coset_mod255(i);
                for &e in coset.iter().take(coset_len) {
                    seen[e] = true;
                }
                let (minpoly_bits, mdeg) = minimal_poly_bits(&exp, &log, &coset, coset_len);

                // Multiply g_full by the new minimal polynomial: GF(2)
                // carryless polynomial multiplication (addition = XOR).
                let mut new_g: u128 = 0;
                for b in 0..=mdeg {
                    if (minpoly_bits >> b) & 1 == 1 {
                        new_g ^= g_full << b;
                    }
                }
                g_full = new_g;
                deg_g += mdeg;
            }
            i += 2;
        }

        if deg_g == 0 || deg_g >= 128 {
            return Err(FecError::InvalidParam(
                "computed BCH generator degree out of supported range",
            ));
        }

        let k = N - deg_g;
        Ok(BchCode {
            t,
            n: N,
            k,
            nk: deg_g,
            g_full,
            exp,
            log,
        })
    }

    /// Block length $n$ (always `255` for this $GF(2^8)$-based code).
    pub fn n(&self) -> usize {
        self.n
    }

    /// Information (dimension) length $k = n - \deg g(x)$.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Designed error-correction capability $t$.
    pub fn t(&self) -> usize {
        self.t
    }

    /// Number of parity bits, $n - k = \deg g(x)$.
    pub fn parity_len(&self) -> usize {
        self.nk
    }

    #[inline]
    fn mul(&self, a: u8, b: u8) -> u8 {
        gf_mul(&self.exp, &self.log, a, b)
    }

    #[inline]
    fn inv(&self, a: u8) -> u8 {
        gf_inv(&self.exp, &self.log, a)
    }

    #[inline]
    fn alpha_pow(&self, p: usize) -> u8 {
        self.exp[p % 255]
    }

    /// Run the systematic-encoder LFSR over `info` and return the `nk`-bit
    /// remainder $x^{n-k} i(x) \bmod g(x)$, packed into a `u128` (bit $j$ =
    /// coefficient of $x^j$).
    ///
    /// This is a bit-serial division register of width `nk`, structurally
    /// identical to [`crate::crc::Crc24::compute`] but generalized to a
    /// `u128` register (`nk <= 127` always holds for `t <= 10`). No dynamic
    /// allocation.
    #[inline]
    fn lfsr_remainder(&self, info: &[u8]) -> u128 {
        let nk = self.nk;
        let mask: u128 = (1u128 << nk) - 1;
        let poly = self.g_full & mask; // low `nk` bits; leading bit implicit.
        let mut reg: u128 = 0;
        for &bit in info {
            let feedback = ((reg >> (nk - 1)) as u8 & 1) ^ (bit & 1);
            reg = (reg << 1) & mask;
            if feedback != 0 {
                reg ^= poly;
            }
        }
        reg
    }

    /// Systematically encode `k` information bits into an `n`-bit codeword.
    ///
    /// $$ c(x) = x^{n-k} i(x) + \bigl(x^{n-k} i(x) \bmod g(x)\bigr) $$
    ///
    /// # Arguments
    ///
    /// * `info` — Exactly `k` bits (`0`/`1` `u8` values), MSB-first.
    /// * `codeword_out` — Output buffer, exactly `n` bytes; filled with
    ///   `info` followed by the `n-k` parity bits.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success; `codeword_out` holds the systematic codeword.
    ///
    /// # Errors
    ///
    /// * [`FecError::InvalidParam`] if `info.len() != k`, or any element is
    ///   not `0`/`1`.
    /// * [`FecError::BufferTooSmall`] if `codeword_out.len() != n`.
    ///
    /// # Examples
    ///
    /// ```
    /// use glezer_rsv::BchCode;
    /// let bch = BchCode::new(1).unwrap();
    /// let info = vec![0u8; bch.k()];
    /// let mut codeword = vec![0u8; bch.n()];
    /// bch.encode(&info, &mut codeword).unwrap();
    /// // The all-zero message encodes to the all-zero codeword.
    /// assert!(codeword.iter().all(|&b| b == 0));
    /// ```
    pub fn encode(&self, info: &[u8], codeword_out: &mut [u8]) -> Result<(), FecError> {
        if info.len() != self.k {
            return Err(FecError::InvalidParam("info length must equal k"));
        }
        if codeword_out.len() != self.n {
            return Err(FecError::BufferTooSmall {
                required: self.n,
                provided: codeword_out.len(),
            });
        }
        if info.iter().any(|&b| b > 1) {
            return Err(FecError::InvalidParam("info bits must be 0 or 1"));
        }

        let remainder = self.lfsr_remainder(info);
        codeword_out[..self.k].copy_from_slice(info);
        let nk = self.nk;
        for offset in 0..nk {
            codeword_out[self.k + offset] = ((remainder >> (nk - 1 - offset)) & 1) as u8;
        }
        Ok(())
    }

    /// Encode using a *shortened* BCH code: only `k_short <= k` information
    /// bits are transmitted, with the high-order `k - k_short` information
    /// positions treated as implicit, always-zero "virtual" bits (never
    /// stored or transmitted).
    ///
    /// This is the NAND-flash controller pattern: ECC is sized to a
    /// page/sector boundary far smaller than the full $n=255$ codeword, so
    /// only the low-order information bits and all parity bits are actually
    /// written to the flash page.
    ///
    /// # Arguments
    ///
    /// * `k_short` — Number of information bits actually carried, `<= k`.
    /// * `info` — Exactly `k_short` bits.
    /// * `codeword_out` — Output buffer, exactly `k_short + (n-k)` bytes.
    ///
    /// # Returns
    ///
    /// `Ok(())` on success.
    ///
    /// # Errors
    ///
    /// * [`FecError::InvalidParam`] if `k_short > k`, `info.len() !=
    ///   k_short`, or any bit is not `0`/`1`.
    /// * [`FecError::BufferTooSmall`] if `codeword_out` has the wrong length.
    ///
    /// # Examples
    ///
    /// ```
    /// use glezer_rsv::BchCode;
    /// let bch = BchCode::new(4).unwrap();
    /// let k_short = 64; // a small NAND ECC region
    /// let info = vec![1u8; k_short];
    /// let mut codeword = vec![0u8; k_short + bch.parity_len()];
    /// bch.shortened_encode(k_short, &info, &mut codeword).unwrap();
    /// assert_eq!(codeword.len(), k_short + bch.parity_len());
    /// ```
    pub fn shortened_encode(
        &self,
        k_short: usize,
        info: &[u8],
        codeword_out: &mut [u8],
    ) -> Result<(), FecError> {
        if k_short > self.k {
            return Err(FecError::InvalidParam("k_short must be <= k"));
        }
        if info.len() != k_short {
            return Err(FecError::InvalidParam("info length must equal k_short"));
        }
        let out_len = k_short + self.nk;
        if codeword_out.len() != out_len {
            return Err(FecError::BufferTooSmall {
                required: out_len,
                provided: codeword_out.len(),
            });
        }
        if info.iter().any(|&b| b > 1) {
            return Err(FecError::InvalidParam("info bits must be 0 or 1"));
        }

        // Leading zero bits do not perturb a zero-initialized LFSR register
        // (feedback stays 0 until a nonzero bit enters), so running the LFSR
        // over just the short info slice gives the identical remainder as
        // running it over `k - k_short` implicit zeros followed by `info`.
        let remainder = self.lfsr_remainder(info);
        codeword_out[..k_short].copy_from_slice(info);
        let nk = self.nk;
        for offset in 0..nk {
            codeword_out[k_short + offset] = ((remainder >> (nk - 1 - offset)) & 1) as u8;
        }
        Ok(())
    }

    /// Compute syndromes $S_1, \ldots, S_{2t}$ via Horner evaluation.
    ///
    /// Only odd-indexed syndromes require a full $O(n)$ Horner pass; each
    /// even-indexed syndrome reachable by repeated doubling is derived via
    /// $S_{2i} = S_i^2$ (binary-field Frobenius identity), halving the
    /// number of $O(n)$ passes needed.
    fn syndromes(&self, codeword: &[u8]) -> ([u8; MAX_TWO_T], bool) {
        let two_t = 2 * self.t;
        let mut s = [0u8; MAX_TWO_T];
        let mut any_nonzero = false;
        let mut j = 1usize;
        while j <= two_t {
            let beta = self.alpha_pow(j);
            let mut val = 0u8;
            for &bit in codeword.iter() {
                val = self.mul(val, beta) ^ (bit & 1);
            }
            s[j - 1] = val;
            any_nonzero |= val != 0;

            let mut idx = j;
            let mut v = val;
            loop {
                idx *= 2;
                if idx > two_t {
                    break;
                }
                v = self.mul(v, v);
                s[idx - 1] = v;
                any_nonzero |= v != 0;
            }
            j += 2;
        }
        (s, any_nonzero)
    }

    /// Full algebraic decode over an `n`-length codeword buffer (private
    /// core, shared by [`BchCode::decode`] and [`BchCode::shortened_decode`]).
    fn decode_full(&self, codeword: &mut [u8]) -> Result<usize, FecError> {
        if codeword.len() != self.n {
            return Err(FecError::BufferTooSmall {
                required: self.n,
                provided: codeword.len(),
            });
        }

        let two_t = 2 * self.t;
        let (s, any_nonzero) = self.syndromes(codeword);
        if !any_nonzero {
            return Ok(0);
        }

        // --- Berlekamp-Massey: find the error-locator polynomial Lambda(x).
        // C(x) is the current locator candidate, B(x) the previous one at
        // the last length-change step. All polys have degree <= t <= 10, so
        // fixed-size [u8; MAX_T + 1] arrays suffice (no heap allocation).
        let mut c = [0u8; MAX_T + 1];
        let mut b_poly = [0u8; MAX_T + 1];
        c[0] = 1;
        b_poly[0] = 1;
        let mut l = 0usize;
        let mut m = 1usize;
        let mut b = 1u8;

        for n_ in 0..two_t {
            let mut delta = s[n_];
            for i in 1..=l {
                if i > MAX_T {
                    // l grew beyond what a <= t-error pattern could produce:
                    // uncorrectable, bail out before any further indexing.
                    return Err(FecError::DecoderNotConverged);
                }
                delta ^= self.mul(c[i], s[n_ - i]);
            }

            if delta == 0 {
                m += 1;
            } else if 2 * l <= n_ {
                let t_poly = c; // [u8; N] is Copy - stack copy, no allocation.
                let coef = self.mul(delta, self.inv(b));
                for idx in 0..=MAX_T {
                    if idx >= m {
                        c[idx] ^= self.mul(coef, b_poly[idx - m]);
                    }
                }
                let new_l = n_ + 1 - l;
                if new_l > MAX_T {
                    return Err(FecError::DecoderNotConverged);
                }
                l = new_l;
                b_poly = t_poly;
                b = delta;
                m = 1;
            } else {
                let coef = self.mul(delta, self.inv(b));
                for idx in 0..=MAX_T {
                    if idx >= m {
                        c[idx] ^= self.mul(coef, b_poly[idx - m]);
                    }
                }
                m += 1;
            }
        }

        if l == 0 || l > self.t {
            return Err(FecError::DecoderNotConverged);
        }

        // --- Chien search: find roots of Lambda(x) among alpha^0..alpha^254.
        let mut corrected = 0usize;
        for u in 0..255usize {
            let beta = self.exp[u];
            let mut val = c[l];
            for jx in (0..l).rev() {
                val = self.mul(val, beta) ^ c[jx];
            }
            if val == 0 {
                let idx = (u + 254) % 255; // maps root exponent -> bit position
                codeword[idx] ^= 1;
                corrected += 1;
            }
        }

        if corrected != l {
            // Root count doesn't match Lambda's degree: more errors than the
            // code can correct were present (bounded-distance decoding
            // failure), report rather than trust a partial/inconsistent fix.
            return Err(FecError::DecoderNotConverged);
        }

        Ok(corrected)
    }

    /// Decode and correct an `n`-bit codeword in place using Berlekamp–Massey
    /// and Chien search.
    ///
    /// # Arguments
    ///
    /// * `codeword` — Mutable buffer of exactly `n` bits (`0`/`1` `u8`
    ///   values); corrected in place.
    ///
    /// # Returns
    ///
    /// `Ok(corrected)` — the number of bit errors found and fixed (`0` if the
    /// codeword was already valid).
    ///
    /// # Errors
    ///
    /// [`FecError::DecoderNotConverged`] if more than `t` errors are
    /// detected (Berlekamp–Massey/Chien inconsistency — locator degree
    /// exceeds `t`, or the root count does not match the locator degree).
    /// [`FecError::BufferTooSmall`] if `codeword.len() != n`.
    ///
    /// # Examples
    ///
    /// ```
    /// use glezer_rsv::BchCode;
    /// let bch = BchCode::new(2).unwrap();
    /// let info = vec![1u8; bch.k()];
    /// let mut codeword = vec![0u8; bch.n()];
    /// bch.encode(&info, &mut codeword).unwrap();
    /// codeword[10] ^= 1; // single bit error
    /// let corrected = bch.decode(&mut codeword).unwrap();
    /// assert_eq!(corrected, 1);
    /// assert_eq!(&codeword[..bch.k()], &info[..]);
    /// ```
    pub fn decode(&self, codeword: &mut [u8]) -> Result<usize, FecError> {
        self.decode_full(codeword)
    }

    /// Decode a shortened codeword produced by
    /// [`BchCode::shortened_encode`].
    ///
    /// Internally reconstructs the full `n`-bit codeword by prepending the
    /// `k - k_short` implicit zero bits into a fixed-size stack buffer (no
    /// heap allocation), runs the ordinary algebraic decoder, then copies
    /// the corrected information/parity bits back into `codeword`.
    ///
    /// # Arguments
    ///
    /// * `k_short` — Number of information bits carried, `<= k` (must match
    ///   the value used at encode time).
    /// * `codeword` — Mutable buffer of exactly `k_short + (n-k)` bits.
    ///
    /// # Returns
    ///
    /// `Ok(corrected)` — number of bit errors found and fixed.
    ///
    /// # Errors
    ///
    /// * [`FecError::InvalidParam`] if `k_short > k`.
    /// * [`FecError::BufferTooSmall`] if `codeword` has the wrong length.
    /// * [`FecError::DecoderNotConverged`] if more than `t` errors are
    ///   detected, or if the underlying decoder places a "correction" inside
    ///   the implicit always-zero prefix (which is never itself a bit error
    ///   for a validly shortened codeword — treated defensively as
    ///   uncorrectable).
    ///
    /// # Examples
    ///
    /// ```
    /// use glezer_rsv::BchCode;
    /// let bch = BchCode::new(3).unwrap();
    /// let k_short = 32;
    /// let info = vec![1u8, 0].into_iter().cycle().take(k_short).collect::<Vec<u8>>();
    /// let mut codeword = vec![0u8; k_short + bch.parity_len()];
    /// bch.shortened_encode(k_short, &info, &mut codeword).unwrap();
    /// codeword[5] ^= 1;
    /// let corrected = bch.shortened_decode(k_short, &mut codeword).unwrap();
    /// assert_eq!(corrected, 1);
    /// assert_eq!(&codeword[..k_short], &info[..]);
    /// ```
    pub fn shortened_decode(&self, k_short: usize, codeword: &mut [u8]) -> Result<usize, FecError> {
        if k_short > self.k {
            return Err(FecError::InvalidParam("k_short must be <= k"));
        }
        let nk = self.nk;
        let out_len = k_short + nk;
        if codeword.len() != out_len {
            return Err(FecError::BufferTooSmall {
                required: out_len,
                provided: codeword.len(),
            });
        }

        let pad = self.k - k_short;
        let mut full = [0u8; N];
        full[pad..pad + k_short].copy_from_slice(&codeword[..k_short]);
        full[self.k..self.n].copy_from_slice(&codeword[k_short..k_short + nk]);

        let corrected = self.decode_full(&mut full)?;

        if full[..pad].iter().any(|&b| b != 0) {
            return Err(FecError::DecoderNotConverged);
        }

        codeword[..k_short].copy_from_slice(&full[pad..pad + k_short]);
        codeword[k_short..k_short + nk].copy_from_slice(&full[self.k..self.n]);
        Ok(corrected)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal deterministic xorshift64 PRNG for reproducible tests (same
    /// shift triplet as `channel_sim::AwgnChannel`).
    struct Xorshift64 {
        state: u64,
    }

    impl Xorshift64 {
        fn new(seed: u64) -> Self {
            Self { state: seed | 1 }
        }

        fn next_u64(&mut self) -> u64 {
            let mut x = self.state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.state = x;
            x
        }

        fn next_bit(&mut self) -> u8 {
            (self.next_u64() & 1) as u8
        }

        /// Uniform integer in `0..bound`.
        fn next_range(&mut self, bound: usize) -> usize {
            (self.next_u64() % bound as u64) as usize
        }
    }

    fn random_bits(rng: &mut Xorshift64, len: usize) -> Vec<u8> {
        (0..len).map(|_| rng.next_bit()).collect()
    }

    /// Flip exactly `count` distinct, randomly chosen bit positions in
    /// `buf`, returning the positions flipped.
    fn inject_errors(rng: &mut Xorshift64, buf: &mut [u8], count: usize) -> Vec<usize> {
        let mut used = [false; N];
        let mut positions = Vec::with_capacity(count);
        while positions.len() < count {
            let pos = rng.next_range(buf.len());
            if !used[pos] {
                used[pos] = true;
                positions.push(pos);
                buf[pos] ^= 1;
            }
        }
        positions
    }

    // -----------------------------------------------------------------
    // Structural tests
    // -----------------------------------------------------------------

    #[test]
    fn generator_degree_equals_n_minus_k() {
        for t in 1..=10usize {
            let bch = BchCode::new(t).unwrap();
            assert_eq!(bch.parity_len(), bch.n() - bch.k(), "t={t}");
            // Sanity: deg g(x) should be strictly positive and monotonically
            // non-decreasing as t grows (more errors require more parity).
            assert!(bch.parity_len() > 0, "t={t}");
        }
        // Report the actually-computed (t, n, k) table (not hardcoded).
        for t in 1..=10usize {
            let bch = BchCode::new(t).unwrap();
            println!(
                "t={} -> n={} k={} parity={}",
                t,
                bch.n(),
                bch.k(),
                bch.parity_len()
            );
        }
    }

    #[test]
    fn dimension_shrinks_as_t_grows() {
        let mut prev_k = usize::MAX;
        for t in 1..=10usize {
            let bch = BchCode::new(t).unwrap();
            assert!(
                bch.k() <= prev_k,
                "k should not increase as t grows: t={t} k={} prev_k={prev_k}",
                bch.k()
            );
            prev_k = bch.k();
        }
    }

    #[test]
    fn invalid_t_rejected() {
        assert!(BchCode::new(0).is_err());
        assert!(BchCode::new(11).is_err());
    }

    /// g(x) must divide x^255 + 1 over GF(2): x^255 mod g(x) == 1.
    #[test]
    fn generator_divides_x255_plus_1() {
        for t in 1..=10usize {
            let bch = BchCode::new(t).unwrap();
            let nk = bch.parity_len();
            let mask: u128 = (1u128 << nk) - 1;
            let g_low = bch.g_full & mask;
            // Iteratively compute x^255 mod g(x) via repeated "multiply by x,
            // reduce" steps starting from the constant polynomial 1.
            let mut r: u128 = 1;
            for _ in 0..255 {
                r <<= 1;
                if (r >> nk) & 1 == 1 {
                    r ^= (1u128 << nk) | g_low;
                }
            }
            assert_eq!(r, 1, "x^255 mod g(x) should be 1 for t={t}");
        }
    }

    /// Every encoded codeword must have all-zero syndromes (decode(0 errors)
    /// == Ok(0)).
    #[test]
    fn encoded_codewords_have_zero_syndromes() {
        let mut rng = Xorshift64::new(12345);
        for t in [1usize, 2, 4, 8] {
            let bch = BchCode::new(t).unwrap();
            for _ in 0..10 {
                let info = random_bits(&mut rng, bch.k());
                let mut codeword = vec![0u8; bch.n()];
                bch.encode(&info, &mut codeword).unwrap();
                let mut check = codeword.clone();
                assert_eq!(bch.decode(&mut check), Ok(0), "t={t}");
                assert_eq!(check, codeword, "t={t}: no bits should change");
            }
        }
    }

    // -----------------------------------------------------------------
    // Round-trip tests: exactly j errors, j = 0..=t
    // -----------------------------------------------------------------

    #[test]
    fn round_trip_exact_error_counts() {
        for t in [1usize, 2, 4, 8] {
            let bch = BchCode::new(t).unwrap();
            for j in 0..=t {
                for trial in 0..10u64 {
                    let mut rng = Xorshift64::new(1000 * t as u64 + 10 * j as u64 + trial + 1);
                    let info = random_bits(&mut rng, bch.k());
                    let mut codeword = vec![0u8; bch.n()];
                    bch.encode(&info, &mut codeword).unwrap();

                    let mut received = codeword.clone();
                    inject_errors(&mut rng, &mut received, j);

                    let result = bch.decode(&mut received);
                    assert_eq!(
                        result,
                        Ok(j),
                        "t={t} j={j} trial={trial}: expected {j} corrected bits"
                    );
                    assert_eq!(
                        received, codeword,
                        "t={t} j={j} trial={trial}: decoded codeword mismatch"
                    );
                    assert_eq!(
                        &received[..bch.k()],
                        &info[..],
                        "t={t} j={j} trial={trial}: info bits mismatch"
                    );
                }
            }
        }
    }

    // -----------------------------------------------------------------
    // Beyond-capacity behavior: must never panic; Err or valid miscorrection.
    // -----------------------------------------------------------------

    #[test]
    fn beyond_capacity_never_panics() {
        let t = 2usize;
        let bch = BchCode::new(t).unwrap();
        for extra_errors in [4usize, 5usize] {
            for trial in 0..30u64 {
                let mut rng = Xorshift64::new(50_000 + extra_errors as u64 * 100 + trial + 1);
                let info = random_bits(&mut rng, bch.k());
                let mut codeword = vec![0u8; bch.n()];
                bch.encode(&info, &mut codeword).unwrap();

                let mut received = codeword.clone();
                inject_errors(&mut rng, &mut received, extra_errors);

                match bch.decode(&mut received) {
                    Err(_) => {
                        // Acceptable: decoder correctly detected an
                        // uncorrectable pattern.
                    }
                    Ok(_) => {
                        // Acceptable only if it landed on *some* valid
                        // codeword (zero syndromes on re-check).
                        let mut recheck = received.clone();
                        assert_eq!(
                            bch.decode(&mut recheck),
                            Ok(0),
                            "miscorrection must land on a valid codeword \
                             (extra_errors={extra_errors} trial={trial})"
                        );
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------
    // Shortened code round-trip.
    // -----------------------------------------------------------------

    #[test]
    fn shortened_round_trip() {
        let mut rng = Xorshift64::new(777);
        for t in [1usize, 2, 4] {
            let bch = BchCode::new(t).unwrap();
            let k_short = bch.k() / 4; // simulate a small NAND ECC region
            for j in 0..=t {
                let info = random_bits(&mut rng, k_short);
                let mut codeword = vec![0u8; k_short + bch.parity_len()];
                bch.shortened_encode(k_short, &info, &mut codeword).unwrap();

                let mut received = codeword.clone();
                inject_errors(&mut rng, &mut received, j);

                let result = bch.shortened_decode(k_short, &mut received);
                assert_eq!(result, Ok(j), "t={t} k_short={k_short} j={j}");
                assert_eq!(&received[..k_short], &info[..], "t={t} j={j}");
            }
        }
    }

    #[test]
    fn shortened_encode_rejects_bad_length() {
        let bch = BchCode::new(2).unwrap();
        let info = vec![0u8; 5];
        let mut out = vec![0u8; 5 + bch.parity_len()];
        // k_short too large.
        assert!(bch.shortened_encode(bch.k() + 1, &info, &mut out).is_err());
    }

    #[test]
    fn encode_rejects_wrong_lengths() {
        let bch = BchCode::new(2).unwrap();
        let mut codeword = vec![0u8; bch.n()];
        let bad_info = vec![0u8; bch.k() - 1];
        assert!(bch.encode(&bad_info, &mut codeword).is_err());

        let info = vec![0u8; bch.k()];
        let mut short_codeword = vec![0u8; bch.n() - 1];
        assert!(bch.encode(&info, &mut short_codeword).is_err());
    }

    #[test]
    fn decode_rejects_wrong_length() {
        let bch = BchCode::new(2).unwrap();
        let mut bad = vec![0u8; bch.n() - 1];
        assert!(bch.decode(&mut bad).is_err());
    }
}
