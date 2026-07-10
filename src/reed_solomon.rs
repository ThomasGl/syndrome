//! Reed–Solomon encoder/decoder (systematic, erasure-only repair).
//!
//! This implementation provides an efficient, systematic encoder and an
//! erasure-only decoder capable of reconstructing missing data shards when
//! the number of erasures does not exceed the number of parity shards. It
//! uses precomputed log/exp tables for GF(256) arithmetic (primitive
//! polynomial 0x11D) and precomputes encoding coefficients.

use std::cmp;

use crate::error::FecError;

#[repr(align(64))]
struct AlignedExp([u8; 512]);

pub struct GfTables {
    exp: Box<AlignedExp>, // length 512 for easy wrap
    log: Box<[u8; 256]>,
}

impl Default for GfTables {
    fn default() -> Self {
        Self::new()
    }
}

impl GfTables {
    pub fn new() -> Self {
        // Build exp and log tables with primitive polynomial 0x11D
        let mut exp = [0u8; 512];
        let mut log = [0u8; 256];

        let mut x: u8 = 1;
        for i in 0..255usize {
            exp[i] = x;
            log[x as usize] = i as u8;
            // multiply by primitive (x *= 2)
            let hi = (x & 0x80) != 0;
            x <<= 1;
            if hi {
                x ^= 0x1D;
            }
        }
        for i in 255usize..512usize {
            exp[i] = exp[i - 255];
        }

        GfTables {
            exp: Box::new(AlignedExp(exp)),
            log: Box::new(log),
        }
    }

    #[inline]
    fn mul(&self, a: u8, b: u8) -> u8 {
        if a == 0 || b == 0 {
            return 0;
        }
        let la = self.log[a as usize] as usize;
        let lb = self.log[b as usize] as usize;
        let idx = la + lb;
        self.exp.0[idx]
    }

    #[inline]
    fn div(&self, a: u8, b: u8) -> u8 {
        if b == 0 {
            panic!("division by zero in GF(256)");
        }
        if a == 0 {
            return 0;
        }
        let la = self.log[a as usize] as isize;
        let lb = self.log[b as usize] as isize;
        let mut idx = la - lb;
        while idx < 0 {
            idx += 255;
        }
        self.exp.0[idx as usize]
    }

    #[inline]
    fn pow_alpha(&self, power: usize) -> u8 {
        // return alpha^power
        self.exp.0[power % 255]
    }
}

/// Systematic Reed-Solomon erasure codec over GF(256).
///
/// The recommended API surface is small:
///
/// 1. [`ReedSolomon::new`] — build the codec for a `(data, parity)` geometry.
/// 2. [`ReedSolomon::precompute_mul_tables`] — one-time table build (64 KiB)
///    that unlocks the fast encode/decode paths; skip it only if memory is
///    tighter than throughput.
/// 3. [`ReedSolomon::encode_with_avx2`] — the fast path: runtime-detects
///    AVX2 on x86_64 (NEON on AArch64 is used unconditionally — it is
///    mandatory on that architecture) and falls back to portable
///    table/scalar code, so it is safe to call unconditionally on any
///    hardware.
/// 4. [`ReedSolomon::encode_into`] (zero-allocation, caller buffer) or
///    [`ReedSolomon::encode`] (convenience, allocates) — portable scalar
///    paths that need no precomputed tables.
/// 5. [`ReedSolomon::decode`] — erasure repair; automatically uses the same
///    table/SIMD machinery as the encoder.
///
/// The crate also ships several `#[doc(hidden)]` encode variants
/// (`encode_with_tables`, `encode_with_tables_chunked`, `encode_simd`,
/// `encode_optimized`, `encode_soa`). They are intermediate steps of the
/// reproducible optimization study in `bench/` — kept public so the
/// benchmark harness and its published numbers stay honest — but they are
/// not part of the supported API and may change without notice.
pub struct ReedSolomon {
    pub data_shards: usize,
    pub parity_shards: usize,
    // encoding coefficients: parity_shards x data_shards, row-major
    coeffs: Vec<u8>,
    tables: GfTables,
    // optional multiplication tables indexed by coefficient value; stored as a
    // flat, contiguous Vec of fixed-size LUTs (no per-table heap indirection)
    mul_tables: Option<Vec<[u8; 256]>>,
    // Companion 16-entry lo/hi nibble decompositions of `mul_tables`, one
    // `[lo, hi]` pair per coefficient (256 x 32 B = 8 KiB total), consumed by
    // the AVX2 VPSHUFB kernel. `nibbles[c][0][v] = GF_mul(c, v)` and
    // `nibbles[c][1][v] = GF_mul(c, v << 4)` for `v` in `0..16`, exploiting
    // GF(256) linearity: `GF_mul(c, x) = GF_mul(c, x & 0x0F) ^
    // GF_mul(c, x & 0xF0)`. Previously these 32 bytes were re-derived from
    // `mul_tables` inside `encode_with_avx2_inner`/`gf_muladd_bulk` on every
    // (coefficient, row) pair per call; now they are built once alongside
    // `mul_tables` and only indexed on the hot path.
    nibble_tables: Option<Vec<[[u8; 16]; 2]>>,
}

impl ReedSolomon {
    /// Validate the `data`/`parity_out` shapes shared by every `encode_*`
    /// variant, and return the common shard length on success.
    ///
    /// This is an entry-level precondition check only — it runs once per
    /// `encode_*` call, never inside the per-byte inner loops.
    ///
    /// # Errors
    ///
    /// Returns [`FecError::InvalidParam`] if the codec was constructed with
    /// zero data shards, if `data.len()` does not equal
    /// [`ReedSolomon::data_shards`], or if the shards have inconsistent
    /// lengths. Returns [`FecError::BufferTooSmall`] if `parity_out` cannot
    /// hold `parity_shards * shard_len` bytes.
    #[inline]
    fn validate_encode_inputs(&self, data: &[&[u8]], parity_out: &[u8]) -> Result<usize, FecError> {
        if self.data_shards == 0 {
            return Err(FecError::InvalidParam("data_shards must be > 0"));
        }
        if data.len() != self.data_shards {
            return Err(FecError::InvalidParam("data length must equal data_shards"));
        }
        let shard_len = data.first().map_or(0, |s| s.len());
        if data.iter().any(|s| s.len() != shard_len) {
            return Err(FecError::InvalidParam("shard size mismatch"));
        }
        let required = self.parity_shards * shard_len;
        if parity_out.len() < required {
            return Err(FecError::BufferTooSmall {
                required,
                provided: parity_out.len(),
            });
        }
        Ok(shard_len)
    }

    /// Precompute multiplication lookup tables for all 256 possible coefficients.
    /// This trades memory for faster inner loops by replacing GF multiplications
    /// with a single table lookup per byte.
    pub fn precompute_mul_tables(&mut self) {
        if self.mul_tables.is_some() {
            return;
        }
        let mut tables: Vec<[u8; 256]> = Vec::with_capacity(256);
        let mut nibbles: Vec<[[u8; 16]; 2]> = Vec::with_capacity(256);
        for coef in 0u8..=255u8 {
            let mut tbl = [0u8; 256];
            if coef == 0 {
                // all zeros
            } else {
                for v in 0u8..=255u8 {
                    tbl[v as usize] = self.tables.mul(coef, v);
                }
            }
            // Companion lo/hi nibble decomposition for the AVX2 VPSHUFB
            // kernel (see the field doc on `nibble_tables`): built here,
            // once per coefficient, instead of being re-derived from `tbl`
            // on every hot-path call.
            let mut lo = [0u8; 16];
            let mut hi = [0u8; 16];
            for v in 0usize..16 {
                lo[v] = tbl[v];
                hi[v] = tbl[v << 4];
            }
            tables.push(tbl);
            nibbles.push([lo, hi]);
        }
        self.mul_tables = Some(tables);
        self.nibble_tables = Some(nibbles);
    }

    /// Encode using precomputed multiplication tables for coefficients.
    ///
    /// Benchmark-study step (see the struct-level docs); prefer
    /// [`ReedSolomon::encode_with_avx2`] or [`ReedSolomon::encode_into`].
    ///
    /// # Errors
    ///
    /// Returns [`FecError::InvalidParam`] if [`ReedSolomon::precompute_mul_tables`]
    /// has not been called yet, or if `data.len()` / shard lengths are
    /// inconsistent. Returns [`FecError::BufferTooSmall`] if `parity_out` is
    /// too small.
    #[doc(hidden)]
    pub fn encode_with_tables(
        &self,
        data: &[&[u8]],
        parity_out: &mut [u8],
    ) -> Result<(), FecError> {
        let d = self.data_shards;
        let p = self.parity_shards;
        let Some(tables) = self.mul_tables.as_ref() else {
            return Err(FecError::InvalidParam(
                "mul tables not precomputed — call precompute_mul_tables() first",
            ));
        };
        let shard_len = self.validate_encode_inputs(data, parity_out)?;
        for b in parity_out.iter_mut().take(p * shard_len) {
            *b = 0;
        }

        for j in 0..d {
            let dj = data[j];
            for i in 0..p {
                let coef = self.coeffs[i * d + j];
                if coef == 0 {
                    continue;
                }
                let table = &tables[coef as usize];
                let parity_row = &mut parity_out[i * shard_len..i * shard_len + shard_len];
                for k in 0..shard_len {
                    parity_row[k] ^= table[dj[k] as usize];
                }
            }
        }
        Ok(())
    }

    /// Encode using precomputed multiplication tables but assemble 8-byte
    /// words from table lookups and XOR them as `u64` to reduce per-byte
    /// overhead. This combines table lookups (fast mul) with chunked
    /// word-wise XOR to approach the speed of pure XOR paths.
    ///
    /// Benchmark-study step (see the struct-level docs); prefer
    /// [`ReedSolomon::encode_with_avx2`], which falls back to this exact
    /// path when AVX2 is unavailable.
    ///
    /// # Errors
    ///
    /// Returns [`FecError::InvalidParam`] if [`ReedSolomon::precompute_mul_tables`]
    /// has not been called yet, or if `data.len()` / shard lengths are
    /// inconsistent. Returns [`FecError::BufferTooSmall`] if `parity_out` is
    /// too small.
    #[doc(hidden)]
    pub fn encode_with_tables_chunked(
        &self,
        data: &[&[u8]],
        parity_out: &mut [u8],
    ) -> Result<(), FecError> {
        let d = self.data_shards;
        let p = self.parity_shards;
        let Some(tables) = self.mul_tables.as_ref() else {
            return Err(FecError::InvalidParam(
                "mul tables not precomputed — call precompute_mul_tables() first",
            ));
        };
        let shard_len = self.validate_encode_inputs(data, parity_out)?;
        for b in parity_out.iter_mut().take(p * shard_len) {
            *b = 0;
        }

        for j in 0..d {
            let dj = data[j];
            for i in 0..p {
                let coef = self.coeffs[i * d + j];
                if coef == 0 {
                    continue;
                }
                let table = &tables[coef as usize];
                let parity_row = &mut parity_out[i * shard_len..i * shard_len + shard_len];

                let mut k = 0usize;
                while k + 8 <= shard_len {
                    let mut acc_bytes = [0u8; 8];
                    for t in 0..8 {
                        acc_bytes[t] = table[dj[k + t] as usize];
                    }
                    let acc_u64 = u64::from_ne_bytes(acc_bytes);
                    let p_u64 = u64::from_ne_bytes(parity_row[k..k + 8].try_into().unwrap());
                    let v = p_u64 ^ acc_u64;
                    parity_row[k..k + 8].copy_from_slice(&v.to_ne_bytes());
                    k += 8;
                }
                for t in k..shard_len {
                    parity_row[t] ^= table[dj[t] as usize];
                }
            }
        }
        Ok(())
    }

    /// Encode using SIMD for XOR-case (coef==1) to accelerate pure XOR
    /// accumulation. Other coefficients currently fall back to scalar mul.
    ///
    /// Benchmark-study step (see the struct-level docs); prefer
    /// [`ReedSolomon::encode_with_avx2`] or [`ReedSolomon::encode_into`].
    ///
    /// # Errors
    ///
    /// Returns [`FecError::InvalidParam`] if `data.len()` / shard lengths are
    /// inconsistent. Returns [`FecError::BufferTooSmall`] if `parity_out` is
    /// too small.
    #[doc(hidden)]
    pub fn encode_simd(&self, data: &[&[u8]], parity_out: &mut [u8]) -> Result<(), FecError> {
        // Use stable word-sized XOR (u64) for the pure-XOR coefficient case
        // to accelerate accumulation without relying on unstable `std::simd`.
        let d = self.data_shards;
        let p = self.parity_shards;
        let shard_len = self.validate_encode_inputs(data, parity_out)?;
        for b in parity_out.iter_mut().take(p * shard_len) {
            *b = 0;
        }

        for j in 0..d {
            let dj = data[j];
            for i in 0..p {
                let coef = self.coeffs[i * d + j];
                if coef == 0 {
                    continue;
                }
                let parity_row = &mut parity_out[i * shard_len..i * shard_len + shard_len];
                if coef == 1 {
                    // process 8 bytes at a time using u64 XOR
                    let mut k = 0usize;
                    while k + 8 <= shard_len {
                        let a = u64::from_ne_bytes(parity_row[k..k + 8].try_into().unwrap());
                        let b = u64::from_ne_bytes(dj[k..k + 8].try_into().unwrap());
                        let v = a ^ b;
                        parity_row[k..k + 8].copy_from_slice(&v.to_ne_bytes());
                        k += 8;
                    }
                    for t in k..shard_len {
                        parity_row[t] ^= dj[t];
                    }
                } else {
                    for k in 0..shard_len {
                        parity_row[k] ^= self.tables.mul(coef, dj[k]);
                    }
                }
            }
        }
        Ok(())
    }
    /// Encode using the fastest SIMD kernel available for the running
    /// architecture, applied to all non-zero coefficients.
    ///
    /// For each coefficient, decomposes the 256-byte GF multiply table into two
    /// 16-byte nibble tables and applies them via a native SIMD table lookup:
    /// AVX2 VPSHUFB (32 bytes/iteration) on x86_64, or NEON `vqtbl1q_u8` (16
    /// bytes/iteration) on AArch64 (selected by a private per-architecture
    /// dispatcher). Falls back to the portable chunked-table path (or, if
    /// `precompute_mul_tables` was never called, a plain scalar encode) when
    /// no SIMD kernel is available at runtime.
    ///
    /// # Errors
    ///
    /// Returns [`FecError::InvalidParam`] if `data.len()` / shard lengths are
    /// inconsistent. Returns [`FecError::BufferTooSmall`] if `parity_out` is
    /// too small. A missing `precompute_mul_tables()` call is not an error
    /// here — this method falls back to a scalar encode instead.
    pub fn encode_with_avx2(&self, data: &[&[u8]], parity_out: &mut [u8]) -> Result<(), FecError> {
        let shard_len = self.validate_encode_inputs(data, parity_out)?;
        if self.mul_tables.is_some() {
            if self.encode_simd_dispatch(data, parity_out) {
                return Ok(());
            }
            // No SIMD kernel available for this target/runtime.
            return self.encode_with_tables_chunked(data, parity_out);
        }
        // Defensive: tables not ready; use scalar path via encode_into.
        let d = self.data_shards;
        let p = self.parity_shards;
        for b in parity_out.iter_mut().take(p * shard_len) {
            *b = 0;
        }
        for j in 0..d {
            for i in 0..p {
                let coef = self.coeffs[i * d + j];
                if coef == 0 {
                    continue;
                }
                let pr = &mut parity_out[i * shard_len..i * shard_len + shard_len];
                for k in 0..shard_len {
                    pr[k] ^= self.tables.mul(coef, data[j][k]);
                }
            }
        }
        Ok(())
    }

    /// Private dispatcher: picks the fastest SIMD encode kernel for the
    /// running architecture. Requires `self.mul_tables`/`nibble_tables` to
    /// already be built (caller-checked).
    ///
    /// # Returns
    ///
    /// `true` if a SIMD kernel ran (and wrote all of `parity_out`); `false`
    /// if none is available (no AVX2 at runtime on x86_64, or a target that
    /// is neither x86_64 nor aarch64), in which case the caller must fall
    /// back to a portable path itself.
    fn encode_simd_dispatch(&self, data: &[&[u8]], parity_out: &mut [u8]) -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                self.encode_with_avx2_inner(data, parity_out);
                return true;
            }
        }
        // AArch64: NEON is mandatory, so no runtime detection is needed.
        #[cfg(target_arch = "aarch64")]
        {
            self.encode_with_neon_inner(data, parity_out);
            return true;
        }
        #[allow(unreachable_code)]
        {
            let _ = (data, parity_out);
            false
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn encode_with_avx2_inner(&self, data: &[&[u8]], parity_out: &mut [u8]) {
        let d = self.data_shards;
        let p = self.parity_shards;
        let tables = self.mul_tables.as_ref().unwrap();
        // Built together with `mul_tables` in `precompute_mul_tables`, so
        // this is always `Some` whenever `mul_tables` is.
        let nibbles = self.nibble_tables.as_ref().unwrap();
        let shard_len = data[0].len();
        for b in parity_out.iter_mut().take(p * shard_len) {
            *b = 0;
        }

        for j in 0..d {
            let dj = data[j];
            for i in 0..p {
                let coef = self.coeffs[i * d + j];
                if coef == 0 {
                    continue;
                }
                let full_table: &[u8; 256] = &tables[coef as usize];
                // Precomputed 16-byte lo/hi nibble tables (see
                // `nibble_tables`'s field doc) -- indexed, not rebuilt.
                let [lo_tbl, hi_tbl] = &nibbles[coef as usize];
                let parity_row = &mut parity_out[i * shard_len..i * shard_len + shard_len];

                // SAFETY: AVX2 presence was verified above by is_x86_feature_detected.
                unsafe {
                    crate::simd_avx2::gf256_muladd_avx2(dj, parity_row, lo_tbl, hi_tbl, full_table);
                }
            }
        }
    }

    /// NEON counterpart of [`Self::encode_with_avx2_inner`]. NEON is
    /// mandatory on AArch64, so this is unconditionally safe to call (no
    /// runtime feature check needed).
    #[cfg(target_arch = "aarch64")]
    fn encode_with_neon_inner(&self, data: &[&[u8]], parity_out: &mut [u8]) {
        let d = self.data_shards;
        let p = self.parity_shards;
        let tables = self.mul_tables.as_ref().unwrap();
        // Built together with `mul_tables` in `precompute_mul_tables`, so
        // this is always `Some` whenever `mul_tables` is.
        let nibbles = self.nibble_tables.as_ref().unwrap();
        let shard_len = data[0].len();
        for b in parity_out.iter_mut().take(p * shard_len) {
            *b = 0;
        }

        for j in 0..d {
            let dj = data[j];
            for i in 0..p {
                let coef = self.coeffs[i * d + j];
                if coef == 0 {
                    continue;
                }
                let full_table: &[u8; 256] = &tables[coef as usize];
                // Precomputed 16-byte lo/hi nibble tables (see
                // `nibble_tables`'s field doc) -- indexed, not rebuilt.
                let [lo_tbl, hi_tbl] = &nibbles[coef as usize];
                let parity_row = &mut parity_out[i * shard_len..i * shard_len + shard_len];

                // SAFETY: NEON is mandatory on AArch64.
                unsafe {
                    crate::simd_neon::gf256_muladd_neon(dj, parity_row, lo_tbl, hi_tbl, full_table);
                }
            }
        }
    }

    /// Create a new Reed-Solomon codec. It precomputes GF tables and the
    /// encoding coefficient matrix for a systematic encoding where data
    /// shards are left unchanged and parity shards are linear combinations of
    /// data shards.
    pub fn new(data_shards: usize, parity_shards: usize) -> Self {
        let tables = GfTables::new();
        let mut coeffs = vec![0u8; parity_shards * data_shards];
        for i in 0..parity_shards {
            for j in 0..data_shards {
                // coefficient = alpha^{i * j}
                coeffs[i * data_shards + j] = tables.pow_alpha(i * j);
            }
        }
        ReedSolomon {
            data_shards,
            parity_shards,
            coeffs,
            tables,
            mul_tables: None,
            nibble_tables: None,
        }
    }

    /// Efficiently encode into a preallocated flat parity buffer. `data` is
    /// a slice of references to data shards (each with equal length). `parity_out`
    /// must be at least `parity_shards * shard_len` bytes. Parity rows are
    /// stored consecutively: row0[..], row1[..], ...
    ///
    /// # Errors
    ///
    /// Returns [`FecError::InvalidParam`] if `data.len()` / shard lengths are
    /// inconsistent. Returns [`FecError::BufferTooSmall`] if `parity_out` is
    /// too small.
    pub fn encode_into(&self, data: &[&[u8]], parity_out: &mut [u8]) -> Result<(), FecError> {
        let d = self.data_shards;
        let p = self.parity_shards;
        let shard_len = self.validate_encode_inputs(data, parity_out)?;

        // zero parity buffer
        for b in parity_out.iter_mut().take(p * shard_len) {
            *b = 0;
        }

        // Optimized inner loop ordering: iterate data shards outer, parity rows
        // inner. This reduces repeated indexing into data and keeps parity rows
        // contiguous for writes which helps cache locality for typical parity
        // sizes.
        for j in 0..d {
            let dj = data[j];
            for i in 0..p {
                let coef = self.coeffs[i * d + j];
                if coef == 0 {
                    continue;
                }
                let parity_row = &mut parity_out[i * shard_len..i * shard_len + shard_len];
                if coef == 1 {
                    // XOR slice
                    for k in 0..shard_len {
                        parity_row[k] ^= dj[k];
                    }
                } else {
                    for k in 0..shard_len {
                        parity_row[k] ^= self.tables.mul(coef, dj[k]);
                    }
                }
            }
        }
        Ok(())
    }

    /// Alternative optimized encoder that processes in chunks for slightly
    /// better inner-loop locality. This is a drop-in replacement for
    /// microbenchmarking and may be extended with SIMD later.
    ///
    /// Benchmark-study step (see the struct-level docs); prefer
    /// [`ReedSolomon::encode_with_avx2`] or [`ReedSolomon::encode_into`].
    ///
    /// # Errors
    ///
    /// Returns [`FecError::InvalidParam`] if `data.len()` / shard lengths are
    /// inconsistent. Returns [`FecError::BufferTooSmall`] if `parity_out` is
    /// too small.
    #[doc(hidden)]
    pub fn encode_optimized(&self, data: &[&[u8]], parity_out: &mut [u8]) -> Result<(), FecError> {
        let d = self.data_shards;
        let p = self.parity_shards;
        let shard_len = self.validate_encode_inputs(data, parity_out)?;
        let chunk = 16usize;
        for i in 0..(p * shard_len) {
            parity_out[i] = 0;
        }
        for j in 0..d {
            let dj = data[j];
            for i in 0..p {
                let coef = self.coeffs[i * d + j];
                if coef == 0 {
                    continue;
                }
                let parity_row = &mut parity_out[i * shard_len..i * shard_len + shard_len];
                if coef == 1 {
                    for k in (0..shard_len).step_by(chunk) {
                        let end = cmp::min(k + chunk, shard_len);
                        for t in k..end {
                            parity_row[t] ^= dj[t];
                        }
                    }
                } else {
                    for k in (0..shard_len).step_by(chunk) {
                        let end = cmp::min(k + chunk, shard_len);
                        for t in k..end {
                            parity_row[t] ^= self.tables.mul(coef, dj[t]);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Encode using a Struct-of-Arrays (SoA) temporary to improve cache
    /// locality for the per-byte accumulation across data shards.
    ///
    /// Benchmark-study step (see the struct-level docs); prefer
    /// [`ReedSolomon::encode_with_avx2`] or [`ReedSolomon::encode_into`].
    ///
    /// # Errors
    ///
    /// Returns [`FecError::InvalidParam`] if `data.len()` / shard lengths are
    /// inconsistent. Returns [`FecError::BufferTooSmall`] if `parity_out` is
    /// too small.
    #[doc(hidden)]
    pub fn encode_soa(&self, data: &[&[u8]], parity_out: &mut [u8]) -> Result<(), FecError> {
        let d = self.data_shards;
        let p = self.parity_shards;
        let shard_len = self.validate_encode_inputs(data, parity_out)?;

        // Transpose data into SoA: transposed[k * d + j] = data[j][k]
        let mut transposed = vec![0u8; shard_len * d];
        for j in 0..d {
            let dj = data[j];
            for k in 0..shard_len {
                transposed[k * d + j] = dj[k];
            }
        }

        // zero parity
        for b in parity_out.iter_mut().take(p * shard_len) {
            *b = 0;
        }

        for i in 0..p {
            let parity_row = &mut parity_out[i * shard_len..i * shard_len + shard_len];
            let coeff_row = &self.coeffs[i * d..i * d + d];
            for j in 0..d {
                let coef = coeff_row[j];
                if coef == 0 {
                    continue;
                }
                if coef == 1 {
                    for k in 0..shard_len {
                        parity_row[k] ^= transposed[k * d + j];
                    }
                } else {
                    for k in 0..shard_len {
                        parity_row[k] ^= self.tables.mul(coef, transposed[k * d + j]);
                    }
                }
            }
        }
        Ok(())
    }

    /// Backwards-compatible encode: takes `Vec<Vec<u8>>` and returns parity as
    /// `Vec<Vec<u8>>`. Internally uses `encode_into` to avoid repeated allocation
    /// overhead when possible.
    ///
    /// # Errors
    ///
    /// Returns [`FecError::InvalidParam`] if `data.len()` does not equal
    /// `data_shards` or shard lengths are inconsistent.
    pub fn encode(&self, data: &[Vec<u8>]) -> Result<Vec<Vec<u8>>, FecError> {
        let shard_len = data.first().map(|v| v.len()).unwrap_or(0);
        let mut flat: Vec<u8> = vec![0u8; self.parity_shards * shard_len];
        let refs: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
        self.encode_into(&refs, &mut flat)?;
        let mut out = Vec::with_capacity(self.parity_shards);
        for i in 0..self.parity_shards {
            let start = i * shard_len;
            out.push(flat[start..start + shard_len].to_vec());
        }
        Ok(out)
    }

    /// Erasure-only decoder: attempts to reconstruct missing data shards in
    /// `shards` (length must be `data_shards + parity_shards`) where missing
    /// entries are `None`. On success the missing data shards are filled and
    /// `Ok(())` is returned. Requires that the number of missing data shards
    /// is <= `parity_shards` and that at least that many parity shards are
    /// present.
    ///
    /// # Performance
    ///
    /// The $O(p^3)$ matrix inversion is negligible for the small erasure
    /// counts this decoder targets ($p \le$ `parity_shards`, typically a
    /// handful). The dominant cost is the reconstruction multiply, which is
    /// $O(p^2 \cdot \text{shard\_len})$ GF(256) multiply-accumulates. That
    /// step is routed through `Self::gf_muladd_bulk`, which — in order of
    /// preference — uses the runtime-detected AVX2 VPSHUFB kernel (same
    /// kernel [`Self::encode_with_avx2`] uses), falls back to the
    /// precomputed 256×256 tables from [`Self::precompute_mul_tables`]
    /// chunked 8 bytes at a time, and finally falls back to the plain
    /// log/exp `mul` if tables were never precomputed. This mirrors the
    /// table/SIMD machinery the encoder already uses instead of the
    /// per-byte scalar loop the original implementation used.
    ///
    /// # Errors
    ///
    /// Returns [`FecError::BufferTooSmall`] if `shards.len()` does not match
    /// `data_shards + parity_shards`. Returns [`FecError::InvalidParam`] if
    /// shard lengths are inconsistent, if too many data shards are missing
    /// to recover, if not enough parity shards are present, or if the
    /// erasure/parity selection yields a singular recovery matrix.
    pub fn decode(&self, shards: &mut [Option<Vec<u8>>]) -> Result<(), FecError> {
        let total = self.data_shards + self.parity_shards;
        if shards.len() != total {
            return Err(FecError::BufferTooSmall {
                required: total,
                provided: shards.len(),
            });
        }

        // determine shard length
        let shard_len = shards
            .iter()
            .find_map(|s| s.as_ref().map(|v| v.len()))
            .unwrap_or(0);
        for s in shards.iter().filter_map(|s| s.as_ref()) {
            if s.len() != shard_len {
                return Err(FecError::InvalidParam("shard size mismatch"));
            }
        }

        // collect missing data indices
        let mut missing_data = Vec::new();
        for i in 0..self.data_shards {
            if shards[i].is_none() {
                missing_data.push(i);
            }
        }
        if missing_data.is_empty() {
            return Ok(());
        }
        let m = missing_data.len();
        if m > self.parity_shards {
            return Err(FecError::InvalidParam("too many data erasures to recover"));
        }

        // collect available parity rows indices
        let mut available_parity_idx = Vec::new();
        for i in 0..self.parity_shards {
            if shards[self.data_shards + i].is_some() {
                available_parity_idx.push(i);
            }
        }
        if available_parity_idx.len() < m {
            return Err(FecError::InvalidParam("not enough parity shards present"));
        }

        // select first m available parity rows
        let parity_rows_sel = &available_parity_idx[0..m];

        // Build A matrix (m x m) where A[r][c] = coeff for parity_row r and missing_data[c]
        let mut a = vec![0u8; m * m];
        for (r_idx, &pr) in parity_rows_sel.iter().enumerate() {
            for (c_idx, &md) in missing_data.iter().enumerate() {
                a[r_idx * m + c_idx] = self.coeffs[pr * self.data_shards + md];
            }
        }

        // invert A. This is the only O(p^3) step and operates on a tiny m x m
        // matrix (m <= parity_shards), so it is negligible next to the
        // O(p^2 * shard_len) reconstruction multiply below.
        let inv_a = invert_matrix_gf(&a, m, &self.tables)
            .ok_or(FecError::InvalidParam("matrix not invertible"))?;

        // Bulk reconstruction, restructured as flat SoA buffers (m rows x
        // shard_len) so the per-byte work happens inside `gf_muladd_bulk`
        // (AVX2 / table / scalar) rather than a per-byte scalar loop with a
        // heap allocation on every iteration. These two buffers are the only
        // allocations in the whole reconstruction and are call-scoped
        // scratch, not per-byte allocations.
        let mut rhs = vec![0u8; m * shard_len];
        for (r_idx, &pr) in parity_rows_sel.iter().enumerate() {
            let row = &mut rhs[r_idx * shard_len..(r_idx + 1) * shard_len];
            let parity_shard = shards[self.data_shards + pr].as_ref().unwrap();
            row.copy_from_slice(parity_shard);
            // subtract (== XOR, in GF(2^8)) known data contributions in bulk
            for j in 0..self.data_shards {
                if let Some(dj) = shards[j].as_ref() {
                    let coef = self.coeffs[pr * self.data_shards + j];
                    if coef != 0 {
                        self.gf_muladd_bulk(dj, row, coef);
                    }
                }
            }
        }

        // outputs = invA * rhs, computed row-by-row as bulk multiply-accumulates
        // instead of a per-byte dot product.
        let mut outputs = vec![0u8; m * shard_len];
        for c_idx in 0..m {
            let out_row = &mut outputs[c_idx * shard_len..(c_idx + 1) * shard_len];
            for r_idx in 0..m {
                let coef = inv_a[c_idx * m + r_idx];
                if coef == 0 {
                    continue;
                }
                let rhs_row = &rhs[r_idx * shard_len..(r_idx + 1) * shard_len];
                self.gf_muladd_bulk(rhs_row, out_row, coef);
            }
        }

        // place outputs into shards
        for (c_idx, &md) in missing_data.iter().enumerate() {
            shards[md] = Some(outputs[c_idx * shard_len..(c_idx + 1) * shard_len].to_vec());
        }

        Ok(())
    }

    /// Bulk GF(256) multiply-accumulate: `dst[i] ^= GF_mul(coef, src[i])` for
    /// all `i`, with zero heap allocation.
    ///
    /// Used by [`Self::decode`] to route its reconstruction multiply through
    /// the same table/SIMD machinery [`Self::encode_with_avx2`] uses instead
    /// of a per-byte scalar `mul` call. In order of preference:
    ///
    /// 1. Runtime-detected AVX2 VPSHUFB nibble kernel on x86_64 (requires
    ///    [`Self::precompute_mul_tables`] to have been called and AVX2 to be
    ///    available at runtime), or the NEON `vqtbl1q_u8` nibble kernel on
    ///    AArch64 (NEON is mandatory there, so no runtime check is needed) —
    ///    same kernels the encoder uses.
    /// 2. Precomputed 256×256 multiplication tables, applied 8 bytes at a
    ///    time (chunked for cache/ILP), when tables exist but no SIMD kernel
    ///    ran (no AVX2 at runtime on x86_64, or a non-x86_64/aarch64 target).
    /// 3. Plain log/exp `mul`, byte by byte, when tables were never
    ///    precomputed. This keeps `decode` correct and allocation-free even
    ///    if the caller skipped `precompute_mul_tables`; call it first for
    ///    the fast paths above.
    ///
    /// # Arguments
    ///
    /// * `src`  — source bytes.
    /// * `dst`  — accumulator, XOR'd in place; must have the same length as `src`.
    /// * `coef` — GF(256) coefficient to multiply by; a no-op if `0`.
    #[inline]
    fn gf_muladd_bulk(&self, src: &[u8], dst: &mut [u8], coef: u8) {
        debug_assert_eq!(src.len(), dst.len());
        if coef == 0 {
            return;
        }

        #[cfg(target_arch = "x86_64")]
        {
            if let Some(tables) = self.mul_tables.as_ref() {
                if is_x86_feature_detected!("avx2") {
                    let full_table = &tables[coef as usize];
                    // Precomputed 16-byte lo/hi nibble tables (see
                    // `nibble_tables`'s field doc) -- indexed, not rebuilt.
                    // Always `Some` when `mul_tables` is (built together in
                    // `precompute_mul_tables`).
                    let [lo_tbl, hi_tbl] = &self.nibble_tables.as_ref().unwrap()[coef as usize];
                    // SAFETY: AVX2 presence was verified above by is_x86_feature_detected.
                    unsafe {
                        crate::simd_avx2::gf256_muladd_avx2(src, dst, lo_tbl, hi_tbl, full_table);
                    }
                    return;
                }
            }
        }

        // AArch64: NEON is mandatory, so no runtime feature check is needed
        // (mirrors the x86_64 AVX2 branch above).
        #[cfg(target_arch = "aarch64")]
        {
            if let Some(tables) = self.mul_tables.as_ref() {
                let full_table = &tables[coef as usize];
                let [lo_tbl, hi_tbl] = &self.nibble_tables.as_ref().unwrap()[coef as usize];
                // SAFETY: NEON is mandatory on AArch64.
                unsafe {
                    crate::simd_neon::gf256_muladd_neon(src, dst, lo_tbl, hi_tbl, full_table);
                }
                return;
            }
        }

        if let Some(tables) = self.mul_tables.as_ref() {
            let table = &tables[coef as usize];
            let len = src.len();
            let mut k = 0usize;
            while k + 8 <= len {
                let mut acc_bytes = [0u8; 8];
                for t in 0..8 {
                    acc_bytes[t] = table[src[k + t] as usize];
                }
                let acc_u64 = u64::from_ne_bytes(acc_bytes);
                let d_u64 = u64::from_ne_bytes(dst[k..k + 8].try_into().unwrap());
                let v = d_u64 ^ acc_u64;
                dst[k..k + 8].copy_from_slice(&v.to_ne_bytes());
                k += 8;
            }
            for t in k..len {
                dst[t] ^= table[src[t] as usize];
            }
            return;
        }

        // No precomputed tables: fall back to plain log/exp multiply.
        for k in 0..dst.len() {
            dst[k] ^= self.tables.mul(coef, src[k]);
        }
    }

    /// Reference erasure decoder kept for equivalence testing only: the
    /// original per-byte scalar implementation (plain log/exp `mul`, no
    /// tables, no SIMD) that [`Self::decode`] replaced. Semantics —
    /// including error cases — are identical to `decode`; only the inner
    /// multiply strategy differs.
    #[cfg(test)]
    fn decode_scalar_reference(&self, shards: &mut [Option<Vec<u8>>]) -> Result<(), FecError> {
        let total = self.data_shards + self.parity_shards;
        if shards.len() != total {
            return Err(FecError::BufferTooSmall {
                required: total,
                provided: shards.len(),
            });
        }

        let shard_len = shards
            .iter()
            .find_map(|s| s.as_ref().map(|v| v.len()))
            .unwrap_or(0);
        for s in shards.iter().filter_map(|s| s.as_ref()) {
            if s.len() != shard_len {
                return Err(FecError::InvalidParam("shard size mismatch"));
            }
        }

        let mut missing_data = Vec::new();
        for i in 0..self.data_shards {
            if shards[i].is_none() {
                missing_data.push(i);
            }
        }
        if missing_data.is_empty() {
            return Ok(());
        }
        let m = missing_data.len();
        if m > self.parity_shards {
            return Err(FecError::InvalidParam("too many data erasures to recover"));
        }

        let mut available_parity_idx = Vec::new();
        for i in 0..self.parity_shards {
            if shards[self.data_shards + i].is_some() {
                available_parity_idx.push(i);
            }
        }
        if available_parity_idx.len() < m {
            return Err(FecError::InvalidParam("not enough parity shards present"));
        }

        let parity_rows_sel = &available_parity_idx[0..m];

        let mut a = vec![0u8; m * m];
        for (r_idx, &pr) in parity_rows_sel.iter().enumerate() {
            for (c_idx, &md) in missing_data.iter().enumerate() {
                a[r_idx * m + c_idx] = self.coeffs[pr * self.data_shards + md];
            }
        }

        let inv_a = invert_matrix_gf(&a, m, &self.tables)
            .ok_or(FecError::InvalidParam("matrix not invertible"))?;

        let mut outputs: Vec<Vec<u8>> = vec![vec![0u8; shard_len]; m];

        for k in 0..shard_len {
            let mut rhs = vec![0u8; m];
            for (r_idx, &pr) in parity_rows_sel.iter().enumerate() {
                let parity_shard = shards[self.data_shards + pr].as_ref().unwrap();
                let mut acc = parity_shard[k];
                for j in 0..self.data_shards {
                    if let Some(ref dj) = shards[j] {
                        let coef = self.coeffs[pr * self.data_shards + j];
                        if coef != 0 {
                            let prod = self.tables.mul(coef, dj[k]);
                            acc ^= prod;
                        }
                    }
                }
                rhs[r_idx] = acc;
            }

            let x = multiply_matrix_vector_gf(&inv_a, m, &rhs, &self.tables);
            for (c_idx, val) in x.iter().enumerate() {
                outputs[c_idx][k] = *val;
            }
        }

        for (c_idx, &md) in missing_data.iter().enumerate() {
            shards[md] = Some(outputs[c_idx].clone());
        }

        Ok(())
    }
}

fn invert_matrix_gf(mat: &[u8], n: usize, tables: &GfTables) -> Option<Vec<u8>> {
    // Gauss-Jordan to invert matrix over GF(256). mat is row-major n*n.
    let mut a = vec![0u8; n * n * 2]; // augmented matrix [mat | I]
    // left
    for r in 0..n {
        for c in 0..n {
            a[r * (2 * n) + c] = mat[r * n + c];
        }
        a[r * (2 * n) + n + r] = 1;
    }

    // forward elimination
    for col in 0..n {
        // find pivot
        let mut pivot = col;
        while pivot < n && a[pivot * (2 * n) + col] == 0 {
            pivot += 1;
        }
        if pivot == n {
            return None;
        }
        if pivot != col {
            // swap rows
            for k in 0..2 * n {
                a.swap(col * (2 * n) + k, pivot * (2 * n) + k);
            }
        }

        // normalize row col
        let pivot_val = a[col * (2 * n) + col];
        if pivot_val == 0 {
            return None;
        }
        let inv_pivot = tables.div(1, pivot_val);
        for k in 0..2 * n {
            if a[col * (2 * n) + k] != 0 {
                a[col * (2 * n) + k] = tables.mul(a[col * (2 * n) + k], inv_pivot);
            }
        }

        // eliminate other rows
        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = a[r * (2 * n) + col];
            if factor != 0 {
                for k in 0..2 * n {
                    let left = a[r * (2 * n) + k];
                    let sub = tables.mul(factor, a[col * (2 * n) + k]);
                    a[r * (2 * n) + k] = left ^ sub;
                }
            }
        }
    }

    // extract inverse
    let mut inv = vec![0u8; n * n];
    for r in 0..n {
        for c in 0..n {
            inv[r * n + c] = a[r * (2 * n) + n + c];
        }
    }
    Some(inv)
}

/// Used only by [`ReedSolomon::decode_scalar_reference`], the pre-optimization
/// reference implementation kept for equivalence testing.
#[cfg(test)]
fn multiply_matrix_vector_gf(mat: &[u8], n: usize, vecv: &[u8], tables: &GfTables) -> Vec<u8> {
    let mut out = vec![0u8; n];
    for r in 0..n {
        let mut acc = 0u8;
        for c in 0..n {
            let coef = mat[r * n + c];
            if coef != 0 && vecv[c] != 0 {
                acc ^= tables.mul(coef, vecv[c]);
            }
        }
        out[r] = acc;
    }
    out
}

// Tests
#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn basic_encode_decode_erasure() {
        let rs = ReedSolomon::new(4, 2);
        let data: Vec<Vec<u8>> = vec![vec![1u8; 16], vec![2u8; 16], vec![3u8; 16], vec![4u8; 16]];
        let parity = rs.encode(&data).unwrap();
        // form shards
        let mut shards: Vec<Option<Vec<u8>>> = Vec::new();
        for d in &data {
            shards.push(Some(d.clone()));
        }
        for p in &parity {
            shards.push(Some(p.clone()));
        }

        // erase one data shard
        shards[1] = None;
        assert!(rs.decode(&mut shards).is_ok());
        assert!(shards[1].is_some());
        assert_eq!(shards[1].as_ref().unwrap(), &data[1]);
    }

    proptest! {
        #[test]
        fn proptest_random_small(len in 1usize..20usize, data0 in proptest::collection::vec(0u8..=255u8, 1..20), data1 in proptest::collection::vec(0u8..=255u8, 1..20)) {
            let len = len.min(20);
            let d0 = data0.into_iter().cycle().take(len).collect::<Vec<u8>>();
            let d1 = data1.into_iter().cycle().take(len).collect::<Vec<u8>>();
            // Use small RS
            let rs = ReedSolomon::new(2,1);
            let data = vec![d0.clone(), d1.clone()];
            let parity = rs.encode(&data).unwrap();
            let mut shards: Vec<Option<Vec<u8>>> = Vec::new();
            for d in &data { shards.push(Some(d.clone())); }
            for p in &parity { shards.push(Some(p.clone())); }
            shards[0] = None;
            rs.decode(&mut shards).unwrap();
            assert_eq!(shards[0].as_ref().unwrap(), &data[0]);
        }
    }

    /// Deterministic xorshift PRNG byte stream (no external RNG dependency).
    fn xorshift_bytes(len: usize, mut seed: u64) -> Vec<u8> {
        if seed == 0 {
            seed = 0xDEAD_BEEF_CAFE_F00D;
        }
        (0..len)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed >> 56) as u8
            })
            .collect()
    }

    /// Equivalence: the optimized `decode` (AVX2 / table / scalar bulk
    /// multiply, per CLAUDE.md's zero-allocation-hot-path and flat-buffer
    /// rules) must reproduce byte-for-byte both the pre-erasure data AND the
    /// original per-byte scalar reference implementation, across many
    /// (data, parity, shard_len, erasure-pattern, precompute) combinations.
    #[test]
    fn decode_matches_scalar_reference_and_original_data() {
        let combos: &[(usize, usize, usize)] = &[
            (4, 2, 1),
            (4, 2, 7),
            (2, 1, 3),
            (3, 3, 33),
            (6, 3, 1000),
            (10, 4, 4096),
            (10, 4, 4097), // not a multiple of 8/32: exercises SIMD/chunk tails
            (1, 1, 5000),
            (5, 5, 64),
        ];

        let mut seed = 0x1234_5678_9abc_def0u64;

        for &(d, p, shard_len) in combos {
            for precompute in [false, true] {
                let mut rs = ReedSolomon::new(d, p);
                if precompute {
                    rs.precompute_mul_tables();
                }

                seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let data: Vec<Vec<u8>> = (0..d)
                    .map(|j| {
                        seed = seed.wrapping_add(j as u64 * 7 + 1);
                        xorshift_bytes(shard_len, seed)
                    })
                    .collect();
                let parity = rs.encode(&data).unwrap();

                // Try every erasure count from 1..=p, and a couple of index
                // patterns per count (front-loaded and scattered).
                for m in 1..=p {
                    let patterns: Vec<Vec<usize>> = vec![
                        (0..m).collect(),
                        (0..m).map(|i| (i * 2) % d).collect(),
                        (0..m).map(|i| d - 1 - (i % d)).collect(),
                    ];
                    for pattern in patterns {
                        let mut missing: Vec<usize> = pattern;
                        missing.sort_unstable();
                        missing.dedup();
                        if missing.len() != m || missing.iter().any(|&i| i >= d) {
                            continue;
                        }

                        let build_shards = || -> Vec<Option<Vec<u8>>> {
                            let mut shards: Vec<Option<Vec<u8>>> = Vec::new();
                            for dd in &data {
                                shards.push(Some(dd.clone()));
                            }
                            for pp in &parity {
                                shards.push(Some(pp.clone()));
                            }
                            for &i in &missing {
                                shards[i] = None;
                            }
                            shards
                        };

                        let mut shards_new = build_shards();
                        let mut shards_ref = build_shards();

                        let res_new = rs.decode(&mut shards_new);
                        let res_ref = rs.decode_scalar_reference(&mut shards_ref);

                        assert!(
                            res_new.is_ok(),
                            "optimized decode failed for d={d} p={p} len={shard_len} missing={missing:?} precompute={precompute}"
                        );
                        assert!(
                            res_ref.is_ok(),
                            "reference decode failed for d={d} p={p} len={shard_len} missing={missing:?} precompute={precompute}"
                        );

                        for &i in &missing {
                            assert_eq!(
                                shards_new[i].as_ref().unwrap(),
                                &data[i],
                                "optimized decode mismatch vs original data: d={d} p={p} len={shard_len} missing={missing:?} precompute={precompute}"
                            );
                            assert_eq!(
                                shards_new[i].as_ref().unwrap(),
                                shards_ref[i].as_ref().unwrap(),
                                "optimized decode mismatch vs scalar reference: d={d} p={p} len={shard_len} missing={missing:?} precompute={precompute}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Error-path parity: the optimized `decode` must return the same `Err`
    /// cases as the scalar reference (length mismatch, too many erasures,
    /// not enough parity shards present).
    #[test]
    fn decode_error_cases_match_reference() {
        let rs = ReedSolomon::new(4, 2);
        let data: Vec<Vec<u8>> = (0..4).map(|j| vec![(j + 1) as u8; 32]).collect();
        let parity = rs.encode(&data).unwrap();

        // Wrong total length.
        let mut too_short: Vec<Option<Vec<u8>>> = data.iter().cloned().map(Some).take(3).collect();
        assert_eq!(
            rs.decode(&mut too_short),
            Err(FecError::BufferTooSmall {
                required: 6,
                provided: 3
            })
        );
        assert_eq!(
            rs.decode_scalar_reference(&mut too_short),
            Err(FecError::BufferTooSmall {
                required: 6,
                provided: 3
            })
        );

        // Too many data erasures (3 missing, only 2 parity shards).
        let build = || -> Vec<Option<Vec<u8>>> {
            let mut shards: Vec<Option<Vec<u8>>> = Vec::new();
            for d in &data {
                shards.push(Some(d.clone()));
            }
            for p in &parity {
                shards.push(Some(p.clone()));
            }
            shards[0] = None;
            shards[1] = None;
            shards[2] = None;
            shards
        };
        let mut too_many_new = build();
        let mut too_many_ref = build();
        assert_eq!(
            rs.decode(&mut too_many_new),
            Err(FecError::InvalidParam("too many data erasures to recover"))
        );
        assert_eq!(
            rs.decode_scalar_reference(&mut too_many_ref),
            Err(FecError::InvalidParam("too many data erasures to recover"))
        );

        // Not enough parity shards present (2 data missing, only 1 parity present).
        let build2 = || -> Vec<Option<Vec<u8>>> {
            let mut shards: Vec<Option<Vec<u8>>> = Vec::new();
            for d in &data {
                shards.push(Some(d.clone()));
            }
            for p in &parity {
                shards.push(Some(p.clone()));
            }
            shards[0] = None;
            shards[1] = None;
            shards[4] = None; // drop one parity shard too
            shards
        };
        let mut np_new = build2();
        let mut np_ref = build2();
        assert_eq!(
            rs.decode(&mut np_new),
            Err(FecError::InvalidParam("not enough parity shards present"))
        );
        assert_eq!(
            rs.decode_scalar_reference(&mut np_ref),
            Err(FecError::InvalidParam("not enough parity shards present"))
        );
    }

    /// Task-4 equivalence guard: `encode_with_avx2` (which consumes the
    /// precomputed `nibble_tables` on AVX2 hosts, or the NEON `vqtbl1q_u8`
    /// kernel on AArch64 hosts) must produce parity byte-identical to the
    /// plain scalar `encode_into` across many (data, parity, shard_len)
    /// shapes, including lengths that are not multiples of 32/16/8 (SIMD
    /// tail handling). On hosts with no SIMD kernel this still verifies the
    /// table-chunked fallback against the scalar path.
    #[test]
    fn encode_avx2_nibble_tables_match_scalar() {
        let combos: &[(usize, usize, usize)] = &[
            (4, 2, 1),
            (4, 2, 31),
            (2, 1, 32),
            (3, 3, 33),
            (6, 3, 1000),
            (10, 4, 4097), // not a multiple of 8/32: exercises SIMD tails
            (5, 5, 64),
        ];
        let mut seed = 0xFEED_FACE_1234_5678u64;

        for &(d, p, shard_len) in combos {
            let mut rs = ReedSolomon::new(d, p);
            rs.precompute_mul_tables();

            let data: Vec<Vec<u8>> = (0..d)
                .map(|_| {
                    seed = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
                    xorshift_bytes(shard_len, seed)
                })
                .collect();
            let refs: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();

            let mut parity_avx2 = vec![0u8; p * shard_len];
            let mut parity_scalar = vec![0u8; p * shard_len];
            rs.encode_with_avx2(&refs, &mut parity_avx2).unwrap();
            rs.encode_into(&refs, &mut parity_scalar).unwrap();

            assert_eq!(
                parity_avx2, parity_scalar,
                "AVX2 (precomputed nibble tables) vs scalar parity mismatch: d={d} p={p} shard_len={shard_len}"
            );
        }
    }

    /// NEON-specific equivalence guard: [`crate::simd_neon::gf256_muladd_neon`]
    /// must produce output byte-identical to the plain scalar
    /// `full_table[data[i]]` reference across random coefficients and
    /// lengths, including lengths that are not multiples of 16 (SIMD tail
    /// handling). Only compiled on AArch64; runs unconditionally there since
    /// NEON is mandatory (no runtime detection to skip).
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn gf256_muladd_neon_matches_scalar_full_table() {
        let mut rs = ReedSolomon::new(1, 1);
        rs.precompute_mul_tables();
        let mul_tables = rs.mul_tables.as_ref().unwrap();
        let nibbles = rs.nibble_tables.as_ref().unwrap();

        let mut seed = 0xABCD_EF01_2345_6789u64;
        let lens: &[usize] = &[0, 1, 5, 15, 16, 17, 31, 32, 33, 200, 4097];
        for &len in lens {
            for trial in 0..8 {
                seed = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
                let coef = (seed >> 32) as u8;
                let data = xorshift_bytes(len, seed);

                let full_table = &mul_tables[coef as usize];
                let [lo_tbl, hi_tbl] = &nibbles[coef as usize];

                let mut parity_neon = vec![0u8; len];
                let mut parity_scalar = vec![0u8; len];
                // SAFETY: NEON is mandatory on AArch64.
                unsafe {
                    crate::simd_neon::gf256_muladd_neon(
                        &data,
                        &mut parity_neon,
                        lo_tbl,
                        hi_tbl,
                        full_table,
                    );
                }
                for i in 0..len {
                    parity_scalar[i] ^= full_table[data[i] as usize];
                }

                assert_eq!(
                    parity_neon, parity_scalar,
                    "trial {trial} coef={coef} len={len}: NEON vs scalar full-table mismatch"
                );
            }
        }
    }
}
