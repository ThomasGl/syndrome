//! AVX2-accelerated LOMS layer kernel for x86-64.
//!
//! This module provides [`decode_layer_passes_avx2`], which replaces the two
//! scalar pass loops (min-tracking and R-update) in
//! [`QcLdpcDecoder::decode_layered_offset_min_sum`] when AVX2 is detected at
//! runtime. The Q-build loop is kept scalar in the caller because it is a
//! gather-style LLR read that benefits less from vectorisation.
//!
//! The kernel processes 8 z-positions per SIMD iteration (256-bit / f32x8).
//! Z values that are not a multiple of 8 are handled by a scalar tail.

#![cfg(target_arch = "x86_64")]

use std::arch::x86_64::*;

use crate::quantize::MSG_MAX;

/// Process passes 1 and 2 of one LOMS layer using AVX2 (8-wide f32 registers).
///
/// # Safety
///
/// Caller must verify AVX2 support at runtime, e.g. with
/// `is_x86_feature_detected!("avx2")`. `min1`, `min2`, and `sxor` must each
/// have length `≥ z`. `q_row` must have length `≥ row_degree * z`. `edge_r`
/// must cover `(layer_begin + row_degree - 1) * z + z` elements. `min1`,
/// `min2`, and `sxor` are additionally accessed with **aligned** AVX2
/// loads/stores (`_mm256_load_ps`/`_mm256_load_si256`), so their backing
/// allocations must start on a 32-byte boundary -- the only caller,
/// `QcLdpcDecoder::decode_layered_offset_min_sum_dispatch`, guarantees this
/// via `#[repr(align(64))]`-wrapped locals, always sliced from index 0.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn decode_layer_passes_avx2(
    z: usize,
    row_degree: usize,
    offset_beta: f32,
    q_row: &[f32],
    edge_r: &mut [f32],
    layer_begin: usize,
    submatrix_cols: &[usize],
    submatrix_shifts: &[i16],
    llr: &mut [f32],
    min1: &mut [f32],
    min2: &mut [f32],
    sxor: &mut [u32],
) {
    // Rust 2024: unsafe operations inside an `unsafe fn` still require an
    // explicit `unsafe {}` block. Wrap the entire body since all work here
    // requires the AVX2 target feature verified by the caller.
    unsafe {
        decode_layer_passes_avx2_body(
            z,
            row_degree,
            offset_beta,
            q_row,
            edge_r,
            layer_begin,
            submatrix_cols,
            submatrix_shifts,
            llr,
            min1,
            min2,
            sxor,
        )
    }
}

#[target_feature(enable = "avx2")]
unsafe fn decode_layer_passes_avx2_body(
    z: usize,
    row_degree: usize,
    offset_beta: f32,
    q_row: &[f32],
    edge_r: &mut [f32],
    layer_begin: usize,
    submatrix_cols: &[usize],
    submatrix_shifts: &[i16],
    llr: &mut [f32],
    min1: &mut [f32],
    min2: &mut [f32],
    sxor: &mut [u32],
) {
    // Rust 2024: explicit unsafe block required even inside `unsafe fn`.
    unsafe {
        // Constants hoisted outside all loops.
        let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFFi32));
        let sign_mask = _mm256_set1_epi32(0x8000_0000u32 as i32);
        let beta_v = _mm256_set1_ps(offset_beta);
        let zero_v = _mm256_setzero_ps();

        // Number of z positions covered by full 8-wide chunks (tail ≤ 7 handled scalarly).
        let full = z & !7;

        // ── Pass 1: init scratch, then accumulate min1 / min2 / sign-XOR ───────
        // Aligned stores: min1/min2/sxor are guaranteed 32-byte-aligned by
        // the caller (see this fn's SAFETY doc).
        for i in (0..full).step_by(8) {
            _mm256_store_ps(min1.as_mut_ptr().add(i), _mm256_set1_ps(f32::MAX));
            _mm256_store_ps(min2.as_mut_ptr().add(i), _mm256_set1_ps(f32::MAX));
            _mm256_store_si256(
                sxor[i..].as_mut_ptr() as *mut __m256i,
                _mm256_setzero_si256(),
            );
        }
        for i in full..z {
            min1[i] = f32::MAX;
            min2[i] = f32::MAX;
            sxor[i] = 0;
        }

        for edge in 0..row_degree {
            let q_ptr = q_row[edge * z..].as_ptr();
            let m1_ptr = min1.as_mut_ptr();
            let m2_ptr = min2.as_mut_ptr();

            // AVX2 chunks (8 z-positions per iteration).
            for chunk in (0..full).step_by(8) {
                let q_v = _mm256_loadu_ps(q_ptr.add(chunk));
                let abs_v = _mm256_and_ps(q_v, abs_mask);
                let sign_v = _mm256_and_si256(_mm256_castps_si256(q_v), sign_mask);

                // XOR each lane's sign bit into the running accumulator.
                // Aligned load+store: sxor is guaranteed 32-byte-aligned.
                let sx_ptr = sxor[chunk..].as_mut_ptr() as *mut __m256i;
                _mm256_store_si256(
                    sx_ptr,
                    _mm256_xor_si256(_mm256_load_si256(sx_ptr as *const __m256i), sign_v),
                );

                // Branchless min1 / min2 update using blendv:
                //   case 1 (abs ≤ m1): push m1→m2, set m1=abs
                //   case 2 (m1 < abs < m2): set m2=abs
                // Aligned loads: min1/min2 are guaranteed 32-byte-aligned.
                let m1_v = _mm256_load_ps(m1_ptr.add(chunk));
                let m2_v = _mm256_load_ps(m2_ptr.add(chunk));

                let le = _mm256_cmp_ps(abs_v, m1_v, _CMP_LE_OQ);
                let new_m2 = _mm256_blendv_ps(m2_v, m1_v, le); // case 1: m2 ← old m1
                let new_m1 = _mm256_blendv_ps(m1_v, abs_v, le); // case 1: m1 ← abs

                // case 2: abs < original m2, but NOT already handled by case 1
                let lt2_only = _mm256_andnot_ps(le, _mm256_cmp_ps(abs_v, m2_v, _CMP_LT_OQ));
                let new_m2 = _mm256_blendv_ps(new_m2, abs_v, lt2_only);

                _mm256_store_ps(m1_ptr.add(chunk), new_m1);
                _mm256_store_ps(m2_ptr.add(chunk), new_m2);
            }
            // Scalar tail (at most 7 elements).
            for i in full..z {
                let bits = q_row[edge * z + i].to_bits();
                let abs_q = f32::from_bits(bits & 0x7FFF_FFFF);
                sxor[i] ^= bits & 0x8000_0000;
                let m1 = min1[i];
                if abs_q <= m1 {
                    min2[i] = m1;
                    min1[i] = abs_q;
                } else if abs_q < min2[i] {
                    min2[i] = abs_q;
                }
            }
        }

        // Clamp MAX sentinels (triggered only when row_degree == 0 or all Q = ±inf).
        for i in 0..z {
            if min1[i] == f32::MAX {
                min1[i] = 0.0;
            }
            if min2[i] == f32::MAX {
                min2[i] = min1[i];
            }
        }

        // ── Pass 2: compute new R and delta-update LLR ──────────────────────────
        for edge in 0..row_degree {
            let col_block = submatrix_cols[layer_begin + edge];
            let shift = submatrix_shifts[layer_begin + edge] as usize;
            let base_edge = (layer_begin + edge) * z;
            let q_ptr = q_row[edge * z..].as_ptr();
            let er_ptr = edge_r[base_edge..].as_mut_ptr();
            let m1_ptr = min1.as_ptr();
            let m2_ptr = min2.as_ptr();
            let var_base = col_block * z;

            for chunk in (0..full).step_by(8) {
                let q_v = _mm256_loadu_ps(q_ptr.add(chunk));
                let q_int = _mm256_castps_si256(q_v);
                let abs_v = _mm256_and_ps(q_v, abs_mask);

                // Exclusive sign: XOR of all edges' signs, with this edge removed.
                // Aligned loads: sxor/min1/min2 are guaranteed 32-byte-aligned.
                let sx_v = _mm256_load_si256(sxor[chunk..].as_ptr() as *const __m256i);
                let excl_sign =
                    _mm256_castsi256_ps(_mm256_xor_si256(sx_v, _mm256_and_si256(q_int, sign_mask)));

                // Exclusive min: use min2 where abs equals min1 (float equality is
                // exact — value was stored directly from this q_row slot in pass 1).
                let m1_v = _mm256_load_ps(m1_ptr.add(chunk));
                let m2_v = _mm256_load_ps(m2_ptr.add(chunk));
                let eq_m1 = _mm256_cmp_ps(abs_v, m1_v, _CMP_EQ_OQ);
                let min_excl = _mm256_blendv_ps(m1_v, m2_v, eq_m1);

                // Magnitude after offset correction, clamped to ≥ 0.
                let mag_v = _mm256_max_ps(_mm256_sub_ps(min_excl, beta_v), zero_v);

                // Apply sign via bitwise OR (mag ≥ 0, so its sign bit is already 0).
                let new_r_v = _mm256_or_ps(mag_v, excl_sign);

                // Update edge_r and scatter-add delta into LLR.
                let old_r_v = _mm256_loadu_ps(er_ptr.add(chunk));
                _mm256_storeu_ps(er_ptr.add(chunk), new_r_v);

                // The 8 var_idx values may not be contiguous when the shift wraps
                // around the block boundary, so we scatter scalarly. For z=384 and
                // typical shifts, this touches a contiguous run in most chunks.
                let mut delta = [0f32; 8];
                _mm256_storeu_ps(delta.as_mut_ptr(), _mm256_sub_ps(new_r_v, old_r_v));
                for i in 0..8 {
                    let s = chunk + i + shift;
                    // SAFETY: col_block < num_col_blocks (from BG entries), s % z < z,
                    // so var_idx < num_col_blocks * z == llr.len().
                    let var_idx = var_base + if s >= z { s - z } else { s };
                    *llr.get_unchecked_mut(var_idx) += delta[i];
                }
            }
            // Scalar tail.
            for i in full..z {
                let q_bits = q_row[edge * z + i].to_bits();
                let abs_q = f32::from_bits(q_bits & 0x7FFF_FFFF);
                let sign_excl = sxor[i] ^ (q_bits & 0x8000_0000);
                let min_excl = if abs_q == min1[i] { min2[i] } else { min1[i] };
                let mag = (min_excl - offset_beta).max(0.0);
                let new_r = f32::from_bits(mag.to_bits() | sign_excl);
                let old_r = edge_r[base_edge + i];
                edge_r[base_edge + i] = new_r;
                let s = i + shift;
                let var_idx = var_base + if s >= z { s - z } else { s };
                llr[var_idx] += new_r - old_r;
            }
        }
    } // end unsafe block
}

// ---------------------------------------------------------------------------
// GF(256) multiply-XOR using AVX2 VPSHUFB — 32 bytes per cycle
// ---------------------------------------------------------------------------

/// GF(256) multiply-accumulate: `parity[i] ^= GF_mul(coef, data[i])` for all i.
///
/// Uses nibble decomposition: `GF_mul(c, x) = lo_tbl[x & 0xF] ^ hi_tbl[x >> 4]`.
/// Processes 32 bytes per SIMD iteration via VPSHUFB; scalar tail handles the rest.
///
/// # Arguments
///
/// * `data`       — input bytes.
/// * `parity`     — accumulator (XOR'd in-place); must have the same length as `data`.
/// * `lo_tbl`     — 16-byte table: `lo_tbl[i] = GF_mul(coef, i)` for i in 0..16.
/// * `hi_tbl`     — 16-byte table: `hi_tbl[i] = GF_mul(coef, i << 4)` for i in 0..16.
/// * `full_table` — 256-byte table for the scalar tail: `full_table[v] = GF_mul(coef, v)`.
///
/// # Safety
///
/// Caller must have verified AVX2 availability with `is_x86_feature_detected!("avx2")`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn gf256_muladd_avx2(
    data: &[u8],
    parity: &mut [u8],
    lo_tbl: &[u8; 16],
    hi_tbl: &[u8; 16],
    full_table: &[u8; 256],
) {
    unsafe {
        let len = data.len().min(parity.len());
        let full = len & !31; // floor to multiple of 32

        let lo_tbl_v =
            _mm256_broadcastsi128_si256(_mm_loadu_si128(lo_tbl.as_ptr() as *const __m128i));
        let hi_tbl_v =
            _mm256_broadcastsi128_si256(_mm_loadu_si128(hi_tbl.as_ptr() as *const __m128i));
        let lo_mask = _mm256_set1_epi8(0x0Fu8 as i8);

        let mut i = 0usize;
        while i < full {
            let data_v = _mm256_loadu_si256(data.as_ptr().add(i) as *const __m256i);

            // Low nibble index: mask off high nibble.
            let lo_idx = _mm256_and_si256(data_v, lo_mask);
            // High nibble: srli_epi16 shifts 16-bit lanes right by 4, then mask.
            // After shift: each byte's high nibble lands in the low nibble position.
            let hi_idx = _mm256_and_si256(_mm256_srli_epi16(data_v, 4), lo_mask);

            let lo_val = _mm256_shuffle_epi8(lo_tbl_v, lo_idx);
            let hi_val = _mm256_shuffle_epi8(hi_tbl_v, hi_idx);
            let prod = _mm256_xor_si256(lo_val, hi_val);

            let par_v = _mm256_loadu_si256(parity.as_ptr().add(i) as *const __m256i);
            _mm256_storeu_si256(
                parity.as_mut_ptr().add(i) as *mut __m256i,
                _mm256_xor_si256(par_v, prod),
            );
            i += 32;
        }

        // Scalar tail.
        for j in i..len {
            parity[j] ^= full_table[data[j] as usize];
        }
    }
}

// ---------------------------------------------------------------------------
// GF(256) multiply-XOR using GFNI — one instruction per 32 bytes
// ---------------------------------------------------------------------------

/// Build the packed $8 \times 8$ $\mathbb{F}\_2$ bit matrix that
/// `_mm256_gf2p8affine_epi64_epi8` expects, for the linear map "multiply by
/// a fixed `GF(256)` constant `c`", from `c`'s 256-entry multiplication
/// table (`table[v] = GF_mul(c, v)`, as already built by
/// [`crate::reed_solomon::ReedSolomon::precompute_mul_tables`]).
///
/// # Why this table already determines the matrix
///
/// Multiplication by a fixed field element is $\mathbb{F}\_2$-linear in the
/// byte's bit representation: $c \cdot (x \oplus y) = c \cdot x \oplus c
/// \cdot y$. A linear map on $\mathbb{F}\_2^8$ is fully determined by its
/// action on the 8 standard basis vectors, so column $i$ of the matrix is
/// exactly `table[1 << i]` (`c` times the byte with only bit `i` set).
///
/// # Bit packing convention
///
/// `GF2P8AFFINEQB` reads its matrix operand as 8 row-bytes packed into a
/// `u64`; row `j`'s bit `i` selects whether column `i` contributes to output
/// bit `j`. Row 0 is the matrix's most significant byte (bits 56..64), row 7
/// its least significant (bits 0..8) — pinned by
/// `gfni_matrix_reproduces_every_coefficient_table` below, which checks all
/// 256 coefficients against the plain scalar table, not just derived by
/// reading the manual.
#[cfg(target_arch = "x86_64")]
pub(crate) fn gf2p8_affine_matrix(table: &[u8; 256]) -> u64 {
    let mut rows = [0u8; 8];
    for i in 0..8usize {
        let col = table[1usize << i];
        for j in 0..8usize {
            rows[j] |= ((col >> j) & 1) << i;
        }
    }
    u64::from_be_bytes(rows)
}

/// GF(256) multiply-accumulate via GFNI: `parity[i] ^= GF_mul(coef, data[i])`.
///
/// Applies the $\mathbb{F}\_2$-linear "multiply by `coef`" bit matrix
/// directly with `_mm256_gf2p8affine_epi64_epi8` — one instruction per 32
/// bytes, versus the shuffle-mask-shuffle-blend sequence
/// [`gf256_muladd_avx2`] needs for the same 32 bytes. `matrix` must be
/// `coef`'s packed matrix from [`gf2p8_affine_matrix`]; `full_table` is used
/// for the scalar tail exactly as in the AVX2 kernel.
///
/// # Arguments
///
/// * `data`       — input bytes.
/// * `parity`     — accumulator (XOR'd in-place); must have the same length as `data`.
/// * `matrix`     — packed affine matrix for `coef`, from [`gf2p8_affine_matrix`].
/// * `full_table` — 256-byte table for the scalar tail: `full_table[v] = GF_mul(coef, v)`.
///
/// # Safety
///
/// Caller must have verified GFNI and AVX2 availability with
/// `is_x86_feature_detected!("gfni")` and `is_x86_feature_detected!("avx2")`.
#[target_feature(enable = "gfni,avx2")]
pub(crate) unsafe fn gf256_muladd_gfni(
    data: &[u8],
    parity: &mut [u8],
    matrix: u64,
    full_table: &[u8; 256],
) {
    unsafe {
        let len = data.len().min(parity.len());
        let full = len & !31; // floor to multiple of 32

        let matrix_v = _mm256_set1_epi64x(matrix as i64);

        let mut i = 0usize;
        while i < full {
            let data_v = _mm256_loadu_si256(data.as_ptr().add(i) as *const __m256i);
            // imm8 = 0: no additive constant, this is pure linear multiply.
            let prod = _mm256_gf2p8affine_epi64_epi8(data_v, matrix_v, 0);

            let par_v = _mm256_loadu_si256(parity.as_ptr().add(i) as *const __m256i);
            _mm256_storeu_si256(
                parity.as_mut_ptr().add(i) as *mut __m256i,
                _mm256_xor_si256(par_v, prod),
            );
            i += 32;
        }

        // Scalar tail.
        for j in i..len {
            parity[j] ^= full_table[data[j] as usize];
        }
    }
}

// ---------------------------------------------------------------------------
// AVX2 Q-build: assemble V→C messages for one edge via contiguous LLR spans
// ---------------------------------------------------------------------------

/// AVX2 Q-build for one LOMS edge: `q_row[q_base..q_base+z] = llr[...] - edge_r[...]`.
///
/// The cyclic shift `shift` splits the z LLR reads into two contiguous spans:
/// - Span 1 (len = z − shift): `llr[var_base+shift .. var_base+z]`
/// - Span 2 (len = shift):     `llr[var_base .. var_base+shift]`
///
/// Each span is contiguous, so it is vectorized with 8-wide f32 loads without scatter.
///
/// # Safety
///
/// Caller must have verified AVX2 availability. All index ranges must be in-bounds.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn q_build_edge_avx2(
    z: usize,
    shift: usize,
    var_base: usize,
    q_base: usize,
    base_edge: usize,
    llr: &[f32],
    edge_r: &[f32],
    q_row: &mut [f32],
) {
    unsafe {
        // Span 1: q_row[q_base..q_base+span1] = llr[var_base+shift..var_base+z] - edge_r[base_edge..]
        let span1 = z - shift;
        let full1 = span1 & !7;
        let mut i = 0usize;
        while i < full1 {
            let lv = _mm256_loadu_ps(llr.as_ptr().add(var_base + shift + i));
            let ev = _mm256_loadu_ps(edge_r.as_ptr().add(base_edge + i));
            _mm256_storeu_ps(q_row.as_mut_ptr().add(q_base + i), _mm256_sub_ps(lv, ev));
            i += 8;
        }
        for j in i..span1 {
            q_row[q_base + j] = llr[var_base + shift + j] - edge_r[base_edge + j];
        }

        // Span 2: q_row[q_base+span1..q_base+z] = llr[var_base..var_base+shift] - edge_r[base_edge+span1..]
        if shift > 0 {
            let full2 = shift & !7;
            let mut i = 0usize;
            while i < full2 {
                let lv = _mm256_loadu_ps(llr.as_ptr().add(var_base + i));
                let ev = _mm256_loadu_ps(edge_r.as_ptr().add(base_edge + span1 + i));
                _mm256_storeu_ps(
                    q_row.as_mut_ptr().add(q_base + span1 + i),
                    _mm256_sub_ps(lv, ev),
                );
                i += 8;
            }
            for j in i..shift {
                q_row[q_base + span1 + j] = llr[var_base + j] - edge_r[base_edge + span1 + j];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fixed-point (i8) LOMS layer kernel — 32 z-positions per instruction
// ---------------------------------------------------------------------------
//
// # Why the cyclic shift is handled per 16-lane group
//
// A layer's edge reads and writes the posterior at `var_base + (z_idx +
// shift) mod z`, so consecutive z-positions are consecutive `app` slots
// except at the single point where the index wraps back to `var_base`. The
// obvious way to vectorize that is to split the whole `z` range into the two
// contiguous runs either side of the wrap and chunk each one — and it is a
// trap. Every run then carries its own scalar tail, so the tail work is
// `~2 * 16` elements *per edge* regardless of `z`, and at `Z = 128` that is
// a quarter of the layer. Measured against the `f32` kernel that made the
// fixed-point path *slower* below `Z ≈ 192`, with a decode time almost
// independent of `Z`: the per-edge overhead, not the per-element work, was
// setting the pace.
//
// What is done instead: chunk the arithmetic over the full `z` range and
// test each 16-lane group for whether *it* straddles the wrap. At most one
// group per edge does, so the scalar work is one group per edge plus the
// `z mod 32` tail the vector loop cannot cover.

/// Add 16 already-widened `i16` deltas into the posterior, clamping to
/// `±app_clamp`.
///
/// `z_idx` is the first of the 16 z-positions; the posterior indices are
/// `var_base + (z_idx + j + shift) mod z`. When those 16 indices do not wrap
/// they are consecutive and the whole group is one load-add-store; the one
/// group per edge that does wrap falls back to scalar.
///
/// # Safety
///
/// Caller must have verified AVX2 availability. `var_base + z` must be within
/// `app`, `z_idx + 16 <= z`, and `app_clamp >= 0`.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn apply_delta16_i8_avx2(
    app: &mut [i16],
    var_base: usize,
    shift: usize,
    z: usize,
    z_idx: usize,
    delta: __m256i,
    clamp_lo: __m256i,
    clamp_hi: __m256i,
    app_clamp: i16,
) {
    unsafe {
        let s = z_idx + shift;
        let start = if s >= z { s - z } else { s };
        if start + 16 <= z {
            let p = app.as_mut_ptr().add(var_base + start) as *mut __m256i;
            let v = _mm256_adds_epi16(_mm256_loadu_si256(p as *const __m256i), delta);
            _mm256_storeu_si256(p, _mm256_min_epi16(_mm256_max_epi16(v, clamp_lo), clamp_hi));
        } else {
            let mut tmp = [0i16; 16];
            _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, delta);
            for (j, &d) in tmp.iter().enumerate() {
                let sj = start + j;
                let idx = var_base + if sj >= z { sj - z } else { sj };
                let slot = &mut *app.get_unchecked_mut(idx);
                *slot = slot.saturating_add(d).clamp(-app_clamp, app_clamp);
            }
        }
    }
}

/// AVX2 Q-build for one LOMS edge in the fixed-point path:
/// `q_row[q_base + i] = sat8(app[var_base + (i + shift) mod z] - edge_r[base_edge + i])`.
///
/// 16 z-positions per iteration: one `__m256i` of `i16` posteriors against
/// one `__m128i` of `i8` messages, widened before the subtraction so an
/// out-of-range posterior cannot wrap. The wrap in the posterior index is
/// handled per group exactly as in [`apply_delta16_i8_avx2`].
///
/// # Safety
///
/// Caller must have verified AVX2 availability. `app` must cover
/// `var_base + z` elements, `edge_r` must cover `base_edge + z`, and `q_row`
/// must cover `q_base + z`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn q_build_edge_i8_avx2(
    z: usize,
    shift: usize,
    var_base: usize,
    q_base: usize,
    base_edge: usize,
    app: &[i16],
    edge_r: &[i8],
    q_row: &mut [i8],
) {
    unsafe {
        let msg_hi = _mm256_set1_epi16(MSG_MAX as i16);
        let msg_lo = _mm256_set1_epi16(-(MSG_MAX as i16));
        let full = z & !15;

        let mut i = 0usize;
        while i < full {
            let s = i + shift;
            let start = if s >= z { s - z } else { s };
            let a = if start + 16 <= z {
                _mm256_loadu_si256(app.as_ptr().add(var_base + start) as *const __m256i)
            } else {
                // The one group per edge whose posterior reads wrap: gather
                // the 16 values, then rejoin the vector path.
                let mut tmp = [0i16; 16];
                for (j, slot) in tmp.iter_mut().enumerate() {
                    let sj = start + j;
                    *slot = *app.get_unchecked(var_base + if sj >= z { sj - z } else { sj });
                }
                _mm256_loadu_si256(tmp.as_ptr() as *const __m256i)
            };

            let r8 = _mm_loadu_si128(edge_r.as_ptr().add(base_edge + i) as *const __m128i);
            let r = _mm256_cvtepi8_epi16(r8);
            // Saturating i16 subtract then clamp to the message range: for an
            // in-range posterior the saturation never engages, and where it
            // would it lands on the same side as the clamp, so this matches
            // the scalar `(app as i32 - r as i32).clamp(..)` exactly.
            let d = _mm256_subs_epi16(a, r);
            let dc = _mm256_min_epi16(_mm256_max_epi16(d, msg_lo), msg_hi);

            // packs_epi16 works per 128-bit lane, so `packs(dc, dc)` puts
            // lanes 0..8 in the low 64 bits of the low half and lanes 8..16
            // in the low 64 bits of the high half. Values are already inside
            // i8 range, so the saturation in the pack is a no-op.
            let packed = _mm256_packs_epi16(dc, dc);
            _mm_storel_epi64(
                q_row.as_mut_ptr().add(q_base + i) as *mut __m128i,
                _mm256_castsi256_si128(packed),
            );
            _mm_storel_epi64(
                q_row.as_mut_ptr().add(q_base + i + 8) as *mut __m128i,
                _mm256_extracti128_si256(packed, 1),
            );
            i += 16;
        }

        // Scalar tail (at most 15 elements).
        for j in i..z {
            let s = j + shift;
            let var_idx = var_base + if s >= z { s - z } else { s };
            let d =
                *app.get_unchecked(var_idx) as i32 - *edge_r.get_unchecked(base_edge + j) as i32;
            *q_row.get_unchecked_mut(q_base + j) = d.clamp(-(MSG_MAX as i32), MSG_MAX as i32) as i8;
        }
    }
}

/// Process passes 1 and 2 of one fixed-point LOMS layer using AVX2 (32-wide
/// `i8` registers).
///
/// This is the reason the fixed-point path exists: the magnitude and sign
/// arithmetic that the `f32` kernel does 8 lanes at a time runs 32 lanes at a
/// time here, on the same 256-bit registers.
///
/// # Safety
///
/// Caller must verify AVX2 support at runtime, e.g. with
/// `is_x86_feature_detected!("avx2")`. `min1`, `min2` and `sxor` must each
/// have length `≥ z`; `q_row` must have length `≥ row_degree * z`; `edge_r`
/// must cover `(layer_begin + row_degree - 1) * z + z` elements; every
/// `submatrix_cols[layer_begin + e] * z + z` must be within `app`. All
/// accesses are unaligned loads/stores, so no alignment guarantee is
/// required. `app_clamp` must be non-negative, `beta_q` in `[0, 127]`, and
/// every `q_row` value in `[-127, 127]`.
///
/// `row_degree` must be `≥ 2`. A layer of degree 0 or 1 has no second
/// smallest magnitude, and the sentinel fix-up for that case lives only in
/// the scalar path — the caller routes such layers there. No 3GPP or IEEE
/// 802.11 base graph contains one (minimum row degree is 3 for BG1 and BG2).
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn decode_layer_passes_i8_avx2(
    z: usize,
    row_degree: usize,
    beta_q: i8,
    app_clamp: i16,
    q_row: &[i8],
    edge_r: &mut [i8],
    layer_begin: usize,
    submatrix_cols: &[usize],
    submatrix_shifts: &[i16],
    app: &mut [i16],
    min1: &mut [i8],
    min2: &mut [i8],
    sxor: &mut [u8],
) {
    unsafe {
        let sign_mask = _mm256_set1_epi8(0x80u8 as i8);
        let msg_max_v = _mm256_set1_epi8(MSG_MAX);
        let ones = _mm256_set1_epi8(1);
        let beta_v = _mm256_set1_epi8(beta_q);
        let zero = _mm256_setzero_si256();
        let clamp_hi = _mm256_set1_epi16(app_clamp);
        let clamp_lo = _mm256_set1_epi16(-app_clamp);
        let full = z & !31;

        // ── Pass 1: min1 / min2 / sign-XOR across the layer's edges ─────────
        // min1 and min2 start at MSG_MAX, which is the true upper bound on a
        // message magnitude rather than an out-of-range sentinel, so no
        // fix-up pass is needed afterwards (contrast the f32 kernel, which
        // must clamp its f32::MAX sentinels).
        for i in (0..full).step_by(32) {
            _mm256_storeu_si256(min1.as_mut_ptr().add(i) as *mut __m256i, msg_max_v);
            _mm256_storeu_si256(min2.as_mut_ptr().add(i) as *mut __m256i, msg_max_v);
            _mm256_storeu_si256(sxor.as_mut_ptr().add(i) as *mut __m256i, zero);
        }
        for i in full..z {
            *min1.get_unchecked_mut(i) = MSG_MAX;
            *min2.get_unchecked_mut(i) = MSG_MAX;
            *sxor.get_unchecked_mut(i) = 0;
        }

        for edge in 0..row_degree {
            let q_ptr = q_row.as_ptr().add(edge * z);
            for c in (0..full).step_by(32) {
                let q = _mm256_loadu_si256(q_ptr.add(c) as *const __m256i);
                let a = _mm256_abs_epi8(q);

                let sx_ptr = sxor.as_mut_ptr().add(c) as *mut __m256i;
                _mm256_storeu_si256(
                    sx_ptr,
                    _mm256_xor_si256(
                        _mm256_loadu_si256(sx_ptr as *const __m256i),
                        _mm256_and_si256(q, sign_mask),
                    ),
                );

                // Branch-free sorted insertion of `a` into the pair
                // (min1 ≤ min2): min1 ← min(min1, a), min2 ← min(min2,
                // max(min1, a)). Three instructions, and it reproduces the
                // scalar two-branch form exactly, ties included.
                let m1_ptr = min1.as_mut_ptr().add(c) as *mut __m256i;
                let m2_ptr = min2.as_mut_ptr().add(c) as *mut __m256i;
                let m1 = _mm256_loadu_si256(m1_ptr as *const __m256i);
                let m2 = _mm256_loadu_si256(m2_ptr as *const __m256i);
                _mm256_storeu_si256(m1_ptr, _mm256_min_epi8(m1, a));
                _mm256_storeu_si256(m2_ptr, _mm256_min_epi8(m2, _mm256_max_epi8(m1, a)));
            }
            // Scalar tail (at most 31 elements).
            for i in full..z {
                let q = *q_row.get_unchecked(edge * z + i);
                let a = q.saturating_abs();
                *sxor.get_unchecked_mut(i) ^= (q as u8) & 0x80;
                let m1 = *min1.get_unchecked(i);
                *min1.get_unchecked_mut(i) = m1.min(a);
                let m2 = *min2.get_unchecked(i);
                *min2.get_unchecked_mut(i) = m2.min(m1.max(a));
            }
        }

        // ── Pass 2: new R, then delta-update the posterior ───────────────────
        for edge in 0..row_degree {
            let col_block = *submatrix_cols.get_unchecked(layer_begin + edge);
            let shift = *submatrix_shifts.get_unchecked(layer_begin + edge) as usize;
            let base_edge = (layer_begin + edge) * z;
            let var_base = col_block * z;
            let q_ptr = q_row.as_ptr().add(edge * z);
            let r_ptr = edge_r.as_mut_ptr().add(base_edge);

            for c in (0..full).step_by(32) {
                let q = _mm256_loadu_si256(q_ptr.add(c) as *const __m256i);
                let a = _mm256_abs_epi8(q);
                let m1 = _mm256_loadu_si256(min1.as_ptr().add(c) as *const __m256i);
                let m2 = _mm256_loadu_si256(min2.as_ptr().add(c) as *const __m256i);
                let sx = _mm256_loadu_si256(sxor.as_ptr().add(c) as *const __m256i);

                // Exclusive minimum: min2 for the edge that *was* the layer
                // minimum, min1 for every other edge. Integer equality is
                // exact, and where two edges tie for the minimum min2 == min1
                // anyway.
                let eq = _mm256_cmpeq_epi8(a, m1);
                let min_excl = _mm256_blendv_epi8(m1, m2, eq);

                // Offset correction, floored at zero. Both operands are in
                // [0, 127], so the subtraction stays inside i8.
                let mag = _mm256_max_epi8(_mm256_sub_epi8(min_excl, beta_v), zero);

                // Exclusive sign: XOR of every edge's sign with this edge
                // removed. `sign_epi8` multiplies by the signum of its second
                // operand, so OR-ing in 1 turns the sign bit into a ±1
                // selector (a plain OR would be wrong here — these are
                // two's-complement values, not sign-magnitude floats).
                let sign_excl = _mm256_xor_si256(sx, _mm256_and_si256(q, sign_mask));
                let new_r = _mm256_sign_epi8(mag, _mm256_or_si256(sign_excl, ones));

                let old_r = _mm256_loadu_si256(r_ptr.add(c) as *const __m256i);
                _mm256_storeu_si256(r_ptr.add(c) as *mut __m256i, new_r);

                // Widen both halves to i16 and delta-update the posterior.
                let d_lo = _mm256_sub_epi16(
                    _mm256_cvtepi8_epi16(_mm256_castsi256_si128(new_r)),
                    _mm256_cvtepi8_epi16(_mm256_castsi256_si128(old_r)),
                );
                let d_hi = _mm256_sub_epi16(
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(new_r, 1)),
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(old_r, 1)),
                );
                apply_delta16_i8_avx2(
                    app, var_base, shift, z, c, d_lo, clamp_lo, clamp_hi, app_clamp,
                );
                apply_delta16_i8_avx2(
                    app,
                    var_base,
                    shift,
                    z,
                    c + 16,
                    d_hi,
                    clamp_lo,
                    clamp_hi,
                    app_clamp,
                );
            }

            // Scalar tail (at most 31 elements).
            for i in full..z {
                let q = *q_row.get_unchecked(edge * z + i);
                let a = q.saturating_abs();
                let sign_excl = *sxor.get_unchecked(i) ^ ((q as u8) & 0x80);
                let m1 = *min1.get_unchecked(i);
                let min_excl = if a == m1 { *min2.get_unchecked(i) } else { m1 };
                let mag = (min_excl - beta_q).max(0);
                // Branch-free negate; see the scalar kernel in `qc_ldpc` for
                // why a sign branch is not affordable here.
                let neg = (sign_excl as i8) >> 7;
                let new_r = (mag ^ neg).wrapping_sub(neg);
                let old_r = *edge_r.get_unchecked(base_edge + i);
                *edge_r.get_unchecked_mut(base_edge + i) = new_r;
                let delta = new_r as i16 - old_r as i16;
                let s = i + shift;
                let var_idx = var_base + if s >= z { s - z } else { s };
                let slot = &mut *app.get_unchecked_mut(var_idx);
                *slot = slot.saturating_add(delta).clamp(-app_clamp, app_clamp);
            }
        }
    }
}
