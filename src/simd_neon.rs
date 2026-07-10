//! NEON-accelerated LOMS layer kernel for AArch64.
//!
//! This module provides [`decode_layer_passes_neon`], which replaces the two
//! scalar pass loops in `decode_layered_offset_min_sum` on ARM cores.
//! NEON is mandatory on AArch64, so no runtime detection is required; the
//! module compiles and runs whenever `target_arch = "aarch64"`.
//!
//! The kernel processes 4 z-positions per SIMD iteration (128-bit / f32x4).
//! Z values that are not a multiple of 4 are handled by a scalar tail.

#![cfg(target_arch = "aarch64")]

use core::arch::aarch64::*;

/// Process passes 1 and 2 of one LOMS layer using ARM NEON (4-wide f32x4).
///
/// # Safety
///
/// NEON is guaranteed on AArch64; no runtime detection is required.
/// `min1`, `min2`, and `sxor` must each have length `≥ z`.
/// `q_row` must have length `≥ row_degree * z`. `edge_r` must cover
/// `(layer_begin + row_degree - 1) * z + z` elements.
pub(crate) unsafe fn decode_layer_passes_neon(
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
    // is intrinsic-driven (same pattern as the AVX2 kernel).
    unsafe {
        decode_layer_passes_neon_body(
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

/// Body of [`decode_layer_passes_neon`]; see its safety contract.
///
/// # Safety
///
/// Same requirements as [`decode_layer_passes_neon`].
#[allow(clippy::too_many_arguments)]
unsafe fn decode_layer_passes_neon_body(
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
        let abs_mask = vdupq_n_u32(0x7FFF_FFFFu32);
        let sign_mask = vdupq_n_u32(0x8000_0000u32);
        let beta_v = vdupq_n_f32(offset_beta);
        let zero_v = vdupq_n_f32(0.0f32);
        let max_v = vdupq_n_f32(f32::MAX);

        // Number of z positions covered by full 4-wide NEON chunks.
        let full = z & !3;

        // ── Pass 1: init scratch, accumulate min1 / min2 / sign-XOR ────────────
        for i in (0..full).step_by(4) {
            vst1q_f32(min1.as_mut_ptr().add(i), max_v);
            vst1q_f32(min2.as_mut_ptr().add(i), max_v);
            vst1q_u32(sxor.as_mut_ptr().add(i), vdupq_n_u32(0));
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

            for chunk in (0..full).step_by(4) {
                let q_v = vld1q_f32(q_ptr.add(chunk));
                let q_u = vreinterpretq_u32_f32(q_v);
                let abs_v = vreinterpretq_f32_u32(vandq_u32(q_u, abs_mask));
                let sign_v = vandq_u32(q_u, sign_mask);

                // XOR sign bits into accumulator.
                let sx_ptr = sxor.as_mut_ptr().add(chunk);
                let sx_v = vld1q_u32(sx_ptr);
                vst1q_u32(sx_ptr, veorq_u32(sx_v, sign_v));

                // Branchless min1 / min2 update.
                let m1_v = vld1q_f32(m1_ptr.add(chunk));
                let m2_v = vld1q_f32(m2_ptr.add(chunk));

                // case 1: abs ≤ m1 → push m1→m2, set m1=abs
                let le = vcleq_f32(abs_v, m1_v); // uint32x4_t mask
                let new_m2 = vbslq_f32(le, m1_v, m2_v); // le: m2←m1, else keep m2
                let new_m1 = vbslq_f32(le, abs_v, m1_v); // le: m1←abs, else keep m1

                // case 2: m1 < abs < original m2
                let lt2 = vcltq_f32(abs_v, m2_v);
                let lt2_only = vandq_u32(vmvnq_u32(le), lt2); // NOT case1 AND case2
                let new_m2 = vbslq_f32(lt2_only, abs_v, new_m2);

                vst1q_f32(m1_ptr.add(chunk), new_m1);
                vst1q_f32(m2_ptr.add(chunk), new_m2);
            }
            // Scalar tail.
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

        // Clamp MAX sentinels.
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

            for chunk in (0..full).step_by(4) {
                let q_v = vld1q_f32(q_ptr.add(chunk));
                let q_u = vreinterpretq_u32_f32(q_v);
                let abs_v = vreinterpretq_f32_u32(vandq_u32(q_u, abs_mask));

                // Exclusive sign.
                let sx_v = vld1q_u32(sxor[chunk..].as_ptr());
                let excl_sign = veorq_u32(sx_v, vandq_u32(q_u, sign_mask));

                // Exclusive min.
                let m1_v = vld1q_f32(m1_ptr.add(chunk));
                let m2_v = vld1q_f32(m2_ptr.add(chunk));
                let eq_m1 = vceqq_f32(abs_v, m1_v);
                let min_excl = vbslq_f32(eq_m1, m2_v, m1_v);

                // Magnitude after offset correction.
                let mag_v = vmaxq_f32(vsubq_f32(min_excl, beta_v), zero_v);

                // Apply sign.
                let new_r_v =
                    vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(mag_v), excl_sign));

                // Update edge_r.
                let old_r_v = vld1q_f32(er_ptr.add(chunk));
                vst1q_f32(er_ptr.add(chunk), new_r_v);

                // Scatter-add delta into LLR (scalar for correctness across wraparound).
                let mut delta = [0f32; 4];
                let delta_v = vsubq_f32(new_r_v, old_r_v);
                vst1q_f32(delta.as_mut_ptr(), delta_v);
                for i in 0..4 {
                    let s = chunk + i + shift;
                    let var_idx = var_base + if s >= z { s - z } else { s };
                    // SAFETY: same argument as AVX2 kernel — var_idx < llr.len().
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
// GF(256) multiply-XOR using NEON VTBL — 16 bytes per iteration
// ---------------------------------------------------------------------------

/// GF(256) multiply-accumulate: `parity[i] ^= GF_mul(coef, data[i])` for all i.
///
/// NEON counterpart of [`crate::simd_avx2::gf256_muladd_avx2`]. Uses the same
/// nibble decomposition (`GF_mul(c, x) = lo_tbl[x & 0xF] ^ hi_tbl[x >> 4]`),
/// but `vqtbl1q_u8` is a native 16-byte table lookup (vs. AVX2's 32-byte
/// VPSHUFB), so this kernel processes 16 bytes per SIMD iteration instead of
/// 32; the scalar tail handles the remainder.
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
/// NEON is guaranteed on AArch64; no runtime detection is required.
/// `parity` must have length `>= data.len()` (only `data.len().min(parity.len())`
/// bytes are touched).
pub(crate) unsafe fn gf256_muladd_neon(
    data: &[u8],
    parity: &mut [u8],
    lo_tbl: &[u8; 16],
    hi_tbl: &[u8; 16],
    full_table: &[u8; 256],
) {
    // Rust 2024: explicit unsafe block required even inside `unsafe fn`.
    unsafe {
        let len = data.len().min(parity.len());
        let full = len & !15; // floor to multiple of 16

        let lo_tbl_v = vld1q_u8(lo_tbl.as_ptr());
        let hi_tbl_v = vld1q_u8(hi_tbl.as_ptr());
        let nibble_mask = vdupq_n_u8(0x0Fu8);

        let mut i = 0usize;
        while i < full {
            let data_v = vld1q_u8(data.as_ptr().add(i));

            // Low nibble index: mask off high nibble.
            let lo_idx = vandq_u8(data_v, nibble_mask);
            // High nibble index: shift right 4 (unsigned, so top nibble
            // lands directly in 0..16 without needing an extra mask).
            let hi_idx = vshrq_n_u8::<4>(data_v);

            let lo_val = vqtbl1q_u8(lo_tbl_v, lo_idx);
            let hi_val = vqtbl1q_u8(hi_tbl_v, hi_idx);
            let prod = veorq_u8(lo_val, hi_val);

            let par_v = vld1q_u8(parity.as_ptr().add(i));
            vst1q_u8(parity.as_mut_ptr().add(i), veorq_u8(par_v, prod));
            i += 16;
        }

        // Scalar tail.
        for j in i..len {
            parity[j] ^= full_table[data[j] as usize];
        }
    }
}
