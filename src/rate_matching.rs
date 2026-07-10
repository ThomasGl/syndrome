//! Rate matching for 5G NR QC-LDPC coded bits (TS 38.212 §5.4.2).
//!
//! Rate matching selects `E` coded bits from the circular buffer of length
//! $N_{cb}$, starting at redundancy-version offset $k_0$ (Table 5.4.2.1-2),
//! and applies bit interleaving.  Filler/`<NULL>` positions are skipped
//! during selection.
//!
//! # Mathematical Summary
//!
//! The circular buffer for code block $r$ contains $N_{cb}$ entries ordered as:
//!
//! $$d_j, \quad j = 0, 1, \ldots, N_{cb} - 1$$
//!
//! where $d[0..N_{systematic}]$ = systematic bits and
//! $d[N_{systematic}..N_{cb}]$ = parity bits.  The first $2Z$ systematic
//! positions correspond to punctured bits (transmitted as erasures, i.e.
//! absent from $d$).  Filler bits at positions $K' .. K$ within the systematic
//! section are marked and skipped.
//!
//! The bit-selection loop (§5.4.2.1) is:
//!
//! $$e_k = d_{(k_0 + j) \bmod N_{cb}}, \quad k = 0, \ldots, E-1$$
//!
//! skipping positions that are marked as `<NULL>` (filler).
//!
//! The bit interleaver (§5.4.2.2) writes columns into an $E/Q_m \times Q_m$
//! matrix row-by-row and reads out column-by-column.

use crate::error::FecError;
use crate::qc_ldpc::BaseGraph;

// ---------------------------------------------------------------------------
// RV starting offset table (TS 38.212 Table 5.4.2.1-2)
// ---------------------------------------------------------------------------

/// $k_0$ starting positions indexed by `[bg_idx][rv]` where `bg_idx = 0` for
/// BG1 and `bg_idx = 1` for BG2.
///
/// For BG1: $N = 66Z$; offsets are 0, 17Z, 33Z, 56Z of the circular buffer
/// (expressed as fractions of $N$ per the spec; we compute per-Z at runtime).
const RV_K0_SETS: [[usize; 4]; 2] = [
    // BG1: floor(17/66 * Ncb), floor(33/66 * Ncb), floor(56/66 * Ncb)
    // Expressed as (numerator, denominator) pairs; computed as floor(num*z).
    // [0, 17Z, 33Z, 56Z] but the spec uses floor((j*Ncb)/66) for j in {0,17,33,56}.
    [0, 17, 33, 56],
    // BG2: [0, 13, 25, 43] (same pattern for 50Z denominator)
    [0, 13, 25, 43],
];

/// Compute the RV starting offset $k_0$ for `rv` ∈ 0..=3.
///
/// Per TS 38.212 Table 5.4.2.1-2:
/// $$k_0 = \lfloor (j \cdot N_{cb}) / n_b \rfloor$$
/// where $j$ is the row from the table and $n_b$ is 66 (BG1) or 50 (BG2).
fn rv_k0(bg: BaseGraph, rv: usize, z: usize) -> usize {
    debug_assert!(rv < 4, "RV must be 0..=3");
    // n_b is 66 (BG1) / 50 (BG2); kept in the comment as the spec derivation.
    let bg_idx = match bg {
        BaseGraph::Bg1 => 0usize,
        BaseGraph::Bg2 => 1usize,
    };
    let j = RV_K0_SETS[bg_idx][rv];
    // k0 = floor(j * Ncb / n_b) = j * z, since Ncb = n_b * z.
    j * z
}

// ---------------------------------------------------------------------------
// Bit interleaver (§5.4.2.2)
// ---------------------------------------------------------------------------

/// Apply the TS 38.212 §5.4.2.2 bit interleaver: write `e` row-by-row into a
/// $E/Q_m \times Q_m$ matrix, then read column-by-column into `out`.
///
/// # Arguments
///
/// * `e`  - Selected bits (length `E`), modified in-place.
/// * `qm` - Modulation order (1=BPSK, 2=QPSK, 4=16QAM, 6=64QAM, 8=256QAM).
fn interleave(e: &mut [u8], qm: usize) {
    debug_assert_eq!(e.len() % qm, 0, "E must be divisible by Qm");
    let rows = e.len() / qm;
    let mut out = vec![0u8; e.len()];
    // Write row-by-row (natural order), read column-by-column.
    for col in 0..qm {
        for row in 0..rows {
            out[col * rows + row] = e[row * qm + col];
        }
    }
    e.copy_from_slice(&out);
}

/// Invert the §5.4.2.2 bit interleaver on soft values.
fn deinterleave_f32(e: &mut [f32], qm: usize) {
    debug_assert_eq!(e.len() % qm, 0, "E must be divisible by Qm");
    let rows = e.len() / qm;
    let mut out = vec![0.0f32; e.len()];
    // Reverse: input was written as col*rows+row, read as row*qm+col.
    for col in 0..qm {
        for row in 0..rows {
            out[row * qm + col] = e[col * rows + row];
        }
    }
    e.copy_from_slice(&out);
}

// ---------------------------------------------------------------------------
// Rate matching (encode direction)
// ---------------------------------------------------------------------------

/// Select and interleave `E` bits from the codeword circular buffer.
///
/// Implements TS 38.212 §5.4.2.1 (bit selection) and §5.4.2.2 (bit
/// interleaving) for a single code block.
///
/// # Arguments
///
/// * `codeword`  - Full encoded codeword of length $N = n_b \cdot Z$
///   (bits 0/1, systematic then parity).
/// * `e_out`     - Output buffer of length `e_bits`.  Filled with the
///   rate-matched, interleaved bits.
/// * `rv`        - Redundancy version (0..=3).
/// * `qm`        - Modulation order ($Q_m$); `e_bits` must be divisible by `qm`.
/// * `bg`        - Base graph.
/// * `z`         - Lifting size.
/// * `n_filler`  - Number of filler bits ($K - K'$) at positions
///   $(K' .. K)$ of the systematic section. These positions are
///   skipped (treated as `<NULL>`) during selection.
///
/// # Errors
///
/// Returns [`FecError::InvalidParam`] if `rv` ≥ 4, `qm == 0`, `z == 0`, or
/// `e_bits` is not divisible by `qm`.
///
/// # Examples
///
/// ```
/// use glezer_rsv::rate_matching::rate_match;
/// use glezer_rsv::qc_ldpc::BaseGraph;
///
/// let z = 2usize;
/// let n = 66 * z; // BG1 codeword length
/// let codeword = vec![0u8; n];
/// let e_bits = 32;
/// let mut e_out = vec![0u8; e_bits];
/// rate_match(&codeword, &mut e_out, 0, 1, BaseGraph::Bg1, z, 0).unwrap();
/// ```
pub fn rate_match(
    codeword: &[u8],
    e_out: &mut [u8],
    rv: usize,
    qm: usize,
    bg: BaseGraph,
    z: usize,
    n_filler: usize,
) -> Result<(), FecError> {
    if rv >= 4 {
        return Err(FecError::InvalidParam("RV must be 0..=3"));
    }
    if qm == 0 {
        return Err(FecError::InvalidParam("Qm must be > 0"));
    }
    if z == 0 {
        return Err(FecError::InvalidParam("lifting size Z must be > 0"));
    }
    let e_bits = e_out.len();
    if e_bits % qm != 0 {
        return Err(FecError::InvalidParam("E must be divisible by Qm"));
    }

    let n_b: usize = match bg {
        BaseGraph::Bg1 => 66,
        BaseGraph::Bg2 => 50,
    };
    let k_b: usize = match bg {
        BaseGraph::Bg1 => 22,
        BaseGraph::Bg2 => 10,
    };
    let ncb = n_b * z;
    let k = k_b * z;

    // The filler bit range in the circular buffer (systematic section only).
    // Circular buffer excludes the first 2Z punctured systematic bits, so:
    //   circular[0 .. k-2z] = systematic[2z .. k]
    //   circular[k-2z .. ncb-2z] = parity
    // Filler positions in systematic = [k_prime .. k], i.e. the last n_filler
    // systematic positions after removing the first 2Z punctured ones.
    let k_prime = k.saturating_sub(n_filler);
    // In the circular buffer (0-indexed, punctured columns excluded):
    // filler occupies [k_prime - 2z .. k - 2z]
    let two_z = 2 * z;
    let filler_start_cb = k_prime.saturating_sub(two_z);
    let filler_end_cb = k.saturating_sub(two_z);

    let k0 = rv_k0(bg, rv, z);
    let mut k_sel = 0usize;
    let mut j = 0usize;
    while k_sel < e_bits {
        let pos = (k0 + j) % ncb;
        // Skip filler/NULL positions.
        if pos >= filler_start_cb && pos < filler_end_cb {
            j += 1;
            continue;
        }
        // Map circular buffer position back to the full codeword. Both the
        // systematic and parity portions shift by the same 2Z that was excluded
        // at the start of the buffer, so the mapping is uniform.
        let cw_pos = pos + two_z;
        e_out[k_sel] = if cw_pos < codeword.len() {
            codeword[cw_pos]
        } else {
            0
        };
        k_sel += 1;
        j += 1;
        if j > 2 * ncb {
            break; // guard against infinite loop if ncb is tiny
        }
    }

    // Bit interleaving (§5.4.2.2).
    let mut e_vec = e_out.to_vec();
    interleave(&mut e_vec, qm);
    e_out.copy_from_slice(&e_vec);

    Ok(())
}

// ---------------------------------------------------------------------------
// Rate de-matching (decode direction, LLR domain)
// ---------------------------------------------------------------------------

/// De-interleave and scatter received soft LLRs back into a code-block
/// accumulation buffer for HARQ combining.
///
/// This is the inverse of [`rate_match`]: it reverses the §5.4.2.2
/// interleaver, then writes each received LLR into the circular buffer at its
/// original bit-selection position.  The accumulation buffer is updated with
/// saturating addition (LLR combining via simple accumulation).
///
/// # Arguments
///
/// * `e_llr`     - Received soft LLRs (length `E`), one per coded bit.
/// * `cb_llr`    - Circular-buffer LLR accumulator of length $N_{cb}$.
///   Updated (add) by this call — zero-initialise for first
///   transmission, leave non-zero for HARQ combining.
/// * `rv`        - Redundancy version (0..=3).
/// * `qm`        - Modulation order.
/// * `bg`        - Base graph.
/// * `z`         - Lifting size.
/// * `n_filler`  - Number of filler bits.
///
/// # Errors
///
/// Returns [`FecError::InvalidParam`] if `rv` ≥ 4, `qm == 0`, `z == 0`, or
/// `E % Qm ≠ 0`.
///
/// # Examples
///
/// ```
/// use glezer_rsv::rate_matching::rate_dematch_llr;
/// use glezer_rsv::qc_ldpc::BaseGraph;
///
/// let z = 2usize;
/// let ncb = 66 * z;
/// let e_llr = vec![1.0f32; 32];
/// let mut cb_llr = vec![0.0f32; ncb];
/// rate_dematch_llr(&e_llr, &mut cb_llr, 0, 1, BaseGraph::Bg1, z, 0).unwrap();
/// ```
pub fn rate_dematch_llr(
    e_llr: &[f32],
    cb_llr: &mut [f32],
    rv: usize,
    qm: usize,
    bg: BaseGraph,
    z: usize,
    n_filler: usize,
) -> Result<(), FecError> {
    if rv >= 4 {
        return Err(FecError::InvalidParam("RV must be 0..=3"));
    }
    if qm == 0 {
        return Err(FecError::InvalidParam("Qm must be > 0"));
    }
    if z == 0 {
        return Err(FecError::InvalidParam("lifting size Z must be > 0"));
    }
    let e_bits = e_llr.len();
    if e_bits % qm != 0 {
        return Err(FecError::InvalidParam("E must be divisible by Qm"));
    }

    let n_b: usize = match bg {
        BaseGraph::Bg1 => 66,
        BaseGraph::Bg2 => 50,
    };
    let k_b: usize = match bg {
        BaseGraph::Bg1 => 22,
        BaseGraph::Bg2 => 10,
    };
    let ncb = n_b * z;
    let k = k_b * z;
    let k_prime = k.saturating_sub(n_filler);
    let two_z = 2 * z;
    let filler_start_cb = k_prime.saturating_sub(two_z);
    let filler_end_cb = k.saturating_sub(two_z);

    // Reverse bit interleaver first.
    let mut e_deint = e_llr.to_vec();
    deinterleave_f32(&mut e_deint, qm);

    let k0 = rv_k0(bg, rv, z);
    let mut k_sel = 0usize;
    let mut j = 0usize;
    while k_sel < e_bits {
        let pos = (k0 + j) % ncb;
        if pos >= filler_start_cb && pos < filler_end_cb {
            j += 1;
            continue;
        }
        if pos < cb_llr.len() {
            cb_llr[pos] += e_deint[k_sel];
        }
        k_sel += 1;
        j += 1;
        if j > 2 * ncb {
            break;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rv_k0_bg1_zero() {
        // RV=0 always starts at 0.
        assert_eq!(rv_k0(BaseGraph::Bg1, 0, 384), 0);
        assert_eq!(rv_k0(BaseGraph::Bg2, 0, 128), 0);
    }

    #[test]
    fn rv_k0_bg1_rv1() {
        // RV=1 BG1: j=17, z=384 → k0 = 17*384 = 6528
        assert_eq!(rv_k0(BaseGraph::Bg1, 1, 384), 17 * 384);
    }

    #[test]
    fn rate_match_output_length() {
        let z = 2usize;
        let n = 66 * z;
        let codeword = vec![0u8; n];
        let e_bits = 24;
        let mut e_out = vec![0u8; e_bits];
        rate_match(&codeword, &mut e_out, 0, 1, BaseGraph::Bg1, z, 0).unwrap();
        assert_eq!(e_out.len(), e_bits);
    }

    #[test]
    fn interleave_roundtrip() {
        let mut v: Vec<u8> = (0..16).collect();
        let orig = v.clone();
        interleave(&mut v, 4);
        // After interleave, must be different from original (unless trivial case).
        // De-interleave as f32 and verify round-trip.
        let mut vf: Vec<f32> = v.iter().map(|&b| b as f32).collect();
        deinterleave_f32(&mut vf, 4);
        let recovered: Vec<u8> = vf.iter().map(|&f| f as u8).collect();
        assert_eq!(recovered, orig);
    }

    #[test]
    fn rate_dematch_accumulates_llr() {
        let z = 2usize;
        let ncb = 66 * z;
        let e_bits = 16;
        let e_llr = vec![1.0f32; e_bits];
        let mut cb = vec![0.0f32; ncb];
        rate_dematch_llr(&e_llr, &mut cb, 0, 1, BaseGraph::Bg1, z, 0).unwrap();
        // At least some positions in the cb must be non-zero.
        assert!(cb.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn invalid_rv_rejected() {
        let codeword = vec![0u8; 66 * 2];
        let mut e_out = vec![0u8; 8];
        assert!(rate_match(&codeword, &mut e_out, 4, 1, BaseGraph::Bg1, 2, 0).is_err());
    }

    /// FINDING 2/4 regression guard (was `finding_rate_match_qm_zero_panics`
    /// / `finding_rate_dematch_llr_qm_zero_panics` in tests/robustness.rs,
    /// `#[should_panic]`): `qm == 0` used to divide-by-zero via `e_bits %
    /// qm` before any validation ran.
    #[test]
    fn qm_zero_rejected() {
        let codeword = vec![0u8; 66 * 2];
        let mut e_out = vec![0u8; 8];
        assert!(rate_match(&codeword, &mut e_out, 0, 0, BaseGraph::Bg1, 2, 0).is_err());

        let e_llr = vec![1.0f32; 8];
        let mut cb = vec![0.0f32; 200];
        assert!(rate_dematch_llr(&e_llr, &mut cb, 0, 0, BaseGraph::Bg1, 2, 0).is_err());
    }

    /// FINDING 3/5 regression guard (was `finding_rate_match_z_zero_panics`
    /// / `finding_rate_dematch_llr_z_zero_panics` in tests/robustness.rs,
    /// `#[should_panic]`): `z == 0` used to divide-by-zero via
    /// `(k0 + j) % ncb` (`ncb = n_b * z == 0`) before any validation ran.
    #[test]
    fn z_zero_rejected() {
        let codeword = vec![0u8; 66 * 2];
        let mut e_out = vec![0u8; 8];
        assert!(rate_match(&codeword, &mut e_out, 0, 1, BaseGraph::Bg1, 0, 0).is_err());

        let e_llr = vec![1.0f32; 8];
        let mut cb = vec![0.0f32; 200];
        assert!(rate_dematch_llr(&e_llr, &mut cb, 0, 1, BaseGraph::Bg1, 0, 0).is_err());
    }
}
