//! Rate matching for 5G NR QC-LDPC coded bits (TS 38.212 §5.4.2).
//!
//! Rate matching selects `E` coded bits from the circular buffer of length
//! $N_{cb}$, starting at redundancy-version offset $k_0$ (Table 5.4.2.1-2),
//! and applies bit interleaving.  Filler/`<NULL>` positions are skipped
//! during selection.
//!
//! # Why this exists, in plain English
//!
//! An LDPC encoder always emits a fixed-size codeword (systematic bits +
//! parity bits), but a real transmission needs some other, arbitrary number
//! of bits `E` -- exactly enough to fill the modulation symbols scheduled for
//! this transmission opportunity, no more, no less. Rate matching is the
//! adapter between those two sizes: it walks the codeword bits in a fixed,
//! wraparound (circular) order and copies out exactly `E` of them, skipping
//! placeholder "filler" bits that carry no information. Different
//! redundancy versions (RVs) start that walk from different offsets, which
//! is what lets HARQ retransmissions send different (or overlapping)
//! subsets of the same codeword rather than a byte-identical repeat.
//!
//! ```text
//! codeword:      [ 2Z punctured ][   systematic bits   ][   parity bits   ]
//!                                 └── K'..K = filler ───┘
//!                                        (skipped)
//! circular buf:  [   systematic (minus punctured/filler)  ][ parity ]
//!                 0                                                N_cb-1
//!                        ▲
//!                        k0 (RV-dependent start offset)
//!
//! bit selection:  walk (k0 + j) mod N_cb, j = 0, 1, 2, ...
//!                 skip positions inside the filler range
//!                 take the first E survivors  ──▶  e[0..E]
//!
//! bit interleaving: write e[] row-by-row into a (E/Qm) x Qm matrix,
//!                   read it back out column-by-column
//! ```
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

use crate::alloc_prelude::*;
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
/// # Returns
///
/// `Ok(())` on success; `e_out` holds the `E` rate-matched, interleaved bits.
///
/// # Errors
///
/// Returns [`FecError::InvalidParam`] if `rv` ≥ 4, `qm == 0`, `z == 0`, or
/// `e_bits` is not divisible by `qm`.
///
/// # Examples
///
/// ```
/// use syndrome::rate_matching::rate_match;
/// use syndrome::qc_ldpc::BaseGraph;
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
    if !e_bits.is_multiple_of(qm) {
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
/// # Returns
///
/// `Ok(())` on success; `cb_llr` has the de-interleaved LLRs added in.
///
/// # Errors
///
/// Returns [`FecError::InvalidParam`] if `rv` ≥ 4, `qm == 0`, `z == 0`, or
/// `E % Qm ≠ 0`.
///
/// # Examples
///
/// ```
/// use syndrome::rate_matching::rate_dematch_llr;
/// use syndrome::qc_ldpc::BaseGraph;
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
    if !e_bits.is_multiple_of(qm) {
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
// Cached rate-matching index tables (hot path for multi-code-block TBs)
// ---------------------------------------------------------------------------

/// Identifies one rate-matching "shape": everything [`rate_match`] /
/// [`rate_dematch_llr`]'s bit-selection walk depends on *except* the
/// codeword/LLR content itself.
///
/// All `C` code blocks of a transport block share one of these (see
/// `DlSchEncoder::encode`'s doc), so the selection/interleave index tables
/// derived from it are safe to reuse verbatim across every code block,
/// rebuilding only when the key actually changes (e.g. a new RV on HARQ
/// retransmission).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct RateMatchKey {
    bg: BaseGraph,
    z: usize,
    rv: usize,
    qm: usize,
    n_filler: usize,
    e_bits: usize,
}

/// Map an interleaved-domain (wire order) index to its selection-domain
/// (pre-interleave / [`rv_k0`]-walk order) index.
///
/// Derived directly from [`interleave`]'s forward definition
/// `out[col*rows+row] = e[row*qm+col]`: solving `p = col*rows+row` for
/// `(row, col)` given `0 <= row < rows` gives `row = p % rows`,
/// `col = p / rows`, so the value written to interleaved position `p` was
/// read from selection-domain position `(p % rows)*qm + (p / rows)`.
/// [`deinterleave_f32`]'s reverse definition
/// (`out[row*qm+col] = e[col*rows+row]`) needs the identical formula for
/// its input index `q` (`q = col*rows+row` also gives `row = q % rows`,
/// `col = q / rows`), so encode and decode share this one helper.
#[inline]
fn interleaved_to_selection_index(x: usize, qm: usize, rows: usize) -> usize {
    (x % rows) * qm + (x / rows)
}

/// Run the §5.4.2.1 `k0 + j` bit-selection walk (shared by [`rate_match`]
/// and [`rate_dematch_llr`]) once, returning the **circular-buffer**
/// position selected for each of the `e_bits` selection-domain slots
/// (filler/`<NULL>` positions are skipped over, exactly as in the original
/// inline walk). This is the part of rate matching that depends only on
/// `(bg, z, rv, n_filler, e_bits)` -- not on `qm` (interleaving) or on
/// codeword content -- and therefore the part [`RateMatchCache`] memoizes.
fn selection_walk(
    bg: BaseGraph,
    z: usize,
    rv: usize,
    n_filler: usize,
    e_bits: usize,
) -> Vec<usize> {
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

    let k0 = rv_k0(bg, rv, z);
    let mut sel_pos = vec![0usize; e_bits];
    let mut k_sel = 0usize;
    let mut j = 0usize;
    while k_sel < e_bits {
        let pos = (k0 + j) % ncb;
        if pos >= filler_start_cb && pos < filler_end_cb {
            j += 1;
            continue;
        }
        sel_pos[k_sel] = pos;
        k_sel += 1;
        j += 1;
        if j > 2 * ncb {
            break; // guard against infinite loop if ncb is tiny
        }
    }
    sel_pos
}

/// Preallocated, key-memoized rate-matching index tables, reused across the
/// `C` code blocks of one transport block (or across a `HarqBuffer`'s
/// repeated `combine` calls at a fixed RV).
///
/// The naive per-call path recomputes the whole §5.4.2.1 `k0 + j` selection
/// walk *and* allocates fresh interleave/deinterleave scratch `Vec`s on
/// every call, even though every code block in a transport block shares the
/// same `(bg, z, rv, qm, n_filler, e_bits)` -- so that work (and those
/// allocations) is identical `C` times over. This cache fuses the
/// selection walk and the §5.4.2.2 interleaver/deinterleaver into one flat
/// gather-index table per direction (`enc_idx`/`dec_idx`, both length
/// `e_bits`), built once per distinct key, and reused thereafter with only
/// an `O(e_bits)` gather/scatter loop -- no per-call allocation, no
/// per-call re-derivation of the index sequence.
///
/// # Examples
///
/// ```
/// use syndrome::rate_matching::{RateMatchCache, rate_match};
/// use syndrome::qc_ldpc::BaseGraph;
///
/// let z = 2usize;
/// let n = 66 * z;
/// let codeword = vec![1u8; n];
/// let mut cache = RateMatchCache::new();
/// let mut e_cached = vec![0u8; 32];
/// let mut e_direct = vec![0u8; 32];
/// cache
///     .rate_match_into(&codeword, &mut e_cached, 0, 1, BaseGraph::Bg1, z, 0)
///     .unwrap();
/// rate_match(&codeword, &mut e_direct, 0, 1, BaseGraph::Bg1, z, 0).unwrap();
/// assert_eq!(e_cached, e_direct);
/// ```
#[derive(Default)]
pub struct RateMatchCache {
    key: Option<RateMatchKey>,
    /// `enc_idx[p]` = codeword bit index to read for rate-matched +
    /// interleaved output position `p` (fused selection + interleave
    /// gather).
    enc_idx: Vec<usize>,
    /// `dec_idx[q]` = circular-buffer index to accumulate into for
    /// received-order (interleaved-domain) LLR position `q` (fused
    /// deinterleave + selection scatter).
    dec_idx: Vec<usize>,
}

impl RateMatchCache {
    /// Create an empty cache. The first call to either `_into` method below
    /// builds the tables; subsequent calls with the same key reuse them.
    ///
    /// # Returns
    ///
    /// A [`RateMatchCache`] with no tables built yet (built lazily on first
    /// use).
    ///
    /// # Examples
    ///
    /// ```
    /// use syndrome::rate_matching::RateMatchCache;
    /// let cache = RateMatchCache::new();
    /// drop(cache); // no tables allocated until first rate_match_into call
    /// ```
    pub fn new() -> Self {
        Self {
            key: None,
            enc_idx: Vec::new(),
            dec_idx: Vec::new(),
        }
    }

    /// Rebuild `enc_idx`/`dec_idx` for `key` if they don't already match it.
    fn ensure(&mut self, key: RateMatchKey, two_z: usize) {
        if self.key == Some(key) {
            return;
        }
        let sel_pos = selection_walk(key.bg, key.z, key.rv, key.n_filler, key.e_bits);
        let rows = key.e_bits / key.qm;

        self.enc_idx.clear();
        self.enc_idx.extend((0..key.e_bits).map(|p| {
            let k_sel = interleaved_to_selection_index(p, key.qm, rows);
            sel_pos[k_sel] + two_z
        }));

        self.dec_idx.clear();
        self.dec_idx.extend((0..key.e_bits).map(|q| {
            let k_sel = interleaved_to_selection_index(q, key.qm, rows);
            sel_pos[k_sel]
        }));

        self.key = Some(key);
    }

    /// Cached equivalent of [`rate_match`]: identical bit-selection +
    /// interleaving, but reuses this cache's index tables when `(bg, z,
    /// rv, qm, n_filler, e_out.len())` matches the previous call, and
    /// gathers directly from `codeword` into `e_out` with no intermediate
    /// `Vec` allocation.
    ///
    /// # Arguments
    ///
    /// Same as [`rate_match`], plus `self`, which owns the memoized index
    /// tables (rebuilt only when the `(bg, z, rv, qm, n_filler, e_bits)` key
    /// changes from the previous call on this cache).
    ///
    /// # Returns
    ///
    /// `Ok(())` on success; `e_out` holds the `E` rate-matched, interleaved
    /// bits, bit-for-bit identical to what [`rate_match`] would produce for
    /// the same inputs.
    ///
    /// # Errors
    ///
    /// Same conditions as [`rate_match`]: `rv >= 4`, `qm == 0`, `z == 0`, or
    /// `e_out.len() % qm != 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use syndrome::rate_matching::{RateMatchCache, rate_match};
    /// use syndrome::qc_ldpc::BaseGraph;
    ///
    /// let z = 2usize;
    /// let codeword = vec![1u8; 66 * z];
    /// let mut cache = RateMatchCache::new();
    /// let mut e_cached = vec![0u8; 32];
    /// cache
    ///     .rate_match_into(&codeword, &mut e_cached, 0, 1, BaseGraph::Bg1, z, 0)
    ///     .unwrap();
    /// let mut e_direct = vec![0u8; 32];
    /// rate_match(&codeword, &mut e_direct, 0, 1, BaseGraph::Bg1, z, 0).unwrap();
    /// assert_eq!(e_cached, e_direct);
    /// ```
    pub fn rate_match_into(
        &mut self,
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
        if !e_bits.is_multiple_of(qm) {
            return Err(FecError::InvalidParam("E must be divisible by Qm"));
        }

        let key = RateMatchKey {
            bg,
            z,
            rv,
            qm,
            n_filler,
            e_bits,
        };
        self.ensure(key, 2 * z);

        for (out_bit, &idx) in e_out.iter_mut().zip(self.enc_idx.iter()) {
            *out_bit = if idx < codeword.len() {
                codeword[idx]
            } else {
                0
            };
        }
        Ok(())
    }

    /// Cached equivalent of [`rate_dematch_llr`]: identical deinterleaving +
    /// scatter-accumulate, but reuses this cache's index table when the key
    /// matches, with no intermediate `Vec` allocation.
    ///
    /// # Arguments
    ///
    /// Same as [`rate_dematch_llr`], plus `self`, which owns the memoized
    /// index table (rebuilt only when the `(bg, z, rv, qm, n_filler,
    /// e_bits)` key changes from the previous call on this cache).
    ///
    /// # Returns
    ///
    /// `Ok(())` on success; `cb_llr` has the de-interleaved LLRs added in,
    /// bit-for-bit identical to what [`rate_dematch_llr`] would produce for
    /// the same inputs.
    ///
    /// # Errors
    ///
    /// Same conditions as [`rate_dematch_llr`]: `rv >= 4`, `qm == 0`, `z ==
    /// 0`, or `e_llr.len() % qm != 0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use syndrome::rate_matching::{RateMatchCache, rate_dematch_llr};
    /// use syndrome::qc_ldpc::BaseGraph;
    ///
    /// let z = 2usize;
    /// let e_llr = vec![1.0f32; 32];
    /// let mut cache = RateMatchCache::new();
    /// let mut cb_cached = vec![0.0f32; 66 * z];
    /// cache
    ///     .rate_dematch_llr_into(&e_llr, &mut cb_cached, 0, 1, BaseGraph::Bg1, z, 0)
    ///     .unwrap();
    /// let mut cb_direct = vec![0.0f32; 66 * z];
    /// rate_dematch_llr(&e_llr, &mut cb_direct, 0, 1, BaseGraph::Bg1, z, 0).unwrap();
    /// assert_eq!(cb_cached, cb_direct);
    /// ```
    pub fn rate_dematch_llr_into(
        &mut self,
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
        if !e_bits.is_multiple_of(qm) {
            return Err(FecError::InvalidParam("E must be divisible by Qm"));
        }

        let key = RateMatchKey {
            bg,
            z,
            rv,
            qm,
            n_filler,
            e_bits,
        };
        self.ensure(key, 2 * z);

        for (&llr, &idx) in e_llr.iter().zip(self.dec_idx.iter()) {
            if idx < cb_llr.len() {
                cb_llr[idx] += llr;
            }
        }
        Ok(())
    }
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

    use crate::test_util::Xorshift64;

    /// Task-2 equivalence guard: [`RateMatchCache::rate_match_into`] must
    /// match the uncached [`rate_match`] bit-for-bit, across several (bg,
    /// z, rv, qm, n_filler, e_bits) shapes and many random codewords each
    /// -- including re-using one cache instance across shapes, to exercise
    /// both the "key unchanged, reuse tables" and "key changed, rebuild"
    /// paths.
    #[test]
    fn cached_rate_match_matches_uncached_random() {
        struct Cfg {
            bg: BaseGraph,
            z: usize,
            rv: usize,
            qm: usize,
            n_filler: usize,
            e_bits: usize,
        }
        let configs = [
            Cfg {
                bg: BaseGraph::Bg1,
                z: 4,
                rv: 0,
                qm: 1,
                n_filler: 0,
                e_bits: 64,
            },
            Cfg {
                bg: BaseGraph::Bg1,
                z: 8,
                rv: 1,
                qm: 2,
                n_filler: 6,
                e_bits: 120,
            },
            Cfg {
                bg: BaseGraph::Bg2,
                z: 6,
                rv: 2,
                qm: 4,
                n_filler: 4,
                e_bits: 96,
            },
            Cfg {
                bg: BaseGraph::Bg2,
                z: 16,
                rv: 3,
                qm: 6,
                n_filler: 20,
                e_bits: 180,
            },
        ];

        let mut cache = RateMatchCache::new();
        let mut rng = Xorshift64::new(0xCACE_5EED);

        for cfg in &configs {
            let n_b = match cfg.bg {
                BaseGraph::Bg1 => 66,
                BaseGraph::Bg2 => 50,
            };
            let n = n_b * cfg.z;
            for _trial in 0..30 {
                let codeword: Vec<u8> = (0..n).map(|_| (rng.next_u64() & 1) as u8).collect();

                let mut e_cached = vec![0u8; cfg.e_bits];
                let mut e_direct = vec![0u8; cfg.e_bits];
                cache
                    .rate_match_into(
                        &codeword,
                        &mut e_cached,
                        cfg.rv,
                        cfg.qm,
                        cfg.bg,
                        cfg.z,
                        cfg.n_filler,
                    )
                    .unwrap();
                rate_match(
                    &codeword,
                    &mut e_direct,
                    cfg.rv,
                    cfg.qm,
                    cfg.bg,
                    cfg.z,
                    cfg.n_filler,
                )
                .unwrap();

                assert_eq!(
                    e_cached, e_direct,
                    "cached vs uncached rate_match mismatch: bg={:?} z={} rv={} qm={} n_filler={}",
                    cfg.bg, cfg.z, cfg.rv, cfg.qm, cfg.n_filler
                );
            }
        }
    }

    /// Same as above but for the decode-direction (`rate_dematch_llr` vs
    /// [`RateMatchCache::rate_dematch_llr_into`]), comparing the resulting
    /// accumulator buffers after applying several transmissions (so HARQ's
    /// use pattern of repeated `combine` calls at possibly-changing RVs is
    /// also exercised).
    #[test]
    fn cached_rate_dematch_matches_uncached_random() {
        struct Cfg {
            bg: BaseGraph,
            z: usize,
            qm: usize,
            n_filler: usize,
            e_bits: usize,
            rvs: &'static [usize],
        }
        let configs = [
            Cfg {
                bg: BaseGraph::Bg1,
                z: 4,
                qm: 1,
                n_filler: 0,
                e_bits: 64,
                rvs: &[0, 0, 1],
            },
            Cfg {
                bg: BaseGraph::Bg2,
                z: 10,
                qm: 4,
                n_filler: 8,
                e_bits: 160,
                rvs: &[0, 2, 3],
            },
        ];

        let mut cache = RateMatchCache::new();
        let mut rng = Xorshift64::new(0xDECA_11ED);

        for cfg in &configs {
            let n_b = match cfg.bg {
                BaseGraph::Bg1 => 66,
                BaseGraph::Bg2 => 50,
            };
            let ncb = n_b * cfg.z;

            let mut cb_cached = vec![0.0f32; ncb];
            let mut cb_direct = vec![0.0f32; ncb];

            for &rv in cfg.rvs {
                let e_llr: Vec<f32> = (0..cfg.e_bits)
                    .map(|_| (rng.next_u64() % 2000) as f32 / 100.0 - 10.0)
                    .collect();

                cache
                    .rate_dematch_llr_into(
                        &e_llr,
                        &mut cb_cached,
                        rv,
                        cfg.qm,
                        cfg.bg,
                        cfg.z,
                        cfg.n_filler,
                    )
                    .unwrap();
                rate_dematch_llr(
                    &e_llr,
                    &mut cb_direct,
                    rv,
                    cfg.qm,
                    cfg.bg,
                    cfg.z,
                    cfg.n_filler,
                )
                .unwrap();

                for (i, (&c, &d)) in cb_cached.iter().zip(cb_direct.iter()).enumerate() {
                    assert!(
                        (c - d).abs() < 1e-6,
                        "cached vs uncached rate_dematch_llr mismatch at cb[{i}]: {c} vs {d} (bg={:?} z={} rv={rv} qm={})",
                        cfg.bg,
                        cfg.z,
                        cfg.qm
                    );
                }
            }
        }
    }
}
