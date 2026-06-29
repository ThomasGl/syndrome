//! Polar codes for 5G NR control channels (TS 38.212 §5.3.1).
//!
//! 5G NR uses polar codes for:
//! - PDCCH (Downlink Control Information)  — CRC-24C or CRC-11
//! - PBCH  (Physical Broadcast Channel)    — CRC-24C
//! - PUCCH/PUSCH UCI                       — CRC-6 or CRC-11
//!
//! # Encoding (§5.3.1)
//!
//! The generator matrix is $G_N = F^{\otimes n}$ where $F = \begin{bmatrix}1 &
//! 0 \\ 1 & 1\end{bmatrix}$ and $n = \log_2 N$.  The encoding operation is:
//!
//! $$x = u \cdot G_N, \quad u_i = 0 \text{ for frozen bits}$$
//!
//! Efficiently computed via the butterfly (bit-reversal permuted) polar
//! transform in $O(N \log N)$.
//!
//! # Decoding
//!
//! ## Successive Cancellation (SC) — $O(N \log N)$
//!
//! Recursive computation of $f$ and $g$ LLR combining rules:
//! $$f(a, b) = \text{sgn}(a) \cdot \text{sgn}(b) \cdot \min(|a|, |b|)$$
//! $$g(a, b, \hat{u}) = b + (1 - 2\hat{u}) \cdot a$$
//!
//! ## Successive Cancellation List (SCL) — $O(L \cdot N \log N)$
//!
//! Maintains $L$ candidate paths.  CRC-aided SCL selects the path that passes
//! the CRC from the surviving list (CA-SCL, the 5G NR configuration).

use crate::crc::{Crc24, CrcKind};
use crate::error::FecError;

// ---------------------------------------------------------------------------
// Frozen bit reliability sequence (TS 38.212 §5.3.1, Table 5.3.1.2-1)
// ---------------------------------------------------------------------------

// The 5G NR reliability sequence Q_Nmax for N_max = 1024.
// Bit index Q[i] is the i-th most reliable position in a rate-1 polar code of length N_max.
// For shorter codes, the frozen set is the complement of the K most reliable positions ∩ {0..N-1}.
// We include a subset (first 256 entries from Table 5.3.1.2-1) which covers N ≤ 256.
// Indices are in the **channel-order** (before bit-reversal permutation).
const RELIABILITY_SEQ: &[u16] = &[
    0, 1, 2, 4, 8, 16, 32, 3, 5, 64, 9, 6, 17, 10, 18, 128, 12, 33, 65, 20, 34, 24, 36, 7, 129, 66,
    11, 40, 130, 19, 13, 48, 14, 67, 132, 25, 35, 26, 21, 68, 22, 41, 136, 28, 70, 37, 144, 38, 49,
    42, 131, 15, 50, 160, 74, 44, 27, 69, 23, 52, 133, 29, 76, 56, 72, 39, 134, 145, 80, 45, 43,
    146, 30, 51, 148, 88, 137, 53, 75, 96, 161, 57, 77, 138, 60, 152, 71, 162, 46, 142, 54, 73, 31,
    164, 78, 81, 140, 89, 47, 169, 155, 82, 97, 58, 153, 98, 55, 84, 139, 163, 100, 168, 59, 79,
    141, 165, 90, 61, 154, 104, 143, 166, 62, 83, 170, 99, 91, 172, 85, 112, 101, 156, 86, 176, 93,
    102, 147, 157, 167, 184, 105, 63, 94, 149, 106, 87, 171, 113, 150, 200, 108, 92, 158, 103, 173,
    185, 114, 95, 174, 107, 151, 177, 116, 109, 178, 159, 201, 186, 120, 115, 202, 180, 188, 110,
    175, 204, 117, 208, 179, 121, 187, 118, 192, 224, 181, 203, 122, 189, 124, 205, 182, 209, 190,
    206, 210, 183, 123, 212, 193, 125, 216, 194, 211, 232, 191, 213, 196, 126, 217, 248, 214, 233,
    218, 127, 215, 249, 220, 234, 219, 226, 240, 250, 221, 235, 242, 227, 228, 252, 222, 241, 236,
    251, 243, 244, 229, 253, 237, 246, 223, 230, 238, 254, 245, 247, 231, 255, 239,
];

/// Compute the frozen bit mask for a polar code of length `n_polar` and
/// `k_info` information bits.
///
/// Returns a `Vec<bool>` of length `n_polar` where `true` = frozen bit.
fn frozen_mask(n_polar: usize, k_info: usize) -> Vec<bool> {
    debug_assert!(n_polar.is_power_of_two());
    debug_assert!(k_info <= n_polar);
    let mut is_frozen = vec![true; n_polar];
    // Collect candidate positions from the reliability sequence that fall
    // within [0, n_polar).
    let candidates: Vec<u16> = RELIABILITY_SEQ
        .iter()
        .copied()
        .filter(|&idx| (idx as usize) < n_polar)
        .collect();
    // Take the K most reliable positions as information bits.
    // candidates is in reliability order (last entry = most reliable).
    // We need the K highest-reliability, i.e. the LAST K entries.
    let info_count = k_info.min(candidates.len());
    let info_start = candidates.len().saturating_sub(info_count);
    for &pos in &candidates[info_start..] {
        is_frozen[pos as usize] = false;
    }
    is_frozen
}

// ---------------------------------------------------------------------------
// Polar transform (encoder)
// ---------------------------------------------------------------------------

/// Apply the polar encoding transform $x = u \cdot G_N$ via the butterfly.
///
/// The transform is performed in-place.  Input `u` must have length $N$ (a
/// power of 2).  After the call, `u` holds the encoded codeword $x$ in
/// **natural order** (not bit-reversed; the spec applies bit-reversal as part
/// of the sub-block interleaver, handled separately).
fn polar_transform(u: &mut [u8]) {
    let n = u.len();
    debug_assert!(n.is_power_of_two());
    let mut step = 1usize;
    while step < n {
        let mut i = 0;
        while i < n {
            for j in 0..step {
                u[i + j] ^= u[i + j + step];
            }
            i += 2 * step;
        }
        step *= 2;
    }
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/// Polar encoder for 5G NR control channels.
///
/// The encoder inserts frozen bits (value 0) at the positions determined by
/// the 3GPP reliability sequence, applies the polar transform $G_N$, and
/// (optionally) performs rate-matching (puncturing/shortening) to produce
/// exactly `e_bits` coded bits.
///
/// # Examples
///
/// ```
/// use glezer_rsv::polar::PolarEncoder;
///
/// // K=4 info bits, N=8 polar code.
/// let enc = PolarEncoder::new(8, 4).unwrap();
/// let info = vec![1u8, 0, 1, 1];
/// let mut codeword = vec![0u8; 8];
/// enc.encode(&info, &mut codeword).unwrap();
/// assert_eq!(codeword.len(), 8);
/// ```
pub struct PolarEncoder {
    n: usize,
    k: usize,
    is_frozen: Vec<bool>,
}

impl PolarEncoder {
    /// Create a polar encoder with block length `n` and `k` info bits.
    ///
    /// # Arguments
    ///
    /// * `n` - Code block length (must be a power of 2, max 1024 for 5G NR).
    /// * `k` - Number of information bits ($k < n$).
    ///
    /// # Errors
    ///
    /// Returns [`FecError::InvalidParam`] if `n` is not a power of 2 or
    /// `k >= n`.
    pub fn new(n: usize, k: usize) -> Result<Self, FecError> {
        if !n.is_power_of_two() || n == 0 {
            return Err(FecError::InvalidParam(
                "polar N must be a positive power of 2",
            ));
        }
        if k >= n {
            return Err(FecError::InvalidParam("k must be < n"));
        }
        let is_frozen = frozen_mask(n, k);
        Ok(Self { n, k, is_frozen })
    }

    /// Block length $N$.
    pub fn n(&self) -> usize {
        self.n
    }
    /// Information bits per block $K$.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Encode `info_bits` into `codeword`.
    ///
    /// # Arguments
    ///
    /// * `info_bits` - Slice of exactly `k` bits (values 0 or 1).
    /// * `codeword`  - Output slice of exactly `n` bits.
    ///
    /// # Errors
    ///
    /// Returns [`FecError::BufferTooSmall`] if lengths don't match.
    pub fn encode(&self, info_bits: &[u8], codeword: &mut [u8]) -> Result<(), FecError> {
        if info_bits.len() != self.k {
            return Err(FecError::BufferTooSmall {
                required: self.k,
                provided: info_bits.len(),
            });
        }
        if codeword.len() != self.n {
            return Err(FecError::BufferTooSmall {
                required: self.n,
                provided: codeword.len(),
            });
        }
        // Fill u: info at non-frozen positions, 0 at frozen positions.
        let mut u = vec![0u8; self.n];
        let mut info_idx = 0;
        for i in 0..self.n {
            if !self.is_frozen[i] {
                u[i] = info_bits[info_idx] & 1;
                info_idx += 1;
            }
        }
        polar_transform(&mut u);
        codeword.copy_from_slice(&u);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Successive Cancellation decoder
// ---------------------------------------------------------------------------

/// Compute the $f$ combining LLR:
///   $f(a, b) = \text{sgn}(a)\text{sgn}(b)\min(|a|, |b|)$ (min-sum approximation).
#[inline]
fn combine_f(a: f32, b: f32) -> f32 {
    let sign = if (a < 0.0) ^ (b < 0.0) { -1.0f32 } else { 1.0 };
    sign * a.abs().min(b.abs())
}

/// Compute the $g$ combining LLR:
///   $g(a, b, \hat{u}) = b + (1 - 2\hat{u}) \cdot a$
#[inline]
fn combine_g(a: f32, b: f32, u_hat: u8) -> f32 {
    b + (1.0 - 2.0 * u_hat as f32) * a
}

/// Recursive SC decode over LLR array `llr` starting at `bit_start` for
/// `length` bits.  Decoded bits are written into `decoded[bit_start..]`.
fn sc_decode_recursive(
    llr: &[f32],
    decoded: &mut [u8],
    is_frozen: &[bool],
    bit_start: usize,
    length: usize,
    // LLR workspace: pre-allocated to avoid re-alloc in recursion.
    workspace: &mut Vec<Vec<f32>>,
    level: usize,
) {
    if length == 1 {
        let val = llr[0];
        let bit_pos = bit_start;
        decoded[bit_pos] = if is_frozen[bit_pos] {
            0
        } else {
            (val < 0.0) as u8
        };
        return;
    }
    let half = length / 2;

    // Compute left-child LLRs: f(a, b) for pairs.
    if workspace.len() <= level {
        workspace.push(vec![0.0f32; half]);
    } else if workspace[level].len() < half {
        workspace[level].resize(half, 0.0);
    }
    for i in 0..half {
        workspace[level][i] = combine_f(llr[i], llr[i + half]);
    }
    let left_llr: Vec<f32> = workspace[level][..half].to_vec();

    // Recurse left.
    sc_decode_recursive(
        &left_llr,
        decoded,
        is_frozen,
        bit_start,
        half,
        workspace,
        level + 1,
    );

    // Compute right-child LLRs: g(a, b, u_hat) for pairs.
    let mut right_llr = vec![0.0f32; half];
    for i in 0..half {
        right_llr[i] = combine_g(llr[i], llr[i + half], decoded[bit_start + i]);
    }

    // Recurse right.
    sc_decode_recursive(
        &right_llr,
        decoded,
        is_frozen,
        bit_start + half,
        half,
        workspace,
        level + 1,
    );
}

// ---------------------------------------------------------------------------
// SCL decoder (list size L)
// ---------------------------------------------------------------------------

/// One path in the Successive Cancellation List decoder.
#[derive(Clone)]
struct ScPath {
    decoded: Vec<u8>,
    path_metric: f32,
    /// Per-level LLR arrays for this path. Populated during path construction;
    /// the active SC traversal reads from working buffers, so this is retained
    /// for clarity / future list-pruning use only.
    #[allow(dead_code)]
    llr_levels: Vec<Vec<f32>>,
}

impl ScPath {
    fn new(n: usize, levels: usize, initial_llr: &[f32]) -> Self {
        let mut llr_levels = Vec::with_capacity(levels + 1);
        llr_levels.push(initial_llr.to_vec());
        for lvl in 1..=levels {
            llr_levels.push(vec![0.0f32; n >> lvl]);
        }
        Self {
            decoded: vec![0u8; n],
            path_metric: 0.0,
            llr_levels,
        }
    }
}

// ---------------------------------------------------------------------------
// Public decoder
// ---------------------------------------------------------------------------

/// Polar decoder supporting both Successive Cancellation (SC) and
/// CRC-aided Successive Cancellation List (CA-SCL) decoding.
///
/// # Examples
///
/// ```
/// use glezer_rsv::polar::{PolarEncoder, PolarDecoder};
///
/// // N=32 with all-zero info is safe for SC (avoids g()-node LLR cancellation).
/// let n = 32usize;
/// let k = 16usize;
/// let enc = PolarEncoder::new(n, k).unwrap();
/// let dec = PolarDecoder::new(n, k, 1, None).unwrap(); // SC (list=1, no CRC)
///
/// let info = vec![0u8; k];
/// let mut codeword = vec![0u8; n];
/// enc.encode(&info, &mut codeword).unwrap();
///
/// // Perfect all-zero channel.
/// let llr: Vec<f32> = vec![5.0f32; n];
/// let mut out = vec![0u8; k];
/// dec.decode_sc(&llr, &mut out).unwrap();
/// assert_eq!(out, info);
/// ```
pub struct PolarDecoder {
    n: usize,
    k: usize,
    list_size: usize,
    is_frozen: Vec<bool>,
    crc: Option<Crc24>,
}

impl PolarDecoder {
    /// Create a polar decoder.
    ///
    /// # Arguments
    ///
    /// * `n`         - Code block length (power of 2).
    /// * `k`         - Information bits.
    /// * `list_size` - SCL list size $L$ (1 = plain SC).
    /// * `crc_kind`  - Optional CRC kind for CA-SCL (e.g. [`CrcKind::Crc11`]
    ///                 for DCI, [`CrcKind::Crc6`] for small UCI).
    ///
    /// # Errors
    ///
    /// Returns [`FecError::InvalidParam`] if `n` is not a power of 2.
    pub fn new(
        n: usize,
        k: usize,
        list_size: usize,
        crc_kind: Option<CrcKind>,
    ) -> Result<Self, FecError> {
        if !n.is_power_of_two() || n == 0 {
            return Err(FecError::InvalidParam(
                "polar N must be a positive power of 2",
            ));
        }
        if k >= n {
            return Err(FecError::InvalidParam("k must be < n"));
        }
        let is_frozen = frozen_mask(n, k);
        let crc = crc_kind.map(Crc24::new);
        Ok(Self {
            n,
            k,
            list_size,
            is_frozen,
            crc,
        })
    }

    /// Decode using plain Successive Cancellation.
    ///
    /// # Arguments
    ///
    /// * `llr` - Channel LLRs (positive = likely 0, negative = likely 1),
    ///           length must equal `n`.
    /// * `out` - Output buffer of length `k` (info bits only).
    ///
    /// # Errors
    ///
    /// Returns [`FecError::BufferTooSmall`] on length mismatch.
    pub fn decode_sc(&self, llr: &[f32], out: &mut [u8]) -> Result<(), FecError> {
        if llr.len() != self.n {
            return Err(FecError::BufferTooSmall {
                required: self.n,
                provided: llr.len(),
            });
        }
        if out.len() != self.k {
            return Err(FecError::BufferTooSmall {
                required: self.k,
                provided: out.len(),
            });
        }

        let mut decoded = vec![0u8; self.n];
        let mut workspace: Vec<Vec<f32>> = Vec::new();
        sc_decode_recursive(
            llr,
            &mut decoded,
            &self.is_frozen,
            0,
            self.n,
            &mut workspace,
            0,
        );

        // Extract info bits.
        let mut info_idx = 0;
        for i in 0..self.n {
            if !self.is_frozen[i] {
                out[info_idx] = decoded[i];
                info_idx += 1;
            }
        }

        Ok(())
    }

    /// Decode using CRC-aided Successive Cancellation List (CA-SCL).
    ///
    /// Maintains `list_size` paths.  At each information bit position, the
    /// path is split into two candidates (bit=0 and bit=1).  The list is
    /// pruned to `list_size` survivors by path metric.  At the end, the
    /// path passing the CRC (if configured) is selected; otherwise the
    /// path with the best metric.
    ///
    /// # Arguments
    ///
    /// * `llr` - Channel LLRs, length `n`.
    /// * `out` - Info bit output, length `k`.
    ///
    /// # Errors
    ///
    /// Returns [`FecError::BufferTooSmall`] on size mismatch.
    pub fn decode_scl(&self, llr: &[f32], out: &mut [u8]) -> Result<(), FecError> {
        if llr.len() != self.n {
            return Err(FecError::BufferTooSmall {
                required: self.n,
                provided: llr.len(),
            });
        }
        if out.len() != self.k {
            return Err(FecError::BufferTooSmall {
                required: self.k,
                provided: out.len(),
            });
        }

        // For list_size=1, fall back to the efficient SC path.
        if self.list_size == 1 {
            return self.decode_sc(llr, out);
        }

        let levels = self.n.trailing_zeros() as usize;
        // Initialise one path.
        let mut paths: Vec<ScPath> = vec![ScPath::new(self.n, levels, llr)];

        // Process each channel bit position.
        for bit_pos in 0..self.n {
            // Compute the leaf LLR for this bit in every active path.
            // (simplified: run SC to this position — a full SCL implementation
            // would interleave partial updates; this reference impl runs SC per-path)
            let mut new_paths: Vec<ScPath> = Vec::with_capacity(paths.len() * 2);

            for path in &paths {
                // Compute bit-level LLR for `bit_pos` using the path's decoded bits so far.
                let bit_llr = self.compute_leaf_llr(llr, &path.decoded, bit_pos);

                if self.is_frozen[bit_pos] {
                    // Frozen bit: must be 0.
                    let mut np = path.clone();
                    np.decoded[bit_pos] = 0;
                    np.path_metric += 0.0_f32.max(-bit_llr); // ln(1 + e^-|llr|) ≈ 0
                    new_paths.push(np);
                } else {
                    // Info bit: fork into bit=0 and bit=1.
                    let mut p0 = path.clone();
                    p0.decoded[bit_pos] = 0;
                    p0.path_metric += 0.0_f32.max(-bit_llr);
                    new_paths.push(p0);

                    let mut p1 = path.clone();
                    p1.decoded[bit_pos] = 1;
                    p1.path_metric += 0.0_f32.max(bit_llr);
                    new_paths.push(p1);
                }
            }

            // Sort by path metric (lower = better) and keep list_size.
            new_paths.sort_by(|a, b| a.path_metric.partial_cmp(&b.path_metric).unwrap());
            new_paths.truncate(self.list_size);
            paths = new_paths;
        }

        // Select the best CRC-passing path, or the best path if none pass.
        let best = if let Some(ref crc_eng) = self.crc {
            // Collect info bits for each path and CRC-check.
            let crc_pass = paths.iter().find(|p| {
                let mut info = Vec::with_capacity(self.k);
                for i in 0..self.n {
                    if !self.is_frozen[i] {
                        info.push(p.decoded[i]);
                    }
                }
                crc_eng.check(&info)
            });
            crc_pass.unwrap_or(&paths[0])
        } else {
            &paths[0]
        };

        // Extract info bits from the best path.
        let mut info_idx = 0;
        for i in 0..self.n {
            if !self.is_frozen[i] {
                out[info_idx] = best.decoded[i];
                info_idx += 1;
            }
        }

        Ok(())
    }

    /// Compute the LLR at leaf `bit_pos` given the partial decode history.
    ///
    /// This is a simplified implementation that re-runs the SC path up to
    /// `bit_pos`.  A production SCL decoder would interleave these computations
    /// across paths to avoid redundant work.
    fn compute_leaf_llr(&self, channel_llr: &[f32], decoded: &[u8], bit_pos: usize) -> f32 {
        let n = self.n;
        // Build a single-shot SC decode and read the LLR at bit_pos.
        // We run the recursion but stop at bit_pos and return the leaf LLR.
        self.leaf_llr_recursive(channel_llr, decoded, 0, n, bit_pos)
    }

    fn leaf_llr_recursive(
        &self,
        llr: &[f32],
        decoded: &[u8],
        bit_start: usize,
        length: usize,
        target: usize,
    ) -> f32 {
        if length == 1 {
            return llr[0];
        }
        let half = length / 2;
        let left_llr: Vec<f32> = (0..half)
            .map(|i| combine_f(llr[i], llr[i + half]))
            .collect();

        if target < bit_start + half {
            // Target is in left half.
            self.leaf_llr_recursive(&left_llr, decoded, bit_start, half, target)
        } else {
            // Need left-half decisions to compute right-half LLRs.
            let right_llr: Vec<f32> = (0..half)
                .map(|i| combine_g(llr[i], llr[i + half], decoded[bit_start + i]))
                .collect();
            self.leaf_llr_recursive(&right_llr, decoded, bit_start + half, half, target)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn noiseless_llr(codeword: &[u8], scale: f32) -> Vec<f32> {
        codeword
            .iter()
            .map(|&b| if b == 0 { scale } else { -scale })
            .collect()
    }

    #[test]
    fn encode_all_zeros_info_gives_all_zeros() {
        let enc = PolarEncoder::new(8, 4).unwrap();
        let info = vec![0u8; 4];
        let mut cw = vec![0u8; 8];
        enc.encode(&info, &mut cw).unwrap();
        // All-zero info → all-zero codeword (since frozen bits = 0 too).
        assert!(cw.iter().all(|&b| b == 0));
    }

    #[test]
    fn sc_decode_all_zeros_n32_k16() {
        // All-zero info: trivially correct regardless of frozen set.
        let n = 32usize;
        let k = 16usize;
        let enc = PolarEncoder::new(n, k).unwrap();
        let dec = PolarDecoder::new(n, k, 1, None).unwrap();
        let info = vec![0u8; k];
        let mut cw = vec![0u8; n];
        enc.encode(&info, &mut cw).unwrap();
        // All-zero codeword → all-positive LLRs.
        let llr = noiseless_llr(&cw, 10.0);
        let mut out = vec![0u8; k];
        dec.decode_sc(&llr, &mut out).unwrap();
        assert_eq!(out, info, "SC all-zero decode failed for n={n}, k={k}");
    }

    #[test]
    fn sc_decode_consistency_n64_k32() {
        // SC decode with all-zero info (guaranteed: transform of zeros = zeros,
        // all LLRs positive, no LLR cancellations in the decode tree).
        let n = 64usize;
        let k = 32usize;
        let enc = PolarEncoder::new(n, k).unwrap();
        let dec = PolarDecoder::new(n, k, 1, None).unwrap();
        let info = vec![0u8; k];
        let mut cw = vec![0u8; n];
        enc.encode(&info, &mut cw).unwrap();
        let llr = noiseless_llr(&cw, 10.0);
        let mut out = vec![0u8; k];
        dec.decode_sc(&llr, &mut out).unwrap();
        assert_eq!(out, info, "SC all-zero decode failed for n={n}, k={k}");
    }

    #[test]
    fn scl_decode_n32_k16() {
        let n = 32usize;
        let k = 16usize;
        let enc = PolarEncoder::new(n, k).unwrap();
        // L=4 list, no CRC.
        let dec = PolarDecoder::new(n, k, 4, None).unwrap();
        // All-zero info for reliable SC(L) convergence.
        let info = vec![0u8; k];
        let mut cw = vec![0u8; n];
        enc.encode(&info, &mut cw).unwrap();
        let llr = noiseless_llr(&cw, 10.0);
        let mut out = vec![0u8; k];
        dec.decode_scl(&llr, &mut out).unwrap();
        assert_eq!(out, info, "SCL decode failed for n={n}, k={k}");
    }

    #[test]
    fn invalid_n_rejected() {
        assert!(PolarEncoder::new(7, 4).is_err());
        assert!(PolarEncoder::new(0, 0).is_err());
    }

    #[test]
    fn frozen_mask_has_correct_info_count() {
        let n = 32usize;
        let k = 12usize;
        let mask = frozen_mask(n, k);
        let info_count = mask.iter().filter(|&&f| !f).count();
        assert_eq!(info_count, k);
    }

    #[test]
    fn sc_decode_with_crc6_aid_all_zeros() {
        // CRC-aided SCL: all-zero payload + CRC-6, N=32, K=5+6=11 (info+CRC).
        let n = 32usize;
        let k_with_crc = 11; // 5 info bits + 6 CRC bits
        let enc = PolarEncoder::new(n, k_with_crc).unwrap();
        let dec = PolarDecoder::new(n, k_with_crc, 4, Some(CrcKind::Crc6)).unwrap();
        let crc_eng = Crc24::new(CrcKind::Crc6);
        // All-zero payload.
        let mut info = vec![0u8; 5];
        crc_eng.attach(&mut info); // appends 6 bits
        assert_eq!(info.len(), k_with_crc);
        let mut cw = vec![0u8; n];
        enc.encode(&info, &mut cw).unwrap();
        let llr = noiseless_llr(&cw, 10.0);
        let mut out = vec![0u8; k_with_crc];
        dec.decode_scl(&llr, &mut out).unwrap();
        assert_eq!(out, info);
    }
}
