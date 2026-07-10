//! QC-LDPC layered offset min-sum decoder and encoder.
//!
//! Implements the Layered Offset Min-Sum (LOMS) algorithm for 5G NR
//! QC-LDPC codes using the real BG1 and BG2 base graphs from
//! 3GPP TS 38.212 Tables 5.3.2-2 and 5.3.2-3.
//!
//! # Lifting size sets (3GPP Table 5.3.2-1)
//!
//! | iLS | Lifting sizes Z                           |
//! |-----|-------------------------------------------|
//! | 0   | 2, 4, 8, 16, 32, 64, 128, 256            |
//! | 1   | 3, 6, 12, 24, 48, 96, 192, 384           |
//! | 2   | 5, 10, 20, 40, 80, 160, 320              |
//! | 3   | 7, 14, 28, 56, 112, 224                  |
//! | 4   | 9, 18, 36, 72, 144, 288                  |
//! | 5   | 11, 22, 44, 88, 176, 352                 |
//! | 6   | 13, 26, 52, 104, 208                     |
//! | 7   | 15, 30, 60, 120, 240                     |

use crate::bg_tables::{
    BG1_COLS, BG1_ENTRIES, BG1_ENTRY_COUNT, BG1_ROWS, BG2_COLS, BG2_ENTRIES, BG2_ENTRY_COUNT,
    BG2_ROWS,
};
use crate::error::FecError;

/// Supported 5G NR QC-LDPC base graph identifiers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BaseGraph {
    Bg1,
    Bg2,
}

// ---------------------------------------------------------------------------
// Lifting size lookup (3GPP TS 38.212 Table 5.3.2-1)
// ---------------------------------------------------------------------------

/// The 8 lifting-size sets indexed by iLS (0..=7).
/// Each inner slice lists the valid Z values in ascending order.
const LIFTING_SETS: [&[usize]; 8] = [
    &[2, 4, 8, 16, 32, 64, 128, 256],
    &[3, 6, 12, 24, 48, 96, 192, 384],
    &[5, 10, 20, 40, 80, 160, 320],
    &[7, 14, 28, 56, 112, 224],
    &[9, 18, 36, 72, 144, 288],
    &[11, 22, 44, 88, 176, 352],
    &[13, 26, 52, 104, 208],
    &[15, 30, 60, 120, 240],
];

/// Return the iLS (set index, 0..=7) for the given lifting size `z`, or
/// `None` if `z` is not a valid 3GPP lifting size.
fn ils_for_z(z: usize) -> Option<usize> {
    for (ils, set) in LIFTING_SETS.iter().enumerate() {
        if set.contains(&z) {
            return Some(ils);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// BG entry lookup helpers
// ---------------------------------------------------------------------------

/// Compute the actual shift value for BG entry index `ei` and lifting size `z`
/// given its iLS.  Returns `(col_block, shift_mod_z)`.
///
/// The raw shift value `v[iLS]` from the spec is taken modulo Z to get the
/// actual cyclic shift.
#[inline(always)]
fn entry_col_shift(entry: &(u8, u8, [i16; 8]), ils: usize, z: usize) -> (usize, usize) {
    let col = entry.1 as usize;
    let raw = entry.2[ils] as usize;
    let shift = raw % z;
    (col, shift)
}

// ---------------------------------------------------------------------------
// Scalar hot-path kernel (allocation-free)
// ---------------------------------------------------------------------------

/// Scalar implementation of the per-z-position layered offset min-sum update.
///
/// This is the allocation-free baseline that MUST remain free of any heap
/// allocation. The SIMD variants (under feature flags) follow the same
/// interface.
///
/// # Arguments
///
/// * `row_degree`       - Number of edges (non-null columns) in this check row.
/// * `z_idx`            - Which z-position within the block is being processed.
/// * `z`                - Lifting size.
/// * `q_row`            - V→C message buffer for this layer `[edge * z + z_idx]`.
/// * `edge_r`           - Flat C→V extrinsic buffer (persistent across iterations).
/// * `layer_begin`      - Global edge offset for the first edge of this layer.
/// * `submatrix_cols`   - Column-block index per global edge.
/// * `submatrix_shifts` - Cyclic shift per global edge.
/// * `llr`              - A-posteriori LLR buffer (updated in-place).
/// * `offset_beta`      - Offset correction factor $\beta$.
// Scalar reference implementation of a single Z-position update, kept as a
// readable companion to the SIMD kernels; not on any active code path yet.
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn process_z_position_scalar(
    row_degree: usize,
    z_idx: usize,
    z: usize,
    q_row: &mut [f32],
    edge_r: &mut [f32],
    layer_begin: usize,
    submatrix_cols: &[usize],
    submatrix_shifts: &[i16],
    llr: &mut [f32],
    offset_beta: f32,
) {
    let mut min1 = f32::INFINITY;
    let mut min2 = f32::INFINITY;
    let mut min1_edge = usize::MAX;
    let mut sign_prod = 1.0f32;

    for edge in 0..row_degree {
        let q = q_row[edge * z + z_idx];
        let abs_q = q.abs();
        let sign = if q.is_sign_negative() { -1.0 } else { 1.0 };
        sign_prod *= sign;

        if abs_q <= min1 {
            min2 = min1;
            min1 = abs_q;
            min1_edge = edge;
        } else if abs_q < min2 {
            min2 = abs_q;
        }
    }

    if min1.is_infinite() {
        min1 = 0.0;
    }
    if min2.is_infinite() {
        min2 = min1;
    }

    for edge in 0..row_degree {
        let q = q_row[edge * z + z_idx];
        let sign = if q.is_sign_negative() { -1.0 } else { 1.0 };
        let min_excluding = if edge == min1_edge { min2 } else { min1 };
        let check_value = (min_excluding - offset_beta).max(0.0);
        let new_r = sign_prod * sign * check_value;

        let base_edge = (layer_begin + edge) * z;
        let prev_r = edge_r[base_edge + z_idx];
        edge_r[base_edge + z_idx] = new_r;

        let col_block = submatrix_cols[layer_begin + edge];
        let shift = submatrix_shifts[layer_begin + edge] as usize;
        let var_idx = col_block * z + ((z_idx + shift) % z);
        llr[var_idx] += new_r - prev_r;
    }
}

// NOTE: SIMD-specialized implementations can be added under the `simd`
// feature. For maintainability we keep scalar path as the canonical
// reference and add intrinsics-based acceleration in separate commits.

#[cfg(feature = "simd")]
use core::simd::Simd;

#[cfg(feature = "simd")]
fn process_z_positions_simd(
    row_degree: usize,
    z_idx: usize,
    z: usize,
    q_row: &mut [f32],
    edge_r: &mut [f32],
    layer_begin: usize,
    submatrix_cols: &[usize],
    submatrix_shifts: &[i16],
    llr: &mut [f32],
    offset_beta: f32,
) {
    type Vf = Simd<f32, 4>;
    let lanes = 4usize;

    let mut min1 = [f32::INFINITY; 4];
    let mut min2 = [f32::INFINITY; 4];
    let mut min1_edge = [usize::MAX; 4];
    let mut sign_prod = Vf::splat(1.0);

    for edge in 0..row_degree {
        let mut vals = [0.0f32; 4];
        for lane in 0..lanes {
            vals[lane] = q_row[edge * z + z_idx + lane];
        }
        let v = Vf::from_array(vals);
        let abs_v = v.abs();
        let sign_v =
            v.simd_lt(Vf::splat(0.0)).to_int().cast::<f32>() * Vf::splat(-2.0) + Vf::splat(1.0);
        sign_prod *= sign_v;

        for lane in 0..lanes {
            let a = abs_v[lane];
            if a <= min1[lane] {
                min2[lane] = min1[lane];
                min1[lane] = a;
                min1_edge[lane] = edge;
            } else if a < min2[lane] {
                min2[lane] = a;
            }
        }
    }

    for lane in 0..lanes {
        if min1[lane].is_infinite() {
            min1[lane] = 0.0;
        }
        if min2[lane].is_infinite() {
            min2[lane] = min1[lane];
        }
    }

    for edge in 0..row_degree {
        let mut arr = [0.0f32; 4];
        for lane in 0..lanes {
            arr[lane] = q_row[edge * z + z_idx + lane];
        }

        for lane in 0..lanes {
            let sign = if arr[lane].is_sign_negative() {
                -1.0
            } else {
                1.0
            };
            let min_excluding = if edge == min1_edge[lane] {
                min2[lane]
            } else {
                min1[lane]
            };
            let check_value = (min_excluding - offset_beta).max(0.0);
            let new_r = sign_prod[lane] * sign * check_value;

            let base_edge = (layer_begin + edge) * z;
            let prev_r = edge_r[base_edge + z_idx + lane];
            edge_r[base_edge + z_idx + lane] = new_r;

            let col_block = submatrix_cols[layer_begin + edge];
            let shift = submatrix_shifts[layer_begin + edge] as usize;
            let var_idx = col_block * z + ((z_idx + lane + shift) % z);
            llr[var_idx] += new_r - prev_r;
        }
    }
}

// ---------------------------------------------------------------------------
// Dynamic parameters (built once at construction time, heap-allocated)
// ---------------------------------------------------------------------------

/// Runtime-computed layout parameters for a QC-LDPC base graph at a specific
/// lifting size.  Heap-allocated once at construction; the decoder hot path
/// only borrows slices into these `Vec`s — no allocation during decode.
#[derive(Clone)]
pub struct QcLdpcParams {
    /// Base graph these parameters were expanded from.
    pub bg: BaseGraph,
    /// Lifting size $Z$.
    pub z: usize,
    /// Number of check-node block rows.
    pub num_row_blocks: usize,
    /// Number of variable-node block columns.
    pub num_col_blocks: usize,
    /// Prefix-sum of per-layer edge counts.  Length = `num_row_blocks + 1`.
    pub layer_offsets: Vec<usize>,
    /// Column-block index per global edge.
    pub submatrix_cols: Vec<usize>,
    /// Cyclic shift per global edge (already reduced modulo Z).
    pub submatrix_shifts: Vec<i16>,
    /// Maximum degree across all layers (used for scratch sizing).
    pub max_layer_degree: usize,
}

impl QcLdpcParams {
    /// Build the runtime parameters for `bg` at lifting size `z`.
    ///
    /// # Errors
    ///
    /// Returns [`FecError::InvalidParam`] if `z` is not a valid 3GPP lifting size.
    pub fn new(bg: BaseGraph, z: usize) -> Result<Self, FecError> {
        let ils =
            ils_for_z(z).ok_or(FecError::InvalidParam("z is not a valid 3GPP lifting size"))?;

        let (num_row_blocks, num_col_blocks, entries, entry_count) = match bg {
            BaseGraph::Bg1 => (BG1_ROWS, BG1_COLS, BG1_ENTRIES.as_ref(), BG1_ENTRY_COUNT),
            BaseGraph::Bg2 => (BG2_ROWS, BG2_COLS, BG2_ENTRIES.as_ref(), BG2_ENTRY_COUNT),
        };

        // Count edges per row block.
        let mut row_degrees = vec![0usize; num_row_blocks];
        for ei in 0..entry_count {
            let r = entries[ei].0 as usize;
            row_degrees[r] += 1;
        }

        // Build layer_offsets (prefix sum).
        let mut layer_offsets = vec![0usize; num_row_blocks + 1];
        for r in 0..num_row_blocks {
            layer_offsets[r + 1] = layer_offsets[r] + row_degrees[r];
        }
        let total_edges = layer_offsets[num_row_blocks];

        // Build submatrix_cols and submatrix_shifts sorted by row then by
        // entry order.  We fill each row's slice in a second pass.
        let mut submatrix_cols = vec![0usize; total_edges];
        let mut submatrix_shifts = vec![0i16; total_edges];
        let mut row_fill = vec![0usize; num_row_blocks];

        for ei in 0..entry_count {
            let r = entries[ei].0 as usize;
            let (col, shift) = entry_col_shift(&entries[ei], ils, z);
            let pos = layer_offsets[r] + row_fill[r];
            submatrix_cols[pos] = col;
            submatrix_shifts[pos] = shift as i16;
            row_fill[r] += 1;
        }

        let max_layer_degree = row_degrees.iter().copied().max().unwrap_or(0);

        Ok(Self {
            bg,
            z,
            num_row_blocks,
            num_col_blocks,
            layer_offsets,
            submatrix_cols,
            submatrix_shifts,
            max_layer_degree,
        })
    }
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

/// QC-LDPC layered offset min-sum decoder.
///
/// The decoder itself holds no per-decode allocation; all working buffers are
/// passed in by the caller via [`QcLdpcDecoder::decode_layered_offset_min_sum`].
///
/// # Examples
///
/// ```
/// use glezer_rsv::qc_ldpc::{BaseGraph, QcLdpcDecoder};
///
/// let decoder = QcLdpcDecoder::new(BaseGraph::Bg1, 0.25);
/// let n = decoder.variable_node_count();
/// let mut llr = vec![0.5f32; n];
/// let mut edge_r = vec![0.0f32; decoder.required_edge_buffer()];
/// let mut scratch = vec![0.0f32; decoder.required_layer_buffer()];
/// let mut hard = vec![0u8; n];
/// decoder.decode_layered_offset_min_sum(&mut llr, &mut edge_r, &mut scratch, &mut hard, 10).unwrap();
/// ```
#[derive(Clone)]
pub struct QcLdpcDecoder {
    params: QcLdpcParams,
    offset_beta: f32,
}

impl QcLdpcDecoder {
    /// Create a decoder with the default lifting size for each base graph.
    ///
    /// Default lifting sizes: BG1 → Z=384 (iLS=1), BG2 → Z=128 (iLS=0).
    /// These are the largest lifting sizes for each graph, giving the largest
    /// code block supported for basic testing.
    ///
    /// # Arguments
    ///
    /// * `base_graph`   - [`BaseGraph::Bg1`] or [`BaseGraph::Bg2`].
    /// * `offset_beta`  - Offset correction $\beta$ for the min-sum update.
    pub fn new(base_graph: BaseGraph, offset_beta: f32) -> Self {
        let default_z = match base_graph {
            BaseGraph::Bg1 => 384,
            BaseGraph::Bg2 => 128,
        };
        Self::with_lifting_size(base_graph, default_z, offset_beta)
            .expect("default lifting sizes are always valid")
    }

    /// Create a decoder with an explicit lifting size.
    ///
    /// # Arguments
    ///
    /// * `bg`           - Base graph variant.
    /// * `z`            - Lifting size (must be a valid 3GPP value from Table 5.3.2-1).
    /// * `offset_beta`  - Offset correction $\beta$.
    ///
    /// # Errors
    ///
    /// Returns [`FecError::InvalidParam`] if `z` is not a valid 3GPP lifting size.
    pub fn with_lifting_size(bg: BaseGraph, z: usize, offset_beta: f32) -> Result<Self, FecError> {
        let params = QcLdpcParams::new(bg, z)?;
        Ok(Self {
            params,
            offset_beta,
        })
    }

    /// Returns the number of variable nodes in the expanded QC-LDPC graph
    /// ($N = n_b \cdot Z$).
    pub fn variable_node_count(&self) -> usize {
        self.params.num_col_blocks * self.params.z
    }

    /// Returns the number of check nodes in the expanded QC-LDPC graph
    /// ($M = m_b \cdot Z$).
    pub fn check_node_count(&self) -> usize {
        self.params.num_row_blocks * self.params.z
    }

    /// Returns the number of edge messages required for the flat
    /// check-to-variable extrinsic buffer.
    pub fn required_edge_buffer(&self) -> usize {
        self.params.submatrix_shifts.len() * self.params.z
    }

    /// Returns the required per-layer temporary scratch buffer size in floats.
    pub fn required_layer_buffer(&self) -> usize {
        self.params.max_layer_degree * self.params.z
    }

    /// 5G NR-compliant decode wrapper (TS 38.212 §5.3.2).
    ///
    /// Initialises LLRs for the two 3GPP-punctured systematic columns and for
    /// filler bit positions, then calls [`QcLdpcDecoder::decode_layered_offset_min_sum`].
    ///
    /// The first $2Z$ systematic positions are punctured (never transmitted) and
    /// arrive as channel erasures (LLR = 0.0).  Filler bits at positions
    /// $K' .. K$ (where $K = k_b \cdot Z$) are known zeros and must be
    /// initialised to a large positive LLR value before decoding.
    ///
    /// # Arguments
    ///
    /// * `llr`       - Channel LLR buffer of length $N = n_b \cdot Z$.
    ///   The caller must have filled positions $[2Z .. K']$ and
    ///   $[K .. N]$ with received channel LLRs, and left
    ///   positions $[0 .. 2Z]$ at 0.0 (punctured erasure).
    ///   This function fills $[K' .. K]$ with the filler-bit LLR.
    /// * `n_filler`  - Number of filler bits ($K - K'$).
    /// * `edge_r`    - Caller-owned C→V buffer (length ≥ [`QcLdpcDecoder::required_edge_buffer`]).
    /// * `scratch`   - Caller-owned per-layer scratch (length ≥ [`QcLdpcDecoder::required_layer_buffer`]).
    /// * `hard`      - Hard-decision output of length $N$.
    /// * `iterations`- Number of layered passes.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`QcLdpcDecoder::decode_layered_offset_min_sum`].
    ///
    /// # Examples
    ///
    /// ```
    /// use glezer_rsv::qc_ldpc::{BaseGraph, QcLdpcDecoder};
    ///
    /// let dec = QcLdpcDecoder::with_lifting_size(BaseGraph::Bg1, 2, 0.25).unwrap();
    /// let n   = dec.variable_node_count();
    /// let k   = dec.info_bit_count_5g();
    /// let mut llr    = vec![5.0f32; n]; // strong all-zero channel
    /// let mut edge_r = vec![0.0f32; dec.required_edge_buffer()];
    /// let mut scratch= vec![0.0f32; dec.required_layer_buffer()];
    /// let mut hard   = vec![0u8; n];
    /// dec.decode_5g(&mut llr, 0, &mut edge_r, &mut scratch, &mut hard, 5).unwrap();
    /// ```
    pub fn decode_5g(
        &self,
        llr: &mut [f32],
        n_filler: usize,
        edge_r: &mut [f32],
        layer_scratch: &mut [f32],
        hard_output: &mut [u8],
        iterations: usize,
    ) -> Result<usize, FecError> {
        let z = self.params.z;
        let k_b = self.params.num_col_blocks - self.params.num_row_blocks;
        let k = k_b * z;

        // Filler bits: positions (k - n_filler)..k are known-zero → very positive LLR.
        // Value of 1e6 is large enough to never be overturned by belief propagation
        // but small enough to avoid f32 overflow in the min-sum accumulator.
        const LLR_FILLER: f32 = 1_000_000.0;
        let k_prime = k.saturating_sub(n_filler);
        for filler_llr in &mut llr[k_prime..k] {
            *filler_llr = LLR_FILLER;
        }

        // Punctured positions: first 2*Z systematic columns arrive as erasures.
        // If the caller already zeroed them (the normal path), this is a no-op;
        // otherwise we enforce the 3GPP convention.
        for punct_llr in &mut llr[..2 * z] {
            *punct_llr = 0.0;
        }

        self.decode_layered_offset_min_sum(llr, edge_r, layer_scratch, hard_output, iterations)
    }

    /// Number of information bits visible to the 5G NR chain: $K = k_b \cdot Z$
    /// (includes any filler bits; subtract `n_filler` for actual info payload).
    pub fn info_bit_count_5g(&self) -> usize {
        let k_b = self.params.num_col_blocks - self.params.num_row_blocks;
        k_b * self.params.z
    }

    /// Decode a block of LLRs using layered offset min-sum with caller-owned
    /// message and scratch buffers.
    ///
    /// No heap allocation occurs inside this function; the hot path is
    /// strictly allocation-free.
    ///
    /// # Arguments
    ///
    /// * `llr`           - Channel LLR input; overwritten with a-posteriori values.
    ///   Length must equal [`QcLdpcDecoder::variable_node_count`].
    /// * `edge_r`        - Preallocated flat C→V extrinsic buffer.
    ///   Minimum length: [`QcLdpcDecoder::required_edge_buffer`].
    /// * `layer_scratch` - Per-layer V→C scratch buffer.
    ///   Minimum length: [`QcLdpcDecoder::required_layer_buffer`].
    /// * `hard_output`   - Bit-wise hard-decision output.
    ///   Length must equal [`QcLdpcDecoder::variable_node_count`].
    /// * `iterations`    - Number of full layered passes.
    ///
    /// # Errors
    ///
    /// Returns [`FecError::InvalidParam`] if `llr` or `hard_output` do not have
    /// exactly length [`QcLdpcDecoder::variable_node_count`]. Returns
    /// [`FecError::BufferTooSmall`] if `edge_r` or `layer_scratch` are smaller
    /// than [`QcLdpcDecoder::required_edge_buffer`] /
    /// [`QcLdpcDecoder::required_layer_buffer`] respectively.
    ///
    /// On success returns the number of iterations actually performed.  When
    /// the syndrome check passes before `iterations` complete, the loop exits
    /// early and the returned count will be less than `iterations`.
    pub fn decode_layered_offset_min_sum(
        &self,
        llr: &mut [f32],
        edge_r: &mut [f32],
        layer_scratch: &mut [f32],
        hard_output: &mut [u8],
        iterations: usize,
    ) -> Result<usize, FecError> {
        let n = self.variable_node_count();
        if llr.len() != n {
            return Err(FecError::InvalidParam(
                "llr length must equal variable_node_count()",
            ));
        }
        if hard_output.len() != n {
            return Err(FecError::InvalidParam(
                "hard_output length must equal variable_node_count()",
            ));
        }
        let edge_count = self.required_edge_buffer();
        if edge_r.len() < edge_count {
            return Err(FecError::BufferTooSmall {
                required: edge_count,
                provided: edge_r.len(),
            });
        }
        let layer_scratch_len = self.required_layer_buffer();
        if layer_scratch.len() < layer_scratch_len {
            return Err(FecError::BufferTooSmall {
                required: layer_scratch_len,
                provided: layer_scratch.len(),
            });
        }

        // Initialize extrinsic messages to zero once before the first iteration.
        for r in edge_r.iter_mut().take(edge_count) {
            *r = 0.0;
        }

        let z = self.params.z;
        let offset_beta = self.offset_beta;

        // Per-z scratch for the layered min-tracking kernel.
        // Z_MAX = 384 is the largest valid 3GPP lifting size; all slices are
        // taken as &mut [..z] so only the first z elements are live.
        // Stack cost: 384 * (4+4+4) = 4.5 KiB — well within thread-stack limits.
        const Z_MAX: usize = 384;
        let mut min1_buf = [f32::MAX; Z_MAX];
        let mut min2_buf = [f32::MAX; Z_MAX];
        let mut sxor_buf = [0u32; Z_MAX];

        // Detect SIMD capability once per call (is_x86_feature_detected! is a
        // single cached atomic read — negligible overhead per call).
        #[cfg(target_arch = "x86_64")]
        let use_avx2 = is_x86_feature_detected!("avx2");
        #[cfg(not(target_arch = "x86_64"))]
        let use_avx2 = false;

        let mut iters_used = 0usize;
        for _ in 0..iterations {
            iters_used += 1;
            for layer in 0..self.params.num_row_blocks {
                let layer_begin = self.params.layer_offsets[layer];
                let layer_end = self.params.layer_offsets[layer + 1];
                let row_degree = layer_end - layer_begin;

                let q_row = &mut layer_scratch[..row_degree * z];
                let min1 = &mut min1_buf[..z];
                let min2 = &mut min2_buf[..z];
                let sxor = &mut sxor_buf[..z];

                // ── Q-build: V→C messages for this layer ─────────────────────
                // When AVX2 is available, q_build_edge_avx2 exploits the fact
                // that (z_idx + shift) % z partitions LLR reads into exactly two
                // contiguous spans, each vectorisable with 8-wide f32 loads.
                // The scalar path uses conditional subtract (avoids % / divide).
                for edge in 0..row_degree {
                    let col_block = self.params.submatrix_cols[layer_begin + edge];
                    let shift = self.params.submatrix_shifts[layer_begin + edge] as usize;
                    let base_edge = (layer_begin + edge) * z;
                    let var_base = col_block * z;
                    let q_base = edge * z;

                    #[cfg(target_arch = "x86_64")]
                    if use_avx2 {
                        unsafe {
                            crate::simd_avx2::q_build_edge_avx2(
                                z, shift, var_base, q_base, base_edge, llr, edge_r, q_row,
                            );
                        }
                        continue;
                    }

                    for z_idx in 0..z {
                        let s = z_idx + shift;
                        let var_idx = var_base + if s >= z { s - z } else { s };
                        q_row[q_base + z_idx] = llr[var_idx] - edge_r[base_edge + z_idx];
                    }
                }

                // ── Passes 1+2: SIMD or scalar ───────────────────────────────
                #[cfg(target_arch = "x86_64")]
                if use_avx2 {
                    unsafe {
                        crate::simd_avx2::decode_layer_passes_avx2(
                            z,
                            row_degree,
                            offset_beta,
                            q_row,
                            edge_r,
                            layer_begin,
                            &self.params.submatrix_cols,
                            &self.params.submatrix_shifts,
                            llr,
                            min1,
                            min2,
                            sxor,
                        );
                    }
                    continue;
                }
                #[cfg(target_arch = "aarch64")]
                {
                    unsafe {
                        crate::simd_neon::decode_layer_passes_neon(
                            z,
                            row_degree,
                            offset_beta,
                            q_row,
                            edge_r,
                            layer_begin,
                            &self.params.submatrix_cols,
                            &self.params.submatrix_shifts,
                            llr,
                            min1,
                            min2,
                            sxor,
                        );
                    }
                    continue;
                }

                // Scalar fallback (non-x86_64/aarch64 or no AVX2 at runtime).
                //
                // Pass 1: accumulate min1, min2, sign-XOR across edges.
                // The inner loop over z is the vectorisation axis (Z=384 trips,
                // independent across z_idx). LLVM emits AVX2 on x86-64 without
                // explicit intrinsics when AVX2 is the compile-time target.
                min1.iter_mut().for_each(|v| *v = f32::MAX);
                min2.iter_mut().for_each(|v| *v = f32::MAX);
                sxor.iter_mut().for_each(|v| *v = 0);

                for edge in 0..row_degree {
                    let q_base = edge * z;
                    for z_idx in 0..z {
                        let bits = q_row[q_base + z_idx].to_bits();
                        let abs_q = f32::from_bits(bits & 0x7FFF_FFFF);
                        sxor[z_idx] ^= bits & 0x8000_0000;
                        let m1 = min1[z_idx];
                        if abs_q <= m1 {
                            min2[z_idx] = m1;
                            min1[z_idx] = abs_q;
                        } else if abs_q < min2[z_idx] {
                            min2[z_idx] = abs_q;
                        }
                    }
                }

                for z_idx in 0..z {
                    if min1[z_idx] == f32::MAX {
                        min1[z_idx] = 0.0;
                    }
                    if min2[z_idx] == f32::MAX {
                        min2[z_idx] = min1[z_idx];
                    }
                }

                // Pass 2: update edge_r and LLR.
                for edge in 0..row_degree {
                    let col_block = self.params.submatrix_cols[layer_begin + edge];
                    let shift = self.params.submatrix_shifts[layer_begin + edge] as usize;
                    let base_edge = (layer_begin + edge) * z;
                    let q_base = edge * z;
                    for z_idx in 0..z {
                        let q_bits = q_row[q_base + z_idx].to_bits();
                        let abs_q = f32::from_bits(q_bits & 0x7FFF_FFFF);
                        let sign_excl_bit = sxor[z_idx] ^ (q_bits & 0x8000_0000);
                        let min_excl = if abs_q == min1[z_idx] {
                            min2[z_idx]
                        } else {
                            min1[z_idx]
                        };
                        let mag = min_excl - offset_beta;
                        let mag = if mag < 0.0 { 0.0 } else { mag };
                        let new_r = f32::from_bits(mag.to_bits() | sign_excl_bit);
                        let old_r = edge_r[base_edge + z_idx];
                        edge_r[base_edge + z_idx] = new_r;
                        let s = z_idx + shift;
                        let var_idx = col_block * z + if s >= z { s - z } else { s };
                        llr[var_idx] += new_r - old_r;
                    }
                }
            }

            // Early termination: all parity checks satisfied → quit.
            if self.check_syndrome_f32(llr) {
                break;
            }
        }

        for i in 0..n {
            hard_output[i] = (llr[i] < 0.0) as u8;
        }

        Ok(iters_used)
    }

    /// Check whether every parity equation is satisfied by the current hard
    /// decisions derived from `llr`.
    ///
    /// Returns `true` iff all parity checks pass (zero syndrome).  Terminates
    /// at the first failed check to keep the overhead low in the common
    /// (converged) case.
    ///
    /// No heap allocation; runs in $O(\text{total\_edges} \cdot Z)$.
    fn check_syndrome_f32(&self, llr: &[f32]) -> bool {
        let z = self.params.z;
        for layer in 0..self.params.num_row_blocks {
            let layer_begin = self.params.layer_offsets[layer];
            let layer_end = self.params.layer_offsets[layer + 1];
            let row_degree = layer_end - layer_begin;
            for z_idx in 0..z {
                let mut parity = 0u8;
                for edge in 0..row_degree {
                    let col = self.params.submatrix_cols[layer_begin + edge];
                    let shift = self.params.submatrix_shifts[layer_begin + edge] as usize;
                    let s = z_idx + shift;
                    let var = col * z + if s >= z { s - z } else { s };
                    parity ^= (llr[var] < 0.0) as u8;
                }
                if parity != 0 {
                    return false;
                }
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/// GF(2) row over packed `u64` words.  Bit `i` is stored at word `i/64`, bit
/// position `i % 64` (LSB = bit 0).
type GfRow = Vec<u64>;

/// XOR `src` into `dst` (both the same length in words).
#[inline]
fn gf_row_xor(dst: &mut [u64], src: &[u64]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d ^= s;
    }
}

#[inline]
fn gf_bit_get(row: &[u64], bit: usize) -> bool {
    (row[bit >> 6] >> (bit & 63)) & 1 == 1
}

#[inline]
fn gf_bit_set(row: &mut [u64], bit: usize) {
    row[bit >> 6] ^= 1u64 << (bit & 63);
}

/// GF(2) Gaussian elimination result: a row-reduced generator matrix mapping
/// systematic bits to parity bits.
///
/// Stored as packed `u64` rows of length `ceil(K/64)` words.  Precomputed
/// once at encoder construction; [`apply`] is an inner product over GF(2).
struct ParityGenerator {
    /// Number of parity bits $M = m_b \cdot Z$.
    m: usize,
    /// Number of information bits $K = k_b \cdot Z$. Retained for clarity of the
    /// generator's dimensions; not read on the current code path.
    #[allow(dead_code)]
    k: usize,
    /// `rows[i]` is a packed bit vector of length $\lceil K/64 \rceil$ words.
    /// Parity bit $i$ = popcount(rows\[i\] AND packed\_systematic) mod 2.
    rows: Vec<GfRow>,
    /// Words per row = ceil(K/64).
    words_per_row: usize,
}

impl ParityGenerator {
    /// Build the parity generator from the expanded H matrix using packed GF(2)
    /// Gaussian elimination.
    ///
    /// Augmented matrix layout: `[Hp (M cols) | Hs (K cols)]`, packed into
    /// `ceil((M+K)/64)` words per row.  After elimination on the Hp block, the
    /// Hs block contains the generator.
    fn build(params: &QcLdpcParams) -> Result<Self, FecError> {
        let z = params.z;
        let m_b = params.num_row_blocks;
        let n_b = params.num_col_blocks;
        let k_b = n_b - m_b;
        let m = m_b * z;
        let k = k_b * z;

        let aug_bits = m + k;
        let aug_words = aug_bits.div_ceil(64);
        let mut aug: Vec<GfRow> = vec![vec![0u64; aug_words]; m];

        // Populate the augmented matrix [Hp | Hs].
        for layer in 0..m_b {
            let layer_begin = params.layer_offsets[layer];
            let layer_end = params.layer_offsets[layer + 1];
            for edge in layer_begin..layer_end {
                let col_block = params.submatrix_cols[edge];
                let shift = params.submatrix_shifts[edge] as usize;
                for z_idx in 0..z {
                    let check_row = layer * z + z_idx;
                    let var_col = col_block * z + ((z_idx + shift) % z);
                    // Hp occupies bits 0..m; Hs occupies bits m..m+k.
                    let bit = if var_col < k {
                        m + var_col
                    } else {
                        var_col - k
                    };
                    gf_bit_set(&mut aug[check_row], bit);
                }
            }
        }

        // Packed GF(2) Gaussian elimination on columns 0..m (Hp block).
        for col in 0..m {
            let pivot = (col..m).find(|&r| gf_bit_get(&aug[r], col));
            let pivot = pivot.ok_or(FecError::InvalidParam(
                "parity matrix is singular — cannot encode",
            ))?;
            if pivot != col {
                aug.swap(col, pivot);
            }
            // Eliminate this column from all other rows.
            // Temporarily remove the pivot row to satisfy the borrow checker.
            let pivot_row = aug[col].clone();
            for row in 0..m {
                if row != col && gf_bit_get(&aug[row], col) {
                    gf_row_xor(&mut aug[row], &pivot_row);
                }
            }
        }

        // Extract the Hs block (bits m..m+k) into packed K-bit rows.
        let words_per_row = k.div_ceil(64);
        let rows: Vec<GfRow> = aug
            .into_iter()
            .map(|full_row| {
                // Shift bits m..m+k down to 0..k, packing into words_per_row words.
                let mut gen_row = vec![0u64; words_per_row];
                for bit in 0..k {
                    if gf_bit_get(&full_row, m + bit) {
                        gen_row[bit >> 6] |= 1u64 << (bit & 63);
                    }
                }
                gen_row
            })
            .collect();

        Ok(Self {
            m,
            k,
            rows,
            words_per_row,
        })
    }

    /// Compute `m` parity bits from `k` systematic bits (packed as input).
    ///
    /// Each parity bit = popcount(rows\[i\] AND packed\_sys) mod 2.
    ///
    /// `sys_packed` is a fixed-size stack array (no heap allocation): the
    /// largest $K$ across all valid 3GPP (BG, Z) combinations is
    /// $22 \cdot 384 = 8448$ bits = 132 `u64` words, well under
    /// [`GEN_WORDS_MAX`]. This path is only reached via the
    /// [`EncodeStrategy::Dense`] fallback or the test-only
    /// `encode_dense_reference`, never the default hot path.
    #[inline]
    fn apply(&self, systematic: &[u8], parity_out: &mut [u8]) {
        const GEN_WORDS_MAX: usize = 140;
        debug_assert!(self.words_per_row <= GEN_WORDS_MAX);

        // Pack systematic bits into u64 words for fast inner product.
        let mut sys_packed = [0u64; GEN_WORDS_MAX];
        for (j, &b) in systematic.iter().enumerate() {
            if b & 1 != 0 {
                sys_packed[j >> 6] |= 1u64 << (j & 63);
            }
        }

        for i in 0..self.m {
            let row = &self.rows[i];
            let mut acc = 0u64;
            for w in 0..self.words_per_row {
                acc ^= (row[w] & sys_packed[w]).count_ones() as u64;
            }
            parity_out[i] = (acc & 1) as u8;
        }
    }
}

// ---------------------------------------------------------------------------
// Sparse structured encoding (3GPP TS 38.212 §5.3.2)
// ---------------------------------------------------------------------------
//
// 5G NR base graphs are structured as H = [A | B | I_ext]: `k_b` systematic
// column-blocks, then 4 "core" parity column-blocks (p1..p4) forming a
// double-diagonal over base rows 0..3, then an identity extension over the
// remaining `m_b - 4` rows (row `i` connects, among the parity columns, only
// to its own diagonal column `k_b + i`, with shift 0).
//
// Reading the actual table entries (not folklore) for both BG1 and BG2
// confirms the following invariant structure for the core 4x4 block, using
// column offsets 0..3 for p1..p4:
//
//   row0: { p1 (shift s_edge), p2 (shift s0_p2) }
//   row3: { p1 (shift s_edge — same value as row0), p4 (shift s3_p4) }
//   {row1, row2}: one of them additionally carries a *second*, generally
//     different, shift for p1 (call that row `row_x`, shift `s_x`); the other
//     ("row_y") does not touch p1 at all. Both always carry p3 between them;
//     row1 always also carries p2, row2 always also carries p4.
//
// XOR-summing all four core-row parity-check equations cancels every term
// that appears in exactly two rows with an *equal* shift: p1's row0/row3
// pair cancels (equal shift), p2's row0/row1 pair cancels, p3's row1/row2
// pair cancels, p4's row2/row3 pair cancels — leaving exactly
// `rotate(p1, s_x)` on the left. That pins down p1 in closed form; p2, p3,
// p4 then follow by direct back-substitution through rows 0, 1, 2 (row 3 is
// the redundant/consistency equation, checked only in debug builds).
//
// Extension rows (i >= 4) are a direct identity: `p_i = lambda_i XOR (that
// row's edges into p1..p4, already known at this point)`.

/// Z ≤ 384 is the largest valid 3GPP lifting size (Table 5.3.2-1); every
/// scratch block used by the sparse encoder is a fixed `[u8; Z_MAX]` stack
/// array sized for the worst case, never a heap allocation.
const Z_MAX: usize = 384;

/// `dst[i] ^= src[(i + shift) mod z]` for `i in 0..z`, without a per-element
/// modulo (the wraparound is exactly two contiguous spans).
#[inline]
fn rotl_xor_into(dst: &mut [u8], src: &[u8], shift: usize, z: usize) {
    if shift == 0 {
        for i in 0..z {
            dst[i] ^= src[i];
        }
    } else {
        let head = z - shift;
        for i in 0..head {
            dst[i] ^= src[i + shift];
        }
        for i in head..z {
            dst[i] ^= src[i - head];
        }
    }
}

/// `dst[i] = src[(i + shift) mod z]` for `i in 0..z` (assignment, not XOR).
#[inline]
fn rotl_assign(dst: &mut [u8], src: &[u8], shift: usize, z: usize) {
    if shift == 0 {
        dst[..z].copy_from_slice(&src[..z]);
    } else {
        let head = z - shift;
        dst[..head].copy_from_slice(&src[shift..shift + head]);
        dst[head..z].copy_from_slice(&src[..shift]);
    }
}

/// Accumulate $\lambda_{\text{row}} = \bigoplus_j \mathrm{rotate}(\text{info}_j, \text{shift}_j)$
/// over the info-column edges (`col < k_b`) of base row `row` into `lambda`
/// (which the caller must have zeroed). Reads the systematic section of
/// `codeword` (`codeword[..k_b*z]`), which must already be populated.
#[inline]
fn accumulate_lambda(
    params: &QcLdpcParams,
    row: usize,
    k_b: usize,
    z: usize,
    codeword: &[u8],
    lambda: &mut [u8],
) {
    let begin = params.layer_offsets[row];
    let end = params.layer_offsets[row + 1];
    for e in begin..end {
        let col = params.submatrix_cols[e];
        if col < k_b {
            let shift = params.submatrix_shifts[e] as usize;
            let src = &codeword[col * z..col * z + z];
            rotl_xor_into(&mut lambda[..z], src, shift, z);
        }
    }
}

/// Collect `(column_offset, shift)` pairs for entries of base `row` whose
/// column falls in the 4-wide core-parity block `[k_b, k_b + 4)`. Returns
/// the filled prefix and its length; capped at 4 slots defensively (a valid
/// base graph never has more than 3 such entries in any one core row).
fn core_entries(params: &QcLdpcParams, row: usize, k_b: usize) -> ([(usize, usize); 4], usize) {
    let mut out = [(0usize, 0usize); 4];
    let mut n = 0usize;
    let begin = params.layer_offsets[row];
    let end = params.layer_offsets[row + 1];
    for e in begin..end {
        let col = params.submatrix_cols[e];
        if col >= k_b && col < k_b + 4 && n < 4 {
            out[n] = (col - k_b, params.submatrix_shifts[e] as usize);
            n += 1;
        }
    }
    (out, n)
}

/// Look up the shift for column-offset `offset` within a row's core-entry
/// list, or `None` if that column is not connected in this row.
fn shift_for(entries: &[(usize, usize)], offset: usize) -> Option<usize> {
    entries.iter().find(|&&(o, _)| o == offset).map(|&(_, s)| s)
}

/// Derived 4x4 core double-diagonal solve structure for one `(BaseGraph, Z)`
/// combination. Computed once at [`QcLdpcEncoder::new`] time; the fields are
/// the shifts and row-role bit needed by the closed-form back-substitution
/// described above.
#[derive(Clone, Copy, Debug)]
struct CoreLayout {
    /// Shift shared by p1's appearances in row0 and row3.
    shift_p1_edge: usize,
    /// Shift of p1's extra appearance in `row_x`.
    shift_p1_x: usize,
    /// `true` if `row_x == 1` (row1 carries the extra p1 edge), `false` if
    /// `row_x == 2`.
    row_x_is_1: bool,
    /// Shift of p2 in row0.
    shift0_p2: usize,
    /// Shift of p2 in row1.
    shift1_p2: usize,
    /// Shift of p3 in row1.
    shift1_p3: usize,
    /// Shift of p3 in row2.
    shift2_p3: usize,
    /// Shift of p4 in row2.
    shift2_p4: usize,
    /// Shift of p4 in row3 (only used by the debug-only row-3 consistency
    /// check; row 3 is redundant for the forward solve). Unread in release
    /// builds, where that check compiles out — hence the targeted allow.
    #[cfg_attr(not(debug_assertions), allow(dead_code))]
    shift3_p4: usize,
}

impl CoreLayout {
    /// Derive the core solve structure from the actual base-graph table
    /// entries for `params` (with `k_b` systematic column-blocks).
    ///
    /// Returns `None` if the entries do not match the double-diagonal
    /// pattern the closed-form solve below relies on — the caller must then
    /// fall back to the dense generator for this `(bg, z)` combination.
    fn derive(params: &QcLdpcParams, k_b: usize) -> Option<Self> {
        let m_b = params.num_row_blocks;
        if m_b < 4 || params.num_col_blocks < k_b + 4 {
            return None;
        }

        let (row0, n0) = core_entries(params, 0, k_b);
        let (row1, n1) = core_entries(params, 1, k_b);
        let (row2, n2) = core_entries(params, 2, k_b);
        let (row3, n3) = core_entries(params, 3, k_b);
        let row0 = &row0[..n0];
        let row1 = &row1[..n1];
        let row2 = &row2[..n2];
        let row3 = &row3[..n3];

        // Row 0 must connect exactly {p1, p2}; row 3 exactly {p1, p4}, with
        // p1 carrying the same shift in both (the pair that self-cancels
        // when the four core-row equations are XOR-summed).
        if n0 != 2 || n3 != 2 {
            return None;
        }
        let shift0_p1 = shift_for(row0, 0)?;
        let shift0_p2 = shift_for(row0, 1)?;
        let shift3_p1 = shift_for(row3, 0)?;
        let shift3_p4 = shift_for(row3, 3)?;
        if shift0_p1 != shift3_p1 {
            return None;
        }

        // Exactly one of row1/row2 carries the extra p1 edge that survives
        // the XOR-sum and pins down p1.
        let row1_p1 = shift_for(row1, 0);
        let row2_p1 = shift_for(row2, 0);
        let (row_x_is_1, shift_p1_x) = match (row1_p1, row2_p1) {
            (Some(s), None) => (true, s),
            (None, Some(s)) => (false, s),
            _ => return None,
        };

        let shift1_p2 = shift_for(row1, 1)?;
        let shift1_p3 = shift_for(row1, 2)?;
        let shift2_p3 = shift_for(row2, 2)?;
        let shift2_p4 = shift_for(row2, 3)?;

        let expected_n1 = if row_x_is_1 { 3 } else { 2 };
        let expected_n2 = if row_x_is_1 { 2 } else { 3 };
        if n1 != expected_n1 || n2 != expected_n2 {
            return None;
        }

        // Extension rows (i >= 4) must be a pure identity on parity column
        // k_b + i (shift 0), optionally plus edges back into p1..p4.
        for row in 4..m_b {
            let begin = params.layer_offsets[row];
            let end = params.layer_offsets[row + 1];
            let mut identity_hits = 0usize;
            for e in begin..end {
                let col = params.submatrix_cols[e];
                if col >= k_b + 4 {
                    identity_hits += 1;
                    if col != k_b + row || params.submatrix_shifts[e] != 0 {
                        return None;
                    }
                }
            }
            if identity_hits != 1 {
                return None;
            }
        }

        Some(Self {
            shift_p1_edge: shift0_p1,
            shift_p1_x,
            row_x_is_1,
            shift0_p2,
            shift1_p2,
            shift1_p3,
            shift2_p3,
            shift2_p4,
            shift3_p4,
        })
    }
}

/// Which parity-computation strategy an encoder instance uses.
///
/// [`CoreLayout::derive`] succeeds for every 3GPP (BG, Z) combination in the
/// currently generated tables, so `Dense` is a defensive fallback rather
/// than a path exercised in practice — but it is a genuine, correct
/// fallback (not a stub) should a future/edited base-graph table not match
/// the assumed double-diagonal structure.
enum EncodeStrategy {
    Sparse(CoreLayout),
    Dense(ParityGenerator),
}

/// QC-LDPC systematic encoder for 5G NR BG1/BG2.
///
/// Uses the standard 3GPP structured encoding (TS 38.212 §5.3.2): the 4
/// "core" parity blocks are solved in closed form from the base graph's
/// double-diagonal structure, and the remaining `m_b - 4` parity blocks
/// follow directly from the identity extension — total cost
/// $O(E \cdot Z)$ (E = number of base-graph edges) instead of the dense
/// $O(M \cdot K)$ generator-matrix multiply. See the module-level comment
/// above `CoreLayout` (private to this module) for the derivation.
///
/// `CoreLayout::derive` is attempted once at construction time; it
/// succeeds for every (BG, Z) combination in the current 3GPP tables. If it
/// ever fails for some future/edited table, the encoder falls back to the
/// dense GF(2) generator (Gaussian elimination, precomputed once at
/// construction) so correctness is preserved either way — see
/// [`QcLdpcEncoder::base_graph`]/[`QcLdpcEncoder::lifting_size`] to identify
/// which combination fell back, if any.
///
/// # Examples
///
/// ```
/// use glezer_rsv::qc_ldpc::{BaseGraph, QcLdpcEncoder};
///
/// let enc = QcLdpcEncoder::new(BaseGraph::Bg1, 2).unwrap();
/// let k = enc.info_bit_count();
/// let n = enc.codeword_bit_count();
/// let info = vec![0u8; k];
/// let mut codeword = vec![0u8; n];
/// enc.encode(&info, &mut codeword).unwrap();
/// ```
pub struct QcLdpcEncoder {
    params: QcLdpcParams,
    /// Sparse structured solve (default) or dense GF(2) generator fallback.
    strategy: EncodeStrategy,
    /// Number of information bits $K = k_b \cdot Z$.
    k_bits: usize,
    /// Number of systematic column-blocks $k_b$.
    k_b_blocks: usize,
}

impl QcLdpcEncoder {
    /// Create an encoder for `bg` at lifting size `z`.
    ///
    /// Attempts to derive the sparse structured-encoding layout
    /// (`CoreLayout::derive`) from the base graph's table entries; this is
    /// $O(E)$ and allocation-free. Only if that derivation fails (never the
    /// case for the current 3GPP tables) does construction fall back to
    /// building the dense GF(2) generator via Gaussian elimination, which
    /// allocates $O(M \cdot (M+K))$ memory and runs in $O(M^2 \cdot N)$ time.
    ///
    /// # Arguments
    ///
    /// * `bg` - Base graph variant.
    /// * `z`  - Lifting size (must be a valid 3GPP value from Table 5.3.2-1).
    ///
    /// # Errors
    ///
    /// Returns [`FecError::InvalidParam`] if `z` is not valid, or (dense
    /// fallback only) if the parity matrix is singular.
    pub fn new(bg: BaseGraph, z: usize) -> Result<Self, FecError> {
        let params = QcLdpcParams::new(bg, z)?;
        let k_b_blocks = params.num_col_blocks - params.num_row_blocks;
        let k_bits = k_b_blocks * params.z;
        let strategy = match CoreLayout::derive(&params, k_b_blocks) {
            Some(layout) => EncodeStrategy::Sparse(layout),
            None => EncodeStrategy::Dense(ParityGenerator::build(&params)?),
        };
        Ok(Self {
            params,
            strategy,
            k_bits,
            k_b_blocks,
        })
    }

    /// Returns `true` if this encoder is using the sparse $O(E \cdot Z)$
    /// structured encoding path; `false` if it fell back to the dense
    /// generator (see [`QcLdpcEncoder::new`]).
    pub fn is_sparse(&self) -> bool {
        matches!(self.strategy, EncodeStrategy::Sparse(_))
    }

    /// Number of information bits ($K = k_b \cdot Z$).
    pub fn info_bit_count(&self) -> usize {
        self.k_bits
    }

    /// Total codeword length ($N = n_b \cdot Z$).
    pub fn codeword_bit_count(&self) -> usize {
        self.params.num_col_blocks * self.params.z
    }

    /// Base graph this encoder was constructed for.
    pub fn base_graph(&self) -> BaseGraph {
        self.params.bg
    }

    /// Lifting size $Z$ this encoder was constructed for.
    pub fn lifting_size(&self) -> usize {
        self.params.z
    }

    /// 5G NR-compliant encode wrapper (TS 38.212 §5.3.2).
    ///
    /// Accepts $K'$ info bits, pads with $n_{filler}$ zero filler bits to
    /// reach $K = k_b \cdot Z$, then calls [`QcLdpcEncoder::encode`].  The first $2Z$
    /// systematic bits in the output codeword are **not** transmitted (the rate
    /// matcher handles that puncturing separately).
    ///
    /// # Arguments
    ///
    /// * `info_bits` - Slice of exactly $K' = K - n_{filler}$ information bits.
    /// * `n_filler`  - Number of zero filler bits to append (= $K - K'$).
    /// * `codeword`  - Output buffer of length $N = n_b \cdot Z$.
    ///
    /// # Errors
    ///
    /// Returns [`FecError::InvalidParam`] if `n_filler` exceeds the encoder's
    /// `k_bits`. Returns [`FecError::BufferTooSmall`] if
    /// `info_bits.len() != k_bits - n_filler` or the codeword buffer has the
    /// wrong length.
    ///
    /// # Examples
    ///
    /// ```
    /// use glezer_rsv::qc_ldpc::{BaseGraph, QcLdpcEncoder};
    ///
    /// let enc = QcLdpcEncoder::new(BaseGraph::Bg1, 2).unwrap();
    /// let k_prime = enc.info_bit_count() - 4; // pretend 4 filler bits
    /// let n       = enc.codeword_bit_count();
    /// let info    = vec![0u8; k_prime];
    /// let mut codeword = vec![0u8; n];
    /// enc.encode_5g(&info, 4, &mut codeword).unwrap();
    /// ```
    pub fn encode_5g(
        &self,
        info_bits: &[u8],
        n_filler: usize,
        codeword: &mut [u8],
    ) -> Result<(), FecError> {
        let k = self.k_bits;
        let k_prime = k
            .checked_sub(n_filler)
            .ok_or(FecError::InvalidParam("n_filler exceeds k_bits"))?;
        if info_bits.len() != k_prime {
            return Err(FecError::BufferTooSmall {
                required: k_prime,
                provided: info_bits.len(),
            });
        }
        // Build full-K info buffer with filler zeros at positions k_prime..k.
        // This is a setup path (not the hot loop) so a stack Vec is acceptable.
        let mut full_info = vec![0u8; k];
        full_info[..k_prime].copy_from_slice(info_bits);
        // filler positions [k_prime..k] remain zero.
        self.encode(&full_info, codeword)
    }

    /// Encode systematic `info_bits` into `codeword`.
    ///
    /// Copies the $K$ systematic bits into codeword positions $[0..K]$, then
    /// computes the $M$ parity bits and writes them into positions
    /// $[K..N]$. Uses the sparse $O(E \cdot Z)$ structured solve by default
    /// (see `CoreLayout`); falls back to the dense GF(2) generator only if
    /// [`QcLdpcEncoder::is_sparse`] is `false` for this instance.
    ///
    /// No heap allocation occurs on the sparse path: all scratch is fixed-size
    /// `[u8; Z_MAX]` stack arrays (Z ≤ 384, the largest valid 3GPP lifting
    /// size).
    ///
    /// # Arguments
    ///
    /// * `info_bits` - Systematic bits of length [`QcLdpcEncoder::info_bit_count`].
    /// * `codeword`  - Output buffer of length [`QcLdpcEncoder::codeword_bit_count`].
    ///
    /// # Errors
    ///
    /// Returns [`FecError::BufferTooSmall`] if `info_bits.len() !=
    /// info_bit_count()` or `codeword.len() != codeword_bit_count()`.
    pub fn encode(&self, info_bits: &[u8], codeword: &mut [u8]) -> Result<(), FecError> {
        let k = self.k_bits;
        let n = self.codeword_bit_count();
        if info_bits.len() != k {
            return Err(FecError::BufferTooSmall {
                required: k,
                provided: info_bits.len(),
            });
        }
        if codeword.len() != n {
            return Err(FecError::BufferTooSmall {
                required: n,
                provided: codeword.len(),
            });
        }
        codeword[..k].copy_from_slice(info_bits);
        match &self.strategy {
            EncodeStrategy::Sparse(layout) => self.encode_sparse(layout, codeword),
            EncodeStrategy::Dense(generator) => generator.apply(info_bits, &mut codeword[k..]),
        }
        Ok(())
    }

    /// Sparse structured parity solve (the default `encode` path). See the
    /// module-level comment above [`CoreLayout`] for the derivation.
    ///
    /// Requires `codeword[..k_bits]` to already hold the systematic bits;
    /// writes the $M$ parity bits into `codeword[k_bits..]`.
    fn encode_sparse(&self, layout: &CoreLayout, codeword: &mut [u8]) {
        let z = self.params.z;
        let k_b = self.k_b_blocks;
        let k = self.k_bits;
        let m_b = self.params.num_row_blocks;

        // λ_i for each of the 4 core rows: XOR of rotate(info_j, shift) over
        // that row's info-column edges.
        let mut lambda0 = [0u8; Z_MAX];
        let mut lambda1 = [0u8; Z_MAX];
        let mut lambda2 = [0u8; Z_MAX];
        let mut lambda3 = [0u8; Z_MAX];
        accumulate_lambda(&self.params, 0, k_b, z, codeword, &mut lambda0[..z]);
        accumulate_lambda(&self.params, 1, k_b, z, codeword, &mut lambda1[..z]);
        accumulate_lambda(&self.params, 2, k_b, z, codeword, &mut lambda2[..z]);
        accumulate_lambda(&self.params, 3, k_b, z, codeword, &mut lambda3[..z]);

        let inv = |s: usize| (z - s) % z;

        // p1 = rotate(λ0 ^ λ1 ^ λ2 ^ λ3, inv(shift_p1_x)).
        let mut p1 = [0u8; Z_MAX];
        p1[..z].copy_from_slice(&lambda0[..z]);
        for i in 0..z {
            p1[i] ^= lambda1[i];
        }
        for i in 0..z {
            p1[i] ^= lambda2[i];
        }
        for i in 0..z {
            p1[i] ^= lambda3[i];
        }
        let mut tmp = [0u8; Z_MAX];
        tmp[..z].copy_from_slice(&p1[..z]);
        rotl_assign(&mut p1[..z], &tmp[..z], inv(layout.shift_p1_x), z);

        // p2 = rotate(λ0 ^ rotate(p1, shift_p1_edge), inv(shift0_p2)).
        let mut p2 = [0u8; Z_MAX];
        p2[..z].copy_from_slice(&lambda0[..z]);
        rotl_xor_into(&mut p2[..z], &p1[..z], layout.shift_p1_edge, z);
        tmp[..z].copy_from_slice(&p2[..z]);
        rotl_assign(&mut p2[..z], &tmp[..z], inv(layout.shift0_p2), z);

        // p3 = rotate(λ1 ^ rotate(p2, shift1_p2) [^ rotate(p1, shift_p1_x) if
        // row_x == 1], inv(shift1_p3)).
        let mut p3 = [0u8; Z_MAX];
        p3[..z].copy_from_slice(&lambda1[..z]);
        rotl_xor_into(&mut p3[..z], &p2[..z], layout.shift1_p2, z);
        if layout.row_x_is_1 {
            rotl_xor_into(&mut p3[..z], &p1[..z], layout.shift_p1_x, z);
        }
        tmp[..z].copy_from_slice(&p3[..z]);
        rotl_assign(&mut p3[..z], &tmp[..z], inv(layout.shift1_p3), z);

        // p4 = rotate(λ2 ^ rotate(p3, shift2_p3) [^ rotate(p1, shift_p1_x) if
        // row_x == 2], inv(shift2_p4)).
        let mut p4 = [0u8; Z_MAX];
        p4[..z].copy_from_slice(&lambda2[..z]);
        rotl_xor_into(&mut p4[..z], &p3[..z], layout.shift2_p3, z);
        if !layout.row_x_is_1 {
            rotl_xor_into(&mut p4[..z], &p1[..z], layout.shift_p1_x, z);
        }
        tmp[..z].copy_from_slice(&p4[..z]);
        rotl_assign(&mut p4[..z], &tmp[..z], inv(layout.shift2_p4), z);

        // Row 3 is the redundant core equation; checking it is a cheap O(Z)
        // self-validation of the solve above (debug builds only — no cost
        // in release, and the mandatory tests validate every (bg, z) case
        // via the full syndrome check regardless).
        #[cfg(debug_assertions)]
        {
            let mut check = [0u8; Z_MAX];
            check[..z].copy_from_slice(&lambda3[..z]);
            rotl_xor_into(&mut check[..z], &p1[..z], layout.shift_p1_edge, z);
            rotl_xor_into(&mut check[..z], &p4[..z], layout.shift3_p4, z);
            debug_assert!(
                check[..z].iter().all(|&b| b == 0),
                "sparse QC-LDPC core solve failed its row-3 consistency check"
            );
        }

        codeword[k..k + z].copy_from_slice(&p1[..z]);
        codeword[k + z..k + 2 * z].copy_from_slice(&p2[..z]);
        codeword[k + 2 * z..k + 3 * z].copy_from_slice(&p3[..z]);
        codeword[k + 3 * z..k + 4 * z].copy_from_slice(&p4[..z]);

        // Extension rows (i >= 4): direct identity, p_i = λ_i XOR (this
        // row's edges into the already-known p1..p4 blocks, if any).
        let core = [&p1[..z], &p2[..z], &p3[..z], &p4[..z]];
        let mut lambda_ext = [0u8; Z_MAX];
        for row in 4..m_b {
            lambda_ext[..z].fill(0);
            let begin = self.params.layer_offsets[row];
            let end = self.params.layer_offsets[row + 1];
            for e in begin..end {
                let col = self.params.submatrix_cols[e];
                let shift = self.params.submatrix_shifts[e] as usize;
                if col < k_b {
                    let src = &codeword[col * z..col * z + z];
                    rotl_xor_into(&mut lambda_ext[..z], src, shift, z);
                } else if col < k_b + 4 {
                    rotl_xor_into(&mut lambda_ext[..z], core[col - k_b], shift, z);
                }
                // col >= k_b + 4 is this row's own identity output column;
                // it is the unknown being solved for, not a summand.
            }
            codeword[k + row * z..k + (row + 1) * z].copy_from_slice(&lambda_ext[..z]);
        }
    }

    /// Dense GF(2) reference encode, kept for equivalence testing against
    /// the sparse path. Builds a fresh [`ParityGenerator`] on every call
    /// (Gaussian elimination is a one-time, non-hot-path cost); this is
    /// intentionally decoupled from `self.strategy` so it exercises the
    /// dense math regardless of which strategy a given `(bg, z)` selected.
    #[cfg(test)]
    fn encode_dense_reference(
        &self,
        info_bits: &[u8],
        codeword: &mut [u8],
    ) -> Result<(), FecError> {
        let k = self.k_bits;
        let n = self.codeword_bit_count();
        if info_bits.len() != k {
            return Err(FecError::BufferTooSmall {
                required: k,
                provided: info_bits.len(),
            });
        }
        if codeword.len() != n {
            return Err(FecError::BufferTooSmall {
                required: n,
                provided: codeword.len(),
            });
        }
        let generator = ParityGenerator::build(&self.params)?;
        codeword[..k].copy_from_slice(info_bits);
        generator.apply(info_bits, &mut codeword[k..]);
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
    fn ils_lookup() {
        assert_eq!(ils_for_z(2), Some(0));
        assert_eq!(ils_for_z(384), Some(1));
        assert_eq!(ils_for_z(320), Some(2));
        assert_eq!(ils_for_z(224), Some(3));
        assert_eq!(ils_for_z(7), Some(3));
        assert_eq!(ils_for_z(1), None);
        assert_eq!(ils_for_z(383), None);
    }

    #[test]
    fn decoder_buffers_bg1_z384() {
        let dec = QcLdpcDecoder::new(BaseGraph::Bg1, 0.25);
        // BG1: 46 rows, 68 cols, Z=384
        assert_eq!(dec.variable_node_count(), 68 * 384);
        assert_eq!(dec.check_node_count(), 46 * 384);
        assert!(dec.required_edge_buffer() > 0);
        assert!(dec.required_layer_buffer() > 0);
    }

    #[test]
    fn decoder_buffers_bg2_z128() {
        let dec = QcLdpcDecoder::new(BaseGraph::Bg2, 0.25);
        // BG2: 42 rows, 52 cols, Z=128
        assert_eq!(dec.variable_node_count(), 52 * 128);
        assert_eq!(dec.check_node_count(), 42 * 128);
    }

    #[test]
    fn qc_ldpc_decoder_roundtrip() {
        // Smoke test with a small lifting size that is still in the 3GPP set.
        let z = 2usize;
        let dec = QcLdpcDecoder::with_lifting_size(BaseGraph::Bg1, z, 0.25).unwrap();
        let n = dec.variable_node_count();
        let mut llr = vec![0.5f32; n];
        let mut edge_r = vec![0.0f32; dec.required_edge_buffer()];
        let mut scratch = vec![0.0f32; dec.required_layer_buffer()];
        let mut hard = vec![0u8; n];
        dec.decode_layered_offset_min_sum(&mut llr, &mut edge_r, &mut scratch, &mut hard, 2)
            .expect("decoder should run");
        assert!(hard.iter().all(|b| *b == 0 || *b == 1));
    }

    #[test]
    fn syndrome_check_early_termination() {
        // All-zero codeword is a valid codeword for any linear code.
        // Strong positive LLR = confident "0" → syndrome satisfied after ≤2 iterations.
        let z = 2usize;
        let dec = QcLdpcDecoder::with_lifting_size(BaseGraph::Bg2, z, 0.25).unwrap();
        let n = dec.variable_node_count();
        let mut llr = vec![10.0f32; n];
        let mut edge_r = vec![0.0f32; dec.required_edge_buffer()];
        let mut scratch = vec![0.0f32; dec.required_layer_buffer()];
        let mut hard = vec![0u8; n];
        // Allow up to 50 iterations but the syndrome gate must fire long before that.
        let iters = dec
            .decode_layered_offset_min_sum(&mut llr, &mut edge_r, &mut scratch, &mut hard, 50)
            .expect("decoder should succeed");
        assert!(
            iters <= 5,
            "all-zero codeword should converge in ≤5 iterations, used {iters}"
        );
        assert!(
            hard.iter().all(|&b| b == 0),
            "all-zero codeword should decode to all zeros"
        );
    }

    #[test]
    fn encoder_dimensions_bg1_z384() {
        let enc = QcLdpcEncoder::new(BaseGraph::Bg1, 384).unwrap();
        // K = (68-46)*384 = 22*384 = 8448, N = 68*384 = 26112
        assert_eq!(enc.info_bit_count(), 22 * 384);
        assert_eq!(enc.codeword_bit_count(), 68 * 384);
    }

    #[test]
    fn encoder_dimensions_bg2_z128() {
        let enc = QcLdpcEncoder::new(BaseGraph::Bg2, 128).unwrap();
        // K = (52-42)*128 = 10*128 = 1280, N = 52*128 = 6656
        assert_eq!(enc.info_bit_count(), 10 * 128);
        assert_eq!(enc.codeword_bit_count(), 52 * 128);
    }

    #[test]
    fn encoder_smoke_bg1_z2() {
        let z = 2usize;
        let enc = QcLdpcEncoder::new(BaseGraph::Bg1, z).unwrap();
        let k = enc.info_bit_count();
        let n = enc.codeword_bit_count();
        let info = vec![0u8; k];
        let mut codeword = vec![0u8; n];
        enc.encode(&info, &mut codeword)
            .expect("encode should succeed");
        // All-zero info should produce all-zero codeword.
        assert!(codeword.iter().all(|&b| b == 0));
    }

    // -- Sparse structured encoder: equivalence + syndrome validation ------

    /// Tiny deterministic PRNG (xorshift64*) so tests don't need a `rand`
    /// dependency; only used to generate reproducible pseudo-random info
    /// bits for the tests below.
    struct XorShift64(u64);
    impl XorShift64 {
        fn new(seed: u64) -> Self {
            Self(seed ^ 0x9E37_79B9_7F4A_7C15)
        }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn random_bits(&mut self, n: usize) -> Vec<u8> {
            let mut bits = Vec::with_capacity(n);
            let mut word = 0u64;
            let mut avail = 0u32;
            for _ in 0..n {
                if avail == 0 {
                    word = self.next_u64();
                    avail = 64;
                }
                bits.push((word & 1) as u8);
                word >>= 1;
                avail -= 1;
            }
            bits
        }
    }

    /// Check every parity equation directly over hard `0/1` bits — the
    /// bit-per-u8 analogue of [`QcLdpcDecoder::check_syndrome_f32`], used to
    /// validate that a fast-encoded codeword satisfies $H \cdot c = 0$.
    fn codeword_satisfies_all_checks(params: &QcLdpcParams, codeword: &[u8]) -> bool {
        let z = params.z;
        for layer in 0..params.num_row_blocks {
            let begin = params.layer_offsets[layer];
            let end = params.layer_offsets[layer + 1];
            for z_idx in 0..z {
                let mut parity = 0u8;
                for e in begin..end {
                    let col = params.submatrix_cols[e];
                    let shift = params.submatrix_shifts[e] as usize;
                    let s = z_idx + shift;
                    let var = col * z + if s >= z { s - z } else { s };
                    parity ^= codeword[var];
                }
                if parity != 0 {
                    return false;
                }
            }
        }
        true
    }

    /// Valid 3GPP lifting sizes from the requested test matrix, restricted to
    /// those actually valid (all of them are, for both BG1 and BG2, but we
    /// check via `ils_for_z` rather than assume it).
    const CANDIDATE_Z: [usize; 7] = [2, 16, 52, 96, 128, 208, 384];

    fn valid_test_sizes() -> Vec<usize> {
        CANDIDATE_Z
            .iter()
            .copied()
            .filter(|&z| ils_for_z(z).is_some())
            .collect()
    }

    #[test]
    fn sparse_encoder_used_for_all_tested_bg_z_combos() {
        // The sparse structured solve must be the active strategy for every
        // (bg, z) in the test matrix; if CoreLayout::derive ever regresses
        // to a silent dense fallback, fail loudly here rather than let the
        // equivalence test mask it.
        for &z in &valid_test_sizes() {
            for bg in [BaseGraph::Bg1, BaseGraph::Bg2] {
                let enc = QcLdpcEncoder::new(bg, z).unwrap();
                assert!(
                    enc.is_sparse(),
                    "expected sparse strategy for {bg:?} z={z}, got dense fallback"
                );
            }
        }
    }

    #[test]
    fn sparse_matches_dense_reference_bg1() {
        for &z in &valid_test_sizes() {
            let enc = QcLdpcEncoder::new(BaseGraph::Bg1, z).unwrap();
            let k = enc.info_bit_count();
            let n = enc.codeword_bit_count();
            let mut rng = XorShift64::new(0xC0FFEE ^ z as u64);
            let info = rng.random_bits(k);

            let mut sparse_cw = vec![0u8; n];
            let mut dense_cw = vec![0u8; n];
            enc.encode(&info, &mut sparse_cw).unwrap();
            enc.encode_dense_reference(&info, &mut dense_cw).unwrap();

            assert_eq!(
                sparse_cw, dense_cw,
                "BG1 z={z}: sparse and dense encodes diverged"
            );
        }
    }

    #[test]
    fn sparse_matches_dense_reference_bg2() {
        for &z in &valid_test_sizes() {
            let enc = QcLdpcEncoder::new(BaseGraph::Bg2, z).unwrap();
            let k = enc.info_bit_count();
            let n = enc.codeword_bit_count();
            let mut rng = XorShift64::new(0xBADC0DE ^ z as u64);
            let info = rng.random_bits(k);

            let mut sparse_cw = vec![0u8; n];
            let mut dense_cw = vec![0u8; n];
            enc.encode(&info, &mut sparse_cw).unwrap();
            enc.encode_dense_reference(&info, &mut dense_cw).unwrap();

            assert_eq!(
                sparse_cw, dense_cw,
                "BG2 z={z}: sparse and dense encodes diverged"
            );
        }
    }

    #[test]
    fn sparse_encoded_codewords_satisfy_syndrome() {
        for &z in &valid_test_sizes() {
            for bg in [BaseGraph::Bg1, BaseGraph::Bg2] {
                let params = QcLdpcParams::new(bg, z).unwrap();
                let enc = QcLdpcEncoder::new(bg, z).unwrap();
                let k = enc.info_bit_count();
                let n = enc.codeword_bit_count();
                let mut rng = XorShift64::new(0x5EED ^ ((bg as u64) << 32) ^ z as u64);
                let info = rng.random_bits(k);

                let mut codeword = vec![0u8; n];
                enc.encode(&info, &mut codeword).unwrap();

                assert!(
                    codeword_satisfies_all_checks(&params, &codeword),
                    "{bg:?} z={z}: sparse-encoded codeword failed H*c=0"
                );
            }
        }
    }

    #[test]
    fn sparse_all_zero_info_yields_all_zero_codeword() {
        // All-zero is always a valid codeword; also exercises the row_x==2
        // (BG2) and row_x==1 (BG1) branches trivially.
        for &z in &valid_test_sizes() {
            for bg in [BaseGraph::Bg1, BaseGraph::Bg2] {
                let enc = QcLdpcEncoder::new(bg, z).unwrap();
                let k = enc.info_bit_count();
                let n = enc.codeword_bit_count();
                let info = vec![0u8; k];
                let mut codeword = vec![0u8; n];
                enc.encode(&info, &mut codeword).unwrap();
                assert!(
                    codeword.iter().all(|&b| b == 0),
                    "{bg:?} z={z}: all-zero info should yield all-zero codeword"
                );
            }
        }
    }
}
