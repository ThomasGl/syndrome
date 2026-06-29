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
    /// Returns `Err` if `z` is not a valid 3GPP lifting size.
    pub fn new(bg: BaseGraph, z: usize) -> Result<Self, &'static str> {
        let ils = ils_for_z(z).ok_or("z is not a valid 3GPP lifting size")?;

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
/// passed in by the caller via [`decode_layered_offset_min_sum`].
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
    /// Returns `Err` if `z` is not a valid 3GPP lifting size.
    pub fn with_lifting_size(
        bg: BaseGraph,
        z: usize,
        offset_beta: f32,
    ) -> Result<Self, &'static str> {
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
    /// filler bit positions, then calls [`decode_layered_offset_min_sum`].
    ///
    /// The first $2Z$ systematic positions are punctured (never transmitted) and
    /// arrive as channel erasures (LLR = 0.0).  Filler bits at positions
    /// $K' .. K$ (where $K = k_b \cdot Z$) are known zeros and must be
    /// initialised to a large positive LLR value before decoding.
    ///
    /// # Arguments
    ///
    /// * `llr`       - Channel LLR buffer of length $N = n_b \cdot Z$.
    ///                 The caller must have filled positions $[2Z .. K']$ and
    ///                 $[K .. N]$ with received channel LLRs, and left
    ///                 positions $[0 .. 2Z]$ at 0.0 (punctured erasure).
    ///                 This function fills $[K' .. K]$ with the filler-bit LLR.
    /// * `n_filler`  - Number of filler bits ($K - K'$).
    /// * `edge_r`    - Caller-owned C→V buffer (length ≥ [`required_edge_buffer()`]).
    /// * `scratch`   - Caller-owned per-layer scratch (length ≥ [`required_layer_buffer()`]).
    /// * `hard`      - Hard-decision output of length $N$.
    /// * `iterations`- Number of layered passes.
    ///
    /// # Errors
    ///
    /// Propagates any error from [`decode_layered_offset_min_sum`].
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
    ) -> Result<usize, &'static str> {
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
    ///                     Length must equal [`variable_node_count()`].
    /// * `edge_r`        - Preallocated flat C→V extrinsic buffer.
    ///                     Minimum length: [`required_edge_buffer()`].
    /// * `layer_scratch` - Per-layer V→C scratch buffer.
    ///                     Minimum length: [`required_layer_buffer()`].
    /// * `hard_output`   - Bit-wise hard-decision output.
    ///                     Length must equal [`variable_node_count()`].
    /// * `iterations`    - Number of full layered passes.
    ///
    /// # Errors
    ///
    /// Returns `Err` if any buffer length is insufficient.
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
    ) -> Result<usize, &'static str> {
        let n = self.variable_node_count();
        if llr.len() != n {
            return Err("llr length mismatch");
        }
        if hard_output.len() != n {
            return Err("hard_output length mismatch");
        }
        let edge_count = self.required_edge_buffer();
        if edge_r.len() < edge_count {
            return Err("edge_r buffer too small");
        }
        let layer_scratch_len = self.required_layer_buffer();
        if layer_scratch.len() < layer_scratch_len {
            return Err("layer_scratch buffer too small");
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
    fn build(params: &QcLdpcParams) -> Result<Self, &'static str> {
        let z = params.z;
        let m_b = params.num_row_blocks;
        let n_b = params.num_col_blocks;
        let k_b = n_b - m_b;
        let m = m_b * z;
        let k = k_b * z;

        let aug_bits = m + k;
        let aug_words = (aug_bits + 63) / 64;
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
            let pivot = pivot.ok_or("parity matrix is singular — cannot encode")?;
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
        let words_per_row = (k + 63) / 64;
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
    #[inline]
    fn apply(&self, systematic: &[u8], parity_out: &mut [u8]) {
        // Pack systematic bits into u64 words for fast inner product.
        let mut sys_packed = vec![0u64; self.words_per_row];
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

/// QC-LDPC systematic encoder for 5G NR BG1/BG2.
///
/// Precomputes a GF(2) parity generator at construction time via Gaussian
/// elimination; the [`encode`] call is then a simple matrix–vector multiply
/// over GF(2).
///
/// The encoder may allocate during construction; it is not intended for
/// the latency-critical decode hot path.
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
    /// Precomputed GF(2) generator: parity[i] = rows[i] · systematic.
    generator: ParityGenerator,
    /// Number of information bits $K = k_b \cdot Z$.
    k_bits: usize,
}

impl QcLdpcEncoder {
    /// Create an encoder for `bg` at lifting size `z`.
    ///
    /// Performs GF(2) Gaussian elimination on the parity portion of the
    /// expanded H matrix to precompute the generator.  This allocates $O(M \cdot (M+K))$
    /// memory and runs in $O(M^2 \cdot N)$ time (one-time cost).
    ///
    /// # Arguments
    ///
    /// * `bg` - Base graph variant.
    /// * `z`  - Lifting size (must be a valid 3GPP value from Table 5.3.2-1).
    ///
    /// # Errors
    ///
    /// Returns `Err` if `z` is not valid, or if the parity matrix is singular.
    pub fn new(bg: BaseGraph, z: usize) -> Result<Self, &'static str> {
        let params = QcLdpcParams::new(bg, z)?;
        let k_blocks = params.num_col_blocks - params.num_row_blocks;
        let k_bits = k_blocks * params.z;
        let generator = ParityGenerator::build(&params)?;
        Ok(Self {
            params,
            generator,
            k_bits,
        })
    }

    /// Number of information bits ($K = k_b \cdot Z$).
    pub fn info_bit_count(&self) -> usize {
        self.k_bits
    }

    /// Total codeword length ($N = n_b \cdot Z$).
    pub fn codeword_bit_count(&self) -> usize {
        self.params.num_col_blocks * self.params.z
    }

    /// 5G NR-compliant encode wrapper (TS 38.212 §5.3.2).
    ///
    /// Accepts $K'$ info bits, pads with $n_{filler}$ zero filler bits to
    /// reach $K = k_b \cdot Z$, then calls [`encode`].  The first $2Z$
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
    /// Returns `Err` if `info_bits.len() != k_bits - n_filler` or the codeword
    /// buffer has the wrong length.
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
    ) -> Result<(), &'static str> {
        let k = self.k_bits;
        let k_prime = k.checked_sub(n_filler).ok_or("n_filler exceeds k_bits")?;
        if info_bits.len() != k_prime {
            return Err("info_bits length must equal k_bits - n_filler");
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
    /// computes the $M$ parity bits via the precomputed GF(2) generator and
    /// writes them into positions $[K..N]$.
    ///
    /// # Arguments
    ///
    /// * `info_bits` - Systematic bits of length [`info_bit_count()`].
    /// * `codeword`  - Output buffer of length [`codeword_bit_count()`].
    ///
    /// # Errors
    ///
    /// Returns `Err` if buffer lengths are incorrect.
    pub fn encode(&self, info_bits: &[u8], codeword: &mut [u8]) -> Result<(), &'static str> {
        let k = self.k_bits;
        let n = self.codeword_bit_count();
        if info_bits.len() != k {
            return Err("info_bits length must equal info_bit_count()");
        }
        if codeword.len() != n {
            return Err("codeword length must equal codeword_bit_count()");
        }
        codeword[..k].copy_from_slice(info_bits);
        self.generator.apply(info_bits, &mut codeword[k..]);
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
}
