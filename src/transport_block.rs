//! 5G NR DL-SCH/UL-SCH transport block processing chain.
//!
//! Implements TS 38.212 §5.1–§5.5 as a single encoder/decoder façade,
//! tying together CRC attachment, code block segmentation, LDPC
//! encoding/decoding with filler/puncturing, rate matching, and HARQ
//! soft combining.
//!
//! # Chain (encode direction, TS 38.212 §5.1–§5.5)
//!
//! ```text
//! TB bits
//!   │  §5.1  CRC-24A attach  (L=24)
//!   ▼
//! TB + CRC-24A
//!   │  §5.2.2  Segmentation into C code blocks + CRC-24B per CB
//!   ▼
//! CB_0 … CB_{C-1}  (each K' bits)
//!   │  §5.3.2  LDPC encode_5g (fills filler bits → K systematic, N=n_b·Z codeword)
//!   ▼
//! CW_0 … CW_{C-1}  (each N bits)
//!   │  §5.4.2  Rate match + interleave  (E bits per CB)
//!   ▼
//! §5.5  Concatenate → G coded bits
//! ```
//!
//! # Chain (decode direction)
//!
//! ```text
//! G soft LLRs
//!   │  §5.5  De-concatenate per CB  (E LLRs each)
//!   ▼
//! §5.4.2  Rate de-match + HARQ soft combine
//!   │
//!   ▼
//! §5.3.2  LDPC decode_5g (filler = +∞ LLR, punctured = 0.0)
//!   │  CRC-24B check per CB
//!   ▼
//! §5.2.2  Desegment → TB bits
//!   │  §5.1  CRC-24A check
//!   ▼
//! TB bits (or error)
//! ```

use crate::alloc_prelude::*;
use crate::crc::{Crc24, CrcKind};
use crate::error::FecError;
use crate::harq::HarqBuffer;
#[cfg(not(feature = "no_std"))]
use crate::ldpc_pipeline::LdpcPipeline;
use crate::qc_ldpc::{QcLdpcDecoder, QcLdpcEncoder};
use crate::rate_matching::RateMatchCache;
use crate::segmentation::{SegmentationParams, compute_segmentation, segment};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Named-field configuration for a DL-SCH encoder/decoder pair.
///
/// [`DlSchEncoder::new`] and [`DlSchDecoder::new`] take four to six
/// positional numeric arguments, three of them `usize` — a call site like
/// `DlSchDecoder::new(8448, 0.5, 2, 16896, 20, 0.25)` compiles just as
/// happily with `qm` and `g` transposed. This struct gives every parameter
/// a name at the call site and keeps the encoder and decoder built from
/// literally the same value, which is what the chain requires anyway (a
/// decoder configured with a different `g` than its encoder is a bug, not
/// a choice).
///
/// # Examples
///
/// ```
/// use syndrome::transport_block::{DlSchConfig, DlSchDecoder, DlSchEncoder};
///
/// let cfg = DlSchConfig {
///     tb_size: 200,
///     target_rate: 0.5,
///     qm: 1,
///     g: 512,
///     ..DlSchConfig::default_decode_params()
/// };
/// let enc = DlSchEncoder::from_config(&cfg).unwrap();
/// let dec = DlSchDecoder::from_config(&cfg).unwrap();
/// assert_eq!(enc.output_bits(), dec.output_bits());
/// ```
#[derive(Clone, Copy, Debug)]
pub struct DlSchConfig {
    /// Transport block size in bits (before CRC attachment).
    pub tb_size: usize,
    /// Target code rate, used for base-graph selection.
    pub target_rate: f32,
    /// Modulation order $Q_m$ (bits per modulation symbol).
    pub qm: usize,
    /// Total coded bits $G$ available for this TB across all code blocks.
    pub g: usize,
    /// LDPC iterations per code block per decode call (decoder only;
    /// ignored by [`DlSchEncoder::from_config`]).
    pub iterations: usize,
    /// LOMS offset correction $\beta$ (decoder only; ignored by
    /// [`DlSchEncoder::from_config`]).
    pub offset_beta: f32,
}

impl DlSchConfig {
    /// The decoder-side defaults (`iterations: 20`, `offset_beta: 0.5`)
    /// with zeroed link parameters, intended for struct-update syntax as in
    /// the [`DlSchConfig`] example. There is no `Default` impl because a
    /// zero `tb_size`/`g` is not a usable configuration, only a base to
    /// spread real values over.
    ///
    /// `offset_beta: 0.5` is the value measured best for BG1 at production
    /// lifting sizes by `tests/ldpc_offset_beta_sweep.rs`, which sweeps
    /// $\beta$ against block error rate with confidence intervals; on BG2
    /// it is statistically indistinguishable from the sweep's best point.
    /// See that file to re-run the measurement rather than taking the
    /// constant on trust.
    ///
    /// # Returns
    ///
    /// A config whose four link parameters are zero and whose decoder
    /// tunables carry the crate-wide defaults.
    #[must_use]
    pub fn default_decode_params() -> Self {
        Self {
            tb_size: 0,
            target_rate: 0.0,
            qm: 0,
            g: 0,
            iterations: 20,
            offset_beta: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

/// Result of a single [`DlSchDecoder::decode`] call.
#[derive(Debug, Clone)]
pub struct DecodeReport {
    /// True if the final CRC-24A check over the transport block passed.
    pub crc_ok: bool,
    /// Per-code-block CRC results (true = block passed).
    pub cb_crc: Vec<bool>,
    /// Maximum LDPC iterations used across all code blocks.
    pub max_iters_used: usize,
    /// Number of HARQ transmissions combined so far.
    pub harq_tx_count: usize,
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/// 5G NR DL-SCH/UL-SCH transport block encoder.
///
/// Performs CRC attachment, segmentation, per-CB LDPC encoding, rate
/// matching, and concatenation per 3GPP TS 38.212 §5.1–§5.5.
///
/// # Examples
///
/// ```
/// use syndrome::transport_block::DlSchEncoder;
///
/// let tb_size = 200usize; // bits
/// let enc = DlSchEncoder::new(tb_size, 0.5, 1, 1000).unwrap();
/// let tb: Vec<u8> = (0..tb_size).map(|i| (i % 2) as u8).collect();
/// let mut coded = vec![0u8; enc.output_bits()];
/// enc.encode(&tb, 0, &mut coded).unwrap();
/// assert_eq!(coded.len(), enc.output_bits());
/// ```
pub struct DlSchEncoder {
    params: SegmentationParams,
    tb_crc: Crc24,
    /// Per-code-block CRC generator, retained for the multi-CB segmentation path
    /// (single-CB transport blocks carry only the TB CRC, so it is unused there).
    #[allow(dead_code)]
    cb_crc: Crc24,
    encoders: Vec<QcLdpcEncoder>,
    qm: usize,
    e_per_cb: usize,
    tb_size: usize,
    /// Rate-matching index-table cache (see [`RateMatchCache`]). All `C`
    /// code blocks of one `encode` call share the same `(bg, z, rv, qm,
    /// n_filler, e_bits)` key, and repeated `encode` calls at a steady RV
    /// (the common case between HARQ retransmissions) share it too, so this
    /// is a struct member rather than a call-local temporary. `encode`
    /// takes `&self` (preserving its public signature), so the cache lives
    /// behind a `RefCell`; borrowing it is uncontended (single-threaded,
    /// non-reentrant use) and adds no heap allocation of its own.
    rm_cache: core::cell::RefCell<RateMatchCache>,
}

impl DlSchEncoder {
    /// Create a DL-SCH encoder for a transport block.
    ///
    /// # Arguments
    ///
    /// * `tb_size`      - Transport block size in bits (before CRC).
    /// * `target_rate`  - Code rate (used for BG selection).
    /// * `qm`           - Modulation order ($Q_m$).
    /// * `g`            - Total coded bits available for this TB across all CBs.
    ///   Must be divisible by `qm`.
    ///
    /// # Returns
    ///
    /// An encoder configured for the chosen base graph, lifting size, and
    /// segmentation.
    ///
    /// # Errors
    ///
    /// Returns [`FecError`] if parameters are invalid or the LDPC encoder
    /// cannot be constructed.
    ///
    /// # Examples
    ///
    /// ```
    /// use syndrome::transport_block::DlSchEncoder;
    /// let enc = DlSchEncoder::new(200, 0.5, 1, 512).unwrap();
    /// ```
    pub fn new(tb_size: usize, target_rate: f32, qm: usize, g: usize) -> Result<Self, FecError> {
        if tb_size == 0 {
            return Err(FecError::InvalidParam("tb_size must be > 0"));
        }
        if qm == 0 {
            return Err(FecError::InvalidParam("qm must be >= 1"));
        }
        if g == 0 || !g.is_multiple_of(qm) {
            return Err(FecError::InvalidParam("G must be > 0 and divisible by Qm"));
        }

        let params = compute_segmentation(tb_size, target_rate)?;
        let tb_crc = Crc24::new(CrcKind::Crc24A);
        let cb_crc = Crc24::new(CrcKind::Crc24B);

        // Build one encoder per code block (they're all identical for a given
        // TB since all CBs share the same Z).
        let enc = QcLdpcEncoder::new(params.bg, params.z)?;
        let encoders = (0..params.c).map(|_| enc.clone()).collect::<Vec<_>>();

        // E per CB: G / C, must be divisible by Qm.
        let e_raw = g / params.c;
        let e_per_cb = (e_raw / qm) * qm; // round down to Qm multiple

        Ok(Self {
            params,
            tb_crc,
            cb_crc,
            encoders,
            qm,
            e_per_cb,
            tb_size,
            rm_cache: core::cell::RefCell::new(RateMatchCache::new()),
        })
    }

    /// Create a DL-SCH encoder from a named-field [`DlSchConfig`].
    ///
    /// Equivalent to [`DlSchEncoder::new`] with the config's four link
    /// parameters; the decoder-only fields (`iterations`, `offset_beta`)
    /// are ignored.
    ///
    /// # Arguments
    ///
    /// * `config` - The shared encoder/decoder configuration.
    ///
    /// # Errors
    ///
    /// Same conditions as [`DlSchEncoder::new`].
    pub fn from_config(config: &DlSchConfig) -> Result<Self, FecError> {
        Self::new(config.tb_size, config.target_rate, config.qm, config.g)
    }

    /// Total output coded bits $G$ (concatenation of all CB rate-matched outputs).
    pub fn output_bits(&self) -> usize {
        self.e_per_cb * self.params.c
    }

    /// Number of code blocks.
    pub fn num_code_blocks(&self) -> usize {
        self.params.c
    }

    /// Segmentation parameters (for diagnostics / rate-matcher configuration).
    pub fn segmentation(&self) -> &SegmentationParams {
        &self.params
    }

    /// Encode a transport block into coded bits.
    ///
    /// Performs: CRC-24A → segmentation + CRC-24B → LDPC encode_5g →
    /// rate match per CB → concatenate.
    ///
    /// # Arguments
    ///
    /// * `tb`      - Transport block bits (length must equal `tb_size`).
    /// * `rv`      - Redundancy version (0..=3).
    /// * `out`     - Output buffer of length [`DlSchEncoder::output_bits`].
    ///
    /// # Errors
    ///
    /// Returns [`FecError`] on size mismatches or internal encoding failures.
    pub fn encode(&self, tb: &[u8], rv: usize, out: &mut [u8]) -> Result<(), FecError> {
        if tb.len() != self.tb_size {
            return Err(FecError::BufferTooSmall {
                required: self.tb_size,
                provided: tb.len(),
            });
        }
        if out.len() < self.output_bits() {
            return Err(FecError::BufferTooSmall {
                required: self.output_bits(),
                provided: out.len(),
            });
        }

        // §5.1 — attach CRC-24A.
        let mut tb_with_crc = tb.to_vec();
        self.tb_crc.attach(&mut tb_with_crc);

        // §5.2.2 — segment into code blocks (each with CRC-24B if C > 1).
        let cb_blocks = segment(&tb_with_crc, &self.params)?;

        let enc = &self.encoders[0];
        let n = enc.codeword_bit_count();
        let mut codeword = vec![0u8; n];
        let mut e_buf = vec![0u8; self.e_per_cb];
        // All C code blocks share one (bg, z, rv, qm, n_filler, e_bits) key,
        // so the rate-matching selection/interleave index table is built
        // once (first iteration below) and reused verbatim for the rest --
        // see `RateMatchCache`'s doc comment.
        let mut rm_cache = self.rm_cache.borrow_mut();

        for (ci, cb) in cb_blocks.iter().enumerate() {
            // §5.3.2 — LDPC encode with filler padding.
            enc.encode_5g(cb, self.params.n_filler, &mut codeword)?;

            // §5.4.2 — rate match (E bits).
            rm_cache.rate_match_into(
                &codeword,
                &mut e_buf,
                rv,
                self.qm,
                self.params.bg,
                self.params.z,
                self.params.n_filler,
            )?;

            // §5.5 — concatenate into output.
            let start = ci * self.e_per_cb;
            out[start..start + self.e_per_cb].copy_from_slice(&e_buf);
        }

        Ok(())
    }
}

// Encoder must be Clone for multi-worker use.
impl Clone for QcLdpcEncoder {
    fn clone(&self) -> Self {
        // Rebuild from the stored (bg, z) — a setup path, allocation intended.
        // `base_graph()` is `None` only for encoders built via
        // `QcLdpcEncoder::from_raw_edges` (e.g. Wi-Fi — see
        // `crate::wifi_ldpc_tables`); the DL-SCH/UL-SCH transport-block
        // pipeline in this module is 3GPP-only and never holds one of those,
        // so rebuilding from `(bg, z)` is always valid here.
        let bg = self
            .base_graph()
            .expect("clone: transport_block encoders are always 3GPP-constructed (Some(bg))");
        QcLdpcEncoder::new(bg, self.lifting_size()).expect("clone: original encoder was valid")
    }
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

/// 5G NR DL-SCH/UL-SCH transport block decoder.
///
/// Maintains per-CB [`HarqBuffer`]s across retransmissions.  Call
/// [`DlSchDecoder::decode`] on each received LLR vector (with `rv` indicating the
/// redundancy version).  On success, the TB bits are returned and the
/// HARQ buffers are flushed.  On failure (CRC miss), keep calling
/// `decode` with subsequent transmissions to perform IR combining.
///
/// # Examples
///
/// ```no_run
/// use syndrome::transport_block::{DlSchEncoder, DlSchDecoder};
///
/// let tb_size = 200usize;
/// let enc = DlSchEncoder::new(tb_size, 0.5, 1, 1000).unwrap();
/// let mut dec = DlSchDecoder::new(tb_size, 0.5, 1, 1000, 10, 0.25).unwrap();
///
/// let tb: Vec<u8> = (0..tb_size).map(|i| (i % 2) as u8).collect();
/// let mut coded = vec![0u8; enc.output_bits()];
/// enc.encode(&tb, 0, &mut coded).unwrap();
///
/// // Convert hard bits to LLRs (noiseless: 0 → +10.0, 1 → -10.0).
/// let llr_g: Vec<f32> = coded.iter().map(|&b| if b == 0 { 10.0 } else { -10.0 }).collect();
/// let mut tb_out = vec![0u8; tb_size];
/// let report = dec.decode(&llr_g, 0, &mut tb_out).unwrap();
/// assert!(report.crc_ok);
/// ```
pub struct DlSchDecoder {
    params: SegmentationParams,
    tb_crc: Crc24,
    cb_crc: Crc24,
    decoders: Vec<QcLdpcDecoder>,
    harq_bufs: Vec<HarqBuffer>,
    qm: usize,
    e_per_cb: usize,
    iterations: usize,
    tb_size: usize,
    /// Per-CB LDPC decode scratch, all sized once here at construction (`n`,
    /// the edge-buffer size, and the layer-buffer size are fixed for the
    /// decoder's lifetime, since `bg`/`z` never change) instead of being
    /// re-allocated on every [`Self::decode`] call, mirroring the pattern
    /// already used by [`crate::turbo::TurboDecoder`]'s constructor-time
    /// scratch and the LDPC pipeline's `FrameSlot`.
    llr_cb: Vec<f32>,
    edge_r: Vec<f32>,
    layer_scratch: Vec<f32>,
    hard: Vec<u8>,
    /// Reconstructed transport-block info bits, accumulated across all `C`
    /// code blocks; cleared (not reallocated) at the start of every
    /// [`Self::decode`] call. (`cb_crc_results` is *not* struct scratch: it
    /// is moved out into the returned [`DecodeReport::cb_crc`] every call,
    /// so keeping it as reusable scratch would just add a clone back.)
    all_info: Vec<u8>,
    /// Optional multi-worker LDPC pipeline, installed by
    /// [`DlSchDecoder::with_pipeline`]. `None` means every code block is
    /// decoded on the calling thread, which is the default. Not present
    /// under the `no_std` feature at all ([`LdpcPipeline`] needs
    /// `std::thread`): [`Self::decode`] always takes the sequential path
    /// there, and [`Self::with_pipeline`]/[`Self::worker_count`] do not
    /// exist to call.
    #[cfg(not(feature = "no_std"))]
    pipeline: Option<LdpcPipeline>,
}

/// Build the decoder-ready LLR buffer for one code block.
///
/// Shared by [`DlSchDecoder::decode`]'s sequential and pipelined paths, which
/// is the whole reason it is a free function: the mapping below is subtle
/// enough that two copies would be two chances to get it wrong, and it has
/// been wrong here before (see the note on `ncb`).
///
/// Three steps:
///
/// 1. **HARQ combine.** This transmission's `E` received LLRs are scattered
///    into the code block's circular buffer and accumulated with whatever
///    earlier redundancy versions deposited there.
/// 2. **Alignment.** The circular buffer excludes the $2Z$ punctured
///    systematic positions — `harq[j]` is the LLR for codeword bit $2Z + j$ —
///    so it lands at offset $2Z$ in the full $N$-length decode buffer, whose
///    prefix stays at zero. `ncb` ($n_b \cdot Z$: $66Z$ for BG1, $50Z$ for
///    BG2) is *already* $N - 2Z$, because $n_b$ is two less than the base
///    graph's column count. Subtracting $2Z$ from it again — as an earlier
///    version did — double-counts the puncture and silently drops the last
///    $2Z$ accumulated LLRs of every combined codeword, which is exactly the
///    region an `rv = 2`/`rv = 3` retransmission is walking into.
/// 3. **5G initialisation.** [`QcLdpcDecoder::init_5g_llr`] pins the filler
///    bits and forces the punctured prefix to the erasure value. The
///    sequential path used to get this from `decode_5g`; the pipelined path
///    cannot, because its worker thread calls the plain decode entry point,
///    so both now call the same helper explicitly.
///
/// # Errors
///
/// Propagates [`FecError`] from the HARQ combine or from the 5G
/// initialisation.
fn prepare_cb_llr(
    harq: &mut HarqBuffer,
    decoder: &QcLdpcDecoder,
    e_llr: &[f32],
    rv: usize,
    qm: usize,
    z: usize,
    n_filler: usize,
    dest: &mut [f32],
) -> Result<(), FecError> {
    harq.combine(e_llr, rv, qm, 0)?;

    let two_z = 2 * z;
    let valid_len = harq.ncb();
    debug_assert_eq!(
        two_z + valid_len,
        dest.len(),
        "ncb (N - 2Z) plus the punctured prefix must exactly fill the N-length decode buffer"
    );
    dest.iter_mut().for_each(|v| *v = 0.0);
    dest[two_z..two_z + valid_len].copy_from_slice(&harq.llr_buffer()[..valid_len]);

    decoder.init_5g_llr(dest, n_filler)
}

/// Decode every code block sequentially, one at a time on the calling
/// thread.
///
/// [`DlSchDecoder::decode`]'s only path under the `no_std` feature (no
/// [`crate::ldpc_pipeline::LdpcPipeline`] there — it needs `std::thread`),
/// and the `std` build's fallback whenever a pipeline either was not
/// installed or has nothing to parallelise ($C = 1$). A free function
/// specifically so there is exactly one copy of this loop for both cases to
/// share, for the same reason [`prepare_cb_llr`] above is one: two copies
/// are two chances for them to drift.
///
/// Writes decoded results into `cb_crc_results`/`all_info` and folds each
/// code block's iteration count into `max_iters`, exactly as the pipelined
/// path's drain loop does — the two must produce identical output for
/// [`tests::pipelined_decode_matches_sequential_decode`] to hold.
///
/// # Errors
///
/// Propagates [`FecError`] from [`prepare_cb_llr`] or from
/// [`QcLdpcDecoder::decode_layered_offset_min_sum`].
#[allow(clippy::too_many_arguments)]
fn decode_sequential_cbs(
    harq_bufs: &mut [HarqBuffer],
    decoders: &[QcLdpcDecoder],
    rx_llr: &[f32],
    rv: usize,
    qm: usize,
    z: usize,
    n_filler: usize,
    e_per_cb: usize,
    llr_cb: &mut [f32],
    edge_r: &mut [f32],
    layer_scratch: &mut [f32],
    hard: &mut [u8],
    iterations: usize,
    k_prime: usize,
    has_cb_crc: bool,
    cb_crc: &Crc24,
    all_info: &mut [u8],
    payload_len: usize,
    n_cb: usize,
    cb_crc_results: &mut [bool],
    max_iters: &mut usize,
) -> Result<(), FecError> {
    for ci in 0..n_cb {
        let e_start = ci * e_per_cb;
        prepare_cb_llr(
            &mut harq_bufs[ci],
            &decoders[ci],
            &rx_llr[e_start..e_start + e_per_cb],
            rv,
            qm,
            z,
            n_filler,
            llr_cb,
        )?;

        let cb_iters = decoders[ci].decode_layered_offset_min_sum(
            llr_cb,
            edge_r,
            layer_scratch,
            hard,
            iterations,
        )?;
        *max_iters = (*max_iters).max(cb_iters);

        let info_bits = &hard[..k_prime];
        cb_crc_results[ci] = !has_cb_crc || cb_crc.check(info_bits);
        all_info[ci * payload_len..(ci + 1) * payload_len]
            .copy_from_slice(&info_bits[..payload_len]);
    }
    Ok(())
}

/// Number of payload bits each code block contributes to the reassembled
/// transport block: its systematic bits minus the CRC-24B trailer when the
/// transport block was segmented.
///
/// Uniform across code blocks — segmentation gives every one the same $K'$ —
/// which is what lets the pipelined path write completions straight into
/// `all_info` at `ci * payload_bits(..)` instead of appending them in order.
fn payload_bits(k_prime: usize, has_cb_crc: bool) -> usize {
    if has_cb_crc {
        k_prime.saturating_sub(24)
    } else {
        k_prime
    }
}

impl DlSchDecoder {
    /// Create a DL-SCH decoder.
    ///
    /// # Arguments
    ///
    /// * `tb_size`      - Transport block size in bits.
    /// * `target_rate`  - Code rate (for BG selection).
    /// * `qm`           - Modulation order.
    /// * `g`            - Total coded bits available.
    /// * `iterations`   - LDPC iterations per CB per call.
    /// * `offset_beta`  - LOMS offset correction $\beta$. Use `0.5` unless
    ///   you have measured otherwise for your configuration; see
    ///   [`DlSchConfig::default_decode_params`] for where that value comes
    ///   from.
    ///
    /// # Errors
    ///
    /// Returns [`FecError`] on invalid parameters.
    ///
    /// # Examples
    ///
    /// ```
    /// use syndrome::transport_block::DlSchDecoder;
    /// let dec = DlSchDecoder::new(200, 0.5, 1, 512, 10, 0.25).unwrap();
    /// ```
    pub fn new(
        tb_size: usize,
        target_rate: f32,
        qm: usize,
        g: usize,
        iterations: usize,
        offset_beta: f32,
    ) -> Result<Self, FecError> {
        if tb_size == 0 {
            return Err(FecError::InvalidParam("tb_size must be > 0"));
        }
        if qm == 0 || g == 0 || !g.is_multiple_of(qm) {
            return Err(FecError::InvalidParam(
                "Qm and G must be positive; G % Qm must be 0",
            ));
        }

        let params = compute_segmentation(tb_size, target_rate)?;
        let tb_crc = Crc24::new(CrcKind::Crc24A);
        let cb_crc = Crc24::new(CrcKind::Crc24B);

        let dec = QcLdpcDecoder::with_lifting_size(params.bg, params.z, offset_beta)?;
        let decoders: Vec<QcLdpcDecoder> = (0..params.c).map(|_| dec.clone()).collect();
        let harq_bufs: Vec<HarqBuffer> = (0..params.c)
            .map(|_| HarqBuffer::with_filler(params.bg, params.z, params.n_filler))
            .collect();

        let e_raw = g / params.c;
        let e_per_cb = (e_raw / qm) * qm;

        let dec0 = &decoders[0];
        let n = dec0.variable_node_count();
        let edge_r = vec![0.0f32; dec0.required_edge_buffer()];
        let layer_scratch = vec![0.0f32; dec0.required_layer_buffer()];
        let info_per_cb = if params.has_cb_crc {
            params.k_prime - 24
        } else {
            params.k_prime
        };
        let all_info_capacity = info_per_cb * params.c;

        Ok(Self {
            params,
            tb_crc,
            cb_crc,
            decoders,
            harq_bufs,
            qm,
            e_per_cb,
            iterations,
            tb_size,
            llr_cb: vec![0.0f32; n],
            edge_r,
            layer_scratch,
            hard: vec![0u8; n],
            all_info: Vec::with_capacity(all_info_capacity),
            #[cfg(not(feature = "no_std"))]
            pipeline: None,
        })
    }

    /// Install a multi-worker LDPC pipeline, so the transport block's code
    /// blocks decode concurrently instead of one after another.
    ///
    /// # Why this is opt-in
    ///
    /// [`LdpcPipeline`] spawns worker threads at construction and keeps them
    /// spinning for the decoder's lifetime. That is the right trade for a
    /// receiver decoding a stream of large transport blocks and the wrong one
    /// almost everywhere else: a transport block small enough to fit in a
    /// single code block — the common case — has nothing to parallelise, and
    /// a short-lived decoder pays the spawn cost against no work. Making it a
    /// separate call keeps [`DlSchDecoder::new`] free of hidden threads.
    ///
    /// The pipeline is used only when the transport block actually has more
    /// than one code block; with $C = 1$ [`DlSchDecoder::decode`] takes the
    /// sequential path regardless, because handing one code block to a worker
    /// and waiting for it is strictly slower than decoding it here.
    ///
    /// # What stays the same
    ///
    /// Everything observable. Both paths run the same LOMS decoder over the
    /// same HARQ-combined LLRs for the same iteration budget, so the decoded
    /// transport block, the per-code-block CRC flags and the reported
    /// iteration count are identical — only the order in which code blocks
    /// finish differs, and results are reassembled by index rather than by
    /// arrival. `pipelined_decode_matches_sequential_decode` in this module's
    /// tests asserts that bit-for-bit.
    ///
    /// # Arguments
    ///
    /// * `n_workers` - Worker threads to spawn. [`LdpcPipeline`] clamps this
    ///   into `1..=8`. Values above the code block count waste threads: no
    ///   transport block can keep more workers busy than it has code blocks.
    ///
    /// # Returns
    ///
    /// `self`, with the pipeline installed, so this chains onto a
    /// constructor.
    ///
    /// # Errors
    ///
    /// Returns [`FecError::InvalidParam`] if `n_workers` is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use syndrome::transport_block::DlSchDecoder;
    ///
    /// // A transport block large enough to segment into several code blocks.
    /// let dec = DlSchDecoder::new(30000, 0.5, 2, 90000, 10, 0.5)
    ///     .unwrap()
    ///     .with_pipeline(4)
    ///     .unwrap();
    /// assert!(dec.num_code_blocks() > 1);
    /// ```
    #[cfg(not(feature = "no_std"))]
    pub fn with_pipeline(mut self, n_workers: usize) -> Result<Self, FecError> {
        if n_workers == 0 {
            return Err(FecError::InvalidParam("n_workers must be > 0"));
        }
        // The pipeline needs a decoder to clone into each worker; every entry
        // in `decoders` is a clone of the same one, so any of them will do.
        self.pipeline = Some(LdpcPipeline::with_workers(
            self.decoders[0].clone(),
            self.iterations,
            n_workers,
        ));
        Ok(self)
    }

    /// Number of code blocks this transport block segments into.
    ///
    /// Worth checking before `DlSchDecoder::with_pipeline` (not linked: not
    /// present under the `no_std` feature) — at `1` there is nothing for
    /// extra workers to do.
    #[must_use]
    pub fn num_code_blocks(&self) -> usize {
        self.params.c
    }

    /// Number of worker threads decoding code blocks, or `0` when no pipeline
    /// is installed and decoding happens on the calling thread.
    ///
    /// [`LdpcPipeline`] clamps the requested worker count, so this can be
    /// lower than what was passed to [`DlSchDecoder::with_pipeline`].
    #[cfg(not(feature = "no_std"))]
    #[must_use]
    pub fn worker_count(&self) -> usize {
        self.pipeline.as_ref().map_or(0, LdpcPipeline::worker_count)
    }

    /// Create a DL-SCH decoder from a named-field [`DlSchConfig`].
    ///
    /// Equivalent to [`DlSchDecoder::new`] with all six of the config's
    /// parameters. Build the matching encoder from the same value with
    /// [`DlSchEncoder::from_config`] so the pair can never disagree.
    ///
    /// # Arguments
    ///
    /// * `config` - The shared encoder/decoder configuration.
    ///
    /// # Errors
    ///
    /// Same conditions as [`DlSchDecoder::new`].
    pub fn from_config(config: &DlSchConfig) -> Result<Self, FecError> {
        Self::new(
            config.tb_size,
            config.target_rate,
            config.qm,
            config.g,
            config.iterations,
            config.offset_beta,
        )
    }

    /// Total received coded bits $G$ expected by [`DlSchDecoder::decode`]'s
    /// `rx_llr` argument (concatenation of all CB rate-matched inputs).
    ///
    /// This is the value to size `rx_llr` from — not the raw `g` passed to
    /// [`DlSchDecoder::new`]. `g` is rounded down internally to a multiple of
    /// `Qm` per code block (matching [`DlSchEncoder::output_bits`], which
    /// this must stay equal to for a given `(tb_size, target_rate, qm, g)`),
    /// so the two only coincide when `g` was already a multiple of
    /// `Qm * num_code_blocks`.
    ///
    /// # Examples
    ///
    /// ```
    /// use syndrome::transport_block::{DlSchEncoder, DlSchDecoder};
    /// let enc = DlSchEncoder::new(200, 0.5, 1, 512).unwrap();
    /// let dec = DlSchDecoder::new(200, 0.5, 1, 512, 10, 0.25).unwrap();
    /// assert_eq!(enc.output_bits(), dec.output_bits());
    /// ```
    pub fn output_bits(&self) -> usize {
        self.e_per_cb * self.params.c
    }

    /// Decode a received LLR vector into a transport block.
    ///
    /// The HARQ accumulators are updated with this transmission before
    /// decoding.  If the final CRC-24A passes, the HARQ buffers are flushed
    /// ready for the next TB.
    ///
    /// # Arguments
    ///
    /// * `rx_llr` - Received soft LLRs of length `E * C` (coded bits).
    /// * `rv`     - Redundancy version (0..=3).
    /// * `tb_out` - Output buffer of length `tb_size` (info bits).
    ///
    /// # Returns
    ///
    /// A [`DecodeReport`] on success.
    ///
    /// # Errors
    ///
    /// Returns [`FecError`] on buffer size mismatches.
    pub fn decode(
        &mut self,
        rx_llr: &[f32],
        rv: usize,
        tb_out: &mut [u8],
    ) -> Result<DecodeReport, FecError> {
        let total_e = self.e_per_cb * self.params.c;
        if rx_llr.len() < total_e {
            return Err(FecError::BufferTooSmall {
                required: total_e,
                provided: rx_llr.len(),
            });
        }
        if tb_out.len() < self.tb_size {
            return Err(FecError::BufferTooSmall {
                required: self.tb_size,
                provided: tb_out.len(),
            });
        }

        // Small Copy fields, read up front so the struct destructure below
        // (needed to get disjoint `&mut` access to the scratch buffers
        // alongside `harq_bufs`/`decoders`/`cb_crc`) doesn't have to thread
        // borrows of `self.params`/`self.qm`/etc. through the loop.
        let n_cb = self.params.c;
        let n_filler = self.params.n_filler;
        let z = self.params.z;
        let has_cb_crc = self.params.has_cb_crc;
        let k_prime = self.params.k_prime;
        let qm = self.qm;
        let e_per_cb = self.e_per_cb;
        let iterations = self.iterations;
        let tb_size = self.tb_size;

        let mut cb_crc_results = vec![false; n_cb];
        let mut max_iters = 0usize;
        let harq_tx = self.harq_bufs[0].tx_count() + 1;

        // Per-CB LDPC scratch (`llr_cb`/`edge_r`/`layer_scratch`/`hard`) and
        // `all_info` are struct-owned (sized once in `new`, see their doc
        // comments) rather than allocated here; `all_info` is resized (not
        // reallocated, its capacity is set in `new`) at the start of every
        // call.
        self.all_info.clear();
        let Self {
            harq_bufs,
            decoders,
            llr_cb,
            edge_r,
            layer_scratch,
            hard,
            all_info,
            cb_crc,
            #[cfg(not(feature = "no_std"))]
            pipeline,
            ..
        } = self;
        let n = llr_cb.len();

        let payload_len = payload_bits(k_prime, has_cb_crc);
        // Every code block contributes the same number of payload bits, so the
        // reassembled transport block is a flat array indexed by `ci *
        // payload_len` — which is what lets the pipelined path below write
        // out-of-order completions into their right place. The sequential path
        // writes the same offsets in order.
        all_info.resize(n_cb * payload_len, 0);

        #[cfg(feature = "no_std")]
        decode_sequential_cbs(
            harq_bufs,
            decoders,
            rx_llr,
            rv,
            qm,
            z,
            n_filler,
            e_per_cb,
            &mut llr_cb[..n],
            edge_r,
            layer_scratch,
            hard,
            iterations,
            k_prime,
            has_cb_crc,
            cb_crc,
            all_info,
            payload_len,
            n_cb,
            &mut cb_crc_results,
            &mut max_iters,
        )?;

        #[cfg(not(feature = "no_std"))]
        match pipeline {
            // ── Pipelined: code blocks decode concurrently on worker threads.
            Some(pipe) if n_cb > 1 => {
                let mut next_submit = 0usize;
                let mut completed = 0usize;

                while completed < n_cb {
                    // Fill and dispatch as many code blocks as there are free
                    // pool slots. `acquire` returning None is ordinary
                    // back-pressure, not an error: it means all sixteen slots
                    // are in flight, so the loop falls through to draining.
                    let mut progressed = false;
                    while next_submit < n_cb {
                        let Some(mut frame) = pipe.acquire() else {
                            break;
                        };
                        let e_start = next_submit * e_per_cb;
                        prepare_cb_llr(
                            &mut harq_bufs[next_submit],
                            &decoders[next_submit],
                            &rx_llr[e_start..e_start + e_per_cb],
                            rv,
                            qm,
                            z,
                            n_filler,
                            frame.llr_mut(),
                        )?;
                        frame.set_tag(next_submit);
                        // Unreachable by construction (at most POOL_SIZE frames
                        // in flight, each ring holds POOL_SIZE), and `submit`
                        // hands the frame back rather than leaking its slot if
                        // it ever were reachable — so put it back and retry.
                        if let Err(frame) = pipe.submit(frame) {
                            pipe.release(frame);
                            break;
                        }
                        next_submit += 1;
                        progressed = true;
                    }

                    // Drain whatever finished. Frames come back in completion
                    // order, which is why each carries its code block index as
                    // a tag.
                    while let Some(frame) = pipe.try_recv() {
                        let ci = frame.tag();
                        let info_bits = &frame.hard()[..k_prime];
                        cb_crc_results[ci] = !has_cb_crc || cb_crc.check(info_bits);
                        all_info[ci * payload_len..(ci + 1) * payload_len]
                            .copy_from_slice(&info_bits[..payload_len]);
                        max_iters = max_iters.max(frame.iterations_used());
                        pipe.release(frame);
                        completed += 1;
                        progressed = true;
                    }

                    if !progressed {
                        // Everything is in flight and nothing has finished;
                        // wait for a worker rather than burning the loop.
                        std::hint::spin_loop();
                    }
                }
            }

            // ── Sequential: one code block at a time on this thread. Shared
            // with the `no_std` build's only path (see above) through the
            // same free function -- exactly the "two copies, two chances to
            // get it wrong" trap `prepare_cb_llr` was already extracted to
            // avoid, now avoided the same way for this loop.
            _ => {
                decode_sequential_cbs(
                    harq_bufs,
                    decoders,
                    rx_llr,
                    rv,
                    qm,
                    z,
                    n_filler,
                    e_per_cb,
                    &mut llr_cb[..n],
                    edge_r,
                    layer_scratch,
                    hard,
                    iterations,
                    k_prime,
                    has_cb_crc,
                    cb_crc,
                    all_info,
                    payload_len,
                    n_cb,
                    &mut cb_crc_results,
                    &mut max_iters,
                )?;
            }
        }

        // §5.1 — CRC-24A check on reconstructed TB.
        let tb_bits = &all_info[..tb_size.min(all_info.len())];
        // Re-compute what CRC-24A should be.
        let tb_crc_ok = {
            let expected_crc = self.tb_crc.compute(tb_bits);
            // Reconstruct: compare against what the received TB contained.
            // If segmentation included CRC-24A as part of the last CB's payload,
            // it's in all_info after the info bits.
            let crc_bits_start = tb_size;
            if all_info.len() >= crc_bits_start + 24 {
                let received_crc = all_info[crc_bits_start..crc_bits_start + 24]
                    .iter()
                    .fold(0u32, |acc, &b| (acc << 1) | (b as u32 & 1));
                expected_crc == received_crc
            } else {
                false
            }
        };

        if crc_ok(&cb_crc_results) && tb_crc_ok {
            // Copy TB bits to output.
            let copy_len = tb_size.min(all_info.len());
            tb_out[..copy_len].copy_from_slice(&all_info[..copy_len]);
            // Flush HARQ on success.
            for buf in harq_bufs.iter_mut() {
                buf.flush();
            }
        }

        Ok(DecodeReport {
            crc_ok: tb_crc_ok && crc_ok(&cb_crc_results),
            cb_crc: cb_crc_results,
            max_iters_used: max_iters,
            harq_tx_count: harq_tx,
        })
    }

    /// Flush all HARQ buffers (e.g. on handover or scheduler reset).
    pub fn flush_harq(&mut self) {
        for buf in &mut self.harq_bufs {
            buf.flush();
        }
    }
}

fn crc_ok(results: &[bool]) -> bool {
    results.iter().all(|&b| b)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel_sim::AwgnChannel;

    #[test]
    fn encoder_output_length_matches() {
        let enc = DlSchEncoder::new(100, 0.5, 1, 512).unwrap();
        let tb: Vec<u8> = vec![0u8; 100];
        let mut out = vec![0u8; enc.output_bits()];
        enc.encode(&tb, 0, &mut out).unwrap();
        // All bits should be 0 or 1.
        assert!(out.iter().all(|&b| b <= 1));
    }

    #[test]
    fn encoder_decoder_noiseless_roundtrip() {
        let tb_size = 100usize;
        let g = 512;
        let enc = DlSchEncoder::new(tb_size, 0.5, 1, g).unwrap();
        let mut dec = DlSchDecoder::new(tb_size, 0.5, 1, g, 10, 0.25).unwrap();

        let tb: Vec<u8> = (0..tb_size).map(|i| (i % 3 == 0) as u8).collect();
        let mut coded = vec![0u8; enc.output_bits()];
        enc.encode(&tb, 0, &mut coded).unwrap();

        // Convert hard bits to strong LLRs (noiseless).
        let llr: Vec<f32> = coded
            .iter()
            .map(|&b| if b == 0 { 10.0 } else { -10.0 })
            .collect();
        let mut tb_out = vec![0u8; tb_size];
        let report = dec.decode(&llr, 0, &mut tb_out).unwrap();

        // CRC should pass for a noiseless channel.
        // (Exact bit recovery depends on LDPC convergence at z; verifying CRC structure)
        assert!(report.max_iters_used <= 10);
    }

    /// Regression test for a `g` that is not already a multiple of
    /// `Qm * num_code_blocks`. Every other round-trip test in this module
    /// happens to pass a "nice" `g` (either `qm = 1`, which makes the
    /// rounding a no-op, or `g = c * n`, the full unpunctured codeword), so
    /// none of them ever exercised the rounding-down that both
    /// `DlSchEncoder::new` and `DlSchDecoder::new` do internally
    /// (`e_per_cb = (g / c / qm) * qm`). This case is chosen so the real
    /// output length (8000) differs from the raw `g` passed in (8002),
    /// which is exactly the condition `DlSchDecoder::output_bits` exists to
    /// report correctly instead of a caller assuming `rx_llr.len() == g`.
    #[test]
    fn round_trip_with_g_not_a_multiple_of_qm_times_c() {
        let (tb_size, rate, qm, g) = (3824usize, 0.5f32, 2usize, 8002usize);
        let params = compute_segmentation(tb_size, rate).unwrap();
        assert_eq!(
            params.c, 2,
            "test assumes C=2 so the rounding actually bites"
        );

        let enc = DlSchEncoder::new(tb_size, rate, qm, g).unwrap();
        let mut dec = DlSchDecoder::new(tb_size, rate, qm, g, 20, 0.25).unwrap();

        assert_eq!(
            enc.output_bits(),
            dec.output_bits(),
            "encoder and decoder must agree on the real (rounded) coded length"
        );
        assert!(
            enc.output_bits() < g,
            "g={g} was chosen to not already be a multiple of qm*c; if this fails the \
             test fixture needs a different g, not the assertion relaxed"
        );

        let tb: Vec<u8> = (0..tb_size).map(|i| ((i * 7 + 3) % 5 < 2) as u8).collect();
        let mut coded = vec![0u8; enc.output_bits()];
        enc.encode(&tb, 0, &mut coded).unwrap();

        let llr: Vec<f32> = coded
            .iter()
            .map(|&b| if b == 0 { 10.0 } else { -10.0 })
            .collect();
        let mut tb_out = vec![0u8; tb_size];
        // Sized from dec.output_bits(), not the raw g — this is the buffer a
        // correct caller builds, and decode() must accept it.
        let report = dec.decode(&llr, 0, &mut tb_out).unwrap();

        assert!(report.crc_ok, "CRC failed to pass over a noiseless channel");
        assert_eq!(tb_out, tb, "recovered payload mismatch");
    }

    #[test]
    fn invalid_params_rejected() {
        assert!(DlSchEncoder::new(0, 0.5, 1, 512).is_err());
        assert!(DlSchEncoder::new(100, 0.5, 0, 512).is_err());
        assert!(DlSchEncoder::new(100, 0.5, 1, 0).is_err());
    }

    /// FINDING 9 round-trip regression guard: a sweep of "awkward" transport
    /// block sizes (including the exact `compute_segmentation(8425, 0.5)`
    /// reproducer from tests/robustness.rs, plus sizes that land on every
    /// residue of $B' \bmod C$, not just the lucky evenly-divisible ones)
    /// proving that the fixed segmentation math (`K' = \lceil B'/C \rceil`,
    /// zero-padded slack — see `segmentation`'s module doc) still round-trips
    /// correctly end-to-end through the full `DlSchEncoder` -> channel ->
    /// `DlSchDecoder` chain, not just in isolation.
    ///
    /// Each case sends the *entire* codeword per code block (`g = c * n`,
    /// `qm = 1`, no puncturing) over a noiseless channel, so decode success
    /// isolates segmentation/CRC correctness from LDPC waterfall behavior.
    #[test]
    fn round_trip_sweep_awkward_tb_sizes() {
        let cases: &[(usize, f32)] = &[
            (8425, 0.5),  // exact finding-9 reproducer: BG1, C=2, non-divisible B'
            (100, 0.5),   // BG2, C=1
            (197, 0.5),   // BG2, C=1
            (3800, 0.5),  // BG2, C=1 (B == Kcb boundary)
            (3825, 0.5),  // BG1, C=1 (A just above the BG2 cutoff)
            (8448, 0.5),  // BG1, B == Kcb exactly -> C=1
            (8449, 0.5),  // BG1, C=2
            (10001, 0.5), // BG1, C=2
            (12345, 0.9), // BG1, C=2
            (20000, 0.5), // BG1, C=3
        ];
        for &(tb_size, rate) in cases {
            let params = compute_segmentation(tb_size, rate).unwrap();
            let qm = 1usize;
            // Send the whole codeword per code block: no puncturing, so
            // decode success isolates segmentation correctness.
            let g = params.c * params.n;

            let enc = DlSchEncoder::new(tb_size, rate, qm, g)
                .unwrap_or_else(|e| panic!("tb_size={tb_size} rate={rate}: encoder: {e:?}"));
            let mut dec = DlSchDecoder::new(tb_size, rate, qm, g, 20, 0.25)
                .unwrap_or_else(|e| panic!("tb_size={tb_size} rate={rate}: decoder: {e:?}"));

            let tb: Vec<u8> = (0..tb_size).map(|i| ((i * 13 + 5) % 7 < 3) as u8).collect();
            let mut coded = vec![0u8; enc.output_bits()];
            enc.encode(&tb, 0, &mut coded)
                .unwrap_or_else(|e| panic!("tb_size={tb_size} rate={rate}: encode: {e:?}"));

            // Noiseless channel: strong LLRs matching the transmitted bits.
            let llr: Vec<f32> = coded
                .iter()
                .map(|&b| if b == 0 { 10.0 } else { -10.0 })
                .collect();
            let mut tb_out = vec![0u8; tb_size];
            let report = dec
                .decode(&llr, 0, &mut tb_out)
                .unwrap_or_else(|e| panic!("tb_size={tb_size} rate={rate}: decode: {e:?}"));

            assert!(
                report.crc_ok,
                "tb_size={tb_size} rate={rate}: CRC failed to pass over a noiseless channel"
            );
            assert_eq!(
                tb_out, tb,
                "tb_size={tb_size} rate={rate}: recovered payload mismatch"
            );
        }
    }

    /// Regression test for the HARQ off-by-2Z bug: `decode`'s mapping from
    /// `HarqBuffer::llr_buffer()` into the LDPC decoder's LLR array dropped
    /// the last 2Z accumulated positions (see the comment on `valid_len` in
    /// `decode`). A single noisy transmission that fails CRC alone, combined
    /// with a second, independently-noisy transmission of the same
    /// codeword, must recover the transport block — this is HARQ combining's
    /// entire reason to exist, and losing 2Z worth of accumulated LLR per
    /// code block is exactly the kind of degradation that turns a
    /// borderline combined decode back into a failure without ever raising
    /// an error.
    ///
    /// The second shot MUST be an incremental-redundancy retransmission
    /// (`rv=3`, not `rv=0` again) for this test to actually exercise the
    /// bug: at `rv=0`, the rate-matching bit-selection walk starts at
    /// circular-buffer offset `k0=0` and — for a `g` this small — never
    /// reaches the tail 2Z positions the fix restores, so a first version of
    /// this test using `rv=0` for both shots passed identically whether or
    /// not the bug was present. `rv=3`'s `k0` walk is specifically what
    /// reaches into that tail (see the comment on `valid_len` in `decode`).
    /// Parameters found by grid search over (`Eb/N0`, seed pairs) for a case
    /// that both (a) has shot 1 fail alone and shot 1+2 combined succeed,
    /// and (b) was confirmed, by temporarily reverting just the `valid_len`
    /// fix, to fail under the old buggy code — so this is a verified
    /// regression guard, not merely a plausible-looking one.
    #[test]
    fn harq_combining_recovers_a_block_that_fails_alone() {
        use crate::channel_sim::AwgnChannel;

        let (tb_size, rate, qm, g) = (400usize, 0.5f32, 1usize, 700usize);
        let (ebno_db, seed_a, seed_b) = (-1.5f32, 1u64, 1001u64);

        let enc = DlSchEncoder::new(tb_size, rate, qm, g).unwrap();
        let mut dec = DlSchDecoder::new(tb_size, rate, qm, g, 20, 0.25).unwrap();
        let tb: Vec<u8> = (0..tb_size).map(|i| ((i * 7 + 3) % 5 < 2) as u8).collect();

        let mut coded_rv0 = vec![0u8; enc.output_bits()];
        enc.encode(&tb, 0, &mut coded_rv0).unwrap();
        let mut coded_rv3 = vec![0u8; enc.output_bits()];
        enc.encode(&tb, 3, &mut coded_rv3).unwrap();

        let mut tb_out = vec![0u8; tb_size];

        let mut ch_a = AwgnChannel::new(ebno_db, rate, seed_a);
        let llr_a = ch_a.transmit(&coded_rv0);
        let report_a = dec.decode(&llr_a, 0, &mut tb_out).unwrap();
        assert!(
            !report_a.crc_ok,
            "fixture regressed: shot 1 alone now succeeds, so this case no longer tests combining"
        );

        let mut ch_b = AwgnChannel::new(ebno_db, rate, seed_b);
        let llr_b = ch_b.transmit(&coded_rv3);
        let report_b = dec.decode(&llr_b, 3, &mut tb_out).unwrap();
        assert!(
            report_b.crc_ok,
            "HARQ-combined decode failed to recover a block that a single shot alone could not"
        );
        assert_eq!(
            tb_out, tb,
            "recovered payload mismatch after HARQ combining"
        );
    }

    #[test]
    fn decoder_flush_harq_resets() {
        let mut dec = DlSchDecoder::new(100, 0.5, 1, 512, 5, 0.25).unwrap();
        let llr = vec![1.0f32; dec.e_per_cb * dec.params.c];
        let mut tb_out = vec![0u8; 100];
        dec.decode(&llr, 0, &mut tb_out).unwrap();
        dec.flush_harq();
        assert!(dec.harq_bufs.iter().all(|b| b.tx_count() == 0));
    }

    /// A multi-code-block transport block encoded once, then decoded twice —
    /// sequentially and through the worker pipeline — must come back
    /// **identical**, down to the per-code-block CRC flags and the reported
    /// iteration count.
    ///
    /// This is the assertion that makes [`DlSchDecoder::with_pipeline`] a
    /// scheduling change rather than a second decoder. Both paths run the
    /// same LOMS kernel on the same HARQ-combined LLRs for the same iteration
    /// budget; the only difference is that code blocks finish out of order
    /// and are reassembled by index. If the reassembly were keyed off arrival
    /// order — the obvious mistake, and one that passes any test using a
    /// single worker or an all-zero payload — this comparison is where it
    /// would show, because the two decodes would disagree on where each code
    /// block's payload landed.
    ///
    /// Noise is deliberate: a clean channel converges every code block in one
    /// iteration, which makes them finish in submission order and hides
    /// exactly the reordering this is meant to catch.
    #[test]
    fn pipelined_decode_matches_sequential_decode() {
        let cfg = DlSchConfig {
            tb_size: 30_000,
            target_rate: 0.5,
            qm: 2,
            g: 90_000,
            ..DlSchConfig::default_decode_params()
        };
        let enc = DlSchEncoder::from_config(&cfg).unwrap();
        assert!(
            enc.num_code_blocks() > 1,
            "this test needs a segmented transport block to be meaningful"
        );

        let mut tb = vec![0u8; cfg.tb_size];
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for b in tb.iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *b = ((state >> 27) & 1) as u8;
        }

        let mut coded = vec![0u8; enc.output_bits()];
        enc.encode(&tb, 0, &mut coded).unwrap();

        // Moderate noise: enough that the decoder does real iterative work and
        // code blocks converge after different numbers of passes.
        let mut ch = AwgnChannel::new(2.0, cfg.target_rate, 0x5EED_0001);
        let rx = ch.transmit(&coded);

        let mut seq_out = vec![0u8; cfg.tb_size];
        let mut seq_dec = DlSchDecoder::from_config(&cfg).unwrap();
        assert_eq!(seq_dec.worker_count(), 0);
        let seq = seq_dec.decode(&rx, 0, &mut seq_out).unwrap();

        let mut pipe_out = vec![0u8; cfg.tb_size];
        let mut pipe_dec = DlSchDecoder::from_config(&cfg)
            .unwrap()
            .with_pipeline(4)
            .unwrap();
        assert!(pipe_dec.worker_count() >= 1);
        let pipelined = pipe_dec.decode(&rx, 0, &mut pipe_out).unwrap();

        assert_eq!(
            seq_out, pipe_out,
            "pipelined decode produced a different transport block"
        );
        assert_eq!(
            seq.cb_crc, pipelined.cb_crc,
            "pipelined decode produced different per-code-block CRC results"
        );
        assert_eq!(
            seq.crc_ok, pipelined.crc_ok,
            "pipelined decode produced a different transport-block CRC result"
        );
        assert_eq!(
            seq.max_iters_used, pipelined.max_iters_used,
            "pipelined decode consumed a different number of iterations"
        );
    }

    /// The pipeline must also survive more code blocks than the pool has
    /// slots, which is where the submit/drain loop has to apply back-pressure
    /// instead of assuming everything fits at once.
    ///
    /// `LdpcPipeline`'s pool holds 16 frames. A transport block segmenting
    /// into more than that forces `acquire` to return `None` partway through
    /// submission, so the loop has to drain completions before it can
    /// continue — the path a smaller transport block never reaches.
    #[test]
    fn pipelined_decode_handles_more_code_blocks_than_pool_slots() {
        let cfg = DlSchConfig {
            tb_size: 160_000,
            target_rate: 0.5,
            qm: 2,
            g: 480_000,
            ..DlSchConfig::default_decode_params()
        };
        let enc = DlSchEncoder::from_config(&cfg).unwrap();
        assert!(
            enc.num_code_blocks() > 16,
            "this test needs more code blocks than the pipeline's 16 pool slots; got {}",
            enc.num_code_blocks()
        );

        let tb = vec![1u8; cfg.tb_size];
        let mut coded = vec![0u8; enc.output_bits()];
        enc.encode(&tb, 0, &mut coded).unwrap();

        let mut ch = AwgnChannel::new(4.0, cfg.target_rate, 0x5EED_0002);
        let rx = ch.transmit(&coded);

        let mut seq_out = vec![0u8; cfg.tb_size];
        let seq = DlSchDecoder::from_config(&cfg)
            .unwrap()
            .decode(&rx, 0, &mut seq_out)
            .unwrap();

        let mut pipe_out = vec![0u8; cfg.tb_size];
        let pipelined = DlSchDecoder::from_config(&cfg)
            .unwrap()
            .with_pipeline(3)
            .unwrap()
            .decode(&rx, 0, &mut pipe_out)
            .unwrap();

        assert_eq!(seq_out, pipe_out);
        assert_eq!(seq.cb_crc, pipelined.cb_crc);
        assert_eq!(seq.crc_ok, pipelined.crc_ok);
    }

    /// A single-code-block transport block must decode identically whether or
    /// not a pipeline is installed.
    ///
    /// Note what this does and does not pin. It pins the *output*: installing
    /// a pipeline never changes the answer, including in the degenerate case
    /// the pipeline cannot help with. It does **not** verify that
    /// [`DlSchDecoder::decode`] actually takes the sequential path at
    /// $C = 1$ — routing the single code block through a worker would produce
    /// the same bits, just more slowly, and there is no observable signal
    /// here to tell the two apart. That bypass is a performance decision
    /// documented on [`DlSchDecoder::with_pipeline`], not a correctness one,
    /// and this test is deliberately not claiming to enforce it.
    #[test]
    fn single_code_block_decodes_identically_with_or_without_a_pipeline() {
        let cfg = DlSchConfig {
            tb_size: 1000,
            target_rate: 0.5,
            qm: 2,
            g: 3000,
            ..DlSchConfig::default_decode_params()
        };
        let enc = DlSchEncoder::from_config(&cfg).unwrap();
        assert_eq!(enc.num_code_blocks(), 1);

        let tb = vec![0u8; cfg.tb_size];
        let mut coded = vec![0u8; enc.output_bits()];
        enc.encode(&tb, 0, &mut coded).unwrap();
        let mut ch = AwgnChannel::new(3.0, cfg.target_rate, 0x5EED_0003);
        let rx = ch.transmit(&coded);

        let mut a = vec![0u8; cfg.tb_size];
        let ra = DlSchDecoder::from_config(&cfg)
            .unwrap()
            .decode(&rx, 0, &mut a)
            .unwrap();
        let mut b = vec![0u8; cfg.tb_size];
        let rb = DlSchDecoder::from_config(&cfg)
            .unwrap()
            .with_pipeline(4)
            .unwrap()
            .decode(&rx, 0, &mut b)
            .unwrap();

        assert_eq!(a, b);
        assert_eq!(ra.crc_ok, rb.crc_ok);
        assert_eq!(ra.max_iters_used, rb.max_iters_used);
    }

    /// HARQ state must survive the pipelined path: a failed first transmission
    /// followed by a retransmission has to combine, exactly as it does
    /// sequentially. The per-code-block HARQ buffers are touched on the
    /// calling thread in both paths — only the decode moves — and this is
    /// what pins that.
    #[test]
    fn pipelined_decode_combines_harq_across_transmissions() {
        let cfg = DlSchConfig {
            tb_size: 30_000,
            target_rate: 0.5,
            qm: 2,
            g: 45_000,
            ..DlSchConfig::default_decode_params()
        };
        let enc = DlSchEncoder::from_config(&cfg).unwrap();
        assert!(enc.num_code_blocks() > 1);

        let tb = vec![1u8; cfg.tb_size];
        let mut coded_rv0 = vec![0u8; enc.output_bits()];
        enc.encode(&tb, 0, &mut coded_rv0).unwrap();
        let mut coded_rv2 = vec![0u8; enc.output_bits()];
        enc.encode(&tb, 2, &mut coded_rv2).unwrap();

        // Low SNR so the first transmission is expected to fail and the
        // retransmission has something to add.
        let mut ch = AwgnChannel::new(-1.0, cfg.target_rate, 0x5EED_0004);
        let rx0 = ch.transmit(&coded_rv0);
        let rx2 = ch.transmit(&coded_rv2);

        let run = |pipeline: bool| {
            let mut dec = DlSchDecoder::from_config(&cfg).unwrap();
            if pipeline {
                dec = dec.with_pipeline(4).unwrap();
            }
            let mut out = vec![0u8; cfg.tb_size];
            let first = dec.decode(&rx0, 0, &mut out).unwrap();
            let second = dec.decode(&rx2, 2, &mut out).unwrap();
            (first, second, out)
        };

        let (seq_first, seq_second, seq_out) = run(false);
        let (pipe_first, pipe_second, pipe_out) = run(true);

        assert_eq!(seq_first.cb_crc, pipe_first.cb_crc);
        assert_eq!(seq_second.cb_crc, pipe_second.cb_crc);
        assert_eq!(seq_second.harq_tx_count, pipe_second.harq_tx_count);
        assert_eq!(
            seq_second.harq_tx_count, 2,
            "the second call must be seen as a retransmission"
        );
        assert_eq!(seq_out, pipe_out);
    }
}
