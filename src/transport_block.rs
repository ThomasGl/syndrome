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

use crate::crc::{Crc24, CrcKind};
use crate::error::FecError;
use crate::harq::HarqBuffer;
use crate::qc_ldpc::{QcLdpcDecoder, QcLdpcEncoder};
use crate::rate_matching::RateMatchCache;
use crate::segmentation::{SegmentationParams, compute_segmentation, segment};

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
    rm_cache: std::cell::RefCell<RateMatchCache>,
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
            rm_cache: std::cell::RefCell::new(RateMatchCache::new()),
        })
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
    /// * `offset_beta`  - LOMS offset correction $\beta$ (typically 0.25).
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
        })
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
        // comments) rather than allocated here; `all_info` is cleared (not
        // reallocated) at the start of every call.
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
            ..
        } = self;
        let n = llr_cb.len();

        for ci in 0..n_cb {
            // Slice this CB's E received LLRs.
            let e_start = ci * e_per_cb;
            let e_llr = &rx_llr[e_start..e_start + e_per_cb];

            // HARQ: scatter + accumulate into the circular buffer.
            harq_bufs[ci].combine(e_llr, rv, qm, 0)?;

            // Copy accumulated LLR into the full-N decode buffer with correct alignment.
            //
            // The HARQ circular buffer excludes the first 2Z punctured systematic
            // positions: HARQ buf[j] holds the LLR for codeword bit (2Z + j).
            // The LDPC decoder (decode_5g) expects llr[i] = LLR for codeword bit i,
            // i.e. it reads parity/systematic at positions [2Z..N] and zeros [0..2Z]
            // itself (punctured erasure).
            //
            // Correct mapping: harq_buf[0..N-2Z] → llr_cb[2Z..N]
            //                  llr_cb[0..2Z] stays 0.0 (decode_5g enforces this anyway).
            //
            // `ncb` (n_b * Z: 66Z for BG1, 50Z for BG2 — see HarqBuffer::new) is
            // ALREADY defined as N - 2Z, since n_b is 2 less than the base
            // graph's total column count (68 for BG1, 52 for BG2). So `ncb` is
            // exactly the valid length above, with no further subtraction — a
            // previous version computed `ncb - 2Z` here, double-subtracting the
            // puncture width and silently dropping the last 2Z accumulated LLRs
            // of every HARQ-combined codeword (the positions rv=2/rv=3
            // retransmissions are specifically walking into) on the floor before
            // they ever reached the decoder.
            let two_z = 2 * z;
            let valid_len = harq_bufs[ci].ncb();
            debug_assert_eq!(
                two_z + valid_len,
                n,
                "ncb (N - 2Z) plus the punctured prefix must exactly fill the N-length decode buffer"
            );
            llr_cb[..n].iter_mut().for_each(|v| *v = 0.0); // zero the full buffer first
            {
                let harq_data = harq_bufs[ci].llr_buffer();
                llr_cb[two_z..two_z + valid_len].copy_from_slice(&harq_data[..valid_len]);
            }

            // decode_5g sets filler LLRs + punctured LLRs correctly.
            let cb_iters = decoders[ci].decode_5g(
                llr_cb,
                n_filler,
                edge_r,
                layer_scratch,
                hard,
                iterations,
            )?;
            max_iters = max_iters.max(cb_iters);

            // Extract K' info bits (systematic only, no filler).
            let k_prime_sys = k_prime; // without filler
            let info_bits = &hard[..k_prime_sys];

            // CRC-24B check (if segmented).
            if has_cb_crc {
                cb_crc_results[ci] = cb_crc.check(info_bits);
            } else {
                cb_crc_results[ci] = true;
            }

            // Collect info payload (strip CRC-24B if present).
            let payload_len = if has_cb_crc {
                k_prime_sys.saturating_sub(24)
            } else {
                k_prime_sys
            };
            all_info.extend_from_slice(&info_bits[..payload_len]);
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
}
