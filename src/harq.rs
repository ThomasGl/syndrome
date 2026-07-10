//! HARQ soft-combining buffer for 5G NR incremental redundancy.
//!
//! Hybrid Automatic Repeat reQuest (HARQ) with incremental redundancy (IR)
//! allows a receiver to combine LLRs from multiple transmissions of the same
//! transport block (using different redundancy versions RV0–RV3) before
//! decoding.  Chase combining uses repeated RV0; IR uses different RVs each
//! time.
//!
//! # Combining Equation
//!
//! Let $L^{(t)}_j$ be the accumulated LLR for circular-buffer position $j$
//! after $t$ transmissions.  Each call to [`HarqBuffer::combine`] adds:
//!
//! $$L^{(t)}_j = L^{(t-1)}_j + l^{(t)}_j$$
//!
//! where $l^{(t)}_j$ is the de-rate-matched LLR from transmission $t$.
//! The combined buffer is then passed to the LDPC decoder.
//!
//! # Usage
//!
//! ```
//! use glezer_rsv::harq::HarqBuffer;
//! use glezer_rsv::qc_ldpc::BaseGraph;
//!
//! let mut buf = HarqBuffer::new(BaseGraph::Bg1, 2); // z=2
//! let e_llr = vec![1.0f32; 32];
//! buf.combine(&e_llr, 0, 1, 0).unwrap(); // RV=0, Qm=1
//!
//! // Check that some positions are non-zero after combining.
//! let cb = buf.llr_buffer();
//! assert!(cb.iter().any(|&v| v != 0.0));
//! ```

use crate::error::FecError;
use crate::qc_ldpc::BaseGraph;
use crate::rate_matching::rate_dematch_llr;

/// HARQ soft-combining accumulator for one code block.
///
/// Holds a $N_{cb}$-length f32 LLR buffer that accumulates soft information
/// across multiple HARQ transmissions.  Call [`HarqBuffer::flush`] to reset for the next
/// transport block.
pub struct HarqBuffer {
    bg: BaseGraph,
    z: usize,
    n_filler: usize,
    /// Accumulated LLR circular buffer, length $N_{cb} = n_b \cdot Z$.
    acc: Vec<f32>,
    /// Number of transmissions combined so far.
    tx_count: usize,
}

impl HarqBuffer {
    /// Create a new HARQ buffer for `bg` at lifting size `z`.
    ///
    /// # Arguments
    ///
    /// * `bg` - Base graph (determines $n_b$ and therefore $N_{cb}$).
    /// * `z`  - Lifting size.
    ///
    /// # Examples
    ///
    /// ```
    /// use glezer_rsv::harq::HarqBuffer;
    /// use glezer_rsv::qc_ldpc::BaseGraph;
    ///
    /// let buf = HarqBuffer::new(BaseGraph::Bg1, 2);
    /// assert_eq!(buf.ncb(), 66 * 2);
    /// ```
    pub fn new(bg: BaseGraph, z: usize) -> Self {
        let n_b: usize = match bg {
            BaseGraph::Bg1 => 66,
            BaseGraph::Bg2 => 50,
        };
        let ncb = n_b * z;
        Self {
            bg,
            z,
            n_filler: 0,
            acc: vec![0.0f32; ncb],
            tx_count: 0,
        }
    }

    /// Create a HARQ buffer with a known filler count (for segmented TBs).
    pub fn with_filler(bg: BaseGraph, z: usize, n_filler: usize) -> Self {
        let mut h = Self::new(bg, z);
        h.n_filler = n_filler;
        h
    }

    /// Circular buffer size $N_{cb} = n_b \cdot Z$.
    pub fn ncb(&self) -> usize {
        self.acc.len()
    }

    /// Number of transmissions combined so far.
    pub fn tx_count(&self) -> usize {
        self.tx_count
    }

    /// Add a new received LLR vector into the accumulation buffer.
    ///
    /// Internally calls [`rate_dematch_llr`] to scatter `e_llr` back to
    /// circular-buffer positions, then adds into the accumulator.
    ///
    /// # Arguments
    ///
    /// * `e_llr` - Received soft LLRs of length $E$ (one per coded bit).
    /// * `rv`    - Redundancy version used for this transmission (0..=3).
    /// * `qm`    - Modulation order ($Q_m$).
    /// * `n_filler_override` - Pass 0 to use the filler count set at construction,
    ///   or a non-zero value to override (useful for first-time setup).
    ///
    /// # Errors
    ///
    /// Propagates any error from [`rate_dematch_llr`].
    ///
    /// # Examples
    ///
    /// ```
    /// use glezer_rsv::harq::HarqBuffer;
    /// use glezer_rsv::qc_ldpc::BaseGraph;
    ///
    /// let mut buf = HarqBuffer::new(BaseGraph::Bg2, 4);
    /// let e = vec![0.5f32; 16];
    /// buf.combine(&e, 0, 1, 0).unwrap();
    /// buf.combine(&e, 2, 1, 0).unwrap(); // second transmission, RV=2 (IR)
    /// assert_eq!(buf.tx_count(), 2);
    /// ```
    pub fn combine(
        &mut self,
        e_llr: &[f32],
        rv: usize,
        qm: usize,
        n_filler_override: usize,
    ) -> Result<(), FecError> {
        let nf = if n_filler_override > 0 {
            n_filler_override
        } else {
            self.n_filler
        };
        rate_dematch_llr(e_llr, &mut self.acc, rv, qm, self.bg, self.z, nf)?;
        self.tx_count += 1;
        Ok(())
    }

    /// Return the accumulated LLR circular buffer (read-only view).
    ///
    /// Pass this to the LDPC decoder as the channel LLR input.  The first
    /// $2Z$ positions correspond to the punctured systematic bits and may be
    /// zero (erasures); the decoder handles this via [`crate::qc_ldpc::QcLdpcDecoder::decode_5g`].
    pub fn llr_buffer(&self) -> &[f32] {
        &self.acc
    }

    /// Copy the accumulated LLRs into `dst`.
    ///
    /// # Errors
    ///
    /// Returns [`FecError::BufferTooSmall`] if `dst` is shorter than `ncb`.
    pub fn copy_llr_into(&self, dst: &mut [f32]) -> Result<(), FecError> {
        if dst.len() < self.acc.len() {
            return Err(FecError::BufferTooSmall {
                required: self.acc.len(),
                provided: dst.len(),
            });
        }
        dst[..self.acc.len()].copy_from_slice(&self.acc);
        Ok(())
    }

    /// Reset the accumulator to zero (start of a new HARQ process).
    ///
    /// Call this after a successful decode (CRC pass) or after reaching the
    /// maximum retransmission count.
    pub fn flush(&mut self) {
        self.acc.iter_mut().for_each(|v| *v = 0.0);
        self.tx_count = 0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_buffer_all_zeros() {
        let buf = HarqBuffer::new(BaseGraph::Bg1, 2);
        assert_eq!(buf.ncb(), 66 * 2);
        assert!(buf.llr_buffer().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn combine_accumulates() {
        let mut buf = HarqBuffer::new(BaseGraph::Bg1, 2);
        let e = vec![1.0f32; 16];
        buf.combine(&e, 0, 1, 0).unwrap();
        let sum1: f32 = buf.llr_buffer().iter().sum();
        buf.combine(&e, 0, 1, 0).unwrap(); // Chase combine: same RV=0
        let sum2: f32 = buf.llr_buffer().iter().sum();
        assert!(sum2 > sum1, "second combine must increase total LLR energy");
        assert_eq!(buf.tx_count(), 2);
    }

    #[test]
    fn flush_resets_to_zero() {
        let mut buf = HarqBuffer::new(BaseGraph::Bg2, 4);
        buf.combine(&[2.0f32; 16], 0, 1, 0).unwrap();
        buf.flush();
        assert_eq!(buf.tx_count(), 0);
        assert!(buf.llr_buffer().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn copy_llr_into_error_on_short_buffer() {
        let buf = HarqBuffer::new(BaseGraph::Bg1, 2);
        let mut dst = vec![0.0f32; 1];
        assert!(buf.copy_llr_into(&mut dst).is_err());
    }

    /// FINDING 3/5 regression guard: `HarqBuffer::new(bg, 0)` followed by
    /// `combine` used to divide-by-zero inside `rate_dematch_llr` (`z == 0`
    /// makes `ncb == 0`). `rate_dematch_llr` now validates `z > 0` up front,
    /// and `combine` propagates that `Err` rather than panicking.
    #[test]
    fn combine_with_zero_lifting_size_returns_err_not_panic() {
        let mut buf = HarqBuffer::new(BaseGraph::Bg1, 0);
        let e = vec![1.0f32; 8];
        assert!(buf.combine(&e, 0, 1, 0).is_err());
    }
}
