//! syndrome: 5G NR Forward Error Correction library
//!
//! A protocol-aware FEC library implementing the 3GPP TS 38.212 transport block
//! processing chain: CRC attachment, code block segmentation, QC-LDPC encode/decode
//! (BG1/BG2, LOMS), rate matching, HARQ soft combining, Reed-Solomon erasure coding,
//! Viterbi convolutional decoding, Polar codes, LTE Turbo codes, BCH codes, and
//! the extended binary Golay code.
//!
//! Designed for zero-allocation hot paths, AVX2 (x86-64) and NEON (AArch64)
//! SIMD acceleration, and lock-free SPSC pipeline concurrency.

pub mod affinity;
pub mod bch;
pub mod bg_tables;
pub mod channel_sim;
pub mod crc;
pub mod error;
pub mod golay;
pub mod hamming;
pub mod harq;
pub mod ldpc_pipeline;
pub mod polar;
pub mod qc_ldpc;
pub mod quantize;
pub mod rate_matching;
pub mod reed_solomon;
pub mod segmentation;
pub mod sixg;
pub mod spsc_queue;
#[cfg(test)]
pub(crate) mod test_util;
pub mod transport_block;
pub mod turbo;
pub mod viterbi;
pub mod wifi;
pub mod wifi_ldpc_tables;
pub mod wifi_rate_matching;

// Architecture-specific SIMD kernels (pub(crate) — called from qc_ldpc only).
#[cfg(target_arch = "x86_64")]
pub(crate) mod simd_avx2;
#[cfg(target_arch = "aarch64")]
pub(crate) mod simd_neon;

pub use affinity::pin_to_core;
pub use bch::BchCode;
pub use bg_tables::*;
pub use crc::{Crc24, CrcKind};
pub use error::FecError;
pub use golay::GolayCode;
pub use hamming::{Hamming74, decode_hamming_7_4, encode_hamming_7_4};
pub use harq::HarqBuffer;
pub use ldpc_pipeline::{LdpcFrame, LdpcPipeline};
pub use polar::{PolarDecoder, PolarEncoder};
pub use qc_ldpc::{BaseGraph, QcLdpcDecoder, QcLdpcEncoder};
pub use quantize::{dequantize_llr, quantize_llr};
pub use rate_matching::{rate_dematch_llr, rate_match};
pub use reed_solomon::ReedSolomon;
pub use segmentation::{SegmentationParams, compute_segmentation, segment};
pub use spsc_queue::SpscRing;
pub use transport_block::{DecodeReport, DlSchDecoder, DlSchEncoder};
pub use turbo::{TurboDecoder, TurboEncoder};
pub use viterbi::ViterbiDecoder;
