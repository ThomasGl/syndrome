//! syndrome: 5G NR Forward Error Correction library
//!
//! A protocol-aware FEC library implementing the 3GPP TS 38.212 transport block
//! processing chain: CRC attachment, code block segmentation, QC-LDPC encode/decode
//! (BG1/BG2, LOMS), rate matching, HARQ soft combining, Reed-Solomon erasure and
//! errors-and-erasures coding, Viterbi convolutional decoding (zero-terminated and
//! tail-biting), Polar codes, LTE Turbo codes, BCH codes, the
//! extended binary Golay code, IEEE 802.11 Wi-Fi LDPC with shortening and
//! puncturing, and the Bluetooth FEC profiles (LE Coded PHY, BR/EDR).
//!
//! The QC-LDPC decoder has two number formats: an `f32` path with AVX2 and
//! NEON kernels, and a fixed-point path carrying `i8` messages in a 16-bit
//! posterior, with an AVX2 kernel that works 32 lanes at a time. There is no
//! NEON kernel for the fixed-point path — on AArch64 it runs its scalar
//! reference. `tests/ldpc_int8_quantization_loss.rs` measures what the
//! fixed-point format costs in error-rate terms.
//!
//! Designed for zero-allocation hot paths, AVX2 (x86-64) and NEON (AArch64)
//! SIMD acceleration, and lock-free SPSC pipeline concurrency.

/// This crate's version, captured at compile time from `Cargo.toml`.
///
/// Exists so downstream wrappers (for example the Python binding) can report
/// which version of the core library they were compiled against, rather than
/// their own version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod affinity;
pub mod bch;
pub mod bg_tables;
pub mod bits;
pub mod bluetooth;
pub mod channel_sim;
pub mod crc;
pub mod error;
pub mod golay;
pub mod hamming;
pub mod harq;
pub mod ldpc_pipeline;
pub mod montecarlo;
pub mod polar;
pub mod qc_ldpc;
pub mod quantize;
pub mod rate_matching;
pub mod reed_solomon;
pub mod segmentation;
pub mod sixg;
pub mod spsc_queue;
pub(crate) mod sync_shim;
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
pub use bits::{bits_to_bytes, bytes_to_bits, hard_decision};
pub use crc::{Crc24, CrcKind};
pub use error::FecError;
pub use golay::GolayCode;
pub use hamming::{Hamming74, decode_hamming_7_4, encode_hamming_7_4};
pub use harq::HarqBuffer;
pub use ldpc_pipeline::{LdpcFrame, LdpcPipeline};
pub use polar::{AdaptiveDecodeReport, PolarDecoder, PolarEncoder};
pub use qc_ldpc::{BaseGraph, LdpcWorkspace, QcLdpcDecoder, QcLdpcEncoder};
pub use quantize::{QuantParams, dequantize_llr, quantize_llr, quantize_llr_i16};
pub use rate_matching::{rate_dematch_llr, rate_match};
pub use reed_solomon::ReedSolomon;
pub use segmentation::{SegmentationParams, compute_segmentation, segment};
pub use spsc_queue::SpscRing;
pub use transport_block::{DecodeReport, DlSchConfig, DlSchDecoder, DlSchEncoder};
pub use turbo::{TurboDecoder, TurboEncoder};
pub use viterbi::ViterbiDecoder;
