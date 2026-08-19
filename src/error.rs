//! Crate-wide error type.
//!
//! Every fallible public API in this crate returns [`FecError`] — there is a
//! single error type for the whole processing chain, no per-module
//! `&'static str` returns.

use core::fmt;

/// Errors returned by the 5G NR FEC processing chain.
///
/// This is the single error type returned by every fallible public API in
/// `syndrome`. It implements [`core::error::Error`], so it composes with
/// `anyhow`/`Box<dyn Error>` via `?`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FecError {
    /// An input parameter is outside the valid range (e.g. an invalid 3GPP
    /// lifting size, a precondition that was not met such as forgetting to
    /// call `precompute_mul_tables()`, or an internally-inconsistent shape).
    InvalidParam(&'static str),
    /// The code block or transport block CRC check failed.
    CrcMismatch,
    /// The LDPC decoder did not converge within the allowed iterations.
    DecoderNotConverged,
    /// A buffer provided by the caller has the wrong size (too small, or not
    /// exactly the required length).
    BufferTooSmall { required: usize, provided: usize },
}

impl fmt::Display for FecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FecError::InvalidParam(msg) => write!(f, "invalid parameter: {msg}"),
            FecError::CrcMismatch => write!(f, "CRC check failed"),
            FecError::DecoderNotConverged => write!(f, "LDPC decoder did not converge"),
            FecError::BufferTooSmall { required, provided } => write!(
                f,
                "buffer too small: required {required} bytes, got {provided}"
            ),
        }
    }
}

// `core::error::Error` (stable since 1.81) rather than `std::error::Error`:
// the latter is just a re-export of the former, so this is identical under
// `std` and additionally compiles under the `no_std` feature.
impl core::error::Error for FecError {}
