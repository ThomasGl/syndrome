//! Crate-wide error type.
//!
//! New modules return [`FecError`]; legacy modules keep `&'static str` returns
//! and convert via [`From`] automatically.

use core::fmt;

/// Errors returned by the 5G NR FEC processing chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FecError {
    /// An input parameter is outside the valid 3GPP range.
    InvalidParam(&'static str),
    /// The code block or transport block CRC check failed.
    CrcMismatch,
    /// The LDPC decoder did not converge within the allowed iterations.
    DecoderNotConverged,
    /// A buffer provided by the caller is too small.
    BufferTooSmall { required: usize, provided: usize },
    /// A legacy string error (from pre-FecError code).
    Legacy(&'static str),
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
            FecError::Legacy(msg) => write!(f, "{msg}"),
        }
    }
}

impl From<&'static str> for FecError {
    fn from(s: &'static str) -> Self {
        FecError::Legacy(s)
    }
}
