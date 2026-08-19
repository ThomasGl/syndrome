//! C ABI for the 5G NR QC-LDPC encoder/decoder.
//!
//! Lets C, C++, or any language with a C FFI (Python `ctypes`/`cffi`, a
//! CMake project via `corrosion`, GNU Radio's `gr-fec` OOT module pattern)
//! call this crate's flagship codec without linking Rust. Every function
//! here is a thin wrapper: it converts raw pointers to the same flat slices
//! [`crate::qc_ldpc`] already takes, calls straight through, and converts
//! [`crate::error::FecError`] into an [`SyndromeStatus`] code. No new
//! algorithm lives in this file.
//!
//! # Scope
//!
//! Covers 5G NR QC-LDPC encode/decode ([`QcLdpcEncoder::encode_5g`],
//! [`QcLdpcDecoder::decode_5g`]) only — the crate's most complete, most
//! heavily tested path, and the one every piece of prior-art research
//! (GNU Radio's `gr-dvbs2rx`, `daniestevez/ldpc-toolbox`'s C-callable
//! staticlib) actually integrates through. The other 8 codecs in this
//! crate have no C entry point yet; extending this module to them is
//! straightforward (the pattern below is the same for each) but not done.
//!
//! # Buffer ownership
//!
//! Every buffer (LLR, scratch, hard-decision output, info bits, codeword)
//! is caller-allocated and caller-owned, exactly like the underlying Rust
//! API — this module performs no heap allocation beyond the decoder/encoder
//! handle itself. Use the `*_required_*` size-query functions to size
//! buffers correctly; passing an undersized buffer returns
//! [`SyndromeStatus::BufferTooSmall`] rather than reading or writing out of
//! bounds.
//!
//! # Error handling and panics
//!
//! Every function returns an `i32` [`SyndromeStatus`] rather than using
//! Rust's `Result`/`panic!` across the FFI boundary (a Rust panic unwinding
//! into C is undefined behavior). Each function body runs inside
//! [`std::panic::catch_unwind`]; an internal panic is caught and reported as
//! [`SyndromeStatus::Panic`] instead of aborting the caller's process.
//!
//! # Building the C-callable artifact
//!
//! This module only compiles with `--features capi`. The crate additionally
//! declares `staticlib`/`cdylib` crate-types unconditionally (see `[lib]` in
//! `Cargo.toml`), so a C project links against
//! `target/release/libsyndrome.{a,so}` after building with:
//!
//! ```text
//! cargo build --release --features capi
//! ```
//!
//! A minimal C header matching this module is not generated automatically
//! (no `cbindgen` dependency has been added — see `Cargo.toml`'s
//! keep-dependencies-light policy); the function signatures below are the
//! source of truth until one is.

use crate::error::FecError;
use crate::qc_ldpc::{BaseGraph, QcLdpcDecoder, QcLdpcEncoder};
use std::panic::{AssertUnwindSafe, catch_unwind};

/// Status code returned by every function in this module. Mirrors
/// [`FecError`]'s variants plus two FFI-specific ones.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SyndromeStatus {
    /// Success.
    Ok = 0,
    /// See [`FecError::InvalidParam`] (also returned for a null pointer or
    /// an out-of-range `base_graph` selector).
    InvalidParam = -1,
    /// See [`FecError::CrcMismatch`]. Never returned by the functions in
    /// this module today (5G LDPC encode/decode does not check a CRC
    /// itself), reserved for when this module's scope grows to
    /// [`crate::transport_block`].
    CrcMismatch = -2,
    /// See [`FecError::DecoderNotConverged`].
    DecoderNotConverged = -3,
    /// See [`FecError::BufferTooSmall`].
    BufferTooSmall = -4,
    /// A required output or handle pointer argument was null.
    NullPointer = -5,
    /// An internal panic was caught at the FFI boundary. Indicates a bug in
    /// this crate; please report it rather than working around it.
    Panic = -6,
}

impl From<FecError> for SyndromeStatus {
    fn from(e: FecError) -> Self {
        match e {
            FecError::InvalidParam(_) => SyndromeStatus::InvalidParam,
            FecError::CrcMismatch => SyndromeStatus::CrcMismatch,
            FecError::DecoderNotConverged => SyndromeStatus::DecoderNotConverged,
            FecError::BufferTooSmall { .. } => SyndromeStatus::BufferTooSmall,
        }
    }
}

fn base_graph_from_i32(bg: i32) -> Result<BaseGraph, SyndromeStatus> {
    match bg {
        0 => Ok(BaseGraph::Bg1),
        1 => Ok(BaseGraph::Bg2),
        _ => Err(SyndromeStatus::InvalidParam),
    }
}

/// Run `body`, converting an internal panic into [`SyndromeStatus::Panic`]
/// instead of unwinding across the FFI boundary (undefined behavior).
fn guard(body: impl FnOnce() -> SyndromeStatus) -> i32 {
    catch_unwind(AssertUnwindSafe(body)).unwrap_or(SyndromeStatus::Panic) as i32
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

/// Opaque handle to a [`QcLdpcDecoder`]. Never dereferenced by the caller;
/// only ever passed back into `syndrome_ldpc_decoder_*` functions, and freed
/// exactly once with [`syndrome_ldpc_decoder_destroy`].
pub struct SyndromeLdpcDecoder(QcLdpcDecoder);

/// Create a 5G NR QC-LDPC decoder for the given base graph and lifting size.
///
/// # Arguments
///
/// * `base_graph` — `0` for BG1, `1` for BG2.
/// * `z` — 3GPP lifting size (TS 38.212 Table 5.3.2-1); see
///   [`QcLdpcDecoder::with_lifting_size`] for the valid set.
/// * `offset_beta` — layered offset min-sum offset (`0.5` is this crate's
///   documented default).
/// * `out_decoder` — receives the created handle on success. On any
///   non-[`SyndromeStatus::Ok`] return, `*out_decoder` is set to null and
///   nothing needs freeing.
///
/// # Safety
///
/// `out_decoder` must be a valid, non-null, properly aligned pointer to a
/// `*mut SyndromeLdpcDecoder` the caller owns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn syndrome_ldpc_decoder_create(
    base_graph: i32,
    z: usize,
    offset_beta: f32,
    out_decoder: *mut *mut SyndromeLdpcDecoder,
) -> i32 {
    guard(|| {
        if out_decoder.is_null() {
            return SyndromeStatus::NullPointer;
        }
        // SAFETY: caller contract above; `out_decoder` is non-null and valid.
        unsafe {
            *out_decoder = std::ptr::null_mut();
        }
        let bg = match base_graph_from_i32(base_graph) {
            Ok(bg) => bg,
            Err(s) => return s,
        };
        match QcLdpcDecoder::with_lifting_size(bg, z, offset_beta) {
            Ok(dec) => {
                let boxed = Box::new(SyndromeLdpcDecoder(dec));
                // SAFETY: see above.
                unsafe {
                    *out_decoder = Box::into_raw(boxed);
                }
                SyndromeStatus::Ok
            }
            Err(e) => e.into(),
        }
    })
}

/// Free a decoder created by [`syndrome_ldpc_decoder_create`].
///
/// # Safety
///
/// `decoder` must be either null (a no-op) or a handle previously returned
/// by [`syndrome_ldpc_decoder_create`] and not already freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn syndrome_ldpc_decoder_destroy(decoder: *mut SyndromeLdpcDecoder) {
    if decoder.is_null() {
        return;
    }
    // SAFETY: caller contract above.
    let _ = unsafe { Box::from_raw(decoder) };
}

/// Number of LLR values ($N = n_b \cdot Z$) this decoder expects — see
/// [`QcLdpcDecoder::variable_node_count`].
///
/// # Safety
///
/// `decoder` must be a live handle from [`syndrome_ldpc_decoder_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn syndrome_ldpc_decoder_variable_node_count(
    decoder: *const SyndromeLdpcDecoder,
) -> usize {
    // SAFETY: caller contract above.
    unsafe { (*decoder).0.variable_node_count() }
}

/// Number of systematic info bits ($K = k_b \cdot Z$) this decoder expects —
/// see [`QcLdpcDecoder::info_bit_count_5g`].
///
/// # Safety
///
/// `decoder` must be a live handle from [`syndrome_ldpc_decoder_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn syndrome_ldpc_decoder_info_bit_count(
    decoder: *const SyndromeLdpcDecoder,
) -> usize {
    // SAFETY: caller contract above.
    unsafe { (*decoder).0.info_bit_count_5g() }
}

/// Required length of the `edge_r` scratch buffer for
/// [`syndrome_ldpc_decode_5g`] — see [`QcLdpcDecoder::required_edge_buffer`].
///
/// # Safety
///
/// `decoder` must be a live handle from [`syndrome_ldpc_decoder_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn syndrome_ldpc_decoder_required_edge_buffer(
    decoder: *const SyndromeLdpcDecoder,
) -> usize {
    // SAFETY: caller contract above.
    unsafe { (*decoder).0.required_edge_buffer() }
}

/// Required length of the `layer_scratch` buffer for
/// [`syndrome_ldpc_decode_5g`] — see
/// [`QcLdpcDecoder::required_layer_buffer`].
///
/// # Safety
///
/// `decoder` must be a live handle from [`syndrome_ldpc_decoder_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn syndrome_ldpc_decoder_required_layer_buffer(
    decoder: *const SyndromeLdpcDecoder,
) -> usize {
    // SAFETY: caller contract above.
    unsafe { (*decoder).0.required_layer_buffer() }
}

/// 5G NR-compliant LDPC decode (TS 38.212 §5.3.2) — see
/// [`QcLdpcDecoder::decode_5g`] for the algorithm and buffer layout this
/// wraps.
///
/// # Arguments
///
/// * `decoder` — handle from [`syndrome_ldpc_decoder_create`].
/// * `llr`, `llr_len` — channel LLR buffer, modified in place; `llr_len`
///   must equal [`syndrome_ldpc_decoder_variable_node_count`].
/// * `n_filler` — number of filler bits, see [`QcLdpcDecoder::decode_5g`].
/// * `edge_r`, `edge_r_len` — scratch buffer, at least
///   [`syndrome_ldpc_decoder_required_edge_buffer`] long.
/// * `layer_scratch`, `layer_scratch_len` — scratch buffer, at least
///   [`syndrome_ldpc_decoder_required_layer_buffer`] long.
/// * `hard_output`, `hard_output_len` — hard-decision output, `hard_output_len`
///   must equal `llr_len`.
/// * `iterations` — number of layered passes.
/// * `out_iterations_used` — if non-null, receives the number of iterations
///   actually run before convergence (or `iterations`, if it never
///   converged early) on [`SyndromeStatus::Ok`].
///
/// # Returns
///
/// [`SyndromeStatus::Ok`] on success, [`SyndromeStatus::BufferTooSmall`] if
/// any buffer is shorter than required, [`SyndromeStatus::DecoderNotConverged`]
/// if the layered min-sum recursion did not converge within `iterations`, or
/// [`SyndromeStatus::NullPointer`]/[`SyndromeStatus::InvalidParam`] as
/// documented on [`SyndromeStatus`].
///
/// # Safety
///
/// `decoder` must be a live handle. `llr`/`edge_r`/`layer_scratch`/
/// `hard_output` must each be valid for reads and writes (as applicable) for
/// their stated lengths, non-overlapping, and properly aligned for their
/// element type. `out_iterations_used`, if non-null, must be valid for a
/// single `usize` write.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn syndrome_ldpc_decode_5g(
    decoder: *mut SyndromeLdpcDecoder,
    llr: *mut f32,
    llr_len: usize,
    n_filler: usize,
    edge_r: *mut f32,
    edge_r_len: usize,
    layer_scratch: *mut f32,
    layer_scratch_len: usize,
    hard_output: *mut u8,
    hard_output_len: usize,
    iterations: usize,
    out_iterations_used: *mut usize,
) -> i32 {
    guard(|| {
        if decoder.is_null()
            || llr.is_null()
            || edge_r.is_null()
            || layer_scratch.is_null()
            || hard_output.is_null()
        {
            return SyndromeStatus::NullPointer;
        }
        // SAFETY: non-null checked above; remaining validity (length,
        // alignment, non-aliasing) is the caller's contract per this
        // function's doc.
        let dec = unsafe { &(*decoder).0 };
        let llr = unsafe { std::slice::from_raw_parts_mut(llr, llr_len) };
        let edge_r = unsafe { std::slice::from_raw_parts_mut(edge_r, edge_r_len) };
        let layer_scratch =
            unsafe { std::slice::from_raw_parts_mut(layer_scratch, layer_scratch_len) };
        let hard_output = unsafe { std::slice::from_raw_parts_mut(hard_output, hard_output_len) };

        match dec.decode_5g(
            llr,
            n_filler,
            edge_r,
            layer_scratch,
            hard_output,
            iterations,
        ) {
            Ok(used) => {
                if !out_iterations_used.is_null() {
                    // SAFETY: non-null checked; validity per this function's doc.
                    unsafe {
                        *out_iterations_used = used;
                    }
                }
                SyndromeStatus::Ok
            }
            Err(e) => e.into(),
        }
    })
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/// Opaque handle to a [`QcLdpcEncoder`]. Never dereferenced by the caller;
/// only ever passed back into `syndrome_ldpc_encoder_*` functions, and freed
/// exactly once with [`syndrome_ldpc_encoder_destroy`].
pub struct SyndromeLdpcEncoder(QcLdpcEncoder);

/// Create a 5G NR QC-LDPC encoder for the given base graph and lifting size.
///
/// # Arguments
///
/// * `base_graph` — `0` for BG1, `1` for BG2.
/// * `z` — 3GPP lifting size (TS 38.212 Table 5.3.2-1).
/// * `out_encoder` — receives the created handle on success. On any
///   non-[`SyndromeStatus::Ok`] return, `*out_encoder` is set to null and
///   nothing needs freeing.
///
/// # Safety
///
/// `out_encoder` must be a valid, non-null, properly aligned pointer to a
/// `*mut SyndromeLdpcEncoder` the caller owns.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn syndrome_ldpc_encoder_create(
    base_graph: i32,
    z: usize,
    out_encoder: *mut *mut SyndromeLdpcEncoder,
) -> i32 {
    guard(|| {
        if out_encoder.is_null() {
            return SyndromeStatus::NullPointer;
        }
        // SAFETY: caller contract above.
        unsafe {
            *out_encoder = std::ptr::null_mut();
        }
        let bg = match base_graph_from_i32(base_graph) {
            Ok(bg) => bg,
            Err(s) => return s,
        };
        match QcLdpcEncoder::new(bg, z) {
            Ok(enc) => {
                let boxed = Box::new(SyndromeLdpcEncoder(enc));
                // SAFETY: see above.
                unsafe {
                    *out_encoder = Box::into_raw(boxed);
                }
                SyndromeStatus::Ok
            }
            Err(e) => e.into(),
        }
    })
}

/// Free an encoder created by [`syndrome_ldpc_encoder_create`].
///
/// # Safety
///
/// `encoder` must be either null (a no-op) or a handle previously returned
/// by [`syndrome_ldpc_encoder_create`] and not already freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn syndrome_ldpc_encoder_destroy(encoder: *mut SyndromeLdpcEncoder) {
    if encoder.is_null() {
        return;
    }
    // SAFETY: caller contract above.
    let _ = unsafe { Box::from_raw(encoder) };
}

/// Number of systematic info bits ($K = k_b \cdot Z$) this encoder expects —
/// see [`QcLdpcEncoder::info_bit_count`].
///
/// # Safety
///
/// `encoder` must be a live handle from [`syndrome_ldpc_encoder_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn syndrome_ldpc_encoder_info_bit_count(
    encoder: *const SyndromeLdpcEncoder,
) -> usize {
    // SAFETY: caller contract above.
    unsafe { (*encoder).0.info_bit_count() }
}

/// Number of codeword bits ($N = n_b \cdot Z$) this encoder produces — see
/// [`QcLdpcEncoder::codeword_bit_count`].
///
/// # Safety
///
/// `encoder` must be a live handle from [`syndrome_ldpc_encoder_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn syndrome_ldpc_encoder_codeword_bit_count(
    encoder: *const SyndromeLdpcEncoder,
) -> usize {
    // SAFETY: caller contract above.
    unsafe { (*encoder).0.codeword_bit_count() }
}

/// 5G NR-compliant LDPC encode (TS 38.212 §5.3.2) — see
/// [`QcLdpcEncoder::encode_5g`].
///
/// # Arguments
///
/// * `encoder` — handle from [`syndrome_ldpc_encoder_create`].
/// * `info_bits`, `info_len` — systematic input bits; `info_len` must equal
///   [`syndrome_ldpc_encoder_info_bit_count`] minus `n_filler`.
/// * `n_filler` — number of filler bits.
/// * `codeword`, `codeword_len` — output buffer; `codeword_len` must equal
///   [`syndrome_ldpc_encoder_codeword_bit_count`].
///
/// # Safety
///
/// `encoder` must be a live handle. `info_bits` must be valid for reads for
/// `info_len` elements; `codeword` must be valid for writes for
/// `codeword_len` elements; the two must not overlap.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn syndrome_ldpc_encode_5g(
    encoder: *const SyndromeLdpcEncoder,
    info_bits: *const u8,
    info_len: usize,
    n_filler: usize,
    codeword: *mut u8,
    codeword_len: usize,
) -> i32 {
    guard(|| {
        if encoder.is_null() || info_bits.is_null() || codeword.is_null() {
            return SyndromeStatus::NullPointer;
        }
        // SAFETY: non-null checked above; remaining validity per this
        // function's doc.
        let enc = unsafe { &(*encoder).0 };
        let info_bits = unsafe { std::slice::from_raw_parts(info_bits, info_len) };
        let codeword = unsafe { std::slice::from_raw_parts_mut(codeword, codeword_len) };

        match enc.encode_5g(info_bits, n_filler, codeword) {
            Ok(()) => SyndromeStatus::Ok,
            Err(e) => e.into(),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end round trip through the C ABI only -- no direct call into
    /// `QcLdpcEncoder`/`QcLdpcDecoder` -- exercising create/encode/decode/
    /// destroy for both handles exactly as a C caller would.
    #[test]
    fn round_trip_through_c_abi() {
        unsafe {
            let mut encoder: *mut SyndromeLdpcEncoder = std::ptr::null_mut();
            let status = syndrome_ldpc_encoder_create(0, 4, &mut encoder);
            assert_eq!(status, SyndromeStatus::Ok as i32);
            assert!(!encoder.is_null());

            let mut decoder: *mut SyndromeLdpcDecoder = std::ptr::null_mut();
            let status = syndrome_ldpc_decoder_create(0, 4, 0.5, &mut decoder);
            assert_eq!(status, SyndromeStatus::Ok as i32);
            assert!(!decoder.is_null());

            let k = syndrome_ldpc_encoder_info_bit_count(encoder);
            let n = syndrome_ldpc_encoder_codeword_bit_count(encoder);
            assert_eq!(n, syndrome_ldpc_decoder_variable_node_count(decoder));
            assert_eq!(k, syndrome_ldpc_decoder_info_bit_count(decoder));

            let n_filler = 0usize;
            let info: Vec<u8> = (0..k).map(|i| (i % 3 == 0) as u8).collect();
            let mut codeword = vec![0u8; n];
            let status = syndrome_ldpc_encode_5g(
                encoder,
                info.as_ptr(),
                info.len(),
                n_filler,
                codeword.as_mut_ptr(),
                codeword.len(),
            );
            assert_eq!(status, SyndromeStatus::Ok as i32);

            // Strong noiseless LLRs from the codeword: 0 -> +5.0, 1 -> -5.0.
            let mut llr: Vec<f32> = codeword
                .iter()
                .map(|&b| if b == 0 { 5.0 } else { -5.0 })
                .collect();
            let edge_len = syndrome_ldpc_decoder_required_edge_buffer(decoder);
            let layer_len = syndrome_ldpc_decoder_required_layer_buffer(decoder);
            let mut edge_r = vec![0.0f32; edge_len];
            let mut layer_scratch = vec![0.0f32; layer_len];
            let mut hard = vec![0u8; n];
            let mut iters_used = 0usize;

            let status = syndrome_ldpc_decode_5g(
                decoder,
                llr.as_mut_ptr(),
                llr.len(),
                n_filler,
                edge_r.as_mut_ptr(),
                edge_r.len(),
                layer_scratch.as_mut_ptr(),
                layer_scratch.len(),
                hard.as_mut_ptr(),
                hard.len(),
                10,
                &mut iters_used,
            );
            assert_eq!(status, SyndromeStatus::Ok as i32);
            assert_eq!(&hard[..k], &info[..], "decoded info bits must match input");

            syndrome_ldpc_encoder_destroy(encoder);
            syndrome_ldpc_decoder_destroy(decoder);
        }
    }

    /// `guard` exists specifically so an internal panic can never unwind
    /// across the `extern "C"` boundary (undefined behavior). Pin that it
    /// actually catches one, rather than just compiling.
    #[test]
    fn guard_catches_panic_instead_of_unwinding() {
        // Suppress the default panic-hook backtrace print for this
        // deliberately-panicking closure -- it's expected, not a failure.
        let prev_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let status = guard(|| panic!("deliberate test panic"));
        std::panic::set_hook(prev_hook);
        assert_eq!(status, SyndromeStatus::Panic as i32);
    }

    #[test]
    fn null_out_pointer_is_rejected() {
        let status = unsafe { syndrome_ldpc_encoder_create(0, 4, std::ptr::null_mut()) };
        assert_eq!(status, SyndromeStatus::NullPointer as i32);
    }

    #[test]
    fn null_handle_is_rejected() {
        let status = unsafe {
            syndrome_ldpc_encode_5g(
                std::ptr::null(),
                std::ptr::null(),
                0,
                0,
                std::ptr::null_mut(),
                0,
            )
        };
        assert_eq!(status, SyndromeStatus::NullPointer as i32);
    }

    #[test]
    fn invalid_base_graph_selector_is_rejected() {
        let mut encoder: *mut SyndromeLdpcEncoder = std::ptr::null_mut();
        let status = unsafe { syndrome_ldpc_encoder_create(2, 4, &mut encoder) };
        assert_eq!(status, SyndromeStatus::InvalidParam as i32);
        assert!(encoder.is_null());
    }

    #[test]
    fn undersized_buffer_is_reported_not_ub() {
        unsafe {
            let mut encoder: *mut SyndromeLdpcEncoder = std::ptr::null_mut();
            syndrome_ldpc_encoder_create(0, 4, &mut encoder);
            let k = syndrome_ldpc_encoder_info_bit_count(encoder);
            let n = syndrome_ldpc_encoder_codeword_bit_count(encoder);
            let info = vec![0u8; k];
            let mut short_codeword = vec![0u8; n - 1]; // deliberately undersized
            let status = syndrome_ldpc_encode_5g(
                encoder,
                info.as_ptr(),
                info.len(),
                0,
                short_codeword.as_mut_ptr(),
                short_codeword.len(),
            );
            assert_eq!(status, SyndromeStatus::BufferTooSmall as i32);
            syndrome_ldpc_encoder_destroy(encoder);
        }
    }
}
