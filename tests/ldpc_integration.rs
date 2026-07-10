//! Integration tests for the QC-LDPC encoder + decoder pipeline.
//!
//! These tests use real 3GPP BG1/BG2 parameters and verify the
//! encode→decode round-trip under controlled channel conditions.

use syndrome::{BaseGraph, QcLdpcDecoder, QcLdpcEncoder};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert a codeword (bits 0/1) into channel LLRs.
///
/// LLR convention: positive → bit-0 likely, negative → bit-1 likely.
/// `llr_magnitude` controls the channel reliability (higher = more reliable).
fn bits_to_llrs(codeword: &[u8], llr_magnitude: f32) -> Vec<f32> {
    codeword
        .iter()
        .map(|&b| {
            if b == 0 {
                llr_magnitude
            } else {
                -llr_magnitude
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Test 1: Encode → Decode round-trip at high SNR (no errors)
// ---------------------------------------------------------------------------

/// A codeword under strong LLRs (±5.0) should decode back to the original
/// codeword without any bit errors.
#[test]
fn encode_decode_roundtrip_bg1_no_errors() {
    let z = 16usize; // small but valid (iLS=0), fast test
    let enc = QcLdpcEncoder::new(BaseGraph::Bg1, z).unwrap();
    let dec = QcLdpcDecoder::with_lifting_size(BaseGraph::Bg1, z, 0.25).unwrap();

    let k = enc.info_bit_count();
    let n = enc.codeword_bit_count();
    assert_eq!(n, dec.variable_node_count());

    // Use a deterministic non-trivial info pattern (alternating 0/1).
    let info: Vec<u8> = (0..k).map(|i| (i % 2) as u8).collect();
    let mut codeword = vec![0u8; n];
    enc.encode(&info, &mut codeword)
        .expect("encode must succeed");

    // Verify systematic part matches.
    assert_eq!(
        &codeword[..k],
        info.as_slice(),
        "systematic bits must be preserved"
    );

    // Build channel LLRs at high reliability.
    let mut llr = bits_to_llrs(&codeword, 5.0);

    let mut edge_r = vec![0.0f32; dec.required_edge_buffer()];
    let mut scratch = vec![0.0f32; dec.required_layer_buffer()];
    let mut hard = vec![0u8; n];

    dec.decode_layered_offset_min_sum(&mut llr, &mut edge_r, &mut scratch, &mut hard, 10)
        .expect("decode must succeed");

    // At high SNR the decoder should recover the exact codeword.
    let errors: usize = codeword
        .iter()
        .zip(hard.iter())
        .filter(|&(&c, &h)| c != h)
        .count();
    assert_eq!(errors, 0, "Expected 0 bit errors at high SNR, got {errors}");
}

/// Same test for BG2.
#[test]
fn encode_decode_roundtrip_bg2_no_errors() {
    let z = 16usize; // iLS=0
    let enc = QcLdpcEncoder::new(BaseGraph::Bg2, z).unwrap();
    let dec = QcLdpcDecoder::with_lifting_size(BaseGraph::Bg2, z, 0.25).unwrap();

    let k = enc.info_bit_count();
    let n = enc.codeword_bit_count();

    let info: Vec<u8> = (0..k).map(|i| (i % 3 == 0) as u8).collect();
    let mut codeword = vec![0u8; n];
    enc.encode(&info, &mut codeword)
        .expect("encode must succeed");

    let mut llr = bits_to_llrs(&codeword, 5.0);
    let mut edge_r = vec![0.0f32; dec.required_edge_buffer()];
    let mut scratch = vec![0.0f32; dec.required_layer_buffer()];
    let mut hard = vec![0u8; n];

    dec.decode_layered_offset_min_sum(&mut llr, &mut edge_r, &mut scratch, &mut hard, 10)
        .expect("decode must succeed");

    let errors: usize = codeword
        .iter()
        .zip(hard.iter())
        .filter(|&(&c, &h)| c != h)
        .count();
    assert_eq!(errors, 0, "Expected 0 bit errors at high SNR, got {errors}");
}

// ---------------------------------------------------------------------------
// Test 2: Encode → 1-bit-flip → Decode (error correction)
// ---------------------------------------------------------------------------

/// Flip one LLR to the wrong sign; the decoder should correct it within
/// 10 iterations at a moderate SNR.  We flip a systematic-bit LLR (position
/// k/2) so the test is reproducible regardless of parity structure.
#[test]
fn encode_decode_one_bit_error_corrected_bg1() {
    let z = 16usize;
    let enc = QcLdpcEncoder::new(BaseGraph::Bg1, z).unwrap();
    let dec = QcLdpcDecoder::with_lifting_size(BaseGraph::Bg1, z, 0.25).unwrap();

    let k = enc.info_bit_count();
    let n = enc.codeword_bit_count();

    let info: Vec<u8> = (0..k).map(|i| (i % 2) as u8).collect();
    let mut codeword = vec![0u8; n];
    enc.encode(&info, &mut codeword)
        .expect("encode must succeed");

    // Build LLRs then flip one.
    let mut llr = bits_to_llrs(&codeword, 5.0);
    let flip_pos = k / 2;
    llr[flip_pos] = -llr[flip_pos]; // flip to wrong sign

    let mut edge_r = vec![0.0f32; dec.required_edge_buffer()];
    let mut scratch = vec![0.0f32; dec.required_layer_buffer()];
    let mut hard = vec![0u8; n];

    dec.decode_layered_offset_min_sum(&mut llr, &mut edge_r, &mut scratch, &mut hard, 10)
        .expect("decode must succeed");

    let errors: usize = codeword
        .iter()
        .zip(hard.iter())
        .filter(|&(&c, &h)| c != h)
        .count();
    assert_eq!(
        errors, 0,
        "Expected 1-bit error to be corrected, {errors} errors remain"
    );
}

/// Same 1-bit-flip test for BG2.
#[test]
fn encode_decode_one_bit_error_corrected_bg2() {
    let z = 16usize;
    let enc = QcLdpcEncoder::new(BaseGraph::Bg2, z).unwrap();
    let dec = QcLdpcDecoder::with_lifting_size(BaseGraph::Bg2, z, 0.25).unwrap();

    let k = enc.info_bit_count();
    let n = enc.codeword_bit_count();

    let info: Vec<u8> = (0..k).map(|i| (i % 3 == 0) as u8).collect();
    let mut codeword = vec![0u8; n];
    enc.encode(&info, &mut codeword)
        .expect("encode must succeed");

    let mut llr = bits_to_llrs(&codeword, 5.0);
    let flip_pos = k / 2;
    llr[flip_pos] = -llr[flip_pos];

    let mut edge_r = vec![0.0f32; dec.required_edge_buffer()];
    let mut scratch = vec![0.0f32; dec.required_layer_buffer()];
    let mut hard = vec![0u8; n];

    dec.decode_layered_offset_min_sum(&mut llr, &mut edge_r, &mut scratch, &mut hard, 10)
        .expect("decode must succeed");

    let errors: usize = codeword
        .iter()
        .zip(hard.iter())
        .filter(|&(&c, &h)| c != h)
        .count();
    assert_eq!(
        errors, 0,
        "Expected 1-bit error to be corrected, {errors} errors remain"
    );
}

// ---------------------------------------------------------------------------
// Test 3: Dimension checks for chosen Z
// ---------------------------------------------------------------------------

/// Verify variable_node_count() and check_node_count() match expected N, M
/// for the lifting sizes used in the other tests.
#[test]
fn dimension_checks_bg1_z16() {
    let z = 16usize;
    let dec = QcLdpcDecoder::with_lifting_size(BaseGraph::Bg1, z, 0.25).unwrap();
    // BG1: 68 col blocks, 46 row blocks
    assert_eq!(dec.variable_node_count(), 68 * z, "N = 68*Z for BG1");
    assert_eq!(dec.check_node_count(), 46 * z, "M = 46*Z for BG1");
}

#[test]
fn dimension_checks_bg2_z16() {
    let z = 16usize;
    let dec = QcLdpcDecoder::with_lifting_size(BaseGraph::Bg2, z, 0.25).unwrap();
    // BG2: 52 col blocks, 42 row blocks
    assert_eq!(dec.variable_node_count(), 52 * z, "N = 52*Z for BG2");
    assert_eq!(dec.check_node_count(), 42 * z, "M = 42*Z for BG2");
}

/// Dimension check at the default (largest) lifting sizes.
#[test]
fn dimension_checks_default_lifting() {
    let dec_bg1 = QcLdpcDecoder::new(BaseGraph::Bg1, 0.25);
    assert_eq!(
        dec_bg1.variable_node_count(),
        68 * 384,
        "BG1 default N = 68*384"
    );
    assert_eq!(
        dec_bg1.check_node_count(),
        46 * 384,
        "BG1 default M = 46*384"
    );

    let dec_bg2 = QcLdpcDecoder::new(BaseGraph::Bg2, 0.25);
    assert_eq!(
        dec_bg2.variable_node_count(),
        52 * 128,
        "BG2 default N = 52*128"
    );
    assert_eq!(
        dec_bg2.check_node_count(),
        42 * 128,
        "BG2 default M = 42*128"
    );
}
