//! Integration tests for the real IEEE 802.11 LDPC encode + decode pipeline.
//!
//! These tests use the actual 802.11 Annex R/F shift matrices (see
//! `src/wifi_ldpc_tables.rs` for sourcing) via
//! `syndrome::wifi_ldpc_tables::{wifi_ldpc_encoder, wifi_ldpc_decoder}` and
//! verify the encode -> AWGN -> decode round-trip for a **full, unshortened,
//! unpunctured codeword** ($K$ info bits exactly fill $N = 24Z$ coded bits).
//!
//! Shortening and puncturing (802.11 rate-matching for a specific MCS) are
//! out of scope for this pass — see the module docs on
//! `syndrome::wifi_ldpc_tables` and the CHANGELOG.

use syndrome::channel_sim::AwgnChannel;
use syndrome::wifi_ldpc_tables::{wifi_ldpc_decoder, wifi_ldpc_encoder};

/// The 12 supported (Z, rate) combinations, mirroring
/// `wifi_ldpc_tables::WIFI_LDPC_MATRICES`.
const ALL_12: [(usize, usize, usize); 12] = [
    (27, 1, 2),
    (27, 2, 3),
    (27, 3, 4),
    (27, 5, 6),
    (54, 1, 2),
    (54, 2, 3),
    (54, 3, 4),
    (54, 5, 6),
    (81, 1, 2),
    (81, 2, 3),
    (81, 3, 4),
    (81, 5, 6),
];

/// Encode a deterministic info pattern, push it through the AWGN channel at
/// a high-reliability SNR, decode, and assert a bit-exact codeword
/// recovery. Run for all 12 (Z, rate) combinations.
#[test]
fn encode_awgn_decode_roundtrip_all_12_combinations() {
    for (z, rn, rd) in ALL_12 {
        let enc = wifi_ldpc_encoder(z, rn, rd)
            .unwrap_or_else(|e| panic!("encoder build failed for Z={z} rate={rn}/{rd}: {e}"));
        let dec = wifi_ldpc_decoder(z, rn, rd, 0.25)
            .unwrap_or_else(|e| panic!("decoder build failed for Z={z} rate={rn}/{rd}: {e}"));

        let k = enc.info_bit_count();
        let n = enc.codeword_bit_count();
        assert_eq!(n, 24 * z, "N = 24Z for Z={z} rate={rn}/{rd}");
        assert_eq!(n, dec.variable_node_count());

        // Deterministic non-trivial info pattern (alternating 0/1).
        let info: Vec<u8> = (0..k).map(|i| (i % 2) as u8).collect();
        let mut codeword = vec![0u8; n];
        enc.encode(&info, &mut codeword)
            .unwrap_or_else(|e| panic!("encode failed for Z={z} rate={rn}/{rd}: {e}"));

        assert_eq!(
            &codeword[..k],
            info.as_slice(),
            "systematic bits must be preserved (Z={z} rate={rn}/{rd})"
        );

        // High Eb/No: the decoder should recover the exact codeword.
        let rate = rn as f32 / rd as f32;
        let mut ch = AwgnChannel::new(6.0, rate, 0xC0FFEE ^ (z as u64) ^ ((rn as u64) << 8));
        let mut llr = ch.transmit(&codeword);

        let mut edge_r = vec![0.0f32; dec.required_edge_buffer()];
        let mut scratch = vec![0.0f32; dec.required_layer_buffer()];
        let mut hard = vec![0u8; n];

        dec.decode_layered_offset_min_sum(&mut llr, &mut edge_r, &mut scratch, &mut hard, 20)
            .unwrap_or_else(|e| panic!("decode failed for Z={z} rate={rn}/{rd}: {e}"));

        let errors: usize = codeword
            .iter()
            .zip(hard.iter())
            .filter(|&(&c, &h)| c != h)
            .count();
        assert_eq!(
            errors, 0,
            "Z={z} rate={rn}/{rd}: expected 0 bit errors at high SNR, got {errors}"
        );
    }
}

/// A single hard bit-flip should still be corrected by the LOMS decoder,
/// exercised at Z=81 across all 4 rates (mirrors the 3GPP
/// `encode_decode_one_bit_error_corrected_*` tests in
/// `tests/ldpc_integration.rs`).
#[test]
fn one_bit_error_corrected_z81_all_rates() {
    for &(z, rn, rd) in ALL_12.iter().filter(|&&(z, _, _)| z == 81) {
        let enc = wifi_ldpc_encoder(z, rn, rd).unwrap();
        let dec = wifi_ldpc_decoder(z, rn, rd, 0.25).unwrap();

        let k = enc.info_bit_count();
        let n = enc.codeword_bit_count();
        let info: Vec<u8> = (0..k).map(|i| (i % 5 == 0) as u8).collect();
        let mut codeword = vec![0u8; n];
        enc.encode(&info, &mut codeword).unwrap();

        let mut llr: Vec<f32> = codeword
            .iter()
            .map(|&b| if b == 0 { 5.0 } else { -5.0 })
            .collect();
        let flip_pos = k / 2;
        llr[flip_pos] = -llr[flip_pos];

        let mut edge_r = vec![0.0f32; dec.required_edge_buffer()];
        let mut scratch = vec![0.0f32; dec.required_layer_buffer()];
        let mut hard = vec![0u8; n];

        dec.decode_layered_offset_min_sum(&mut llr, &mut edge_r, &mut scratch, &mut hard, 20)
            .unwrap();

        let errors: usize = codeword
            .iter()
            .zip(hard.iter())
            .filter(|&(&c, &h)| c != h)
            .count();
        assert_eq!(
            errors, 0,
            "Z={z} rate={rn}/{rd}: 1-bit error should be corrected, {errors} remain"
        );
    }
}

/// Structural sanity: for every one of the 12 combinations, the all-zero
/// codeword must satisfy every parity check (syndrome zero), verified via a
/// 0-iteration decode call (no correction needed/possible; this only
/// exercises `check_syndrome_f32`'s early-termination path).
#[test]
fn all_zero_codeword_satisfies_syndrome_all_12() {
    for (z, rn, rd) in ALL_12 {
        let dec = wifi_ldpc_decoder(z, rn, rd, 0.25).unwrap();
        let n = dec.variable_node_count();

        // Strong "all zero bits transmitted" LLRs (+5.0 => hard decision 0).
        let mut llr = vec![5.0f32; n];
        let mut edge_r = vec![0.0f32; dec.required_edge_buffer()];
        let mut scratch = vec![0.0f32; dec.required_layer_buffer()];
        let mut hard = vec![0u8; n];

        let iters = dec
            .decode_layered_offset_min_sum(&mut llr, &mut edge_r, &mut scratch, &mut hard, 10)
            .unwrap();

        // The syndrome check is satisfied trivially by the all-zero
        // codeword, so the decoder should terminate on the first pass.
        assert_eq!(
            iters, 1,
            "Z={z} rate={rn}/{rd}: all-zero codeword should satisfy the syndrome on pass 1"
        );
        assert!(
            hard.iter().all(|&b| b == 0),
            "Z={z} rate={rn}/{rd}: all-zero input should decode to all-zero output"
        );
    }
}
