//! Integration tests for Wi-Fi LDPC shortening and puncturing
//! (`syndrome::wifi_rate_matching`).
//!
//! These mirror `tests/wifi_ldpc_integration.rs`'s encode -> AWGN -> decode
//! shape, but for a payload genuinely smaller than $K$ (shortened) and a
//! transmitted length genuinely smaller than the post-shortening budget
//! (punctured), across all 12 real 802.11 (Z, R) matrices.

use syndrome::channel_sim::AwgnChannel;
use syndrome::wifi_ldpc_tables::{wifi_ldpc_decoder, wifi_ldpc_encoder};
use syndrome::wifi_rate_matching::{decode_shortened, encode_shortened};

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

/// Encode a payload at roughly 40% of K, puncture the transmitted length
/// down to roughly 90% of the post-shortening budget, push it through a
/// high-reliability AWGN channel, decode, and assert a bit-exact payload
/// recovery. Run for all 12 (Z, rate) combinations.
#[test]
fn shortened_punctured_awgn_decode_roundtrip_all_12_combinations() {
    for (z, rn, rd) in ALL_12 {
        let enc = wifi_ldpc_encoder(z, rn, rd)
            .unwrap_or_else(|e| panic!("encoder build failed for Z={z} rate={rn}/{rd}: {e}"));
        let dec = wifi_ldpc_decoder(z, rn, rd, 0.25)
            .unwrap_or_else(|e| panic!("decoder build failed for Z={z} rate={rn}/{rd}: {e}"));

        let k = enc.info_bit_count();
        let n = enc.codeword_bit_count();

        let payload_bits = (2 * k) / 5;
        let payload: Vec<u8> = (0..payload_bits).map(|i| (i % 3 == 0) as u8).collect();

        let n_shrt = k - payload_bits;
        let max_coded = n - n_shrt;
        let target_coded_bits = max_coded - max_coded / 10;

        let mut coded = vec![0u8; target_coded_bits];
        encode_shortened(&enc, &payload, target_coded_bits, &mut coded)
            .unwrap_or_else(|e| panic!("encode_shortened failed for Z={z} rate={rn}/{rd}: {e}"));

        let effective_rate = target_coded_bits as f32 / payload_bits as f32;
        let mut ch = AwgnChannel::new(
            6.0,
            1.0 / effective_rate,
            0xC0FFEE ^ (z as u64) ^ ((rn as u64) << 8),
        );
        let rx_llr = ch.transmit(&coded);

        let mut codeword_llr = vec![0.0f32; n];
        let mut edge_r = vec![0.0f32; dec.required_edge_buffer()];
        let mut layer_scratch = vec![0.0f32; dec.required_layer_buffer()];
        let mut hard_output = vec![0u8; n];

        decode_shortened(
            &dec,
            payload_bits,
            &rx_llr,
            &mut codeword_llr,
            &mut edge_r,
            &mut layer_scratch,
            &mut hard_output,
            20,
        )
        .unwrap_or_else(|e| panic!("decode_shortened failed for Z={z} rate={rn}/{rd}: {e}"));

        let errors: usize = hard_output[..payload_bits]
            .iter()
            .zip(payload.iter())
            .filter(|&(&h, &p)| h != p)
            .count();
        assert_eq!(
            errors, 0,
            "Z={z} rate={rn}/{rd}: expected 0 payload bit errors at high SNR, got {errors}"
        );
    }
}

/// Rejects a payload that doesn't fit one codeword and a puncturing target
/// that exceeds the post-shortening budget, for a representative (Z, R).
#[test]
fn rejects_out_of_range_shortening_and_puncturing_targets() {
    let enc = wifi_ldpc_encoder(27, 1, 2).unwrap();
    let k = enc.info_bit_count();
    let n = enc.codeword_bit_count();

    let too_big_payload = vec![0u8; k + 1];
    let mut out = vec![0u8; n];
    assert!(encode_shortened(&enc, &too_big_payload, n, &mut out).is_err());

    let payload = vec![0u8; k / 2];
    let mut out = vec![0u8; n]; // budget after shortening (N - K/2) is below N
    assert!(encode_shortened(&enc, &payload, n, &mut out).is_err());
}
