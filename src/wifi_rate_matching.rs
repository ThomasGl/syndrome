//! Shortening and puncturing for IEEE 802.11 Wi-Fi LDPC codewords.
//!
//! [`crate::wifi_ldpc_tables`] and [`crate::wifi`] provide the real 802.11
//! Annex R/F shift matrices and full-codeword encode/decode ($K$ info bits
//! in, $N$ coded bits out, exactly), but a real transmission almost never
//! needs exactly that: a payload is usually smaller than $K$, and the
//! number of coded bits that actually fit the scheduled OFDM symbols is
//! usually different from $N$ too. This module is the adapter between
//! those two sizes, for a single LDPC codeword:
//!
//! * **Shortening** pads a payload smaller than $K$ with known, unsent
//!   zero bits at the end of the systematic block. Both sides already know
//!   these bits are zero, so they cost the decoder nothing but the LLR
//!   buffer position — it feeds them back in as a high-confidence LLR
//!   before decoding, rather than an unpaid channel observation.
//! * **Puncturing** drops trailing parity bits so the transmitted length
//!   fits the coded-bit budget for a given rate/MCS. The decoder feeds
//!   those positions back in as an *erasure* ($LLR = 0$) — the opposite
//!   confidence of a shortened bit, since nothing is known about them.
//!
//! ```text
//! codeword:  [ payload_bits (real) ][ n_shrt (known-0, unsent) ][ parity, transmitted ][ n_punc (unsent) ]
//!             \_____________________ K = payload_bits + n_shrt _/\__________________ N - K ________________/
//! transmit:  [ payload_bits (real) ][ transmitted parity ]                      <- target_coded_bits total ->
//! ```
//!
//! # What this does not do
//!
//! This implements the shortening/puncturing *mechanism* — the bit
//! insertion/removal and LLR reconstruction — exactly and generically for
//! any $(K, N)$ pair, including all 12 real 802.11 $(Z, R)$ combinations.
//! It does **not** implement:
//! * Multi-codeword segmentation: a payload larger than one codeword's $K$
//!   is rejected, not split across several LDPC codewords (matching the
//!   existing limitation of [`crate::wifi::select_wifi_ldpc`]).
//! * The 802.11 PPDU-level formula that derives *how many* coded bits are
//!   actually available for a given MCS, bandwidth, and PSDU length (IEEE
//!   802.11-2020 §19.5.3.2, which depends on $N_{CBPS}$/$N_{SYM}$ figures
//!   this crate's [`crate::wifi::WifiMcs`] table does not carry). The
//!   caller supplies `target_coded_bits` directly, the same way
//!   [`crate::transport_block::DlSchEncoder`] takes `g` directly rather
//!   than deriving it from a 5G scheduling grant.
//! * The rest of the 802.11 PHY chain (scrambling, interleaving, OFDM
//!   subcarrier mapping) — out of scope for this FEC-only crate.

use crate::error::FecError;
use crate::qc_ldpc::{QcLdpcDecoder, QcLdpcEncoder};

/// High-confidence LLR for a shortened (known-zero) bit position.
///
/// Matches the sign convention used throughout this crate — positive LLR
/// favours bit 0 (see `crate::channel_sim` / `crate::turbo` module docs) —
/// and the same magnitude as the 5G rate-matching path's internal filler
/// constant: large enough that the LOMS decoder treats the bit as certain
/// without overflowing the `f32` min-sum arithmetic.
const LLR_KNOWN_ZERO: f32 = 1_000_000.0;

/// Compute the shortening and puncturing bit counts for a single codeword.
///
/// # Arguments
///
/// * `k` - Information bits of the full codeword ($K$).
/// * `n` - Coded bits of the full codeword ($N$).
/// * `payload_bits` - Real information bits to protect ($\le K$).
/// * `target_coded_bits` - Bits actually transmitted.
///
/// # Returns
///
/// `(n_shrt, n_punc)`: the number of shortened (unsent, known-zero
/// systematic) bits and punctured (unsent, erased parity) bits.
/// $n_{shrt} = K - \text{payload\_bits}$, and $n_{punc} = (N - n_{shrt}) -
/// \text{target\_coded\_bits}$.
///
/// # Errors
///
/// Returns [`FecError::InvalidParam`] if `payload_bits > k` (the payload
/// does not fit in one codeword; multi-codeword segmentation is not
/// implemented), if `target_coded_bits < payload_bits` (every real
/// information bit is always transmitted, so the budget can never be
/// smaller than the payload itself), or if `target_coded_bits` exceeds the
/// codeword's coded-bit budget after shortening ($N - n_{shrt}$) — puncturing
/// can only remove bits, not add them back.
///
/// # Examples
///
/// ```
/// use syndrome::wifi_rate_matching::shortening_and_puncture_counts;
///
/// // K=324, N=648 (Z=27, R=1/2): a 100-bit payload, budget for 400 coded bits.
/// let (n_shrt, n_punc) = shortening_and_puncture_counts(324, 648, 100, 400).unwrap();
/// assert_eq!(n_shrt, 224); // 324 - 100
/// assert_eq!(n_punc, 24); // (648 - 224) - 400
/// ```
pub fn shortening_and_puncture_counts(
    k: usize,
    n: usize,
    payload_bits: usize,
    target_coded_bits: usize,
) -> Result<(usize, usize), FecError> {
    // `k` and `n` are free parameters here, not read off a validated
    // matrix, so a caller can supply a pair no real code could have. Every
    // block code satisfies n >= k (rate at most 1); without this check the
    // `n - n_shrt` below underflows whenever n < k - payload_bits, which
    // panics in a debug build and — worse — wraps silently in a release
    // build, yielding an enormous `max_coded` that lets the range check
    // pass and returns a meaningless puncture count.
    if n < k {
        return Err(FecError::InvalidParam(
            "codeword length N must be at least the information length K",
        ));
    }
    if payload_bits > k {
        return Err(FecError::InvalidParam(
            "payload_bits exceeds this codeword's K; multi-codeword segmentation is not implemented",
        ));
    }
    if target_coded_bits < payload_bits {
        return Err(FecError::InvalidParam(
            "target_coded_bits must be at least payload_bits: every real information bit is always transmitted",
        ));
    }
    let n_shrt = k - payload_bits;
    let max_coded = n - n_shrt;
    if target_coded_bits > max_coded {
        return Err(FecError::InvalidParam(
            "target_coded_bits exceeds N minus the shortened bits (N - n_shrt); puncturing cannot add bits back",
        ));
    }
    Ok((n_shrt, max_coded - target_coded_bits))
}

/// Encode a payload of up to $K$ bits into a shortened, punctured codeword.
///
/// Pads `payload` with known-zero bits up to the encoder's full $K$, runs
/// the ordinary systematic LDPC encode, then transmits only the real
/// systematic bits (never the shortened padding) followed by as many
/// parity bits as fit `target_coded_bits`, dropping the rest (puncturing).
///
/// # Arguments
///
/// * `encoder` - Encoder for the target $(Z, R)$ 802.11 codeword (see
///   [`crate::wifi::WifiLdpcParams::build_encoder`]).
/// * `payload` - Real information bits, one bit per byte ($\le K$).
/// * `target_coded_bits` - Number of bits to actually transmit.
/// * `out` - Output buffer; must have length exactly `target_coded_bits`.
///
/// # Errors
///
/// Propagates [`shortening_and_puncture_counts`]'s errors, and returns
/// [`FecError::BufferTooSmall`] if `out.len() != target_coded_bits`.
///
/// # Examples
///
/// ```
/// use syndrome::wifi::{select_wifi_ldpc, WifiStandard};
/// use syndrome::wifi_rate_matching::encode_shortened;
///
/// let params = select_wifi_ldpc(41, WifiStandard::WiFi6, 0.5); // Z=27, R=1/2, K=324
/// let encoder = params.build_encoder().unwrap();
///
/// let payload = vec![1u8, 0, 1, 1, 0, 0, 1, 0]; // 8 real info bits, far below K
/// let target_coded_bits = 200; // fewer than N=648: puncture the parity down
/// let mut coded = vec![0u8; target_coded_bits];
/// encode_shortened(&encoder, &payload, target_coded_bits, &mut coded).unwrap();
/// assert_eq!(&coded[..payload.len()], &payload[..]);
/// ```
pub fn encode_shortened(
    encoder: &QcLdpcEncoder,
    payload: &[u8],
    target_coded_bits: usize,
    out: &mut [u8],
) -> Result<(), FecError> {
    let k = encoder.info_bit_count();
    let n = encoder.codeword_bit_count();
    let payload_bits = payload.len();
    let (_n_shrt, n_punc) = shortening_and_puncture_counts(k, n, payload_bits, target_coded_bits)?;

    if out.len() != target_coded_bits {
        return Err(FecError::BufferTooSmall {
            required: target_coded_bits,
            provided: out.len(),
        });
    }

    // Setup path, not a hot loop: a heap Vec here is acceptable (same
    // reasoning as `QcLdpcEncoder::encode_5g`).
    let mut full_info = vec![0u8; k];
    full_info[..payload_bits].copy_from_slice(payload);
    // Shortened positions [payload_bits..k] remain zero.

    let mut codeword = vec![0u8; n];
    encoder.encode(&full_info, &mut codeword)?;

    let parity_transmit_len = (n - k) - n_punc;
    out[..payload_bits].copy_from_slice(payload);
    out[payload_bits..].copy_from_slice(&codeword[k..k + parity_transmit_len]);
    Ok(())
}

/// Reconstruct a full-length LLR vector from a received shortened/punctured
/// buffer and decode it.
///
/// Places the received LLRs at their real systematic and transmitted-parity
/// positions, fills the shortened positions with a high-confidence
/// known-zero LLR, fills the punctured positions with an erasure ($LLR =
/// 0$), and decodes the reconstructed full-length buffer.
///
/// # Arguments
///
/// * `decoder` - Decoder for the target $(Z, R)$ 802.11 codeword (see
///   [`crate::wifi::WifiLdpcParams::build_decoder`]).
/// * `payload_bits` - Number of real information bits that were sent (the
///   receiver must know this out-of-band, same as `n_filler` for 5G).
/// * `rx_llr` - Received channel LLRs, one entry per transmitted bit
///   (length = the `target_coded_bits` used at encode time).
/// * `codeword_llr` - Caller-owned scratch of length
///   [`QcLdpcDecoder::variable_node_count`] ($N$); overwritten with the
///   reconstructed full-length LLR buffer.
/// * `edge_r`, `layer_scratch`, `hard_output`, `iterations` - Passed
///   through to [`QcLdpcDecoder::decode_layered_offset_min_sum`] unchanged;
///   see its docs for sizing.
///
/// # Returns
///
/// The number of LOMS iterations actually used (early-exit on a satisfied
/// syndrome). The recovered payload is `hard_output[..payload_bits]`.
///
/// # Errors
///
/// Propagates [`shortening_and_puncture_counts`]'s errors, returns
/// [`FecError::BufferTooSmall`] if `codeword_llr.len() != N`, and
/// propagates any error from
/// [`QcLdpcDecoder::decode_layered_offset_min_sum`].
///
/// # Examples
///
/// ```
/// use syndrome::wifi::{select_wifi_ldpc, WifiStandard};
/// use syndrome::wifi_rate_matching::{decode_shortened, encode_shortened};
///
/// let params = select_wifi_ldpc(41, WifiStandard::WiFi6, 0.5);
/// let encoder = params.build_encoder().unwrap();
/// let decoder = params.build_decoder(0.25).unwrap();
///
/// let payload = vec![1u8, 0, 1, 1, 0, 0, 1, 0];
/// let target_coded_bits = 200;
/// let mut coded = vec![0u8; target_coded_bits];
/// encode_shortened(&encoder, &payload, target_coded_bits, &mut coded).unwrap();
///
/// // Strong noiseless channel: bit 0 -> +LLR, bit 1 -> -LLR.
/// let rx_llr: Vec<f32> = coded.iter().map(|&b| if b == 0 { 8.0 } else { -8.0 }).collect();
///
/// let n = decoder.variable_node_count();
/// let mut codeword_llr = vec![0f32; n];
/// let mut edge_r = vec![0f32; decoder.required_edge_buffer()];
/// let mut layer_scratch = vec![0f32; decoder.required_layer_buffer()];
/// let mut hard_output = vec![0u8; n];
///
/// decode_shortened(
///     &decoder, payload.len(), &rx_llr, &mut codeword_llr,
///     &mut edge_r, &mut layer_scratch, &mut hard_output, 20,
/// ).unwrap();
/// assert_eq!(&hard_output[..payload.len()], &payload[..]);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn decode_shortened(
    decoder: &QcLdpcDecoder,
    payload_bits: usize,
    rx_llr: &[f32],
    codeword_llr: &mut [f32],
    edge_r: &mut [f32],
    layer_scratch: &mut [f32],
    hard_output: &mut [u8],
    iterations: usize,
) -> Result<usize, FecError> {
    let n = decoder.variable_node_count();
    let k = n - decoder.check_node_count();
    let target_coded_bits = rx_llr.len();
    let (_n_shrt, n_punc) = shortening_and_puncture_counts(k, n, payload_bits, target_coded_bits)?;

    if codeword_llr.len() != n {
        return Err(FecError::BufferTooSmall {
            required: n,
            provided: codeword_llr.len(),
        });
    }

    codeword_llr[..payload_bits].copy_from_slice(&rx_llr[..payload_bits]);
    for llr in &mut codeword_llr[payload_bits..k] {
        *llr = LLR_KNOWN_ZERO;
    }
    let parity_transmit_len = (n - k) - n_punc;
    codeword_llr[k..k + parity_transmit_len]
        .copy_from_slice(&rx_llr[payload_bits..payload_bits + parity_transmit_len]);
    for llr in &mut codeword_llr[k + parity_transmit_len..] {
        *llr = 0.0;
    }

    decoder.decode_layered_offset_min_sum(
        codeword_llr,
        edge_r,
        layer_scratch,
        hard_output,
        iterations,
    )
}

/// [`decode_shortened`] with every buffer taken from an
/// [`LdpcWorkspace`](crate::qc_ldpc::LdpcWorkspace) instead of four
/// separate caller-sized slices.
///
/// This is the recommended entry point unless you need exact control over
/// each allocation: build the workspace once with
/// [`QcLdpcDecoder::workspace`], reuse it for every received frame, and
/// each decode stays allocation-free.
///
/// # Arguments
///
/// * `decoder` - Decoder for the target $(Z, R)$ 802.11 codeword.
/// * `payload_bits` - Number of real information bits that were sent.
/// * `rx_llr` - Received channel LLRs, one entry per transmitted bit.
/// * `workspace` - Buffers from [`QcLdpcDecoder::workspace`]. After the
///   call, the recovered payload is
///   `&workspace.hard_output()[..payload_bits]` and the reconstructed
///   full-length a-posteriori LLRs are in `workspace.posterior_llr()`.
/// * `iterations` - Number of LOMS passes.
///
/// # Returns
///
/// The number of LOMS iterations actually used.
///
/// # Errors
///
/// Same conditions as [`decode_shortened`]; a workspace built for a
/// different decoder is rejected, never silently misused.
///
/// # Examples
///
/// ```
/// use syndrome::wifi::{select_wifi_ldpc, WifiStandard};
/// use syndrome::wifi_rate_matching::{decode_shortened_with_workspace, encode_shortened};
///
/// let params = select_wifi_ldpc(41, WifiStandard::WiFi6, 0.5);
/// let encoder = params.build_encoder().unwrap();
/// let decoder = params.build_decoder(0.25).unwrap();
///
/// let payload = vec![1u8, 0, 1, 1, 0, 0, 1, 0];
/// let mut coded = vec![0u8; 200];
/// encode_shortened(&encoder, &payload, coded.len(), &mut coded).unwrap();
///
/// // Strong noiseless channel: bit 0 -> +LLR, bit 1 -> -LLR.
/// let rx_llr: Vec<f32> = coded.iter().map(|&b| if b == 0 { 8.0 } else { -8.0 }).collect();
///
/// let mut ws = decoder.workspace();
/// decode_shortened_with_workspace(&decoder, payload.len(), &rx_llr, &mut ws, 20).unwrap();
/// assert_eq!(&ws.hard_output()[..payload.len()], &payload[..]);
/// ```
pub fn decode_shortened_with_workspace(
    decoder: &QcLdpcDecoder,
    payload_bits: usize,
    rx_llr: &[f32],
    workspace: &mut crate::qc_ldpc::LdpcWorkspace,
    iterations: usize,
) -> Result<usize, FecError> {
    // Split borrows: the codeword staging buffer is rebuilt by
    // decode_shortened while the other three buffers feed the core decoder.
    let crate::qc_ldpc::LdpcWorkspace {
        codeword_llr,
        edge_r,
        layer_scratch,
        hard,
    } = workspace;
    decode_shortened(
        decoder,
        payload_bits,
        rx_llr,
        codeword_llr,
        edge_r,
        layer_scratch,
        hard,
        iterations,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel_sim::AwgnChannel;
    use crate::wifi_ldpc_tables::{wifi_ldpc_decoder, wifi_ldpc_encoder};

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

    #[test]
    fn counts_example_matches_doctest() {
        let (n_shrt, n_punc) = shortening_and_puncture_counts(324, 648, 100, 400).unwrap();
        assert_eq!(n_shrt, 224);
        assert_eq!(n_punc, 24);
    }

    #[test]
    fn counts_no_shortening_no_puncturing() {
        let (n_shrt, n_punc) = shortening_and_puncture_counts(324, 648, 324, 648).unwrap();
        assert_eq!(n_shrt, 0);
        assert_eq!(n_punc, 0);
    }

    #[test]
    fn counts_rejects_payload_larger_than_k() {
        assert!(shortening_and_puncture_counts(324, 648, 325, 648).is_err());
    }

    #[test]
    fn counts_rejects_target_below_payload() {
        assert!(shortening_and_puncture_counts(324, 648, 100, 99).is_err());
    }

    #[test]
    fn counts_rejects_target_above_budget() {
        // n_shrt = 224, so max_coded = 648 - 224 = 424; 425 must be rejected.
        assert!(shortening_and_puncture_counts(324, 648, 100, 425).is_err());
    }

    /// Noiseless round trip (deterministic payload, strong LLRs) across all
    /// 12 real 802.11 (Z, R) matrices, each with genuine shortening (payload
    /// well below K) and genuine puncturing (target well below the
    /// post-shortening budget).
    #[test]
    fn roundtrip_shortened_and_punctured_all_12_combinations() {
        for (z, rn, rd) in ALL_12 {
            let enc = wifi_ldpc_encoder(z, rn, rd).unwrap();
            let dec = wifi_ldpc_decoder(z, rn, rd, 0.25).unwrap();

            let k = enc.info_bit_count();
            let n = enc.codeword_bit_count();
            let payload_bits = k / 3;
            let payload: Vec<u8> = (0..payload_bits).map(|i| (i % 2) as u8).collect();

            let n_shrt = k - payload_bits;
            let max_coded = n - n_shrt;
            let target_coded_bits = max_coded - max_coded / 5; // genuine puncturing

            let mut coded = vec![0u8; target_coded_bits];
            encode_shortened(&enc, &payload, target_coded_bits, &mut coded)
                .unwrap_or_else(|e| panic!("encode failed for Z={z} rate={rn}/{rd}: {e}"));
            assert_eq!(&coded[..payload_bits], payload.as_slice());

            let rx_llr: Vec<f32> = coded
                .iter()
                .map(|&b| if b == 0 { 8.0 } else { -8.0 })
                .collect();

            let mut codeword_llr = vec![0f32; n];
            let mut edge_r = vec![0f32; dec.required_edge_buffer()];
            let mut layer_scratch = vec![0f32; dec.required_layer_buffer()];
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
            .unwrap_or_else(|e| panic!("decode failed for Z={z} rate={rn}/{rd}: {e}"));

            assert_eq!(
                &hard_output[..payload_bits],
                payload.as_slice(),
                "Z={z} rate={rn}/{rd}: shortened+punctured payload must round-trip exactly"
            );
        }
    }

    /// A shortened/punctured codeword still corrects real channel noise —
    /// the point of doing LDPC at all, not just a data-movement exercise.
    /// Z=81, R=1/2 at a moderate Eb/N0, with roughly a third of K shortened
    /// away and roughly 10% of the remaining budget punctured.
    #[test]
    fn shortened_punctured_codeword_corrects_awgn_noise() {
        let enc = wifi_ldpc_encoder(81, 1, 2).unwrap();
        let dec = wifi_ldpc_decoder(81, 1, 2, 0.25).unwrap();

        let k = enc.info_bit_count();
        let n = enc.codeword_bit_count();
        let payload_bits = (2 * k) / 3;
        let payload: Vec<u8> = (0..payload_bits).map(|i| (i % 7 == 0) as u8).collect();

        let n_shrt = k - payload_bits;
        let max_coded = n - n_shrt;
        let target_coded_bits = max_coded - max_coded / 10;

        let mut coded = vec![0u8; target_coded_bits];
        encode_shortened(&enc, &payload, target_coded_bits, &mut coded).unwrap();

        let rate = target_coded_bits as f32 / payload_bits as f32;
        let mut ch = AwgnChannel::new(3.0, 1.0 / rate, 0xBEEF);
        let rx_llr = ch.transmit(&coded);

        let mut codeword_llr = vec![0f32; n];
        let mut edge_r = vec![0f32; dec.required_edge_buffer()];
        let mut layer_scratch = vec![0f32; dec.required_layer_buffer()];
        let mut hard_output = vec![0u8; n];

        decode_shortened(
            &dec,
            payload_bits,
            &rx_llr,
            &mut codeword_llr,
            &mut edge_r,
            &mut layer_scratch,
            &mut hard_output,
            25,
        )
        .unwrap();

        let errors: usize = hard_output[..payload_bits]
            .iter()
            .zip(payload.iter())
            .filter(|&(&h, &p)| h != p)
            .count();
        assert_eq!(
            errors, 0,
            "expected 0 payload bit errors at 3 dB with a moderately punctured code, got {errors}"
        );
    }
}
