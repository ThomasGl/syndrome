//! Viterbi K=7 convolutional decoding: hard vs. soft decisions.
//!
//! Teaches: throwing away information hurts. A "hard" demodulator collapses
//! each received sample to a single bit (0 or 1) before decoding; a "soft"
//! demodulator instead hands the decoder a log-likelihood ratio (LLR) that
//! also encodes *how confident* the receiver is. Feeding that confidence
//! into the trellis search (max-log-MAP branch metric) recovers more errors
//! at the exact same channel SNR. Industry context: this rate-1/2 K=7 code
//! (generators 0o133/0o171) is the classic NASA/3GPP convolutional code used
//! for LTE PDCCH and countless satellite links before turbo/LDPC took over.
//!
//! Run with: `cargo run --example 04_viterbi_convolutional`

use syndrome::channel_sim::AwgnChannel;
use syndrome::viterbi::ViterbiDecoder;

fn main() {
    let dec = ViterbiDecoder::new(7).expect("K=7, rate 1/2 (standard 0o133/0o171 generators)");
    println!(
        "Viterbi decoder: constraint length K={}, rate 1/2, {} trellis states",
        dec.constraint_length,
        1usize << (dec.constraint_length - 1)
    );

    // A 48-bit "message" -- the encoder zero-terminates it internally so the
    // trellis always starts and ends in the known state 0.
    let info: Vec<u8> = (0..48).map(|i| ((i * 7 + 3) % 5 < 2) as u8).collect();
    let coded = dec.encode(&info);
    println!(
        "Encoded {} info bits -> {} coded bits (rate 1/2 + zero-termination tail)",
        info.len(),
        coded.len()
    );

    // A single AWGN channel at a deliberately harsh Eb/N0 so that both
    // decode paths see the *same* noisy samples but the hard path throws
    // away confidence information the soft path keeps.
    let ebno_db = 0.0;
    let mut channel = AwgnChannel::new(ebno_db, 0.5, 32);
    let llrs = channel.transmit(&coded);

    // Hard decision: collapse each LLR to its sign *before* decoding. This
    // is what a naive "slicer" demodulator does -- a +0.01 LLR (barely more
    // likely to be 0) is treated identically to a +9.0 LLR (near-certain 0).
    let hard_bits: Vec<u8> = llrs.iter().map(|&l| if l >= 0.0 { 0 } else { 1 }).collect();
    let hard_decoded = dec.decode_hard(&hard_bits);

    // Soft decision: hand the raw LLRs straight to the decoder. Weak,
    // ambiguous samples contribute less to the trellis path metric than
    // strong, confident ones -- exactly the information the hard path lost.
    let soft_decoded = dec.decode_soft(&llrs);

    let hard_errors = AwgnChannel::count_errors(&hard_decoded, &info);
    let soft_errors = AwgnChannel::count_errors(&soft_decoded, &info);
    let hard_ber = AwgnChannel::bit_error_rate(&hard_decoded, &info);
    let soft_ber = AwgnChannel::bit_error_rate(&soft_decoded, &info);

    println!(
        "\nSame noisy channel, Eb/N0 = {ebno_db} dB, {} info bits:",
        info.len()
    );
    println!("  Hard-decision decode: {hard_errors} bit errors (BER = {hard_ber:.4})");
    println!("  Soft-decision decode: {soft_errors} bit errors (BER = {soft_ber:.4})");

    if soft_errors <= hard_errors {
        println!(
            "\nSoft decoding matched or beat hard decoding using *identical* channel \
             noise -- the ~2 dB \"soft-decision gain\" textbooks describe comes entirely \
             from not discarding confidence information before decoding."
        );
    } else {
        // Rare with this seed/SNR, but reported honestly rather than assumed.
        println!(
            "\nOn this particular noise realization hard decoding happened to do \
             better; rerun at a lower Eb/N0 to see the soft-decision advantage more \
             clearly -- averaged over many frames, soft decoding always wins."
        );
    }
}
