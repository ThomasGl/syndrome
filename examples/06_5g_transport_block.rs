//! The flagship: a full 5G NR DL-SCH transport block, end to end.
//!
//! Teaches: this is the whole TS 38.212 §5.1-5.5 chain wired together --
//! CRC-24A attach, code-block segmentation, QC-LDPC encode, rate matching,
//! transmission over a simulated radio channel, LDPC belief-propagation
//! decode, and the final CRC check -- exactly the pipeline every 5G phone
//! and base station runs on every downlink transport block. Industry
//! context: the "waterfall curve" swept below (BER/CRC pass-rate vs. Eb/N0)
//! is the standard way link-level simulations characterize a channel code:
//! there is a fairly sharp SNR threshold below which the iterative decoder
//! stops converging at all.
//!
//! Run with: `cargo run --example 06_5g_transport_block`

use syndrome::channel_sim::AwgnChannel;
use syndrome::transport_block::{DlSchDecoder, DlSchEncoder};

fn main() {
    // ~100-byte payload -- a small IP packet or a VoIP frame.
    let tb_size = 800usize; // bits
    let target_rate = 0.5f32; // code rate R (selects BG1 vs BG2 + lifting size Z)
    let qm = 2usize; // QPSK: 2 bits per modulated symbol
    let g = 3200usize; // total coded bits budget for this TB (must be divisible by qm)

    let encoder = DlSchEncoder::new(tb_size, target_rate, qm, g)
        .expect("valid TB size / rate / Qm / G combination");
    println!(
        "DL-SCH encoder: {tb_size}-bit TB -> {} code block(s) -> {} coded bits (rate {target_rate})",
        encoder.num_code_blocks(),
        encoder.output_bits()
    );

    // A deterministic "payload" (e.g. a VoIP frame) -- fixed pattern so the
    // whole example is reproducible byte-for-byte across runs.
    let tb: Vec<u8> = (0..tb_size).map(|i| ((i * 13 + 5) % 7 < 3) as u8).collect();

    let mut coded = vec![0u8; encoder.output_bits()];
    encoder
        .encode(&tb, /* redundancy version */ 0, &mut coded)
        .expect("encode buffers are correctly sized above");

    // Sweep Eb/N0 to trace out the classic LDPC "waterfall": well above
    // threshold the decoder converges in a couple of iterations; right at
    // threshold it needs many more; below threshold it burns the full
    // iteration budget and still fails CRC.
    println!("\nEb/N0 sweep (same transport block, same code, 20-iteration budget per attempt):");
    println!(
        "{:>8} | {:>8} | {:>10} | {:>12}",
        "Eb/N0 dB", "CRC OK", "LDPC iters", "bit errors"
    );
    println!("{:->8}-+-{:->8}-+-{:->10}-+-{:->12}", "", "", "", "");

    for ebno_db in [4.0f32, -1.0, -1.5, -2.0] {
        // Fresh decoder per attempt: each carries its own per-CB HARQ buffers,
        // and we want each Eb/N0 point to be an independent first transmission
        // (rv=0), not an HARQ-combined retransmission of the previous one.
        let mut decoder = DlSchDecoder::new(tb_size, target_rate, qm, g, 20, 0.25)
            .expect("same parameters as the encoder");

        let mut channel = AwgnChannel::new(ebno_db, target_rate, 99);
        let rx_llr = channel.transmit(&coded);

        let mut tb_out = vec![0u8; tb_size];
        let report = decoder
            .decode(&rx_llr, 0, &mut tb_out)
            .expect("LLR/output buffers are correctly sized above");
        let bit_errors = AwgnChannel::count_errors(&tb_out, &tb);

        println!(
            "{ebno_db:>8.1} | {:>8} | {:>10} | {:>12}",
            report.crc_ok, report.max_iters_used, bit_errors
        );
    }

    println!(
        "\nNotice the cliff: a ~0.5 dB drop between -1.5 dB (still converges, using more \
         iterations) and -2 dB (fails completely, hundreds of residual bit errors) is \
         the LDPC waterfall in miniature. This is exactly why real schedulers pick a \
         modulation-and-coding scheme (MCS) with a safety margin above the measured SNR, \
         and why HARQ retransmission combining (see DlSchDecoder::decode's `rv` parameter) \
         exists to rescue transport blocks that land just below threshold."
    );
}
