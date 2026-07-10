//! Polar codes: 5G's choice for short, latency-critical control channels.
//!
//! Teaches: polar codes are built by "polarizing" N correlated bit channels
//! into a mix of nearly-perfect and nearly-useless ones (via the recursive
//! butterfly transform), then only sending real data on the good channels
//! ("frozen" bits carry a known 0 on the bad ones). Successive Cancellation
//! (SC) decoding walks that same recursive structure top-down, combining
//! LLRs with f()/g() rules and feeding each hard decision forward into the
//! next one -- which is also *why* SC is fragile: one wrong early decision
//! propagates. Industry context: 5G NR uses polar codes for PDCCH (DCI) and
//! PBCH -- small, latency-critical control payloads where LDPC's larger
//! block sizes and iterative decoding don't pay off.
//!
//! Run with: `cargo run --example 05_polar_code`

use syndrome::channel_sim::AwgnChannel;
use syndrome::polar::{PolarDecoder, PolarEncoder};

/// Minimal deterministic xorshift64 PRNG, seeded for a reproducible demo
/// message (same shift triplet as `channel_sim::AwgnChannel`).
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    fn next_bit(&mut self) -> u8 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x & 1) as u8
    }
}

fn main() {
    // N=32 codeword bits, K=16 information bits.
    let n = 32usize;
    let k = 16usize;

    let encoder = PolarEncoder::new(n, k).expect("N must be a power of 2 and K < N");
    let decoder = PolarDecoder::new(n, k, 1, None).expect("same N/K as the encoder"); // list=1 => plain SC

    println!(
        "Polar code: N={n} (codeword bits), K={k} (info bits), rate = {:.2}",
        k as f32 / n as f32
    );
    println!("(5G NR uses exactly this scheme for PDCCH/PBCH control channels.)");

    // A normal, mixed-bit 16-bit message (fixed seed, for a reproducible
    // demo) -- not a single-flag special case. SC decode correctly
    // reconstructs any information pattern, not just low-weight ones.
    let mut rng = Xorshift64::new(0x00C0_FFEE);
    let info: Vec<u8> = (0..k).map(|_| rng.next_bit()).collect();
    let mut codeword = vec![0u8; n];
    encoder.encode(&info, &mut codeword).unwrap();
    println!("\nInfo bits (random 16-bit message): {info:?}");

    // Mild noise: control channels are usually power-boosted / low-rate
    // specifically so they survive worse conditions than the data channel.
    let ebno_db = 3.0;
    let mut channel = AwgnChannel::new(ebno_db, k as f32 / n as f32, 12);
    let llr = channel.transmit(&codeword);

    // Successive Cancellation decode: recursively combines LLR pairs with the
    // f() (worse-case) and g() (conditional) rules until each of the N leaf
    // bits is resolved -- frozen leaves are forced to 0, info leaves take the
    // sign of their computed LLR, and each level's decoded left sub-block is
    // re-encoded to recover the partial sum that feeds the next g() step.
    let mut decoded = vec![0u8; k];
    decoder.decode_sc(&llr, &mut decoded).unwrap();

    let errors = AwgnChannel::count_errors(&decoded, &info);
    println!("Decoded bits                              : {decoded:?}");
    println!("\nEb/N0 = {ebno_db} dB -> {errors} bit error(s) out of {k} after SC decoding.");

    assert_eq!(decoded, info, "message decodes correctly at this Eb/N0");
    println!(
        "Message recovered correctly. In production 5G modems, PDCCH uses \
         CRC-aided List decoding (CA-SCL, see PolarDecoder::decode_scl) precisely \
         because plain SC has no way to recover from an early wrong decision -- \
         the list keeps several candidate paths alive and lets the CRC pick the \
         survivor."
    );
}
