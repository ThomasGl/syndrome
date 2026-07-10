//! CRC-24A: detection without correction.
//!
//! Teaches: a CRC is a *detector*, not a corrector -- it tells you "this
//! frame is damaged" with overwhelming probability, but gives no clue which
//! bit(s) flipped or how to fix them. Industry context: every 5G transport
//! block (TS 38.212 5.1) is CRC-24A checked *after* LDPC decoding, because
//! LDPC can converge to a plausible-but-wrong codeword and the CRC is the
//! final tripwire that catches it.
//!
//! Run with: `cargo run --example 02_crc_error_detection`

use syndrome::crc::{Crc24, CrcKind};

fn main() {
    let crc = Crc24::new(CrcKind::Crc24A);

    // A CRC works over a bit-string, MSB-first, matching the 3GPP convention
    // used throughout this crate (see crc.rs). Sixteen payload bits here.
    let mut payload: Vec<u8> = vec![1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1];
    println!("Payload ({} bits): {:?}", payload.len(), payload);

    // `attach` appends 24 parity bits computed by dividing the payload by
    // the CRC-24A generator polynomial (mod-2 polynomial division). Those
    // 24 bits are redundant *by construction*: a correct receiver can always
    // recompute them from the payload alone.
    crc.attach(&mut payload);
    println!(
        "After CRC-24A attach ({} bits total, last 24 are parity)",
        payload.len()
    );

    // Case 1: untouched frame -- must pass.
    assert!(crc.check(&payload), "an unmodified frame must always pass");
    println!("\nCase 1 -- untouched frame:  CRC check = PASS (as expected)");

    // Case 2: a single bit flips somewhere in transit (radio fading, thermal
    // noise, whatever). The CRC recomputation over the corrupted payload
    // will not match the transmitted parity bits with near-certain probability.
    let mut corrupted = payload.clone();
    corrupted[3] ^= 1;
    let ok = crc.check(&corrupted);
    println!("Case 2 -- 1 bit flipped:    CRC check = {}", pass_fail(ok));
    assert!(!ok, "CRC-24A must catch a single flipped bit");

    // Case 3: the CRC tells us *that* something is wrong, but not *what*.
    // Unlike Hamming(7,4) in example 01, there is no syndrome-to-bit-position
    // mapping here -- the only correct response is "discard and ask again"
    // (ARQ) or "hand the frame to a code that *can* correct it" (LDPC).
    println!(
        "\nNotice: the CRC gives a single yes/no bit. It cannot say *which* \
         bit flipped, unlike Hamming(7,4)'s syndrome. That's the whole point --\n\
         a 24-bit CRC over a huge payload is cheap insurance, not a repair kit."
    );

    // Why 5G pairs CRC with LDPC:
    // LDPC's iterative belief-propagation decoder can *converge* to a
    // codeword that satisfies every parity-check equation yet is still not
    // the transmitted one (rare, but happens at low SNR / few iterations).
    // The CRC-24A is the independent, cheap, final check that catches this
    // "confidently wrong" case before the payload is passed up the stack.
    println!(
        "\n5G reasoning: LDPC (see example 06) fixes most bit errors, but its \
         iterative decoder can occasionally converge to the wrong codeword.\n\
         CRC-24A is attached *outside* the LDPC codeword specifically to catch \
         that residual failure mode -- cheap, independent, and (near) infallible."
    );
}

fn pass_fail(ok: bool) -> &'static str {
    if ok { "PASS" } else { "FAIL (detected!)" }
}
