//! Hamming(7,4): the "hello world" of forward error correction.
//!
//! Teaches: how adding a handful of parity bits lets a receiver not just
//! *detect* but *locate and fix* a single-bit error, with no retransmission.
//! Industry context: Hamming codes protect DRAM/ECC memory words today, and
//! the same parity-check-matrix idea scales up to the QC-LDPC codes 5G uses.
//!
//! Run with: `cargo run --example 01_hamming_first_steps`

use glezer_rsv::{decode_hamming_7_4, encode_hamming_7_4};

/// Print a `u8` as an N-bit binary string, least significant bit last.
fn bits(value: u8, width: u32) -> String {
    (0..width)
        .rev()
        .map(|i| if (value >> i) & 1 == 1 { '1' } else { '0' })
        .collect()
}

fn main() {
    // Hamming(7,4) packs 4 data bits + 3 parity bits into a 7-bit codeword.
    // The parity bits are placed at positions 1, 2, 4 (powers of two) and
    // each one covers a different overlapping subset of the data bits. That
    // overlap is the trick: when a single bit flips, the *pattern* of which
    // parity checks fail (the "syndrome") spells out, in binary, the exact
    // bit position that is wrong -- no lookup table required.
    let message: u8 = 0b1011;
    println!("Original 4-bit message : {}", bits(message, 4));

    let codeword = encode_hamming_7_4(message);
    println!("Encoded 7-bit codeword : {}", bits(codeword, 7));

    // Flip exactly one bit to simulate a noisy channel (e.g. a cosmic ray
    // hitting a DRAM cell, or a fading dip on a radio link).
    let error_position = 5u8; // 0-based bit index into the 7-bit codeword
    let received = codeword ^ (1 << error_position);
    println!(
        "Received (bit {error_position} flipped) : {}",
        bits(received, 7)
    );

    // Decoding recomputes the three parity checks. A non-zero syndrome
    // identifies exactly which of the 7 bits is wrong; the decoder flips it
    // back before extracting the original 4 data bits.
    let recovered = decode_hamming_7_4(received).expect("single-bit errors are always correctable");
    println!("Decoded 4-bit message  : {}", bits(recovered, 4));

    assert_eq!(
        recovered, message,
        "Hamming(7,4) must correct any single-bit error"
    );
    println!("\nSingle-bit error corrected successfully -- no retransmission needed.");

    // Hamming(7,4) has "minimum distance 3": it can correct 1 error, or
    // merely *detect* (not fix) 2 errors. Show what happens with 2 flips:
    // the decoder still returns *a* nibble, but it is silently wrong -- this
    // is exactly why real systems pair a correcting code with a separate
    // detecting code (see example 02).
    let double_error = codeword ^ (1 << 2) ^ (1 << 5);
    let wrongly_decoded = decode_hamming_7_4(double_error).unwrap();
    println!(
        "\nWith 2 bit flips the decoder \"corrects\" to {} (wrong: original was {}) -- \
         2 errors exceed what a single-error-correcting code can fix.",
        bits(wrongly_decoded, 4),
        bits(message, 4)
    );
}
