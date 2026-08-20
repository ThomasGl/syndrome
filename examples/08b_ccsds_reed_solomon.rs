//! CCSDS 131.0-B-3 outer Reed-Solomon(255,223) coding, at every interleaving
//! depth the standard permits.
//!
//! Teaches: `CcsdsReedSolomon::new(interleaving)` builds a codec for
//! CCSDS's actual outer code -- an evaluation-based RS(255,223) with first
//! consecutive root 112 and primitive-element step 11 over `GF(2^8)`
//! (field polynomial `1 + x + x^2 + x^7 + x^8`) -- a genuinely different
//! mathematical construction from `syndrome::reed_solomon`'s Cauchy-matrix
//! erasure code (see both modules' docs for why one cannot stand in for
//! the other). This is the outer half of CCSDS 131.0-B-3's concatenated
//! FEC chain; see `08_ccsds_convolutional.rs` for the inner convolutional
//! code.
//!
//! What this does **not** cover: CCSDS 131.0-B-3's frame synchronization
//! markers and pseudo-randomization (scrambling) sequence are not
//! implemented anywhere in this crate. A real CCSDS 131.0-B-3 downlink
//! needs that framing and derandomization from elsewhere.
//!
//! Run with: `cargo run --example 08b_ccsds_reed_solomon`

use syndrome::CcsdsReedSolomon;

fn main() {
    println!(
        "CCSDS 131.0-B-3 outer Reed-Solomon(255,223) code, every permitted interleaving depth:\n"
    );

    for &interleaving in &[1usize, 2, 3, 4, 5, 8] {
        let rs = CcsdsReedSolomon::new(interleaving).expect("1,2,3,4,5,8 are all valid depths");

        // A deterministic but non-trivial data pattern, one full interleaved
        // block's worth (interleaving * 223 bytes).
        let data: Vec<u8> = (0..rs.data_len() as u32)
            .map(|i| ((i * 41 + 17) % 256) as u8)
            .collect();

        let mut block = vec![0u8; rs.block_len()];
        rs.encode(&data, &mut block).unwrap();

        // Corrupt up to 16 bytes in every underlying RS(255,223) stream --
        // the code's designed capacity (E=16) -- distributed across the
        // interleaved block the way a burst error on the channel would be.
        let errors_per_stream = 16usize;
        for stream in 0..interleaving {
            for e in 0..errors_per_stream {
                let pos = (e * interleaving + stream) % rs.block_len();
                block[pos] ^= 0x5A;
            }
        }

        let corrected = rs.decode(&mut block).unwrap();
        let recovered_ok = block[..rs.data_len()] == data[..];

        println!(
            "  interleaving={interleaving:<2} block_len={:<5} corrected={corrected:<4} data recovered: {}",
            rs.block_len(),
            if recovered_ok { "yes" } else { "NO" }
        );
        assert!(
            recovered_ok,
            "interleaving={interleaving}: data not recovered"
        );
    }

    println!(
        "\nHigher interleaving depths spread the same total per-stream error \
         budget (16 symbols per underlying RS(255,223) codeword) across a \
         wider channel-byte span, which is exactly why CCSDS specifies \
         interleaving at all: a real channel burst error corrupts \
         consecutive bytes, and interleaving turns a long burst that would \
         overwhelm one codeword into a short run inside each of several, \
         each still within its own correction capacity."
    );
}
