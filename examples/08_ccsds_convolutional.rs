//! CCSDS 131.0-B-3 convolutional coding: the standard's whole K family.
//!
//! Teaches: `ViterbiDecoder::new(k)` for k in {3, 5, 7, 9} selects exactly
//! the connection polynomials CCSDS 131.0-B-3 §3 ("Convolutional Coding")
//! specifies for its rate-1/2 convolutional code family — k=7 (generators
//! 0o133/0o171) most notably, the historical "Voyager code" NASA/JPL flew
//! on Voyager and Cassini, and the same code many 3GPP profiles reuse. That
//! is a citable fact about the generator polynomials matching the published
//! standard, not a claim this has been checked bit-for-bit against real
//! CCSDS reference vectors or flight hardware -- no such vectors were
//! available to test against when this crate was built.
//!
//! This example covers the inner convolutional code only -- CCSDS
//! 131.0-B-3 also concatenates it with an *outer* Reed-Solomon (255,223)
//! code at a specific interleaving depth, which this crate implements
//! separately as `syndrome::ccsds_rs::CcsdsReedSolomon` (see
//! `08b_ccsds_reed_solomon.rs` for that half of the chain, and note it is
//! a genuinely different construction from `syndrome::reed_solomon`'s
//! Cauchy-matrix erasure code -- see both modules' docs). What neither
//! example nor this crate covers: CCSDS 131.0-B-3's frame synchronization
//! markers and pseudo-randomization (scrambling) sequence are not
//! implemented anywhere here. A real CCSDS 131.0-B-3 downlink needs that
//! framing and derandomization from elsewhere; this crate provides
//! conformant inner and outer channel codes.
//!
//! Run with: `cargo run --example 08_ccsds_convolutional`

use syndrome::channel_sim::AwgnChannel;
use syndrome::viterbi::ViterbiDecoder;

fn main() {
    // The CCSDS 131.0-B-3 rate-1/2 convolutional code family. k=7 is the
    // baseline every CCSDS mission profile built on this family starts
    // from; k=3/5/9 are the shorter/longer constraint-length variants the
    // standard also specifies.
    const CCSDS_CONSTRAINT_LENGTHS: [usize; 4] = [3, 5, 7, 9];

    println!(
        "CCSDS 131.0-B-3 rate-1/2 convolutional code family (inner code -- \
         see 08b_ccsds_reed_solomon.rs for the outer code):\n"
    );

    for &k in &CCSDS_CONSTRAINT_LENGTHS {
        let dec =
            ViterbiDecoder::new(k).expect("k in {3, 5, 7, 9} are all valid constraint lengths");
        let n_states = 1usize << (k - 1);

        // A 64-bit message, deterministic but not trivially periodic, so
        // decode failures would not hide behind an all-zero or all-one
        // pattern.
        let info: Vec<u8> = (0..64).map(|i| ((i * 11 + 5) % 7 < 3) as u8).collect();
        let coded = dec.encode(&info);

        // A moderately noisy channel -- harsh enough that a shorter
        // constraint length (weaker code) visibly struggles more than a
        // longer one, at the same seed and Eb/N0.
        let ebno_db = 1.5;
        let mut channel = AwgnChannel::new(ebno_db, 0.5, 100 + k as u64);
        let llrs = channel.transmit(&coded);
        let decoded = dec.decode_soft(&llrs);

        let errors = AwgnChannel::count_errors(&decoded, &info);
        let ber = AwgnChannel::bit_error_rate(&decoded, &info);

        println!(
            "  K={k:<2} ({n_states:>4} states): {errors:>2} bit errors / {} \
             (BER = {ber:.4}) at Eb/N0 = {ebno_db} dB",
            info.len()
        );
    }

    println!(
        "\nA longer constraint length gives the decoder a longer memory of past \
         input bits to weigh each decision against -- more trellis states to \
         search, but (usually) fewer residual errors at the same channel SNR. \
         This is exactly why CCSDS standardized the whole K=3..9 family rather \
         than a single fixed code: different missions trade decoder \
         complexity against coding gain differently."
    );
}
