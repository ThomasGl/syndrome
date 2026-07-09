//! Reed-Solomon RS(10,4): surviving packet loss, not just bit flips.
//!
//! Teaches: RS codes operate on whole *symbols* (here: 1 KiB shards) rather
//! than individual bits, which makes them a natural fit for *erasures* --
//! packets that are entirely missing rather than merely corrupted. Industry
//! context: this is exactly how video-streaming CDNs and object-storage
//! systems (and QUIC/FEC transports) recover from lost UDP packets or dead
//! disks without a retransmission round-trip.
//!
//! Run with: `cargo run --example 03_reed_solomon_packet_loss`

use glezer_rsv::ReedSolomon;

const SHARD_LEN: usize = 1024; // 1 KiB per shard, like a network packet or disk block.
const DATA_SHARDS: usize = 10;
const PARITY_SHARDS: usize = 4;

/// Deterministic pseudo-random byte generator (xorshift32) so every run
/// produces identical "video frame" data -- no external RNG crate needed.
fn fill_deterministic(seed: u32, len: usize) -> Vec<u8> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state & 0xFF) as u8
        })
        .collect()
}

fn main() {
    // RS(10,4): 10 data shards + 4 parity shards. Think of this as one GOP
    // (group of pictures) sliced into 10 packets, with 4 extra "repair"
    // packets computed from them.
    let rs = ReedSolomon::new(DATA_SHARDS, PARITY_SHARDS);
    println!(
        "Reed-Solomon RS({DATA_SHARDS},{PARITY_SHARDS}): {DATA_SHARDS} data shards + \
         {PARITY_SHARDS} parity shards, {SHARD_LEN} bytes each (like {DATA_SHARDS} \
         video-stream packets protected by {PARITY_SHARDS} repair packets)."
    );

    // Build 10 deterministic "video packets".
    let data_shards: Vec<Vec<u8>> = (0..DATA_SHARDS)
        .map(|i| fill_deterministic(0xC0FFEE + i as u32, SHARD_LEN))
        .collect();

    // Encode: parity shards are linear combinations of the data shards over
    // GF(256), computed once and sent alongside the data.
    let parity_shards = rs.encode(&data_shards);
    println!(
        "Encoded {} parity shards ({} bytes each).",
        parity_shards.len(),
        SHARD_LEN
    );

    // Assemble the full 14-shard transmission: data shards first, then parity.
    let mut shards: Vec<Option<Vec<u8>>> = Vec::with_capacity(DATA_SHARDS + PARITY_SHARDS);
    shards.extend(data_shards.iter().cloned().map(Some));
    shards.extend(parity_shards.iter().cloned().map(Some));

    // Simulate network packet loss: 4 packets vanish in transit -- a mix of
    // data and parity shards, which RS handles identically (it doesn't care
    // *which* symbols are missing, only *how many*).
    let lost_indices = [1usize, 4, 7, 11]; // 2 data + 1 data + 1 parity
    for &idx in &lost_indices {
        shards[idx] = None;
    }
    println!(
        "\nSimulated packet loss: shards {lost_indices:?} dropped ({} of {} total) -- \
         exactly at the {PARITY_SHARDS}-shard recovery limit.",
        lost_indices.len(),
        DATA_SHARDS + PARITY_SHARDS
    );

    // Recover: RS(10,4) can reconstruct up to `parity_shards` (4) missing
    // shards, whether they were data or parity, by solving a small linear
    // system over GF(256) using whichever shards did arrive.
    rs.decode(&mut shards)
        .expect("4 erasures is exactly the RS(10,4) recovery limit");

    // Verify every recovered data shard matches the original bytes exactly.
    let mut all_ok = true;
    for i in 0..DATA_SHARDS {
        let recovered = shards[i].as_ref().expect("data shard must be filled in");
        let matches = recovered == &data_shards[i];
        all_ok &= matches;
        if lost_indices.contains(&i) {
            println!("  shard {i} (was lost): reconstructed bytes match original = {matches}");
        }
    }

    println!(
        "\nAll {DATA_SHARDS} data shards verified byte-for-byte identical to the \
         originals: {all_ok}. The video frame is fully intact -- zero \
         retransmissions were needed."
    );
    assert!(all_ok, "recovered data must exactly match the source data");
}
