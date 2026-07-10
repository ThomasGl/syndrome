//! Fuzz target: `ReedSolomon::decode` under adversarial shard patterns.
//!
//! Byte-stream layout:
//! - `data[0]` -> `data_shards` in `1..=8`.
//! - `data[1]` -> `parity_shards` in `1..=4`.
//! - `data[2]` -> `shard_len` in `1..=16`.
//! - `data[3..]` -> shard content, then an erasure pattern (which shards
//!   become `None`) derived byte-by-byte from the same stream.

#![no_main]

use libfuzzer_sys::fuzz_target;
use syndrome::reed_solomon::ReedSolomon;

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }
    let d = 1 + (data[0] as usize % 8);
    let p = 1 + (data[1] as usize % 4);
    let shard_len = 1 + (data[2] as usize % 16);

    let rs = ReedSolomon::new(d, p);

    let rest = &data[3..];
    if rest.is_empty() {
        return;
    }

    let mut data_shards: Vec<Vec<u8>> = Vec::with_capacity(d);
    for i in 0..d {
        let mut shard = vec![0u8; shard_len];
        for (j, slot) in shard.iter_mut().enumerate() {
            *slot = rest[(i * shard_len + j) % rest.len()];
        }
        data_shards.push(shard);
    }
    let parity = match rs.encode(&data_shards) {
        Ok(p) => p,
        // Geometry is valid by construction, but a graceful Err must never
        // panic the harness either.
        Err(_) => return,
    };

    let mut shards: Vec<Option<Vec<u8>>> = Vec::with_capacity(d + p);
    for s in &data_shards {
        shards.push(Some(s.clone()));
    }
    for s in &parity {
        shards.push(Some(s.clone()));
    }

    // Byte-driven erasure pattern -- can erase anywhere from none to all
    // shards (including patterns that make recovery impossible, which must
    // return `Err`, never panic).
    for (i, slot) in shards.iter_mut().enumerate() {
        let control = rest[i % rest.len()];
        if control % 3 == 0 {
            *slot = None;
        }
    }
    let _ = rs.decode(&mut shards);

    // Also fuzz decode() directly against a raw byte-derived shard vector
    // shape (wrong total length, mismatched shard lengths) rather than a
    // guaranteed-consistent encode() output.
    let raw_total = 1 + (rest[0] as usize % (d + p + 4));
    let mut raw_shards: Vec<Option<Vec<u8>>> = Vec::with_capacity(raw_total);
    for i in 0..raw_total {
        let ctrl = rest[i % rest.len()];
        if ctrl % 4 == 0 {
            raw_shards.push(None);
        } else {
            let len = 1 + (ctrl as usize % 8);
            raw_shards.push(Some(vec![ctrl; len]));
        }
    }
    let _ = rs.decode(&mut raw_shards);
});
