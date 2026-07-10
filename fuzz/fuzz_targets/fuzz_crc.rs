//! Fuzz target: `Crc24::compute` / `attach` / `check`.
//!
//! Byte-stream layout:
//! - `data[0]` -> selects the [`CrcKind`].
//! - `data[1..]` -> raw bit-string, **not** masked to `{0, 1}` (every byte
//!   value 0..=255 is used as-is), directly exercising the ">1 bit value"
//!   contract question from the task brief.

#![no_main]

use glezer_rsv::crc::{Crc24, CrcKind};
use libfuzzer_sys::fuzz_target;

const KINDS: [CrcKind; 6] = [
    CrcKind::Crc24A,
    CrcKind::Crc24B,
    CrcKind::Crc24C,
    CrcKind::Crc16,
    CrcKind::Crc11,
    CrcKind::Crc6,
];

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    let kind = KINDS[data[0] as usize % KINDS.len()];
    let crc = Crc24::new(kind);

    // Deliberately unmasked bit values (0..=255, not just 0/1).
    let bits: Vec<u8> = data[1..].to_vec();
    let _ = crc.compute(&bits);
    let _ = crc.check(&bits);

    let mut attach_buf = bits.clone();
    crc.attach(&mut attach_buf);
    let _ = crc.check(&attach_buf);

    // Flip a byte-derived subset of bits post-attach and re-check -- must
    // never panic regardless of how many/which bits are corrupted.
    if !attach_buf.is_empty() {
        for (i, b) in attach_buf.iter_mut().enumerate() {
            let control = data[i % data.len()];
            if control % 7 == 0 {
                *b ^= 1;
            }
        }
        let _ = crc.check(&attach_buf);
    }
});
