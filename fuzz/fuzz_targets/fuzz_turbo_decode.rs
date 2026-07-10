//! Fuzz target: `TurboDecoder::decode`.
//!
//! Byte-stream layout:
//! - `data[0]` -> selects `K` from the 8 supported 3GPP QPP block lengths
//!   (also implicitly covers "every unsupported K" by construction, since
//!   [`SUPPORTED_K`] is exhaustive and `TurboDecoder::new` rejects anything
//!   else -- exercised directly by the stable-toolchain robustness suite's
//!   `fuzz_turbo_unsupported_k` test instead of here, to keep this target
//!   focused on the decode hot path).
//! - `data[1]` -> `max_iters` in `1..=8`.
//! - `data[2..]` -> channel LLRs (`3*K + 12` of them), two bytes per value.

#![no_main]

use glezer_rsv::turbo::TurboDecoder;
use libfuzzer_sys::fuzz_target;

const SUPPORTED_K: [usize; 8] = [40, 104, 256, 512, 1024, 2048, 4096, 6144];

fn byte_pair_to_llr(hi: u8, lo: u8) -> f32 {
    match hi % 8 {
        0 => f32::NAN,
        1 => -f32::NAN,
        2 => f32::INFINITY,
        3 => f32::NEG_INFINITY,
        4 => 0.0,
        _ => {
            let raw = (hi as i16 - 128) * 256 + lo as i16;
            raw as f32 / 64.0
        }
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }
    let k = SUPPORTED_K[data[0] as usize % SUPPORTED_K.len()];
    let Ok(mut dec) = TurboDecoder::new(k) else {
        return;
    };
    let max_iters = 1 + (data[1] as usize % 8);

    let required = 3 * k + 12;
    let rest = &data[2..];
    if rest.is_empty() {
        return;
    }

    let mut llr = vec![0.0f32; required];
    for (i, slot) in llr.iter_mut().enumerate() {
        let hi = rest[(2 * i) % rest.len()];
        let lo = rest[(2 * i + 1) % rest.len()];
        *slot = byte_pair_to_llr(hi, lo);
    }

    let mut out = vec![0u8; k];
    let _ = dec.decode(&llr, &mut out, max_iters);

    // Deliberately too-short LLR buffer -- must return Err, not panic.
    let short_llr = &llr[..llr.len() / 2];
    let mut out2 = vec![0u8; k];
    let _ = dec.decode(short_llr, &mut out2, max_iters);
});
