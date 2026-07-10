//! Fuzz target: `PolarDecoder::decode_sc` / `decode_scl`.
//!
//! Byte-stream layout:
//! - `data[0]` -> `n = 2^(1 + data[0] % 8)`, i.e. `n` in `{2, 4, ..., 256}`.
//! - `data[1]` -> `k = data[1] % n` (includes the `k == 0` and `k == n-1`
//!   edges).
//! - `data[2]` -> SCL `list_size` in `1..=8`.
//! - `data[3..]` -> LLRs, two bytes per value via [`byte_pair_to_llr`],
//!   with periodic NaN/Inf/zero injection.

#![no_main]

use libfuzzer_sys::fuzz_target;
use syndrome::polar::PolarDecoder;

/// Map a `(hi, lo)` byte pair to an `f32` LLR: `hi % 8` selects an
/// occasional NaN/+-Inf/zero special case (probability 4/8), otherwise a
/// bounded normal value derived from both bytes.
fn byte_pair_to_llr(hi: u8, lo: u8) -> f32 {
    match hi % 8 {
        0 => f32::NAN,
        1 => -f32::NAN,
        2 => f32::INFINITY,
        3 => f32::NEG_INFINITY,
        4 => 0.0,
        _ => {
            let raw = (hi as i16 - 128) * 256 + lo as i16; // -32768..=32767
            raw as f32 / 64.0 // bounded, roughly +-512.0
        }
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }
    let n_pow = 1 + (data[0] % 8) as u32; // 1..=8 -> n in {2,...,256}
    let n = 1usize << n_pow;
    let k = (data[1] as usize) % n; // always < n; includes k=0 and k=n-1
    let list_size = 1 + (data[2] as usize % 8);

    let Ok(dec) = PolarDecoder::new(n, k, list_size, None) else {
        return;
    };

    let rest = &data[3..];
    if rest.is_empty() {
        return;
    }
    let mut llr = vec![0.0f32; n];
    for (i, slot) in llr.iter_mut().enumerate() {
        let hi = rest[(2 * i) % rest.len()];
        let lo = rest[(2 * i + 1) % rest.len()];
        *slot = byte_pair_to_llr(hi, lo);
    }

    let mut out_sc = vec![0u8; k];
    let _ = dec.decode_sc(&llr, &mut out_sc);

    let mut out_scl = vec![0u8; k];
    let _ = dec.decode_scl(&llr, &mut out_scl);

    // Also probe with a deliberately wrong-length LLR/out buffer.
    if !rest.is_empty() {
        let bad_len = n.saturating_sub(1).max(1);
        let bad_llr = vec![0.0f32; bad_len];
        let mut out3 = vec![0u8; k];
        let _ = dec.decode_sc(&bad_llr, &mut out3);
    }
});
