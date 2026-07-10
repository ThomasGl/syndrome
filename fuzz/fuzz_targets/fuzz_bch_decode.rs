//! Fuzz target: `BchCode::decode` / `BchCode::shortened_decode`.
//!
//! Byte-stream layout:
//! - `data[0]`  -> `t` in `1..=10` (error-correction capability).
//! - `data[1]`  -> `k_short` seed for the shortened-decode path.
//! - `data[2..]` -> codeword bits, one byte per bit position (cycled),
//!   mostly masked to `{0, 1}` but occasionally left as a raw garbage byte
//!   (`> 1`) to exercise BCH's behavior on out-of-contract bit values.

#![no_main]

use glezer_rsv::bch::BchCode;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }

    let t = 1 + (data[0] as usize % 10);
    let Ok(bch) = BchCode::new(t) else {
        return;
    };
    let n = bch.n();
    let k = bch.k();

    let rest = &data[2..];

    let mut codeword = vec![0u8; n];
    for (i, slot) in codeword.iter_mut().enumerate() {
        let b = rest[i % rest.len()];
        // Every 5th position keeps the raw byte (possibly > 1); the rest
        // are masked to a valid bit, so most cases still exercise the
        // "genuine near-codeword with some bit errors" path while a
        // minority probe the out-of-contract >1 case.
        *slot = if b % 5 == 0 { b } else { b & 1 };
    }
    let _ = bch.decode(&mut codeword);

    // Shortened decode with an adversarial k_short (including > k).
    let k_short = (data[1] as usize) % (k + 8);
    let out_len = k_short + bch.parity_len();
    if out_len > 0 {
        let mut short_cw = vec![0u8; out_len];
        for (i, slot) in short_cw.iter_mut().enumerate() {
            *slot = rest[i % rest.len()] & 1;
        }
        let _ = bch.shortened_decode(k_short, &mut short_cw);
    }

    // Also probe encode() with adversarial (possibly wrong-length, possibly
    // >1-valued) info bits derived from the same byte stream.
    let info_len = (data[1] as usize) % (k + 4);
    let mut info = vec![0u8; info_len];
    for (i, slot) in info.iter_mut().enumerate() {
        *slot = rest[i % rest.len()];
    }
    let mut cw_out = vec![0u8; n];
    let _ = bch.encode(&info, &mut cw_out);
});
