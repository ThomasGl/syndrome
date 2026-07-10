//! Fuzz target: `QcLdpcDecoder::decode_layered_offset_min_sum` / `decode_5g`.
//!
//! Byte-stream layout:
//! - `data[0]` -> base graph (BG1/BG2, from the low bit).
//! - `data[1]` -> lifting size `Z`, chosen from a subset of valid 3GPP
//!   values (kept small so each fuzz iteration stays fast).
//! - `data[2]` -> offset-min-sum beta in `[0.0, 1.0]`.
//! - `data[3..]` -> LLRs (one per variable node), two bytes per value, plus
//!   a couple of trailing control bytes for `iterations`/`n_filler`.

#![no_main]

use libfuzzer_sys::fuzz_target;
use syndrome::qc_ldpc::{BaseGraph, QcLdpcDecoder};

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

// A subset of valid 3GPP lifting sizes, kept small so decode buffers (which
// scale with `num_col_blocks * Z`) stay cheap enough for many fuzz
// iterations per second.
const SMALL_VALID_Z: [usize; 10] = [2, 3, 4, 5, 6, 7, 8, 10, 12, 16];

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }
    let bg = if data[0] & 1 == 0 {
        BaseGraph::Bg1
    } else {
        BaseGraph::Bg2
    };
    let z = SMALL_VALID_Z[data[1] as usize % SMALL_VALID_Z.len()];
    let beta = (data[2] as f32) / 255.0;

    let Ok(dec) = QcLdpcDecoder::with_lifting_size(bg, z, beta) else {
        return;
    };
    let n = dec.variable_node_count();

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

    let mut edge_r = vec![0.0f32; dec.required_edge_buffer()];
    let mut scratch = vec![0.0f32; dec.required_layer_buffer()];
    let mut hard = vec![0u8; n];
    let iterations = 1 + (rest[0] as usize % 8);

    let mut llr_direct = llr.clone();
    let _ = dec.decode_layered_offset_min_sum(
        &mut llr_direct,
        &mut edge_r,
        &mut scratch,
        &mut hard,
        iterations,
    );

    // 5G wrapper with an adversarial n_filler (including > K).
    let n_filler =
        (rest.first().copied().unwrap_or(0) as usize) % (dec.info_bit_count_5g().saturating_add(8));
    let mut llr_5g = llr;
    let _ = dec.decode_5g(
        &mut llr_5g,
        n_filler,
        &mut edge_r,
        &mut scratch,
        &mut hard,
        iterations,
    );

    // Deliberately wrong-length buffers -- must return Err, not panic.
    let mut short_llr = vec![0.0f32; n.saturating_sub(1)];
    let _ = dec.decode_layered_offset_min_sum(
        &mut short_llr,
        &mut edge_r,
        &mut scratch,
        &mut hard,
        iterations,
    );
});
