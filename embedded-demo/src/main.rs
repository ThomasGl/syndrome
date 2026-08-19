//! Bare-metal Cortex-M4F firmware demo of `syndrome`'s `no_std` QC-LDPC
//! path: encode a real BG2 (Z=128) codeword, quantize it to the crate's
//! fixed-point `i8` message format, and decode it back with the layered
//! offset min-sum kernel -- the same algorithm the `std` build runs, built
//! and linked here with no OS, no allocator but the tiny static heap below,
//! and no libc.
//!
//! # What this does and does not prove
//!
//! This links into a real ELF for `thumbv7em-none-eabihf` (a genuine
//! Cortex-M4F target) -- proof the `no_std` feature is not just a
//! `cargo check` checkbox, it produces a real firmware image, with a
//! measurable size (see the crate's `README.md`, section "5.6 no_std
//! embedded firmware footprint", for that number and how it was measured).
//!
//! It has **not** been run: there is no hardware or cycle-accurate
//! simulator available to verify execution or measure real throughput, and
//! this crate's own documentation policy is to publish no benchmark number
//! that was not actually measured. `memory.x`'s FLASH/RAM layout is
//! generic and illustrative (see that file), not a specific board; adapt
//! it before flashing anything built from this demo to real hardware.

#![no_std]
#![no_main]

extern crate alloc;

use alloc::vec;
use cortex_m_rt::entry;
use embedded_alloc::LlffHeap as Heap;
use panic_halt as _;
use syndrome::qc_ldpc::{BaseGraph, QcLdpcDecoder, QcLdpcEncoder};
use syndrome::quantize::{QuantParams, quantize_llr_i16};

#[global_allocator]
static HEAP: Heap = Heap::empty();

/// BG2 Z=128 int8 needs tens of KiB, not hundreds (see the crate's "Memory
/// reality" note) -- 64 KiB of static heap comfortably covers the encoder,
/// decoder, and every scratch buffer this demo allocates.
const HEAP_SIZE: usize = 64 * 1024;

#[entry]
fn main() -> ! {
    {
        static mut HEAP_MEM: [u8; HEAP_SIZE] = [0u8; HEAP_SIZE];
        // SAFETY: `entry` runs this closure exactly once, before anything
        // else touches `HEAP_MEM` or allocates through `HEAP` -- the
        // standard embedded-alloc initialization pattern.
        unsafe {
            #[allow(static_mut_refs)]
            HEAP.init(HEAP_MEM.as_mut_ptr() as usize, HEAP_SIZE);
        }
    }

    if run_ldpc_round_trip() {
        // Success: park here. A real board would blink an LED or emit over
        // UART/RTT instead -- both are board-specific, so this demo stops
        // at the point every board diverges rather than guessing one.
        loop {
            cortex_m::asm::wfi();
        }
    } else {
        // Failure: halt at a breakpoint a debugger can catch, rather than
        // spinning silently.
        cortex_m::asm::bkpt();
        loop {
            cortex_m::asm::wfi();
        }
    }
}

/// Encode, quantize, and decode one BG2 (Z=128) codeword through the exact
/// fixed-point path `tests/ldpc_int8_quantization_loss.rs` measures in the
/// `std` build -- see `src/quantize.rs`'s module doc for the measured
/// $E_b/N_0$ cost of this format. Returns whether the decoded info bits
/// matched the encoded input.
fn run_ldpc_round_trip() -> bool {
    let bg = BaseGraph::Bg2;
    let z = 128usize;

    let Ok(enc) = QcLdpcEncoder::new(bg, z) else {
        return false;
    };
    let Ok(dec) = QcLdpcDecoder::with_lifting_size(bg, z, 0.5) else {
        return false;
    };

    let k = enc.info_bit_count();
    let n = enc.codeword_bit_count();
    if n != dec.variable_node_count() {
        return false;
    }

    let info: alloc::vec::Vec<u8> = (0..k).map(|i| ((i * 7) % 3 == 0) as u8).collect();
    let mut codeword = vec![0u8; n];
    if enc.encode(&info, &mut codeword).is_err() {
        return false;
    }

    // Noiseless BPSK channel: 0 -> +5.0 LLR, 1 -> -5.0 LLR (strongly
    // confident, since this demo is a plumbing smoke test, not a BER
    // measurement -- see the caveat above on what this does and does not
    // prove).
    let quant = QuantParams::default();
    let llr: alloc::vec::Vec<f32> = codeword
        .iter()
        .map(|&b| if b == 0 { 5.0 } else { -5.0 })
        .collect();
    let mut app = vec![0i16; n];
    quantize_llr_i16(&llr, &mut app, quant.scale);

    let mut edge_r = vec![0i8; dec.required_edge_buffer()];
    let mut layer_scratch = vec![0i8; dec.required_layer_buffer()];
    let mut hard = vec![0u8; n];
    let Ok(_iterations_used) = dec.decode_layered_offset_min_sum_i8(
        &mut app,
        &mut edge_r,
        &mut layer_scratch,
        &mut hard,
        10,
        quant,
    ) else {
        return false;
    };

    hard[..k] == info[..]
}
