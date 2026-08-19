//! Bare-metal Cortex-M4F firmware demo of `syndrome`'s `no_std` QC-LDPC
//! path: encode a real BG2 (Z=128) codeword, quantize it to the crate's
//! fixed-point `i8` message format, and decode it back with the layered
//! offset min-sum kernel -- the same algorithm the `std` build runs, built
//! and linked here with no OS, no allocator but the static heap below, and
//! no libc.
//!
//! # What this does and does not prove
//!
//! This links into a real ELF for `thumbv7em-none-eabihf` (a genuine
//! Cortex-M4F target) -- proof the `no_std` feature is not just a
//! `cargo check` checkbox, it produces a real firmware image, with a
//! measurable size (see the crate's `README.md`, section "5.6 no_std
//! embedded firmware footprint", for that number and how it was measured).
//!
//! It **has** been run: under QEMU's `netduinoplus2` (Cortex-M4F) machine
//! model, over ARM semihosting -- see `README.md`'s "Running under QEMU"
//! section for the exact command and its actual output. That is real
//! execution of this real firmware image encoding, quantizing, and decoding
//! a real codeword, not a host-side simulation of the algorithm standing in
//! for one. That execution caught a real bug in this demo the first time it
//! ran: 64 KiB of static heap (the original, unverified guess) was not
//! enough and failed a real allocation; 96 KiB is the measured minimum that
//! runs to completion, and `HEAP_SIZE` below ships with headroom above that
//! floor rather than sitting exactly on it. See "Running under QEMU" for
//! how to reproduce the bisection.
//!
//! What it is **not**: a hardware timing measurement. This firmware also
//! tried reporting a DWT cycle-counter delta around each phase; every
//! reading came back exactly zero under `netduinoplus2` -- QEMU's Cortex-M4
//! model for this machine does not implement the DWT cycle counter, a real
//! (if regrettable) gap in QEMU's own peripheral emulation, not a bug in
//! this demo. Reporting a number that is always zero would be worse than
//! reporting none, so none is reported: this crate's policy is to publish
//! no benchmark it did not actually measure, and "the counter reads zero"
//! is not a measurement of decode cost. A real hardware timing (or a
//! simulator that actually implements DWT) is still open work.
//!
//! `memory.x`'s FLASH/RAM layout is generic and illustrative (see that
//! file) -- comfortably inside `netduinoplus2`'s real STM32F405 capacity,
//! which is why it runs under QEMU without modification, but still not a
//! specific verified board; adjust it before flashing anything built from
//! this demo to real hardware.

#![no_std]
#![no_main]

extern crate alloc;

use alloc::vec;
use cortex_m_rt::entry;
use cortex_m_semihosting::{debug, hprintln};
use embedded_alloc::LlffHeap as Heap;
use panic_semihosting as _;
use syndrome::qc_ldpc::{BaseGraph, QcLdpcDecoder, QcLdpcEncoder};
use syndrome::quantize::{QuantParams, quantize_llr_i16};

#[global_allocator]
static HEAP: Heap = Heap::empty();

/// Measured under QEMU (see the module doc): 64 KiB fails a real allocation
/// partway through decode; 96 KiB is the smallest size that ran to
/// completion in that bisection. 128 KiB ships here as that measured floor
/// plus headroom, not the floor itself -- a peak-allocation profile on real
/// hardware would be needed to trust a number any tighter than this.
const HEAP_SIZE: usize = 128 * 1024;

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

    let report = run_ldpc_round_trip();

    hprintln!(
        "syndrome-embedded-demo: bg=BG2 z=128 k={} n={} iterations_used={}",
        report.k,
        report.n,
        report.iterations_used
    );

    if report.decoded_correctly {
        hprintln!("RESULT: PASS -- decoded info bits matched the encoded input exactly");
        debug::exit(debug::EXIT_SUCCESS);
    } else {
        hprintln!("RESULT: FAIL -- decoded info bits did NOT match the encoded input");
        debug::exit(debug::EXIT_FAILURE);
    }

    // `debug::exit` above requests QEMU terminate the process (it does, on
    // the netduinoplus2 + `-semihosting-config enable=on,target=native`
    // setup this demo is run under -- see README.md). On real hardware
    // there is no host to exit to, so this loop is what a board actually
    // falls into.
    loop {
        cortex_m::asm::wfi();
    }
}

/// Outcome of one [`run_ldpc_round_trip`] call: enough to report a
/// meaningful line over semihosting without a `Debug` impl (not available
/// without pulling in `core::fmt` formatting machinery this demo otherwise
/// has no reason to need).
struct RoundTripReport {
    k: usize,
    n: usize,
    iterations_used: usize,
    decoded_correctly: bool,
}

/// Encode, quantize, and decode one BG2 (Z=128) codeword through the exact
/// fixed-point path `tests/ldpc_int8_quantization_loss.rs` measures in the
/// `std` build -- see `src/quantize.rs`'s module doc for the measured
/// $E_b/N_0$ cost of this format.
fn run_ldpc_round_trip() -> RoundTripReport {
    let bg = BaseGraph::Bg2;
    let z = 128usize;

    let enc = QcLdpcEncoder::new(bg, z).expect("BG2 Z=128 is a valid 3GPP lifting size");
    let dec = QcLdpcDecoder::with_lifting_size(bg, z, 0.5)
        .expect("BG2 Z=128 is a valid 3GPP lifting size");

    let k = enc.info_bit_count();
    let n = enc.codeword_bit_count();
    assert_eq!(
        n,
        dec.variable_node_count(),
        "encoder/decoder built from the same (bg, z) must agree on N"
    );

    let info: alloc::vec::Vec<u8> = (0..k).map(|i| ((i * 7) % 3 == 0) as u8).collect();
    let mut codeword = vec![0u8; n];
    enc.encode(&info, &mut codeword)
        .expect("info/codeword buffers are sized from the same encoder");

    // Noiseless BPSK channel: 0 -> +5.0 LLR, 1 -> -5.0 LLR (strongly
    // confident, since this demo is a plumbing smoke test, not a BER
    // measurement -- see the module doc above on what this does and does
    // not prove).
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
    let iterations_used = dec
        .decode_layered_offset_min_sum_i8(
            &mut app,
            &mut edge_r,
            &mut layer_scratch,
            &mut hard,
            10,
            quant,
        )
        .expect("buffers are sized from required_edge_buffer()/required_layer_buffer()");

    RoundTripReport {
        k,
        n,
        iterations_used,
        decoded_correctly: hard[..k] == info[..],
    }
}
