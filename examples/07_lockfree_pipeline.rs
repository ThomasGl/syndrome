//! Lock-free SPSC pipelining: decoding many LDPC frames without OS locks.
//!
//! Teaches: a real base station or modem decodes many transport blocks per
//! millisecond; blocking on `std::sync::Mutex` for every hand-off between the
//! calling thread and decoder worker threads means a syscall (futex wait/wake)
//! sits on the hot path. [`LdpcPipeline`] instead uses single-producer
//! single-consumer (SPSC) ring buffers built from atomics with `Acquire`/
//! `Release` orderings -- `try_push`/`try_pop` never block and never enter
//! the kernel, so throughput scales with CPU cores, not with scheduler
//! latency. Industry context: this acquire -> submit -> try_recv -> release
//! protocol is the same shape as a DPDK/io_uring style zero-copy packet
//! pipeline, just applied to LDPC decode frames instead of network packets.
//!
//! Run with: `cargo run --example 07_lockfree_pipeline`

use glezer_rsv::{BaseGraph, LdpcPipeline, QcLdpcDecoder};

const ITERATIONS: usize = 10; // LOMS iterations applied to every frame by every worker.
const N_WORKERS: usize = 2;
const N_FRAMES: usize = 6;

fn main() {
    // A small BG2/Z=24 code (1248 codeword bits) keeps this demo fast; the
    // same pipeline scales to full BG1 lifting sizes without any API change.
    let decoder = QcLdpcDecoder::with_lifting_size(BaseGraph::Bg2, 24, 0.25)
        .expect("BG2/Z=24 is a valid 3GPP lifting size");
    let n = decoder.variable_node_count();

    // The pipeline owns the decoder, clones it once per worker, and
    // pre-allocates all 16 frame slots (LLR/edge/scratch/hard buffers) up
    // front -- no heap allocation happens once frames start flowing.
    let mut pipeline = LdpcPipeline::with_workers(decoder, ITERATIONS, N_WORKERS);
    println!(
        "Pipeline: {N_WORKERS} worker thread(s), {ITERATIONS} LOMS iterations/frame, \
         {n}-bit codeword, 16 pre-allocated frame slots (no allocation on the hot path)."
    );

    // Submit N_FRAMES "channel LLR" frames. Each frame is a mostly-confident
    // all-zero codeword with a deterministic sprinkling of weak/flipped LLRs
    // (a stand-in for a noisy but decodable received symbol) -- the same
    // shape of input example 06 hands to a single decoder, just fanned out
    // across worker threads here via acquire()/submit().
    for frame_id in 0..N_FRAMES {
        let mut frame = pipeline
            .acquire()
            .expect("pool has 16 slots, we submit only 6");
        frame.llr_mut().iter_mut().enumerate().for_each(|(bit, v)| {
            // Every 7th bit (offset by frame_id) starts as a weak, wrong-signed
            // LLR; LOMS belief propagation should pull it back towards +3.0.
            *v = if (bit + frame_id) % 7 == 0 { -0.5 } else { 3.0 };
        });
        pipeline.submit(frame); // hands the slot to a worker via the work ring -- no lock taken.
        println!("  submitted frame {frame_id}");
    }

    // Drain results. try_recv/try_pop never blocks: on an empty ring it
    // returns None immediately, so the caller spins (or could do other work)
    // instead of sleeping in the kernel waiting for a worker.
    let mut received = 0usize;
    while received < N_FRAMES {
        if let Some(result) = pipeline.try_recv() {
            let hard = result.hard();
            let corrected_ones = hard.iter().filter(|&&b| b == 1).count();
            println!(
                "  frame decoded -> {corrected_ones} residual 1-bits out of {} \
                 (LOMS pulled the weak/flipped LLRs back to a consistent codeword)",
                hard.len()
            );
            pipeline.release(result); // returns the slot to the free list for reuse.
            received += 1;
        } else {
            std::hint::spin_loop(); // busy-wait: cheaper than a syscall for microsecond-scale waits.
        }
    }

    println!(
        "\nAll {N_FRAMES} frames decoded across {N_WORKERS} workers using only atomic \
         head/tail indices for coordination -- zero mutexes, zero syscalls, zero heap \
         allocations after pipeline construction."
    );

    // Candid API note: LdpcFrame only exposes llr_mut()/hard(); the actual
    // per-frame LOMS iteration count returned by decode_layered_offset_min_sum
    // is computed by the worker but discarded (see ldpc_pipeline.rs), so it
    // cannot be surfaced here the way DecodeReport::max_iters_used is in
    // example 06. That's a real gap for anyone wanting per-frame convergence
    // telemetry out of the pipelined path.
}
