//! Pipeline throughput benchmark: measures frames/s and Melem/s when the
//! LOMS decoder runs in a background thread behind two SPSC rings.
//!
//! Run:
//!   cargo run --release --bin ldpc_pipeline_bench
//!
//! Outputs bench/results/ldpc_pipeline_rust.json.

use glezer_rsv::{BaseGraph, LdpcPipeline, QcLdpcDecoder};
use std::time::Instant;

const DECODE_ITERS: usize = 10;
const BENCH_FRAMES: usize = 500; // total frames submitted for timing
const WARMUP_FRAMES: usize = 50; // discarded

fn main() {
    let decoder = QcLdpcDecoder::with_lifting_size(BaseGraph::Bg1, 384, 0.25).unwrap();
    let n = decoder.variable_node_count();
    println!("BG1 Z=384: {n} variable nodes, {DECODE_ITERS} decode iterations");
    println!("Pipeline pool: 16 slots (SPSC work+done rings, 1 worker thread)");

    let mut pipe = LdpcPipeline::new(decoder, DECODE_ITERS);

    // Warm-up: submit and drain WARMUP_FRAMES frames.
    let mut submitted = 0usize;
    let mut received = 0usize;
    while received < WARMUP_FRAMES {
        // Acquire and submit if pool has room.
        if submitted < WARMUP_FRAMES {
            if let Some(mut frame) = pipe.acquire() {
                frame.llr_mut().iter_mut().for_each(|v| *v = 5.0);
                pipe.submit(frame);
                submitted += 1;
            }
        }
        // Drain completed frames.
        if let Some(result) = pipe.try_recv() {
            pipe.release(result);
            received += 1;
        } else {
            std::hint::spin_loop();
        }
    }

    // Timed run: BENCH_FRAMES frames, start timer after first submit.
    submitted = 0;
    received = 0;
    let start = Instant::now();

    while received < BENCH_FRAMES {
        if submitted < BENCH_FRAMES {
            if let Some(mut frame) = pipe.acquire() {
                frame.llr_mut().iter_mut().for_each(|v| *v = 5.0);
                pipe.submit(frame);
                submitted += 1;
            }
        }
        if let Some(result) = pipe.try_recv() {
            pipe.release(result);
            received += 1;
        } else {
            std::hint::spin_loop();
        }
    }

    let elapsed_ns = start.elapsed().as_nanos() as f64;
    let ns_per_frame = elapsed_ns / BENCH_FRAMES as f64;
    let melem_per_s = n as f64 * DECODE_ITERS as f64 / (ns_per_frame * 1e-9) / 1e6;
    let frames_per_s = BENCH_FRAMES as f64 / (elapsed_ns * 1e-9);

    println!("Median ns/frame : {ns_per_frame:.0}");
    println!("Frames/s        : {frames_per_s:.1}");
    println!("Melem/s         : {melem_per_s:.2}");

    let result = serde_json::json!({
        "lang": "rust",
        "impl": "loms_pipeline_spsc",
        "backend": "avx2_if_available",
        "n_variable_nodes": n,
        "n_iters": DECODE_ITERS,
        "n_frames": BENCH_FRAMES,
        "ns_per_frame": ns_per_frame,
        "frames_per_s": frames_per_s,
        "melem_per_s": melem_per_s,
    });

    let out_path = "bench/results/ldpc_pipeline_rust.json";
    std::fs::write(out_path, serde_json::to_string_pretty(&result).unwrap()).unwrap();
    println!("Wrote {out_path}");
}
