//! Pipeline throughput benchmark: measures frames/s and Melem/s when the
//! LOMS decoder runs in a background thread behind two SPSC rings.
//!
//! Every frame carries a genuinely noisy LLR vector — a real BG1 Z=384
//! codeword, encoded with [`QcLdpcEncoder`], corrupted through
//! [`AwgnChannel`] at a fixed `Eb/N0` and PRNG seed (same channel model
//! `src/bin/ldpc_convergence_export.rs` uses) — instead of an all-`+5.0`
//! error-free codeword. An error-free codeword satisfies the syndrome check
//! on the first pass, so the decoder would return after one iteration no
//! matter how large the configured iteration budget is, making any
//! throughput figure derived from the configured budget fictional.
//!
//! Because the actual iteration count matters, `melem_per_s` here is derived
//! from the iteration counts [`syndrome::ldpc_pipeline::LdpcFrame::iterations_used`] reports for the
//! frames actually timed, not from the configured `DECODE_ITERS` budget.
//!
//! Run:
//!   cargo run --release --bin ldpc_pipeline_bench
//!
//! Outputs bench/results/ldpc_pipeline_rust.json.

use std::time::Instant;
use syndrome::channel_sim::AwgnChannel;
use syndrome::{BaseGraph, LdpcPipeline, QcLdpcDecoder, QcLdpcEncoder};

/// LOMS iteration budget passed to the pipeline. The decoder may return
/// early (see the module doc above) — the actual per-frame count is read
/// back via [`syndrome::LdpcFrame::iterations_used`], never assumed to
/// equal this constant.
const DECODE_ITERS: usize = 10;
const BENCH_FRAMES: usize = 500; // total frames submitted for timing
const WARMUP_FRAMES: usize = 50; // discarded

/// `Eb/N0` operating point for the AWGN channel. Chosen (same value used by
/// `src/bin/algo_bench_export.rs`'s QC-LDPC BG1 section) to sit near the
/// LOMS waterfall for this code, so decoding real noisy frames actually
/// spends multiple layered passes instead of exiting on the first one.
const EB_N0_DB: f32 = 2.0;
/// Fixed PRNG seed so the corrupted codeword — and therefore every reported
/// number — is exactly reproducible across runs.
const CHANNEL_SEED: u64 = 42;

fn main() {
    let z = 384usize;
    let encoder =
        QcLdpcEncoder::new(BaseGraph::Bg1, z).expect("BG1 Z=384 is a valid 3GPP lifting size");
    let decoder = QcLdpcDecoder::with_lifting_size(BaseGraph::Bg1, z, 0.25)
        .expect("BG1 Z=384 is a valid 3GPP lifting size");
    let n = decoder.variable_node_count();
    assert_eq!(
        n,
        encoder.codeword_bit_count(),
        "encoder/decoder must agree on codeword length"
    );
    println!("BG1 Z=384: {n} variable nodes, {DECODE_ITERS}-iteration LOMS budget");
    println!("Pipeline pool: 16 slots (SPSC work+done rings, 1 worker thread)");

    // Deterministic pseudo-random payload (Knuth multiplicative hash), same
    // style as `src/bin/ldpc_convergence_export.rs` and
    // `tests/media_reconstruction.rs`.
    let k = encoder.info_bit_count();
    let info_bits: Vec<u8> = (0..k)
        .map(|i| u8::from((i.wrapping_mul(2_654_435_761) >> 31) & 1 == 1))
        .collect();
    let mut codeword = vec![0u8; n];
    encoder
        .encode(&info_bits, &mut codeword)
        .expect("encode failed");

    // Corrupt the real codeword through a real AWGN channel. The same noisy
    // LLR vector is reused for every frame — exactly the pattern
    // `src/bin/algo_bench_export.rs`'s QC-LDPC decode benchmark uses — so
    // channel-simulation cost (Box-Muller draws) is paid once, up front,
    // and never pollutes the timed pipeline-throughput measurement.
    let code_rate = k as f32 / n as f32;
    let mut channel = AwgnChannel::new(EB_N0_DB, code_rate, CHANNEL_SEED);
    let noisy_llr = channel.transmit(&codeword);

    let mut pipe = LdpcPipeline::new(decoder, DECODE_ITERS);

    // Warm-up: submit and drain WARMUP_FRAMES frames.
    let mut submitted = 0usize;
    let mut received = 0usize;
    while received < WARMUP_FRAMES {
        // Acquire and submit if pool has room.
        if submitted < WARMUP_FRAMES {
            if let Some(mut frame) = pipe.acquire() {
                frame.llr_mut().copy_from_slice(&noisy_llr);
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
    let mut total_iterations_used = 0u64;
    let start = Instant::now();

    while received < BENCH_FRAMES {
        if submitted < BENCH_FRAMES {
            if let Some(mut frame) = pipe.acquire() {
                frame.llr_mut().copy_from_slice(&noisy_llr);
                pipe.submit(frame);
                submitted += 1;
            }
        }
        if let Some(result) = pipe.try_recv() {
            total_iterations_used += result.iterations_used() as u64;
            pipe.release(result);
            received += 1;
        } else {
            std::hint::spin_loop();
        }
    }

    let elapsed_ns = start.elapsed().as_nanos() as f64;
    let ns_per_frame = elapsed_ns / BENCH_FRAMES as f64;
    let frames_per_s = BENCH_FRAMES as f64 / (elapsed_ns * 1e-9);
    let mean_iters_per_frame = total_iterations_used as f64 / BENCH_FRAMES as f64;
    // Real element-updates actually performed, not `n * DECODE_ITERS`: every
    // frame's iteration count is read back from the decoder via
    // `iterations_used()`, so a frame that converges early (or one that
    // never converges and burns the whole budget) is counted for what it
    // actually cost, not for what a fixed budget assumes it cost.
    let total_elements = n as f64 * total_iterations_used as f64;
    let melem_per_s = total_elements / (elapsed_ns * 1e-9) / 1e6;

    println!("Median ns/frame        : {ns_per_frame:.0}");
    println!("Frames/s               : {frames_per_s:.1}");
    println!("Mean iterations/frame  : {mean_iters_per_frame:.2}");
    println!("Melem/s                : {melem_per_s:.2}");

    let result = serde_json::json!({
        "lang": "rust",
        "impl": "loms_pipeline_spsc",
        "backend": "avx2_if_available",
        "n_variable_nodes": n,
        "n_iters_budget": DECODE_ITERS,
        "n_frames": BENCH_FRAMES,
        "mean_iters_per_frame": mean_iters_per_frame,
        "ns_per_frame": ns_per_frame,
        "frames_per_s": frames_per_s,
        "melem_per_s": melem_per_s,
    });

    let out_path = "bench/results/ldpc_pipeline_rust.json";
    std::fs::write(out_path, serde_json::to_string_pretty(&result).unwrap()).unwrap();
    println!("Wrote {out_path}");
}
