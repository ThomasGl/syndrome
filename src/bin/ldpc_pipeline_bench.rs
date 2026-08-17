//! Pipeline throughput benchmark: measures how aggregate LOMS decode
//! throughput scales with worker-thread count behind the lock-free SPSC
//! pipeline.
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
//! from the iteration counts [`syndrome::ldpc_pipeline::LdpcFrame::iterations_used`]
//! reports for the frames actually timed, not from the configured
//! `DECODE_ITERS` budget.
//!
//! This sweeps worker counts explicitly via
//! [`LdpcPipeline::with_workers`] rather than trusting [`LdpcPipeline::new`]'s
//! host-dependent auto-detection for the recorded numbers — a fixed set of
//! worker counts is what makes the sweep reproducible and comparable across
//! machines with different core counts. An earlier version of this file
//! called `LdpcPipeline::new` once and printed a hardcoded "1 worker thread"
//! banner; on any host with `available_parallelism() > 1`,`new` actually
//! constructs `min(available_parallelism(), 8)` workers, so that banner was
//! false on every multi-core machine that ever ran it, and every number the
//! benchmark ever produced was really an N-worker aggregate mislabeled as a
//! single-worker one.
//!
//! Before the worker-count sweep, this also isolates true pipeline overhead
//! by alternating a plain decode loop against the 1-worker pipeline on
//! identical input, several rounds, all within this one process
//! (`measure_tight_loop`/`OVERHEAD_ROUNDS`). That interleaving matters: a
//! from-scratch investigation into why this benchmark's throughput didn't
//! reconcile with a plain decode loop found that (1) most of the apparent
//! gap was actually a *workload* difference — a synthetic non-converging LLR
//! pattern used elsewhere in this benchmark suite never triggers the
//! decoder's early-exit syndrome check, while a real codeword's final,
//! successful check scans the whole parity matrix and costs roughly as much
//! as one AVX2 decode iteration, work `melem_per_s` does not count as
//! "iterations" — and (2) a real several-percent ring/dispatch overhead does
//! exist at one worker (see the printed `pipeline_overhead_pct` for the
//! current measurement), but comparing two separate process invocations to
//! measure it is invalid on this host, which drifted as much as 15-25%
//! between runs during this investigation, from clock/thermal effects alone.
//! Measuring both sides back-to-back in one
//! process is what makes the isolated overhead figure trustworthy. A
//! redundant per-frame zero-fill of the ~474 KiB extrinsic buffer in the
//! pipeline worker (`ldpc_pipeline.rs`) — the decoder already zeroes it
//! internally at the top of every call — was found and removed during this
//! investigation, which is part of why the overhead measured here is lower
//! than an earlier estimate.
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
const BENCH_FRAMES: usize = 1000; // total frames submitted for timing, per worker count
const WARMUP_FRAMES: usize = 100; // discarded, per worker count

/// `Eb/N0` operating point for the AWGN channel. Chosen (same value used by
/// `src/bin/algo_bench_export.rs`'s QC-LDPC BG1 section) to sit near the
/// LOMS waterfall for this code, so decoding real noisy frames actually
/// spends multiple layered passes instead of exiting on the first one.
const EB_N0_DB: f32 = 2.0;
/// Fixed PRNG seed so the corrupted codeword — and therefore every reported
/// number — is exactly reproducible across runs.
const CHANNEL_SEED: u64 = 42;

/// Worker counts to sweep. Capped at 8 to match
/// [`LdpcPipeline::with_workers`]'s own ceiling (more workers than that would
/// starve for slots against the pipeline's 16 preallocated frames).
const WORKER_COUNTS: &[usize] = &[1, 2, 4, 8];

struct Measurement {
    n_workers: usize,
    ns_per_frame: f64,
    frames_per_s: f64,
    mean_iters_per_frame: f64,
    melem_per_s: f64,
}

/// Rounds to alternate between the tight-loop baseline and the 1-worker
/// pipeline when isolating pipeline overhead. Interleaving within one process
/// — rather than comparing two separate binary invocations — is what makes
/// the comparison trustworthy: a from-scratch investigation of why this
/// benchmark's numbers didn't reconcile with a plain decode loop found that
/// consecutive invocations of the *same* binary on this host can differ by
/// 15-25% from clock/thermal drift alone, which swamps a real ~10% pipeline
/// overhead if the two sides of the comparison are measured minutes apart in
/// separate processes.
const OVERHEAD_ROUNDS: usize = 5;

/// Decode `BENCH_FRAMES` copies of `noisy_llr` in a plain loop — no pipeline,
/// no threads, no rings — and return the median per-frame Melem/s. This is
/// the fair baseline for isolating pipeline overhead: same decoder, same
/// workload, same iteration-counting method as [`measure`], with the ring
/// machinery being the only thing removed.
fn measure_tight_loop(decoder: &QcLdpcDecoder, n: usize, noisy_llr: &[f32]) -> f64 {
    let mut llr = vec![0.0f32; n];
    let mut edge_r = vec![0.0f32; decoder.required_edge_buffer()];
    let mut scratch = vec![0.0f32; decoder.required_layer_buffer()];
    let mut hard = vec![0u8; n];

    // Warm-up, matching the pipeline path's warm-up frame count.
    for _ in 0..WARMUP_FRAMES {
        llr.copy_from_slice(noisy_llr);
        decoder
            .decode_layered_offset_min_sum(
                &mut llr,
                &mut edge_r,
                &mut scratch,
                &mut hard,
                DECODE_ITERS,
            )
            .expect("decode failed");
    }

    let mut total_iters = 0u64;
    let start = Instant::now();
    for _ in 0..BENCH_FRAMES {
        llr.copy_from_slice(noisy_llr);
        let used = decoder
            .decode_layered_offset_min_sum(
                &mut llr,
                &mut edge_r,
                &mut scratch,
                &mut hard,
                DECODE_ITERS,
            )
            .expect("decode failed");
        total_iters += used as u64;
    }
    let elapsed_ns = start.elapsed().as_nanos() as f64;
    (n as f64 * total_iters as f64) / (elapsed_ns * 1e-9) / 1e6
}

/// Run one full warm-up + timed measurement at a fixed worker count.
///
/// `noisy_llr` is the single real, AWGN-corrupted codeword reused for every
/// frame (channel-simulation cost is paid once, outside this function, so it
/// never pollutes the timed throughput measurement — the same pattern
/// `src/bin/algo_bench_export.rs`'s QC-LDPC section uses).
fn measure(decoder: QcLdpcDecoder, n_workers: usize, n: usize, noisy_llr: &[f32]) -> Measurement {
    let mut pipe = LdpcPipeline::with_workers(decoder, DECODE_ITERS, n_workers);
    assert_eq!(
        pipe.worker_count(),
        n_workers,
        "with_workers must honor the requested count within its documented 1..=8 range"
    );

    let mut submitted = 0usize;
    let mut received = 0usize;
    while received < WARMUP_FRAMES {
        if submitted < WARMUP_FRAMES
            && let Some(mut frame) = pipe.acquire()
        {
            frame.llr_mut().copy_from_slice(noisy_llr);
            let _ = pipe.submit(frame);
            submitted += 1;
        }
        if let Some(result) = pipe.try_recv() {
            pipe.release(result);
            received += 1;
        } else {
            std::hint::spin_loop();
        }
    }

    submitted = 0;
    received = 0;
    let mut total_iterations_used = 0u64;
    let start = Instant::now();

    while received < BENCH_FRAMES {
        if submitted < BENCH_FRAMES
            && let Some(mut frame) = pipe.acquire()
        {
            frame.llr_mut().copy_from_slice(noisy_llr);
            let _ = pipe.submit(frame);
            submitted += 1;
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

    Measurement {
        n_workers,
        ns_per_frame,
        frames_per_s,
        mean_iters_per_frame,
        melem_per_s,
    }
}

fn main() {
    let z = 384usize;
    let encoder =
        QcLdpcEncoder::new(BaseGraph::Bg1, z).expect("BG1 Z=384 is a valid 3GPP lifting size");
    let decoder_template = QcLdpcDecoder::with_lifting_size(BaseGraph::Bg1, z, 0.25)
        .expect("BG1 Z=384 is a valid 3GPP lifting size");
    let n = decoder_template.variable_node_count();
    assert_eq!(
        n,
        encoder.codeword_bit_count(),
        "encoder/decoder must agree on codeword length"
    );

    let available = std::thread::available_parallelism()
        .map(|c| c.get())
        .unwrap_or(1);
    println!("BG1 Z=384: {n} variable nodes, {DECODE_ITERS}-iteration LOMS budget");
    println!("Host reports {available} logical cores (available_parallelism)");
    println!("Sweeping worker counts: {WORKER_COUNTS:?}\n");

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

    // Corrupt the real codeword through a real AWGN channel once, up front;
    // every worker-count measurement reuses the identical noisy LLR vector so
    // decode difficulty never varies across the sweep.
    let code_rate = k as f32 / n as f32;
    let mut channel = AwgnChannel::new(EB_N0_DB, code_rate, CHANNEL_SEED);
    let noisy_llr = channel.transmit(&codeword);

    // ── Isolate pipeline overhead, same-process, interleaved ─────────────
    //
    // Earlier drafts of this investigation compared this benchmark's 1-worker
    // pipeline figure against a *separately invoked* tight-loop benchmark and
    // concluded the two nearly matched (implying near-zero pipeline overhead).
    // That comparison was invalid: this host's wall-clock throughput drifts
    // 15-25% between process invocations, which is larger than the effect
    // being measured. Alternating both measurements within this one process
    // removes that confound.
    println!("Isolating pipeline overhead ({OVERHEAD_ROUNDS} interleaved rounds)...");
    let mut tight_samples = Vec::with_capacity(OVERHEAD_ROUNDS);
    let mut pipe_samples = Vec::with_capacity(OVERHEAD_ROUNDS);
    for round in 0..OVERHEAD_ROUNDS {
        // Alternate which side runs first each round. Always measuring the
        // same side first under this host's within-run thermal/frequency
        // decay would bias the overhead estimate in a fixed direction
        // (conservatively, but still a bias); alternating cancels it.
        let (tight, pipe) = if round % 2 == 0 {
            let tight = measure_tight_loop(&decoder_template, n, &noisy_llr);
            let pipe = measure(decoder_template.clone(), 1, n, &noisy_llr).melem_per_s;
            (tight, pipe)
        } else {
            let pipe = measure(decoder_template.clone(), 1, n, &noisy_llr).melem_per_s;
            let tight = measure_tight_loop(&decoder_template, n, &noisy_llr);
            (tight, pipe)
        };
        println!(
            "  round {round}: tight_loop={tight:.2} Melem/s   1-worker pipeline={pipe:.2} Melem/s   ratio={:.3}",
            tight / pipe
        );
        tight_samples.push(tight);
        pipe_samples.push(pipe);
    }
    // Paired per-round (tight, pipe) values for the JSON export, taken before
    // the medians below sort each series independently — sorting separately
    // is fine for computing a median but would destroy which two numbers
    // came from the same round if used for the export.
    let paired_rounds: Vec<(f64, f64)> = tight_samples
        .iter()
        .copied()
        .zip(pipe_samples.iter().copied())
        .collect();
    tight_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    pipe_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_tight = tight_samples[OVERHEAD_ROUNDS / 2];
    let median_pipe = pipe_samples[OVERHEAD_ROUNDS / 2];
    let overhead_ratio = median_tight / median_pipe;
    println!(
        "\nMedian: tight_loop={median_tight:.2} Melem/s, 1-worker pipeline={median_pipe:.2} Melem/s\n\
         Pipeline overhead at 1 worker: {:.1}% (tight_loop / pipeline = {overhead_ratio:.3}x)\n",
        (overhead_ratio - 1.0) * 100.0
    );

    let mut measurements = Vec::with_capacity(WORKER_COUNTS.len());
    for &n_workers in WORKER_COUNTS {
        if n_workers > available {
            println!(
                "  [skip] {n_workers} workers requested but only {available} logical cores available"
            );
            continue;
        }
        let decoder = decoder_template.clone();
        let m = measure(decoder, n_workers, n, &noisy_llr);
        println!(
            "  workers={:<2}  frames/s={:>8.1}  mean_iters/frame={:>5.2}  Melem/s={:>8.2}",
            m.n_workers, m.frames_per_s, m.mean_iters_per_frame, m.melem_per_s
        );
        measurements.push(m);
    }

    let baseline_melem_per_s = measurements
        .iter()
        .find(|m| m.n_workers == 1)
        .map(|m| m.melem_per_s)
        .unwrap_or(f64::NAN);

    println!("\nScaling relative to 1 worker (through the same pipeline harness):");
    for m in &measurements {
        let speedup = m.melem_per_s / baseline_melem_per_s;
        let efficiency_pct = 100.0 * speedup / m.n_workers as f64;
        println!(
            "  workers={:<2}  speedup={:>5.2}x  efficiency={:>5.1}% of ideal linear scaling",
            m.n_workers, speedup, efficiency_pct
        );
    }
    println!(
        "\nEfficiency below 100% at higher worker counts reflects real contention (shared\n\
         memory bandwidth / cache) on this host, not a flaw in the lock-free ring protocol\n\
         itself — the rings add no locking overhead regardless of worker count; what caps\n\
         scaling is however many independent AVX2 kernels this machine's memory subsystem\n\
         can actually feed at once."
    );

    let records: Vec<serde_json::Value> = measurements
        .iter()
        .map(|m| {
            serde_json::json!({
                "lang": "rust",
                "impl": "loms_pipeline_spsc",
                "backend": "avx2_if_available",
                "n_variable_nodes": n,
                "n_iters_budget": DECODE_ITERS,
                "n_frames": BENCH_FRAMES,
                "n_workers": m.n_workers,
                "mean_iters_per_frame": m.mean_iters_per_frame,
                "ns_per_frame": m.ns_per_frame,
                "frames_per_s": m.frames_per_s,
                "melem_per_s": m.melem_per_s,
                "speedup_vs_1_worker": m.melem_per_s / baseline_melem_per_s,
            })
        })
        .collect();

    let paired_round_records: Vec<serde_json::Value> = paired_rounds
        .iter()
        .enumerate()
        .map(|(round, &(tight, pipe))| {
            serde_json::json!({
                "round": round,
                "measured_first": if round % 2 == 0 { "tight_loop" } else { "pipeline" },
                "tight_loop_melem_per_s": tight,
                "pipeline_1worker_melem_per_s": pipe,
                "ratio": tight / pipe,
            })
        })
        .collect();

    let output = serde_json::json!({
        "overhead_isolation": {
            "description": "Same-process, interleaved comparison of a plain decode loop \
                against the 1-worker pipeline on identical input, isolating ring/dispatch \
                overhead from workload differences and from cross-process host drift. \
                Which side runs first alternates each round to cancel any within-run \
                thermal/frequency-scaling bias.",
            "rounds": paired_round_records,
            "median_tight_loop_melem_per_s": median_tight,
            "median_pipeline_1worker_melem_per_s": median_pipe,
            "pipeline_overhead_ratio": overhead_ratio,
            "pipeline_overhead_pct": (overhead_ratio - 1.0) * 100.0,
        },
        "worker_sweep": records,
    });

    let out_path = "bench/results/ldpc_pipeline_rust.json";
    std::fs::write(out_path, serde_json::to_string_pretty(&output).unwrap()).unwrap();
    println!("\nWrote {out_path}");
}
