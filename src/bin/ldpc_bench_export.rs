//! LDPC decode benchmark exporter — times QC-LDPC LOMS decode (BG1, Z=384, 10 iter)
//! and writes bench/results/ldpc_rust.json for the dashboard Rust vs C++ comparison.
//!
//! Measures two kernel selections over the identical decode workload:
//! `"loms_runtime_simd"` (the default entry point, which probes the host CPU
//! at runtime and prefers AVX2 on x86_64 / NEON on aarch64) and
//! `"loms_scalar"` (forced onto the pure scalar fallback via
//! [`QcLdpcDecoder::decode_layered_offset_min_sum_scalar`] on every
//! architecture). Both share the same LOMS implementation and differ only
//! in which kernel is selected, so this is a fair scalar-vs-vectorized
//! comparison rather than two different algorithms.
//!
//! Usage: `cargo run --release --bin ldpc_bench_export`
//! Output: `bench/results/ldpc_rust.json`

use std::time::Instant;
use syndrome::qc_ldpc::{BaseGraph, QcLdpcDecoder};

const Z: usize = 384;
const DECODE_ITERS: usize = 10;
const BENCH_REPS: usize = 200;

/// Fill LLR with alternating +0.5 / -0.5 (matches C++ bench init).
fn init_llr(llr: &mut [f32]) {
    for (i, v) in llr.iter_mut().enumerate() {
        *v = if i & 1 == 0 { 0.5 } else { -0.5 };
    }
}

/// Time `BENCH_REPS` decode calls (resetting the LLR buffer before each rep)
/// and return the median nanoseconds/call, using `decode` to run either the
/// runtime-dispatched or forced-scalar kernel.
fn median_decode_ns(
    llr: &mut [f32],
    edge_r: &mut [f32],
    scratch: &mut [f32],
    hard: &mut [u8],
    mut decode: impl FnMut(&mut [f32], &mut [f32], &mut [f32], &mut [u8]),
) -> f64 {
    let mut samples: Vec<u128> = Vec::with_capacity(BENCH_REPS);
    for _ in 0..BENCH_REPS {
        init_llr(llr);
        let t0 = Instant::now();
        decode(llr, edge_r, scratch, hard);
        samples.push(t0.elapsed().as_nanos());
    }
    samples.sort_unstable();
    samples[BENCH_REPS / 2] as f64
}

fn main() {
    let out_dir = "bench/results";
    std::fs::create_dir_all(out_dir).expect("cannot create bench/results");

    let decoder = QcLdpcDecoder::with_lifting_size(BaseGraph::Bg1, Z, 0.25)
        .expect("BG1 Z=384 is a valid 3GPP lifting size");

    let n = decoder.variable_node_count(); // 68 * 384 = 26112
    let edge_buf_len = decoder.required_edge_buffer();
    let scratch_len = decoder.required_layer_buffer();

    let mut llr = vec![0.0f32; n];
    let mut edge_r = vec![0.0f32; edge_buf_len];
    let mut scratch = vec![0.0f32; scratch_len];
    let mut hard = vec![0u8; n];

    println!("BG1 Z={Z}: {n} variable nodes, {DECODE_ITERS} decode iterations, {BENCH_REPS} reps");

    // ── Runtime-dispatched kernel (AVX2/NEON if available, else scalar) ──
    let median_ns_simd = median_decode_ns(
        &mut llr,
        &mut edge_r,
        &mut scratch,
        &mut hard,
        |llr, edge_r, scratch, hard| {
            decoder
                .decode_layered_offset_min_sum(llr, edge_r, scratch, hard, DECODE_ITERS)
                .expect("decode failed");
        },
    );
    let melem_per_s_simd = (n as f64 * DECODE_ITERS as f64) / (median_ns_simd * 1e-9) / 1e6;

    println!("[loms_runtime_simd] Median ns/iter : {median_ns_simd:.1}");
    println!("[loms_runtime_simd] Melem/s        : {melem_per_s_simd:.2}");

    // ── Forced-scalar kernel (every architecture) ────────────────────────
    let median_ns_scalar = median_decode_ns(
        &mut llr,
        &mut edge_r,
        &mut scratch,
        &mut hard,
        |llr, edge_r, scratch, hard| {
            decoder
                .decode_layered_offset_min_sum_scalar(llr, edge_r, scratch, hard, DECODE_ITERS)
                .expect("decode failed");
        },
    );
    let melem_per_s_scalar = (n as f64 * DECODE_ITERS as f64) / (median_ns_scalar * 1e-9) / 1e6;

    println!("[loms_scalar]       Median ns/iter : {median_ns_scalar:.1}");
    println!("[loms_scalar]       Melem/s        : {melem_per_s_scalar:.2}");

    let record_simd = format!(
        r#"  {{"lang":"rust","impl":"loms_runtime_simd","shard_len":0,"data_shards":0,"parity_shards":0,"payload_bytes":{n},"ns_per_iter":{median_ns_simd:.1},"mib_per_s":0,"melem_per_s":{melem_per_s_simd:.2},"n_variable_nodes":{n},"n_iters":{DECODE_ITERS}}}"#
    );
    let record_scalar = format!(
        r#"  {{"lang":"rust","impl":"loms_scalar","shard_len":0,"data_shards":0,"parity_shards":0,"payload_bytes":{n},"ns_per_iter":{median_ns_scalar:.1},"mib_per_s":0,"melem_per_s":{melem_per_s_scalar:.2},"n_variable_nodes":{n},"n_iters":{DECODE_ITERS}}}"#
    );

    let json = format!("[\n{record_simd},\n{record_scalar}\n]\n");
    let json_path = format!("{out_dir}/ldpc_rust.json");
    std::fs::write(&json_path, &json).expect("cannot write ldpc_rust.json");
    println!("Wrote {json_path}");
}
