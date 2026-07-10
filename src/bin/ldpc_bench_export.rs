//! LDPC decode benchmark exporter — times QC-LDPC LOMS decode (BG1, Z=384, 10 iter)
//! and writes bench/results/ldpc_rust.json for the dashboard Rust vs C++ comparison.
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

    // Collect one timing sample per rep (reset LLR each time so all reps are
    // equivalent — edge_r is re-zeroed inside decode_layered_offset_min_sum).
    let mut samples: Vec<u128> = Vec::with_capacity(BENCH_REPS);

    for _ in 0..BENCH_REPS {
        init_llr(&mut llr);
        let t0 = Instant::now();
        decoder
            .decode_layered_offset_min_sum(
                &mut llr,
                &mut edge_r,
                &mut scratch,
                &mut hard,
                DECODE_ITERS,
            )
            .expect("decode failed");
        samples.push(t0.elapsed().as_nanos());
    }

    // Median ns/iter
    samples.sort_unstable();
    let median_ns = samples[BENCH_REPS / 2] as f64;

    let melem_per_s = (n as f64 * DECODE_ITERS as f64) / (median_ns * 1e-9) / 1e6;

    println!("Median ns/iter : {median_ns:.1}");
    println!("Melem/s        : {melem_per_s:.2}");

    let record = format!(
        r#"  {{"lang":"rust","impl":"loms_scalar","shard_len":0,"data_shards":0,"parity_shards":0,"payload_bytes":{n},"ns_per_iter":{median_ns:.1},"mib_per_s":0,"melem_per_s":{melem_per_s:.2},"n_variable_nodes":{n},"n_iters":{DECODE_ITERS}}}"#
    );

    let json = format!("[\n{record}\n]\n");
    let json_path = format!("{out_dir}/ldpc_rust.json");
    std::fs::write(&json_path, &json).expect("cannot write ldpc_rust.json");
    println!("Wrote {json_path}");
}
