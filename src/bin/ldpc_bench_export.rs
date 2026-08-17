//! LDPC decode benchmark exporter — times QC-LDPC LOMS decode (BG1, Z=384, 10 iter)
//! and writes bench/results/ldpc_rust.json for the dashboard Rust vs C++ comparison.
//!
//! Measures four kernel selections over the identical decode workload:
//! `"loms_runtime_simd"` (the default entry point, which probes the host CPU
//! at runtime and prefers AVX2 on x86_64 / NEON on aarch64) and
//! `"loms_scalar"` (forced onto the pure scalar fallback via
//! [`QcLdpcDecoder::decode_layered_offset_min_sum_scalar`] on every
//! architecture), then the same pair for the fixed-point path,
//! `"loms_i8_runtime_simd"` and `"loms_i8_scalar"`. Within each number
//! format the two entries share one LOMS implementation and differ only in
//! which kernel is selected, so each is a fair scalar-vs-vectorized
//! comparison rather than two different algorithms.
//!
//! The fixed-point entries decode the *same* channel values, quantized: the
//! `f32` LLR buffer is passed through
//! [`syndrome::quantize::quantize_llr_i16`] at the crate's default scale.
//! Comparing the two formats on throughput is only meaningful because the
//! error-rate cost of that quantization has been measured separately and is
//! small — see `tests/ldpc_int8_quantization_loss.rs`. The cross-language
//! checksum below stays on the `f32` scalar kernel, since the C++ reference
//! has no fixed-point path.
//!
//! Usage: `cargo run --release --bin ldpc_bench_export`
//! Output: `bench/results/ldpc_rust.json`

use std::time::Instant;
use syndrome::qc_ldpc::{BaseGraph, QcLdpcDecoder};
use syndrome::quantize::{QuantParams, quantize_llr_i16};

const Z: usize = 384;
const DECODE_ITERS: usize = 10;
const BENCH_REPS: usize = 200;

/// Fill LLR with alternating +0.5 / -0.5 (matches C++ bench init).
fn init_llr(llr: &mut [f32]) {
    for (i, v) in llr.iter_mut().enumerate() {
        *v = if i & 1 == 0 { 0.5 } else { -0.5 };
    }
}

/// FNV-1a over the decoder's hard-decision output, for the cross-language
/// correctness gate.
///
/// The Reed-Solomon gate compares parity bytes for exact equality because
/// GF(256) encoding is integer arithmetic: any two correct implementations
/// must agree bit for bit. LOMS is floating point, and `g++ -O3
/// -march=native` and `rustc` are free to contract multiply-adds into FMAs,
/// reassociate, and vectorize differently — so identical `f32` LLRs are not
/// a property either compiler guarantees, and demanding them would produce a
/// gate that fails for reasons unrelated to correctness.
///
/// Hashing the *hard decisions* instead compares the thing that actually has
/// to match: which codeword the decoder settled on. That is an integer
/// quantity, it is what every downstream stage consumes, and it is invariant
/// to the last-ulp differences the compilers are entitled to introduce.
fn hard_decision_checksum(hard: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in hard {
        h ^= u64::from(b & 1);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
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

/// [`median_decode_ns`] for the fixed-point path: the posterior buffer is
/// re-quantized from the same `f32` LLR values before every rep, so each
/// timed call starts from exactly the state the `f32` measurement starts
/// from.
fn median_decode_ns_i8(
    llr: &[f32],
    app: &mut [i16],
    edge_r: &mut [i8],
    scratch: &mut [i8],
    hard: &mut [u8],
    quant: QuantParams,
    mut decode: impl FnMut(&mut [i16], &mut [i8], &mut [i8], &mut [u8]),
) -> f64 {
    let mut samples: Vec<u128> = Vec::with_capacity(BENCH_REPS);
    for _ in 0..BENCH_REPS {
        quantize_llr_i16(llr, app, quant.scale);
        let t0 = Instant::now();
        decode(app, edge_r, scratch, hard);
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

    // ── Fixed-point kernels over the same workload ───────────────────────
    let quant = QuantParams::default();
    let mut app = vec![0i16; n];
    let mut edge_r_i8 = vec![0i8; edge_buf_len];
    let mut scratch_i8 = vec![0i8; scratch_len];
    init_llr(&mut llr);
    let llr_ref = llr.clone();

    let median_ns_i8_simd = median_decode_ns_i8(
        &llr_ref,
        &mut app,
        &mut edge_r_i8,
        &mut scratch_i8,
        &mut hard,
        quant,
        |app, edge_r, scratch, hard| {
            decoder
                .decode_layered_offset_min_sum_i8(app, edge_r, scratch, hard, DECODE_ITERS, quant)
                .expect("decode failed");
        },
    );
    let melem_per_s_i8_simd = (n as f64 * DECODE_ITERS as f64) / (median_ns_i8_simd * 1e-9) / 1e6;

    println!("[loms_i8_runtime_simd] Median ns/iter : {median_ns_i8_simd:.1}");
    println!("[loms_i8_runtime_simd] Melem/s        : {melem_per_s_i8_simd:.2}");

    let median_ns_i8_scalar = median_decode_ns_i8(
        &llr_ref,
        &mut app,
        &mut edge_r_i8,
        &mut scratch_i8,
        &mut hard,
        quant,
        |app, edge_r, scratch, hard| {
            decoder
                .decode_layered_offset_min_sum_i8_scalar(
                    app,
                    edge_r,
                    scratch,
                    hard,
                    DECODE_ITERS,
                    quant,
                )
                .expect("decode failed");
        },
    );
    let melem_per_s_i8_scalar =
        (n as f64 * DECODE_ITERS as f64) / (median_ns_i8_scalar * 1e-9) / 1e6;

    println!("[loms_i8_scalar]       Median ns/iter : {median_ns_i8_scalar:.1}");
    println!("[loms_i8_scalar]       Melem/s        : {melem_per_s_i8_scalar:.2}");

    let record_simd = format!(
        r#"  {{"lang":"rust","impl":"loms_runtime_simd","shard_len":0,"data_shards":0,"parity_shards":0,"payload_bytes":{n},"ns_per_iter":{median_ns_simd:.1},"mib_per_s":0,"melem_per_s":{melem_per_s_simd:.2},"n_variable_nodes":{n},"n_iters":{DECODE_ITERS}}}"#
    );
    let record_scalar = format!(
        r#"  {{"lang":"rust","impl":"loms_scalar","shard_len":0,"data_shards":0,"parity_shards":0,"payload_bytes":{n},"ns_per_iter":{median_ns_scalar:.1},"mib_per_s":0,"melem_per_s":{melem_per_s_scalar:.2},"n_variable_nodes":{n},"n_iters":{DECODE_ITERS}}}"#
    );

    let record_i8_simd = format!(
        r#"  {{"lang":"rust","impl":"loms_i8_runtime_simd","shard_len":0,"data_shards":0,"parity_shards":0,"payload_bytes":{n},"ns_per_iter":{median_ns_i8_simd:.1},"mib_per_s":0,"melem_per_s":{melem_per_s_i8_simd:.2},"n_variable_nodes":{n},"n_iters":{DECODE_ITERS}}}"#
    );
    let record_i8_scalar = format!(
        r#"  {{"lang":"rust","impl":"loms_i8_scalar","shard_len":0,"data_shards":0,"parity_shards":0,"payload_bytes":{n},"ns_per_iter":{median_ns_i8_scalar:.1},"mib_per_s":0,"melem_per_s":{melem_per_s_i8_scalar:.2},"n_variable_nodes":{n},"n_iters":{DECODE_ITERS}}}"#
    );

    let json =
        format!("[\n{record_simd},\n{record_scalar},\n{record_i8_simd},\n{record_i8_scalar}\n]\n");
    let json_path = format!("{out_dir}/ldpc_rust.json");
    std::fs::write(&json_path, &json).expect("cannot write ldpc_rust.json");
    println!("Wrote {json_path}");

    // ── Correctness checksum ─────────────────────────────────────────────
    // Run one fresh decode with the *scalar* kernel: the C++ reference is
    // scalar, so this compares like with like. The SIMD kernels already have
    // their own scalar-equivalence tests inside the crate.
    init_llr(&mut llr);
    edge_r.fill(0.0);
    decoder
        .decode_layered_offset_min_sum_scalar(
            &mut llr,
            &mut edge_r,
            &mut scratch,
            &mut hard,
            DECODE_ITERS,
        )
        .expect("checksum decode failed");
    let checksum = hard_decision_checksum(&hard);
    let ones = hard.iter().filter(|&&b| b == 1).count();
    let checksum_path = format!("{out_dir}/ldpc_rust.checksum");
    std::fs::write(&checksum_path, format!("{checksum:016x}\n"))
        .expect("cannot write ldpc_rust.checksum");
    println!("Hard-decision checksum: {checksum:016x} ({ones} ones of {n})");
    println!("Wrote {checksum_path}");
}
