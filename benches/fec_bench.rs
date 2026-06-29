//! Reproducible Criterion benchmarks for glezer_rsv.
//!
//! Run:    cargo bench --bench fec_bench
//! Output: target/criterion/** (HTML reports + raw estimates.json per bench)
//!
//! These measure ONLY what is actually implemented and correct today:
//!   - Reed-Solomon encode variants (the meaningful cross-language story)
//!   - SPSC ring push/pop round-trip
//!   - QC-LDPC LOMS kernel on the CURRENT base graph (a 2x3 toy graph until
//!     `data/bg_tables.json` is generated — see BENCHMARK_PLAN.md). Throughput
//!     numbers here are NOT representative of real 5G BG1/BG2 yet; the bench
//!     exists to prove the harness and exercise the kernel.
//!
//! Throughput is reported in bytes/elements so results convert directly to
//! MB/s and line up with the C++/Python comparison drivers.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use glezer_rsv::{BaseGraph, QcLdpcDecoder, QcLdpcEncoder, ReedSolomon, SpscRing};

/// Reed-Solomon encode, swept over shard length. Data payload = 10 shards.
fn bench_reed_solomon(c: &mut Criterion) {
    let mut group = c.benchmark_group("reed_solomon_encode");
    let data_shards = 10usize;
    let parity_shards = 4usize;

    for &shard_len in &[256usize, 1024, 4096, 16384] {
        let payload_bytes = (data_shards * shard_len) as u64;
        group.throughput(Throughput::Bytes(payload_bytes));

        let owned: Vec<Vec<u8>> = (0..data_shards).map(|i| vec![i as u8; shard_len]).collect();
        let refs: Vec<&[u8]> = owned.iter().map(|v| v.as_slice()).collect();
        let mut parity = vec![0u8; parity_shards * shard_len];

        let mut rs = ReedSolomon::new(data_shards, parity_shards);
        rs.precompute_mul_tables();

        // The two variants worth comparing: the straightforward encoder and the
        // cache-chunked table encoder (fastest in ad-hoc timing).
        group.bench_with_input(
            BenchmarkId::new("encode_into", shard_len),
            &shard_len,
            |b, _| b.iter(|| rs.encode_into(black_box(&refs), black_box(&mut parity))),
        );
        group.bench_with_input(
            BenchmarkId::new("encode_with_tables_chunked", shard_len),
            &shard_len,
            |b, _| {
                b.iter(|| rs.encode_with_tables_chunked(black_box(&refs), black_box(&mut parity)))
            },
        );
    }
    group.finish();
}

/// SPSC ring: cost of a single-threaded push+pop round-trip.
fn bench_spsc(c: &mut Criterion) {
    let mut group = c.benchmark_group("spsc_ring");
    group.throughput(Throughput::Elements(1));
    let ring = SpscRing::<u32, 1024>::new();
    group.bench_function("push_pop_roundtrip", |b| {
        b.iter(|| {
            ring.try_push(black_box(7u32)).ok();
            black_box(ring.try_pop());
        })
    });
    group.finish();
}

/// QC-LDPC LOMS kernel on the CURRENT (toy) base graph. Labeled `toy` so the
/// reported numbers are never mistaken for real 5G BG1/BG2 throughput.
fn bench_qc_ldpc_kernel(c: &mut Criterion) {
    let mut group = c.benchmark_group("qc_ldpc_loms_toy");
    let decoder = QcLdpcDecoder::new(BaseGraph::Bg1, 0.25);
    let n = decoder.variable_node_count();
    group.throughput(Throughput::Elements(n as u64));

    let mut llr = vec![0.5f32; n];
    let mut edge_r = vec![0.0f32; decoder.required_edge_buffer()];
    let mut scratch = vec![0.0f32; decoder.required_layer_buffer()];
    let mut hard = vec![0u8; n];

    group.bench_function("decode_5iter", |b| {
        b.iter(|| {
            // reset mutable channel input each iteration so work is identical
            for v in llr.iter_mut() {
                *v = 0.5;
            }
            for v in edge_r.iter_mut() {
                *v = 0.0;
            }
            decoder
                .decode_layered_offset_min_sum(
                    black_box(&mut llr),
                    black_box(&mut edge_r),
                    black_box(&mut scratch),
                    black_box(&mut hard),
                    5,
                )
                .unwrap();
        })
    });
    group.finish();
}

/// QC-LDPC LOMS decode + encode on real 5G NR BG1, Z=384.
///
/// This is the real 5G NR code (BG1, iLS=1, Z=384, rate ~1/3).
/// Throughput is reported per decoded codeword (N = 68*384 = 26112 variable nodes).
fn bench_qc_ldpc_real(c: &mut Criterion) {
    let mut group = c.benchmark_group("qc_ldpc_loms_bg1_z384");

    let encoder = QcLdpcEncoder::new(BaseGraph::Bg1, 384).unwrap();
    let decoder = QcLdpcDecoder::with_lifting_size(BaseGraph::Bg1, 384, 0.25).unwrap();

    let k = encoder.info_bit_count();
    let n = decoder.variable_node_count();

    group.throughput(Throughput::Elements(n as u64));

    // Prepare a fixed info pattern and encode once to get the reference codeword.
    let info: Vec<u8> = (0..k).map(|i| (i % 2) as u8).collect();
    let mut codeword = vec![0u8; n];
    encoder.encode(&info, &mut codeword).unwrap();

    // Pre-build LLR vector from codeword (high SNR, ±5.0).
    let reference_llr: Vec<f32> = codeword
        .iter()
        .map(|&b| if b == 0 { 5.0f32 } else { -5.0 })
        .collect();

    let mut llr = reference_llr.clone();
    let mut edge_r = vec![0.0f32; decoder.required_edge_buffer()];
    let mut scratch = vec![0.0f32; decoder.required_layer_buffer()];
    let mut hard = vec![0u8; n];

    group.bench_function("decode_10iter", |b| {
        b.iter(|| {
            // Reset LLRs and extrinsic buffer each iteration.
            llr.copy_from_slice(&reference_llr);
            for v in edge_r.iter_mut() {
                *v = 0.0;
            }
            decoder
                .decode_layered_offset_min_sum(
                    black_box(&mut llr),
                    black_box(&mut edge_r),
                    black_box(&mut scratch),
                    black_box(&mut hard),
                    10,
                )
                .unwrap();
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_reed_solomon,
    bench_spsc,
    bench_qc_ldpc_kernel,
    bench_qc_ldpc_real
);
criterion_main!(benches);
