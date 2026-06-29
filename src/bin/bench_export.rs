//! Benchmark exporter: times Reed-Solomon encode variants over the shard-length
//! sweep used by the cross-language comparison and writes structured JSON + a
//! checksum file for the correctness gate.
//!
//! Usage: cargo run --release --bin bench_export
//! Output: bench/results/rust.json, bench/results/rust.checksum

use glezer_rsv::ReedSolomon;
use std::time::Instant;

const DATA_SHARDS: usize = 10;
const PARITY_SHARDS: usize = 4;
const SHARD_LENS: &[usize] = &[256, 1024, 4096, 16384];
/// Number of timing iterations; enough to get stable mean on a warm CPU.
const ITERS: u32 = 500;
/// Warm-up iterations discarded before timing.
const WARMUP: u32 = 50;

/// Build deterministic seed-filled data shards for the checksum.
/// Shard `j` has every byte set to `(j as u8).wrapping_mul(3).wrapping_add(1)`.
fn seed_data(shard_len: usize) -> Vec<Vec<u8>> {
    (0..DATA_SHARDS)
        .map(|j| vec![(j as u8).wrapping_mul(3).wrapping_add(1); shard_len])
        .collect()
}

fn time_encode<F>(mut f: F) -> f64
where
    F: FnMut(),
{
    // warm-up
    for _ in 0..WARMUP {
        f();
    }
    let t0 = Instant::now();
    for _ in 0..ITERS {
        f();
    }
    t0.elapsed().as_nanos() as f64 / ITERS as f64
}

fn mib_per_s(payload_bytes: usize, ns_per_iter: f64) -> f64 {
    let bytes_per_ns = payload_bytes as f64 / ns_per_iter;
    bytes_per_ns * 1e9 / (1024.0 * 1024.0)
}

fn main() {
    let out_dir = "bench/results";
    std::fs::create_dir_all(out_dir).expect("cannot create bench/results");

    let mut records: Vec<String> = Vec::new();
    let mut checksum_parity: Option<Vec<u8>> = None;

    // Use shard_len=1024 for the checksum (compact but representative).
    let checksum_shard_len = 1024usize;

    for &shard_len in SHARD_LENS {
        let payload_bytes = DATA_SHARDS * shard_len;

        // Build owned data once; take refs for encode calls.
        let owned: Vec<Vec<u8>> = (0..DATA_SHARDS).map(|i| vec![i as u8; shard_len]).collect();
        let refs: Vec<&[u8]> = owned.iter().map(|v| v.as_slice()).collect();
        let mut parity = vec![0u8; PARITY_SHARDS * shard_len];

        let mut rs = ReedSolomon::new(DATA_SHARDS, PARITY_SHARDS);
        rs.precompute_mul_tables();

        // --- encode_into ---
        let ns_into = time_encode(|| {
            rs.encode_into(&refs, &mut parity);
        });
        let mib_into = mib_per_s(payload_bytes, ns_into);
        records.push(format!(
            r#"  {{"lang":"rust","impl":"encode_into","shard_len":{shard_len},"data_shards":{DATA_SHARDS},"parity_shards":{PARITY_SHARDS},"payload_bytes":{payload_bytes},"ns_per_iter":{ns_into:.1},"mib_per_s":{mib_into:.1}}}"#
        ));

        // --- encode_with_tables_chunked ---
        let ns_chunked = time_encode(|| {
            rs.encode_with_tables_chunked(&refs, &mut parity);
        });
        let mib_chunked = mib_per_s(payload_bytes, ns_chunked);
        records.push(format!(
            r#"  {{"lang":"rust","impl":"encode_with_tables_chunked","shard_len":{shard_len},"data_shards":{DATA_SHARDS},"parity_shards":{PARITY_SHARDS},"payload_bytes":{payload_bytes},"ns_per_iter":{ns_chunked:.1},"mib_per_s":{mib_chunked:.1}}}"#
        ));

        // --- encode_with_avx2 (runtime-detected; falls back to chunked) ---
        let ns_avx2 = time_encode(|| {
            rs.encode_with_avx2(&refs, &mut parity);
        });
        let mib_avx2 = mib_per_s(payload_bytes, ns_avx2);
        records.push(format!(
            r#"  {{"lang":"rust","impl":"encode_with_avx2","shard_len":{shard_len},"data_shards":{DATA_SHARDS},"parity_shards":{PARITY_SHARDS},"payload_bytes":{payload_bytes},"ns_per_iter":{ns_avx2:.1},"mib_per_s":{mib_avx2:.1}}}"#
        ));

        // Capture checksum from encode_into at the canonical shard length.
        if shard_len == checksum_shard_len {
            let seed = seed_data(shard_len);
            let seed_refs: Vec<&[u8]> = seed.iter().map(|v| v.as_slice()).collect();
            let mut cparity = vec![0u8; PARITY_SHARDS * shard_len];
            rs.encode_into(&seed_refs, &mut cparity);
            checksum_parity = Some(cparity);
        }
    }

    // Write JSON array.
    let json = format!("[\n{}\n]\n", records.join(",\n"));
    let json_path = format!("{out_dir}/rust.json");
    std::fs::write(&json_path, &json).expect("cannot write rust.json");
    println!("Wrote {json_path}");

    // Write checksum: hex of all parity bytes for shard_len=1024.
    let cs_bytes = checksum_parity.expect("checksum not captured");
    let hex: String = cs_bytes.iter().map(|b| format!("{b:02x}")).collect();
    let cs_path = format!("{out_dir}/rust.checksum");
    std::fs::write(&cs_path, &hex).expect("cannot write rust.checksum");
    println!("Wrote {cs_path}");

    // Print a quick summary table.
    println!("\nRust RS encode throughput (MiB/s):");
    println!(
        "{:<32} {:>8} {:>8} {:>8} {:>8}",
        "impl", "256B", "1KiB", "4KiB", "16KiB"
    );

    // Re-run to get per-impl rows for the summary (reuse records).
    // Just parse them back (simpler than storing separately).
    for impl_name in &["encode_into", "encode_with_tables_chunked"] {
        let vals: Vec<&str> = records
            .iter()
            .filter(|r| r.contains(&format!(r#""impl":"{impl_name}""#)))
            .map(|r| {
                let start = r.rfind("\"mib_per_s\":").unwrap() + 12;
                let end = r[start..].find('}').unwrap() + start;
                &r[start..end]
            })
            .collect();
        if vals.len() == SHARD_LENS.len() {
            println!(
                "{:<32} {:>8} {:>8} {:>8} {:>8}",
                impl_name,
                vals[0].trim(),
                vals[1].trim(),
                vals[2].trim(),
                vals[3].trim()
            );
        }
    }
}
