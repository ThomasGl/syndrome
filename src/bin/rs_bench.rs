use std::time::Instant;
use syndrome::ReedSolomon;

fn bench<F: Fn(&ReedSolomon, &[&[u8]], &mut [u8])>(
    name: &str,
    rs: &ReedSolomon,
    data: &[&[u8]],
    parity: &mut [u8],
    f: F,
) {
    let iterations = 1000;
    let start = Instant::now();
    for _ in 0..iterations {
        f(rs, data, parity);
    }
    let duration = start.elapsed();
    println!(
        "{}: {:?} total, {:?} per iter",
        name,
        duration,
        duration / iterations
    );
}

fn main() {
    let rs = ReedSolomon::new(10, 4);
    let shard_len = 1024usize;
    let data: Vec<Vec<u8>> = (0..10).map(|i| vec![i as u8; shard_len]).collect();
    let refs: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
    let mut parity = vec![0u8; 4 * shard_len];

    bench(
        "encode_into",
        &rs,
        &refs,
        &mut parity,
        |rs, data, parity| {
            rs.encode_into(data, parity).unwrap();
        },
    );
    bench(
        "encode_optimized",
        &rs,
        &refs,
        &mut parity,
        |rs, data, parity| {
            rs.encode_optimized(data, parity).unwrap();
        },
    );
    bench("encode_soa", &rs, &refs, &mut parity, |rs, data, parity| {
        rs.encode_soa(data, parity).unwrap();
    });
    // precompute mul tables and benchmark table-based encoder and SIMD encoder
    let mut rs2 = ReedSolomon::new(10, 4);
    rs2.precompute_mul_tables();
    bench(
        "encode_with_tables",
        &rs2,
        &refs,
        &mut parity,
        |rs, data, parity| {
            rs.encode_with_tables(data, parity).unwrap();
        },
    );
    bench(
        "encode_simd",
        &rs2,
        &refs,
        &mut parity,
        |rs, data, parity| {
            rs.encode_simd(data, parity).unwrap();
        },
    );
    bench(
        "encode_with_tables_chunked",
        &rs2,
        &refs,
        &mut parity,
        |rs, data, parity| {
            rs.encode_with_tables_chunked(data, parity).unwrap();
        },
    );
}
