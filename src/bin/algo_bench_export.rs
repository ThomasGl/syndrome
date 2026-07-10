//! All-algorithm benchmark exporter.
//!
//! Times encode and decode for every FEC core in the crate under a uniform
//! metric — **information-bit throughput** (Mbit/s of payload bits, not coded
//! bits) — and writes one JSON record per (algorithm, operation) to
//! `bench/results/algos.json` for `bench/dashboard/gen_charts.py`.
//!
//! Usage: `cargo run --release --bin algo_bench_export`
//!
//! All numbers are measured on this machine at run time — never hand-written.

use std::time::Instant;
use syndrome::channel_sim::AwgnChannel;
use syndrome::crc::{Crc24, CrcKind};
use syndrome::hamming::encode_hamming_7_4;
use syndrome::polar::{PolarDecoder, PolarEncoder};
use syndrome::viterbi::ViterbiDecoder;
use syndrome::{
    BaseGraph, BchCode, GolayCode, QcLdpcDecoder, QcLdpcEncoder, ReedSolomon, TurboDecoder,
    TurboEncoder,
};

/// Timing harness: warm-up then averaged wall-clock ns per call.
fn time_ns<F: FnMut()>(mut f: F, warmup: u32, iters: u32) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        f();
    }
    t0.elapsed().as_nanos() as f64 / iters as f64
}

struct Record {
    algo: &'static str,
    label: String,
    op: &'static str,
    info_bits: usize,
    ns_per_iter: f64,
}

impl Record {
    fn mbit_per_s(&self) -> f64 {
        // info_bits per call / (ns per call) → bits/ns → *1e9 → bits/s → /1e6.
        self.info_bits as f64 / self.ns_per_iter * 1e3
    }

    fn to_json(&self) -> String {
        format!(
            r#"  {{"algo":"{}","label":"{}","op":"{}","info_bits":{},"ns_per_iter":{:.1},"mbit_per_s":{:.3}}}"#,
            self.algo,
            self.label,
            self.op,
            self.info_bits,
            self.ns_per_iter,
            self.mbit_per_s()
        )
    }
}

/// Deterministic pseudo-random bit buffer.
fn random_bits(len: usize, mut seed: u64) -> Vec<u8> {
    (0..len)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 63) as u8
        })
        .collect()
}

fn main() {
    let mut records: Vec<Record> = Vec::new();

    // ── Hamming(7,4) ─────────────────────────────────────────────────────
    // Encode/decode a 4 KiB payload nibble-by-nibble.
    {
        let payload: Vec<u8> = (0..4096u32).map(|i| (i % 16) as u8).collect();
        let info_bits = payload.len() * 4;
        let mut coded = vec![0u8; payload.len()];
        let ns_enc = time_ns(
            || {
                for (c, &nib) in coded.iter_mut().zip(&payload) {
                    *c = encode_hamming_7_4(nib);
                }
            },
            20,
            200,
        );
        records.push(Record {
            algo: "hamming",
            label: "Hamming(7,4)".into(),
            op: "encode",
            info_bits,
            ns_per_iter: ns_enc,
        });
        let coded_fixed = coded.clone();
        let mut out = vec![0u8; payload.len()];
        let ns_dec = time_ns(
            || {
                for (o, &c) in out.iter_mut().zip(&coded_fixed) {
                    *o = syndrome::decode_hamming_7_4(c).unwrap();
                }
            },
            20,
            200,
        );
        records.push(Record {
            algo: "hamming",
            label: "Hamming(7,4)".into(),
            op: "decode",
            info_bits,
            ns_per_iter: ns_dec,
        });
    }

    // ── CRC-24A over a 6144-bit block ────────────────────────────────────
    {
        let crc = Crc24::new(CrcKind::Crc24A);
        let bits = random_bits(6144, 7);
        let ns = time_ns(
            || {
                std::hint::black_box(crc.compute(&bits));
            },
            20,
            500,
        );
        records.push(Record {
            algo: "crc24a",
            label: "CRC-24A (6144-bit block)".into(),
            op: "encode",
            info_bits: 6144,
            ns_per_iter: ns,
        });
    }

    // ── Reed-Solomon RS(10,4), 4 KiB shards ──────────────────────────────
    {
        let (d, p, shard_len) = (10usize, 4usize, 4096usize);
        let mut rs = ReedSolomon::new(d, p);
        rs.precompute_mul_tables();
        let data: Vec<Vec<u8>> = (0..d).map(|j| vec![j as u8 + 1; shard_len]).collect();
        let refs: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
        let mut parity = vec![0u8; p * shard_len];
        let info_bits = d * shard_len * 8;

        let ns_enc = time_ns(|| rs.encode_with_avx2(&refs, &mut parity).unwrap(), 20, 300);
        records.push(Record {
            algo: "reed_solomon",
            label: "Reed-Solomon RS(10,4) GF(256)".into(),
            op: "encode",
            info_bits,
            ns_per_iter: ns_enc,
        });

        // Erasure decode: 4 lost data shards. The per-iteration Vec rebuild is
        // timed separately and subtracted so only recovery math is counted.
        let full: Vec<Vec<u8>> = data
            .iter()
            .cloned()
            .chain(parity.chunks(shard_len).map(|c| c.to_vec()))
            .collect();
        let make_shards = || {
            let mut s: Vec<Option<Vec<u8>>> = full.iter().cloned().map(Some).collect();
            s[0] = None;
            s[3] = None;
            s[5] = None;
            s[9] = None;
            s
        };
        let ns_setup = time_ns(
            || {
                std::hint::black_box(make_shards());
            },
            10,
            100,
        );
        let ns_total = time_ns(
            || {
                let mut shards = make_shards();
                rs.decode(&mut shards).unwrap();
                std::hint::black_box(&shards);
            },
            10,
            100,
        );
        records.push(Record {
            algo: "reed_solomon",
            label: "Reed-Solomon RS(10,4) GF(256)".into(),
            op: "decode",
            info_bits,
            ns_per_iter: (ns_total - ns_setup).max(1.0),
        });
    }

    // ── Viterbi K=7 rate-1/2, 2048 info bits ─────────────────────────────
    {
        let dec = ViterbiDecoder::new(7).unwrap();
        let info = random_bits(2048, 11);
        let coded = dec.encode(&info);
        let llr: Vec<f32> = coded
            .iter()
            .map(|&b| if b == 0 { 3.0 } else { -3.0 })
            .collect();
        let info_bits = info.len();

        let ns_enc = time_ns(
            || {
                std::hint::black_box(dec.encode(&info));
            },
            10,
            100,
        );
        records.push(Record {
            algo: "viterbi",
            label: "Viterbi K=7 R=1/2".into(),
            op: "encode",
            info_bits,
            ns_per_iter: ns_enc,
        });
        let ns_dec = time_ns(
            || {
                std::hint::black_box(dec.decode_soft(&llr));
            },
            5,
            50,
        );
        records.push(Record {
            algo: "viterbi",
            label: "Viterbi K=7 R=1/2".into(),
            op: "decode",
            info_bits,
            ns_per_iter: ns_dec,
        });
    }

    // ── Polar (1024, 512), SC decode ─────────────────────────────────────
    {
        let (n, k) = (1024usize, 512usize);
        let enc = PolarEncoder::new(n, k).unwrap();
        let dec = PolarDecoder::new(n, k, 1, None).unwrap();
        let info = random_bits(k, 13);
        let mut cw = vec![0u8; n];
        enc.encode(&info, &mut cw).unwrap();
        let llr: Vec<f32> = cw
            .iter()
            .map(|&b| if b == 0 { 4.0 } else { -4.0 })
            .collect();
        let mut out = vec![0u8; k];

        let ns_enc = time_ns(|| enc.encode(&info, &mut cw).unwrap(), 10, 200);
        records.push(Record {
            algo: "polar",
            label: "Polar (1024,512) SC".into(),
            op: "encode",
            info_bits: k,
            ns_per_iter: ns_enc,
        });
        let ns_dec = time_ns(|| dec.decode_sc(&llr, &mut out).unwrap(), 5, 100);
        records.push(Record {
            algo: "polar",
            label: "Polar (1024,512) SC".into(),
            op: "decode",
            info_bits: k,
            ns_per_iter: ns_dec,
        });
    }

    // ── Turbo LTE rate-1/3, K=1024, max-log-MAP 8 iterations ─────────────
    {
        let k = 1024usize;
        let enc = TurboEncoder::new(k).unwrap();
        let mut dec = TurboDecoder::new(k).unwrap();
        let info = random_bits(k, 19);
        let mut coded = vec![0u8; enc.output_len()];
        enc.encode(&info, &mut coded).unwrap();

        let ns_enc = time_ns(|| enc.encode(&info, &mut coded).unwrap(), 10, 200);
        records.push(Record {
            algo: "turbo",
            label: "Turbo LTE K=1024 R=1/3".into(),
            op: "encode",
            info_bits: k,
            ns_per_iter: ns_enc,
        });

        // Mild AWGN so the iterative decoder does representative work.
        let mut ch = AwgnChannel::new(1.5, 1.0 / 3.0, 21);
        let llr = ch.transmit(&coded);
        let mut out = vec![0u8; k];
        let ns_dec = time_ns(
            || {
                dec.decode(&llr, &mut out, 8).unwrap();
            },
            3,
            30,
        );
        records.push(Record {
            algo: "turbo",
            label: "Turbo LTE K=1024 R=1/3".into(),
            op: "decode",
            info_bits: k,
            ns_per_iter: ns_dec,
        });
    }

    // ── BCH(255,223,t=4) over GF(2^8) ─────────────────────────────────────
    {
        let bch = BchCode::new(4).unwrap();
        let (n, k) = (bch.n(), bch.k());
        let info = random_bits(k, 23);
        let mut cw = vec![0u8; n];
        bch.encode(&info, &mut cw).unwrap();

        let ns_enc = time_ns(|| bch.encode(&info, &mut cw).unwrap(), 10, 300);
        records.push(Record {
            algo: "bch",
            label: format!("BCH(255,{k},t=4)"),
            op: "encode",
            info_bits: k,
            ns_per_iter: ns_enc,
        });

        // Corrupt t=4 positions; per-iteration memcpy of the 255-byte block is
        // negligible next to the syndrome/BM/Chien work being measured.
        let mut noisy = cw.clone();
        for pos in [7usize, 63, 128, 200] {
            noisy[pos] ^= 1;
        }
        let mut scratch = vec![0u8; n];
        let ns_dec = time_ns(
            || {
                scratch.copy_from_slice(&noisy);
                bch.decode(&mut scratch).unwrap();
            },
            10,
            200,
        );
        records.push(Record {
            algo: "bch",
            label: format!("BCH(255,{k},t=4)"),
            op: "decode",
            info_bits: k,
            ns_per_iter: ns_dec,
        });
    }

    // ── Extended Golay(24,12) — table-driven ─────────────────────────────
    // Throughput over a 1024-block buffer (12,288 info bits per timed call).
    {
        let golay = GolayCode::new();
        let blocks = 1024usize;
        let info = random_bits(12 * blocks, 29);
        let mut coded = vec![0u8; 24 * blocks];
        let info_bits = 12 * blocks;

        let ns_enc = time_ns(
            || {
                for b in 0..blocks {
                    golay
                        .encode(
                            &info[b * 12..(b + 1) * 12],
                            &mut coded[b * 24..(b + 1) * 24],
                        )
                        .unwrap();
                }
            },
            10,
            200,
        );
        records.push(Record {
            algo: "golay",
            label: "Golay(24,12) ext.".into(),
            op: "encode",
            info_bits,
            ns_per_iter: ns_enc,
        });

        // Flip 2 bits in every block, then time correction.
        let mut noisy = coded.clone();
        for b in 0..blocks {
            noisy[b * 24 + 3] ^= 1;
            noisy[b * 24 + 17] ^= 1;
        }
        let mut out = vec![0u8; 12 * blocks];
        let ns_dec = time_ns(
            || {
                for b in 0..blocks {
                    golay
                        .decode(&noisy[b * 24..(b + 1) * 24], &mut out[b * 12..(b + 1) * 12])
                        .unwrap();
                }
            },
            10,
            200,
        );
        records.push(Record {
            algo: "golay",
            label: "Golay(24,12) ext.".into(),
            op: "decode",
            info_bits,
            ns_per_iter: ns_dec,
        });
    }

    // ── QC-LDPC BG1 Z=384, LOMS 10 iterations ────────────────────────────
    {
        let z = 384usize;
        let enc = QcLdpcEncoder::new(BaseGraph::Bg1, z).unwrap();
        let dec = QcLdpcDecoder::with_lifting_size(BaseGraph::Bg1, z, 0.25).unwrap();
        let k = enc.info_bit_count();
        let n = dec.variable_node_count();
        let info = random_bits(k, 17);
        let mut cw = vec![0u8; n];
        enc.encode(&info, &mut cw).unwrap();

        let ns_enc = time_ns(|| enc.encode(&info, &mut cw).unwrap(), 5, 50);
        records.push(Record {
            algo: "qc_ldpc",
            label: "QC-LDPC BG1 Z=384 LOMS".into(),
            op: "encode",
            info_bits: k,
            ns_per_iter: ns_enc,
        });

        // Mildly noisy LLRs from the AWGN channel so the decoder does real
        // iterative work rather than an instant syndrome pass.
        let mut ch = AwgnChannel::new(2.0, 22.0 / 68.0, 42);
        let llr0 = ch.transmit(&cw);
        let mut llr = llr0.clone();
        let mut edge = vec![0.0f32; dec.required_edge_buffer()];
        let mut scratch = vec![0.0f32; dec.required_layer_buffer()];
        let mut hard = vec![0u8; n];
        let ns_dec = time_ns(
            || {
                llr.copy_from_slice(&llr0);
                edge.iter_mut().for_each(|v| *v = 0.0);
                dec.decode_layered_offset_min_sum(&mut llr, &mut edge, &mut scratch, &mut hard, 10)
                    .unwrap();
            },
            3,
            30,
        );
        records.push(Record {
            algo: "qc_ldpc",
            label: "QC-LDPC BG1 Z=384 LOMS".into(),
            op: "decode",
            info_bits: k,
            ns_per_iter: ns_dec,
        });
    }

    // ── write JSON + print summary ───────────────────────────────────────
    std::fs::create_dir_all("bench/results").expect("cannot create bench/results");
    let json = format!(
        "[\n{}\n]\n",
        records
            .iter()
            .map(Record::to_json)
            .collect::<Vec<_>>()
            .join(",\n")
    );
    std::fs::write("bench/results/algos.json", &json).expect("cannot write algos.json");
    println!("Wrote bench/results/algos.json\n");
    println!(
        "{:<36} {:>8} {:>14} {:>14}",
        "algorithm", "op", "ns/call", "Mbit/s (info)"
    );
    for r in &records {
        println!(
            "{:<36} {:>8} {:>14.0} {:>14.2}",
            r.label,
            r.op,
            r.ns_per_iter,
            r.mbit_per_s()
        );
    }
}
