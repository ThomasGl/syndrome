//! BER waterfall simulation — the "impulse response" of the LDPC code.
//!
//! Runs a Monte Carlo AWGN BPSK simulation for BG1 Z=384 across a sweep of
//! Eb/N0 values.  The waterfall region (where BER drops sharply by orders of
//! magnitude within 1–2 dB) demonstrates the coding gain of LDPC over the
//! Shannon limit.
//!
//! All scratch buffers are allocated once; no allocation occurs inside the
//! per-frame decode loop.
//!
//! # Run
//!
//! ```bash
//! cargo run --release --bin ber_sim
//! ```
//!
//! Writes `bench/results/ber_rust.json` and prints a formatted table.

use std::fs;
use std::io::Write as IoWrite;

use syndrome::qc_ldpc::{BaseGraph, QcLdpcDecoder, QcLdpcEncoder};

// ---------------------------------------------------------------------------
// Minimal XorShift64 RNG + Box-Muller Gaussian (no external crates)
// ---------------------------------------------------------------------------

struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0xDEAD_BEEF_CAFE_BABE
            } else {
                seed
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform float in (0, 1) — excludes 0 to avoid log(0) in Box-Muller.
    fn next_f32(&mut self) -> f32 {
        // Upper 24 bits for f32 mantissa precision, shift to (0,1).
        let bits = (self.next_u64() >> 40) as u32;
        (bits as f32 + 0.5) * (1.0 / (1u64 << 24) as f32)
    }

    /// Two independent standard-normal samples via Box-Muller.
    fn next_gaussian_pair(&mut self) -> (f32, f32) {
        let u1 = self.next_f32();
        let u2 = self.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        (r * theta.cos(), r * theta.sin())
    }

    fn next_bit(&mut self) -> u8 {
        (self.next_u64() >> 63) as u8
    }
}

// ---------------------------------------------------------------------------
// Simulation parameters
// ---------------------------------------------------------------------------

// BG1 Z=384: K = 22×384 = 8448 info bits, N = 68×384 = 26112 code bits.
const BG: BaseGraph = BaseGraph::Bg1;
const Z: usize = 384;

/// Stop at this many frame errors per Eb/N0 point (Monte Carlo rule of thumb).
const MAX_FRAME_ERRORS: usize = 50;
/// Or at this many total frames (to bound runtime at strong SNR).
const MAX_FRAMES: usize = 500;

const LOMS_ITERATIONS: usize = 10;
const OFFSET_BETA: f32 = 0.25;

const EB_N0_DB_POINTS: &[f64] = &[-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];

fn main() {
    let enc = QcLdpcEncoder::new(BG, Z).expect("invalid lifting size");
    let dec = QcLdpcDecoder::with_lifting_size(BG, Z, OFFSET_BETA).expect("invalid lifting size");

    let k = enc.info_bit_count();
    let n = enc.codeword_bit_count();
    let code_rate = k as f64 / n as f64;

    // Pre-allocate all scratch buffers — never reallocated inside the loop.
    let mut info_bits = vec![0u8; k];
    let mut codeword = vec![0u8; n];
    let mut channel = vec![0.0f32; n];
    let mut edge_r = vec![0.0f32; dec.required_edge_buffer()];
    let mut scratch = vec![0.0f32; dec.required_layer_buffer()];
    let mut hard = vec![0u8; n];

    let mut rng = Xorshift64::new(0x5EED_1234_ABCD_EF00);

    println!(
        "BER waterfall: BG1 Z={Z}, K={k}, N={n}, rate={:.4}, {} iterations",
        code_rate, LOMS_ITERATIONS
    );
    println!("{:-<64}", "");
    println!(
        "{:>10} | {:>14} | {:>10} | {:>7}",
        "Eb/N0 (dB)", "BER", "BLER", "Frames"
    );
    println!("{:-<64}", "");

    let mut results = Vec::new();

    for &eb_n0_db in EB_N0_DB_POINTS {
        let eb_n0_linear = 10_f64.powf(eb_n0_db / 10.0);
        let sigma = (1.0 / (2.0 * eb_n0_linear * code_rate)).sqrt() as f32;
        let sigma_sq_inv = 1.0 / (sigma * sigma);

        let mut total_bit_errors: u64 = 0;
        let mut total_frame_errors: u64 = 0;
        let mut total_frames: u64 = 0;

        while total_frame_errors < MAX_FRAME_ERRORS as u64 && total_frames < MAX_FRAMES as u64 {
            // 1. Generate random info bits.
            for bit in info_bits.iter_mut() {
                *bit = rng.next_bit();
            }

            // 2. Encode.
            enc.encode(&info_bits, &mut codeword)
                .expect("encode failed");

            // 3. BPSK modulate + AWGN: c=0 → +1, c=1 → -1, then add noise.
            // Generate Gaussian noise in pairs (Box-Muller); consume both samples.
            let mut ni = 0usize;
            while ni + 2 <= n {
                let (g0, g1) = rng.next_gaussian_pair();
                let tx0 = if codeword[ni] == 0 { 1.0f32 } else { -1.0f32 };
                let tx1 = if codeword[ni + 1] == 0 {
                    1.0f32
                } else {
                    -1.0f32
                };
                // LLR = 2y/sigma^2; positive = bit 0 more likely.
                channel[ni] = 2.0 * (tx0 + sigma * g0) * sigma_sq_inv;
                channel[ni + 1] = 2.0 * (tx1 + sigma * g1) * sigma_sq_inv;
                ni += 2;
            }
            if ni < n {
                let (g0, _) = rng.next_gaussian_pair();
                let tx0 = if codeword[ni] == 0 { 1.0f32 } else { -1.0f32 };
                channel[ni] = 2.0 * (tx0 + sigma * g0) * sigma_sq_inv;
            }

            // 4. Decode: decoder zeroes edge_r internally before each call.
            let _ = dec.decode_layered_offset_min_sum(
                &mut channel,
                &mut edge_r,
                &mut scratch,
                &mut hard,
                LOMS_ITERATIONS,
            );

            // 5. Count errors on info bits only (positions 0..k in the codeword).
            let mut frame_errors = 0u64;
            for (&orig, &dec_bit) in info_bits.iter().zip(hard.iter()) {
                if orig != dec_bit {
                    total_bit_errors += 1;
                    frame_errors += 1;
                }
            }
            if frame_errors > 0 {
                total_frame_errors += 1;
            }
            total_frames += 1;

            // Restore channel buffer for next frame (decode_layered overwrote it).
            // The next frame re-fills channel[] before decode, so no reset needed here.
        }

        let ber = total_bit_errors as f64 / (total_frames as f64 * k as f64);
        let bler = total_frame_errors as f64 / total_frames as f64;

        println!(
            "{:>10.2} | {:>14.4e} | {:>10.4e} | {:>7}",
            eb_n0_db, ber, bler, total_frames
        );

        results.push(serde_json::json!({
            "eb_n0_db":     eb_n0_db,
            "eb_n0_linear": (eb_n0_linear * 1000.0).round() / 1000.0,
            "ber":          ber,
            "bler":         bler,
            "bit_errors":   total_bit_errors,
            "frame_errors": total_frame_errors,
            "total_frames": total_frames,
        }));
    }

    println!("{:-<64}", "");

    // Write JSON output.
    let out_dir = "bench/results";
    fs::create_dir_all(out_dir).ok();
    let out_path = format!("{out_dir}/ber_rust.json");
    let json = serde_json::to_string_pretty(&results).expect("JSON serialization failed");
    let mut f = fs::File::create(&out_path).expect("cannot create ber_rust.json");
    f.write_all(json.as_bytes()).expect("write failed");
    println!("\nWrote {out_path}");
}
