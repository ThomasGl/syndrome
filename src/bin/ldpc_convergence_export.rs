//! LDPC iterative-decode convergence exporter — runs a real 802.11 Wi-Fi
//! LDPC encode → BPSK/AWGN corrupt → layered offset min-sum decode cycle and
//! dumps a per-iteration trace to JSON for the README convergence animation.
//!
//! Every number in the output comes from one continuous, real decode call:
//! [`syndrome::qc_ldpc::QcLdpcDecoder::decode_layered_offset_min_sum_traced`]
//! is invoked once, with an observer closure that snapshots the hard-decision
//! bits after each completed layered pass. No error pattern or convergence
//! curve is authored by hand — the bit-error counts are the actual Hamming
//! distance between the decoder's running hard decision and the codeword
//! this program itself transmitted.
//!
//! A small search over `(Eb/N0, seed)` picks one trial whose initial
//! corruption and iteration count make a legible animation (see
//! [`is_good_trial`]); the search itself only selects among real trials, it
//! does not alter any of them.
//!
//! Usage: `cargo run --release --bin ldpc_convergence_export`
//! Output: `bench/results/ldpc_convergence.json`

use std::fs;
use std::io::Write as IoWrite;

use syndrome::channel_sim::AwgnChannel;
use syndrome::wifi_ldpc_tables::{wifi_ldpc_decoder, wifi_ldpc_encoder};

/// 802.11 lifting size for this demonstration code.
const Z: usize = 27;
/// Code rate numerator/denominator (rate 1/2 — the shortest, most legible
/// of the 12 real 802.11 (Z, rate) combinations this crate implements).
const RATE_NUM: usize = 1;
const RATE_DEN: usize = 2;
const OFFSET_BETA: f32 = 0.25;
const MAX_ITERATIONS: usize = 14;

/// Acceptable pre-decode bit-error range: enough visible corruption to read
/// as "damaged" in a 24x27 pixel grid, not so much the code cannot recover.
const MIN_INITIAL_ERRORS: usize = 25;
const MAX_INITIAL_ERRORS: usize = 160;
/// Convergence should take a handful of iterations, not one and not all of them,
/// so the animation actually shows something iterating.
const MIN_ITERS_USED: usize = 4;
const MAX_ITERS_USED: usize = MAX_ITERATIONS - 1;

/// One decode trial: an `Eb/N0` point and an `AwgnChannel` seed.
struct Trial {
    eb_n0_db: f32,
    seed: u64,
}

/// A completed, real decode run kept for JSON export.
struct TrialResult {
    trial: Trial,
    transmitted_codeword: Vec<u8>,
    /// `(iteration, hard_bits, bit_errors)` — one entry per completed
    /// layered pass, plus a synthetic iteration-0 entry for the pre-decode
    /// (channel hard-decision) state.
    frames: Vec<(usize, Vec<u8>, usize)>,
    iterations_used: usize,
    converged: bool,
}

/// True if this trial's real trace makes a legible convergence animation:
/// visible initial corruption, genuine perfect reconstruction, and an
/// iteration count long enough to show iterative progress.
fn is_good_trial(pre_decode_errors: usize, iterations_used: usize, final_errors: usize) -> bool {
    final_errors == 0
        && (MIN_INITIAL_ERRORS..=MAX_INITIAL_ERRORS).contains(&pre_decode_errors)
        && (MIN_ITERS_USED..=MAX_ITERS_USED).contains(&iterations_used)
}

/// Run one full real encode → AWGN → traced-decode cycle for `trial`.
fn run_trial(
    enc: &syndrome::qc_ldpc::QcLdpcEncoder,
    dec: &syndrome::qc_ldpc::QcLdpcDecoder,
    info_bits: &[u8],
    trial: Trial,
) -> TrialResult {
    let n = dec.variable_node_count();
    let mut codeword = vec![0u8; enc.codeword_bit_count()];
    enc.encode(info_bits, &mut codeword).expect("encode failed");

    let code_rate = RATE_NUM as f32 / RATE_DEN as f32;
    let mut channel = AwgnChannel::new(trial.eb_n0_db, code_rate, trial.seed);
    let mut llr = channel.transmit(&codeword);

    // Pre-decode hard decision: the receiver's view before any LOMS pass —
    // this is the "damaged" frame the animation opens on.
    let pre_hard: Vec<u8> = llr.iter().map(|&l| u8::from(l < 0.0)).collect();
    let pre_errors = pre_hard
        .iter()
        .zip(codeword.iter())
        .filter(|(a, b)| a != b)
        .count();

    let mut edge_r = vec![0.0f32; dec.required_edge_buffer()];
    let mut scratch = vec![0.0f32; dec.required_layer_buffer()];
    let mut hard = vec![0u8; n];

    let mut frames: Vec<(usize, Vec<u8>, usize)> = vec![(0, pre_hard, pre_errors)];

    let iterations_used = dec
        .decode_layered_offset_min_sum_traced(
            &mut llr,
            &mut edge_r,
            &mut scratch,
            &mut hard,
            MAX_ITERATIONS,
            |iteration, live_llr| {
                let hard_snapshot: Vec<u8> = live_llr.iter().map(|&l| u8::from(l < 0.0)).collect();
                let errors = hard_snapshot
                    .iter()
                    .zip(codeword.iter())
                    .filter(|(a, b)| a != b)
                    .count();
                frames.push((iteration, hard_snapshot, errors));
            },
        )
        .expect("decode failed");

    let converged = iterations_used < MAX_ITERATIONS;

    TrialResult {
        trial,
        transmitted_codeword: codeword,
        frames,
        iterations_used,
        converged,
    }
}

fn main() {
    let enc = wifi_ldpc_encoder(Z, RATE_NUM, RATE_DEN)
        .expect("802.11 Z=27 rate 1/2 is a supported combination");
    let dec = wifi_ldpc_decoder(Z, RATE_NUM, RATE_DEN, OFFSET_BETA)
        .expect("802.11 Z=27 rate 1/2 is a supported combination");

    let k = enc.info_bit_count();
    let n = enc.codeword_bit_count();
    assert_eq!(n, 24 * Z, "802.11 codeword length must be 24*Z");

    // Deterministic pseudo-random payload (Knuth multiplicative hash), same
    // style as tests/media_reconstruction.rs — fixed across trials so only
    // the channel noise realization changes what gets corrupted.
    let info_bits: Vec<u8> = (0..k)
        .map(|i| u8::from((i.wrapping_mul(2_654_435_761) >> 31) & 1 == 1))
        .collect();

    println!("802.11 Wi-Fi LDPC: Z={Z}, rate={RATE_NUM}/{RATE_DEN}, K={k}, N={n}");
    println!("Searching real (Eb/N0, seed) trials for a legible convergence trace...");

    let eb_n0_candidates: &[f32] = &[-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0];
    let mut best: Option<TrialResult> = None;

    'search: for &eb_n0_db in eb_n0_candidates {
        for seed in 1u64..=400 {
            let result = run_trial(&enc, &dec, &info_bits, Trial { eb_n0_db, seed });
            let pre_errors = result.frames[0].2;
            let final_errors = result.frames.last().map(|f| f.2).unwrap_or(usize::MAX);
            if is_good_trial(pre_errors, result.iterations_used, final_errors) {
                println!(
                    "  selected: Eb/N0={eb_n0_db:.1} dB, seed={seed}, pre-decode errors={pre_errors}, \
                     iterations={}, converged={}",
                    result.iterations_used, result.converged
                );
                best = Some(result);
                break 'search;
            }
        }
    }

    let result = best.expect(
        "no (Eb/N0, seed) trial in the search grid satisfied is_good_trial; widen the ranges",
    );
    assert!(
        result.converged,
        "selected trial must have converged (syndrome satisfied before MAX_ITERATIONS)"
    );
    assert_eq!(
        result.frames.last().unwrap().2,
        0,
        "selected trial must reach zero bit errors — perfect reconstruction"
    );

    let frames_json: Vec<serde_json::Value> = result
        .frames
        .iter()
        .map(|(iteration, hard_bits, bit_errors)| {
            serde_json::json!({
                "iteration": iteration,
                "hard_bits": hard_bits,
                "bit_errors": bit_errors,
            })
        })
        .collect();

    let output = serde_json::json!({
        "code": {
            "standard": "IEEE 802.11ax/be Annex R/F QC-LDPC",
            "z": Z,
            "rate_num": RATE_NUM,
            "rate_den": RATE_DEN,
            "n": n,
            "k": k,
            "offset_beta": OFFSET_BETA,
        },
        "channel": {
            "eb_n0_db": result.trial.eb_n0_db,
            "seed": result.trial.seed,
            "code_rate": RATE_NUM as f32 / RATE_DEN as f32,
        },
        "max_iterations": MAX_ITERATIONS,
        "iterations_used": result.iterations_used,
        "converged": result.converged,
        "transmitted_codeword": result.transmitted_codeword,
        "frames": frames_json,
    });

    let out_dir = "bench/results";
    fs::create_dir_all(out_dir).expect("cannot create bench/results");
    let out_path = format!("{out_dir}/ldpc_convergence.json");
    let json = serde_json::to_string_pretty(&output).expect("JSON serialization failed");
    let mut f = fs::File::create(&out_path).expect("cannot create ldpc_convergence.json");
    f.write_all(json.as_bytes()).expect("write failed");
    println!("Wrote {out_path}");
}
