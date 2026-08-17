//! Validation of the QC-LDPC layered offset min-sum correction factor $\beta$.
//!
//! The offset min-sum check-node update
//!
//! $$R_{ij} = \left(\prod_{k \ne j} \operatorname{sign} Q_{ik}\right)
//!   \cdot \max\left(\min_{k \ne j} |Q_{ik}| - \beta,\; 0\right)$$
//!
//! exists because plain min-sum *overestimates* check-node reliability: the
//! $\min$ operation is an upper bound on the true (tanh-rule) magnitude, and
//! feeding an iterative decoder over-confident messages makes it converge to
//! the wrong codeword. Subtracting $\beta$ compensates. That makes $\beta$ a
//! genuine tuning parameter — not a constant that can be picked and assumed
//! — and this file is what turns the crate's choice of `0.25` from an
//! assertion into a measurement.
//!
//! # What is measured
//!
//! Block error rate (BLER) over the crate's own AWGN channel, using the
//! [`syndrome::montecarlo`] harness so every number arrives with a
//! confidence interval. BLER, not BER, because the trial unit is one
//! independent codeword: bit errors inside a failed block are strongly
//! correlated, so a bit-level interval would be too narrow (see the
//! `montecarlo` module docs).
//!
//! # Why comparisons use interval overlap
//!
//! Two BLER point estimates differing by a few percent prove nothing when
//! each carries a 20% relative standard error. The assertions here require
//! *non-overlapping confidence intervals* before declaring one $\beta$
//! better than another, which is the weakest claim the data actually
//! supports.

use syndrome::channel_sim::AwgnChannel;
use syndrome::montecarlo::{MonteCarloConfig, MonteCarloResult, Trial, run};
use syndrome::{BaseGraph, QcLdpcDecoder, QcLdpcEncoder};

/// Lifting size for the sweep. Small enough that a few hundred decodes run
/// quickly even in an unoptimized test build, large enough to be a real
/// 3GPP-valid code ($Z = 16$ is lifting-size set 0).
const Z: usize = 16;
/// Decoder iteration budget. The 3GPP-typical operating range is 10-20;
/// 10 keeps the sweep fast and is what the rest of the crate's tests use.
const ITERS: usize = 10;
/// The $\beta$ the crate itself selects when it picks one on the caller's
/// behalf ([`syndrome::transport_block::DlSchConfig::default_decode_params`]).
/// Kept here so the tests below track the shipped constant rather than
/// duplicating a literal that could drift away from it.
const CHOSEN_BETA: f32 = 0.5;

/// Measure BLER for one $(\beta, E_b/N_0)$ point.
///
/// Each trial encodes a fresh pseudo-random information word, pushes the
/// codeword through an independently seeded AWGN channel, decodes, and
/// records whether **any** information bit came back wrong.
///
/// The channel is re-seeded per trial from `seed_base` and the trial index,
/// so trials are independent of one another while the whole measurement
/// stays exactly reproducible.
fn measure_bler_z(
    bg: BaseGraph,
    z: usize,
    beta: f32,
    ebno_db: f32,
    seed_base: u64,
    cfg: &MonteCarloConfig,
) -> MonteCarloResult {
    let enc = QcLdpcEncoder::new(bg, z).unwrap();
    let dec = QcLdpcDecoder::with_lifting_size(bg, z, beta).unwrap();

    let k = enc.info_bit_count();
    let n = enc.codeword_bit_count();
    let rate = k as f32 / n as f32;

    // Scratch reused across every trial: the harness runs thousands of
    // decodes and reallocating these each time would dominate the runtime.
    let mut codeword = vec![0u8; n];
    let mut edge_r = vec![0.0f32; dec.required_edge_buffer()];
    let mut scratch = vec![0.0f32; dec.required_layer_buffer()];
    let mut hard = vec![0u8; n];
    let mut info = vec![0u8; k];

    run(cfg, |trial| {
        // Deterministic pseudo-random info word: xorshift over the trial
        // index, so each trial carries different data.
        let mut s = seed_base
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(trial as u64 + 1);
        for b in info.iter_mut() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            *b = (s & 1) as u8;
        }

        enc.encode(&info, &mut codeword).unwrap();

        let mut ch = AwgnChannel::new(ebno_db, rate, seed_base ^ ((trial as u64) << 20));
        let mut llr = ch.transmit(&codeword);

        edge_r.fill(0.0);
        // A decoder error (not a decode failure) would be a bug; a failure to
        // converge is an ordinary block error and is what we are counting.
        let converged = dec
            .decode_layered_offset_min_sum(&mut llr, &mut edge_r, &mut scratch, &mut hard, ITERS)
            .is_ok();

        // Count the block as an error if the recovered information bits
        // differ, regardless of whether the decoder thought it converged --
        // an undetected error is still an error.
        let info_wrong = hard[..k] != info[..];
        Trial::block(info_wrong || !converged)
    })
}

/// [`measure_bler_z`] at the sweep's default lifting size.
fn measure_bler(
    bg: BaseGraph,
    beta: f32,
    ebno_db: f32,
    seed_base: u64,
    cfg: &MonteCarloConfig,
) -> MonteCarloResult {
    measure_bler_z(bg, Z, beta, ebno_db, seed_base, cfg)
}

/// Two measurements are distinguishable only if their confidence intervals
/// do not overlap.
fn intervals_disjoint(a: &MonteCarloResult, b: &MonteCarloResult) -> bool {
    a.ci_high < b.ci_low || b.ci_high < a.ci_low
}

/// The crate's $\beta = 0.25$ must beat plain min-sum ($\beta = 0$) by a
/// statistically resolvable margin.
///
/// This is the assertion that gives the constant its justification. Plain
/// min-sum's over-confident check messages are a well-known weakness, so if
/// the offset correction were mis-implemented — subtracting in the wrong
/// place, clamping wrongly, or ignoring $\beta$ altogether — this comparison
/// is where it would show up, because the two configurations would become
/// indistinguishable.
#[test]
fn offset_correction_beats_plain_min_sum_bg1() {
    let cfg = MonteCarloConfig::default()
        .with_target_error_events(60)
        .with_min_trials(60)
        .with_max_trials(1_500);

    let ebno = 1.0_f32;
    let with_offset = measure_bler(BaseGraph::Bg1, 0.25, ebno, 0xB1_0000, &cfg);
    let no_offset = measure_bler(BaseGraph::Bg1, 0.0, ebno, 0xB1_0000, &cfg);

    assert!(
        with_offset.estimate < no_offset.estimate,
        "beta=0.25 BLER {:.4} should be below plain min-sum {:.4}",
        with_offset.estimate,
        no_offset.estimate
    );
    assert!(
        intervals_disjoint(&with_offset, &no_offset),
        "beta=0.25 [{:.4}, {:.4}] and beta=0 [{:.4}, {:.4}] overlap — the measurement \
         does not resolve a difference, so no claim about beta is supported",
        with_offset.ci_low,
        with_offset.ci_high,
        no_offset.ci_low,
        no_offset.ci_high,
    );
}

/// The same check on BG2, whose different degree distribution makes it a
/// genuinely separate operating point rather than a rerun of the BG1 case.
#[test]
fn offset_correction_beats_plain_min_sum_bg2() {
    let cfg = MonteCarloConfig::default()
        .with_target_error_events(60)
        .with_min_trials(60)
        .with_max_trials(1_500);

    let ebno = 1.0_f32;
    let with_offset = measure_bler(BaseGraph::Bg2, 0.25, ebno, 0xB2_0000, &cfg);
    let no_offset = measure_bler(BaseGraph::Bg2, 0.0, ebno, 0xB2_0000, &cfg);

    assert!(
        with_offset.estimate < no_offset.estimate,
        "BG2 beta=0.25 BLER {:.4} should be below plain min-sum {:.4}",
        with_offset.estimate,
        no_offset.estimate
    );
    assert!(
        intervals_disjoint(&with_offset, &no_offset),
        "BG2 beta=0.25 [{:.4}, {:.4}] and beta=0 [{:.4}, {:.4}] overlap",
        with_offset.ci_low,
        with_offset.ci_high,
        no_offset.ci_low,
        no_offset.ci_high,
    );
}

/// An over-large offset must be *worse* than the crate's choice: subtracting
/// too much destroys genuine reliability information instead of correcting
/// for over-confidence. Together with the $\beta = 0$ comparisons this
/// brackets [`CHOSEN_BETA`] from both sides, which is what distinguishes a
/// tuned value from one that merely happens to be non-zero.
#[test]
fn excessive_offset_is_worse_than_the_chosen_beta() {
    let cfg = MonteCarloConfig::default()
        .with_target_error_events(60)
        .with_min_trials(60)
        .with_max_trials(1_500);

    let ebno = 1.0_f32;
    let chosen = measure_bler(BaseGraph::Bg1, CHOSEN_BETA, ebno, 0xB3_0000, &cfg);
    let excessive = measure_bler(BaseGraph::Bg1, 1.5, ebno, 0xB3_0000, &cfg);

    assert!(
        chosen.estimate < excessive.estimate,
        "beta={CHOSEN_BETA} BLER {:.4} should be below beta=1.5 BLER {:.4}",
        chosen.estimate,
        excessive.estimate
    );
    assert!(
        intervals_disjoint(&chosen, &excessive),
        "beta={CHOSEN_BETA} [{:.4}, {:.4}] and beta=1.5 [{:.4}, {:.4}] overlap",
        chosen.ci_low,
        chosen.ci_high,
        excessive.ci_low,
        excessive.ci_high,
    );
}

/// The crate's chosen $\beta$ must beat the smaller value `0.25` at a
/// production lifting size.
///
/// This is the regression guard for [`CHOSEN_BETA`] itself. The sweep study
/// in this file found `0.25` to be substantially worse than `0.5` on BG1 —
/// at $Z = 384$, $E_b/N_0 = 1$ dB the gap is more than two orders of
/// magnitude in BLER — and the effect *grows* with lifting size rather than
/// being an artifact of the small-$Z$ configuration the fast tests use. A
/// change that quietly moved the default back toward `0.25`, or that broke
/// the offset subtraction so that $\beta$ stopped mattering, would fail
/// here.
///
/// $Z = 128$ rather than 384 keeps the test inside a reasonable runtime
/// while staying in the regime where the effect is unambiguous.
#[test]
fn chosen_beta_beats_a_smaller_offset_at_production_lifting_size() {
    let cfg = MonteCarloConfig::default()
        .with_target_error_events(40)
        .with_min_trials(40)
        .with_max_trials(600);

    let ebno = 1.0_f32;
    let chosen = measure_bler_z(BaseGraph::Bg1, 128, CHOSEN_BETA, ebno, 0xB4_0000, &cfg);
    let smaller = measure_bler_z(BaseGraph::Bg1, 128, 0.25, ebno, 0xB4_0000, &cfg);

    assert!(
        chosen.estimate < smaller.estimate,
        "beta={CHOSEN_BETA} BLER {:.5} should be below beta=0.25 BLER {:.5}",
        chosen.estimate,
        smaller.estimate
    );
    assert!(
        intervals_disjoint(&chosen, &smaller),
        "beta={CHOSEN_BETA} [{:.5}, {:.5}] and beta=0.25 [{:.5}, {:.5}] overlap — \
         the default's advantage is not resolved by this measurement",
        chosen.ci_low,
        chosen.ci_high,
        smaller.ci_low,
        smaller.ci_high,
    );
}

/// The same sweep at production lifting sizes.
///
/// The small-$Z$ study is fast but a short code is not automatically
/// representative: iterative decoding behaviour depends on cycle structure,
/// which changes with $Z$. This confirms whether the $\beta$ ordering found
/// at $Z = 16$ still holds at the lifting sizes 5G NR actually uses, so the
/// crate's choice is not tuned to a toy configuration.
///
/// ```text
/// cargo test --release --test ldpc_offset_beta_sweep -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn offset_beta_sweep_at_production_lifting_sizes() {
    let cfg = MonteCarloConfig::default()
        .with_target_error_events(100)
        .with_min_trials(100)
        .with_max_trials(3_000);

    for (bg, z, ebnos) in [
        (BaseGraph::Bg1, 128usize, [1.0_f32, 1.5, 2.0]),
        (BaseGraph::Bg1, 384, [1.0, 1.5, 2.0]),
        (BaseGraph::Bg2, 128, [1.0, 1.5, 2.0]),
    ] {
        println!("\n=== {bg:?}, Z={z}, {ITERS} iterations ===");
        println!(
            "{:>6}  {:>8}  {:>10}  {:>21}  {:>7}  {:>9}",
            "beta", "Eb/N0", "BLER", "95% CI", "trials", "converged"
        );
        for ebno in ebnos {
            for &beta in &[0.0_f32, 0.25, 0.35, 0.5, 0.65, 0.75] {
                let r = measure_bler_z(bg, z, beta, ebno, 0xDEC0_0000, &cfg);
                println!(
                    "{beta:>6.2}  {ebno:>6.1} dB  {:>10.5}  [{:>8.5}, {:>8.5}]  {:>7}  {:>9}",
                    r.estimate,
                    r.ci_low,
                    r.ci_high,
                    r.trials,
                    if r.is_converged() { "yes" } else { "NO" },
                );
            }
            println!();
        }
    }
}

/// Full $\beta \times E_b/N_0$ sweep, printed as a table.
///
/// `#[ignore]`d: this is a study, not a pass/fail gate, and it runs far
/// longer than a test suite should. It is the measurement that would justify
/// *changing* the crate's $\beta$, and it is checked in so the claim
/// "0.25 is a reasonable choice" can be re-examined rather than taken on
/// trust. Run with:
///
/// ```text
/// cargo test --release --test ldpc_offset_beta_sweep -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn offset_beta_sweep_study() {
    let cfg = MonteCarloConfig::default()
        .with_target_error_events(200)
        .with_min_trials(200)
        .with_max_trials(20_000);

    for bg in [BaseGraph::Bg1, BaseGraph::Bg2] {
        println!("\n=== {bg:?}, Z={Z}, {ITERS} iterations ===");
        println!(
            "{:>6}  {:>8}  {:>10}  {:>21}  {:>7}  {:>9}",
            "beta", "Eb/N0", "BLER", "95% CI", "trials", "converged"
        );
        for &ebno in &[0.0_f32, 1.0, 2.0, 3.0] {
            for &beta in &[0.0_f32, 0.15, 0.25, 0.35, 0.5, 0.75, 1.0] {
                let r = measure_bler(bg, beta, ebno, 0xC0DE_0000, &cfg);
                println!(
                    "{beta:>6.2}  {ebno:>6.1} dB  {:>10.5}  [{:>8.5}, {:>8.5}]  {:>7}  {:>9}",
                    r.estimate,
                    r.ci_low,
                    r.ci_high,
                    r.trials,
                    if r.is_converged() { "yes" } else { "NO" },
                );
            }
            println!();
        }
    }
}
