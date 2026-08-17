//! How much error-rate performance the fixed-point LDPC decoder gives up.
//!
//! `src/quantize.rs` exists to feed a decoder that carries its messages in
//! `i8` instead of `f32`. Quantization is lossy, so the question that decides
//! whether the fixed-point path is usable is not *whether* it costs something
//! but *how much*, expressed the way link budgets are: as the extra
//! $E_b/N_0$ needed to reach the same error rate. This file is what makes
//! that a number this tree produces rather than one cited from somebody
//! else's decoder.
//!
//! # Method
//!
//! **Paired trials.** Every trial draws one channel realisation and hands the
//! *same* received vector to both decoders — the `f32` path directly, the
//! fixed-point path after quantization. Common random numbers take the
//! channel's variance out of the comparison, so a difference between the two
//! paths is resolved far more sharply than two independent sweeps of the same
//! length could resolve it. It also makes the natural statistic a paired one:
//! the trials on which the two paths *disagree* are the only ones carrying
//! information about the difference, and if the two never disagreed the
//! measured difference would be exactly zero with zero uncertainty.
//!
//! **Error-event budget.** Trials run until a target number of block errors
//! has accumulated on the `f32` reference, not for a fixed trial count,
//! because the relative precision of a rate estimate is set by the number of
//! events observed rather than the number of trials — see
//! [`syndrome::montecarlo`]. Block error rate rather than bit error rate is
//! the primary quantity for that module's other reason: one codeword per
//! trial is an independent unit, whereas bit errors inside a failed block are
//! strongly correlated, so a bit-level interval computed from raw bit counts
//! would be too narrow. Bit error rates are reported alongside, without
//! intervals, for comparison against the crate's BER waterfall.
//!
//! **From a rate ratio to decibels.** In the waterfall the block error rate
//! is close to exponential in $E_b/N_0$, $\mathrm{BLER}(E) \approx
//! \mathrm{BLER}(E_0)\thinspace e^{s(E-E_0)}$ with $s < 0$. If the
//! fixed-point curve is the `f32` curve displaced right by $\Delta$ then at
//! any fixed $E$
//!
//! $$\ln \frac{\mathrm{BLER}\_{i8}}{\mathrm{BLER}\_{f32}} = -s\thinspace\Delta
//!   \qquad\Longrightarrow\qquad
//!   \Delta = \frac{1}{\lvert s \rvert}\thinspace
//!   \ln \frac{\mathrm{BLER}\_{i8}}{\mathrm{BLER}\_{f32}} .$$
//!
//! One operating point therefore gives the shift, provided the local slope
//! $s$ is measured too — which [`measure_slope`] does from two neighbouring
//! $E_b/N_0$ values on the `f32` curve. The slope is steep for these codes,
//! of order 10 per dB, and that steepness is what makes a small displacement
//! measurable at all: a shift of 0.01 dB still moves the block error rate by
//! about 10%.
//!
//! **What is claimed.** A difference is claimed only when the confidence
//! interval on the ratio excludes 1. Where it does not, the honest result is
//! a bound — "below $x$ dB at 95% confidence" — and not the point estimate,
//! which at these sample sizes is mostly noise.
//!
//! # What this does not cover
//!
//! The shift is measured on the BPSK AWGN channel of
//! [`syndrome::channel_sim`] at the operating points listed in the study
//! below. Quantization loss depends on the scale relative to the LLR
//! distribution, so it is a function of the operating $E_b/N_0$ and the code
//! rate; the sweep prints enough of the surface to show how flat it is, but
//! a different rate, modulation or fading model needs its own run.
//!
//! # Running the study
//!
//! ```text
//! cargo test --release --test ldpc_int8_quantization_loss -- --ignored --nocapture
//! ```
//!
//! The tests that are not `#[ignore]`d are regression gates on the same
//! machinery, at budgets small enough for an ordinary test run.

use syndrome::channel_sim::AwgnChannel;
use syndrome::montecarlo::{MonteCarloConfig, StopReason, Trial, run, wilson_interval};
use syndrome::qc_ldpc::{BaseGraph, QcLdpcDecoder, QcLdpcEncoder};
use syndrome::quantize::{APP_CLAMP_I8, DEFAULT_SCALE, QuantParams, quantize_llr_i16};

/// 95% two-sided standard-normal quantile, matching the default of
/// [`MonteCarloConfig`].
const Z95: f64 = 1.959_963_984_540_054;

/// Decoder iteration budget. 10 is the 3GPP-typical operating point and what
/// the rest of the crate's LDPC measurements use, so the figure reported here
/// is comparable to them.
const ITERS: usize = 10;

/// The offset $\beta$ the crate selects on the caller's behalf, justified in
/// `tests/ldpc_offset_beta_sweep.rs`. Quantization loss is only meaningful
/// against a decoder that is otherwise tuned.
const BETA: f32 = 0.5;

// ---------------------------------------------------------------------------
// Paired measurement
// ---------------------------------------------------------------------------

/// One paired measurement of the `f32` and fixed-point decoders at a single
/// $E_b/N_0$.
#[derive(Debug, Clone, Copy)]
struct Paired {
    /// Trials executed.
    trials: u64,
    /// Block errors on the `f32` path.
    f32_errors: u64,
    /// Block errors on the fixed-point path.
    i8_errors: u64,
    /// Trials on which only the `f32` path failed.
    f32_only: u64,
    /// Trials on which only the fixed-point path failed.
    i8_only: u64,
    /// Information-bit errors summed over trials, `f32` path.
    f32_bit_errors: u64,
    /// Information-bit errors summed over trials, fixed-point path.
    i8_bit_errors: u64,
    /// Information bits transmitted in total.
    info_bits: u64,
    /// Whether the run met its error-event target.
    converged: bool,
}

impl Paired {
    fn f32_bler(&self) -> f64 {
        self.f32_errors as f64 / self.trials as f64
    }

    fn i8_bler(&self) -> f64 {
        self.i8_errors as f64 / self.trials as f64
    }

    fn f32_bler_ci(&self) -> (f64, f64) {
        wilson_interval(self.f32_errors, self.trials, Z95)
    }

    fn i8_bler_ci(&self) -> (f64, f64) {
        wilson_interval(self.i8_errors, self.trials, Z95)
    }

    fn f32_ber(&self) -> f64 {
        self.f32_bit_errors as f64 / self.info_bits as f64
    }

    fn i8_ber(&self) -> f64 {
        self.i8_bit_errors as f64 / self.info_bits as f64
    }

    /// Point estimate and 95% interval for $\ln(\mathrm{BLER}\_{i8} /
    /// \mathrm{BLER}\_{f32})$, accounting for the pairing.
    ///
    /// The two error counts share every trial on which both paths failed, so
    /// treating them as independent would overstate the uncertainty badly.
    /// Write $a$ for the trials both failed, $b$ for `f32`-only, $c$ for
    /// `i8`-only, $N_f = a + b$ and $N_i = a + c$. The statistic is $D =
    /// \ln N_i - \ln N_f$, and the delta method on the multinomial
    /// $(a, b, c, \cdot)$ gives
    ///
    /// $$\operatorname{Var}(D) \approx a \left(\frac{1}{N_i} -
    ///   \frac{1}{N_f}\right)^2 + \frac{b}{N_f^2} + \frac{c}{N_i^2} .$$
    ///
    /// The linear term of the multinomial quadratic form vanishes identically
    /// here, which is why no $-\left(\sum c_j p_j\right)^2$ correction
    /// appears. The formula has the property the pairing is for: when the two
    /// decoders never disagree ($b = c = 0$, $N_i = N_f$) the variance is
    /// exactly zero, whereas an unpaired interval would still be wide.
    ///
    /// Returns `None` when either path saw no errors, since the log ratio is
    /// then undefined.
    fn log_ratio_ci(&self) -> Option<(f64, f64, f64)> {
        if self.f32_errors == 0 || self.i8_errors == 0 {
            return None;
        }
        let n_f = self.f32_errors as f64;
        let n_i = self.i8_errors as f64;
        let b = self.f32_only as f64;
        let c = self.i8_only as f64;
        let a = (self.f32_errors - self.f32_only) as f64;

        let point = (n_i / n_f).ln();
        let d = 1.0 / n_i - 1.0 / n_f;
        let var = a * d * d + b / (n_f * n_f) + c / (n_i * n_i);
        let half = Z95 * var.sqrt();
        Some((point, point - half, point + half))
    }
}

/// Run paired trials at one $E_b/N_0$.
///
/// The stopping rule counts `f32` block errors, so the reference curve always
/// reaches the requested precision and the fixed-point counts come from the
/// same trials at no extra channel cost.
///
/// # Arguments
///
/// * `bg`, `z` — base graph and lifting size.
/// * `ebno_db` — operating point.
/// * `quant` — fixed-point format under test.
/// * `seed_base` — makes the whole measurement reproducible.
/// * `cfg` — error-event budget and trial bounds.
fn measure_paired(
    bg: BaseGraph,
    z: usize,
    ebno_db: f32,
    quant: QuantParams,
    seed_base: u64,
    cfg: &MonteCarloConfig,
) -> Paired {
    let enc = QcLdpcEncoder::new(bg, z).unwrap();
    let dec = QcLdpcDecoder::with_lifting_size(bg, z, BETA).unwrap();

    let k = enc.info_bit_count();
    let n = enc.codeword_bit_count();
    let rate = k as f32 / n as f32;

    // Allocated once: the loop runs tens of thousands of decodes.
    let mut info = vec![0u8; k];
    let mut codeword = vec![0u8; n];
    let mut llr_f = vec![0.0f32; n];
    let mut edge_f = vec![0.0f32; dec.required_edge_buffer()];
    let mut scratch_f = vec![0.0f32; dec.required_layer_buffer()];
    let mut hard_f = vec![0u8; n];
    let mut app = vec![0i16; n];
    let mut edge_i = vec![0i8; dec.required_edge_buffer()];
    let mut scratch_i = vec![0i8; dec.required_layer_buffer()];
    let mut hard_i = vec![0u8; n];

    let mut acc = Paired {
        trials: 0,
        f32_errors: 0,
        i8_errors: 0,
        f32_only: 0,
        i8_only: 0,
        f32_bit_errors: 0,
        i8_bit_errors: 0,
        info_bits: 0,
        converged: false,
    };

    let result = run(cfg, |trial| {
        let mut s = seed_base
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(trial as u64 + 1);
        for b in info.iter_mut() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            // The low bit of a raw xorshift word is weakly mixed.
            *b = ((s >> 27) & 1) as u8;
        }
        enc.encode(&info, &mut codeword).unwrap();

        let mut ch = AwgnChannel::new(ebno_db, rate, seed_base ^ ((trial as u64) << 20));
        let llr = ch.transmit(&codeword);

        // f32 reference on this realisation.
        llr_f.copy_from_slice(&llr);
        edge_f.fill(0.0);
        dec.decode_layered_offset_min_sum(
            &mut llr_f,
            &mut edge_f,
            &mut scratch_f,
            &mut hard_f,
            ITERS,
        )
        .unwrap();

        // Fixed point on the *same* realisation.
        quantize_llr_i16(&llr, &mut app, quant.scale);
        dec.decode_layered_offset_min_sum_i8(
            &mut app,
            &mut edge_i,
            &mut scratch_i,
            &mut hard_i,
            ITERS,
            quant,
        )
        .unwrap();

        let fb = hard_f[..k]
            .iter()
            .zip(&info)
            .filter(|(a, b)| a != b)
            .count() as u64;
        let ib = hard_i[..k]
            .iter()
            .zip(&info)
            .filter(|(a, b)| a != b)
            .count() as u64;
        let fw = fb != 0;
        let iw = ib != 0;

        acc.trials += 1;
        acc.f32_errors += u64::from(fw);
        acc.i8_errors += u64::from(iw);
        acc.f32_only += u64::from(fw && !iw);
        acc.i8_only += u64::from(iw && !fw);
        acc.f32_bit_errors += fb;
        acc.i8_bit_errors += ib;
        acc.info_bits += k as u64;

        Trial::block(fw)
    });

    acc.converged = result.stop_reason == StopReason::TargetErrorEvents;
    acc
}

/// Local slope $s = \mathrm{d}\ln\mathrm{BLER} / \mathrm{d}(E_b/N_0)$ of the
/// `f32` curve, from two neighbouring operating points.
///
/// Returns `None` if either point saw no errors, since the slope through a
/// zero is undefined.
fn measure_slope(
    bg: BaseGraph,
    z: usize,
    lo_db: f32,
    hi_db: f32,
    seed_base: u64,
    cfg: &MonteCarloConfig,
) -> Option<f64> {
    let q = QuantParams::default();
    let lo = measure_paired(bg, z, lo_db, q, seed_base, cfg);
    let hi = measure_paired(bg, z, hi_db, q, seed_base ^ 0x5555, cfg);
    if lo.f32_errors == 0 || hi.f32_errors == 0 {
        return None;
    }
    Some((hi.f32_bler().ln() - lo.f32_bler().ln()) / f64::from(hi_db - lo_db))
}

/// Convert a paired measurement plus a local slope into an $E_b/N_0$ shift in
/// dB, as `(point, low, high)` at 95% confidence.
///
/// A positive shift means the fixed-point path needs *more* $E_b/N_0$ to
/// reach the same block error rate.
fn shift_db(p: &Paired, slope: f64) -> Option<(f64, f64, f64)> {
    let (point, lo, hi) = p.log_ratio_ci()?;
    let inv = 1.0 / slope.abs();
    Some((point * inv, lo * inv, hi * inv))
}

/// Whether a shift interval resolves a non-zero displacement.
fn resolves_a_shift(shift: (f64, f64, f64)) -> bool {
    shift.1 > 0.0 || shift.2 < 0.0
}

/// Format one paired point as a table row.
fn row(label: &str, ebno: f32, p: &Paired) -> String {
    let (flo, fhi) = p.f32_bler_ci();
    let (ilo, ihi) = p.i8_bler_ci();
    format!(
        "{label:<22} {ebno:>5.2}  {:>9.5} [{:>8.5},{:>8.5}]  {:>9.5} [{:>8.5},{:>8.5}]  \
         {:>9.2e} {:>9.2e}  {:>5} {:>5}  {:>8} {}",
        p.f32_bler(),
        flo,
        fhi,
        p.i8_bler(),
        ilo,
        ihi,
        p.f32_ber(),
        p.i8_ber(),
        p.f32_only,
        p.i8_only,
        p.trials,
        if p.converged { "" } else { " (BUDGET)" },
    )
}

/// Column header matching [`row`].
fn header() -> String {
    format!(
        "{:<22} {:>5}  {:>9} {:>19}  {:>9} {:>19}  {:>9} {:>9}  {:>5} {:>5}  {:>8}",
        "config",
        "Eb/N0",
        "f32 BLER",
        "95% CI",
        "i8 BLER",
        "95% CI",
        "f32 BER",
        "i8 BER",
        "f32only",
        "i8only",
        "trials"
    )
}

// ---------------------------------------------------------------------------
// The study: the measurement the published figure comes from
// ---------------------------------------------------------------------------

/// Full quantization-loss study across both base graphs and two lifting
/// sizes, reporting the $E_b/N_0$ shift with a confidence interval.
///
/// `#[ignore]`d because it runs far longer than a test suite should. This is
/// the run that produces the figure quoted in `src/quantize.rs`, the README
/// and the changelog; re-run it before changing any of them.
///
/// ```text
/// cargo test --release --test ldpc_int8_quantization_loss -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn int8_quantization_loss_study() {
    let cfg = MonteCarloConfig::default()
        .with_target_error_events(2_000)
        .with_min_trials(2_000)
        .with_max_trials(400_000);
    // The slope only needs to be accurate to a few percent, because the shift
    // it divides is itself near zero.
    let slope_cfg = MonteCarloConfig::default()
        .with_target_error_events(400)
        .with_min_trials(400)
        .with_max_trials(200_000);

    let quant = QuantParams::default();
    println!(
        "\ni8 messages, i16 posterior; scale s = {}, beta = {BETA} (beta_q = {}), {ITERS} iterations",
        quant.scale,
        quant.beta_q(BETA)
    );
    println!("BPSK AWGN, paired trials (both decoders see the same received vector)\n");
    println!("{}", header());

    for (bg, z, ebno) in [
        (BaseGraph::Bg1, 128usize, 0.8f32),
        (BaseGraph::Bg1, 384, 0.75),
        (BaseGraph::Bg2, 128, 0.6),
        (BaseGraph::Bg2, 384, 0.6),
    ] {
        let seed = 0x0010_8000 ^ ((z as u64) << 16) ^ (bg as u64);
        let p = measure_paired(bg, z, ebno, quant, seed, &cfg);
        println!("{}", row(&format!("{bg:?} Z={z}"), ebno, &p));

        // Slope bracketing the operating point, not extending beyond it: the
        // waterfall steepens with SNR, so a slope measured to one side would
        // bias the dB conversion.
        let slope = measure_slope(bg, z, ebno - 0.1, ebno + 0.1, seed ^ 0xABCD, &slope_cfg);
        match (slope, shift_db(&p, slope.unwrap_or(f64::NAN))) {
            (Some(s), Some((point, lo, hi))) => {
                println!(
                    "  local slope d(ln BLER)/d(Eb/N0) = {s:.2} per dB  ->  \
                     shift {point:+.4} dB, 95% CI [{lo:+.4}, {hi:+.4}]"
                );
                if resolves_a_shift((point, lo, hi)) {
                    println!("  -> a shift IS resolved at 95% confidence");
                } else {
                    println!(
                        "  -> no shift resolved; the loss is below {:.4} dB at 95% confidence",
                        hi.abs().max(lo.abs())
                    );
                }
            }
            _ => println!("  (no errors on one path — shift undefined at this budget)"),
        }
    }
}

/// How the loss depends on the scale factor $s$.
///
/// The scale is the one parameter of the format that has to be matched to the
/// channel: too small and the LLR distribution is coarsely resolved, too
/// large and its tails clip. This prints the whole curve so the crate's
/// [`DEFAULT_SCALE`] can be read off a measurement instead of asserted, in
/// the same spirit as the $\beta$ sweep.
#[test]
#[ignore]
fn scale_sweep_study() {
    let cfg = MonteCarloConfig::default()
        .with_target_error_events(500)
        .with_min_trials(500)
        .with_max_trials(100_000);

    for (bg, z, ebno) in [
        (BaseGraph::Bg1, 128usize, 0.8f32),
        (BaseGraph::Bg2, 128, 0.6),
    ] {
        println!("\n=== scale sweep: {bg:?} Z={z}, Eb/N0 = {ebno} dB, beta = {BETA} ===");
        println!("{}", header());
        for &scale in &[2.0f32, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 32.0] {
            let quant = QuantParams::default().with_scale(scale);
            let p = measure_paired(bg, z, ebno, quant, 0x5CA1_E000, &cfg);
            println!(
                "{}",
                row(
                    &format!("s={scale:<5} beta_q={}", quant.beta_q(BETA)),
                    ebno,
                    &p
                )
            );
        }
    }
}

/// How much the posterior accumulator width costs.
///
/// The messages have to be `i8` — that is the point of the format. The
/// posterior does not, and this sweep is what decides it: it clamps the
/// accumulator at a range of widths, from the full `i16` down to the same
/// $\pm 127$ the messages use.
#[test]
#[ignore]
fn posterior_width_study() {
    let cfg = MonteCarloConfig::default()
        .with_target_error_events(500)
        .with_min_trials(500)
        .with_max_trials(100_000);

    for (bg, z, ebno) in [
        (BaseGraph::Bg1, 128usize, 0.8f32),
        (BaseGraph::Bg2, 128, 0.6),
    ] {
        println!("\n=== posterior width: {bg:?} Z={z}, Eb/N0 = {ebno} dB, s = {DEFAULT_SCALE} ===");
        println!("{}", header());
        for &clamp in &[i16::MAX, 4095, 2047, 1023, 511, 255, APP_CLAMP_I8] {
            let quant = QuantParams::default().with_app_clamp(clamp);
            let p = measure_paired(bg, z, ebno, quant, 0x0C1A_3000, &cfg);
            println!("{}", row(&format!("app_clamp={clamp}"), ebno, &p));
        }
    }
}

// ---------------------------------------------------------------------------
// Gates
// ---------------------------------------------------------------------------
//
// These run in an ordinary `cargo test`, so they use a small lifting size and
// a small error-event budget. They are not the study — they are the checks
// that would fail if the fixed-point path regressed, or if one of the format
// decisions the study justified were quietly reversed.
//
// Where two fixed-point configurations are compared the comparison is on the
// **paired** log ratio against the shared `f32` reference, not on the two
// marginal Wilson intervals. Both configurations see the same channel
// realisations, so the marginal intervals carry variance that cancels in the
// comparison; requiring those to separate would need several times the
// trials for the same conclusion. Requiring the paired intervals not to
// overlap is still conservative, since they too are positively correlated
// through the shared reference.

/// Budget for the gates: enough error events for the paired comparison to
/// resolve the effects below, small enough to run unoptimized.
fn gate_cfg() -> MonteCarloConfig {
    MonteCarloConfig::default()
        .with_target_error_events(120)
        .with_min_trials(120)
        .with_max_trials(1_500)
}

/// Lifting size for the gates. $Z = 32$ is a valid 3GPP size (iLS 0) and,
/// being a multiple of 32, keeps every AVX2 chunk full while staying small
/// enough to decode thousands of times unoptimized.
const GATE_Z: usize = 32;

/// Operating point for the gates: $\mathrm{BLER} \approx 0.34$ on the `f32`
/// reference at [`GATE_Z`], which reaches the error-event budget in a few
/// hundred trials. Further up the waterfall the same budget costs an order of
/// magnitude more trials for no extra resolving power.
const GATE_EBNO: f32 = 0.75;

/// Paired log-ratio interval, or a panic naming the configuration — every
/// gate needs one and a missing one means the budget was too small.
fn ratio_ci(label: &str, p: &Paired) -> (f64, f64, f64) {
    p.log_ratio_ci().unwrap_or_else(|| {
        panic!(
            "{label}: one path saw no block errors in {} trials (f32 {}, i8 {}), \
             so the comparison has no resolving power at this budget",
            p.trials, p.f32_errors, p.i8_errors
        )
    })
}

/// Whether two intervals are disjoint.
fn disjoint(a: (f64, f64, f64), b: (f64, f64, f64)) -> bool {
    a.1 > b.2 || b.1 > a.2
}

/// Operating point for the posterior-width gate.
///
/// Clamping the posterior does not cost a constant factor: it costs almost
/// nothing where the decoder is already failing often and a great deal where
/// it is not, because a clamped posterior is what turns a converging decode
/// into a stuck one. Measured at [`GATE_Z`], the narrow accumulator is 1.06x
/// worse at 0.75 dB, 1.5x at 1.0 dB and 5.6x at 1.25 dB — it is an error
/// floor, not an offset. 1.0 dB is where the effect is large enough to
/// resolve cheaply without the reference curve becoming too rare to sample.
const GATE_FLOOR_EBNO: f32 = 1.0;

/// Budget for the posterior-width gate: a fixed trial count rather than an
/// error-event target.
///
/// The quantity under test is the ratio between two *fixed-point*
/// configurations, so the reference curve's own precision is not what has to
/// converge; both fixed-point configurations accumulate ample errors in this
/// many trials, while an error-event target on the reference would cost
/// several times the runtime for no extra resolving power.
fn floor_cfg() -> MonteCarloConfig {
    MonteCarloConfig::default()
        .with_target_error_events(u64::MAX)
        .with_min_trials(1_000)
        .with_max_trials(1_000)
}

/// Holding the posterior to the message width must cost a *resolvable*
/// amount, in the direction that makes the wide accumulator the right
/// default.
///
/// This is the assertion behind the two-width design. The posterior is a sum
/// of up to 31 messages for BG1 column 0, so clamping it at $\pm 127$ bites on
/// exactly the variable nodes the decoder is most confident about. If a future
/// change collapsed the accumulator to `i8` — or made the clamp ineffective,
/// which would be a bug in the other direction — this is where it would show.
#[test]
fn narrow_posterior_is_measurably_worse_than_a_wide_one() {
    let cfg = floor_cfg();
    let wide = measure_paired(
        BaseGraph::Bg1,
        GATE_Z,
        GATE_FLOOR_EBNO,
        QuantParams::default(),
        0x0A00_0001,
        &cfg,
    );
    let narrow = measure_paired(
        BaseGraph::Bg1,
        GATE_Z,
        GATE_FLOOR_EBNO,
        QuantParams::default().with_app_clamp(APP_CLAMP_I8),
        0x0A00_0001,
        &cfg,
    );

    let w = ratio_ci("wide posterior", &wide);
    let n = ratio_ci("narrow posterior", &narrow);
    assert!(
        n.0 > w.0,
        "clamping the posterior to +/-{APP_CLAMP_I8} gave log-ratio {:.4} against f32, \
         no worse than the wide accumulator's {:.4}",
        n.0,
        w.0,
    );
    assert!(
        disjoint(n, w),
        "narrow posterior log-ratio [{:.4}, {:.4}] and wide [{:.4}, {:.4}] overlap — \
         this measurement does not resolve a difference, so it does not support \
         choosing the wider accumulator",
        n.1,
        n.2,
        w.1,
        w.2,
    );
}

/// The default format must not be resolvably worse than `f32`, and its loss
/// bound must stay under a tenth of a dB.
///
/// The study measures the loss far more precisely than this; the point here is
/// a regression guard. A mis-quantized $\beta$, a lost saturation, or a scale
/// that no longer suits the LLR distribution would each move the fixed-point
/// path off the `f32` curve by much more than this bound allows.
#[test]
fn default_format_stays_on_the_float_curve() {
    let cfg = gate_cfg();
    let seed = 0xF0A7_0001;
    let p = measure_paired(
        BaseGraph::Bg1,
        GATE_Z,
        GATE_EBNO,
        QuantParams::default(),
        seed,
        &cfg,
    );

    // The slope bracket spans the operating point: the waterfall steepens
    // with SNR, so a slope measured entirely to one side would bias the dB
    // conversion. It needs only a few percent of accuracy, because the shift
    // it divides is itself near zero.
    let slope_cfg = MonteCarloConfig::default()
        .with_target_error_events(80)
        .with_min_trials(80)
        .with_max_trials(1_500);
    let slope = measure_slope(BaseGraph::Bg1, GATE_Z, 0.5, 1.0, seed, &slope_cfg)
        .expect("both slope points must see block errors");
    assert!(
        slope < 0.0,
        "block error rate must fall with Eb/N0; measured slope {slope:.3} per dB"
    );

    let (point, lo, hi) = shift_db(&p, slope).expect("both paths must see block errors");
    assert!(
        lo <= 0.0,
        "the fixed-point path is resolvably worse than f32: shift {point:+.4} dB, \
         95% CI [{lo:+.4}, {hi:+.4}] — the default quantization format has regressed",
    );
    let bound = hi.abs().max(lo.abs());
    assert!(
        bound < 0.10,
        "quantization loss bound {bound:.4} dB (shift {point:+.4} dB, CI [{lo:+.4}, {hi:+.4}]) \
         exceeds 0.10 dB",
    );
}

/// [`DEFAULT_SCALE`] must be bracketed: a much coarser scale and a much
/// clipping-ier one both have to be resolvably worse.
///
/// Bracketing from both sides is what distinguishes a measured choice from a
/// value that merely happens to work — the same standard
/// `tests/ldpc_offset_beta_sweep.rs` holds $\beta$ to. At $s = 2$ one
/// quantization step is half an LLR unit, coarse next to the message
/// magnitudes the decoder compares; at $s = 64$ the representable range is
/// $\pm 2$, so the channel LLR distribution is clipped at well under one
/// standard deviation.
#[test]
fn default_scale_is_bracketed_from_both_sides() {
    let cfg = gate_cfg();
    let seed = 0x5CA1_0001;
    let at = |scale: Option<f32>| {
        let q = match scale {
            Some(s) => QuantParams::default().with_scale(s),
            None => QuantParams::default(),
        };
        measure_paired(BaseGraph::Bg1, GATE_Z, GATE_EBNO, q, seed, &cfg)
    };

    let chosen = at(None);
    let coarse = at(Some(2.0));
    let clipped = at(Some(64.0));

    let c = ratio_ci(&format!("s = {DEFAULT_SCALE}"), &chosen);
    let lo = ratio_ci("s = 2", &coarse);
    let hi = ratio_ci("s = 64", &clipped);

    assert!(
        lo.0 > c.0 && disjoint(lo, c),
        "s = 2 log-ratio [{:.4}, {:.4}] does not sit resolvably above \
         s = {DEFAULT_SCALE}'s [{:.4}, {:.4}]",
        lo.1,
        lo.2,
        c.1,
        c.2,
    );
    assert!(
        hi.0 > c.0 && disjoint(hi, c),
        "s = 64 log-ratio [{:.4}, {:.4}] does not sit resolvably above \
         s = {DEFAULT_SCALE}'s [{:.4}, {:.4}]",
        hi.1,
        hi.2,
        c.1,
        c.2,
    );
}
