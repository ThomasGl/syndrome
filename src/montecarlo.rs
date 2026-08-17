//! Monte Carlo estimation with a statistically-justified stopping rule.
//!
//! Measuring a decoder's bit or block error rate means estimating a
//! probability $p$ from a finite number of random trials. Two questions
//! decide whether such a measurement means anything, and this module exists
//! to answer both explicitly rather than by eyeballing a number:
//!
//! 1. **When is it allowed to stop?** Running a fixed trial count is the
//!    usual shortcut, and it is wrong in both directions: at high SNR a
//!    million trials may produce zero errors (no information about $p$), and
//!    at low SNR a thousand trials are already more than enough. The
//!    statistically meaningful budget is counted in *error events observed*,
//!    not trials run.
//!
//! 2. **How precise is the answer?** A bare point estimate $\hat p = k/n$
//!    invites over-reading. Every result here carries a confidence interval
//!    so a reader can see whether two measurements are actually
//!    distinguishable.
//!
//! # Why the stopping rule counts errors, not trials
//!
//! For $n$ independent Bernoulli trials with success probability $p$, the
//! observed count $k$ has variance $n p (1-p)$, so the estimator
//! $\hat p = k/n$ has relative standard error
//!
//! $$\frac{\sigma_{\hat p}}{p} = \sqrt{\frac{1-p}{p\,n}} \approx
//!   \frac{1}{\sqrt{k}} \quad \text{for small } p .$$
//!
//! The relative precision therefore depends on $k$ — the number of error
//! events — and *not* on $n$. Collecting $k = 100$ error events gives about
//! 10% relative standard error whether that took $10^3$ trials or $10^9$.
//! This is why [`MonteCarloConfig::target_error_events`] is the primary
//! budget, and it is the standard practice in FEC simulation for exactly
//! this reason.
//!
//! # Interval construction
//!
//! Intervals are Wilson score intervals, not the textbook normal
//! approximation $\hat p \pm z\sqrt{\hat p(1-\hat p)/n}$. The normal
//! approximation degenerates precisely where FEC work lives: at small $\hat
//! p$ it produces intervals that reach below zero, and at $k = 0$ it
//! collapses to the zero-width interval $[0, 0]$, asserting certainty from
//! an observation that carries almost none. The Wilson interval solves
//!
//! $$\frac{|\hat p - p|}{\sqrt{p(1-p)/n}} = z$$
//!
//! for $p$, which stays inside $[0, 1]$ at every $k$ and yields a sensible
//! one-sided bound $p \lesssim z^2/n$ when $k = 0$ — the same order as the
//! familiar "rule of three".
//!
//! # What the interval does and does not cover
//!
//! The interval is a statement about **sampling noise only**: how much
//! $\hat p$ would move if the simulation were repeated with different random
//! draws. It says nothing about whether the channel model, the SNR
//! calibration, or the decoder itself is correct, and a tight interval
//! around a wrong number is still wrong.
//!
//! It is also exact only when the counted events are mutually independent.
//! That holds for **block**-level events (one independent codeword per
//! trial, the usual BLER setup). It does **not** hold for **bit**-level
//! events within one codeword: a decoder that fails on a block typically
//! emits a burst of correlated bit errors, so the effective sample size is
//! smaller than the bit count and a bit-level Wilson interval computed from
//! raw bit counts is **too narrow**. For that reason
//! [`MonteCarloResult::ci_low`]/[`MonteCarloResult::ci_high`] should be read
//! as a genuine confidence interval for per-block quantities, and as an
//! optimistic lower bound on the true width for per-bit quantities. Where a
//! trustworthy bit-level interval is needed, make each trial contribute one
//! independent block and treat the per-block bit-error count as the trial's
//! payload — the [`Trial`] type carries both, so the trial count remains the
//! independent unit.
//!
//! # Examples
//!
//! ```
//! use syndrome::montecarlo::{MonteCarloConfig, StopReason, run};
//!
//! // A synthetic source that errs on one trial in eight.
//! let mut i = 0usize;
//! let cfg = MonteCarloConfig::default().with_target_error_events(50);
//! let result = run(&cfg, |_| {
//!     i += 1;
//!     syndrome::montecarlo::Trial::block(i % 8 == 0)
//! });
//!
//! assert_eq!(result.stop_reason, StopReason::TargetErrorEvents);
//! assert!(result.ci_low <= 0.125 && 0.125 <= result.ci_high);
//! ```

/// One trial's contribution to a Monte Carlo estimate.
///
/// A trial reports how many *events* it observed out of how many *samples*
/// it drew. The two common shapes:
///
/// * **Block error rate** — [`Trial::block`]: one sample per trial, one
///   event if the block failed. The trial is the independent unit, so the
///   resulting interval is exact.
/// * **Bit error rate** — [`Trial::bits`]: one decoded codeword per trial
///   contributing its bit-error count out of its bit count. See this
///   module's note on correlated bit errors before trusting the interval
///   width here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Trial {
    /// Number of events (errors) observed in this trial.
    pub events: u64,
    /// Number of samples drawn in this trial. Must be `>= events`.
    pub samples: u64,
}

impl Trial {
    /// A single pass/fail trial: one sample, one event if `failed`.
    ///
    /// # Examples
    ///
    /// ```
    /// use syndrome::montecarlo::Trial;
    ///
    /// assert_eq!(Trial::block(true).events, 1);
    /// assert_eq!(Trial::block(false).events, 0);
    /// assert_eq!(Trial::block(true).samples, 1);
    /// ```
    pub fn block(failed: bool) -> Self {
        Self {
            events: u64::from(failed),
            samples: 1,
        }
    }

    /// A trial contributing `errors` bit errors out of `total` bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use syndrome::montecarlo::Trial;
    ///
    /// let t = Trial::bits(3, 1024);
    /// assert_eq!((t.events, t.samples), (3, 1024));
    /// ```
    pub fn bits(errors: u64, total: u64) -> Self {
        Self {
            events: errors,
            samples: total,
        }
    }
}

/// Why a Monte Carlo run stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// Enough error events accumulated to reach the configured relative
    /// precision. This is the statistically meaningful outcome.
    TargetErrorEvents,
    /// The trial budget was exhausted before enough error events appeared.
    /// The estimate is still usable — with `events == 0` it is best read as
    /// the upper bound [`MonteCarloResult::ci_high`] rather than as a point
    /// estimate — but it did **not** reach the requested precision, and a
    /// caller reporting it should say so.
    MaxTrials,
}

/// Configuration for a Monte Carlo run.
///
/// The defaults target roughly 10% relative standard error (100 error
/// events) with a 95% confidence level, which is the usual operating point
/// for a BER/BLER curve where the interesting differences are factors of
/// two or more.
#[derive(Debug, Clone, Copy)]
pub struct MonteCarloConfig {
    /// Stop once this many error events have accumulated. Relative standard
    /// error is approximately $1/\sqrt{k}$ for $k$ events, so 100 events
    /// gives ~10% and 400 events gives ~5%.
    pub target_error_events: u64,
    /// Always run at least this many trials, even if the target error count
    /// is reached sooner. Guards against a wildly wrong estimate from a
    /// handful of trials that happened to fail.
    pub min_trials: usize,
    /// Never run more than this many trials. Bounds the runtime at high SNR
    /// where errors may be arbitrarily rare.
    pub max_trials: usize,
    /// Standard-normal quantile setting the confidence level: `1.959964`
    /// for 95%, `2.575829` for 99%, `1.644854` for 90%.
    pub z: f64,
}

impl Default for MonteCarloConfig {
    fn default() -> Self {
        Self {
            target_error_events: 100,
            min_trials: 100,
            max_trials: 1_000_000,
            z: 1.959_963_984_540_054,
        }
    }
}

impl MonteCarloConfig {
    /// Set the error-event budget (the primary precision control).
    ///
    /// # Examples
    ///
    /// ```
    /// use syndrome::montecarlo::MonteCarloConfig;
    ///
    /// let cfg = MonteCarloConfig::default().with_target_error_events(400);
    /// assert_eq!(cfg.target_error_events, 400);
    /// ```
    pub fn with_target_error_events(mut self, events: u64) -> Self {
        self.target_error_events = events;
        self
    }

    /// Set the maximum number of trials (the runtime bound).
    ///
    /// # Examples
    ///
    /// ```
    /// use syndrome::montecarlo::MonteCarloConfig;
    ///
    /// let cfg = MonteCarloConfig::default().with_max_trials(5_000);
    /// assert_eq!(cfg.max_trials, 5_000);
    /// ```
    pub fn with_max_trials(mut self, trials: usize) -> Self {
        self.max_trials = trials;
        self
    }

    /// Set the minimum number of trials.
    ///
    /// # Examples
    ///
    /// ```
    /// use syndrome::montecarlo::MonteCarloConfig;
    ///
    /// let cfg = MonteCarloConfig::default().with_min_trials(10);
    /// assert_eq!(cfg.min_trials, 10);
    /// ```
    pub fn with_min_trials(mut self, trials: usize) -> Self {
        self.min_trials = trials;
        self
    }
}

/// The outcome of a Monte Carlo run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MonteCarloResult {
    /// Number of trials executed.
    pub trials: usize,
    /// Total error events accumulated across all trials.
    pub events: u64,
    /// Total samples drawn across all trials.
    pub samples: u64,
    /// Point estimate $\hat p = \text{events} / \text{samples}$.
    pub estimate: f64,
    /// Lower end of the Wilson score interval.
    pub ci_low: f64,
    /// Upper end of the Wilson score interval.
    pub ci_high: f64,
    /// Relative standard error $\sqrt{(1-\hat p)/(\hat p\, n)}$ of the point
    /// estimate; `f64::INFINITY` when no events were observed.
    pub relative_standard_error: f64,
    /// Why the run stopped. Treat [`StopReason::MaxTrials`] as "did not
    /// reach the requested precision".
    pub stop_reason: StopReason,
}

impl MonteCarloResult {
    /// Whether the run reached its configured precision target.
    ///
    /// # Examples
    ///
    /// ```
    /// use syndrome::montecarlo::{MonteCarloConfig, Trial, run};
    ///
    /// let cfg = MonteCarloConfig::default()
    ///     .with_target_error_events(10)
    ///     .with_min_trials(1);
    /// let converged = run(&cfg, |_| Trial::block(true));
    /// assert!(converged.is_converged());
    ///
    /// let cfg = cfg.with_max_trials(50);
    /// let starved = run(&cfg, |_| Trial::block(false));
    /// assert!(!starved.is_converged());
    /// ```
    pub fn is_converged(&self) -> bool {
        self.stop_reason == StopReason::TargetErrorEvents
    }
}

/// Wilson score interval for a binomial proportion.
///
/// Solves $|\hat p - p| = z\sqrt{p(1-p)/n}$ for $p$, giving
///
/// $$\frac{\hat p + \frac{z^2}{2n} \pm
///   z\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}}
///   {1 + \frac{z^2}{n}} .$$
///
/// # Arguments
///
/// * `events` — observed successes $k$.
/// * `samples` — trials $n$.
/// * `z` — standard-normal quantile (`1.959964` for 95%).
///
/// # Returns
///
/// `(low, high)`, clamped to $[0, 1]$. Returns `(0.0, 1.0)` when
/// `samples == 0`, the only honest interval given no data.
///
/// # Examples
///
/// ```
/// use syndrome::montecarlo::wilson_interval;
///
/// // With zero observed events the interval is one-sided but non-degenerate,
/// // unlike the normal approximation which would collapse to [0, 0].
/// let (lo, hi) = wilson_interval(0, 1000, 1.959964);
/// assert_eq!(lo, 0.0);
/// assert!(hi > 0.0 && hi < 0.01);
/// ```
pub fn wilson_interval(events: u64, samples: u64, z: f64) -> (f64, f64) {
    if samples == 0 {
        return (0.0, 1.0);
    }
    let n = samples as f64;
    let p_hat = events as f64 / n;
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let center = (p_hat + z2 / (2.0 * n)) / denom;
    let half = (z / denom) * (p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n)).sqrt();

    // The two extreme counts have exact endpoints, and computing them by
    // subtraction instead would return a rounding artifact rather than the
    // right answer. At `events == 0` the formula reduces to
    // `center == half == (z^2/2n)/denom`, so the lower bound is exactly 0 --
    // but `center - half` is a cancellation of two equal quantities that
    // floating point resolves to a tiny positive number (~1e-20), which
    // `.max(0.0)` cannot clean up because it is on the wrong side of zero.
    // The `events == samples` case is the mirror image at the upper end.
    let low = if events == 0 {
        0.0
    } else {
        (center - half).max(0.0)
    };
    let high = if events == samples {
        1.0
    } else {
        (center + half).min(1.0)
    };
    (low, high)
}

/// Run trials until the error-event target or the trial budget is reached.
///
/// `trial` is called with the zero-based trial index and returns that
/// trial's [`Trial`] contribution. It is called at least
/// [`MonteCarloConfig::min_trials`] times and at most
/// [`MonteCarloConfig::max_trials`] times.
///
/// # Arguments
///
/// * `cfg` — budget and confidence level.
/// * `trial` — the simulation body; receives the trial index.
///
/// # Returns
///
/// A [`MonteCarloResult`] carrying the point estimate, its Wilson interval,
/// the relative standard error, and why the run stopped. Read
/// [`MonteCarloResult::is_converged`] before quoting the point estimate.
///
/// # Examples
///
/// ```
/// use syndrome::montecarlo::{MonteCarloConfig, Trial, run};
///
/// // Deterministic 1-in-4 failure pattern.
/// let cfg = MonteCarloConfig::default().with_target_error_events(40);
/// let r = run(&cfg, |i| Trial::block(i % 4 == 0));
/// assert!(r.is_converged());
/// assert!(r.ci_low <= 0.25 && 0.25 <= r.ci_high);
/// ```
pub fn run<F>(cfg: &MonteCarloConfig, mut trial: F) -> MonteCarloResult
where
    F: FnMut(usize) -> Trial,
{
    let mut events: u64 = 0;
    let mut samples: u64 = 0;
    let mut trials: usize = 0;
    let mut stop_reason = StopReason::MaxTrials;

    while trials < cfg.max_trials {
        let t = trial(trials);
        events = events.saturating_add(t.events);
        samples = samples.saturating_add(t.samples);
        trials += 1;

        if trials >= cfg.min_trials && events >= cfg.target_error_events {
            stop_reason = StopReason::TargetErrorEvents;
            break;
        }
    }

    let estimate = if samples == 0 {
        0.0
    } else {
        events as f64 / samples as f64
    };
    let (ci_low, ci_high) = wilson_interval(events, samples, cfg.z);
    let relative_standard_error = if events == 0 || samples == 0 {
        f64::INFINITY
    } else {
        ((1.0 - estimate) / (estimate * samples as f64)).sqrt()
    };

    MonteCarloResult {
        trials,
        events,
        samples,
        estimate,
        ci_low,
        ci_high,
        relative_standard_error,
        stop_reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic Bernoulli source: xorshift64 compared against a
    /// threshold, so a "random" test is exactly reproducible.
    struct Bernoulli {
        state: u64,
        p: f64,
    }

    impl Bernoulli {
        fn new(p: f64, seed: u64) -> Self {
            let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            Self {
                state: if z == 0 { 1 } else { z },
                p,
            }
        }

        fn draw(&mut self) -> bool {
            let mut x = self.state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.state = x;
            // Use the top 53 bits so the uniform has full f64 mantissa
            // resolution; comparing against `p` then yields P(true) = p.
            let u = (x >> 11) as f64 / (1u64 << 53) as f64;
            u < self.p
        }
    }

    /// The estimate must land close to the true probability, and the
    /// interval must contain it, for a source whose `p` is known exactly.
    #[test]
    fn estimate_and_interval_track_a_known_probability() {
        for &p in &[0.5_f64, 0.25, 0.1, 0.01] {
            let mut src = Bernoulli::new(p, 0xA11CE);
            let cfg = MonteCarloConfig::default()
                .with_target_error_events(2000)
                .with_max_trials(5_000_000);
            let r = run(&cfg, |_| Trial::block(src.draw()));

            assert!(r.is_converged(), "p={p} did not converge");
            assert!(
                r.ci_low <= p && p <= r.ci_high,
                "p={p} outside interval [{}, {}] (estimate {})",
                r.ci_low,
                r.ci_high,
                r.estimate
            );
            // With 2000 events the relative standard error is ~2.2%, so a
            // 10% relative tolerance is a loose but non-vacuous bound.
            assert!(
                (r.estimate - p).abs() / p < 0.10,
                "p={p} estimate {} off by more than 10%",
                r.estimate
            );
        }
    }

    /// The interval's *coverage* is the property that actually matters and
    /// the one most easily got wrong: over many independent repetitions, a
    /// 95% interval must contain the true value about 95% of the time.
    /// Checked here by running 400 independent fixed-size experiments at a
    /// small `p` — the regime where the naive normal-approximation interval
    /// under-covers badly and Wilson does not.
    #[test]
    fn wilson_interval_achieves_nominal_coverage() {
        const P: f64 = 0.05;
        const REPS: usize = 400;
        const N: usize = 300;

        let mut covered = 0usize;
        for rep in 0..REPS {
            let mut src = Bernoulli::new(P, 0xC0FFEE + rep as u64);
            // Fixed-size experiment: no early stop, so the coverage
            // statement is the textbook one.
            let cfg = MonteCarloConfig::default()
                .with_target_error_events(u64::MAX)
                .with_min_trials(N)
                .with_max_trials(N);
            let r = run(&cfg, |_| Trial::block(src.draw()));
            assert_eq!(r.trials, N);
            if r.ci_low <= P && P <= r.ci_high {
                covered += 1;
            }
        }

        let rate = covered as f64 / REPS as f64;
        // Wilson is slightly conservative for small p, so the achieved rate
        // sits at or a little above nominal. The binomial standard error of
        // this coverage measurement itself is sqrt(.95*.05/400) ~ 1.1%, so
        // allow a 4-sigma band below nominal and accept anything above.
        assert!(
            rate >= 0.91,
            "95% interval covered only {rate:.3} of {REPS} repetitions — under-covering"
        );
        assert!(
            rate <= 1.0,
            "coverage rate {rate:.3} is impossible — counting bug"
        );
    }

    /// A run that never sees an error must not claim precision: it stops on
    /// the trial budget, reports an infinite relative standard error, and
    /// yields a one-sided interval whose upper bound is on the order of
    /// $z^2/n$ (the "rule of three" scale) rather than the degenerate
    /// $[0, 0]$ the normal approximation would give.
    #[test]
    fn zero_events_yields_a_bound_not_a_false_certainty() {
        let n = 10_000usize;
        let cfg = MonteCarloConfig::default()
            .with_target_error_events(100)
            .with_min_trials(1)
            .with_max_trials(n);
        let r = run(&cfg, |_| Trial::block(false));

        assert_eq!(r.stop_reason, StopReason::MaxTrials);
        assert!(!r.is_converged());
        assert_eq!(r.events, 0);
        assert_eq!(r.estimate, 0.0);
        assert_eq!(r.ci_low, 0.0);
        assert!(r.relative_standard_error.is_infinite());
        // z^2/n = 3.84/10000 = 3.84e-4; require the same order of magnitude.
        assert!(
            r.ci_high > 1.0 / n as f64 && r.ci_high < 10.0 / n as f64,
            "upper bound {} not on the 1/n scale",
            r.ci_high
        );
    }

    /// `min_trials` must win over an early error-target hit, so a run that
    /// fails its first few trials cannot report a converged estimate from a
    /// tiny sample.
    #[test]
    fn min_trials_overrides_an_early_error_target() {
        let cfg = MonteCarloConfig::default()
            .with_target_error_events(5)
            .with_min_trials(500)
            .with_max_trials(10_000);
        // Every trial fails: the error target is met at trial 5, but
        // min_trials forces 500.
        let r = run(&cfg, |_| Trial::block(true));
        assert_eq!(r.trials, 500);
        assert_eq!(r.events, 500);
        assert!(r.is_converged());
    }

    /// `max_trials` is a hard bound even when the error target is never met.
    #[test]
    fn max_trials_is_a_hard_bound() {
        let cfg = MonteCarloConfig::default()
            .with_target_error_events(u64::MAX)
            .with_min_trials(1)
            .with_max_trials(777);
        let mut calls = 0usize;
        let r = run(&cfg, |_| {
            calls += 1;
            Trial::block(true)
        });
        assert_eq!(calls, 777);
        assert_eq!(r.trials, 777);
        assert_eq!(r.stop_reason, StopReason::MaxTrials);
    }

    /// Relative standard error must follow $1/\sqrt{k}$ in the small-$p$
    /// regime: stopping at 100 events gives ~10%, at 400 events ~5%.
    #[test]
    fn relative_standard_error_follows_inverse_sqrt_of_event_count() {
        for (target, expected) in [(100u64, 0.10_f64), (400, 0.05), (2500, 0.02)] {
            let mut src = Bernoulli::new(0.01, 0xBEEF + target);
            let cfg = MonteCarloConfig::default()
                .with_target_error_events(target)
                .with_max_trials(10_000_000);
            let r = run(&cfg, |_| Trial::block(src.draw()));
            assert!(r.is_converged());
            // Small-p approximation: rse ~ 1/sqrt(k). Allow 15% relative
            // slack for the (1-p) factor and for overshooting the target
            // by at most one event.
            let ratio = r.relative_standard_error / expected;
            assert!(
                (0.85..=1.15).contains(&ratio),
                "target={target}: rse {} not ~{expected} (ratio {ratio:.3})",
                r.relative_standard_error
            );
        }
    }

    /// Bit-level trials aggregate events and samples separately, so a run
    /// over blocks of many bits estimates the *bit* error rate while still
    /// counting whole blocks as the trial unit.
    #[test]
    fn bit_level_trials_estimate_a_bit_error_rate() {
        // Each trial contributes exactly 2 bit errors out of 1000 bits, so
        // the BER is exactly 0.002 with no sampling noise at all.
        let cfg = MonteCarloConfig::default()
            .with_target_error_events(100)
            .with_min_trials(1);
        let r = run(&cfg, |_| Trial::bits(2, 1000));
        assert_eq!(r.trials, 50);
        assert_eq!(r.events, 100);
        assert_eq!(r.samples, 50_000);
        assert!((r.estimate - 0.002).abs() < 1e-12);
    }

    /// `wilson_interval` with no samples cannot bound anything, and says so
    /// rather than dividing by zero.
    #[test]
    fn wilson_interval_with_no_samples_is_maximally_wide() {
        assert_eq!(wilson_interval(0, 0, 1.959_964), (0.0, 1.0));
    }

    /// An all-events run pins the estimate at 1.0 with a one-sided interval,
    /// the mirror image of the zero-events case.
    #[test]
    fn wilson_interval_at_unity_is_one_sided() {
        let (lo, hi) = wilson_interval(1000, 1000, 1.959_964);
        assert_eq!(hi, 1.0);
        assert!(lo > 0.99 && lo < 1.0);
    }
}
