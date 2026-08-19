//! Transcendental `f32`/`f64` methods that `core` does not provide.
//!
//! `std::f32`/`std::f64` have inherent methods like `.round()`, `.powf()`,
//! `.exp()`, `.ln_1p()` — but these need a math library (round to nearest,
//! polynomial series for exp/ln, ...) that `core` deliberately does not
//! assume exists, since `core` must build for targets with no libm at all.
//! `std` supplies them from the platform's libm; under the `no_std` feature
//! there is no platform to ask, so [`libm`] (a pure-Rust reimplementation,
//! the standard fix for exactly this gap) supplies them instead.
//!
//! Every method here is named `*_ext` specifically so it never shadows or
//! competes with the inherent `std` method of the same underlying operation
//! — under `std` this trait just forwards to that inherent method, so a
//! std build's numerics are provably unchanged by this module existing.

/// Extension trait supplying the handful of transcendental operations this
/// crate's algorithms need, that `core` alone does not provide. Implemented
/// for both `f32` and `f64` (only the methods each caller actually needs).
pub(crate) trait FloatExt: Sized {
    /// Round to the nearest integer, ties away from zero — see
    /// `f32::round`/`f64::round`.
    fn round_ext(self) -> Self;
    /// `self` raised to the power `y` — see `f32::powf`/`f64::powf`.
    fn powf_ext(self, y: Self) -> Self;
    /// $e^{\text{self}}$ — see `f32::exp`/`f64::exp`.
    fn exp_ext(self) -> Self;
    /// $\ln(1 + \text{self})$, accurate near `self = 0` (unlike
    /// `(1.0 + self).ln_ext()`, which cancels) — see `f32::ln_1p`.
    fn ln_1p_ext(self) -> Self;
}

#[cfg(not(feature = "no_std"))]
impl FloatExt for f32 {
    #[inline]
    fn round_ext(self) -> Self {
        self.round()
    }
    #[inline]
    fn powf_ext(self, y: Self) -> Self {
        self.powf(y)
    }
    #[inline]
    fn exp_ext(self) -> Self {
        self.exp()
    }
    #[inline]
    fn ln_1p_ext(self) -> Self {
        self.ln_1p()
    }
}

#[cfg(not(feature = "no_std"))]
impl FloatExt for f64 {
    #[inline]
    fn round_ext(self) -> Self {
        self.round()
    }
    #[inline]
    fn powf_ext(self, y: Self) -> Self {
        self.powf(y)
    }
    #[inline]
    fn exp_ext(self) -> Self {
        self.exp()
    }
    #[inline]
    fn ln_1p_ext(self) -> Self {
        self.ln_1p()
    }
}

#[cfg(feature = "no_std")]
impl FloatExt for f32 {
    #[inline]
    fn round_ext(self) -> Self {
        libm::roundf(self)
    }
    #[inline]
    fn powf_ext(self, y: Self) -> Self {
        libm::powf(self, y)
    }
    #[inline]
    fn exp_ext(self) -> Self {
        libm::expf(self)
    }
    #[inline]
    fn ln_1p_ext(self) -> Self {
        libm::log1pf(self)
    }
}

#[cfg(feature = "no_std")]
impl FloatExt for f64 {
    #[inline]
    fn round_ext(self) -> Self {
        libm::round(self)
    }
    #[inline]
    fn powf_ext(self, y: Self) -> Self {
        libm::pow(self, y)
    }
    #[inline]
    fn exp_ext(self) -> Self {
        libm::exp(self)
    }
    #[inline]
    fn ln_1p_ext(self) -> Self {
        libm::log1p(self)
    }
}

#[cfg(test)]
mod tests {
    /// The `no_std`-feature build of this trait routes through
    /// `libm::{roundf,expf,log1pf,powf}` instead of the `std` inherent
    /// methods, but `cargo test --features no_std` does not build (test code
    /// throughout the crate was never ported — see `Cargo.toml`'s `libm`
    /// dev-dependency comment for why that combination is not meaningful).
    /// So this calls `libm::` directly, as a plain dev-dependency, checking
    /// exactly the functions [`FloatExt`]'s `no_std` impls forward to
    /// against `std`'s — in an ordinary `cargo test --all` run, every time,
    /// not only when someone remembers to pass `--features no_std`.
    #[test]
    fn libm_matches_std() {
        // Includes every half-integer tie (0.5, 1.5, 2.5 and their
        // negations) deliberately: `round`'s tie-breaking rule (away from
        // zero, both here) is a real behavioral choice a different libm
        // implementation could get right for non-tie inputs while disagreeing
        // exactly at a tie -- quantize.rs's rounding needs consistent
        // behavior regardless of which path computed it.
        for &x in &[
            -3.7_f32, -2.5, -1.5, -1.0, -0.5, 0.0, 0.3, 0.5, 1.0, 1.5, 2.5, 10.25,
        ] {
            assert!((libm::roundf(x) - x.round()).abs() < 1e-6, "roundf({x})");
            assert!(
                (libm::expf(x) - x.exp()).abs() / x.exp().abs().max(1.0) < 1e-5,
                "expf({x})"
            );
            assert!(
                (libm::log1pf(x.abs()) - x.abs().ln_1p()).abs() < 1e-5,
                "log1pf({x})"
            );
        }
        for &(x, y) in &[(2.0_f32, 0.25), (10.0, 2.0), (0.5, 3.0)] {
            assert!(
                (libm::powf(x, y) - x.powf(y)).abs() < 1e-4,
                "powf({x}, {y})"
            );
        }
        // f64: only `powf_ext` is actually used at f64 (polar.rs), but check
        // the others too since libm ships them and a future f64 call site
        // would silently rely on this being correct.
        for &x in &[-3.7_f64, -1.0, -0.5, 0.0, 0.3, 1.0, 2.5, 10.25] {
            assert!((libm::round(x) - x.round()).abs() < 1e-9, "round({x})");
            assert!(
                (libm::exp(x) - x.exp()).abs() / x.exp().abs().max(1.0) < 1e-9,
                "exp({x})"
            );
            assert!(
                (libm::log1p(x.abs()) - x.abs().ln_1p()).abs() < 1e-9,
                "log1p({x})"
            );
        }
        for &(x, y) in &[(2.0_f64, 0.25), (10.0, 2.0), (0.5, 3.0)] {
            assert!((libm::pow(x, y) - x.powf(y)).abs() < 1e-9, "pow({x}, {y})");
        }
    }
}
