//! Shared test-only utilities. Not part of the public API — this module
//! exists purely to give every unit-test suite in the crate one canonical
//! seeded PRNG instead of the five near-identical copies (differing in
//! field shape, constructor, and even output mixing) that had accumulated
//! across `crc.rs`, `bch.rs`, `polar.rs`, `rate_matching.rs`, and
//! `qc_ldpc.rs` from independent development sessions.
//!
//! `tests/common/mod.rs` is the equivalent for integration tests in
//! `tests/*.rs`, which compile as separate crates and cannot see this
//! `#[cfg(test)]`-gated module.

#![cfg(test)]

/// Deterministic, dependency-free xorshift64 PRNG for seeded randomized
/// tests. Not cryptographically secure and not intended to be — the only
/// requirement is a reproducible, well-distributed stream from a seed.
pub(crate) struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    /// Construct from `seed`. A seed of `0` is folded to `1`: xorshift's
    /// state must never be all-zero, since `x ^= x << k` etc. would then
    /// output zero forever.
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    /// Next pseudo-random `u64`.
    pub(crate) fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Next pseudo-random `u64`, additionally passed through a splitmix-style
    /// output-mixing multiply (the "xorshift64*" construction).
    ///
    /// Plain xorshift's low-order output bits are known to have weaker
    /// statistical quality than its high-order bits (a documented weakness
    /// of the xorshift family, not specific to this implementation). That
    /// weakness only matters when a caller extracts individual *bits*
    /// directly from consecutive raw outputs (e.g. building a bitstream by
    /// reading bit 0, bit 1, ... of successive words) rather than treating
    /// each `next_u64` as one opaque wide sample. `qc_ldpc.rs`'s randomized
    /// info-bit generator for its encoder equivalence tests does exactly
    /// that, so it needs this mixed variant rather than [`Xorshift64::next_u64`].
    pub(crate) fn next_u64_mixed(&mut self) -> u64 {
        self.next_u64().wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Next pseudo-random `bool`.
    pub(crate) fn next_bool(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }

    /// Random length in `1..=max` (inclusive), deliberately usable for
    /// lengths not divisible by common SIMD lane widths (4/8/16/32) so
    /// scalar-tail paths get exercised alongside the vectorized bulk.
    pub(crate) fn next_len(&mut self, max: usize) -> usize {
        1 + (self.next_u64() as usize % max)
    }

    /// Random value in `0..bound` (exclusive). Panics if `bound == 0`.
    pub(crate) fn next_below(&mut self, bound: usize) -> usize {
        (self.next_u64() as usize) % bound
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_seed_does_not_stall_at_zero() {
        let mut rng = Xorshift64::new(0);
        assert_ne!(rng.next_u64(), 0);
    }

    #[test]
    fn distinct_seeds_diverge() {
        let mut a = Xorshift64::new(1);
        let mut b = Xorshift64::new(2);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn next_len_is_within_bounds() {
        let mut rng = Xorshift64::new(42);
        for _ in 0..100 {
            let len = rng.next_len(16);
            assert!((1..=16).contains(&len));
        }
    }

    #[test]
    fn mixed_output_differs_from_raw_output() {
        let mut a = Xorshift64::new(7);
        let mut b = Xorshift64::new(7);
        assert_ne!(a.next_u64(), b.next_u64_mixed());
    }
}
