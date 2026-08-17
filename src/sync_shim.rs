//! Synchronization primitives that swap between `std` and `loom` at compile
//! time.
//!
//! # Why this indirection exists
//!
//! [`crate::spsc_queue::SpscRing`] makes a claim that ordinary tests cannot
//! check: that its `Acquire`/`Release` pairing is sufficient, so no
//! interleaving of the producer and the consumer can lose an item, duplicate
//! one, reorder two, or let one thread touch a slot the other is using. A
//! stress test spawning two threads exercises whatever interleavings the
//! machine happens to produce — on x86-64, which is strongly ordered, that is
//! a small and unrepresentative subset, and code with a genuinely wrong
//! ordering routinely passes millions of iterations there while failing on
//! AArch64.
//!
//! [loom](https://docs.rs/loom) answers the question properly: it models the
//! C11 memory model and *exhaustively* explores the interleavings and
//! permitted reorderings a small program admits, so a passing run is a proof
//! over that program rather than a sample of it. The cost is that loom has to
//! substitute its own instrumented atomics and cells for the standard ones,
//! which is what this module does.
//!
//! # How to run the model
//!
//! ```text
//! RUSTFLAGS="--cfg loom" cargo test --release --test loom_spsc
//! ```
//!
//! Nothing is instrumented in an ordinary build: without `--cfg loom` this
//! module is a thin re-export of `core::sync::atomic` and `core::cell`, and
//! the wrapper below compiles away.
//!
//! # The `UnsafeCell` wrapper
//!
//! loom's `UnsafeCell` cannot hand out a raw pointer and forget about it —
//! it has to know when an access begins and ends in order to detect a data
//! race, so its API is closure-based (`with`, `with_mut`) rather than
//! `get()`. The non-loom wrapper here presents that same closure API over
//! `core::cell::UnsafeCell` so the queue has one shape of source for both
//! builds. It is `#[inline(always)]` and returns the pointer unchanged, so
//! there is nothing left of it after optimization.

#[cfg(loom)]
pub(crate) use loom::cell::UnsafeCell;
#[cfg(loom)]
pub(crate) use loom::sync::atomic::{AtomicUsize, Ordering};

#[cfg(not(loom))]
pub(crate) use core::sync::atomic::{AtomicUsize, Ordering};

/// `core::cell::UnsafeCell` behind loom's closure-based accessor API.
///
/// See the [module-level documentation](self) for why the queue is written
/// against this shape rather than against `get()`.
#[cfg(not(loom))]
#[derive(Debug)]
pub(crate) struct UnsafeCell<T>(core::cell::UnsafeCell<T>);

#[cfg(not(loom))]
impl<T> UnsafeCell<T> {
    #[inline(always)]
    pub(crate) fn new(value: T) -> Self {
        Self(core::cell::UnsafeCell::new(value))
    }

    /// Run `f` with a shared raw pointer to the contents.
    ///
    /// # Safety
    ///
    /// This is not itself `unsafe` — obtaining the pointer is fine; the
    /// caller's dereference inside `f` is what carries the obligation, and
    /// that dereference is already an `unsafe` block at every call site.
    #[inline(always)]
    pub(crate) fn with<F, R>(&self, f: F) -> R
    where
        F: FnOnce(*const T) -> R,
    {
        f(self.0.get())
    }

    /// Run `f` with a mutable raw pointer to the contents.
    ///
    /// Takes `&self`, not `&mut self`, matching loom: the whole point of an
    /// `UnsafeCell` is interior mutability, and the queue's producer mutates
    /// through a shared reference.
    #[inline(always)]
    pub(crate) fn with_mut<F, R>(&self, f: F) -> R
    where
        F: FnOnce(*mut T) -> R,
    {
        f(self.0.get())
    }
}
