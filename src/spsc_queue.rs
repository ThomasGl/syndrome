//! Single-producer single-consumer lock-free ring buffer.
//!
//! # The ring layout
//!
//! [`SpscRing`] stores `N` slots (`N` a power of two) in a flat, preallocated
//! array and tracks occupancy with two monotonically increasing counters,
//! `head` (next write position) and `tail` (next read position), rather than
//! a separate length/count field:
//!
//! ```text
//!            mask = N - 1                     (N = 8 here)
//!        ┌───┬───┬───┬───┬───┬───┬───┬───┐
//! buffer │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │   physical slot index = idx & mask
//!        └───┴───┴───┴───┴───┴───┴───┴───┘
//!                ▲               ▲
//!              tail            head
//!          (next pop)      (next push)
//!
//! occupied slots = [tail & mask .. head & mask), wrapping through slot N-1 → 0
//! ```
//!
//! `head` and `tail` are never reduced modulo `N` themselves — only the
//! *physical index* (`head & mask` / `tail & mask`) wraps. Both counters grow
//! without bound (wrapping around the full `usize` range via
//! [`usize::wrapping_add`] once in the lifetime of a long-running pipeline,
//! which is harmless since only their *difference* is ever inspected). This
//! is what makes `wrapping_sub` sufficient to distinguish "empty" from
//! "full" without a third counter: because `N` is a power of two and both
//! counters advance by exactly one per operation, `head.wrapping_sub(tail)`
//! always yields the true occupancy count in `0..=N` even after either
//! counter wraps around `usize::MAX` — a plain `head - tail` would panic (in
//! debug builds) or silently underflow-wrap incorrectly the moment `head`
//! wrapped past `tail` while `tail` had not yet wrapped.
//!
//! # Memory ordering: the producer/consumer happens-before edge
//!
//! This queue relies on exactly one cross-thread happens-before
//! relationship, established twice (once per direction of data flow):
//!
//! * **Push** ([`SpscRing::try_push`]): the item is written into `buffer`
//!   *first*, then `head` is bumped with [`Ordering::Release`]. Release
//!   guarantees that the buffer write is visible to any thread that
//!   subsequently *observes* the new `head` value with an
//!   [`Ordering::Acquire`] load — i.e. the consumer can never see an
//!   incremented `head` without also seeing the slot data that write
//!   produced. Without this pairing the CPU or compiler could reorder the
//!   store to `head` ahead of the store to `buffer`, and the consumer could
//!   read uninitialized/stale slot memory.
//! * **Pop** ([`SpscRing::try_pop`]): symmetric in the other direction — the
//!   slot is read out *before* `tail` is bumped with `Release`, so the
//!   producer's next `Acquire` load of `tail` (inside `try_push`'s
//!   full-check) cannot observe the freed slot until the read that freed it
//!   has completed, preventing the producer from overwriting a slot the
//!   consumer is still in the middle of reading.
//!
//! Every load of the *other* thread's counter uses `Acquire` and every store
//! of *this* thread's own counter uses `Release`; loads of one's own counter
//! (there are none on the hot path — each side only ever writes its own
//! counter and reads the other's) would be free to use `Relaxed`, but using
//! `Acquire` uniformly costs nothing extra on the load side and keeps the
//! ordering rules easy to audit.
//!
//! # Why `head` and `tail` are on separate cache lines
//!
//! The producer thread writes `head` on every [`SpscRing::try_push`]; the
//! consumer thread writes `tail` on every [`SpscRing::try_pop`]. If both
//! atomics lived in the same 64-byte cache line (the common case for two
//! adjacent `AtomicUsize` fields — 8 bytes apart, one cache line is 64
//! bytes), every push and every pop would force a cross-core
//! cache-coherency invalidation of a line the *other* thread is also
//! actively touching: the classic **false-sharing** pattern, where two
//! logically independent atomics contend for the same physical cache line
//! purely because of their memory layout. This directly defeats the purpose
//! of splitting the queue into an SPSC design in the first place — the
//! entire point of "single producer / single consumer" is that the two
//! sides *shouldn't* need to fight over shared cache state on every
//! operation.
//!
//! `CachePadded` fixes this: `head` and `tail` are each wrapped in their
//! own `#[repr(align(64))]` newtype. Because Rust guarantees a type's size
//! is always a multiple of its alignment, each `CachePadded` occupies a full,
//! exclusive 64-byte block — regardless of how the compiler orders struct
//! fields, `head` and `tail` can never land in the same cache line. See the
//! `head_and_tail_do_not_share_a_cache_line` test below for a
//! `size_of`/`align_of` + address-distance proof.
//!
//! This queue is fully preallocated at construction time. The hot path uses
//! no heap allocations and relies only on atomic pointer arithmetic.

use core::cell::UnsafeCell;
use core::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A single [`AtomicUsize`] padded out to occupy an entire 64-byte cache
/// line by itself.
///
/// # Why this exists
///
/// Two atomics that are written by *different* threads must never share a
/// cache line, or every write by one thread forces a coherency invalidation
/// (and a stall) on the core touching the other. Wrapping each hot atomic in
/// its own `#[repr(align(64))]` newtype is the standard fix: Rust requires a
/// type's size to be a multiple of its alignment, so this struct's size is
/// forced up to 64 bytes even though the payload is 8 — the remaining 56
/// bytes are pure padding that guarantees isolation. See the
/// [module-level documentation](self) for the full false-sharing
/// explanation and why it matters specifically for `head`/`tail` here.
#[repr(align(64))]
struct CachePadded(AtomicUsize);

impl CachePadded {
    #[inline(always)]
    fn new(value: usize) -> Self {
        Self(AtomicUsize::new(value))
    }

    #[inline(always)]
    fn load(&self, order: Ordering) -> usize {
        self.0.load(order)
    }

    #[inline(always)]
    fn store(&self, value: usize, order: Ordering) {
        self.0.store(value, order);
    }
}

/// Single-producer single-consumer lock-free ring buffer.
///
/// This queue is fully preallocated at construction time. The hot path uses
/// no heap allocations and relies only on atomic pointer arithmetic.
///
/// See the [module-level documentation](self) for the ring-index algorithm,
/// the Acquire/Release memory-ordering contract, and why `head`/`tail` are
/// each padded onto their own cache line.
#[repr(align(64))]
pub struct SpscRing<T: Copy, const N: usize> {
    buffer: UnsafeCell<[MaybeUninit<T>; N]>,
    head: CachePadded,
    tail: CachePadded,
    mask: usize,
}

unsafe impl<T: Copy, const N: usize> Sync for SpscRing<T, N> {}

impl<T: Copy, const N: usize> Default for SpscRing<T, N> {
    /// Equivalent to [`SpscRing::new`]; panics if `N` is not a power of two.
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy, const N: usize> SpscRing<T, N> {
    /// Creates a new ring with capacity `N`. `N` must be a power of two.
    pub fn new() -> Self {
        assert!(N.is_power_of_two(), "capacity must be a power of two");
        let buffer = unsafe { MaybeUninit::<[MaybeUninit<T>; N]>::uninit().assume_init() };
        SpscRing {
            buffer: UnsafeCell::new(buffer),
            head: CachePadded::new(0),
            tail: CachePadded::new(0),
            mask: N - 1,
        }
    }

    /// Returns `true` when the queue currently contains no elements.
    pub fn is_empty(&self) -> bool {
        self.head.load(Ordering::Acquire) == self.tail.load(Ordering::Acquire)
    }

    /// Returns `true` when the queue is full.
    pub fn is_full(&self) -> bool {
        self.head
            .load(Ordering::Acquire)
            .wrapping_sub(self.tail.load(Ordering::Acquire))
            == N
    }

    /// Attempt to push an item into the ring. Returns the value back if the
    /// queue is currently full.
    pub fn try_push(&self, item: T) -> Result<(), T> {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        if head.wrapping_sub(tail) == N {
            return Err(item);
        }

        let idx = head & self.mask;
        unsafe { (*self.buffer.get())[idx].write(item) };
        self.head.store(head.wrapping_add(1), Ordering::Release);
        Ok(())
    }

    /// Attempt to pop an item from the ring. Returns `None` when empty.
    pub fn try_pop(&self) -> Option<T> {
        let tail = self.tail.load(Ordering::Acquire);
        let head = self.head.load(Ordering::Acquire);
        if head == tail {
            return None;
        }

        let idx = tail & self.mask;
        let item = unsafe { (*self.buffer.get())[idx].assume_init_read() };
        self.tail.store(tail.wrapping_add(1), Ordering::Release);
        Some(item)
    }

    /// Push as many of `items` as currently fit, returning how many were
    /// taken.
    ///
    /// This is not a convenience wrapper around [`Self::try_push`]: the point
    /// is that the whole batch costs **one** `Release` store to `head`
    /// instead of one per element. That store is the expensive part of an
    /// SPSC hand-off, because it is what the consumer's `Acquire` load must
    /// observe, so it bounces the cache line holding `head` between the two
    /// cores every time. Moving `k` items with `try_push` bounces it `k`
    /// times; moving them with this bounces it once.
    ///
    /// A short return is normal and not an error: the queue is fixed-capacity
    /// and lock-free, so the only honest thing to report is how much fit.
    /// Callers should treat the return value as the amount consumed and
    /// retain the rest.
    ///
    /// # Arguments
    ///
    /// * `items` - Elements to enqueue, in order.
    ///
    /// # Returns
    ///
    /// The number of leading elements of `items` that were enqueued, from `0`
    /// (queue full) to `items.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use syndrome::spsc_queue::SpscRing;
    ///
    /// let ring: SpscRing<u32, 4> = SpscRing::new();
    /// assert_eq!(ring.push_slice(&[1, 2, 3]), 3);
    /// // Only one slot left, so a 3-element batch is truncated.
    /// assert_eq!(ring.push_slice(&[4, 5, 6]), 1);
    /// assert!(ring.is_full());
    /// ```
    pub fn push_slice(&self, items: &[T]) -> usize {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        let free = N - head.wrapping_sub(tail);
        let count = free.min(items.len());
        if count == 0 {
            return 0;
        }

        // The ring is power-of-two sized, so a batch spans at most two
        // contiguous runs: from `start` to the end of the buffer, then from
        // the beginning.
        let start = head & self.mask;
        let first = count.min(N - start);
        // SAFETY: as the sole producer this thread owns every slot in
        // `[head, head + free)`; `count <= free`, so the writes below stay
        // inside that range and cannot race the consumer, which only reads
        // slots below `tail`. The `Release` store afterwards is what
        // publishes them.
        unsafe {
            let buf = &mut *self.buffer.get();
            for i in 0..first {
                buf[start + i].write(items[i]);
            }
            for i in 0..count - first {
                buf[i].write(items[first + i]);
            }
        }
        self.head.store(head.wrapping_add(count), Ordering::Release);
        count
    }

    /// Pop up to `out.len()` items into `out`, returning how many were moved.
    ///
    /// The mirror of [`Self::push_slice`], and for the same reason: one
    /// `Release` store to `tail` for the whole batch rather than one per
    /// element.
    ///
    /// # Arguments
    ///
    /// * `out` - Destination buffer; its length bounds the batch size.
    ///
    /// # Returns
    ///
    /// The number of elements written to the front of `out`, from `0` (queue
    /// empty) to `out.len()`. Elements past that count are left untouched.
    ///
    /// # Examples
    ///
    /// ```
    /// use syndrome::spsc_queue::SpscRing;
    ///
    /// let ring: SpscRing<u32, 8> = SpscRing::new();
    /// ring.push_slice(&[10, 20, 30]);
    ///
    /// let mut out = [0u32; 4];
    /// assert_eq!(ring.pop_slice(&mut out), 3);
    /// assert_eq!(&out[..3], &[10, 20, 30]);
    /// assert!(ring.is_empty());
    /// ```
    pub fn pop_slice(&self, out: &mut [T]) -> usize {
        let tail = self.tail.load(Ordering::Acquire);
        let head = self.head.load(Ordering::Acquire);
        let available = head.wrapping_sub(tail);
        let count = available.min(out.len());
        if count == 0 {
            return 0;
        }

        let start = tail & self.mask;
        let first = count.min(N - start);
        // SAFETY: `count <= head - tail`, so every slot read below was
        // published by the producer's `Release` store to `head` that this
        // thread's `Acquire` load above observed. As the sole consumer, no
        // other thread reads or reclaims them before the store to `tail`.
        unsafe {
            let buf = &*self.buffer.get();
            for i in 0..first {
                out[i] = buf[start + i].assume_init_read();
            }
            for i in 0..count - first {
                out[first + i] = buf[i].assume_init_read();
            }
        }
        self.tail.store(tail.wrapping_add(count), Ordering::Release);
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    /// A batch must move the same values, in the same order, as the same
    /// number of single pushes and pops — including across the point where
    /// the batch wraps the end of the ring buffer, which is the case a
    /// two-segment copy gets wrong.
    #[test]
    fn batched_transfer_matches_the_per_item_path() {
        const CAP: usize = 8;
        let batched: SpscRing<u32, CAP> = SpscRing::new();
        let single: SpscRing<u32, CAP> = SpscRing::new();

        // Offset both rings so the next write starts mid-buffer, forcing the
        // batch to wrap.
        for r in [&batched, &single] {
            for v in 0..5u32 {
                r.try_push(v).unwrap();
            }
            for _ in 0..5 {
                r.try_pop().unwrap();
            }
        }

        let payload: Vec<u32> = (100..106).collect();
        assert_eq!(batched.push_slice(&payload), payload.len());
        for &v in &payload {
            single.try_push(v).unwrap();
        }

        let mut got = vec![0u32; payload.len()];
        assert_eq!(batched.pop_slice(&mut got), payload.len());
        let expect: Vec<u32> = (0..payload.len())
            .map(|_| single.try_pop().unwrap())
            .collect();
        assert_eq!(got, expect, "batched order differs from per-item order");
        assert_eq!(got, payload);
        assert!(batched.is_empty() && single.is_empty());
    }

    /// A short return is the contract, not a failure: `push_slice` takes only
    /// what fits and `pop_slice` yields only what is there.
    #[test]
    fn batched_transfer_reports_partial_progress() {
        let ring: SpscRing<u8, 4> = SpscRing::new();
        assert_eq!(
            ring.push_slice(&[1, 2, 3, 4, 5, 6]),
            4,
            "must stop at capacity"
        );
        assert!(ring.is_full());
        assert_eq!(ring.push_slice(&[7]), 0, "full queue must take nothing");

        let mut out = [0u8; 10];
        assert_eq!(
            ring.pop_slice(&mut out),
            4,
            "must yield only what is queued"
        );
        assert_eq!(&out[..4], &[1, 2, 3, 4]);
        assert_eq!(out[4], 0, "must not touch the tail of the destination");
        assert_eq!(
            ring.pop_slice(&mut out),
            0,
            "empty queue must yield nothing"
        );
    }

    /// An empty batch is a no-op rather than an index error, and must not
    /// disturb the indices.
    #[test]
    fn batched_transfer_handles_empty_slices() {
        let ring: SpscRing<u16, 4> = SpscRing::new();
        assert_eq!(ring.push_slice(&[]), 0);
        assert!(ring.is_empty());
        ring.try_push(9).unwrap();
        let mut none: [u16; 0] = [];
        assert_eq!(ring.pop_slice(&mut none), 0);
        assert_eq!(ring.try_pop(), Some(9), "an empty pop must not consume");
    }

    /// Cross-thread stress: a producer pushing in batches and a consumer
    /// popping in batches must transfer every value exactly once and in
    /// order. Batching is where an SPSC ring is most likely to go wrong —
    /// publishing `head` before the payload writes, or mis-splitting the
    /// wrap — and neither shows up single-threaded.
    #[test]
    fn batched_transfer_is_lossless_and_ordered_across_threads() {
        const CAP: usize = 64;
        const TOTAL: u32 = 200_000;
        let ring: Arc<SpscRing<u32, CAP>> = Arc::new(SpscRing::new());

        let producer = {
            let ring = Arc::clone(&ring);
            thread::spawn(move || {
                let mut next = 0u32;
                let mut pending: Vec<u32> = Vec::new();
                while next < TOTAL {
                    // Vary the batch size so both wrap halves and the
                    // partial-fit path are exercised.
                    let want = (next as usize % 17) + 1;
                    while pending.len() < want && next < TOTAL {
                        pending.push(next);
                        next += 1;
                    }
                    let moved = ring.push_slice(&pending);
                    pending.drain(..moved);
                }
                while !pending.is_empty() {
                    let moved = ring.push_slice(&pending);
                    pending.drain(..moved);
                }
            })
        };

        let consumer = {
            let ring = Arc::clone(&ring);
            thread::spawn(move || {
                let mut expect = 0u32;
                let mut buf = [0u32; 23]; // deliberately not a divisor of CAP
                while expect < TOTAL {
                    let got = ring.pop_slice(&mut buf);
                    for &v in &buf[..got] {
                        assert_eq!(v, expect, "value out of order or duplicated");
                        expect += 1;
                    }
                }
                expect
            })
        };

        producer.join().expect("producer panicked");
        let seen = consumer.join().expect("consumer panicked");
        assert_eq!(seen, TOTAL, "not every value arrived exactly once");
        assert!(ring.is_empty());
    }

    #[test]
    fn spsc_ring_push_and_pop() {
        let queue = SpscRing::<u32, 8>::new();
        assert!(queue.is_empty());
        queue.try_push(42).unwrap();
        assert!(!queue.is_empty());
        assert_eq!(queue.try_pop(), Some(42));
        assert!(queue.is_empty());
    }

    #[test]
    fn spsc_ring_full_behavior() {
        let queue = SpscRing::<u32, 4>::new();
        assert!(queue.try_push(1).is_ok());
        assert!(queue.try_push(2).is_ok());
        assert!(queue.try_push(3).is_ok());
        assert!(queue.try_push(4).is_ok());
        assert!(queue.try_push(5).is_err());
        assert_eq!(queue.try_pop(), Some(1));
        assert!(queue.try_push(5).is_ok());
        assert_eq!(queue.try_pop(), Some(2));
    }

    /// Proves the Task-1 false-sharing fix actually holds: `head` and `tail`
    /// must never land in the same 64-byte-aligned region of memory.
    ///
    /// This checks both the type-level guarantee (`CachePadded` is exactly
    /// one cache line, alignment 64) and the concrete field layout of a real
    /// `SpscRing` instance (the two fields' addresses are at least 64 bytes
    /// apart and each is itself 64-byte aligned).
    #[test]
    fn head_and_tail_do_not_share_a_cache_line() {
        assert_eq!(core::mem::align_of::<CachePadded>(), 64);
        assert_eq!(core::mem::size_of::<CachePadded>(), 64);

        let ring = SpscRing::<u32, 8>::new();
        let head_addr = &ring.head as *const CachePadded as usize;
        let tail_addr = &ring.tail as *const CachePadded as usize;

        assert_eq!(head_addr % 64, 0, "head is not 64-byte aligned");
        assert_eq!(tail_addr % 64, 0, "tail is not 64-byte aligned");
        assert!(
            head_addr.abs_diff(tail_addr) >= 64,
            "head ({head_addr:#x}) and tail ({tail_addr:#x}) share a 64-byte cache line"
        );
    }

    /// Real concurrent stress test with a deliberately tiny ring capacity so
    /// the producer and consumer contend on "full" and "empty" on nearly
    /// every single operation — this is the adversarial case for a lock-free
    /// SPSC ring, since it maximizes how often each side observes the
    /// other's counter mid-flight.
    ///
    /// One producer thread pushes the strictly increasing sequence
    /// `0..N_SMALL`; one consumer thread pops and checks each value against
    /// the next expected counter. Any lost, duplicated, or reordered item
    /// would show up immediately as a mismatched value or a value count
    /// short of `N_SMALL` — this is what actually exercises the
    /// Acquire/Release pairing documented at the module level, rather than
    /// merely asserting on it in prose.
    #[test]
    fn spsc_ring_concurrent_stress_small_capacity() {
        const N_SMALL: u64 = 200_000;
        let ring: Arc<SpscRing<u64, 4>> = Arc::new(SpscRing::new());

        let producer_ring = Arc::clone(&ring);
        let producer = thread::spawn(move || {
            for v in 0..N_SMALL {
                while producer_ring.try_push(v).is_err() {
                    std::hint::spin_loop();
                }
            }
        });

        let consumer_ring = Arc::clone(&ring);
        let consumer = thread::spawn(move || {
            let mut expected = 0u64;
            while expected < N_SMALL {
                match consumer_ring.try_pop() {
                    Some(v) => {
                        assert_eq!(
                            v, expected,
                            "SPSC ring delivered an out-of-order/corrupted value"
                        );
                        expected += 1;
                    }
                    None => std::hint::spin_loop(),
                }
            }
            expected
        });

        producer.join().expect("producer thread panicked");
        let received = consumer.join().expect("consumer thread panicked");
        assert_eq!(
            received, N_SMALL,
            "consumer did not receive every pushed value exactly once"
        );
    }

    /// Same concurrent proof as
    /// [`spsc_ring_concurrent_stress_small_capacity`] but with a much larger
    /// ring (so full/empty contention is rare) and a larger total volume —
    /// this covers the complementary regime where most pushes/pops succeed
    /// on the first try and wraparound of the physical index (`head & mask`)
    /// happens thousands of times over the run.
    #[test]
    fn spsc_ring_concurrent_stress_large_capacity() {
        const N_LARGE: u64 = 1_000_000;
        let ring: Arc<SpscRing<u64, 4096>> = Arc::new(SpscRing::new());

        let producer_ring = Arc::clone(&ring);
        let producer = thread::spawn(move || {
            for v in 0..N_LARGE {
                while producer_ring.try_push(v).is_err() {
                    std::hint::spin_loop();
                }
            }
        });

        let consumer_ring = Arc::clone(&ring);
        let consumer = thread::spawn(move || {
            let mut expected = 0u64;
            while expected < N_LARGE {
                match consumer_ring.try_pop() {
                    Some(v) => {
                        assert_eq!(
                            v, expected,
                            "SPSC ring delivered an out-of-order/corrupted value"
                        );
                        expected += 1;
                    }
                    None => std::hint::spin_loop(),
                }
            }
            expected
        });

        producer.join().expect("producer thread panicked");
        let received = consumer.join().expect("consumer thread panicked");
        assert_eq!(
            received, N_LARGE,
            "consumer did not receive every pushed value exactly once"
        );
    }
}
