//! Exhaustive model check of the SPSC ring's memory ordering.
//!
//! This file only exists under `--cfg loom`:
//!
//! ```text
//! RUSTFLAGS="--cfg loom" cargo test --release --test loom_spsc
//! ```
//!
//! # What a stress test cannot tell you
//!
//! `src/spsc_queue.rs` already has two-thread stress tests moving hundreds of
//! thousands of values and checking that none are lost, duplicated or
//! reordered. They are worth having, and they cannot establish what the
//! module's documentation claims. A stress test samples whichever
//! interleavings the machine happens to produce, and this machine is x86-64:
//! a strongly-ordered architecture where the hardware will not reorder a
//! store past a store at all. Downgrading `Ordering::Release` to
//! `Ordering::Relaxed` in [`SpscRing::try_push`] leaves those stress tests
//! passing indefinitely here, while introducing a real bug that shows up on
//! AArch64 — which is exactly the architecture the crate's other CI job runs
//! on, and exactly the class of bug that is impossible to reproduce once
//! reported.
//!
//! # What loom does instead
//!
//! [loom](https://docs.rs/loom) executes the test body once per distinct
//! interleaving, over a model of the C11 memory ordering rules rather than of
//! any one CPU — so a load is allowed to observe every value the model
//! permits, not merely the ones this processor would produce. For a program
//! small enough to enumerate, a passing run is a proof over all executions
//! rather than a sample of them. `loom::cell::UnsafeCell` additionally tracks
//! every access, so a slot touched by both threads without an intervening
//! happens-before edge is reported as a data race even when the values
//! involved happen to come out right.
//!
//! Two consequences shape everything below. Programs must be **tiny** — the
//! interleaving count grows combinatorially, so these models use a
//! two-element ring and two or three operations per thread, which is enough
//! to exercise the wrap and the full/empty boundaries but nowhere near enough
//! to be a functional test. And loom cannot check anything it does not
//! execute, which is why there is a separate model per entry point rather
//! than one model of "the queue".
//!
//! # These models have teeth
//!
//! A model that passes against a broken queue proves nothing, so each of
//! these was checked against a deliberately weakened `SpscRing` before being
//! committed. All eight defects below are caught:
//!
//! | Injected defect | |
//! |---|---|
//! | `try_push`'s `Release` store to `head` weakened to `Relaxed` | caught |
//! | `try_pop`'s `Release` store to `tail` weakened to `Relaxed` | caught |
//! | `push_slice`'s `Release` store to `head` weakened to `Relaxed` | caught |
//! | `pop_slice`'s `Release` store to `tail` weakened to `Relaxed` | caught |
//! | `try_pop`'s `Acquire` load of `head` weakened to `Relaxed` | caught |
//! | `try_push`'s full check off by one (`== N + 1`) | caught |
//! | `push_slice`'s free-slot count off by one | caught |
//! | `pop_slice`'s available count off by one | caught |
//!
//! Two of those needed a model that did not exist in the first draft — see
//! [`batched_producer_cannot_overrun_the_consumer`], which was written
//! because the controls said the batched coverage had a hole. Re-run the
//! controls whenever one of these models is edited; passing them is the
//! property that makes this file worth its runtime.

#![cfg(loom)]

use loom::sync::Arc;
use loom::thread;
use syndrome::spsc_queue::SpscRing;

/// Ring capacity for every model here.
///
/// Two slots is the smallest capacity that still has an interior: a producer
/// can fill the ring, wrap, and be blocked by a consumer that has not caught
/// up, which is where the `head - tail == N` full check and the `head & mask`
/// wrap both matter. Raising it to 4 multiplies the interleaving count
/// without reaching any state 2 does not.
const CAP: usize = 2;

/// One producer pushing two values, one consumer draining until it has both.
///
/// The assertion is on *content and order*: the consumer must receive
/// `[1, 2]`, never `[2, 1]`, never a repeat, never a value it was not sent.
/// Because the ring holds two items and the producer sends two, some
/// interleavings fill it completely and some drain it completely, so the
/// full and empty boundaries are both reached.
///
/// This is also the model that catches a missing `Release` on `head`: with a
/// relaxed store, loom can schedule the consumer to observe the incremented
/// `head` before the slot write is visible, and `loom::cell::UnsafeCell`
/// reports the resulting concurrent access rather than waiting for the value to
/// come out wrong.
#[test]
fn single_item_handoff_has_no_race() {
    loom::model(|| {
        let ring: Arc<SpscRing<u32, CAP>> = Arc::new(SpscRing::new());
        let producer = ring.clone();

        let handle = thread::spawn(move || {
            for value in 1..=2u32 {
                // A full queue is a legitimate outcome, not a failure: spin
                // until the consumer makes room. loom explores the schedules
                // where this spins and the ones where it does not.
                while producer.try_push(value).is_err() {
                    loom::thread::yield_now();
                }
            }
        });

        let mut received = Vec::new();
        while received.len() < 2 {
            match ring.try_pop() {
                Some(v) => received.push(v),
                None => loom::thread::yield_now(),
            }
        }

        handle.join().unwrap();
        assert_eq!(received, vec![1, 2], "items lost, duplicated or reordered");
        assert!(ring.is_empty(), "queue not drained");
    });
}

/// The same hand-off through the batched entry points.
///
/// `push_slice` and `pop_slice` are not wrappers around the single-item
/// operations: they publish a whole batch with one `Release` store, and they
/// compute how much fits from a snapshot of the *other* thread's counter.
/// That arithmetic is where an off-by-one would let the producer write into a
/// slot the consumer has not finished reading — a bug the single-item model
/// cannot see, because it never writes more than one slot per store.
///
/// On its own this model is *not* sufficient: sending exactly `CAP` items
/// never makes the producer wait for space, so the batch never has to be
/// shortened and the consumer's `tail` never has to be observed. Two real
/// defects survive it. [`batched_producer_cannot_overrun_the_consumer`]
/// covers that case; this one covers the exact-fit path.
#[test]
fn batched_handoff_has_no_race() {
    loom::model(|| {
        let ring: Arc<SpscRing<u32, CAP>> = Arc::new(SpscRing::new());
        let producer = ring.clone();

        let handle = thread::spawn(move || {
            let batch = [1u32, 2];
            let mut sent = 0;
            while sent < batch.len() {
                let n = producer.push_slice(&batch[sent..]);
                if n == 0 {
                    loom::thread::yield_now();
                }
                sent += n;
            }
        });

        let mut received = Vec::new();
        let mut out = [0u32; CAP];
        while received.len() < 2 {
            let n = ring.pop_slice(&mut out);
            if n == 0 {
                loom::thread::yield_now();
            }
            received.extend_from_slice(&out[..n]);
        }

        handle.join().unwrap();
        assert_eq!(received, vec![1, 2], "items lost, duplicated or reordered");
        assert!(ring.is_empty(), "queue not drained");
    });
}

/// A batched producer against a single-item consumer.
///
/// The two sides of this queue are independent, and nothing requires a caller
/// to use matching entry points. Mixing them puts a multi-slot `Release`
/// store against a single-slot `Acquire` load, which is the pairing neither
/// of the models above exercises: `push_slice` publishes two slots at once
/// while `try_pop` claims them one at a time, so the consumer necessarily
/// runs with a `head` that is two ahead of its `tail`.
#[test]
fn batched_producer_against_single_item_consumer_has_no_race() {
    loom::model(|| {
        let ring: Arc<SpscRing<u32, CAP>> = Arc::new(SpscRing::new());
        let producer = ring.clone();

        let handle = thread::spawn(move || {
            let batch = [7u32, 9];
            let mut sent = 0;
            while sent < batch.len() {
                let n = producer.push_slice(&batch[sent..]);
                if n == 0 {
                    loom::thread::yield_now();
                }
                sent += n;
            }
        });

        let mut received = Vec::new();
        while received.len() < 2 {
            match ring.try_pop() {
                Some(v) => received.push(v),
                None => loom::thread::yield_now(),
            }
        }

        handle.join().unwrap();
        assert_eq!(received, vec![7, 9], "items lost, duplicated or reordered");
    });
}

/// The producer must never overrun the consumer, even when it sends more
/// items than the ring can hold at once.
///
/// Three values through a two-slot ring guarantees that at least one push
/// finds the queue full and has to wait for a pop, so the model covers the
/// wrap (`head & mask` returning to 0) and the `head - tail == N` rejection
/// on the same execution. Those are the two places the counters' unbounded
/// growth has to behave, and they are only reachable when the producer gets
/// ahead.
#[test]
fn producer_cannot_overrun_the_consumer() {
    loom::model(|| {
        let ring: Arc<SpscRing<u32, CAP>> = Arc::new(SpscRing::new());
        let producer = ring.clone();

        let handle = thread::spawn(move || {
            for value in 1..=3u32 {
                while producer.try_push(value).is_err() {
                    loom::thread::yield_now();
                }
            }
        });

        let mut received = Vec::new();
        while received.len() < 3 {
            match ring.try_pop() {
                Some(v) => received.push(v),
                None => loom::thread::yield_now(),
            }
        }

        handle.join().unwrap();
        assert_eq!(
            received,
            vec![1, 2, 3],
            "items lost, duplicated or reordered"
        );
    });
}

/// The batched twin of [`producer_cannot_overrun_the_consumer`]: three values
/// through a two-slot ring, both sides using the slice entry points.
///
/// This model exists because the negative controls said it had to. With the
/// batched producer sending exactly `CAP` items,
/// [`batched_handoff_has_no_race`] never makes the producer wait for space —
/// so it never depends on observing the consumer's `tail`, and two real
/// defects survived it: relaxing the `Release` on `tail` in `pop_slice`, and
/// an off-by-one in `push_slice`'s free-slot count (masked there because
/// `items.len()` bounded the batch anyway). Offering more than fits is what
/// forces `push_slice` to compute a *short* count from the consumer's
/// counter and to be right about it. Both defects fail here.
#[test]
fn batched_producer_cannot_overrun_the_consumer() {
    loom::model(|| {
        let ring: Arc<SpscRing<u32, CAP>> = Arc::new(SpscRing::new());
        let producer = ring.clone();

        let handle = thread::spawn(move || {
            let batch = [1u32, 2, 3];
            let mut sent = 0;
            while sent < batch.len() {
                let n = producer.push_slice(&batch[sent..]);
                if n == 0 {
                    loom::thread::yield_now();
                }
                sent += n;
            }
        });

        let mut received = Vec::new();
        let mut out = [0u32; CAP];
        while received.len() < 3 {
            let n = ring.pop_slice(&mut out);
            if n == 0 {
                loom::thread::yield_now();
            }
            received.extend_from_slice(&out[..n]);
        }

        handle.join().unwrap();
        assert_eq!(
            received,
            vec![1, 2, 3],
            "items lost, duplicated or reordered"
        );
    });
}
