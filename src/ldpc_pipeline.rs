//! Lock-free pipelined LDPC decoder backed by SPSC rings.
//!
//! [`LdpcPipeline`] wraps a [`QcLdpcDecoder`] so that frame submission and
//! result collection are fully decoupled. One or more background worker threads
//! run the LOMS kernel; coordination between the caller and the workers uses
//! only [`SpscRing`] SPSC queues — no OS mutexes are touched on the hot path.
//!
//! # Multi-worker architecture
//!
//! Each worker owns exactly one dedicated pair of rings — a *work ring*
//! (caller → worker) and a *done ring* (worker → caller) — rather than all
//! workers sharing one queue. An [`SpscRing`] is single-producer
//! single-consumer by construction, so an MPSC/SPMC shared queue is simply
//! not an option without adding a different (heavier) synchronization
//! primitive; N independent SPSC pairs sidesteps that entirely, at the cost
//! of the caller having to fan work out and fan results back in itself:
//!
//! ```text
//!                    worker 0 ── work_ring[0] ──►│decode│── done_ring[0] ──┐
//!                   /                                                      \
//! submit() ── round robin                                                  ├──► try_recv()
//!                   \                                                      /   (polls each
//!                    worker 1 ── work_ring[1] ──►│decode│── done_ring[1] ──┘    done ring
//!                                                                                in order)
//! ```
//!
//! [`LdpcPipeline::submit`] hands a frame's pool slot index to the next
//! worker in round-robin order; [`LdpcPipeline::try_recv`] polls every done
//! ring in turn and returns the first completed frame it finds. Because
//! each worker gets its own decoder clone (see [`QcLdpcDecoder`]'s `Clone`),
//! there is no shared mutable decode state between workers either — only
//! the ring indices themselves are shared, and those are lock-free atomics
//! (see [`crate::spsc_queue`]).
//!
//! # One frame's path through the pipeline (single worker shown)
//!
//! ```text
//! acquire() ──► fill llr_mut() ──► submit()
//!                                      │ [work ring, round-robin]
//!                                      ▼
//!                               [worker decodes]
//!                                      │ [done ring]
//!                                      ▼
//!                           try_recv() ──► read hard() ──► release()
//! ```
//!
//! All 16 frame slots (the `FrameSlot` pool) are pre-allocated at
//! construction time and never reallocated. The ring protocol guarantees
//! that exactly one entity (the caller, or exactly one worker) holds each
//! slot index at any moment: a slot index only ever exists in one ring (or
//! the caller's `free_list`) at a time, so there is no concurrent access to
//! the `FrameSlot` it points to — this is what makes the raw `*mut
//! FrameSlot` pointers in [`LdpcFrame`] sound to hand across threads without
//! any lock.
//!
//! # Worker count: hardware-aware by default
//!
//! [`LdpcPipeline::new`] sizes the worker pool from
//! [`std::thread::available_parallelism`] (falling back to 1 if the OS
//! cannot report it), clamped to `1..=POOL_SIZE/2` (8) — the same ceiling
//! [`LdpcPipeline::with_workers`] has always enforced, since more workers
//! than that would starve for slots against only 16 preallocated frames.
//! Use [`LdpcPipeline::with_workers`] directly to pick an explicit worker
//! count instead (e.g. exactly 1, for deterministic tests).
//!
//! # Example
//!
//! ```no_run
//! use syndrome::{BaseGraph, QcLdpcDecoder, LdpcPipeline};
//!
//! let decoder  = QcLdpcDecoder::with_lifting_size(BaseGraph::Bg1, 384, 0.25).unwrap();
//! let mut pipe = LdpcPipeline::new(decoder, 10);
//!
//! // Acquire a free slot, fill with channel LLRs, and submit.
//! let mut frame = pipe.acquire().expect("pool has 16 slots");
//! frame.llr_mut().iter_mut().for_each(|v| *v = 5.0);
//! pipe.submit(frame).ok();
//!
//! // Spin until the worker completes the frame.
//! loop {
//!     if let Some(result) = pipe.try_recv() {
//!         let _hard_bits = result.hard();
//!         pipe.release(result);
//!         break;
//!     }
//!     std::hint::spin_loop();
//! }
//! ```

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};

use crate::qc_ldpc::QcLdpcDecoder;
use crate::spsc_queue::SpscRing;

// Power-of-two pool; also used as the SPSC ring capacity.
const POOL_SIZE: usize = 16;

/// Per-frame working memory stored in the pre-allocated pool.
struct FrameSlot {
    llr: Vec<f32>,
    edge_r: Vec<f32>,
    scratch: Vec<f32>,
    hard: Vec<u8>,
    /// Number of LOMS iterations the worker actually consumed decoding this
    /// slot's most recent frame (`<=` the pipeline's configured iteration
    /// budget; smaller when the syndrome check passed early). `0` until the
    /// first decode completes.
    iterations_used: usize,
    /// Caller-owned identifier, carried through the pipeline untouched. See
    /// [`LdpcFrame::set_tag`].
    tag: usize,
}

/// A borrowed decode slot.  Obtained via [`LdpcPipeline::acquire`].
///
/// Fill [`LdpcFrame::llr_mut`] with channel LLR values, then hand ownership to the
/// pipeline with [`LdpcPipeline::submit`].  After receiving a completed
/// frame via [`LdpcPipeline::try_recv`], read results from [`LdpcFrame::hard`], then
/// return the slot with [`LdpcPipeline::release`].
///
/// # Ownership invariant
///
/// Each `LdpcFrame` must be consumed by either `submit` **or** `release`
/// before the pipeline is dropped.  Dropping an `LdpcFrame` without
/// returning it leaks the pool slot (it will never be reused).
pub struct LdpcFrame {
    slot_idx: usize,
    ptr: *mut FrameSlot,
}

// SAFETY: `LdpcFrame` is transferred to the worker thread only via the work
// ring, which enforces single-owner semantics. The raw pointer is valid for
// the lifetime of the pool Vec (which outlives the worker threads — see Drop).
unsafe impl Send for LdpcFrame {}

impl LdpcFrame {
    /// Mutable view of the LLR input buffer. Write channel LLRs here before
    /// calling [`LdpcPipeline::submit`].
    pub fn llr_mut(&mut self) -> &mut [f32] {
        // SAFETY: exclusive access is guaranteed by the pool protocol.
        unsafe { &mut (*self.ptr).llr }
    }

    /// Read the hard-decision decode output. Valid only after receiving via
    /// [`LdpcPipeline::try_recv`].
    pub fn hard(&self) -> &[u8] {
        // SAFETY: exclusive access is guaranteed by the pool protocol.
        unsafe { &(*self.ptr).hard }
    }

    /// Number of LOMS iterations the worker actually consumed decoding this
    /// frame. Valid only after receiving via [`LdpcPipeline::try_recv`].
    ///
    /// This is the real per-frame iteration count returned by
    /// [`QcLdpcDecoder::decode_layered_offset_min_sum`] — smaller than the
    /// iteration budget passed to [`LdpcPipeline::new`]/[`LdpcPipeline::with_workers`]
    /// whenever the syndrome check is satisfied before the budget is
    /// exhausted. Callers computing decode throughput (e.g.
    /// `src/bin/ldpc_pipeline_bench.rs`) should sum/average this value
    /// across frames rather than assuming every frame consumed the full
    /// configured iteration count.
    pub fn iterations_used(&self) -> usize {
        // SAFETY: exclusive access is guaranteed by the pool protocol.
        unsafe { (*self.ptr).iterations_used }
    }

    /// Attach a caller-chosen identifier to this frame, carried through the
    /// pipeline and readable again with [`LdpcFrame::tag`].
    ///
    /// [`LdpcPipeline::try_recv`] returns frames in *completion* order, which
    /// is not submission order once there is more than one worker — a short
    /// frame submitted second can finish first. Any caller that has to put
    /// results back in order therefore needs to know which input a completed
    /// frame came from, and the pool slot index is deliberately not exposed
    /// for that purpose: it is an allocation detail that gets reused, so it
    /// identifies a slot rather than a piece of work.
    ///
    /// [`crate::transport_block::DlSchDecoder`] uses this to carry the code
    /// block index across the pipeline, which is what lets it reassemble the
    /// transport block in order from out-of-order completions.
    ///
    /// # Arguments
    ///
    /// * `tag` - Any value meaningful to the caller. The pipeline never reads
    ///   it.
    pub fn set_tag(&mut self, tag: usize) {
        // SAFETY: exclusive access is guaranteed by the pool protocol.
        unsafe { (*self.ptr).tag = tag };
    }

    /// The identifier set by [`LdpcFrame::set_tag`] before submission, or `0`
    /// if none was set.
    pub fn tag(&self) -> usize {
        // SAFETY: exclusive access is guaranteed by the pool protocol.
        unsafe { (*self.ptr).tag }
    }
}

/// Lock-free pipelined LDPC decoder with optional multi-worker parallelism.
///
/// See [module-level documentation](self) for the frame lifecycle.
pub struct LdpcPipeline {
    /// Pre-allocated slot pool; never reallocated after construction. Each slot
    /// is individually boxed so its address is stable even if the outer `Vec`
    /// were ever resized — the workers hold raw `*mut FrameSlot` into these
    /// allocations, so per-element address stability is a safety requirement
    /// (hence `clippy::vec_box` is intentional here, not a missed simplification).
    #[allow(clippy::vec_box)]
    pool: Vec<Box<FrameSlot>>,
    /// Slot indices available to the caller (main-thread-only, no concurrency).
    free_list: Vec<usize>,
    /// Per-worker work rings: caller → worker[i].
    work_rings: Vec<Arc<SpscRing<usize, POOL_SIZE>>>,
    /// Per-worker done rings: worker[i] → caller.
    done_rings: Vec<Arc<SpscRing<usize, POOL_SIZE>>>,
    n_workers: usize,
    /// Round-robin counter for submit dispatch.
    next_submit: usize,
    stop_flag: Arc<AtomicBool>,
    workers: Vec<JoinHandle<()>>,
}

impl LdpcPipeline {
    /// Create a pipeline sized to the host's available hardware parallelism.
    ///
    /// Earlier versions of this constructor hardcoded exactly one worker
    /// thread regardless of the host, which left every additional core idle
    /// by default — a caller had to already know about
    /// [`LdpcPipeline::with_workers`] to get any parallelism at all. `new`
    /// now queries [`std::thread::available_parallelism`] and sizes the
    /// worker pool to match, falling back to `1` only if the OS cannot
    /// report a core count (e.g. some sandboxes/containers).
    ///
    /// The detected count is passed straight to
    /// [`LdpcPipeline::with_workers`], which clamps it to `1..=POOL_SIZE/2`
    /// (8) — that ceiling is unchanged and still deliberate: with only 16
    /// preallocated frame slots, more than 8 concurrent workers would leave
    /// each worker less than one slot of headroom on average, causing
    /// [`LdpcPipeline::acquire`] to stall constantly instead of gaining
    /// throughput.
    ///
    /// Call [`LdpcPipeline::with_workers`] directly whenever you need to
    /// override the auto-detected count explicitly — for example, pinning
    /// to exactly 1 worker for deterministic single-threaded tests, or
    /// deliberately under-subscribing a shared host.
    ///
    /// # Arguments
    ///
    /// * `decoder`    - Fully constructed decoder. The pipeline takes ownership.
    /// * `iterations` - LOMS iteration count applied to every frame.
    pub fn new(decoder: QcLdpcDecoder, iterations: usize) -> Self {
        let n_workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        Self::with_workers(decoder, iterations, n_workers)
    }

    /// Create a pipeline with `n_workers` background decode threads.
    ///
    /// Each worker gets its own SPSC work ring and done ring. The main thread
    /// distributes submitted frames across workers in round-robin order and
    /// collects completed frames from all done rings in `try_recv`.
    ///
    /// # Arguments
    ///
    /// * `decoder`    - Decoder template; cloned once per additional worker.
    /// * `iterations` - LOMS iteration count per frame.
    /// * `n_workers`  - Number of parallel worker threads (clamped to 1..=8).
    pub fn with_workers(decoder: QcLdpcDecoder, iterations: usize, n_workers: usize) -> Self {
        let n_workers = n_workers.clamp(1, POOL_SIZE / 2);
        let n = decoder.variable_node_count();
        let n_edge = decoder.required_edge_buffer();
        let n_scr = decoder.required_layer_buffer();

        // Allocate all frame slots up front.
        let mut pool: Vec<Box<FrameSlot>> = (0..POOL_SIZE)
            .map(|_| {
                Box::new(FrameSlot {
                    llr: vec![0.0f32; n],
                    edge_r: vec![0.0f32; n_edge],
                    scratch: vec![0.0f32; n_scr],
                    hard: vec![0u8; n],
                    iterations_used: 0,
                    tag: 0,
                })
            })
            .collect();

        // Collect raw pointers (as usize for Send compatibility).
        // SAFETY: `pool` is never reallocated after this point, so the
        // pointers remain valid until `pool` is dropped after all workers join.
        let pool_ptrs: Vec<usize> = pool
            .iter_mut()
            .map(|b| b.as_mut() as *mut FrameSlot as usize)
            .collect();

        let stop_flag = Arc::new(AtomicBool::new(false));
        let mut work_rings = Vec::with_capacity(n_workers);
        let mut done_rings = Vec::with_capacity(n_workers);
        let mut workers = Vec::with_capacity(n_workers);

        for wi in 0..n_workers {
            let work_ring = Arc::new(SpscRing::<usize, POOL_SIZE>::new());
            let done_ring = Arc::new(SpscRing::<usize, POOL_SIZE>::new());

            let work_rx = Arc::clone(&work_ring);
            let done_tx = Arc::clone(&done_ring);
            let stop = Arc::clone(&stop_flag);
            let ptrs = pool_ptrs.clone();
            // Each worker gets its own decoder clone so there is no data sharing
            // on the (mutable) decode scratch.
            let dec: QcLdpcDecoder = decoder.clone();

            let handle = thread::Builder::new()
                .name(format!("ldpc-worker-{wi}"))
                .spawn(move || {
                    // Best-effort affinity: binds this worker to core `wi` when
                    // the `affinity` feature and OS support are available.
                    // The Err path is silently ignored — correctness is unaffected;
                    // affinity only reduces cross-core cache misses.
                    let _ = crate::affinity::pin_to_core(wi);
                    loop {
                        if let Some(idx) = work_rx.try_pop() {
                            // SAFETY: `idx` is in [0, POOL_SIZE). The pool protocol
                            // ensures we are the only thread touching this slot.
                            let slot = unsafe { &mut *(ptrs[idx] as *mut FrameSlot) };

                            // decode_layered_offset_min_sum zeroes edge_r itself at
                            // the top of every call (see qc_ldpc.rs), so doing it
                            // here too was a redundant ~474 KiB memset per frame at
                            // BG1 Z=384 — caught while investigating why the
                            // pipeline measured slower than a plain decode loop on
                            // the same workload.
                            slot.iterations_used = dec
                                .decode_layered_offset_min_sum(
                                    &mut slot.llr,
                                    &mut slot.edge_r,
                                    &mut slot.scratch,
                                    &mut slot.hard,
                                    iterations,
                                )
                                .unwrap_or(0);

                            // Spin if the done ring is momentarily full (caller lagging).
                            while done_tx.try_push(idx).is_err() {
                                std::hint::spin_loop();
                            }
                        } else if stop.load(Ordering::Acquire) {
                            break;
                        } else {
                            std::hint::spin_loop();
                        }
                    }
                })
                .expect("failed to spawn ldpc-worker thread");

            work_rings.push(work_ring);
            done_rings.push(done_ring);
            workers.push(handle);
        }

        let free_list: Vec<usize> = (0..POOL_SIZE).collect();

        Self {
            pool,
            free_list,
            work_rings,
            done_rings,
            n_workers,
            next_submit: 0,
            stop_flag,
            workers,
        }
    }

    /// Returns the number of background worker threads actually running.
    ///
    /// Reflects whatever [`LdpcPipeline::new`] auto-detected (clamped to
    /// `1..=POOL_SIZE/2`) or the explicit value passed to
    /// [`LdpcPipeline::with_workers`] (after the same clamp).
    pub fn worker_count(&self) -> usize {
        self.n_workers
    }

    /// Borrow a free decode slot. Returns `None` when all 16 slots are in
    /// flight (caller is submitting faster than the workers can decode).
    pub fn acquire(&mut self) -> Option<LdpcFrame> {
        let idx = self.free_list.pop()?;
        // SAFETY: pool is never reallocated; idx is in [0, POOL_SIZE).
        let ptr = self.pool[idx].as_mut() as *mut FrameSlot;
        Some(LdpcFrame { slot_idx: idx, ptr })
    }

    /// Submit a frame for decoding, dispatching to the next worker in
    /// round-robin order.
    ///
    /// On success the frame is consumed and its slot belongs to the worker
    /// until it comes back through [`LdpcPipeline::try_recv`]. On failure —
    /// the chosen worker's ring is full — the frame is handed **back** to the
    /// caller in the `Err`, who can retry or [`LdpcPipeline::release`] it.
    ///
    /// Returning the frame rather than a bare `false` is what makes the
    /// failure path leak-free. `LdpcFrame` has no `Drop`, so a frame consumed
    /// by a failed submit would simply vanish and its pool slot would never
    /// return to the free list — the pipeline would quietly lose one of its
    /// sixteen slots per occurrence and eventually stall with no error
    /// anywhere. With the pool and each ring both sized to the same 16 slots, the
    /// failure is currently unreachable by construction (at most `POOL_SIZE`
    /// frames can be in flight at once, so no single ring can overflow), but
    /// a signature that makes the invariant load-bearing is a poor place to
    /// rely on it.
    ///
    /// # Arguments
    ///
    /// * `frame` - The filled frame to decode.
    ///
    /// # Errors
    ///
    /// Returns the frame unchanged if the target worker's ring is full.
    pub fn submit(&mut self, frame: LdpcFrame) -> Result<(), LdpcFrame> {
        let idx = frame.slot_idx;
        let wi = self.next_submit % self.n_workers;
        if self.work_rings[wi].try_push(idx).is_err() {
            return Err(frame);
        }
        // Only advance the round-robin cursor on a successful dispatch, so a
        // full ring does not silently skip a worker's turn.
        self.next_submit = self.next_submit.wrapping_add(1);
        Ok(())
    }

    /// Poll for a completed frame without blocking. Checks all done rings in
    /// order and returns the first available completed frame.
    pub fn try_recv(&self) -> Option<LdpcFrame> {
        for done in &self.done_rings {
            if let Some(idx) = done.try_pop() {
                // SAFETY: pool is never reallocated; idx came from the done ring.
                let ptr = self.pool[idx].as_ref() as *const FrameSlot as *mut FrameSlot;
                return Some(LdpcFrame { slot_idx: idx, ptr });
            }
        }
        None
    }

    /// Return a completed (or abandoned) frame to the free list.
    ///
    /// The frame is consumed: the caller must not use it after this call.
    pub fn release(&mut self, frame: LdpcFrame) {
        // Consumes the frame; `LdpcFrame` has no `Drop`, so dropping it is a no-op.
        let idx = frame.slot_idx;
        self.free_list.push(idx);
    }
}

impl Drop for LdpcPipeline {
    fn drop(&mut self) {
        // Signal all workers to exit after draining any in-flight frames.
        self.stop_flag.store(true, Ordering::Release);
        for handle in self.workers.drain(..) {
            // Joining ensures each worker has released all pool_ptrs before
            // the pool Vec is dropped below.
            handle.join().ok();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qc_ldpc::{BaseGraph, QcLdpcDecoder};

    #[test]
    fn pipeline_roundtrip_single_frame() {
        let decoder = QcLdpcDecoder::new(BaseGraph::Bg1, 0.25);
        let n = decoder.variable_node_count();
        let mut pipe = LdpcPipeline::new(decoder, 5);

        let mut frame = pipe.acquire().expect("pool should have free slots");
        // All-positive LLRs → hard decisions should all be 0.
        frame.llr_mut().iter_mut().for_each(|v| *v = 5.0);
        assert!(pipe.submit(frame).is_ok());

        let result = loop {
            if let Some(r) = pipe.try_recv() {
                break r;
            }
            std::hint::spin_loop();
        };

        assert_eq!(result.hard().len(), n);
        assert!(result.hard().iter().all(|&b| b == 0));
        pipe.release(result);
    }

    #[test]
    fn pipeline_all_slots_cycle() {
        let decoder = QcLdpcDecoder::new(BaseGraph::Bg1, 0.25);
        let mut pipe = LdpcPipeline::new(decoder, 3);

        // Submit all 16 slots, then drain all 16 results.
        for _ in 0..POOL_SIZE {
            let mut frame = pipe.acquire().expect("enough slots");
            frame.llr_mut().iter_mut().for_each(|v| *v = 5.0);
            assert!(pipe.submit(frame).is_ok());
        }

        let mut received = 0usize;
        while received < POOL_SIZE {
            if let Some(result) = pipe.try_recv() {
                pipe.release(result);
                received += 1;
            } else {
                std::hint::spin_loop();
            }
        }
        assert_eq!(received, POOL_SIZE);
    }

    #[test]
    fn pipeline_multi_worker_throughput() {
        let decoder = QcLdpcDecoder::new(BaseGraph::Bg1, 0.25);
        let n = decoder.variable_node_count();
        let mut pipe = LdpcPipeline::with_workers(decoder, 3, 4);

        const FRAMES: usize = 64;
        let mut submitted = 0usize;
        let mut received = 0usize;

        // Submit all frames, draining periodically to avoid exhausting the pool.
        while submitted < FRAMES || received < FRAMES {
            if submitted < FRAMES
                && let Some(mut frame) = pipe.acquire()
            {
                frame.llr_mut().iter_mut().for_each(|v| *v = 5.0);
                assert!(pipe.submit(frame).is_ok());
                submitted += 1;
            }
            if let Some(result) = pipe.try_recv() {
                assert_eq!(result.hard().len(), n);
                assert!(result.hard().iter().all(|&b| b == 0));
                pipe.release(result);
                received += 1;
            } else if submitted == FRAMES && received < FRAMES {
                std::hint::spin_loop();
            }
        }
        assert_eq!(received, FRAMES);
    }

    #[test]
    fn new_uses_available_hardware_parallelism() {
        let decoder = QcLdpcDecoder::new(BaseGraph::Bg1, 0.25);
        let pipe = LdpcPipeline::new(decoder, 3);

        // Compute the expected worker count exactly the way `new` does, so
        // this assertion is correct on any host instead of hardcoding a
        // number that would be wrong (and flaky) on CI runners with a
        // different core count.
        let detected = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let expected = detected.clamp(1, POOL_SIZE / 2);
        assert_eq!(pipe.worker_count(), expected);

        // On any genuinely multi-core host this must exceed the old
        // hardcoded default of 1 worker. Skip the strict `>1` check on a
        // single-core host so the test isn't flaky there.
        if detected > 1 {
            assert!(pipe.worker_count() > 1);
        }
    }

    #[test]
    fn pipeline_worker_count_clamp() {
        // n_workers=0 must behave as 1 worker (no panic, correct output).
        let decoder = QcLdpcDecoder::new(BaseGraph::Bg1, 0.25);
        let mut pipe = LdpcPipeline::with_workers(decoder, 3, 0);
        let mut frame = pipe.acquire().expect("pool has slots");
        frame.llr_mut().iter_mut().for_each(|v| *v = 5.0);
        assert!(pipe.submit(frame).is_ok());
        let result = loop {
            if let Some(r) = pipe.try_recv() {
                break r;
            }
            std::hint::spin_loop();
        };
        assert!(result.hard().iter().all(|&b| b == 0));
        pipe.release(result);
    }
}
