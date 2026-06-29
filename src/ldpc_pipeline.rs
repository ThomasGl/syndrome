//! Lock-free pipelined LDPC decoder backed by SPSC rings.
//!
//! [`LdpcPipeline`] wraps a [`QcLdpcDecoder`] so that frame submission and
//! result collection are fully decoupled. One or more background worker threads
//! run the LOMS kernel; coordination between the caller and the workers uses
//! only [`SpscRing`] SPSC queues — no OS mutexes are touched on the hot path.
//!
//! # Frame lifecycle
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
//! All 16 frame slots are pre-allocated at construction time.  The ring
//! protocol guarantees that exactly one entity (caller or a worker) holds each
//! slot at any moment, so there is no concurrent access.
//!
//! # Example
//!
//! ```no_run
//! use glezer_rsv::{BaseGraph, QcLdpcDecoder, LdpcPipeline};
//!
//! let decoder  = QcLdpcDecoder::with_lifting_size(BaseGraph::Bg1, 384, 0.25).unwrap();
//! let mut pipe = LdpcPipeline::new(decoder, 10);
//!
//! // Acquire a free slot, fill with channel LLRs, and submit.
//! let mut frame = pipe.acquire().expect("pool has 16 slots");
//! frame.llr_mut().iter_mut().for_each(|v| *v = 5.0);
//! pipe.submit(frame);
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
}

/// A borrowed decode slot.  Obtained via [`LdpcPipeline::acquire`].
///
/// Fill [`llr_mut`] with channel LLR values, then hand ownership to the
/// pipeline with [`LdpcPipeline::submit`].  After receiving a completed
/// frame via [`LdpcPipeline::try_recv`], read results from [`hard`], then
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
}

/// Lock-free pipelined LDPC decoder with optional multi-worker parallelism.
///
/// See [module-level documentation](self) for the frame lifecycle.
pub struct LdpcPipeline {
    /// Pre-allocated slot pool; never reallocated after construction.
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
    /// Create a single-worker pipeline. Equivalent to `with_workers(decoder, iterations, 1)`.
    ///
    /// # Arguments
    ///
    /// * `decoder`    - Fully constructed decoder. The pipeline takes ownership.
    /// * `iterations` - LOMS iteration count applied to every frame.
    pub fn new(decoder: QcLdpcDecoder, iterations: usize) -> Self {
        Self::with_workers(decoder, iterations, 1)
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
        let n_workers = n_workers.max(1).min(POOL_SIZE / 2);
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
            // on the (mutable) decode scratch; worker 0 takes ownership of the
            // original to avoid an unnecessary clone.
            let dec: QcLdpcDecoder = if wi == 0 {
                decoder.clone()
            } else {
                decoder.clone()
            };

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

                            // edge_r must start at zero each decode call.
                            slot.edge_r.iter_mut().for_each(|r| *r = 0.0);

                            let _ = dec.decode_layered_offset_min_sum(
                                &mut slot.llr,
                                &mut slot.edge_r,
                                &mut slot.scratch,
                                &mut slot.hard,
                                iterations,
                            );

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

    /// Borrow a free decode slot. Returns `None` when all 16 slots are in
    /// flight (caller is submitting faster than the workers can decode).
    pub fn acquire(&mut self) -> Option<LdpcFrame> {
        let idx = self.free_list.pop()?;
        // SAFETY: pool is never reallocated; idx is in [0, POOL_SIZE).
        let ptr = self.pool[idx].as_mut() as *mut FrameSlot;
        Some(LdpcFrame { slot_idx: idx, ptr })
    }

    /// Submit a frame for decoding. Dispatches to the next worker in
    /// round-robin order. Returns `false` if the chosen ring is full.
    ///
    /// The frame is consumed: the caller must not use it after this call.
    pub fn submit(&mut self, frame: LdpcFrame) -> bool {
        let idx = frame.slot_idx;
        // Prevent the destructor from running — the slot is now owned by the ring.
        std::mem::forget(frame);
        let wi = self.next_submit % self.n_workers;
        self.next_submit = self.next_submit.wrapping_add(1);
        self.work_rings[wi].try_push(idx).is_ok()
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
        let idx = frame.slot_idx;
        std::mem::forget(frame);
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
        assert!(pipe.submit(frame));

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
            assert!(pipe.submit(frame));
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
            if submitted < FRAMES {
                if let Some(mut frame) = pipe.acquire() {
                    frame.llr_mut().iter_mut().for_each(|v| *v = 5.0);
                    assert!(pipe.submit(frame));
                    submitted += 1;
                }
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
    fn pipeline_worker_count_clamp() {
        // n_workers=0 must behave as 1 worker (no panic, correct output).
        let decoder = QcLdpcDecoder::new(BaseGraph::Bg1, 0.25);
        let mut pipe = LdpcPipeline::with_workers(decoder, 3, 0);
        let mut frame = pipe.acquire().expect("pool has slots");
        frame.llr_mut().iter_mut().for_each(|v| *v = 5.0);
        assert!(pipe.submit(frame));
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
