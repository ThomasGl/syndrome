Workspace Guidelines & Instructions: syndrome

This file defines the technical constraints, software paradigms, style guidelines, and execution targets for syndrome developers and AI code assistants.

1. Mathematical Rigor & Code Style

- Naming Conventions: Enforce strict standard Rust naming conventions. Functions and local variables must use snake_case, structures and enumerations must use UpperCamelCase, and constants must use SCREAMING_SNAKE_CASE.
- System Documentation: All public APIs must be documented with descriptive triple-slash `///` docstrings. Include `# Arguments`, `# Returns`, `# Errors` (if returning a `Result`), and executable `# Examples` where appropriate.
- Math Formatting: All computational theories, algorithm descriptions, and mathematical proofs must be formatted strictly in LaTeX (`$...$` for inline, `$$...$$` for block expressions).

2. High-Performance Constraints (Non-Negotiable)

- Zero-Allocation Hot-Paths: The inner loops of the LOMS decoder and the task ring buffers MUST execute without any dynamic heap allocations. Any use of `Box::new`, `Vec::new`, `Vec::push`, `clone()` or other heap allocation inside a hot path is a critical bug.
- Flat Memory Layout: Reject nested vectors (`Vec<Vec<T>>`) or boxed arrays for streaming matrices. Represent all 2D matrices as flat, single-dimensional, cache-aligned slices (`[T; N]`, `&[T]`, or `&mut [T]`).
- Data Alignment: Structural components in SIMD pathways must enforce memory boundary alignment using attributes like `#[repr(align(64))]` to prevent CPU misaligned read stalls.
- SoA Preference: Prefer struct-of-arrays (SoA) for streaming variables and message buffers to preserve contiguous access in SIMD loops.
- Branch-Free Hot Loops: Avoid branch misprediction penalties within internal loops by expressing decision logic through arithmetic, bitwise, or SIMD mask operations when possible.

3. Multi-Architecture SIMD

- Algorithm Target: Implement the Layered Offset Min-Sum (LOMS) algorithm for 5G NR QC-LDPC decoding with exact 3GPP lifting parameter structures.
- Portable SIMD: Leverage `core::simd` / `portable-simd` where stable, and provide fallback scalar implementations for non-vectorized targets.
- Architecture Layers: Support bare-metal ARM Cortex-M `no_std`, aarch64 Neon, and x86_64 AVX2/AVX-512 paths without changing the algorithm semantics.

4. Lock-Free Synchronization Queueing

- No OS Locks: Do not use `std::sync::Mutex`, `std::sync::RwLock`, or other kernel-based locks in the queue hot path.
- Atomic Coordination: Use atomics (`AtomicUsize`, `AtomicBool`) with explicit memory orderings (`Acquire`, `Release`) for queue head/tail and worker coordination.
- SPSC Ring Buffers: Implement single-producer single-consumer ring buffers with a preallocated fixed capacity and no dynamic resizing.
- Affinity Control: Wrap worker threads with platform-specific affinity bindings where available to keep task data on the same physical core and reduce cache invalidation.

5. Incremental Refactor Strategy

- Preserve Existing Implementation: Evolve what exists rather than rewriting from scratch. Maintain backward-compatible entry points while introducing new optimized modules.
- Step 1 — Layout Migration: Refactor nested or hierarchical arrays into flat contiguous buffers first. Keep legacy math active while validating data layout parity.
- Step 2 — Algorithm Transition: Swap legacy FEC kernels toward the layered QC-LDPC LOMS update equations inside preallocated scratch buffers.
- Step 3 — Concurrency Wrapping: Add lock-free SPSC queue wrappers around the decoder pipeline once the core kernel is stable and validated.

6. Repository Updates

- Add new modules incrementally with tests and example usage.
- Update `README.md` and architecture documentation when new components or targets are introduced.
- Keep `Cargo.toml` lightweight and avoid adding dependencies unless the target architecture or queue abstraction strictly requires them.
