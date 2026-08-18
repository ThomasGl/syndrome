Workspace Guidelines & Instructions: syndrome

This file defines the technical constraints, software paradigms, style guidelines, and execution targets for syndrome developers and AI code assistants.

1. Mathematical Rigor & Code Style

- Naming Conventions: Enforce strict standard Rust naming conventions. Functions and local variables must use snake_case, structures and enumerations must use UpperCamelCase, and constants must use SCREAMING_SNAKE_CASE.
- System Documentation: All public APIs must be documented with descriptive triple-slash `///` docstrings. Include `# Arguments`, `# Returns`, `# Errors` (if returning a `Result`), and executable `# Examples` where appropriate.
- Math Formatting: In rustdoc comments (`///`, `//!`) ONLY, format computational theories, algorithm descriptions, and mathematical proofs strictly in LaTeX (`$...$` for inline, `$$...$$` for block expressions), following the CommonMark-escaping convention documented in `katex-header.html`. In README.md, CHANGELOG.md, and system_architecture.md, do NOT use LaTeX $-delimited math at all — use plain-text/Unicode notation instead (β not \beta, × not \times, subscripts as `Eb/N0` not `$E_b/N_0$`, standalone equations as a ```text fenced block). This is not a style preference: crates.io renders the README as a bare HTML fragment with no script or stylesheet injection possible, so LaTeX $...$ syntax can never render there under any convention — it displays as literal backslashes and braces to every visitor landing on the crate's crates.io page. Verified 2026-08 via `https://static.crates.io/readmes/<crate>/<crate>-<version>.html`, which contains no `<script>`/`<link>` tags at all. `tests/doc_math.rs` enforces the rustdoc-only rule for `src/`; there is no equivalent automated check for the three plain-text files, so any edit there must eyeball for a stray $ or backslash macro before committing.

2. High-Performance Constraints (Non-Negotiable)

- Zero-Allocation Hot-Paths: The inner loops of the LOMS decoder and the task ring buffers MUST execute without any dynamic heap allocations. Any use of `Box::new`, `Vec::new`, `Vec::push`, `clone()` or other heap allocation inside a hot path is a critical bug.
- Flat Memory Layout: Reject nested vectors (`Vec<Vec<T>>`) or boxed arrays for streaming matrices. Represent all 2D matrices as flat, single-dimensional, cache-aligned slices (`[T; N]`, `&[T]`, or `&mut [T]`).
- Data Alignment: Structural components in SIMD pathways must enforce memory boundary alignment using attributes like `#[repr(align(64))]` to prevent CPU misaligned read stalls.
- SoA Preference: Prefer struct-of-arrays (SoA) for streaming variables and message buffers to preserve contiguous access in SIMD loops.
- Branch-Free Hot Loops: Avoid branch misprediction penalties within internal loops by expressing decision logic through arithmetic, bitwise, or SIMD mask operations when possible.

3. Multi-Architecture SIMD

- Algorithm Target: Implement the Layered Offset Min-Sum (LOMS) algorithm for 5G NR QC-LDPC decoding with exact 3GPP lifting parameter structures.
- Portable SIMD: Do NOT use `core::simd` / `portable-simd`. It requires nightly Rust and this crate builds on stable, so reaching for it produces code that does not compile for users. SIMD is hand-written `core::arch` intrinsics (`core::arch::x86_64`, `core::arch::aarch64`), and every vectorized path keeps a tested scalar fallback it is proven equivalent to by seeded randomized tests. Revisit only if portable-simd stabilizes.
- Architecture Layers (current): x86_64 AVX2, selected at runtime via `is_x86_feature_detected!`; aarch64 NEON, gated on `#[cfg(target_arch = "aarch64")]`; and the scalar reference everywhere else. Adding a layer must not change algorithm semantics.
- Architecture Layers (future targets, not implemented): bare-metal ARM Cortex-M `no_std` and x86_64 AVX-512. Neither exists in the crate today. Do not declare a Cargo feature, a module, or a documentation claim for either until it gates real, tested code — an advertised switch that gates nothing is a defect, and removing one after publication is a breaking change.

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

7. Public Documentation Voice (Non-Negotiable)

Everything the library ships — `README.md`, `CHANGELOG.md`, `system_architecture.md`, and rustdoc — describes what the library does now, for a reader who has never seen it before. The governing distinction:

- **A statement about a past state of this codebase is a liability. A statement about its current state, including its gaps, is an asset.** These look similar and must be treated as opposites.

- Document the present, not the path taken to it. No "before → after" columns, no speed-up multipliers measured against code that was deleted, no "previously X only did Y", no "now carries", no "every panic it originally discovered is now fixed". A reader has no prior version to compare against, so a relative statement carries no information for them — it only advertises defects in code nobody ever ran.

- Never delete a limitation to make the library look more complete. Statements of present scope are load-bearing and stay: that Wi-Fi multi-codeword segmentation and PPDU-level avbits derivation are not implemented (shortening and puncturing themselves are — see `wifi_rate_matching`), that no AVX-512 code exists in this crate, the deferred list in `system_architecture.md`. A reviewer who finds an undocumented gap concludes the author did not know it was there; one who finds it documented concludes the author scoped deliberately. These statements are what make every other claim in the documentation credible, and removing them costs more than the gap they admit.

- Every published number must be reproducible by running the current tree (`bench/run_all.sh`, `algo_bench_export`). Never hand-write a benchmark figure. A comparison is legitimate only when both sides still exist and are tested — scalar versus SIMD, Rust versus the C++ port — never against a deleted earlier implementation, because nobody can re-run it.

- A changelog is written for users, not for the author. At a version nobody could have installed there is nothing to have `Fixed`, `Changed` or `Performance`-improved; a first release gets a single `Added` list of what the library provides.

- Working notes stay out of the published repository. `/notes/` is gitignored — self-study plans, progress reports, and personal roadmaps do not belong in a public library, and a document that reads as written *to* the author rather than *for* a reader belongs there.

- **Commit history is exempt from this section.** Commit messages are the engineering record and should stay detailed and candid about what was fixed and why. This section governs the documentation the library ships, not git history.
