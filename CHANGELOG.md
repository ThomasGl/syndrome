# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Fixed-point QC-LDPC decode path** (`src/qc_ldpc.rs`, `src/quantize.rs`,
  `src/simd_avx2.rs`): `QcLdpcDecoder::decode_layered_offset_min_sum_i8`,
  its forced-scalar twin `..._i8_scalar`, and the 5G wrapper `decode_5g_i8`
  run the same layered offset min-sum algorithm as the `f32` path with
  8-bit check-to-variable messages and a 16-bit a-posteriori accumulator.
  `quantize.rs` gains `QuantParams` (scale and posterior clamp),
  `quantize_llr_i16`, and the constants `DEFAULT_SCALE`, `MSG_MAX`,
  `APP_CLAMP_WIDE` and `APP_CLAMP_I8`.

  The AVX2 kernel works 32 z-positions per 256-bit register, against 8 for
  the `f32` kernel, and the scalar reference remains the tested definition of
  the path: every operation is integer, so the two are required to agree
  bit-for-bit — `tests/ldpc_int8_kernel_equivalence.rs` asserts that on
  seeded random channel data across both 3GPP base graphs, the 802.11
  matrices, and lifting sizes chosen to exercise the vector body, the
  32-element tail, the 16-element Q-build tail, and the group that straddles
  the cyclic-shift wrap.

  There is no NEON kernel for the fixed-point path: on AArch64 it runs its
  scalar reference, and only the `f32` path is vectorized there.

- **Measured quantization loss** (`tests/ldpc_int8_quantization_loss.rs`):
  the extra $E_b/N_0$ the fixed-point decoder needs to reach the same block
  error rate as the `f32` one, over the crate's BPSK AWGN channel with
  $s = 8$, $\beta = 0.5$ and 10 iterations.

  | Code | $E_b/N_0$ | Shift | 95% CI |
  |---|---|---|---|
  | BG1, $Z = 128$ | 0.80 dB | +0.0031 dB | [+0.0005, +0.0057] |
  | BG1, $Z = 384$ | 0.75 dB | +0.0052 dB | [+0.0035, +0.0070] |
  | BG2, $Z = 128$ | 0.60 dB | +0.0096 dB | [+0.0066, +0.0126] |
  | BG2, $Z = 384$ | 0.60 dB | +0.0067 dB | [+0.0044, +0.0089] |

  Every interval excludes zero, so the loss is real; it is between 0.003 and
  0.010 dB, with every upper bound below 0.013 dB. Resolving a hundredth of a
  dB is possible because the trials are **paired** — each hands the same
  received vector to both decoders, so the channel's variance cancels and
  only the trials the two disagree on carry information — and because the
  waterfall is steep enough that a 0.01 dB displacement still moves the block
  error rate by about 10%. The file also carries the sweeps behind the two
  format decisions: the posterior width (clamping it to the message range
  roughly doubles the block error rate and raises the bit error rate about
  eightyfold, an error floor rather than an offset) and the scale (a broad
  plateau from $s = 8$ to $s = 24$, with $s = 2$ and $s = 32$
  resolvably worse).

  Until now `src/quantize.rs` documented its own loss as unmeasured and
  quoted a published figure for other decoders. That paragraph is replaced by
  this measurement.

- **Rustdoc math CI gate** (`tools/check_doc_math.mjs`, `tools/package.json`,
  CI job `doc-math`): extracts every LaTeX span from the crate's doc
  comments, applies CommonMark's backslash-escape rule, and parses the result
  with the same KaTeX version `katex-header.html` loads. rustdoc renders doc
  comments as CommonMark before KaTeX sees them, and `throwOnError: false`
  means a malformed expression appears as red raw source on the rendered page
  rather than failing the build — so this class of bug survives review of the
  generated pages. The gate also checks the source for row breaks written
  with two backslashes instead of four, which is the failure that *does*
  parse: it reaches KaTeX as a control space and silently collapses a `cases`
  or `bmatrix` block to a single row.

- **Adaptive CA-SCL list size** (`PolarDecoder::decode_scl_adaptive`,
  `AdaptiveDecodeReport`): decodes at $L = 1$ first, and escalates to 2, 4, …
  up to the configured list size only when the CRC rejects the result.

  The error-rate performance is that of the *largest* list, because the ladder
  stops early only when the CRC — a check the decoder cannot satisfy by
  guessing — has confirmed the answer. The cost is where it pays: escalating
  all the way runs $1 + 2 + \dots + L < 2L$ list-units, so the worst case is
  under twice a single full-size decode, and that case only arises on blocks
  the channel damaged badly. At a working SNR most blocks pass at $L = 1$, so
  the average cost falls toward plain successive cancellation while the error
  rate stays at $L$. Measured here at 6 dB on $N = 256$, $K = 128$: one
  list-unit per block against a fixed $L = 8$ decoder's eight.

  Requires a CRC, and says so with an error rather than silently escalating to
  the maximum on every block — without one there is no signal that a decode
  succeeded, so there is nothing to escalate on.

  `AdaptiveDecodeReport` returns the list size that produced the bits, the CRC
  verdict and the attempt count, because the cost actually paid is the whole
  reason to prefer this over a fixed large list and a caller should be able to
  see it.

- **Mutation audit with `cargo-mutants`** (`mutants.toml`, plus the tests it
  demanded): 185 mutants across `bits`, `quantize`, `harq`, `segmentation` and
  `spsc_queue`. Six survived, every one a real gap, and all six are now closed.

  | Survivor | What it meant |
  |---|---|
  | `HarqBuffer::combine`: `n_filler_override > 0` | The override branch had no test at all — every caller in the suite passed `0`, so deleting the parameter would not have failed anything. |
  | `HarqBuffer::copy_llr_into`: `dst.len() < acc.len()` → `<=` | The existing test passed a 1-element destination, which both forms reject. An exactly-`ncb` buffer — the one a caller who read `ncb()` would allocate — was never tried. |
  | `compute_segmentation`: all four BG2 $K_b$ thresholds | The TS 38.212 §5.2.2 ladder could be loosened from `>` to `>=` unnoticed, because $K_b$ is not among the fields `SegmentationParams` reports and no test used a transport block sitting *at* a threshold. |

  The $K_b$ finding is the one worth dwelling on: it is a transcribed spec
  table feeding a value that shifts $Z$, $K$ and the filler count for a *band*
  of transport block sizes while leaving every size outside that band correct
  — the exact shape of a transcription slip, and invisible to round-trip tests
  that sample sizes at random. The ladder is now `lifting_selection_k_b`, a
  separate function purely so it can be tested at each boundary from both
  sides.

  One survivor in `bits_to_bytes` (`|` → `^`) is left as-is: `byte << 1`
  always leaves the low bit clear and the input is validated to be 0 or 1
  before the loop, so the two operators are provably identical there. It is an
  equivalent mutant, not a gap.

  `mutants.toml` records the scoping. A full run is ~4,500 mutants — weeks at
  this suite's runtime — so the audit is per module, and two kinds of module
  are excluded deliberately rather than silently: the SIMD kernels, which have
  a stronger gate already (bit-identical equivalence against a retained scalar
  reference) and mostly produce timeouts under mutation, and the benchmark
  exporters, which have no assertions to violate.

  A side finding worth recording: 28 of the `spsc_queue` mutants were detected
  as TIMEOUT rather than CAUGHT — a mutated ring makes the cross-thread stress
  tests spin forever. The defect *is* caught, after 219 seconds of waiting. The
  loom models added alongside catch the same class in milliseconds and name the
  ordering that broke, which is the better tool for that module; the mutation
  run is what showed how much of the SPSC coverage rested on a hang.

- **Load-aware dispatch in `LdpcPipeline`** (`pick_worker`): frames now go to
  the worker with the fewest outstanding, ties broken by a rotating cursor,
  replacing strict round-robin.

  Round-robin is only fair when every frame costs the same, and LDPC frames do
  not: the decoder stops as soon as the syndrome check passes, so a code block
  from a clean channel finishes in two iterations while a marginal one runs the
  full budget — an order of magnitude apart on one configuration. Dealing in
  rotation therefore hands each worker whatever lands on its turn, and a run of
  expensive frames on one worker leaves the rest idle while it drains. The
  count is maintained on the submitting thread alone (incremented on dispatch,
  decremented when a frame returns), so the policy costs an `n_workers` scan
  and no atomics.

  Ties rotate rather than taking the first minimum, which matters at the start
  of a burst when every worker is at zero: without it the whole burst would go
  to worker 0 until something completed. The policy therefore degrades to
  round-robin exactly when round-robin is right — when nothing distinguishes
  the workers.

  `pick_worker` is a free function over the load vector precisely so it can be
  tested: which worker `submit` picks cannot be asserted from outside, because
  the workers run concurrently and any such test measures the scheduler too.

- **Scoped Miri run over the crate's non-SIMD `unsafe`** (CI job `miri`):
  `cargo miri test --lib spsc_queue` and `--lib ldpc_pipeline`.

  The scope is not a sample. Every other `unsafe` in the crate is a call into
  `simd_avx2`/`simd_neon`, which Miri cannot execute — so those two modules
  *are* the non-SIMD unsafe code: the SPSC ring's cells and the LDPC
  pipeline's raw-pointer frame-pool protocol. The SIMD kernels are held to a
  different standard instead, being proven equal to a retained scalar
  reference.

  Miri sees what a test suite cannot — a pointer outliving its allocation, a
  `&mut` overlapping a `&`, an index that lands inside a different live object
  — and it is the natural complement to the loom work: loom checks *ordering*
  across interleavings, Miri checks *memory validity* along each one. The
  cross-thread stress tests scale themselves down under `cfg(miri)`
  (`STRESS_COUNT`), because Miri interprets at roughly a thousandth of native
  speed and what it checks is structural rather than statistical.

- **`DlSchDecoder` can decode code blocks concurrently**
  (`DlSchDecoder::with_pipeline`): the lock-free [`LdpcPipeline`] was built,
  benchmarked and documented, but nothing in the transport-block chain used
  it — `DlSchDecoder::decode` walked its code blocks one at a time on the
  calling thread, so the concurrency machinery was not on the real path.
  Installing a pipeline dispatches each code block to a worker instead.

  It is opt-in and stays that way. `LdpcPipeline` spawns threads that spin for
  the decoder's lifetime, which is right for a receiver processing a stream of
  large transport blocks and wrong for a short-lived decoder or a transport
  block that fits in one code block — so `DlSchDecoder::new` remains
  thread-free, and `decode` takes the sequential path whenever $C = 1$ even
  with a pipeline installed.

  Nothing observable changes. Both paths run the same LOMS decoder over the
  same HARQ-combined LLRs for the same iteration budget; code blocks merely
  finish out of order and are reassembled by index rather than by arrival.
  `pipelined_decode_matches_sequential_decode` asserts the two agree
  bit-for-bit on the decoded transport block, the per-code-block CRC flags,
  the transport-block CRC and the reported iteration count, over a noisy
  channel — noise deliberately, because a clean one converges every code block
  in a single iteration and they then complete in submission order, hiding the
  reordering the test exists to catch. Three more cover more code blocks than
  the pipeline has pool slots (which forces the back-pressure path), HARQ
  combining across a retransmission, and the single-code-block case.

  Two supporting additions: `QcLdpcDecoder::init_5g_llr` exposes the filler
  and puncture initialisation `decode_5g` used to keep private, because the
  pipelined path has to apply it before submitting a frame to a worker that
  calls the plain decode entry point — one definition rather than two copies
  free to drift; and `LdpcFrame::set_tag`/`tag` carry a caller-chosen
  identifier through the pipeline, which is what lets completions be matched
  back to their code block.

- **Exhaustive model check of the SPSC ring's memory ordering**
  (`tests/loom_spsc.rs`, `src/sync_shim.rs`, CI job `loom`): five
  [loom](https://docs.rs/loom) models covering the single-item entry points,
  the batched ones, the mixed pairing, and the over-capacity case where the
  producer must wait for the consumer to free a slot.

  The ring's `Acquire`/`Release` contract was documented but not checkable.
  The existing two-thread stress tests sample whichever interleavings the
  machine produces, and on x86-64 the hardware will not reorder a store past
  a store — so weakening either `Release` to `Relaxed` leaves them passing
  indefinitely there while introducing a bug that surfaces on AArch64, the
  architecture the crate's other CI job runs on. loom instead models the C11
  ordering rules and enumerates the executions a small program admits, and
  its instrumented cells flag a slot touched by both threads without an
  intervening happens-before edge even when the values come out right.

  Eight deliberately injected defects — each `Release` and `Acquire` weakened
  in turn, and each occupancy count moved by one — are all caught. Two of
  them were not, in the first draft: sending exactly `CAP` items never makes
  the batched producer wait for space, so `push_slice` never had to shorten a
  batch from the consumer's counter. The over-capacity model exists because
  those controls said it had to.

  Only the model check pays for any of this. `loom` is a
  `[target.'cfg(loom)'.dependencies]` entry, so it is absent from an ordinary
  build, from `cargo test`, and from anything a downstream user resolves;
  `src/sync_shim.rs` re-exports `core::sync::atomic` and a zero-cost
  `UnsafeCell` wrapper unless `--cfg loom` is set.

### Changed

- **`LdpcPipeline::submit` returns the frame on failure** instead of a bare
  `bool`: `Result<(), LdpcFrame>` rather than `bool`. `LdpcFrame` has no
  `Drop`, so a frame consumed by a failed submit simply vanished and its pool
  slot never returned to the free list — the pipeline would quietly lose one
  of its sixteen slots per occurrence and eventually stall with nothing
  reported anywhere. The failure is unreachable by construction (at most
  `POOL_SIZE` frames can be in flight and each ring holds `POOL_SIZE`), but a
  signature that makes an invariant load-bearing is a poor place to rely on
  it. `submit` also no longer advances its round-robin cursor on a failed
  dispatch, so a full ring cannot silently skip a worker's turn.

  Breaking for anyone calling `submit` directly; `if pipe.submit(f) { .. }`
  becomes `if pipe.submit(f).is_ok() { .. }`.

- **`SpscRing` now holds one `UnsafeCell` per slot** rather than one around
  the whole buffer array (`src/spsc_queue.rs`). With a single cell the
  producer's write has to form a mutable reference spanning every slot while
  the consumer holds a shared reference over the same range — an aliasing
  violation under Rust's reference rules even though the two touch disjoint
  elements and no byte is ever both read and written. The loom models report
  it as a causality violation, which is how it was found; nothing else in the
  suite could see it. Per-slot cells make the disjointness structural: each
  access borrows exactly the slot it touches. No API or behaviour change.

- `src/bin/ldpc_bench_export.rs` additionally times the fixed-point kernels
  (`loms_i8_runtime_simd`, `loms_i8_scalar`) over the same decode workload,
  quantizing the same `f32` LLR values, and writes them to
  `bench/results/ldpc_rust.json`. The cross-language checksum gate stays on
  the `f32` scalar kernel, since the C++ reference has no fixed-point path.

  README §5.2.2 publishes the result: BG1 Z=384, 10 iterations, the vectorized
  int8 kernel at ~854 Melem/s against the vectorized `f32` kernel's
  ~216 Melem/s, a ~4.0× gap. The two *scalar* rows are the control that makes
  that reading safe — narrowing the number format buys nothing without SIMD,
  and they agree at ~64 Melem/s to within their own run-to-run spread, so the
  gap above is the register width and not the arithmetic.

### Fixed

- **Reed-Solomon could not recover every erasure pattern within its stated
  capability.** The generator matrix is now Cauchy,
  $C_{ij} = (x_i \oplus y_j)^{-1}$ with $x_i = i$ and $y_j = m + j$, replacing
  $C_{ij} = \alpha^{ij}$.

  **This changes parity bytes on the wire.** Data encoded by syndrome 0.4.0 or
  earlier must be decoded with
  `ReedSolomon::with_matrix(k, m, MatrixKind::PowerVandermonde)`, which builds
  the old matrix and is retained for exactly that purpose.

  Recovery inverts the submatrix of $C$ on the missing data columns and on as
  many surviving parity rows as are needed. Writing $x_c = \alpha^{j_c}$, that
  submatrix is $\left[x_c^{\thinspace i_r}\right]$. While the surviving parity
  rows are $0, 1, \dots$ consecutively — which is what happens when no parity
  shard is lost — it is a true Vandermonde and nonsingular. Lose a parity
  shard and the row exponents skip: rows $\lbrace 0, 1, 3 \rbrace$ give
  $\left[1, x, x^3\right]$, a *generalized* Vandermonde whose determinant
  carries an extra symmetric factor, and over GF(256) that factor sometimes
  vanishes. The decoder then reported
  `FecError::InvalidParam("matrix not invertible")` for a shard set that
  should have been recoverable.

  It is not a corner case, and it is not confined to exotic geometries: at
  $k = 12$, $m = 5$ — an ordinary RS(17, 12) — 18 of the 6,187 patterns inside
  the code's capability failed; at $k = 16$, $m = 6$, 254 of 74,612; at
  $k = 20$, $m = 6$, 684 of 230,229. The patterns that break it are exactly
  those losing *both* data and parity shards, which is why an erasure test
  dropping only data shards finds nothing wrong — and why the module's own
  documentation asserted the opposite in good faith.

  Every square submatrix of a Cauchy matrix is a Cauchy matrix, and a Cauchy
  determinant is nonzero whenever the two index sets are distinct within
  themselves and disjoint from each other. All three hold by construction for
  any subset, so recovery now succeeds for **every** pattern; the new tests
  enumerate the full pattern space at each of those geometries and assert zero
  failures for Cauchy against the exact counts above for the old matrix.

  The module documentation's nonsingularity argument was wrong and is
  rewritten. The C++ and Python reference encoders in `bench/` are updated to
  the same construction, so the cross-language checksum gate still compares
  three independent implementations of the same thing — all three agree on the
  new parity.

- **Two aliasing violations in `LdpcPipeline`'s frame pool**, both found by
  Miri and neither observable any other way.

  The worker threads received their slot addresses as `Vec<usize>` — a raw
  pointer is not `Send`, and the integer round trip was the shortcut around
  that. It is undefined behaviour: converting a pointer to an integer and back
  discards its provenance, so the reborrow inside the worker has no claim to
  the allocation it names. Miri rejects it outright ("trying to retag from
  `<wildcard>` ... no exposed tags have suitable permission in the borrow
  stack"). Now a `SlotPtr` newtype carries the pointer intact and asserts
  `Send` in one place where it can be justified.

  With that repaired Miri found a second, deeper one: `acquire` and `try_recv`
  each took a *fresh* `&mut` (or, worse, a `&` cast to `*mut`) out of the pool
  `Vec` to build a frame, and every such borrow invalidates the pointers the
  workers are already holding — even though nothing reads through them at that
  instant. Both now take the pointer from the single `slot_ptrs` vector
  created at construction, so there is one provenance chain rather than three;
  `pool` does nothing after construction but own the allocations and free them
  on drop.

  No behaviour changes and no generated code changes under any compiler that
  exists today, which is exactly why no test could have caught either. This is
  the crate's lock-free showpiece, and it had been shipping undefined
  behaviour since the pipeline was written.

## [0.4.0] — 2026-08-16

### Added

- **Cross-language LDPC correctness gate** (`bench/run_all.sh`,
  `bench/cpp/loms_decode.cpp`, `src/bin/ldpc_bench_export.rs`): the C++ and
  Rust LOMS decoders now each hash their hard decisions after a decode from
  a fixed input and write `ldpc_{cpp,rust}.checksum`, which `run_all.sh`
  compares and fails the run on divergence — the same gate the Reed-Solomon
  path already had, extended to the LDPC benchmark whose numbers were
  previously published without any correctness check tying the two
  implementations together. Hard decisions rather than raw LLRs: RS parity
  can be compared byte-for-byte because GF(256) encoding is integer
  arithmetic, but LOMS is floating point and the two binaries are built by
  different compilers free to contract multiply-adds into FMAs and
  reassociate, so identical `f32` values are not something either toolchain
  guarantees. Which codeword the decoder settles on is integer, is what a
  downstream stage consumes, and is invariant to those differences. The two
  implementations currently agree on all 26,112 decisions.

- **Adversarial coverage for the Bluetooth, Wi-Fi rate-matching, and `bits`
  APIs** (`tests/robustness.rs`): these were the largest gaps in a suite
  whose stated goal is that no public API panics on arbitrary input — every
  Bluetooth entry point takes caller-sized buffers whose required lengths
  are derived from a code rate, which is exactly where a length-check slip
  becomes an out-of-bounds index. Six new fuzz functions drive the LE Coded
  PHY pattern mapper, both BR/EDR codes, the Wi-Fi shortening/puncturing
  path, and the bit/byte conversions with mismatched buffers, non-binary
  "bit" values, `usize::MAX` parameters, and NaN/infinite/subnormal LLRs.
  They found the `shortening_and_puncture_counts` overflow recorded under
  Fixed below.

- **Monte Carlo estimation harness** (`src/montecarlo.rs`): runs trials until
  a target number of *error events* has accumulated rather than a fixed
  trial count, because the relative precision of an error-rate estimate
  depends on the event count ($\approx 1/\sqrt{k}$) and not on how many
  trials produced them — a fixed budget over-runs at low SNR and returns no
  information at high SNR. Every result carries a Wilson score confidence
  interval, which stays inside $[0, 1]$ and gives a usable one-sided bound
  when zero errors were seen, where the textbook normal approximation
  collapses to $[0, 0]$ and asserts certainty from no evidence. The module
  documents what the interval does not cover: it is a statement about
  sampling noise only, and it is exact for block-level events but optimistic
  for bit-level ones, whose errors are correlated within a failed block.
  Includes an empirical coverage test that verifies a nominal 95% interval
  covers the true value about 95% of the time over 400 independent
  experiments.

- **Rayleigh block-fading channel** (`src/channel_sim.rs`):
  `RayleighBlockChannel` models multipath fading with a per-block amplitude
  $h$ drawn from a Rayleigh distribution normalized to $E[h^2] = 1$, so a
  given $E_b/N_0$ means the same average received energy as on the AWGN
  channel and the two are directly comparable. Perfect receiver CSI is
  assumed and the LLR is scaled by the realized gain; channel estimation
  error is not modelled, and the docs say so — results from this channel are
  an optimistic bound on a real receiver. `transmit_with_gains` also returns
  the realized amplitudes.

- **Statistical validation of the channel models** (`src/channel_sim.rs`):
  the existing tests checked determinism and LLR signs, none of which would
  notice noise with the wrong variance, a non-zero mean, or a non-Gaussian
  shape — defects that would silently shift every error-rate curve the crate
  produces while leaving the suite green. Added tests that recover the
  realized noise from the LLRs and check its mean and variance against the
  $E_b/N_0$ calibration, invert the calibration to confirm the measured SNR
  matches the requested one, run a $\chi^2$ goodness-of-fit test against the
  normal distribution, and bound the lag-1 autocorrelation. Both channels
  are additionally checked against the closed-form uncoded-BPSK error
  probabilities ($Q(1/\sigma)$ for AWGN, the standard
  $\tfrac{1}{2}(1-\sqrt{\bar\gamma_b/(1+\bar\gamma_b)})$ for Rayleigh), and
  the fading amplitudes against the Rayleigh moments $E[h] = \sqrt\pi/2$,
  $E[h^2] = 1$, $E[h^4] = 2$.

- **Exact log-MAP for the Turbo decoder** (`src/turbo.rs`): `MapAlgorithm`
  selects between `MaxLog` (the default, unchanged) and `LogMap`, which
  evaluates the full Jacobian correction $\ln(1 + e^{-|a-b|})$ at every BCJR
  combining step instead of dropping it. Both rules share one generic scalar
  kernel parameterized by a const flag, so they cannot drift apart and the
  max-log path monomorphizes back to exactly the branch-free `max` loop it
  was before. Selecting `LogMap` also disables the Vogt-Finger extrinsic
  damping, which exists to correct max-log's over-confidence and would
  discard correctly scaled information under exact log-MAP. Exact log-MAP is
  scalar-only: the AVX2/NEON kernels express one `max` per step and cannot
  carry a transcendental, so requesting `LogMap` overrides the backend
  rather than silently downgrading the algorithm. The docs state the trap
  that makes this option easy to misuse — max-log-MAP is invariant to a
  positive scaling of its input LLRs and exact log-MAP is **not**, so
  `LogMap` requires genuine LLRs and underperforms on arbitrarily scaled
  soft values. The accompanying test measures block error rate for both
  algorithms over identical channel realizations and requires disjoint 95%
  confidence intervals before claiming a difference.

- **Reed-Solomon errors-and-erasures decoding** (`src/reed_solomon.rs`):
  `ReedSolomon::decode_errors_and_erasures` corrects symbols that are
  *present but corrupted* — wrong bytes carrying no erasure flag — alongside
  any number of flagged erasures, whenever $2t + s \le$ `parity_shards`.
  It returns the number of unknown-position errors it actually found.
  The algorithm is a syndrome-verified combinatorial search over candidate
  error positions that reuses the existing Vandermonde erasure decoder as
  its only reconstruction engine, so it introduces no new field arithmetic.
  Berlekamp–Massey and Chien search (as used in `src/bch.rs`) are
  deliberately *not* used here and would be mathematically unsound for this
  code: data positions hold the message polynomial's coefficients while
  parity positions hold its evaluations $p(\alpha^i)$, so the parity-check
  matrix has Kronecker-delta columns at parity positions and the syndromes
  never take the classical $S_i = \sum_k e_k \beta_k^i$ form those algorithms
  require. Cost is $O\!\big(\binom{n}{\text{max\_errors}} \cdot
  \text{shard\_len}\big)$ — cheap for the small parity counts this crate
  targets, and the reason the method takes an explicit `max_errors` bound
  rather than searching unboundedly. Verified against a from-scratch
  exhaustive reference decoder that shares no code with the crate (its own
  Russian-peasant `GF(256)` multiply and Horner evaluator), plus seeded
  random round-trips across 11 $(d, p, t, s)$ shapes, and a case with more
  errors than the distance bound allows that must return
  `FecError::DecoderNotConverged` rather than a silently wrong answer.

- **Tail-biting convolutional codes** (`src/viterbi.rs`):
  `encode_tail_biting`, `decode_hard_tail_biting`, and
  `decode_soft_tail_biting` implement tail-biting termination, where the
  encoder's shift register is preloaded with the message's own final $K-1$
  bits so the trellis starts and ends in the same (unknown) state — no
  zero-tail flush bits, and therefore no rate loss on short blocks. Decoding
  uses the Wrap-Around Viterbi Algorithm: metrics are initialised uniformly
  across all states, carried around the circular trellis for up to four
  laps, and the decode stops as soon as a lap's traceback returns to the
  state it started from (a self-consistent circular path). Hard-decision and
  soft-decision paths both have AVX2 and NEON kernels sharing the same ACS
  step functions as the existing zero-terminated decoders, with seeded
  equivalence tests against the scalar reference. Round-trip correctness is
  tested from all 64 possible starting states for both hard and soft input.

### Changed

- **LDPC offset correction $\beta$ raised from `0.25` to `0.5`** where the
  crate picks it on the caller's behalf
  (`DlSchConfig::default_decode_params`). `0.25` is measurably the wrong
  value: `tests/ldpc_offset_beta_sweep.rs` sweeps $\beta$ against block
  error rate over the crate's AWGN channel using the new
  `montecarlo` harness, and on BG1 at production lifting sizes the gap is
  large and grows with $Z$ — at $Z = 384$, $E_b/N_0 = 1$ dB, 10 iterations,
  BLER is $0.133$ at $\beta = 0.25$ against $3.3 \times 10^{-4}$ at
  $\beta = 0.5$, with disjoint 95% confidence intervals. On BG2 the two
  best points ($0.35$ and $0.5$) are statistically indistinguishable from
  each other and both beat $0.25$. $\beta$ remains a caller-supplied
  parameter everywhere else in the public API; only the value the crate
  selects by itself changed. The sweep is checked in, so the choice can be
  re-measured rather than trusted, and `tests/` now carries fast regression
  tests that fail if $\beta$'s advantage over both $0$ and an over-large
  offset stops being resolvable.

- **`src/viterbi.rs` module documentation** now states the precise limits of
  its "maximum likelihood" claim, which previously read as unqualified. The
  decoder is maximum-likelihood *sequence* detection, not bitwise MAP: it
  minimizes the probability that the whole sequence is wrong and produces no
  per-bit reliability, which is the distinction from the BCJR decoder in
  `src/turbo.rs`. Optimality also holds only under each branch metric's own
  channel model — Hamming distance is ML for a BSC with $p < 1/2$,
  correlation is ML for BPSK over AWGN. And it does not extend to the
  tail-biting decoder at all: WAVA is an approximation to ML decoding of a
  circular trellis, returning a certified tail-biting codeword when its
  self-consistency check succeeds and the best candidate found otherwise,
  with no flag distinguishing the two.

- **`src/quantize.rs` module documentation** now states plainly that this
  crate has *not* measured its own quantization loss, because the vectorized
  `i8` LOMS kernel this module exists to feed is not implemented, so there
  is no decode path here to measure. The previous text asserted a "< 0.1 dB"
  figure with no source. The replacement cites a published survey for
  min-sum decoders generally
  (<https://par.nsf.gov/servlets/purl/10156560>) and says explicitly that
  the figure describes other implementations until this crate has a kernel
  and its own BER/BLER sweep to back a number.

- **AVX2 QC-LDPC kernel alignment contract** (`src/simd_avx2.rs`,
  `src/qc_ldpc.rs`): the `min1`/`min2`/`sxor` scratch buffers are now accessed
  with the aligned AVX2 load/store forms (`_mm256_load_ps`,
  `_mm256_load_si256`) instead of the unaligned ones, and the sole caller
  backs them with `#[repr(align(64))]` locals always sliced from index 0 to
  guarantee that. The safety contract on `decode_layer_passes_avx2` documents
  the requirement. `edge_r` and `q_row` keep the unaligned forms: their
  per-layer offsets are data-dependent running sums, so no unconditional
  alignment guarantee is available for them without a larger layout change.
  Measured LDPC throughput is unchanged (~217 Melem/s) — modern x86 handles
  aligned and unaligned loads at the same rate when the address happens to
  be aligned; this is a correctness/intent change, not a speed-up.

### Fixed

- **Benchmark harness measured the AVX2 kernels while they were still
  ramping** (`src/bin/algo_bench_export.rs`, `src/bin/bench_export.rs`). Both
  exporters warmed up for a fixed *call count* and then reported the mean of
  a single timed block. Two independent defects followed, and the published
  Reed-Solomon figures were the main casualty:

  A fixed call count is not a fixed amount of warm-up. Twenty warm-up calls
  is tens of microseconds — enough for the scalar codecs, far too little for
  the AVX2/GFNI Reed-Solomon kernel to reach a steady frequency. Measured on
  this host, the same timed block reports ~78 Gbit/s after 20 warm-up calls
  against ~165 Gbit/s once fully warmed, and which regime a given run landed
  in varied with host state. The identical measurement against the *scalar*
  encode kernel on the same buffers shows no such ramp, which is what rules
  out cache residency as the cause.

  Separately, a mean cannot reject a timed block that was preempted. Per-call
  times here are heavily right-tailed (99.9th percentile ~7× the median,
  worst case ~50×), so 100 independent blocks timing the identical encode
  spanned 90–163 Gbit/s.

  Both exporters now warm up for a fixed wall-clock duration and report the
  median of 51 timed rounds. Run-to-run spread on the same tree drops from
  28% to ~7% for Reed-Solomon encode, 100% to ~4% for CRC-24A, 62% to ~4% for
  Turbo encode, and 54% to ~5% for Viterbi encode. Every published number in
  `README.md` moved accordingly — Reed-Solomon encode most of all, from a
  reported ~97 Gbit/s to ~162 Gbit/s, because the old figure was measuring a
  partly-cold vector unit rather than the kernel.

- **Unchecked subtraction in
  `wifi_rate_matching::shortening_and_puncture_counts`**: the function
  computed `n - (k - payload_bits)` having checked only
  `payload_bits <= k`, never that `n >= k`. `k` and `n` are free `usize`
  parameters rather than values read off a validated matrix, so a caller
  could supply a pair no real block code has and reach the subtraction. In a
  debug build that panicked; in a release build — the worse case — it wrapped
  to a value near `usize::MAX`, which let the range check below it pass for
  essentially any input and returned a meaningless puncture count instead of
  an error. Now rejected with `FecError::InvalidParam`. Found by the new
  adversarial coverage below, with a minimized reproducer kept in
  `tests/robustness.rs`.

- **Polar code reliability sequence** (`src/polar.rs`): `RELIABILITY_SEQ` now
  embeds the complete 1024-entry 3GPP `Q_Nmax` table (TS 38.212 Table
  5.3.1.2-1), cross-validated byte-for-byte against two independent
  open-source implementations (`robmaunder/polar-3gpp-matlab` and
  OpenAirInterface5G's `nr_polar_sequence_pattern.c`). `frozen_mask` now
  consults the real table for every `N ≤ 1024` — the entire 3GPP-defined
  range, including PBCH (`N=512`) and PDCCH up to `N=1024` — instead of
  falling back to a polarization-weight approximation above `N=256`. The
  previously embedded `N ≤ 256` prefix was also incomplete (247 of the 256
  required entries; nine reliability positions were missing from the
  hand-transcribed table) and is corrected by this replacement. The PW
  fallback remains for `N > 1024`, outside the range 5G NR polar codes are
  defined for.

### Performance

- **Reed-Solomon GFNI acceleration** (`src/reed_solomon.rs`, `src/simd_avx2.rs`):
  `encode_with_avx2` and erasure `decode` now runtime-detect GFNI
  (`_mm256_gf2p8affine_epi64_epi8`) and prefer it over the existing AVX2
  VPSHUFB nibble-table kernel when available, falling back to VPSHUFB where
  GFNI isn't present. Multiplying by a fixed `GF(256)` coefficient is
  $\mathbb{F}_2$-linear, so GFNI applies the coefficient's precomputed
  $8 \times 8$ bit matrix directly in one instruction per 32 bytes, instead
  of VPSHUFB's four-instruction shuffle/mask/blend sequence for the same 32
  bytes. Measured by `reed_solomon::tests::bench_gfni_vs_avx2_nibble`, which
  alternates the two kernels inside a single process (21 rounds × 200
  iterations, medians) so both see the same turbo and thermal state: GFNI is
  1.51×/1.73×/1.65×/1.35× the VPSHUFB kernel's throughput at 256 B/1 KiB/4
  KiB/16 KiB shards. That interleaved form is what the comparison rests on —
  comparing separate `bench/run_all.sh` runs could not have established this
  either way — see the benchmark-harness entry under Fixed for why those
  runs were unstable, and note that this interleaved measurement was
  unaffected by that defect and reproduces to within 0.03x across runs.
  Both kernels are proven byte-identical to the scalar reference; the GFNI
  bit matrix is additionally checked exhaustively against all 256 `GF(256)`
  coefficients.

## [0.3.0] — 2026-08-09

### Added

- **Wi-Fi 802.11 LDPC shortening and puncturing** (`src/wifi_rate_matching.rs`):
  `encode_shortened`/`decode_shortened` accept a payload smaller than a
  codeword's $K$ (padded with known-zero bits that are never transmitted)
  and a transmitted length smaller than the post-shortening budget (parity
  bits dropped from the tail), for any of the 12 real 802.11 $(Z, R)$
  matrices. The decoder reconstructs the full-length LLR buffer before
  decoding: shortened positions get a high-confidence known-zero LLR,
  punctured positions get an erasure ($LLR = 0$). Previously only a full,
  unshortened, unpunctured codeword ($K$ info bits exactly filling $N$
  coded bits) could be encoded or decoded.
  `tests/wifi_shortening_puncturing_integration.rs` verifies the encode →
  AWGN → decode round-trip, with genuine shortening and puncturing applied,
  across all 12 combinations. Still not implemented, and now the documented
  scope boundary: multi-codeword segmentation (a payload larger than one
  codeword's $K$), and the 802.11 PPDU-level formula (§19.5.3.2) that
  derives the available coded-bit count from an MCS, bandwidth, and PSDU
  length — callers supply that length directly instead.
- **Bluetooth FEC profiles** (`src/bluetooth.rs`): the complete set of FEC
  schemes in the Bluetooth Core Specification (unchanged since 5.0; verified
  against 5.0 through 6.2) — the LE Coded PHY convolutional code ($K=4$,
  rate 1/2, $G_0 = 1+x+x^2+x^3$, $G_1 = 1+x^2+x^3$) built on the existing
  Viterbi engine with hard and soft decoding, the $S=8$ pattern
  mapper/soft-demapper, BR/EDR FEC 1/3 (3× repetition, majority decode),
  and BR/EDR FEC 2/3 (the (15,10) shortened Hamming code,
  $g(D)=D^5+D^4+D^2+1$, single-error correction with double-error
  detection). Unit tests reproduce the specification's own sample data
  bit-exactly: the Vol 6 Part C reference packet (every FEC output bit for
  both CI values and the $S=8$ symbol stream) and all ten (15,10)
  generator rows from the Vol 2 FEC sample data, cross-checked against
  libbtbb's independent table. Out of scope, documented: packet assembly,
  whitening, and the Bluetooth CRC/HEC family.
- **Public `bits` module** (`src/bits.rs`): MSB-first `bytes_to_bits` /
  `bits_to_bytes` (caller-buffer and `_vec` forms, validating that every
  element is 0/1) and `hard_decision`, the crate-wide LLR sign rule
  ($L < 0 \Rightarrow 1$). Nearly every API in the crate speaks
  one-bit-per-byte, and until now every user had to write these conversions
  themselves.
- **`LdpcWorkspace`** (`src/qc_ldpc.rs`): an owning bundle of all four LDPC
  decode buffers, built by `QcLdpcDecoder::workspace()`. New
  `decode_with_workspace` / `decode_5g_with_workspace` /
  `wifi_rate_matching::decode_shortened_with_workspace` entry points replace
  the three sizing calls and four `vec![...]` lines previously required
  before a first decode (the Wi-Fi shortened decode drops from 8 parameters
  to 5). The raw slice entry points are unchanged for callers who want exact
  allocation control; each decode remains allocation-free either way, and
  the workspace path is tested bit- and iteration-identical to the raw path.
- **`DlSchConfig`** (`src/transport_block.rs`): named-field configuration for
  the DL-SCH pair, with `DlSchEncoder::from_config` /
  `DlSchDecoder::from_config`, so the six positional numerics of
  `DlSchDecoder::new` can no longer be transposed silently and the encoder
  and decoder can be built from literally the same value.
- **`syndrome::VERSION`**: the crate version captured at compile time, so
  downstream wrappers (e.g. the Python binding) can report which core they
  were compiled against instead of their own version.

### Fixed

- **A build without `data/bg_tables.json` silently produced fake 5G base
  graphs.** `build.rs` emitted a 2-entry placeholder BG1/BG2 when the table
  file was missing — compiling cleanly into a crate whose "base graphs" were
  two-edge toys, with no warning anywhere. The file ships in both git and
  the crates.io tarball, so its absence always means a broken checkout; it
  is now a hard build error naming the regeneration tool. The `bg1`/`bg2`
  keys and their `rows`/`cols` fields are also required now instead of
  being silently skipped or defaulted.

## [0.2.0] — 2026-08-04

### Changed

- **MSRV raised from 1.85 to 1.97** (latest stable at time of release). A
  minor-version bump rather than a patch, because raising the MSRV is a real
  breaking change for anyone still on an older toolchain — Cargo's default
  caret matching (`^0.1.x`) would otherwise have handed this to existing
  dependents automatically. `rust-toolchain.toml` already tracked `channel =
  "stable"` generically, so no pin needed updating there; only the declared
  floor moved.
- Two workarounds that existed solely to reconcile the old MSRV against
  current stable are gone, not just relaxed: three AVX2 helpers in
  `src/turbo.rs` no longer need an inner `unsafe` block plus
  `#[allow(unused_unsafe)]` (Rust 1.87 made `core::arch` intrinsic calls safe
  from within an equally `#[target_feature]`-gated function; MSRV 1.97 is
  past that), and the `manual_is_multiple_of` clippy allow is gone —
  clippy's MSRV-aware lints now correctly suggest `.is_multiple_of()` in
  `rate_matching.rs`, `transport_block.rs`, and a test in `viterbi.rs`
  (`is_multiple_of` stabilized in 1.87, also now behind the floor), and two
  `if`-let-chains simplify nested `if let Some(...) { if ... { ... } }`
  patterns in `reed_solomon.rs`, `ldpc_pipeline.rs`, and
  `src/bin/ldpc_pipeline_bench.rs`. None of this changes behavior; every
  site was verified equivalent by the existing test suite passing unchanged.

## [0.1.3] — 2026-07-31

### Fixed

- **`AwgnChannel::new` seeded xorshift64 with `seed | 1`, silently colliding
  every `(even, even + 1)` seed pair onto identical noise.** `0` and `1`
  produced the same sequence, as did `2`/`3`, `100`/`101`, and so on for
  every consecutive pair — two channels built with "different" seeds could
  transmit byte-identical noise. The seed is now mixed through SplitMix64
  before becoming the xorshift state, which is both collision-free (the
  mixer is a bijection) and free of the weak diffusion between low-bit-similar
  seeds that a raw `seed | 1` (or a naive `if seed == 0 { 1 }`, which merely
  trades the bug for a new collision against the literal seed `1`) would
  still carry. Found while investigating why a HARQ combining test was
  silently combining a transmission with itself.
- **`DlSchDecoder::decode` dropped the last `2Z` accumulated HARQ LLRs of
  every code block before they reached the LDPC decoder.** The mapping from
  the HARQ circular buffer into the decoder's LLR array computed
  `valid_len = ncb - 2Z`, double-subtracting the puncture width: `ncb`
  (`HarqBuffer`'s circular buffer size) is already `N - 2Z` by construction,
  so the correct value is `ncb` with no further subtraction. The bug never
  panicked or returned an error — it silently zeroed real information the
  receiver had (worth roughly one AVX2 decode iteration's amount of data per
  code block), and specifically weakened incremental-redundancy
  retransmissions at RV 2 and RV 3, whose rate-matching walk reaches into
  exactly the dropped region. A test decoding a real, `AwgnChannel`-corrupted
  codeword confirms an RV0 transmission that fails CRC alone is recovered by
  combining it with an RV3 retransmission, and was verified — by temporarily
  reverting only this fix — to fail under the old code.
- `DlSchDecoder` had no way to report the real (rounded) coded length it
  expects, so a caller could only guess it from the raw `g` constructor
  argument — wrong whenever `g` wasn't already a multiple of
  `Qm * num_code_blocks`. Added `DlSchDecoder::output_bits`, mirroring the
  accessor `DlSchEncoder` already had.

## [0.1.2] — 2026-07-29

### Added

- `QcLdpcDecoder::decode_layered_offset_min_sum_scalar` — the same layered
  offset min-sum decode with the scalar kernel forced on every architecture,
  bypassing the runtime AVX2 probe and the compile-gated NEON path. It exists
  so the scalar fallback can be benchmarked against the vectorized kernels on
  one machine; a unit test asserts the two produce identical hard-decision
  output and identical iteration counts on the same input. Kernel selection is
  resolved once per call, before the iteration and layer loops, so the
  runtime-dispatched entry points are unchanged.
- `LdpcFrame::iterations_used` — the number of layered passes the decoder
  actually consumed for that frame, which may be fewer than the configured
  budget when the syndrome check terminates early.

- `QcLdpcDecoder::decode_layered_offset_min_sum_traced` — the same layered
  offset min-sum decode, with a callback invoked once per completed layered
  pass so a caller can record a real per-iteration convergence trace from a
  single continuous run. Re-invoking the untraced decoder once per iteration
  count cannot reproduce the trajectory, because the extrinsic message buffer
  is reset at the start of every call. The callback is a monomorphized
  generic, so the untraced entry point compiles to the same code as before.

## [0.1.1] — 2026-07-26

### Fixed

- LaTeX in the API documentation now renders on docs.rs. rustdoc has no math
  support of its own, so the `$...$` expressions throughout the module and
  function docs were being served as literal text. A KaTeX header is now
  injected via `--html-in-header`, declared under `[package.metadata.docs.rs]`.
  Documentation-only; no code, API, or behaviour changed.

## [0.1.0] — 2026-07-26

First public release.

### Added

- **Nine FEC cores**, each with encode and decode: Hamming(7,4), extended
  Golay(24,12,8), BCH(255,k,t≤10) over GF(2⁸) with Berlekamp–Massey and Chien
  search, Reed-Solomon erasure coding over GF(256), rate-1/2 K=7 Viterbi
  (hard and soft decision), LTE rate-1/3 Turbo (iterative max-log-MAP),
  5G NR QC-LDPC (layered offset min-sum), Polar SC and CA-SCL, and the
  CRC-24A/B/C, CRC-16/11/6 family.
- **Complete 3GPP TS 38.212 transport-block chain**: CRC attachment, code-block
  segmentation with base-graph selection, QC-LDPC encode/decode over BG1 and
  BG2, rate matching with redundancy versions, HARQ soft combining, and the
  `DlSchEncoder` / `DlSchDecoder` end-to-end pair.
- **IEEE 802.11 Wi-Fi 6/7 LDPC**: all twelve 802.11 Annex R/F parity-check
  shift matrices ($Z \in \{27, 54, 81\}$, $R \in \{1/2, 2/3, 3/4, 5/6\}$),
  decoded through the same LOMS kernel as 5G NR. Shortening and
  puncturing/rate-matching are out of scope for this release and documented as
  such in the module docs.
- **SIMD kernels**: AVX2 paths for the LDPC, Reed-Solomon, Viterbi and Turbo
  inner loops, selected at runtime via `is_x86_feature_detected!`; NEON paths
  on AArch64. Each is proven output-equivalent to its scalar reference by
  seeded randomized tests, and the scalar implementations remain available.
- **Lock-free concurrency**: a cache-line-padded single-producer
  single-consumer ring buffer, a multi-worker LDPC pipeline sized from
  `available_parallelism()`, and optional per-core thread affinity behind the
  `affinity` feature.
- **Typed error handling**: fallible public APIs return `Result<_, FecError>`,
  with `FecError` implementing `std::error::Error`.
- **Test suite**: 293 tests on x86-64 (294 on AArch64), including 14
  known-answer conformance vectors pinned to external ground truth — the
  reveng CRC catalogue, CCSDS Reed-Solomon conventions, and published
  Hamming/Golay/BCH generator and parity-check matrices — plus 31 adversarial
  input tests, 73 doctests, and six libFuzzer targets over the decoder entry
  points.
- **Seven runnable teaching examples**, from a single corrected bit flip to a
  full 5G NR transport block surviving an AWGN channel.
- **Reproducible benchmark suite** (`bench/run_all.sh`) comparing Rust, a
  same-algorithm C++ port, and Python, gated on byte-identical output so a
  divergence fails the run rather than producing a misleading number.

Minimum supported Rust version: 1.85 (Rust 2024 edition).
