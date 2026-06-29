Systems & Mathematical Architecture: glezer-rsvglezer-rsv is a multi-architecture Forward Error Correction (FEC) study written in pure Rust. It follows the 3GPP Release 15/16 5G New Radio (NR) specification as a learning and benchmark reference, exploring whether memory-safe Rust can match established C++ implementations such as AFF3CT and OpenAirInterface. It is a portfolio/research project, not a production baseband replacement; see "Current Implementation Status" below for what is actually implemented today.1. Mathematical Framework: 5G NR QC-LDPC Decoding5G NR discards legacy Reed-Solomon and Convolutional codes for data channels, mandating Quasi-Cyclic Low-Density Parity-Check (QC-LDPC) codes due to their capacity to approach the Shannon Limit under high-concurrency implementations.An LDPC code is defined by a sparse parity-check matrix $H$ of size $M \times N$. In QC-LDPC, $H$ is structured into blocks of shifted identity matrices of size $Z \times Z$ (the lifting size), dictated by Base Graph 1 (BG1) for large/high-rate transport blocks, or Base Graph 2 (BG2) for smaller blocks.Base Graph Matrix H (Sparse Block Structure)
[ I(p_0,0)   I(p_0,1)   0          ... ]  --> Map to Z x Z sub-matrices
[ 0          I(p_1,1)   I(p_1,2)   ... ]
The Layered Offset Min-Sum (LOMS) AlgorithmTo minimize latency and maximize L1 cache data reuse, glezer-rsv implements the Layered Offset Min-Sum (LOMS) approximation of Belief Propagation. The decoding space is processed in sub-matrix rows (layers).Let $L_{n}$ be the input Log-Likelihood Ratio (LLR) from the channel demodulator. The extrinsic memory space consists of check-to-variable messages $R_{m,n}$ and variable-to-check messages $Q_{m,n}$.For each layer (row block $m$), the updating equations are calculated sequentially:Variable Message Update:$$Q_{m,n}^{(t)} = L_{n}^{(t-1)} - R_{m,n}^{(t-1)}$$Check Message Update (Min-Sum with Offset $\beta$):$$R_{m,n}^{(t)} = \prod_{n' \in \mathcal{N}(m) \setminus n} \text{sign}\left(Q_{m,n'}^{(t)}\right) \times \max\left( \min_{n' \in \mathcal{N}(m) \setminus n} \left|Q_{m,n'}^{(t)}\right| - \beta, \; 0 \right)$$A Posteriori LLR Update:$$L_{n}^{(t)} = Q_{m,n}^{(t)} + R_{m,n}^{(t)}$$2. Multi-Architecture Hardware Compilation Strategyglezer-rsv abstracts target compilation flags to enforce deterministic data mapping depending on the profile tier:Hardware TierArchitectureVectorization PathwayMemory ConstraintsEdge / IoT MCU (Arduino, STM32)ARM Cortex-Mcore::arch::arm (no_std)Zero heap allocations. Fixed stack arrays.SBC Gateway (Raspberry Pi 4/5)ARM Cortex-A (aarch64)Neon SIMD (std::simd)Thread affinity pinned to performance cores.Desktop / Cloud RAN (AMD, Intel)x86_64AVX2 / AVX-512Ring-buffered NUMA-node local allocations.Memory Layout Optimization for L1/L2 Cache EfficiencyModern CPU memory hierarchies require data layout design that limits cache invalidation:Pre-computed Offset Offloading: To prevent indexing calculations in real-time loops, the column indices of the non-zero sub-matrices of the base graphs are loaded into static, pre-compiled lookup arrays.Struct of Arrays (SoA) over Array of Structs (AoS): LLR metrics and message structures are partitioned into parallel vectors so that contiguous float slices can be loaded directly into AVX-512 or NEON SIMD registers using zero-overhead pointer offsets.Avoidance of Pointer Chasing: Memory representation is bound to sequential 1D vector arrays ([f32; N]). Pointer indirection (such as nested vectors Vec<Vec<T>> or heap-allocated boxes Box<T>) is completely prohibited on computational hot-paths.3. High-Throughput Job Queue & Synchronization RuntimeIn order to scale linearly with the number of CPU physical cores, glezer-rsv operates an asynchronous, lock-free task scheduler. This scheduler mimics industrial C++ libraries (like aff3ct-core) while avoiding common synchronization bottlenecks.Incoming LLR Stream ---> [ Atomic Ring Buffer (SPSC) ] ---> Thread Worker 0 (Core 0 Affinity)
                        [ Atomic Ring Buffer (SPSC) ] ---> Thread Worker 1 (Core 1 Affinity)
                        [ Atomic Ring Buffer (SPSC) ] ---> Thread Worker 2 (Core 2 Affinity)
Lock-Free Atomic RingsThread workers query work using lock-free Single-Producer Single-Consumer (SPSC) rings. Task distribution relies entirely on atomic increments (AtomicUsize) with Ordering::Relaxed or Ordering::Acquire/Ordering::Release state changes. This minimizes kernel context switching and eliminates mutex lock contention overhead under high-volume streaming.Thread Pinning & Hardware AffinityIn high-throughput baseband deployments (gNodeB / RAN), context switches are unacceptable. glezer-rsv integrates platform-specific scheduler controls (core_affinity crate) to pin processing loops to distinct hardware physical threads. This maximizes instruction cache hit rates and keeps processing local to physical CPU sockets (NUMA nodes).4. Benchmarking & Cross-Language Analytics

The project ships a reproducible benchmark suite that measures Reed-Solomon encode throughput in Rust, same-algorithm C++, and Python. All numbers come from actually running the code — never hand-written.

Current Implementation Status (99 tests: 64 unit + 7 integration + 28 doctests)

The following components are real and tested:

5G NR TS 38.212 transport-block chain (complete):
- CRC-24A/B/C, CRC-16/11/6 (src/crc.rs) — all §5.1 generator polynomials; bit-serial LFSR
- Code block segmentation + BG selection (src/segmentation.rs) — §5.2.2; K', Z, n_filler
- QC-LDPC encode_5g / decode_5g (src/qc_ldpc.rs) — filler +∞ LLR, 2-col puncturing
- Rate matching + HARQ (src/rate_matching.rs, src/harq.rs) — §5.4.2.1-2; RV k0; soft combining
- DlSchEncoder / DlSchDecoder (src/transport_block.rs) — full TB chain end-to-end

FEC cores:
- Hamming(7,4) encode/decode (src/hamming.rs)
- Reed-Solomon erasure encoder/decoder (src/reed_solomon.rs) — GF(256) 0x11D; AVX2 VPSHUFB path
- Rate-1/2 K=7 Viterbi decoder (src/viterbi.rs) — hard (Hamming ACS) + soft (max-log-MAP)
- Polar codes SC + CA-SCL (src/polar.rs) — 3GPP reliability sequence; list decode

QC-LDPC kernel:
- BG1 (Z=384, R≈1/3) and BG2 (Z=128) from 3GPP TS 38.212 — 7 integration tests
- Scalar z-inner path: ~65 Melem/s (parity with C++ scalar ~66 Melem/s)
- AVX2 kernel (runtime-detected): ~119 Melem/s (1.80× over C++ scalar)
- NEON kernel (compile-gated, aarch64)
- Syndrome-check early termination (exits before max_iters if all parity checks pass)

Concurrency:
- SPSC lock-free ring buffer (src/spsc_queue.rs) — ~1.1 ns push+pop
- LdpcPipeline multi-worker (src/ldpc_pipeline.rs) — per-core affinity wiring
- i8 LLR quantization (src/quantize.rs) — scale + clamp to [−127, 127]

The following are aspirational / deferred:
- i8 scalar LOMS decoder path (next: extend qc_ldpc.rs with &mut [i8])
- AVX2/AVX-512 i8 LOMS kernel (32-wide _mm256_min_epi8, 4× SIMD density over f32)
- no_std cfg split (core kernel has no std dep; feature gate pending)
- core::simd portable-SIMD path (requires nightly Rust)
- AFF3CT end-to-end comparison on the LDPC path (Phase 2)

Result Schema (bench/results/<lang>.json)

One JSON record per (lang, impl, shard_len):

  {"lang":"rust","impl":"encode_into","shard_len":1024,"data_shards":10,
   "parity_shards":4,"payload_bytes":10240,"ns_per_iter":1465.0,"mib_per_s":682.3}

Fields:
- lang: "rust" | "cpp" | "python_same_algo" | "python_reedsolo"
- impl: encoder variant name
- shard_len: bytes per data shard
- payload_bytes: data_shards * shard_len
- ns_per_iter: mean nanoseconds per encode call (wall-clock, warm)
- mib_per_s: payload_bytes / ns_per_iter * 1e9 / 1048576

All numbers are produced by bench/run_all.sh, which runs the Rust, C++, and Python drivers and validates byte-identical parity output via a checksum gate before writing any results. Numbers are never hand-written.
