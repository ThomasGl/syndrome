# glezer_rsv fuzz targets

This is a standard [`cargo-fuzz`](https://github.com/rust-fuzz/cargo-fuzz)
layout (`libfuzzer-sys` harnesses, one `[[bin]]` per target in
`fuzz/Cargo.toml`). It is its own Cargo workspace (`[workspace]` at the
bottom of `fuzz/Cargo.toml`) and is **not** a member of the root
`glezer_rsv` workspace — the root `Cargo.toml` is untouched.

## Targets

| Target               | Exercises                                                          |
|----------------------|---------------------------------------------------------------------|
| `fuzz_bch_decode`    | `BchCode::decode`, `shortened_decode`, `encode` (all t, adversarial k_short/info) |
| `fuzz_rs_decode`     | `ReedSolomon::decode` (adversarial shard patterns/erasures)          |
| `fuzz_polar_sc`      | `PolarDecoder::decode_sc` / `decode_scl` (adversarial LLRs, n/k/list_size) |
| `fuzz_turbo_decode`  | `TurboDecoder::decode` (adversarial LLRs, every supported K)         |
| `fuzz_ldpc_decode`   | `QcLdpcDecoder::decode_layered_offset_min_sum` / `decode_5g`         |
| `fuzz_crc`           | `Crc24::compute` / `attach` / `check` (unmasked >1 bit values)       |

Each target derives its parameters (lengths, code sizes, RV/Qm/t/K/etc.)
directly from the raw fuzzer byte stream, and turns byte pairs into `f32`
LLRs with periodic NaN/+-Inf/zero injection (see the `byte_pair_to_llr`
helper duplicated in each target file).

## Status in this environment: built AND run (not on nightly)

This machine has **no nightly toolchain installed**
(`rustup toolchain list` shows only `stable` and `1.96.0`), so the intended
`cargo +nightly fuzz build` / `cargo +nightly fuzz run <target> -- \
-max_total_time=30` workflow could not be exercised as specified.

However, `libfuzzer-sys` does not itself require nightly to *build and
link* a working libFuzzer binary (only `cargo-fuzz`'s normal invocation
adds `-Z`-gated sanitizer/coverage instrumentation flags, which do need
nightly). So, as a best-effort substitute, every target here was verified
with the installed **stable** toolchain:

1. `cargo check --all-targets` in this directory — succeeds for all 6
   targets (confirms they are structurally correct against the current
   `src/` public API).
2. `cargo build --release` — succeeds and produces a real, runnable
   libFuzzer binary per target (`target/release/fuzz_<name>`).
3. Each of the 6 binaries was then run directly for ~30s
   (`./target/release/fuzz_<name> -max_total_time=30 corpus/fuzz_<name>`)
   against millions of random inputs. **No crashes, panics, or hangs were
   found** in any target during these runs.

**Important caveat**: because this was a plain `cargo build` (not
`cargo +nightly fuzz build`), the binaries were **not** compiled with
SanitizerCoverage instrumentation — libFuzzer prints
`WARNING: no interesting inputs were found so far. Is the code
instrumented for coverage?` for exactly this reason. Practically, this
means these runs behaved as an *uninstrumented random fuzzer* (still a
real execution of millions of adversarial inputs through each harness,
and still capable of catching a panic/abort/OOB if one were hit) rather
than the coverage-guided search a true `cargo +nightly fuzz run` performs.
Treat the "no crashes" result as a positive but weaker signal than a
proper nightly run would give — re-run with real `cargo-fuzz` + nightly
(instructions below) for full confidence, especially before relying on
this as a release gate.

## How to run properly (nightly + cargo-fuzz)

```sh
rustup toolchain install nightly
cargo install cargo-fuzz

cd fuzz
cargo +nightly fuzz build
for t in fuzz_bch_decode fuzz_rs_decode fuzz_polar_sc fuzz_turbo_decode fuzz_ldpc_decode fuzz_crc; do
    cargo +nightly fuzz run "$t" -- -max_total_time=30
done
```

Findings from a `cargo-fuzz` run land in `fuzz/artifacts/<target>/` as
`crash-<hash>` files; reproduce/minimize with:

```sh
cargo +nightly fuzz run <target> fuzz/artifacts/<target>/crash-<hash>
```

## How to reproduce the stable-toolchain best-effort run performed here

```sh
cd fuzz
cargo build --release
mkdir -p corpus/fuzz_crc   # (etc., one per target)
./target/release/fuzz_crc -max_total_time=30 corpus/fuzz_crc
```

Swap `fuzz_crc` for any of the 6 target names above. Any crash found this
way is written to the current directory as `crash-<hash>`; reproduce with
`./target/release/fuzz_crc crash-<hash>`.
