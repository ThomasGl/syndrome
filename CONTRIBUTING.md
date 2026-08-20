# Contributing to syndrome

Thanks for your interest. This document describes how to build, test, and submit
changes to the library.

## Toolchain

- **MSRV:** Rust **1.97** (Rust 2024 edition). The repo ships a
  `rust-toolchain.toml` that selects the `stable` channel with `rustfmt` and
  `clippy`, so `rustup` will provision everything on first build.

## Local checks (must pass before a PR)

Run these locally before opening a PR — they cover the core lib/test/lint
gate, but CI's real job list is larger than this (no_std on a real ARM
target, the `capi`/`embedded-demo` standalone packages, MSRV, loom, Miri,
the rustdoc-math checker, a security audit, and a fuzz smoke test — see
`.github/workflows/ci.yml` for the full, current list rather than trusting
a copy of it here, since a stale mirror of that list is worse than none):

```bash
cargo build --release            # builds the library + binaries
cargo test                       # unit + integration + tests/*.rs + doctests, ALL of them (no --lib/--doc split)
cargo clippy --all-targets -- -D warnings   # zero-warning lint gate
cargo fmt --all -- --check       # formatting gate
```

Run `cargo test` as shown — plain, no `--lib`/`--doc`/`--test <name>`
restriction. Splitting it into separate `--lib` and `--doc` invocations
looks equivalent but silently skips every test in `tests/*.rs`, including
`tests/doc_math.rs` (the rustdoc-LaTeX-escaping checker) — exactly the gap
that let a rendering bug reach the published v0.6.0 on crates.io. See
`scripts/pre-publish-check.sh`'s own comment for the full story.

The exact test counts change as the library grows, so they aren't hardcoded
here (a stale number in this file is worse than no number). The one place
that's kept current at every commit is the `Tests` badge at the top of
`README.md` — after running `cargo test` locally, the total in your
terminal output should match that badge. `README.md` §4 ("Test Suite") also
breaks the total down by category (unit / integration / media / reference
vectors / robustness / doctests) if you want to see where a new test should
live.

## Before a release

PR-level checks above are necessary but not sufficient for cutting a
release — run the full gate CI actually enforces, matching the real QEMU
version CI uses for the `embedded-demo` check (a locally-installed QEMU can
give a false pass; see the script for why):

```bash
bash scripts/pre-publish-check.sh
```

Push to `master`, wait for CI to report green on that exact commit, **then**
`cargo publish` — never publish before CI confirms the pushed commit,
since a published crate version is immutable and cannot be corrected, only
superseded by a new one.

## Lint policy

`clippy` is enforced as `-D warnings`, so the build fails on any new lint. A
small set of lints is deliberately allowed crate-wide in the `[lints.clippy]`
table of `Cargo.toml`, each with a rationale. The most important category is
**performance-intentional**: per the constraints in `CLAUDE.md`, hot loops use
explicit index-based, branch-free arithmetic for SoA/SIMD friendliness, so
lints such as `needless_range_loop` and `manual_clamp` are allowed rather than
rewritten. If you touch a hot path, keep that style; if you add a genuinely
new allowed lint, document why in the table.

## Performance constraints

The decoder inner loops and ring buffers must stay **allocation-free**. Avoid
`Box::new`, `Vec::push`, `clone()`, or other heap traffic on hot paths, and
prefer flat `&[T]` / `[T; N]` buffers over nested `Vec<Vec<T>>`. See `CLAUDE.md`
for the full set of non-negotiable constraints.

## Benchmarks

Performance numbers in the docs are produced by running code, never hand-written.

```bash
cargo bench --bench fec_bench          # Criterion micro-benchmarks
bash bench/run_all.sh                  # cross-language RS comparison + charts
```

`bench/run_all.sh` includes a **checksum gate**: it fails loudly if the Rust,
C++, and same-algorithm Python encoders do not produce byte-identical parity,
so no speed claim is plotted unless the implementations agree bit-for-bit. Do
not edit result JSON by hand.

## Commit and PR conventions

- Keep commits focused; use clear, conventional-style prefixes
  (`feat:`, `fix:`, `docs:`, `ci:`, `chore:`, `perf:`, `test:`).
- Update `README.md`, `system_architecture.md`, and `CHANGELOG.md` when adding
  components or changing public APIs.
- New modules land with tests and an example.

## License

By contributing, you agree that your contributions are licensed under the
[MIT License](LICENSE).
