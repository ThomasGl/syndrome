# Contributing to syndrome

Thanks for your interest. This document describes how to build, test, and submit
changes to the library.

## Toolchain

- **MSRV:** Rust **1.85** (Rust 2024 edition). The repo ships a
  `rust-toolchain.toml` that selects the `stable` channel with `rustfmt` and
  `clippy`, so `rustup` will provision everything on first build.

## Local checks (must pass before a PR)

CI runs exactly these; run them locally to get a green review:

```bash
cargo build --release            # builds the library + binaries
cargo test --all                 # 81 unit + 7 integration + 4 media tests
cargo test --doc                 # 43 doctests
cargo clippy --all-targets -- -D warnings   # zero-warning lint gate
cargo fmt --all -- --check       # formatting gate
```

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
