#!/usr/bin/env bash
# scripts/pre-publish-check.sh — the exact gate to run before every push to
# master and every `cargo publish`. Exists because v0.6.0 was published with
# a rustdoc-breaking LaTeX escaping bug in src/ccsds_rs.rs that
# tests/doc_math.rs would have caught -- it was never run, because the
# pre-release check that day ran `cargo test --lib` and `cargo test --doc`
# separately instead of a plain `cargo test`, which is the only invocation
# that also runs tests/*.rs (doc_math.rs among them). That mistake reached
# crates.io, where it is permanent: crate versions are immutable, so the
# only fix was a same-day v0.6.1 patch release. This script exists so no
# future release skips a check by running an ad hoc subset of commands
# instead of the real thing.
#
# Usage:  bash scripts/pre-publish-check.sh
# Must be run from the repo root (syndrome/). Requires: stable Rust with the
# thumbv7em-none-eabihf target installed, Docker (for the embedded-demo QEMU
# check, which must run under the same QEMU version CI uses -- see that
# section below for why a locally-installed QEMU is not trustworthy here).
#
# Exit codes:
#   0 — every check passed, safe to push/publish
#   1 — a check failed; do not push or publish until this is clean

set -euo pipefail

step() { echo; echo "=== $1 ==="; }

step "cargo fmt --check"
cargo fmt --check

step "cargo test (the FULL suite -- unit, doc, and every tests/*.rs, not a hand-picked subset)"
cargo test

step "clippy (default features)"
cargo clippy --all-targets -- -D warnings

step "clippy (affinity,bench-export)"
cargo clippy --all-targets --features affinity,bench-export -- -D warnings

step "no_std check + clippy (native host)"
cargo clippy --no-default-features --features no_std --lib -- -D warnings

step "no_std check (real thumbv7em-none-eabihf cross-compile)"
cargo check --no-default-features --features no_std --target thumbv7em-none-eabihf --lib

step "capi package: test + clippy"
(cd capi && cargo test && cargo clippy --all-targets -- -D warnings)

step "embedded-demo package: build + clippy"
(cd embedded-demo && cargo build --release && cargo clippy -- -D warnings)

step "embedded-demo: real QEMU execution, matching CI's exact QEMU version"
# A locally-installed qemu-system-arm is NOT sufficient here: this crate's
# CI installs whatever ubuntu-latest's `apt-get install qemu-system-arm`
# resolves to (QEMU 8.2.2 as of this writing), and QEMU versions genuinely
# disagree on enforcement of the emulated board's memory-region boundaries
# -- QEMU 6.2 silently tolerated a 48 KiB RAM-region overclaim in memory.x
# that QEMU 8.2.2 correctly rejects with a boot-time HardFault. A disposable
# container pinned to the same base image CI uses (ubuntu:24.04) sidesteps
# that gap entirely, and is what this step runs.
if command -v docker >/dev/null 2>&1; then
  docker run --rm -v "$PWD/embedded-demo":/demo ubuntu:24.04 bash -c '
    set -e
    apt-get update -qq >/dev/null 2>&1
    apt-get install -y -qq qemu-system-arm >/dev/null 2>&1
    timeout 30 qemu-system-arm -M netduinoplus2 \
      -semihosting-config enable=on,target=native -nographic \
      -kernel /demo/target/thumbv7em-none-eabihf/release/syndrome-embedded-demo \
      | tee /tmp/qemu-output.log
    grep -q "RESULT: PASS" /tmp/qemu-output.log
  '
else
  echo "docker not available -- SKIPPING the QEMU execution check." >&2
  echo "This means embedded-demo has only been link-checked, not run. Do not" >&2
  echo "treat that as equivalent to a real pass; install Docker or run this" >&2
  echo "step's container command manually before trusting a release." >&2
fi

step "packaged-tarball docs build (what docs.rs actually builds, not the git checkout)"
PKG_TMP=$(mktemp -d)
trap 'rm -rf "$PKG_TMP"' EXIT
cargo package --allow-dirty --no-verify >/dev/null
PKG_VERSION=$(grep -m1 '^version' Cargo.toml | sed -E 's/.*"(.*)".*/\1/')
tar -xzf "target/package/syndrome-${PKG_VERSION}.crate" -C "$PKG_TMP"
(cd "$PKG_TMP/syndrome-${PKG_VERSION}" && RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features)

echo
echo "=== ALL CHECKS PASSED ==="
echo "Safe to push to master. Only after CI is green on that exact pushed"
echo "commit -- never before -- is it safe to 'cargo publish'."
