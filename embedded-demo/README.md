# syndrome-embedded-demo

Bare-metal Cortex-M4F firmware demo of `syndrome`'s `no_std` QC-LDPC path,
proving that feature produces a real, linkable firmware image rather than
just passing `cargo check`.

## What this proves, and what it does not

**Proves:** `syndrome` compiled with `default-features = false, features =
["no_std"]` links successfully into a real ARM EABI5 ELF for
`thumbv7em-none-eabihf`, alongside `cortex-m-rt` (runtime/vector table),
`panic-halt` (panic handler), and `embedded-alloc` (a `#[global_allocator]`
backed by a static byte array — the "microcontroller with a working
allocator" target the `no_std` feature's Cargo.toml doc comment describes,
not a fully allocation-free build).

**Does not prove:** that it runs correctly, or how fast. There is no
hardware and no cycle-accurate simulator (QEMU, `probe-rs`, ...) available
in the environment this was built in, and this crate's own documentation
policy is to publish no benchmark number that was not actually measured —
so none is claimed here. `memory.x`'s FLASH/RAM layout is generic and
illustrative (a common mid-range Cortex-M4 shape), not a specific verified
board; if you flash this to real hardware, adjust it to that board's
datasheet first.

## What it does

`src/main.rs` encodes a BG2 (Z=128) codeword, quantizes it to the crate's
fixed-point `i8` message format (`QuantParams::default()`), and decodes it
back with the layered offset min-sum kernel over a noiseless channel — the
same algorithm and fixed-point format
`tests/ldpc_int8_quantization_loss.rs` measures in the `std` build. On
success it parks in `wfi()`; on failure (which would mean a real bug, since
the channel is noiseless) it hits a breakpoint instead of spinning silently.

## Building

```bash
cd embedded-demo
cargo build --release
```

Produces `target/thumbv7em-none-eabihf/release/syndrome-embedded-demo`, a
real ELF (`file` reports `ELF 32-bit LSB executable, ARM, EABI5`).

## Measured size

Measured 2026-08-19 with `llvm-size` on the `--release` build described
above (`opt-level = "z"`, LTO on, one codegen unit — see this package's
`Cargo.toml`):

| Section | Bytes | What it is |
|---|---|---|
| `.text` | 17,312 (~16.9 KiB) | Code: the QC-LDPC encoder/decoder, `cortex-m-rt`'s runtime, the panic handler, and the allocator — flashed to FLASH. |
| `.data` | 0 | No initialized-nonzero globals. |
| `.bss` | 65,564 (~64.0 KiB) | Zero-initialized RAM: almost entirely the 64 KiB static heap `main.rs` provisions (`HEAP_SIZE`), not a measured minimum — see below. |

**The 64 KiB heap is a deliberately round, generous choice, not a computed
or measured minimum.** No profiling of actual peak allocation was done
(there is nothing to run it on); shrinking `HEAP_SIZE` and confirming the
demo still links is not the same claim as confirming it still has enough
heap at runtime, so that was not attempted here. A real port should profile
actual peak allocation on target hardware (or with an allocator that reports
high-water-mark usage) rather than trust this number.

Re-run `llvm-size target/thumbv7em-none-eabihf/release/syndrome-embedded-demo`
after any change to confirm these numbers rather than trusting this table —
they will drift as the crate and its dependencies change.
