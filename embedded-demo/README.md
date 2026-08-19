# syndrome-embedded-demo

Bare-metal Cortex-M4F firmware demo of `syndrome`'s `no_std` QC-LDPC path —
built, linked, and **actually run** under QEMU, not just checked.

## What this proves, and what it does not

**Proves:** `syndrome` compiled with `default-features = false, features =
["no_std"]` links into a real ARM EABI5 ELF for `thumbv7em-none-eabihf`,
alongside `cortex-m-rt` (runtime/vector table), `panic-semihosting` (panic
handler), `cortex-m-semihosting` (host I/O), and `embedded-alloc` (a
`#[global_allocator]` backed by a static byte array — the "microcontroller
with a working allocator" target the `no_std` feature's Cargo.toml doc
comment describes, not a fully allocation-free build). And, run under
QEMU's `netduinoplus2` (Cortex-M4F) machine model, that it **actually
encodes, quantizes, and decodes a real BG2 (Z=128) codeword correctly** —
see "Running under QEMU" below for the exact command and its real output.

**Does not prove:** hardware timing. QEMU's Cortex-M core runs under TCG
(dynamic binary translation), not a cycle-accurate model of any specific
silicon's pipeline, flash wait-states, or cache behavior. This demo tried
reporting a DWT cycle-counter delta around each phase and found QEMU's
`netduinoplus2` model doesn't implement the DWT cycle counter at all —
every reading came back exactly zero, which is a gap in QEMU's own
peripheral emulation, not a bug here. Publishing "0 cycles" would be a
worse lie than publishing nothing, so no timing number is claimed. A real
measurement needs actual hardware or a simulator that implements DWT,
neither available in the environment this was built in (see below).
`memory.x`'s FLASH/RAM layout is generic and illustrative, not a specific
verified physical board (see that file); it happens to fit inside
`netduinoplus2`'s real STM32F405 capacity, which is why it runs under QEMU
unmodified, but adjust it to your actual board's datasheet before flashing
this to real hardware.

## What it does

`src/main.rs` encodes a BG2 (Z=128) codeword, quantizes it to the crate's
fixed-point `i8` message format (`QuantParams::default()`), and decodes it
back with the layered offset min-sum kernel over a noiseless channel — the
same algorithm and fixed-point format
`tests/ldpc_int8_quantization_loss.rs` measures in the `std` build. It
reports `k`/`n`/iterations used and PASS/FAIL over ARM semihosting, then
exits (under an emulator/debugger that implements semihosting's exit
request — see below) or parks in `wfi()` (on real hardware, which has no
host to exit to).

## Building

```bash
cd embedded-demo
cargo build --release
```

Produces `target/thumbv7em-none-eabihf/release/syndrome-embedded-demo`, a
real ELF (`file` reports `ELF 32-bit LSB executable, ARM, EABI5`).

## Running under QEMU

This needs `qemu-system-arm`. If it's not already on your `PATH`, install
it the normal way for your system (`apt install qemu-system-arm`,
`brew install qemu`, ...). Then:

```bash
qemu-system-arm \
  -M netduinoplus2 \
  -semihosting-config enable=on,target=native \
  -nographic \
  -kernel target/thumbv7em-none-eabihf/release/syndrome-embedded-demo
```

Real output, from an actual run (2026-08-19, QEMU 6.2.0):

```
syndrome-embedded-demo: bg=BG2 z=128 k=1280 n=6656 iterations_used=1
RESULT: PASS -- decoded info bits matched the encoded input exactly
```

QEMU exits with status 0 (the firmware's own `debug::exit(EXIT_SUCCESS)`
semihosting call, not a timeout or a crash).

`netduinoplus2` (an STM32F405, real Cortex-M4F) was picked because it's one
of the machine models this exact QEMU build actually ships (`qemu-system-arm
-machine help` to see the full list on yours); any Cortex-M4-with-FPU model
QEMU supports should work equivalently.

**No sudo/root in your environment either?** `qemu-system-arm` and its
shared-library dependencies can be fetched and run from a normal user
account with no system install, using `apt-get download` (fetches a
package to the current directory without installing it) plus manual
extraction:

```bash
mkdir -p qemu-local/root && cd qemu-local
DEPS=$(apt-cache depends --recurse --no-recommends --no-suggests \
  --no-conflicts --no-breaks --no-replaces --no-enhances qemu-system-arm \
  | grep '^\w')
for pkg in $DEPS; do apt-get download "$pkg"; done
for f in *.deb; do dpkg-deb -x "$f" root; done
export LD_LIBRARY_PATH="$PWD/root/usr/lib/x86_64-linux-gnu:$PWD/root/lib/x86_64-linux-gnu:$PWD/root/usr/lib"
./root/usr/bin/qemu-system-arm --version   # should print a real version -- no sudo used anywhere
```

(Put the `for pkg in $DEPS` loop in a script file and run that rather than
pasting it into an interactive shell — a long inline loop making 100+
`apt-get download` calls in a row is exactly the kind of thing shells
mangle silently.) This is exactly how the run above was produced.

## Measured size

Measured 2026-08-19 with `llvm-size` on the `--release` build described
above (`opt-level = "z"`, LTO on, one codegen unit — see this package's
`Cargo.toml`):

| Section | Bytes | What it is |
|---|---|---|
| `.text` | 35,288 (~34.5 KiB) | Code: the QC-LDPC encoder/decoder, `cortex-m-rt`'s runtime, the panic and semihosting handlers, and the allocator — flashed to FLASH. |
| `.data` | 0 | No initialized-nonzero globals. |
| `.bss` | 131,108 (~128.0 KiB) | Zero-initialized RAM: almost entirely the 128 KiB static heap `main.rs` provisions (`HEAP_SIZE`) — see below for where that number comes from. |

**`HEAP_SIZE` is not a guess — it was bisected by actually running under
QEMU.** The first version of this demo shipped with a 64 KiB heap, chosen
without measurement; the first time it was actually run (see "Running
under QEMU" above), it panicked on a real failed allocation partway through
decode. Rebuilding and re-running at several sizes found: 64 KiB fails,
96 KiB is the smallest size that runs to completion and reports PASS,
128 KiB (what ships here) is that measured floor plus headroom, not the
floor itself. A tighter number would need a peak-allocation profile (an
allocator that reports high-water-mark usage, or real hardware) rather
than a bisection against pass/fail — not attempted here, since 128 KiB
already comfortably fits `netduinoplus2`'s (and most STM32F4-class chips')
real SRAM.

Re-run `llvm-size target/thumbv7em-none-eabihf/release/syndrome-embedded-demo`
after any change to confirm these numbers rather than trusting this table —
they will drift as the crate and its dependencies change.
