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
verified physical board (see that file); it is sized to the real
STM32F405's 128 KiB of *contiguous* SRAM (not the oft-quoted 192 KiB total —
the other 64 KiB, "CCM", sits at a separate, non-contiguous address this
single-region linker script does not reach), which is why it runs under
QEMU unmodified, but adjust it to your actual board's datasheet before
flashing this to real hardware.

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

Real output, from an actual run (2026-08-19, QEMU 8.2.2 — the version this
crate's CI actually installs and runs against; also reproduced against QEMU
6.2.0 for local-dev-loop parity):

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

**QEMU versions can genuinely disagree here, so test against the version
that matters.** This crate's CI once merged clean locally and then failed
in GitHub Actions with a boot-time `HardFault` lockup: `memory.x` declared
176 KiB of RAM (see "Measured size" below for why that number was wrong),
which QEMU 6.2.0 silently tolerated but QEMU 8.2.2 correctly rejected. If
you don't have the exact QEMU version your CI runs available locally, a
disposable container gets an exact match without touching your host:

```bash
docker run --rm -v "$PWD":/demo ubuntu:24.04 bash -c '
  apt-get update -qq && apt-get install -y -qq qemu-system-arm
  qemu-system-arm -M netduinoplus2 -semihosting-config enable=on,target=native \
    -nographic -kernel /demo/target/thumbv7em-none-eabihf/release/syndrome-embedded-demo
'
```

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
| `.bss` | 106,532 (~104.0 KiB) | Zero-initialized RAM: almost entirely the 104 KiB static heap `main.rs` provisions (`HEAP_SIZE`) — see below for where that number comes from. |

**`HEAP_SIZE` is not a guess — it was bisected by actually running under
QEMU, twice.** The first version of this demo shipped with a 64 KiB heap,
chosen without measurement; the first time it was actually run, it panicked
on a real failed allocation partway through decode. Rebuilding and
re-running at several sizes against a (then-undiscovered) incorrectly-sized
176 KiB RAM region found: 64 KiB fails, 96 KiB is the smallest size that
runs to completion, 128 KiB shipped as that floor plus headroom.

That bisection turned out to be built on a wrong assumption: `memory.x`'s
176 KiB RAM claim overran the STM32F405's real 128 KiB of contiguous SRAM
by 48 KiB, which QEMU 6.2.0 didn't enforce but QEMU 8.2.2 (this crate's CI)
correctly does — see "Running under QEMU" above. With `memory.x` corrected
to the real 128 KiB, the old 128 KiB heap doesn't even *link* any more
(`.bss` overflows the region by 36 bytes, caught at build time, not
runtime). Re-bisected against the corrected region using QEMU 8.2.2: 64 KiB
still fails, 96 KiB is still the measured floor (the algorithm's own peak
usage didn't change, only the RAM budget available to hold it did), and
104 KiB ships here as that floor plus roughly 24 KiB of headroom for the
stack — a real number now, not sitting on the floor, but tighter than the
old 128 KiB because there is far less RAM to spare. A tighter number still
would need a peak-allocation profile (an allocator that reports
high-water-mark usage, or real hardware) rather than a bisection against
pass/fail.

Re-run `llvm-size target/thumbv7em-none-eabihf/release/syndrome-embedded-demo`
after any change to confirm these numbers rather than trusting this table —
they will drift as the crate and its dependencies change.
