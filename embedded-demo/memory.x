/* GENERIC Cortex-M4F memory layout -- illustrative, NOT a specific verified
 * board, chosen so the demo links and its size is measurable rather than
 * checked against real hardware (this crate has none). Sized to fit inside
 * a real STM32F405's 1 MiB flash, which is what this demo is actually run
 * against under QEMU's `netduinoplus2` machine model (see README.md's
 * "Running under QEMU" section) -- so both this generic layout and the real
 * chip it runs on happen to agree here, but that is a convenience for
 * testing, not a claim this was verified on that specific physical board.
 * Adjust FLASH/RAM origin and length to your actual board's datasheet
 * before flashing anything built from this linker script.
 *
 * RAM is 128K, not the STM32F405's oft-quoted "192 KiB SRAM": only 128 KiB
 * of that (112 KiB "SRAM1" + 16 KiB "SRAM2") is contiguous at 0x20000000.
 * The remaining 64 KiB ("CCM", core-coupled memory) lives at a separate,
 * non-contiguous address (0x10000000) and is not reachable through this
 * single-region linker script. An earlier version of this file declared
 * 176K here, which QEMU 6.2 silently tolerated but QEMU 8.2.2 (what CI
 * actually runs) correctly rejects at boot with a HardFault lockup -- see
 * `src/main.rs`'s module doc for the full story.
 */
MEMORY
{
  FLASH : ORIGIN = 0x08000000, LENGTH = 512K
  RAM : ORIGIN = 0x20000000, LENGTH = 128K
}
