/* GENERIC Cortex-M4F memory layout -- illustrative, NOT a specific verified
 * board, chosen so the demo links and its size is measurable rather than
 * checked against real hardware (this crate has none). Sized to fit inside
 * a real STM32F405's 1 MiB flash / 192 KiB SRAM, which is what this demo is
 * actually run against under QEMU's `netduinoplus2` machine model (see
 * README.md's "Running under QEMU" section) -- so both this generic layout
 * and the real chip it runs on happen to agree here, but that is a
 * convenience for testing, not a claim this was verified on that specific
 * physical board. Adjust FLASH/RAM origin and length to your actual board's
 * datasheet before flashing anything built from this linker script.
 */
MEMORY
{
  FLASH : ORIGIN = 0x08000000, LENGTH = 512K
  RAM : ORIGIN = 0x20000000, LENGTH = 176K
}
