/* GENERIC Cortex-M4F memory layout -- illustrative, NOT a specific verified
 * board. 512 KiB flash / 128 KiB RAM is a common mid-range STM32F4-class
 * shape, chosen so the demo links and its size is measurable, not because
 * it was checked against real hardware (this crate has none to check
 * against). Adjust FLASH/RAM origin and length to your actual board's
 * datasheet before flashing anything built from this linker script.
 */
MEMORY
{
  FLASH : ORIGIN = 0x08000000, LENGTH = 512K
  RAM : ORIGIN = 0x20000000, LENGTH = 128K
}
