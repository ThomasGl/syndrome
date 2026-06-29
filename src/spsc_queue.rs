use core::cell::UnsafeCell;
use core::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Single-producer single-consumer lock-free ring buffer.
///
/// This queue is fully preallocated at construction time. The hot path uses
/// no heap allocations and relies only on atomic pointer arithmetic.
#[repr(align(64))]
pub struct SpscRing<T: Copy, const N: usize> {
    buffer: UnsafeCell<[MaybeUninit<T>; N]>,
    head: AtomicUsize,
    tail: AtomicUsize,
    mask: usize,
}

unsafe impl<T: Copy, const N: usize> Sync for SpscRing<T, N> {}

impl<T: Copy, const N: usize> Default for SpscRing<T, N> {
    /// Equivalent to [`SpscRing::new`]; panics if `N` is not a power of two.
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy, const N: usize> SpscRing<T, N> {
    /// Creates a new ring with capacity `N`. `N` must be a power of two.
    pub fn new() -> Self {
        assert!(N.is_power_of_two(), "capacity must be a power of two");
        let buffer = unsafe { MaybeUninit::<[MaybeUninit<T>; N]>::uninit().assume_init() };
        SpscRing {
            buffer: UnsafeCell::new(buffer),
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            mask: N - 1,
        }
    }

    /// Returns `true` when the queue currently contains no elements.
    pub fn is_empty(&self) -> bool {
        self.head.load(Ordering::Acquire) == self.tail.load(Ordering::Acquire)
    }

    /// Returns `true` when the queue is full.
    pub fn is_full(&self) -> bool {
        self.head
            .load(Ordering::Acquire)
            .wrapping_sub(self.tail.load(Ordering::Acquire))
            == N
    }

    /// Attempt to push an item into the ring. Returns the value back if the
    /// queue is currently full.
    pub fn try_push(&self, item: T) -> Result<(), T> {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        if head.wrapping_sub(tail) == N {
            return Err(item);
        }

        let idx = head & self.mask;
        unsafe { (*self.buffer.get())[idx].write(item) };
        self.head.store(head.wrapping_add(1), Ordering::Release);
        Ok(())
    }

    /// Attempt to pop an item from the ring. Returns `None` when empty.
    pub fn try_pop(&self) -> Option<T> {
        let tail = self.tail.load(Ordering::Acquire);
        let head = self.head.load(Ordering::Acquire);
        if head == tail {
            return None;
        }

        let idx = tail & self.mask;
        let item = unsafe { (*self.buffer.get())[idx].assume_init_read() };
        self.tail.store(tail.wrapping_add(1), Ordering::Release);
        Some(item)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spsc_ring_push_and_pop() {
        let queue = SpscRing::<u32, 8>::new();
        assert!(queue.is_empty());
        queue.try_push(42).unwrap();
        assert!(!queue.is_empty());
        assert_eq!(queue.try_pop(), Some(42));
        assert!(queue.is_empty());
    }

    #[test]
    fn spsc_ring_full_behavior() {
        let queue = SpscRing::<u32, 4>::new();
        assert!(queue.try_push(1).is_ok());
        assert!(queue.try_push(2).is_ok());
        assert!(queue.try_push(3).is_ok());
        assert!(queue.try_push(4).is_ok());
        assert!(queue.try_push(5).is_err());
        assert_eq!(queue.try_pop(), Some(1));
        assert!(queue.try_push(5).is_ok());
        assert_eq!(queue.try_pop(), Some(2));
    }
}
