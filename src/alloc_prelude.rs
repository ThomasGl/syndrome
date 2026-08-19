//! `Vec`/`Box`/`vec!`/`Rc` re-exported from `alloc` (under the `no_std`
//! feature) or `std` (otherwise), so the rest of the crate can write one
//! `use crate::alloc_prelude::*;` per file instead of two parallel import
//! lists. `String`/`format!` are not re-exported: nothing outside test code
//! in this crate's algorithms needs them, so they are left out rather than
//! re-exported unused (add them here, to both halves, the day something
//! genuinely needs them).
//!
//! Under `std`, the standard prelude already brings `Vec`/`Box`/`vec!` into
//! scope for free, so this module is redundant but harmless there (mirrors
//! [`crate::sync_shim`]'s std/loom split for the same reason: one name to
//! import, regardless of which half of the `cfg` is active).

#[cfg(feature = "no_std")]
pub(crate) use alloc::{boxed::Box, rc::Rc, vec, vec::Vec};

#[cfg(not(feature = "no_std"))]
pub(crate) use std::{boxed::Box, rc::Rc, vec, vec::Vec};
