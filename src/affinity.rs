use crate::error::FecError;

/// Pin the calling thread to a specific physical core.
///
/// # Arguments
///
/// * `core_id` - Zero-based index into the OS-reported core list.
///
/// # Errors
///
/// Returns [`FecError::InvalidParam`] if `core_id` is out of range, if the
/// core list could not be queried, or if the `affinity` feature is not
/// enabled at build time. Affinity is a best-effort optimization: callers
/// may safely ignore this error (it only affects cache locality, not
/// correctness).
#[cfg(feature = "affinity")]
pub fn pin_to_core(core_id: usize) -> Result<(), FecError> {
    match core_affinity::get_core_ids() {
        Some(cores) => {
            if core_id >= cores.len() {
                return Err(FecError::InvalidParam("core_id out of range"));
            }
            core_affinity::set_for_current(cores[core_id]);
            Ok(())
        }
        None => Err(FecError::InvalidParam("unable to query cores")),
    }
}

/// Pin the calling thread to a specific physical core (stub; `affinity`
/// feature not enabled).
///
/// # Arguments
///
/// * `_core_id` - Ignored; see [`pin_to_core`] (the `affinity`-feature
///   variant) for the real implementation.
///
/// # Errors
///
/// Always returns [`FecError::InvalidParam`], since the `affinity` feature
/// was not enabled at build time.
#[cfg(not(feature = "affinity"))]
pub fn pin_to_core(_core_id: usize) -> Result<(), FecError> {
    Err(FecError::InvalidParam("affinity feature not enabled"))
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "affinity"))]
    #[test]
    fn pin_stub_returns_error_when_disabled() {
        assert!(super::pin_to_core(0).is_err());
    }

    #[cfg(feature = "affinity")]
    #[test]
    fn pin_to_core_zero_does_not_panic() {
        // Core 0 exists on any machine; pinning either succeeds or the
        // platform reports failure — both are graceful, non-panicking paths.
        let _ = super::pin_to_core(0);
    }
}
