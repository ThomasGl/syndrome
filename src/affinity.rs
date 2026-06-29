#[cfg(feature = "affinity")]
pub fn pin_to_core(core_id: usize) -> Result<(), &'static str> {
    match core_affinity::get_core_ids() {
        Some(cores) => {
            if core_id >= cores.len() {
                return Err("core_id out of range");
            }
            core_affinity::set_for_current(cores[core_id]);
            Ok(())
        }
        None => Err("unable to query cores"),
    }
}

#[cfg(not(feature = "affinity"))]
pub fn pin_to_core(_core_id: usize) -> Result<(), &'static str> {
    Err("affinity feature not enabled")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pin_stub_returns_error_when_disabled() {
        #[cfg(not(feature = "affinity"))]
        assert!(pin_to_core(0).is_err());
    }
}
