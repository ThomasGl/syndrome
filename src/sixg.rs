//! 6G NR (IMT-2030) FEC research module.
//!
//! This module documents confirmed 6G research directions and speculative
//! proposals as of 2024, based on ITU-R IMT-2030 framework documents and
//! 3GPP Release 18/19 research items.
//!
//! # Standards Status
//!
//! | Item                        | Status                              |
//! |-----------------------------|-------------------------------------|
//! | ITU-R IMT-2030 framework    | Published (ITU-R M.2160, Nov 2023)  |
//! | 3GPP 5G Advanced (Rel-18)   | Completed (June 2023)               |
//! | 3GPP 6G study (Rel-19+)     | Study phase, started 2024           |
//! | LDPC as 6G data channel     | Confirmed research direction        |
//! | 4096-QAM                    | Confirmed research direction        |
//! | AI/ML-integrated decoding   | Confirmed research direction        |
//! | Semantic communication      | Confirmed research direction        |
//! | 16384-QAM                   | **Speculative** — no consensus      |
//! | Extended BG3 (new BG)       | **Speculative** — proposed in papers|
//!
//! # IMT-2030 Peak Performance Targets
//!
//! $$R_{\text{peak}} > 1 \text{ Tbps}, \quad \tau < 0.1 \text{ ms},
//!   \quad P_e < 10^{-7}$$
//!
//! where $R_{\text{peak}}$ is the peak data rate, $\tau$ is the user-plane
//! latency, and $P_e$ is the block error rate (reliability target).
//!
//! # Data Channel FEC Outlook
//!
//! 6G research strongly indicates LDPC will remain the data channel code,
//! extended to longer block lengths.  The 5G NR maximum transport block is
//! approximately 1.28 Mbits (BG1, $Z = 384$, maximum segments).  6G research
//! papers propose up to ~8 Mbits per TB via extended lifting factors or a new
//! "BG3" designed for sub-THz path loss compensation.  No 3GPP specification
//! for BG3 exists as of 2024.

/// 6G NR service profile, corresponding to IMT-2030 use case categories.
///
/// These map to the six IMT-2030 usage scenarios defined in ITU-R M.2160-0
/// (November 2023).  Each profile drives distinct code-rate and modulation
/// constraints in adaptive link design.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SixgProfile {
    /// Enhanced Mobile Broadband — highest throughput, benign channel.
    ///
    /// IMT-2030 target: $> 1$ Tbps peak, $100$ Gbps user experience.
    EnhancedMBB,

    /// Ultra-Reliable Low-Latency Communication.
    ///
    /// IMT-2030 target: $P_e < 10^{-7}$, $\tau < 0.1$ ms.
    /// Achieved via low code rate and simple modulation.
    URLLC,

    /// Massive Machine-Type Communications — IoT density.
    ///
    /// IMT-2030 target: $10^7$ devices/km².  Low-rate, low-power codes.
    MassiveMTC,

    /// Integrated Sensing and Communication (ISAC).
    ///
    /// Dual-function waveforms for radar sensing and data delivery.
    /// Confirmed IMT-2030 scenario; waveform design is an active research area.
    IntegratedSensing,

    /// AI-Native Communication.
    ///
    /// **Research direction** (not yet standardized): neural network–based
    /// end-to-end transceivers and AI-assisted channel estimation/decoding.
    /// 3GPP Rel-18 initiated the first AI/ML RAN study item; full integration
    /// is expected in the 6G standardization window (Rel-20+).
    AIComm,

    /// Semantic / Goal-Oriented Communication.
    ///
    /// **Research direction** (not yet standardized): transmit only task-relevant
    /// information rather than raw bits.  No PHY specification exists as of 2024;
    /// the concept is studied in 3GPP Rel-19 research items and ITU-T FG-AN.
    Semantic,
}

/// Modulation order supported in 6G NR research.
///
/// Up to 4096-QAM ($Q_m = 12$) is a confirmed 6G research direction, appearing
/// in 3GPP Rel-18 NR-NTN and IMT-2030 workshop papers.  16384-QAM is
/// speculative and not included here.
///
/// The relationship between modulation order and spectral efficiency (in an ideal
/// AWGN channel) is:
/// $$\eta = Q_m \cdot R \quad [\text{bits/channel use}]$$
/// where $R$ is the code rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SixgModulation {
    /// BPSK — $Q_m = 1$ bit/symbol.  Used in URLLC and fallback.
    Bpsk,
    /// QPSK — $Q_m = 2$ bits/symbol.
    Qpsk,
    /// 16-QAM — $Q_m = 4$ bits/symbol.
    Qam16,
    /// 64-QAM — $Q_m = 6$ bits/symbol.
    Qam64,
    /// 256-QAM — $Q_m = 8$ bits/symbol.  Already in 5G NR.
    Qam256,
    /// 1024-QAM — $Q_m = 10$ bits/symbol.  Already in Wi-Fi 6/7.
    Qam1024,
    /// 4096-QAM — $Q_m = 12$ bits/symbol.
    ///
    /// **Confirmed 6G research direction** (ITU-R IMT-2030 workshop, 2023;
    /// 3GPP Rel-19 study items).  Requires extremely high receive SNR
    /// ($\gtrsim 40$ dB) and near-perfect channel estimation, realistically
    /// feasible only in D2D or indoor/wired near-field scenarios.
    Qam4096,
}

impl SixgModulation {
    /// Number of coded bits per QAM symbol ($Q_m$).
    ///
    /// # Returns
    ///
    /// $Q_m \in \{1, 2, 4, 6, 8, 10, 12\}$.
    ///
    /// # Examples
    ///
    /// ```
    /// use syndrome::sixg::SixgModulation;
    ///
    /// assert_eq!(SixgModulation::Bpsk.bits_per_symbol(),   1);
    /// assert_eq!(SixgModulation::Qam4096.bits_per_symbol(), 12);
    /// ```
    pub fn bits_per_symbol(&self) -> u8 {
        match self {
            Self::Bpsk => 1,
            Self::Qpsk => 2,
            Self::Qam16 => 4,
            Self::Qam64 => 6,
            Self::Qam256 => 8,
            Self::Qam1024 => 10,
            Self::Qam4096 => 12,
        }
    }

    /// Human-readable modulation label.
    ///
    /// # Returns
    ///
    /// A static string such as `"BPSK"` or `"4096-QAM"`.
    ///
    /// # Examples
    ///
    /// ```
    /// use syndrome::sixg::SixgModulation;
    ///
    /// assert_eq!(SixgModulation::Qam4096.label(), "4096-QAM");
    /// assert_eq!(SixgModulation::Qpsk.label(),    "QPSK");
    /// ```
    pub fn label(&self) -> &'static str {
        match self {
            Self::Bpsk => "BPSK",
            Self::Qpsk => "QPSK",
            Self::Qam16 => "16-QAM",
            Self::Qam64 => "64-QAM",
            Self::Qam256 => "256-QAM",
            Self::Qam1024 => "1024-QAM",
            Self::Qam4096 => "4096-QAM",
        }
    }
}

/// Transport block parameters for a 6G NR transmission.
///
/// Encodes the link configuration for one scheduling grant under the
/// IMT-2030 framework.  Fields without a finalized 3GPP specification are
/// marked with "research direction" notes.
///
/// # Examples
///
/// ```
/// use syndrome::sixg::{SixgTbParams, SixgProfile, SixgModulation, sixg_profile_target_rate};
///
/// let rate = sixg_profile_target_rate(&SixgProfile::EnhancedMBB);
/// let params = SixgTbParams {
///     profile:      SixgProfile::EnhancedMBB,
///     tb_size_bits: 2_000_000,
///     modulation:   SixgModulation::Qam4096,
///     target_rate:  rate,
///     ai_assisted:  false,
/// };
/// assert!((params.target_rate - 0.89).abs() < 1e-6);
/// ```
#[derive(Debug, Clone)]
pub struct SixgTbParams {
    /// IMT-2030 service profile driving rate and modulation constraints.
    pub profile: SixgProfile,

    /// Transport block size in bits (payload before any FEC overhead).
    ///
    /// 6G research targets TB sizes up to [`SIXG_RESEARCH_MAX_TB_BITS`] via
    /// extended lifting or a new BG3.  See also [`NR5G_MAX_TB_BITS`].
    pub tb_size_bits: usize,

    /// Modulation order selected by the scheduler AMC.
    pub modulation: SixgModulation,

    /// Target code rate $R \in (0, 1)$.
    pub target_rate: f32,

    /// Whether this TB uses AI-assisted decoding hints.
    ///
    /// **Research direction** (not yet standardized): in the AI/ML RAN
    /// architecture studied in 3GPP Rel-18/19, neural network side-information
    /// (e.g. belief-propagation initialisation biases) may be fed to the LOMS
    /// decoder to accelerate convergence.  No air interface specification for
    /// this signalling exists as of 2024.
    pub ai_assisted: bool,
}

// ---------------------------------------------------------------------------
// Profile helpers
// ---------------------------------------------------------------------------

/// Return the nominal target code rate for a 6G NR service profile.
///
/// Values are derived from IMT-2030 spectral efficiency and reliability
/// targets as interpreted in the research literature (e.g. 3GPP TR 38.843,
/// TR 38.875, and ITU-R M.2160).  These are **indicative design points**,
/// not frozen 3GPP parameters.
///
/// | Profile              | Rate  | Rationale                            |
/// |----------------------|-------|--------------------------------------|
/// | `EnhancedMBB`        | 0.89  | Near-capacity for high-SNR Tbps links |
/// | `URLLC`              | 0.33  | Low rate for $10^{-7}$ reliability    |
/// | `MassiveMTC`         | 0.50  | Balanced rate for IoT power budgets   |
/// | `IntegratedSensing`  | 0.50  | Half-duplex sensing/comms split       |
/// | `AIComm`             | 0.67  | AI overhead reduces net payload rate  |
/// | `Semantic`           | 0.50  | Placeholder; semantic rate undefined  |
///
/// # Arguments
///
/// * `profile` — IMT-2030 use case category.
///
/// # Returns
///
/// Code rate $R \in (0, 1)$ as `f32`.
///
/// # Examples
///
/// ```
/// use syndrome::sixg::{SixgProfile, sixg_profile_target_rate};
///
/// let r = sixg_profile_target_rate(&SixgProfile::URLLC);
/// assert!((r - 0.33).abs() < 1e-6);
/// ```
pub fn sixg_profile_target_rate(profile: &SixgProfile) -> f32 {
    match profile {
        SixgProfile::EnhancedMBB => 0.89,
        SixgProfile::URLLC => 0.33,
        SixgProfile::MassiveMTC => 0.50,
        SixgProfile::IntegratedSensing => 0.50,
        SixgProfile::AIComm => 0.67,
        SixgProfile::Semantic => 0.50,
    }
}

/// Return the maximum modulation order permitted by a 6G NR service profile.
///
/// Profiles with tight reliability targets (URLLC, mMTC, Semantic) cap
/// modulation at QPSK to keep the constellation robust under fading.
/// eMBB is the only profile that is expected to reach 4096-QAM in favorable
/// channel conditions.
///
/// **Note**: 4096-QAM for eMBB is a confirmed research direction (ITU-R
/// IMT-2030 workshop, 2023) but not a finalized 3GPP specification.
///
/// # Arguments
///
/// * `profile` — IMT-2030 service profile.
///
/// # Returns
///
/// The highest [`SixgModulation`] the profile is permitted to use.
///
/// # Examples
///
/// ```
/// use syndrome::sixg::{SixgProfile, SixgModulation, sixg_profile_max_modulation};
///
/// assert_eq!(sixg_profile_max_modulation(&SixgProfile::EnhancedMBB),  SixgModulation::Qam4096);
/// assert_eq!(sixg_profile_max_modulation(&SixgProfile::URLLC),        SixgModulation::Qpsk);
/// ```
pub fn sixg_profile_max_modulation(profile: &SixgProfile) -> SixgModulation {
    match profile {
        SixgProfile::EnhancedMBB => SixgModulation::Qam4096,
        SixgProfile::URLLC => SixgModulation::Qpsk,
        SixgProfile::MassiveMTC => SixgModulation::Qpsk,
        SixgProfile::IntegratedSensing => SixgModulation::Qam64,
        SixgProfile::AIComm => SixgModulation::Qam256,
        SixgProfile::Semantic => SixgModulation::Qpsk,
    }
}

/// Select the 6G NR modulation order based on current channel $E_b/N_0$.
///
/// Implements a simplified Adaptive Modulation and Coding (AMC) mapping
/// using approximate SNR thresholds drawn from 6G research literature.
/// These thresholds assume an ideal AWGN channel and are suitable for
/// simulation studies; a real scheduler would also account for fading margin,
/// HARQ, and interference.
///
/// The threshold ladder (approximate research values):
///
/// | $E_b/N_0$ range (dB) | Selected modulation | $Q_m$ |
/// |----------------------|---------------------|--------|
/// | $< 3$                | BPSK                | 1      |
/// | $[3, 6)$             | QPSK                | 2      |
/// | $[6, 10)$            | 16-QAM              | 4      |
/// | $[10, 14)$           | 64-QAM              | 6      |
/// | $[14, 19)$           | 256-QAM             | 8      |
/// | $[19, 24)$           | 1024-QAM            | 10     |
/// | $\ge 24$             | 4096-QAM            | 12     |
///
/// # Arguments
///
/// * `ebno_db` — $E_b/N_0$ in decibels.
///
/// # Returns
///
/// The [`SixgModulation`] appropriate for the given SNR.
///
/// # Examples
///
/// ```
/// use syndrome::sixg::{sixg_select_modulation, SixgModulation};
///
/// assert_eq!(sixg_select_modulation(-5.0), SixgModulation::Bpsk);
/// assert_eq!(sixg_select_modulation(4.0),  SixgModulation::Qpsk);
/// assert_eq!(sixg_select_modulation(30.0), SixgModulation::Qam4096);
/// ```
pub fn sixg_select_modulation(ebno_db: f32) -> SixgModulation {
    if ebno_db < 3.0 {
        SixgModulation::Bpsk
    } else if ebno_db < 6.0 {
        SixgModulation::Qpsk
    } else if ebno_db < 10.0 {
        SixgModulation::Qam16
    } else if ebno_db < 14.0 {
        SixgModulation::Qam64
    } else if ebno_db < 19.0 {
        SixgModulation::Qam256
    } else if ebno_db < 24.0 {
        SixgModulation::Qam1024
    } else {
        SixgModulation::Qam4096
    }
}

// ---------------------------------------------------------------------------
// Block length constants
// ---------------------------------------------------------------------------

/// Research target maximum transport block size for 6G NR (bits).
///
/// **Research direction** (not yet standardized): 6G research papers (e.g.
/// 3GPP RP-231741, IMT-2030 workshop contributions) propose extending the
/// maximum TB size to approximately 8 million bits to support Tbps-class
/// throughput with manageable segment counts.  The mechanism under study is
/// either an extended lifting-size table (beyond $Z = 384$) or an entirely new
/// base graph ("BG3") with higher code rate and larger information dimension.
///
/// This constant is a round research target; the actual value will depend on
/// the 3GPP Rel-20+ standardization outcome.
pub const SIXG_RESEARCH_MAX_TB_BITS: usize = 8_000_000;

/// 5G NR maximum transport block size (bits) per 3GPP TS 38.214 Table 5.1.3.2-2.
///
/// In 5G NR, the largest achievable TB uses BG1 with $Z = 384$ and the maximum
/// number of code block segments.  The approximate ceiling is 1,277,992 bits,
/// as tabulated in TS 38.214 for the highest MCS and maximum allocated PRBs.
///
/// This constant is provided as a reference baseline for comparisons with
/// [`SIXG_RESEARCH_MAX_TB_BITS`].
pub const NR5G_MAX_TB_BITS: usize = 1_277_992;

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// All profile target rates must be strictly between 0.0 and 1.0.
    #[test]
    fn profile_rates_are_valid() {
        let profiles = [
            SixgProfile::EnhancedMBB,
            SixgProfile::URLLC,
            SixgProfile::MassiveMTC,
            SixgProfile::IntegratedSensing,
            SixgProfile::AIComm,
            SixgProfile::Semantic,
        ];
        for p in &profiles {
            let r = sixg_profile_target_rate(p);
            assert!(
                r > 0.0 && r <= 1.0,
                "Profile {p:?} has out-of-range rate {r}"
            );
        }
    }

    /// Verify `bits_per_symbol` returns the correct $Q_m$ for each modulation.
    #[test]
    fn modulation_bits_correct() {
        let cases = [
            (SixgModulation::Bpsk, 1u8),
            (SixgModulation::Qpsk, 2),
            (SixgModulation::Qam16, 4),
            (SixgModulation::Qam64, 6),
            (SixgModulation::Qam256, 8),
            (SixgModulation::Qam1024, 10),
            (SixgModulation::Qam4096, 12),
        ];
        for (m, expected) in &cases {
            assert_eq!(
                m.bits_per_symbol(),
                *expected,
                "{} should have Qm={}",
                m.label(),
                expected
            );
        }
    }

    /// AMC selection must be non-decreasing as SNR increases from 0 to 25 dB.
    ///
    /// The [`SixgModulation`] enum derives `PartialOrd`/`Ord` so we can compare
    /// consecutive selections directly.
    #[test]
    fn select_modulation_monotone() {
        let snr_points: Vec<f32> = (0..=25).map(|i| i as f32).collect();
        let mods: Vec<SixgModulation> = snr_points
            .iter()
            .map(|&s| sixg_select_modulation(s))
            .collect();
        for i in 1..mods.len() {
            assert!(
                mods[i] >= mods[i - 1],
                "Modulation decreased from SNR={} ({:?}) to SNR={} ({:?})",
                snr_points[i - 1],
                mods[i - 1],
                snr_points[i],
                mods[i]
            );
        }
    }

    /// 6G research target TB must be strictly larger than the 5G NR maximum.
    #[test]
    fn sixg_tb_larger_than_5g() {
        const {
            assert!(
                SIXG_RESEARCH_MAX_TB_BITS > NR5G_MAX_TB_BITS,
                "6G research max TB must exceed 5G NR max TB"
            );
        }
    }

    /// `SixgTbParams` can be constructed and its rate field round-trips.
    #[test]
    fn tb_params_construction() {
        let profile = SixgProfile::EnhancedMBB;
        let rate = sixg_profile_target_rate(&profile);
        let params = SixgTbParams {
            profile,
            tb_size_bits: 2_000_000,
            modulation: SixgModulation::Qam4096,
            target_rate: rate,
            ai_assisted: false,
        };
        assert!((params.target_rate - 0.89).abs() < 1e-6);
        assert_eq!(params.modulation.bits_per_symbol(), 12);
    }
}
