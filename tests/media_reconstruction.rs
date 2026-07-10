//! End-to-end FEC reconstruction tests: audio, video, Wi-Fi, and 6G scenarios.
use syndrome::channel_sim::AwgnChannel;
use syndrome::transport_block::{DlSchDecoder, DlSchEncoder};

#[allow(clippy::too_many_lines)]
#[test]
fn audio_frame_5g_nr_reconstruction() {
    // ════════════════════════════════════════════════════════════
    //   Audio Frame — Opus-style 100-byte frame over 5G NR channel
    //   TB size: 800 bits (100 bytes) | Rate: 0.5 | Standard: 5G NR
    // ════════════════════════════════════════════════════════════
    //
    // Segmentation: BG2, Z=88, C=1, K'=824, N=4400, G=1600 ≤ N ✓
    // E_per_cb = 1600 (qm=1)

    let tb_size: usize = 800;
    let target_rate: f32 = 0.5;
    let qm: usize = 1;
    let g: usize = 1600;

    let enc = DlSchEncoder::new(tb_size, target_rate, qm, g).unwrap();
    let actual_g = enc.output_bits();

    let tb: Vec<u8> = (0..tb_size).map(|i| ((i * 7 + 13) % 2) as u8).collect();
    let mut coded_bits = vec![0u8; actual_g];
    enc.encode(&tb, 0, &mut coded_bits).unwrap();

    let snr_points: &[f32] = &[-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0];

    println!("════════════════════════════════════════════════════════════");
    println!("  Audio Frame — Opus-style 100-byte frame over 5G NR");
    println!(
        "  TB size: {}bits ({}bytes) | Rate: {} | Standard: 5G NR",
        tb_size,
        tb_size / 8,
        target_rate
    );
    println!("════════════════════════════════════════════════════════════");
    println!(" Eb/No (dB) │ Raw BER     │ FEC BER     │ CRC  │ Iter");
    println!("────────────┼─────────────┼─────────────┼──────┼─────");

    for &ebno_db in snr_points {
        let mut channel = AwgnChannel::new(ebno_db, target_rate, 42);
        let llr = channel.transmit(&coded_bits);

        // Hard-decision raw BER (before FEC).
        let raw_hard: Vec<u8> = llr
            .iter()
            .map(|&l| if l >= 0.0 { 0u8 } else { 1u8 })
            .collect();
        let raw_ber = AwgnChannel::bit_error_rate(&raw_hard, &coded_bits);

        let mut dec = DlSchDecoder::new(tb_size, target_rate, qm, g, 20, 0.25).unwrap();
        let mut tb_out = vec![0u8; tb_size];
        let report = dec.decode(&llr, 0, &mut tb_out).unwrap();

        let fec_ber = AwgnChannel::bit_error_rate(&tb_out, &tb);
        let crc_label = if report.crc_ok {
            "PASS ✓"
        } else {
            "FAIL ✗"
        };
        let ber_str = if fec_ber == 0.0 {
            "0.0000".to_string()
        } else {
            format!("{:.4}", fec_ber)
        };

        println!(
            " {:>10.1}  │ {:.4}      │ {:<12} │ {} │ {}",
            ebno_db, raw_ber, ber_str, crc_label, report.max_iters_used
        );
    }

    // At 5 dB: perfect reconstruction required.
    {
        let mut channel = AwgnChannel::new(5.0, target_rate, 42);
        let llr = channel.transmit(&coded_bits);
        let mut dec = DlSchDecoder::new(tb_size, target_rate, qm, g, 20, 0.25).unwrap();
        let mut tb_out = vec![0u8; tb_size];
        let report = dec.decode(&llr, 0, &mut tb_out).unwrap();
        assert!(report.crc_ok, "Audio: CRC must pass at 5.0 dB Eb/No");
        assert_eq!(
            tb_out, tb,
            "Audio: perfect reconstruction required at 5.0 dB Eb/No"
        );
    }

    // At -1 dB: channel must be noisy (raw BER > 0.001 on coded bits).
    {
        let mut channel = AwgnChannel::new(-1.0, target_rate, 42);
        let llr = channel.transmit(&coded_bits);
        let raw_hard: Vec<u8> = llr
            .iter()
            .map(|&l| if l >= 0.0 { 0u8 } else { 1u8 })
            .collect();
        let raw_ber = AwgnChannel::bit_error_rate(&raw_hard, &coded_bits);
        assert!(
            raw_ber > 0.001,
            "Audio: raw BER at -1.0 dB should be > 0.001 (got {raw_ber:.4}) — channel noise validation"
        );
    }
}

#[allow(clippy::too_many_lines)]
#[test]
fn video_nalu_5g_nr_reconstruction() {
    // ════════════════════════════════════════════════════════════
    //   Video NAL Unit — H.265-style frame chunk over 5G NR
    //   TB size: 8000 bits (1000 bytes) | Rate: 0.33 | Standard: 5G NR
    // ════════════════════════════════════════════════════════════
    //
    // Segmentation: BG1, Z=384, C=1, K'=8024, N=25344, G=24000 ≤ N ✓
    // E_per_cb = 24000 (qm=1)

    let tb_size: usize = 8000;
    let target_rate: f32 = 0.33;
    let qm: usize = 1;
    let g: usize = 24000;

    let enc = DlSchEncoder::new(tb_size, target_rate, qm, g).unwrap();
    let actual_g = enc.output_bits();

    // Knuth multiplicative hash for pseudo-random bits.
    let tb: Vec<u8> = (0usize..tb_size)
        .map(|i| ((i.wrapping_mul(2_654_435_761) >> 31) & 1) as u8)
        .collect();
    let mut coded_bits = vec![0u8; actual_g];
    enc.encode(&tb, 0, &mut coded_bits).unwrap();

    let snr_points: &[f32] = &[1.0, 2.0, 3.0, 4.0];

    println!("════════════════════════════════════════════════════════════");
    println!("  Video NAL Unit — H.265-style frame chunk over 5G NR");
    println!(
        "  TB size: {}bits ({}bytes) | Rate: {} | Standard: 5G NR",
        tb_size,
        tb_size / 8,
        target_rate
    );
    println!("════════════════════════════════════════════════════════════");
    println!(" Eb/No (dB) │ Raw BER     │ FEC BER     │ CRC  │ Iter");
    println!("────────────┼─────────────┼─────────────┼──────┼─────");

    for &ebno_db in snr_points {
        let mut channel = AwgnChannel::new(ebno_db, target_rate, 42);
        let llr = channel.transmit(&coded_bits);

        let raw_hard: Vec<u8> = llr
            .iter()
            .map(|&l| if l >= 0.0 { 0u8 } else { 1u8 })
            .collect();
        let raw_ber = AwgnChannel::bit_error_rate(&raw_hard, &coded_bits);

        let mut dec = DlSchDecoder::new(tb_size, target_rate, qm, g, 20, 0.25).unwrap();
        let mut tb_out = vec![0u8; tb_size];
        let report = dec.decode(&llr, 0, &mut tb_out).unwrap();

        let fec_ber = AwgnChannel::bit_error_rate(&tb_out, &tb);
        let crc_label = if report.crc_ok {
            "PASS ✓"
        } else {
            "FAIL ✗"
        };
        let ber_str = if fec_ber == 0.0 {
            "0.0000".to_string()
        } else {
            format!("{:.4}", fec_ber)
        };

        println!(
            " {:>10.1}  │ {:.4}      │ {:<12} │ {} │ {}",
            ebno_db, raw_ber, ber_str, crc_label, report.max_iters_used
        );
    }

    // At 4.0 dB: perfect reconstruction required.
    {
        let mut channel = AwgnChannel::new(4.0, target_rate, 42);
        let llr = channel.transmit(&coded_bits);
        let mut dec = DlSchDecoder::new(tb_size, target_rate, qm, g, 20, 0.25).unwrap();
        let mut tb_out = vec![0u8; tb_size];
        let report = dec.decode(&llr, 0, &mut tb_out).unwrap();
        assert!(report.crc_ok, "Video: CRC must pass at 4.0 dB Eb/No");
        assert_eq!(
            tb_out, tb,
            "Video: perfect reconstruction required at 4.0 dB Eb/No"
        );
    }
}

#[allow(clippy::too_many_lines)]
#[test]
fn wifi6_frame_reconstruction() {
    // ════════════════════════════════════════════════════════════
    //   Wi-Fi 6 MCS7 (64-QAM 5/6) — audio streaming scenario
    //   LOMS algorithm with 5G NR BG1 proxy matrices
    //   TB size: 500 bits (62 bytes) | Rate: 5/6 ≈ 0.833 | Standard: Wi-Fi 6 proxy
    // ════════════════════════════════════════════════════════════
    //
    // Wi-Fi 6 MCS7: 64-QAM, code rate 5/6.
    // Using 5G NR LDPC (BG1) as a structural proxy — same LOMS algorithm,
    // different shift matrices.  This test demonstrates algorithm portability,
    // not 802.11 matrix compliance.
    //
    // Segmentation for A=500, R=5/6≈0.833:
    //   select_bg(500, 0.833): A=500 > 292, A=500 ≤ 3824 but R=0.833 > 0.67,
    //   R > 0.25 → BG1.
    //   B = 524, K_cb=8448, C=1, K'=524, K_b=22.
    //   22*Z ≥ 524 → Z ≥ 23.8 → Z=24. N=66*24=1584.
    //   G=600 ≤ N=1584 ✓

    println!("Wi-Fi 6 MCS7 (64-QAM 5/6) — LOMS algorithm, 5G BG1 proxy matrices");

    let tb_size: usize = 500;
    let target_rate: f32 = 5.0 / 6.0;
    let qm: usize = 1;
    let g: usize = 600;

    let enc = DlSchEncoder::new(tb_size, target_rate, qm, g).unwrap();
    let actual_g = enc.output_bits();

    let tb: Vec<u8> = (0..tb_size).map(|i| (i % 3 == 0) as u8).collect();
    let mut coded_bits = vec![0u8; actual_g];
    enc.encode(&tb, 0, &mut coded_bits).unwrap();

    let snr_points: &[f32] = &[3.0, 5.0, 7.0];

    println!("════════════════════════════════════════════════════════════");
    println!("  Wi-Fi 6 MCS7 (64-QAM 5/6) — LOMS algorithm, 5G BG1 proxy matrices");
    println!(
        "  TB size: {}bits ({}bytes) | Rate: 5/6 | Standard: Wi-Fi 6 (5G proxy)",
        tb_size,
        tb_size / 8
    );
    println!("════════════════════════════════════════════════════════════");
    println!(" Eb/No (dB) │ Raw BER     │ FEC BER     │ CRC  │ Iter");
    println!("────────────┼─────────────┼─────────────┼──────┼─────");

    for &ebno_db in snr_points {
        let mut channel = AwgnChannel::new(ebno_db, target_rate, 42);
        let llr = channel.transmit(&coded_bits);

        let raw_hard: Vec<u8> = llr
            .iter()
            .map(|&l| if l >= 0.0 { 0u8 } else { 1u8 })
            .collect();
        let raw_ber = AwgnChannel::bit_error_rate(&raw_hard, &coded_bits);

        let mut dec = DlSchDecoder::new(tb_size, target_rate, qm, g, 20, 0.25).unwrap();
        let mut tb_out = vec![0u8; tb_size];
        let report = dec.decode(&llr, 0, &mut tb_out).unwrap();

        let fec_ber = AwgnChannel::bit_error_rate(&tb_out, &tb);
        let crc_label = if report.crc_ok {
            "PASS ✓"
        } else {
            "FAIL ✗"
        };
        let ber_str = if fec_ber == 0.0 {
            "0.0000".to_string()
        } else {
            format!("{:.4}", fec_ber)
        };

        println!(
            " {:>10.1}  │ {:.4}      │ {:<12} │ {} │ {}",
            ebno_db, raw_ber, ber_str, crc_label, report.max_iters_used
        );
    }

    // At 7 dB: perfect reconstruction required.
    {
        let mut channel = AwgnChannel::new(7.0, target_rate, 42);
        let llr = channel.transmit(&coded_bits);
        let mut dec = DlSchDecoder::new(tb_size, target_rate, qm, g, 20, 0.25).unwrap();
        let mut tb_out = vec![0u8; tb_size];
        let report = dec.decode(&llr, 0, &mut tb_out).unwrap();
        assert!(report.crc_ok, "Wi-Fi 6 MCS7: CRC must pass at 7.0 dB Eb/No");
        assert_eq!(
            tb_out, tb,
            "Wi-Fi 6 MCS7: perfect reconstruction required at 7.0 dB Eb/No"
        );
    }
}

#[allow(clippy::too_many_lines)]
#[test]
fn sixg_embb_ultra_reliable() {
    // ════════════════════════════════════════════════════════════
    //   6G NR eMBB — ultra-reliable high-rate transmission
    //   TB size: 2000 bits (250 bytes) | Rate: 0.89 | Standard: 6G NR (research)
    // ════════════════════════════════════════════════════════════
    //
    // 6G eMBB profile: rate 0.89, 4096-QAM capable (confirmed research direction,
    // ITU-R IMT-2030 workshop 2023).  Demonstrated here with 5G NR LOMS kernel
    // (algorithm is expected to carry forward to 6G with extended parameters).
    //
    // Segmentation for A=2000, R=0.89:
    //   select_bg(2000, 0.89): A=2000 > 292, A ≤ 3824 but R=0.89 > 0.67,
    //   R > 0.25 → BG1.
    //   B=2024, K_cb=8448, C=1, K'=2024, K_b=22.
    //   22*Z ≥ 2024 → Z ≥ 92 → Z=96. N=66*96=6336.
    //   G=2246 ≤ N=6336 ✓  (2000/0.89 ≈ 2247.2 → use 2246)

    println!(
        "6G NR eMBB profile (rate 0.89, 4096-QAM capable) — demonstrating ultra-reliable reconstruction"
    );

    let tb_size: usize = 2000;
    let target_rate: f32 = 0.89;
    let qm: usize = 1;
    let g: usize = 2246;

    let enc = DlSchEncoder::new(tb_size, target_rate, qm, g).unwrap();
    let actual_g = enc.output_bits();

    let tb: Vec<u8> = (0..tb_size).map(|i| (i % 3 == 0) as u8).collect();
    let mut coded_bits = vec![0u8; actual_g];
    enc.encode(&tb, 0, &mut coded_bits).unwrap();

    let snr_points: &[f32] = &[8.0, 10.0, 12.0];

    println!("════════════════════════════════════════════════════════════");
    println!("  6G NR eMBB Profile (rate 0.89, 4096-QAM capable)");
    println!(
        "  TB size: {}bits ({}bytes) | Rate: {} | Standard: 6G NR (research)",
        tb_size,
        tb_size / 8,
        target_rate
    );
    println!("════════════════════════════════════════════════════════════");
    println!(" Eb/No (dB) │ Raw BER     │ FEC BER     │ CRC  │ Iter");
    println!("────────────┼─────────────┼─────────────┼──────┼─────");

    for &ebno_db in snr_points {
        let mut channel = AwgnChannel::new(ebno_db, target_rate, 42);
        let llr = channel.transmit(&coded_bits);

        let raw_hard: Vec<u8> = llr
            .iter()
            .map(|&l| if l >= 0.0 { 0u8 } else { 1u8 })
            .collect();
        let raw_ber = AwgnChannel::bit_error_rate(&raw_hard, &coded_bits);

        let mut dec = DlSchDecoder::new(tb_size, target_rate, qm, g, 20, 0.25).unwrap();
        let mut tb_out = vec![0u8; tb_size];
        let report = dec.decode(&llr, 0, &mut tb_out).unwrap();

        let fec_ber = AwgnChannel::bit_error_rate(&tb_out, &tb);
        let crc_label = if report.crc_ok {
            "PASS ✓"
        } else {
            "FAIL ✗"
        };
        let ber_str = if fec_ber == 0.0 {
            "0.0000".to_string()
        } else {
            format!("{:.4}", fec_ber)
        };

        println!(
            " {:>10.1}  │ {:.4}      │ {:<12} │ {} │ {}",
            ebno_db, raw_ber, ber_str, crc_label, report.max_iters_used
        );
    }

    // At 12 dB: perfect reconstruction required.
    {
        let mut channel = AwgnChannel::new(12.0, target_rate, 42);
        let llr = channel.transmit(&coded_bits);
        let mut dec = DlSchDecoder::new(tb_size, target_rate, qm, g, 20, 0.25).unwrap();
        let mut tb_out = vec![0u8; tb_size];
        let report = dec.decode(&llr, 0, &mut tb_out).unwrap();
        assert!(report.crc_ok, "6G eMBB: CRC must pass at 12.0 dB Eb/No");
        assert_eq!(
            tb_out, tb,
            "6G eMBB: perfect reconstruction required at 12.0 dB Eb/No"
        );
    }
}
