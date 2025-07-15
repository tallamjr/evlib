/// Integration tests for EVT2/EVT3 format detection
///
/// This test suite validates that the format detection correctly identifies
/// EVT2 and EVT3 files and integrates properly with the main loading function.

#[cfg(test)]
mod evt_format_detection_tests {
    use evlib::ev_formats::{
        format_detector::{EventFormat, FormatDetector},
        load_events_with_config, LoadConfig,
    };
    use std::fs::File;
    use std::io::Write;
    use std::path::Path;
    use tempfile::TempDir;

    const REAL_EVT2_FILE: &str = "data/eTram/raw/val_2/val_night_007.raw";

    #[test]
    fn test_evt2_format_detection() {
        let file_path = Path::new(REAL_EVT2_FILE);
        if !file_path.exists() {
            println!(
                "Skipping test - real data file not found: {}",
                REAL_EVT2_FILE
            );
            return;
        }

        let result = FormatDetector::detect_format(file_path).unwrap();

        assert_eq!(result.format, EventFormat::EVT2);
        assert!(result.confidence > 0.9);
        assert_eq!(result.metadata.sensor_resolution, Some((1280, 720)));

        println!("EVT2 format detection:");
        println!("  Format: {}", result.format);
        println!("  Confidence: {:.2}", result.confidence);
        println!("  Resolution: {:?}", result.metadata.sensor_resolution);
    }

    #[test]
    fn test_evt3_format_detection() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test_evt3.raw");

        let mut file = File::create(&file_path).unwrap();
        writeln!(file, "% evt 3.0").unwrap();
        writeln!(file, "% format EVT3;height=720;width=1280").unwrap();
        writeln!(file, "% geometry 1280x720").unwrap();
        writeln!(file, "% camera_integrator_name Prophesee").unwrap();
        writeln!(file, "% end").unwrap();

        // Add some dummy binary data
        let dummy_data = vec![0u8; 100];
        file.write_all(&dummy_data).unwrap();

        let result = FormatDetector::detect_format(&file_path).unwrap();

        assert_eq!(result.format, EventFormat::EVT3);
        assert!(result.confidence > 0.9);
        assert_eq!(result.metadata.sensor_resolution, Some((1280, 720)));

        println!("EVT3 format detection:");
        println!("  Format: {}", result.format);
        println!("  Confidence: {:.2}", result.confidence);
        println!("  Resolution: {:?}", result.metadata.sensor_resolution);
    }

    #[test]
    fn test_evt2_vs_evt3_detection() {
        let temp_dir = TempDir::new().unwrap();

        // Create EVT2 file
        let evt2_path = temp_dir.path().join("test_evt2.raw");
        let mut file = File::create(&evt2_path).unwrap();
        writeln!(file, "% evt 2.0").unwrap();
        writeln!(file, "% format EVT2;height=480;width=640").unwrap();
        writeln!(file, "% end").unwrap();
        file.write_all(&[0u8; 100]).unwrap();

        // Create EVT3 file
        let evt3_path = temp_dir.path().join("test_evt3.raw");
        let mut file = File::create(&evt3_path).unwrap();
        writeln!(file, "% evt 3.0").unwrap();
        writeln!(file, "% format EVT3;height=480;width=640").unwrap();
        writeln!(file, "% end").unwrap();
        file.write_all(&[0u8; 100]).unwrap();

        let evt2_result = FormatDetector::detect_format(&evt2_path).unwrap();
        let evt3_result = FormatDetector::detect_format(&evt3_path).unwrap();

        assert_eq!(evt2_result.format, EventFormat::EVT2);
        assert_eq!(evt3_result.format, EventFormat::EVT3);

        assert!(evt2_result.confidence > 0.9);
        assert!(evt3_result.confidence > 0.9);

        println!("EVT2 vs EVT3 detection:");
        println!(
            "  EVT2 format: {}, confidence: {:.2}",
            evt2_result.format, evt2_result.confidence
        );
        println!(
            "  EVT3 format: {}, confidence: {:.2}",
            evt3_result.format, evt3_result.confidence
        );
    }

    #[test]
    fn test_evt2_integration_with_load_function() {
        let file_path = Path::new(REAL_EVT2_FILE);
        if !file_path.exists() {
            println!(
                "Skipping test - real data file not found: {}",
                REAL_EVT2_FILE
            );
            return;
        }

        let config = LoadConfig {
            t_start: None,
            t_end: None,
            min_x: None,
            max_x: None,
            min_y: None,
            max_y: None,
            polarity: None,
            sort: false,
            chunk_size: Some(10000),
            ..Default::default()
        };

        // Test loading with automatic format detection
        let events = load_events_with_config(file_path.to_str().unwrap(), &config).unwrap();

        assert!(events.len() > 0);
        println!("EVT2 integration test:");
        println!("  Events loaded: {}", events.len());

        // Validate events
        for event in events.iter().take(10) {
            assert!(event.t >= 0.0);
            assert!(event.x < 1280);
            assert!(event.y < 720);
            assert!(event.polarity == 1 || event.polarity == -1);
        }
    }

    #[test]
    fn test_evt3_integration_with_load_function() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test_evt3_integration.raw");

        let mut file = File::create(&file_path).unwrap();
        writeln!(file, "% evt 3.0").unwrap();
        writeln!(file, "% format EVT3;height=480;width=640").unwrap();
        writeln!(file, "% end").unwrap();

        // Create synthetic EVT3 data
        let mut binary_data = Vec::new();

        // Time High
        let time_high = ((0x100u16) << 4) | 0x8;
        binary_data.extend_from_slice(&time_high.to_le_bytes());

        // Time Low
        let time_low = ((0x200u16) << 4) | 0x6;
        binary_data.extend_from_slice(&time_low.to_le_bytes());

        // Y address
        let y_addr = ((200u16) << 4) | 0x0;
        binary_data.extend_from_slice(&y_addr.to_le_bytes());

        // X address
        let x_addr = ((1u16) << 15) | ((300u16) << 4) | 0x2;
        binary_data.extend_from_slice(&x_addr.to_le_bytes());

        file.write_all(&binary_data).unwrap();

        let config = LoadConfig {
            sort: true,
            ..Default::default()
        };

        // Test loading with automatic format detection
        let events = load_events_with_config(file_path.to_str().unwrap(), &config).unwrap();

        assert!(events.len() > 0);
        println!("EVT3 integration test:");
        println!("  Events loaded: {}", events.len());

        // Validate events
        for event in &events {
            assert!(event.t >= 0.0);
            assert!(event.x < 640);
            assert!(event.y < 480);
            assert!(event.polarity == 1 || event.polarity == -1);
        }
    }

    #[test]
    fn test_format_descriptions() {
        use evlib::ev_formats::format_detector::FormatDetector;

        let evt2_desc = FormatDetector::get_format_description(&EventFormat::EVT2);
        let evt3_desc = FormatDetector::get_format_description(&EventFormat::EVT3);

        assert_eq!(evt2_desc, "EVT2 format (Prophesee binary events)");
        assert_eq!(
            evt3_desc,
            "EVT3 format (Prophesee vectorized binary events)"
        );

        println!("Format descriptions:");
        println!("  EVT2: {}", evt2_desc);
        println!("  EVT3: {}", evt3_desc);
    }

    #[test]
    fn test_format_detection_with_different_extensions() {
        let temp_dir = TempDir::new().unwrap();

        // Test .raw extension with EVT2
        let evt2_raw_path = temp_dir.path().join("test_evt2.raw");
        let mut file = File::create(&evt2_raw_path).unwrap();
        writeln!(file, "% evt 2.0").unwrap();
        writeln!(file, "% format EVT2;height=480;width=640").unwrap();
        writeln!(file, "% end").unwrap();
        file.write_all(&[0u8; 100]).unwrap();

        // Test .dat extension with EVT3
        let evt3_dat_path = temp_dir.path().join("test_evt3.dat");
        let mut file = File::create(&evt3_dat_path).unwrap();
        writeln!(file, "% evt 3.0").unwrap();
        writeln!(file, "% format EVT3;height=480;width=640").unwrap();
        writeln!(file, "% end").unwrap();
        file.write_all(&[0u8; 100]).unwrap();

        let evt2_result = FormatDetector::detect_format(&evt2_raw_path).unwrap();
        let evt3_result = FormatDetector::detect_format(&evt3_dat_path).unwrap();

        // Content detection should override extension detection
        assert_eq!(evt2_result.format, EventFormat::EVT2);
        assert_eq!(evt3_result.format, EventFormat::EVT3);

        // Both should have high confidence due to content analysis
        assert!(evt2_result.confidence > 0.9);
        assert!(evt3_result.confidence > 0.9);

        println!("Extension vs content detection:");
        println!(
            "  EVT2 (.raw): {}, confidence: {:.2}",
            evt2_result.format, evt2_result.confidence
        );
        println!(
            "  EVT3 (.dat): {}, confidence: {:.2}",
            evt3_result.format, evt3_result.confidence
        );
    }

    #[test]
    fn test_format_detection_performance() {
        let file_path = Path::new(REAL_EVT2_FILE);
        if !file_path.exists() {
            println!(
                "Skipping test - real data file not found: {}",
                REAL_EVT2_FILE
            );
            return;
        }

        let start_time = std::time::Instant::now();
        let result = FormatDetector::detect_format(file_path).unwrap();
        let duration = start_time.elapsed();

        println!("Format detection performance:");
        println!("  Format: {}", result.format);
        println!("  Confidence: {:.2}", result.confidence);
        println!("  Detection time: {:?}", duration);
        println!("  File size: {} bytes", result.metadata.file_size);

        // Format detection should be fast (< 100ms for large files)
        assert!(duration.as_millis() < 100);
        assert_eq!(result.format, EventFormat::EVT2);
    }

    #[test]
    fn test_header_metadata_extraction() {
        let file_path = Path::new(REAL_EVT2_FILE);
        if !file_path.exists() {
            println!(
                "Skipping test - real data file not found: {}",
                REAL_EVT2_FILE
            );
            return;
        }

        let result = FormatDetector::detect_format(file_path).unwrap();

        println!("Header metadata extraction:");
        println!("  Resolution: {:?}", result.metadata.sensor_resolution);
        println!("  File size: {} bytes", result.metadata.file_size);
        println!(
            "  Estimated events: {:?}",
            result.metadata.estimated_event_count
        );

        // Validate metadata
        assert_eq!(result.metadata.sensor_resolution, Some((1280, 720)));
        assert!(result.metadata.file_size > 0);
        assert!(result.metadata.estimated_event_count.is_some());

        if let Some(event_count) = result.metadata.estimated_event_count {
            assert!(event_count > 0);
            println!(
                "  Event density: {:.2} events/byte",
                event_count as f64 / result.metadata.file_size as f64
            );
        }

        // Check header properties
        println!("  Header properties:");
        for (key, value) in &result.metadata.properties {
            println!("    {}: {}", key, value);
        }
    }
}
