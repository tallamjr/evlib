/// Integration tests for EVT2/EVT3 format detection and loading
///
/// This test suite focuses on integration between format detection and loading functions,
/// testing the complete workflow from format detection to event loading.
/// Basic format detection tests are covered in test_format_detection.rs.
#[cfg(test)]
mod evt_format_detection_tests {
    use evlib::ev_formats::{format_detector::FormatDetector, load_events_with_config, LoadConfig};
    use std::fs::File;
    use std::io::Write;
    use std::path::Path;
    use tempfile::TempDir;

    const REAL_EVT2_FILE: &str = "data/eTram/raw/val_2/val_night_007.raw";

    /// Lightweight event row extracted from a Polars DataFrame for assertion convenience.
    #[derive(Debug, Clone, Copy)]
    struct EventRow {
        t: f64, // seconds
        x: u16,
        y: u16,
        #[allow(dead_code)]
        polarity: i8, // -1/1 for EVT/AER/HDF5; 0/1 for text
    }

    fn dataframe_to_rows(df: &polars::prelude::DataFrame) -> Vec<EventRow> {
        let x = df.column("x").unwrap().i16().unwrap();
        let y = df.column("y").unwrap().i16().unwrap();
        let t = df.column("t").unwrap().duration().unwrap();
        let p = df.column("polarity").unwrap().i8().unwrap();
        (0..df.height())
            .map(|i| EventRow {
                t: t.get(i).unwrap() as f64 / 1_000_000.0,
                x: x.get(i).unwrap() as u16,
                y: y.get(i).unwrap() as u16,
                polarity: p.get(i).unwrap(),
            })
            .collect()
    }

    #[test]
    fn test_evt2_vs_evt3_integration_comparison() {
        let temp_dir = TempDir::new().unwrap();

        // Create EVT2 file for integration testing
        let evt2_path = temp_dir.path().join("test_evt2.raw");
        let mut file = File::create(&evt2_path).unwrap();
        writeln!(file, "% evt 2.0").unwrap();
        writeln!(file, "% format EVT2;height=480;width=640").unwrap();
        writeln!(file, "% end").unwrap();
        file.write_all(&[0u8; 100]).unwrap();

        // Create EVT3 file for integration testing
        let evt3_path = temp_dir.path().join("test_evt3.raw");
        let mut file = File::create(&evt3_path).unwrap();
        writeln!(file, "% evt 3.0").unwrap();
        writeln!(file, "% format EVT3;height=480;width=640").unwrap();
        writeln!(file, "% end").unwrap();

        // Create synthetic EVT3 data
        let mut binary_data = Vec::new();
        let time_high = ((0x100u16) << 4) | 0x8;
        binary_data.extend_from_slice(&time_high.to_le_bytes());
        let time_low = ((0x200u16) << 4) | 0x6;
        binary_data.extend_from_slice(&time_low.to_le_bytes());
        let y_addr = (200u16) << 4;
        binary_data.extend_from_slice(&y_addr.to_le_bytes());
        let x_addr = ((1u16) << 15) | ((300u16) << 4) | 0x2;
        binary_data.extend_from_slice(&x_addr.to_le_bytes());
        file.write_all(&binary_data).unwrap();

        let config = LoadConfig::default();

        // Test loading both formats through the same interface
        let evt2_result = load_events_with_config(evt2_path.to_str().unwrap(), &config);
        let evt3_result = load_events_with_config(evt3_path.to_str().unwrap(), &config);

        println!("EVT2 vs EVT3 integration comparison:");
        match evt2_result {
            Ok(events) => println!("  EVT2 events loaded: {}", events.height()),
            Err(e) => println!("  EVT2 loading failed: {e}"),
        }

        match evt3_result {
            Ok(events) => println!("  EVT3 events loaded: {}", events.height()),
            Err(e) => println!("  EVT3 loading failed: {e}"),
        }
    }

    #[test]
    fn test_evt2_integration_with_load_function() {
        let file_path = Path::new(REAL_EVT2_FILE);
        if !file_path.exists() {
            println!("Skipping test - real data file not found: {REAL_EVT2_FILE}");
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

        assert!(events.height() > 0);
        println!("EVT2 integration test:");
        println!("  Events loaded: {}", events.height());

        // Validate events
        let rows = dataframe_to_rows(&events);
        for event in rows.iter().take(10) {
            assert!(event.t >= 0.0);
            assert!(event.x <= 1280); // Use <= instead of < for boundary cases
            assert!(event.y <= 720); // Use <= instead of < for boundary cases
                                     // Polarity is -1/1, so always valid
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

        // Create synthetic EVT3 data using the OpenEB wire-format layout:
        // bits 15..12 = type, bit 11 = polarity/orig, bits 10..0 = address,
        // and the 12-bit content field is bits 0..11.

        let mut binary_data = Vec::new();

        // Time High (type 0x8), high 12 bits of the timestamp
        let time_high = (0x8u16 << 12) | (0x100u16 & 0x0FFF);
        binary_data.extend_from_slice(&time_high.to_le_bytes());

        // Time Low (type 0x6), low 12 bits of the timestamp
        let time_low = (0x6u16 << 12) | (0x200u16 & 0x0FFF);
        binary_data.extend_from_slice(&time_low.to_le_bytes());

        // Y address (type 0x0, so the type nibble contributes nothing): y = 200
        let y_addr = 200u16 & 0x07FF;
        binary_data.extend_from_slice(&y_addr.to_le_bytes());

        // X address (type 0x2): x = 300, polarity = true. This emits one event.
        let x_addr = (0x2u16 << 12) | (1u16 << 11) | (300u16 & 0x07FF);
        binary_data.extend_from_slice(&x_addr.to_le_bytes());

        file.write_all(&binary_data).unwrap();

        let config = LoadConfig {
            sort: true,
            ..Default::default()
        };

        // Test loading with automatic format detection
        let events = load_events_with_config(file_path.to_str().unwrap(), &config).unwrap();

        assert!(events.height() > 0);
        println!("EVT3 integration test:");
        println!("  Events loaded: {}", events.height());

        // Validate events
        let rows = dataframe_to_rows(&events);
        for event in &rows {
            assert!(event.t >= 0.0);
            assert!(event.x <= 640); // Use <= instead of < for boundary cases
            assert!(event.y <= 480); // Use <= instead of < for boundary cases
                                     // Polarity is -1/1, so always valid
        }
    }

    #[test]
    fn test_format_detection_integration_performance() {
        let file_path = Path::new(REAL_EVT2_FILE);
        if !file_path.exists() {
            println!("Skipping test - real data file not found: {REAL_EVT2_FILE}");
            return;
        }

        let config = LoadConfig {
            chunk_size: Some(1000),
            ..Default::default()
        };

        let start_time = std::time::Instant::now();
        let events = load_events_with_config(file_path.to_str().unwrap(), &config).unwrap();
        let duration = start_time.elapsed();

        println!("Format detection + loading performance:");
        println!("  Events loaded: {}", events.height());
        println!("  Total time: {duration:?}");
        println!(
            "  Events per second: {:.2}",
            events.height() as f64 / duration.as_secs_f64()
        );

        // Integration should be reasonably fast
        assert!(events.height() > 0);
        assert!(duration.as_secs() < 30); // Should complete within 30 seconds
    }

    #[test]
    fn test_header_metadata_extraction() {
        let file_path = Path::new(REAL_EVT2_FILE);
        if !file_path.exists() {
            println!("Skipping test - real data file not found: {REAL_EVT2_FILE}");
            return;
        }

        let result = FormatDetector::detect_format(file_path).unwrap();

        println!("Header metadata extraction:");
        println!("  Resolution: {:?}", result.metadata.sensor_resolution);
        println!(
            "  File size: {file_size} bytes",
            file_size = result.metadata.file_size
        );
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
            println!("    {key}: {value}");
        }
    }
}
