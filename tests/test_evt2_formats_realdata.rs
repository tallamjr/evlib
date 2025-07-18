/// Comprehensive tests for EVT2 format reader with real data
///
/// This test suite validates the EVT2 reader implementation against real Prophesee data files
/// and ensures proper binary parsing, timestamp reconstruction, and event validation.

#[cfg(test)]
mod evt2_tests {
    use evlib::ev_formats::{
        evt2_reader::{Evt2Config, Evt2Reader},
        format_detector::FormatDetector,
        LoadConfig,
    };
    use std::path::Path;

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

        // Test format detection
        let detection_result = FormatDetector::detect_format(file_path).unwrap();
        assert_eq!(detection_result.format.to_string(), "EVT2");
        assert!(detection_result.confidence > 0.9);

        // Verify metadata
        assert_eq!(
            detection_result.metadata.sensor_resolution,
            Some((1280, 720))
        );
        assert!(detection_result.metadata.file_size > 0);
        assert!(detection_result.metadata.estimated_event_count.is_some());

        println!("Format detection results:");
        println!("  Format: {}", detection_result.format);
        println!("  Confidence: {:.2}", detection_result.confidence);
        println!(
            "  Resolution: {:?}",
            detection_result.metadata.sensor_resolution
        );
        println!("  File size: {} bytes", detection_result.metadata.file_size);
        println!(
            "  Estimated events: {:?}",
            detection_result.metadata.estimated_event_count
        );
    }

    #[test]
    fn test_evt2_reader_basic() {
        let file_path = Path::new(REAL_EVT2_FILE);
        if !file_path.exists() {
            println!(
                "Skipping test - real data file not found: {}",
                REAL_EVT2_FILE
            );
            return;
        }

        // Create reader with limited event count for testing
        let config = Evt2Config {
            validate_coordinates: true,
            skip_invalid_events: false,
            max_events: Some(10000), // Limit to 10k events for testing
            sensor_resolution: Some((1280, 720)),
            chunk_size: 100000,
            polarity_encoding: None,
        };

        let reader = Evt2Reader::with_config(config);
        let (events, metadata) = reader.read_file(file_path).unwrap();

        println!("EVT2 reader test results:");
        println!("  Events read: {}", events.len());
        println!("  File size: {} bytes", metadata.file_size);
        println!("  Header size: {} bytes", metadata.header_size);
        println!("  Data size: {} bytes", metadata.data_size);
        println!("  Resolution: {:?}", metadata.sensor_resolution);

        // Basic validation
        assert!(events.len() > 0);
        assert!(events.len() <= 10000);
        assert_eq!(metadata.sensor_resolution, Some((1280, 720)));
        assert!(metadata.header_size > 0);
        assert!(metadata.data_size > 0);

        // Validate some events
        for (i, event) in events.iter().take(10).enumerate() {
            println!(
                "  Event {}: t={:.6}, x={}, y={}, p={}",
                i, event.t, event.x, event.y, event.polarity
            );

            // Basic sanity checks
            assert!(event.t >= 0.0);
            assert!(event.x < 1280);
            assert!(event.y < 720);
            // Polarity is bool, so always valid
        }

        // Check timestamp monotonicity (should be approximately sorted)
        let mut monotonic_violations = 0;
        for i in 1..events.len().min(1000) {
            if events[i].t < events[i - 1].t {
                monotonic_violations += 1;
            }
        }

        // Allow some violations due to parallel processing, but should be minimal
        let violation_rate = monotonic_violations as f64 / events.len().min(1000) as f64;
        println!(
            "  Timestamp monotonicity violations: {:.2}%",
            violation_rate * 100.0
        );
        assert!(violation_rate < 0.1); // Less than 10% violations
    }

    #[test]
    fn test_evt2_reader_with_load_config() {
        let file_path = Path::new(REAL_EVT2_FILE);
        if !file_path.exists() {
            println!(
                "Skipping test - real data file not found: {}",
                REAL_EVT2_FILE
            );
            return;
        }

        let reader = Evt2Reader::new();

        // Test with time window filtering
        let load_config = LoadConfig {
            t_start: Some(0.0),
            t_end: Some(1.0), // First second only
            min_x: Some(100),
            max_x: Some(200),
            min_y: Some(100),
            max_y: Some(200),
            polarity: Some(true), // Positive events only
            sort: true,
            ..Default::default()
        };

        let events = reader.read_with_config(file_path, &load_config).unwrap();

        println!("EVT2 reader with filtering test results:");
        println!("  Filtered events: {}", events.len());

        // Validate filtering worked
        for event in &events {
            assert!(event.t >= 0.0);
            assert!(event.t <= 1.0);
            assert!(event.x >= 100);
            assert!(event.x <= 200);
            assert!(event.y >= 100);
            assert!(event.y <= 200);
            assert_eq!(event.polarity, true);
        }

        // Check if events are sorted
        for i in 1..events.len() {
            assert!(events[i].t >= events[i - 1].t);
        }
    }

    #[test]
    fn test_evt2_reader_coordinate_validation() {
        let file_path = Path::new(REAL_EVT2_FILE);
        if !file_path.exists() {
            println!(
                "Skipping test - real data file not found: {}",
                REAL_EVT2_FILE
            );
            return;
        }

        // Test with coordinate validation enabled
        let config = Evt2Config {
            validate_coordinates: true,
            skip_invalid_events: false,
            max_events: Some(1000),
            sensor_resolution: Some((1280, 720)),
            chunk_size: 10000,
            polarity_encoding: None,
        };

        let reader = Evt2Reader::with_config(config);
        let (events, _) = reader.read_file(file_path).unwrap();

        // All events should have valid coordinates
        for event in &events {
            assert!(event.x < 1280);
            assert!(event.y < 720);
        }

        // Test with coordinate validation disabled but skip invalid events
        let config_skip = Evt2Config {
            validate_coordinates: true,
            skip_invalid_events: true,
            max_events: Some(1000),
            sensor_resolution: Some((1280, 720)),
            chunk_size: 10000,
            polarity_encoding: None,
        };

        let reader_skip = Evt2Reader::with_config(config_skip);
        let (events_skip, _) = reader_skip.read_file(file_path).unwrap();

        // Should get similar number of events (assuming most are valid)
        assert!(events_skip.len() >= events.len() * 95 / 100); // Allow 5% difference
    }

    #[test]
    fn test_evt2_reader_performance() {
        let file_path = Path::new(REAL_EVT2_FILE);
        if !file_path.exists() {
            println!(
                "Skipping test - real data file not found: {}",
                REAL_EVT2_FILE
            );
            return;
        }

        let config = Evt2Config {
            validate_coordinates: true,
            skip_invalid_events: false,
            max_events: Some(100000), // 100k events
            sensor_resolution: Some((1280, 720)),
            chunk_size: 100000,
            polarity_encoding: None,
        };

        let reader = Evt2Reader::with_config(config);

        let start_time = std::time::Instant::now();
        let (events, metadata) = reader.read_file(file_path).unwrap();
        let duration = start_time.elapsed();

        println!("EVT2 reader performance test results:");
        println!("  Events read: {}", events.len());
        println!("  Time taken: {:?}", duration);
        println!(
            "  Events per second: {:.2}",
            events.len() as f64 / duration.as_secs_f64()
        );
        println!(
            "  Megabytes per second: {:.2}",
            metadata.data_size as f64 / duration.as_secs_f64() / 1_000_000.0
        );

        // Performance should be reasonable (at least 100k events/second)
        let events_per_second = events.len() as f64 / duration.as_secs_f64();
        assert!(events_per_second > 100_000.0);
    }

    #[test]
    fn test_evt2_reader_chunked_vs_single() {
        let file_path = Path::new(REAL_EVT2_FILE);
        if !file_path.exists() {
            println!(
                "Skipping test - real data file not found: {}",
                REAL_EVT2_FILE
            );
            return;
        }

        let max_events = 10000;

        // Test with large chunks
        let config_large = Evt2Config {
            validate_coordinates: true,
            skip_invalid_events: false,
            max_events: Some(max_events),
            sensor_resolution: Some((1280, 720)),
            chunk_size: 1_000_000,
            polarity_encoding: None,
        };

        // Test with small chunks
        let config_small = Evt2Config {
            validate_coordinates: true,
            skip_invalid_events: false,
            max_events: Some(max_events),
            sensor_resolution: Some((1280, 720)),
            chunk_size: 1000,
            polarity_encoding: None,
        };

        let reader_large = Evt2Reader::with_config(config_large);
        let reader_small = Evt2Reader::with_config(config_small);

        let (events_large, _) = reader_large.read_file(file_path).unwrap();
        let (events_small, _) = reader_small.read_file(file_path).unwrap();

        println!("EVT2 chunked reading test results:");
        println!("  Large chunks: {} events", events_large.len());
        println!("  Small chunks: {} events", events_small.len());

        // Should read the same number of events
        assert_eq!(events_large.len(), events_small.len());

        // Events should be identical (same parsing logic)
        for (i, (event_large, event_small)) in
            events_large.iter().zip(events_small.iter()).enumerate()
        {
            assert_eq!(
                event_large.t, event_small.t,
                "Timestamp mismatch at event {}",
                i
            );
            assert_eq!(
                event_large.x, event_small.x,
                "X coordinate mismatch at event {}",
                i
            );
            assert_eq!(
                event_large.y, event_small.y,
                "Y coordinate mismatch at event {}",
                i
            );
            assert_eq!(
                event_large.polarity, event_small.polarity,
                "Polarity mismatch at event {}",
                i
            );
        }
    }

    #[test]
    fn test_evt2_reader_header_parsing() {
        let file_path = Path::new(REAL_EVT2_FILE);
        if !file_path.exists() {
            println!(
                "Skipping test - real data file not found: {}",
                REAL_EVT2_FILE
            );
            return;
        }

        let reader = Evt2Reader::new();
        let (_, metadata) = reader.read_file(file_path).unwrap();

        println!("EVT2 header parsing test results:");
        println!("  Sensor resolution: {:?}", metadata.sensor_resolution);
        println!("  Header properties: {:#?}", metadata.properties);

        // Validate header parsing
        assert_eq!(metadata.sensor_resolution, Some((1280, 720)));
        assert!(metadata.properties.contains_key("camera_integrator_name"));
        assert!(metadata.properties.contains_key("generation"));
        assert!(metadata.properties.contains_key("serial_number"));
        assert!(metadata.properties.contains_key("system_ID"));

        // Check specific expected values from the real file
        assert_eq!(
            metadata.properties.get("camera_integrator_name"),
            Some(&"Prophesee".to_string())
        );
        assert_eq!(
            metadata.properties.get("generation"),
            Some(&"4.1".to_string())
        );
        assert_eq!(
            metadata.properties.get("plugin_name"),
            Some(&"hal_plugin_gen41_evk3".to_string())
        );
    }

    #[test]
    fn test_evt2_timestamp_reconstruction() {
        let file_path = Path::new(REAL_EVT2_FILE);
        if !file_path.exists() {
            println!(
                "Skipping test - real data file not found: {}",
                REAL_EVT2_FILE
            );
            return;
        }

        let config = Evt2Config {
            validate_coordinates: true,
            skip_invalid_events: false,
            max_events: Some(50000),
            sensor_resolution: Some((1280, 720)),
            chunk_size: 50000,
            polarity_encoding: None,
        };

        let reader = Evt2Reader::with_config(config);
        let (events, _) = reader.read_file(file_path).unwrap();

        println!("EVT2 timestamp reconstruction test results:");
        println!("  Events read: {}", events.len());

        if !events.is_empty() {
            let first_timestamp = events[0].t;
            let last_timestamp = events[events.len() - 1].t;
            let duration = last_timestamp - first_timestamp;

            println!("  First timestamp: {:.6} s", first_timestamp);
            println!("  Last timestamp: {:.6} s", last_timestamp);
            println!("  Duration: {:.6} s", duration);

            // Validate timestamp reconstruction
            assert!(first_timestamp >= 0.0);
            assert!(last_timestamp > first_timestamp);
            assert!(duration > 0.0);
            assert!(duration < 3600.0); // Should be less than 1 hour

            // Check for reasonable timestamp distribution
            let mut monotonic_count = 0;
            for i in 1..events.len() {
                if events[i].t >= events[i - 1].t {
                    monotonic_count += 1;
                }
            }

            let monotonic_rate = monotonic_count as f64 / (events.len() - 1) as f64;
            println!("  Monotonic rate: {:.2}%", monotonic_rate * 100.0);
            assert!(monotonic_rate > 0.8); // At least 80% should be monotonic
        }
    }
}
