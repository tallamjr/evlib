/// Tests for EVT3 format reader implementation
///
/// This test suite validates the EVT3 reader implementation with synthetic data
/// and ensures proper vectorized binary parsing and event reconstruction.
#[cfg(test)]
mod evt3_tests {
    use evlib::ev_formats::{
        evt3_reader::{Evt3Config, Evt3EventType, Evt3Reader, RawEvt3Event},
        format_detector::FormatDetector,
        LoadConfig,
    };
    use std::fs::File;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_evt3_event_type_parsing() {
        // Test all EVT3 event types
        let test_cases = vec![
            (0x0000, Evt3EventType::AddrY),
            (0x0002, Evt3EventType::AddrX),
            (0x0003, Evt3EventType::VectBaseX),
            (0x0004, Evt3EventType::Vect12),
            (0x0005, Evt3EventType::Vect8),
            (0x0006, Evt3EventType::TimeLow),
            (0x0008, Evt3EventType::TimeHigh),
            (0x000A, Evt3EventType::ExtTrigger),
        ];

        for (raw_data, expected_type) in test_cases {
            let raw_event = RawEvt3Event { data: raw_data };
            assert_eq!(raw_event.event_type().unwrap(), expected_type);
        }

        // Test invalid event type
        let raw_event = RawEvt3Event { data: 0x0001 };
        assert!(raw_event.event_type().is_err());
    }

    #[test]
    fn test_evt3_y_addr_event_parsing() {
        // Test Y address event: y=300, orig=true (slave camera)
        let raw_data = (1u16 << 15) | (300u16 << 4);
        let raw_event = RawEvt3Event { data: raw_data };

        let y_event = raw_event.as_y_addr_event().unwrap();
        assert_eq!(y_event.y, 300);
        assert!(y_event.orig);

        // Test Y address event: y=100, orig=false (master camera)
        let raw_data = 100u16 << 4;
        let raw_event = RawEvt3Event { data: raw_data };

        let y_event = raw_event.as_y_addr_event().unwrap();
        assert_eq!(y_event.y, 100);
        assert!(!y_event.orig);
    }

    #[test]
    fn test_evt3_x_addr_event_parsing() {
        // Test X address event: x=500, polarity=true (positive)
        let raw_data = (1u16 << 15) | (500u16 << 4) | 0x2;
        let raw_event = RawEvt3Event { data: raw_data };

        let x_event = raw_event.as_x_addr_event().unwrap();
        assert_eq!(x_event.x, 500);
        assert!(x_event.polarity);

        // Test X address event: x=200, polarity=false (negative)
        let raw_data = (200u16 << 4) | 0x2;
        let raw_event = RawEvt3Event { data: raw_data };

        let x_event = raw_event.as_x_addr_event().unwrap();
        assert_eq!(x_event.x, 200);
        assert!(!x_event.polarity);
    }

    #[test]
    fn test_evt3_vect_base_x_event_parsing() {
        // Test Vector Base X event: x=800, polarity=true
        let raw_data = (1u16 << 15) | (800u16 << 4) | 0x3;
        let raw_event = RawEvt3Event { data: raw_data };

        let vect_base_event = raw_event.as_vect_base_x_event().unwrap();
        assert_eq!(vect_base_event.x, 800);
        assert!(vect_base_event.polarity);
    }

    #[test]
    fn test_evt3_vect12_event_parsing() {
        // Test Vector 12 event with validity mask 0xABC (bits 0, 2, 3, 5, 7, 9, 10, 11 set)
        let raw_data = (0xABCu16 << 4) | 0x4;
        let raw_event = RawEvt3Event { data: raw_data };

        let vect12_event = raw_event.as_vect12_event().unwrap();
        assert_eq!(vect12_event.valid, 0xABC);

        // Count set bits
        let mut set_bits = 0;
        for i in 0..12 {
            if (vect12_event.valid >> i) & 1 != 0 {
                set_bits += 1;
            }
        }
        assert_eq!(set_bits, 8); // 0xABC has 8 bits set
    }

    #[test]
    fn test_evt3_vect8_event_parsing() {
        // Test Vector 8 event with validity mask 0xF0 (bits 4, 5, 6, 7 set)
        let raw_data = (0xF0u16 << 4) | 0x5;
        let raw_event = RawEvt3Event { data: raw_data };

        let vect8_event = raw_event.as_vect8_event().unwrap();
        assert_eq!(vect8_event.valid, 0xF0);

        // Count set bits
        let mut set_bits = 0;
        for i in 0..8 {
            if (vect8_event.valid >> i) & 1 != 0 {
                set_bits += 1;
            }
        }
        assert_eq!(set_bits, 4); // 0xF0 has 4 bits set
    }

    #[test]
    fn test_evt3_time_event_parsing() {
        // Test Time Low event with time=0x123
        let raw_data = (0x123u16 << 4) | 0x6;
        let raw_event = RawEvt3Event { data: raw_data };

        let time_event = raw_event.as_time_event().unwrap();
        assert_eq!(time_event.time, 0x123);
        assert!(!time_event.is_high);

        // Test Time High event with time=0x456
        let raw_data = (0x456u16 << 4) | 0x8;
        let raw_event = RawEvt3Event { data: raw_data };

        let time_event = raw_event.as_time_event().unwrap();
        assert_eq!(time_event.time, 0x456);
        assert!(time_event.is_high);
    }

    #[test]
    fn test_evt3_header_parsing() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test_evt3.raw");

        let mut file = File::create(&file_path).unwrap();
        writeln!(file, "% evt 3.0").unwrap();
        writeln!(file, "% format EVT3;height=480;width=640").unwrap();
        writeln!(file, "% geometry 640x480").unwrap();
        writeln!(file, "% camera_integrator_name Prophesee").unwrap();
        writeln!(file, "% generation 4.1").unwrap();
        writeln!(file, "% end").unwrap();

        // Write some dummy binary data
        let dummy_data = vec![0u8; 32];
        file.write_all(&dummy_data).unwrap();

        // Test format detection
        let detection_result = FormatDetector::detect_format(&file_path).unwrap();
        assert_eq!(detection_result.format.to_string(), "EVT3");
        assert!(detection_result.confidence > 0.9);
        assert_eq!(
            detection_result.metadata.sensor_resolution,
            Some((640, 480))
        );

        // Test header parsing with reader
        let reader = Evt3Reader::new();
        let mut file_handle = File::open(&file_path).unwrap();
        let (metadata, header_size) = reader.parse_header(&mut file_handle).unwrap();

        assert_eq!(metadata.sensor_resolution, Some((640, 480)));
        assert_eq!(
            metadata.properties.get("camera_integrator_name"),
            Some(&"Prophesee".to_string())
        );
        assert_eq!(
            metadata.properties.get("generation"),
            Some(&"4.1".to_string())
        );
        assert!(header_size > 0);
    }

    #[test]
    fn test_evt3_synthetic_data_reading() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test_evt3_synthetic.raw");

        let mut file = File::create(&file_path).unwrap();

        // Write header
        writeln!(file, "% evt 3.0").unwrap();
        writeln!(file, "% format EVT3;height=480;width=640").unwrap();
        writeln!(file, "% geometry 640x480").unwrap();
        writeln!(file, "% end").unwrap();

        // Write synthetic binary data sequence
        let mut binary_data = Vec::new();

        // 1. Time High event (timestamp high bits = 0x100)
        let time_high = ((0x100u16) << 4) | 0x8;
        binary_data.extend_from_slice(&time_high.to_le_bytes());

        // 2. Time Low event (timestamp low bits = 0x200)
        let time_low = ((0x200u16) << 4) | 0x6;
        binary_data.extend_from_slice(&time_low.to_le_bytes());

        // 3. Y address event (y=100)
        let y_addr = (100u16) << 4;
        binary_data.extend_from_slice(&y_addr.to_le_bytes());

        // 4. X address event (x=200, polarity=positive)
        let x_addr = ((1u16) << 15) | ((200u16) << 4) | 0x2;
        binary_data.extend_from_slice(&x_addr.to_le_bytes());

        // 5. Vector Base X event (x=300, polarity=positive)
        let vect_base_x = ((1u16) << 15) | ((300u16) << 4) | 0x3;
        binary_data.extend_from_slice(&vect_base_x.to_le_bytes());

        // 6. Vector 8 event (bits 0, 2, 4 set = 0x15)
        let vect8 = ((0x15u16) << 4) | 0x5;
        binary_data.extend_from_slice(&vect8.to_le_bytes());

        file.write_all(&binary_data).unwrap();

        // Test reading
        let config = Evt3Config {
            validate_coordinates: true,
            skip_invalid_events: false,
            max_events: Some(100),
            sensor_resolution: Some((640, 480)),
            chunk_size: 1000,
            polarity_encoding: None,
        };

        let reader = Evt3Reader::with_config(config);
        let (events, metadata) = reader.read_file(&file_path).unwrap();

        println!("Synthetic EVT3 data test results:");
        println!("  Events read: {}", events.len());
        println!("  Sensor resolution: {:?}", metadata.sensor_resolution);

        // Should have read events: 1 single event + 3 vector events = 4 total
        assert!(!events.is_empty()); // At least the single event
        assert_eq!(metadata.sensor_resolution, Some((640, 480)));

        // Check the first event (single X address event)
        if !events.is_empty() {
            let first_event = &events[0];
            assert_eq!(first_event.x, 200);
            assert_eq!(first_event.y, 100);
            assert!(first_event.polarity);

            // Check timestamp reconstruction (0x100 << 12 | 0x200 = 0x100200)
            let expected_timestamp = 0x100200_u32 as f64 / 1_000_000.0;
            assert_eq!(first_event.t, expected_timestamp);
        }
    }

    #[test]
    fn test_evt3_reader_with_load_config() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test_evt3_filtering.raw");

        let mut file = File::create(&file_path).unwrap();

        // Write header
        writeln!(file, "% evt 3.0").unwrap();
        writeln!(file, "% format EVT3;height=480;width=640").unwrap();
        writeln!(file, "% end").unwrap();

        // Write multiple events with different coordinates and polarities
        let mut binary_data = Vec::new();

        // Time setup
        let time_high = ((0x100u16) << 4) | 0x8;
        binary_data.extend_from_slice(&time_high.to_le_bytes());
        let time_low = ((0x200u16) << 4) | 0x6;
        binary_data.extend_from_slice(&time_low.to_le_bytes());

        // Y address
        let y_addr = (150u16) << 4;
        binary_data.extend_from_slice(&y_addr.to_le_bytes());

        // Event 1: x=100, positive polarity (should be included)
        let x_addr1 = ((1u16) << 15) | ((100u16) << 4) | 0x2;
        binary_data.extend_from_slice(&x_addr1.to_le_bytes());

        // Event 2: x=50, negative polarity (should be excluded by polarity filter)
        let x_addr2 = (50u16 << 4) | 0x2;
        binary_data.extend_from_slice(&x_addr2.to_le_bytes());

        // Event 3: x=300, positive polarity (should be excluded by coordinate filter)
        let x_addr3 = ((1u16) << 15) | ((300u16) << 4) | 0x2;
        binary_data.extend_from_slice(&x_addr3.to_le_bytes());

        file.write_all(&binary_data).unwrap();

        // Test with filtering
        let load_config = LoadConfig {
            min_x: Some(80),
            max_x: Some(200),
            min_y: Some(140),
            max_y: Some(160),
            polarity: Some(true), // Positive events only
            sort: true,
            ..Default::default()
        };

        let reader = Evt3Reader::new();
        let events = reader.read_with_config(&file_path, &load_config).unwrap();

        println!("EVT3 filtering test results:");
        println!("  Filtered events: {}", events.len());

        // Should have only the first event
        assert!(!events.is_empty());

        // Validate all events pass the filters
        for event in &events {
            assert!(event.x >= 80);
            assert!(event.x <= 200);
            assert!(event.y >= 140);
            assert!(event.y <= 160);
            assert!(event.polarity);
        }
    }

    #[test]
    fn test_evt3_coordinate_validation() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test_evt3_validation.raw");

        let mut file = File::create(&file_path).unwrap();

        // Write header with small resolution
        writeln!(file, "% evt 3.0").unwrap();
        writeln!(file, "% format EVT3;height=100;width=100").unwrap();
        writeln!(file, "% end").unwrap();

        // Write binary data with out-of-bounds coordinates
        let mut binary_data = Vec::new();

        // Time setup
        let time_high = ((0x100u16) << 4) | 0x8;
        binary_data.extend_from_slice(&time_high.to_le_bytes());
        let time_low = ((0x200u16) << 4) | 0x6;
        binary_data.extend_from_slice(&time_low.to_le_bytes());

        // Y address (out of bounds)
        let y_addr = (150u16) << 4;
        binary_data.extend_from_slice(&y_addr.to_le_bytes());

        // X address (out of bounds)
        let x_addr = ((1u16) << 15) | ((150u16) << 4) | 0x2;
        binary_data.extend_from_slice(&x_addr.to_le_bytes());

        file.write_all(&binary_data).unwrap();

        // Test with validation enabled and skip_invalid_events = true
        let config = Evt3Config {
            validate_coordinates: true,
            skip_invalid_events: true,
            max_events: Some(100),
            sensor_resolution: Some((100, 100)),
            chunk_size: 1000,
            polarity_encoding: None,
        };

        let reader = Evt3Reader::with_config(config);
        let (events, _) = reader.read_file(&file_path).unwrap();

        // Should have no events (all coordinates out of bounds)
        assert_eq!(events.len(), 0);

        // Test with validation disabled
        let config_no_validation = Evt3Config {
            validate_coordinates: false,
            skip_invalid_events: false,
            max_events: Some(100),
            sensor_resolution: Some((100, 100)),
            chunk_size: 1000,
            polarity_encoding: None,
        };

        let reader_no_validation = Evt3Reader::with_config(config_no_validation);
        let (events_no_validation, _) = reader_no_validation.read_file(&file_path).unwrap();

        // Should have events (validation disabled)
        assert!(!events_no_validation.is_empty());
    }

    #[test]
    fn test_evt3_config_defaults() {
        let config = Evt3Config::default();
        assert!(config.validate_coordinates);
        assert!(!config.skip_invalid_events);
        assert_eq!(config.max_events, None);
        assert_eq!(config.sensor_resolution, None);
        assert_eq!(config.chunk_size, 1_000_000);
        assert_eq!(config.polarity_encoding, None);
    }
}
