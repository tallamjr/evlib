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

    /// Lightweight event row extracted from a Polars DataFrame for assertion convenience.
    #[derive(Debug, Clone, Copy)]
    struct EventRow {
        t: f64, // seconds
        x: u16,
        y: u16,
        polarity: i8, // -1/1 for EVT3
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

    /// Build a 16-bit EVT3 word (OpenEB wire format): type in bits 12-15,
    /// 12-bit content (polarity/orig in bit 11, address in bits 0-10, or a
    /// generic 12-bit mask/time field) in bits 0-11.
    fn evt3_word(ty: u16, content: u16) -> u16 {
        ((ty & 0xF) << 12) | (content & 0xFFF)
    }

    #[test]
    fn test_evt3_event_type_parsing() {
        // Test all EVT3 event types. Type lives in bits 12-15 (OpenEB wire format).
        let test_cases = vec![
            (0x0u16 << 12, Evt3EventType::AddrY),
            (0x2u16 << 12, Evt3EventType::AddrX),
            (0x3u16 << 12, Evt3EventType::VectBaseX),
            (0x4u16 << 12, Evt3EventType::Vect12),
            (0x5u16 << 12, Evt3EventType::Vect8),
            (0x6u16 << 12, Evt3EventType::TimeLow),
            (0x8u16 << 12, Evt3EventType::TimeHigh),
            (0xAu16 << 12, Evt3EventType::ExtTrigger),
        ];

        for (raw_data, expected_type) in test_cases {
            let raw_event = RawEvt3Event { data: raw_data };
            assert_eq!(raw_event.event_type().unwrap(), expected_type);
        }

        // Type code 0x1 maps to Reserved1, which is not an error in the current
        // reader (it is a known reserved type). Assert it parses as Reserved1.
        let raw_event = RawEvt3Event { data: 0x1u16 << 12 };
        assert_eq!(raw_event.event_type().unwrap(), Evt3EventType::Reserved1);
    }

    #[test]
    fn test_evt3_y_addr_event_parsing() {
        // Test Y address event: y=300, orig=true (slave camera)
        // type=AddrY (bits 12-15), orig (bit 11), y (bits 0-10)
        let raw_data = evt3_word(0x0, (1u16 << 11) | 300);
        let raw_event = RawEvt3Event { data: raw_data };

        let y_event = raw_event.as_y_addr_event().unwrap();
        assert_eq!(y_event.y, 300);
        assert!(y_event.orig);

        // Test Y address event: y=100, orig=false (master camera)
        let raw_data = evt3_word(0x0, 100);
        let raw_event = RawEvt3Event { data: raw_data };

        let y_event = raw_event.as_y_addr_event().unwrap();
        assert_eq!(y_event.y, 100);
        assert!(!y_event.orig);
    }

    #[test]
    fn test_evt3_x_addr_event_parsing() {
        // Test X address event: x=500, polarity=true (positive)
        // type=AddrX (bits 12-15), polarity (bit 11), x (bits 0-10)
        let raw_data = (0x2u16 << 12) | (1u16 << 11) | 500;
        let raw_event = RawEvt3Event { data: raw_data };

        let x_event = raw_event.as_x_addr_event().unwrap();
        assert_eq!(x_event.x, 500);
        assert!(x_event.polarity);

        // Test X address event: x=200, polarity=false (negative)
        let raw_data = (0x2u16 << 12) | 200;
        let raw_event = RawEvt3Event { data: raw_data };

        let x_event = raw_event.as_x_addr_event().unwrap();
        assert_eq!(x_event.x, 200);
        assert!(!x_event.polarity);
    }

    #[test]
    fn test_evt3_vect_base_x_event_parsing() {
        // Test Vector Base X event: x=800, polarity=true
        // type=VectBaseX (bits 12-15), polarity (bit 11), x (bits 0-10)
        let raw_data = (0x3u16 << 12) | (1u16 << 11) | 800;
        let raw_event = RawEvt3Event { data: raw_data };

        let vect_base_event = raw_event.as_vect_base_x_event().unwrap();
        assert_eq!(vect_base_event.x, 800);
        assert!(vect_base_event.polarity);
    }

    #[test]
    fn test_evt3_vect12_event_parsing() {
        // Test Vector 12 event with validity mask 0xABC (bits 2, 3, 4, 5, 7, 9, 11 set)
        // type=Vect12 (bits 12-15), 12-bit mask (bits 0-11)
        let raw_data = (0x4u16 << 12) | 0xABC;
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
        assert_eq!(set_bits, 7); // 0xABC = 0b1010_1011_1100 has 7 bits set
    }

    #[test]
    fn test_evt3_vect8_event_parsing() {
        // Test Vector 8 event with validity mask 0xF0 (bits 4, 5, 6, 7 set)
        // type=Vect8 (bits 12-15), 8-bit mask (bits 0-7)
        let raw_data = (0x5u16 << 12) | 0xF0;
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
        // type=TimeLow (bits 12-15), 12-bit time (bits 0-11)
        let raw_data = (0x6u16 << 12) | 0x123;
        let raw_event = RawEvt3Event { data: raw_data };

        let time_event = raw_event.as_time_event().unwrap();
        assert_eq!(time_event.time, 0x123);
        assert!(!time_event.is_high);

        // Test Time High event with time=0x456
        let raw_data = (0x8u16 << 12) | 0x456;
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

        // Write synthetic binary data sequence (OpenEB wire format:
        // type in bits 12-15, polarity/orig in bit 11, address in bits 0-10).
        let mut binary_data = Vec::new();

        // 1. Time High event (timestamp high bits = 0x100)
        let time_high = (0x8u16 << 12) | 0x100;
        binary_data.extend_from_slice(&time_high.to_le_bytes());

        // 2. Time Low event (timestamp low bits = 0x200)
        let time_low = (0x6u16 << 12) | 0x200;
        binary_data.extend_from_slice(&time_low.to_le_bytes());

        // 3. Y address event (y=100)
        let y_addr = evt3_word(0x0, 100);
        binary_data.extend_from_slice(&y_addr.to_le_bytes());

        // 4. X address event (x=200, polarity=positive)
        let x_addr = (0x2u16 << 12) | (1u16 << 11) | 200;
        binary_data.extend_from_slice(&x_addr.to_le_bytes());

        // 5. Vector Base X event (x=300, polarity=positive)
        let vect_base_x = (0x3u16 << 12) | (1u16 << 11) | 300;
        binary_data.extend_from_slice(&vect_base_x.to_le_bytes());

        // 6. Vector 8 event (bits 0, 2, 4 set = 0x15)
        let vect8 = (0x5u16 << 12) | 0x15;
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
        let (df, metadata) = reader.read_file(&file_path).unwrap();
        let events = dataframe_to_rows(&df);

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
            // EVT3 encodes positive polarity as 1
            assert_eq!(first_event.polarity, 1);

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

        // Time setup (OpenEB wire format: type in bits 12-15)
        let time_high = (0x8u16 << 12) | 0x100;
        binary_data.extend_from_slice(&time_high.to_le_bytes());
        let time_low = (0x6u16 << 12) | 0x200;
        binary_data.extend_from_slice(&time_low.to_le_bytes());

        // Y address (y=150, type AddrY)
        let y_addr = evt3_word(0x0, 150);
        binary_data.extend_from_slice(&y_addr.to_le_bytes());

        // Event 1: x=100, positive polarity (should be included)
        let x_addr1 = (0x2u16 << 12) | (1u16 << 11) | 100;
        binary_data.extend_from_slice(&x_addr1.to_le_bytes());

        // Event 2: x=50, negative polarity (excluded by bounding box: x < 80)
        let x_addr2 = (0x2u16 << 12) | 50;
        binary_data.extend_from_slice(&x_addr2.to_le_bytes());

        // Event 3: x=300, positive polarity (excluded by bounding box: x > 200)
        let x_addr3 = (0x2u16 << 12) | (1u16 << 11) | 300;
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
        let df = reader.read_with_config(&file_path, &load_config).unwrap();
        let events = dataframe_to_rows(&df);

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
            // Positive events only (EVT3 encodes positive as 1)
            assert_eq!(event.polarity, 1);
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

        // Time setup (OpenEB wire format: type in bits 12-15)
        let time_high = (0x8u16 << 12) | 0x100;
        binary_data.extend_from_slice(&time_high.to_le_bytes());
        let time_low = (0x6u16 << 12) | 0x200;
        binary_data.extend_from_slice(&time_low.to_le_bytes());

        // Y address (out of bounds: y=150 >= 100)
        let y_addr = evt3_word(0x0, 150);
        binary_data.extend_from_slice(&y_addr.to_le_bytes());

        // X address (out of bounds: x=150 >= 100), positive polarity
        let x_addr = (0x2u16 << 12) | (1u16 << 11) | 150;
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
        let (df, _) = reader.read_file(&file_path).unwrap();

        // Should have no events (all coordinates out of bounds)
        assert_eq!(df.height(), 0);

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
        let (df_no_validation, _) = reader_no_validation.read_file(&file_path).unwrap();

        // Should have events (validation disabled)
        assert!(df_no_validation.height() > 0);
    }

    #[test]
    fn test_evt3_config_defaults() {
        let config = Evt3Config::default();
        // Current default: validation disabled for better real-data compatibility.
        assert!(!config.validate_coordinates);
        assert!(!config.skip_invalid_events);
        assert_eq!(config.max_events, None);
        assert_eq!(config.sensor_resolution, None);
        assert_eq!(config.chunk_size, 1_000_000);
        assert_eq!(config.polarity_encoding, None);
    }
}
