/// Comprehensive tests for AEDAT address decoding
///
/// This test suite validates the correct implementation of address decoding
/// for all AEDAT format versions, ensuring that coordinates and polarity
/// are extracted correctly according to the specifications.
use evlib::ev_formats::aedat_reader::{AedatConfig, AedatReader, AedatVersion};
use std::fs::File;
use std::io::Write;
use tempfile::TempDir;

/// Test AEDAT 1.0 address decoding with various coordinate combinations
#[test]
fn test_aedat_1_0_address_decoding() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("test_aedat_1_0_address.aedat");

    // Create test AEDAT 1.0 file
    let mut file = File::create(&file_path).unwrap();
    writeln!(file, "#!AER-DAT1.0").unwrap();
    writeln!(file, "# sizeX 128").unwrap();
    writeln!(file, "# sizeY 128").unwrap();

    // Test various coordinate combinations
    // AEDAT 1.0 DVS128 format:
    // - Bit 0: Polarity (1 = ON, 0 = OFF)
    // - Bits 1-7: X coordinate (7 bits)
    // - Bits 8-14: Y coordinate (7 bits)
    // - Bit 15: External event flag (0 for DVS events)
    let test_cases = vec![
        // (expected_x, expected_y, expected_polarity, timestamp)
        (0, 0, 1, 1000),     // Corner case: (0,0) with ON polarity
        (0, 0, -1, 2000),    // Corner case: (0,0) with OFF polarity
        (1, 1, 1, 3000),     // Small coordinates
        (127, 127, 1, 4000), // Max coordinates for 7-bit
        (64, 32, -1, 5000),  // Mid-range coordinates
        (10, 50, 1, 6000),   // Arbitrary coordinates
    ];

    for (x, y, polarity, timestamp) in &test_cases {
        // Create address according to AEDAT 1.0 specification
        let address = ((*y as u16) << 8) | ((*x as u16) << 1) | if *polarity == 1 { 1 } else { 0 };
        file.write_all(&address.to_le_bytes()).unwrap();
        file.write_all(&(*timestamp as u32).to_le_bytes()).unwrap();
    }

    // Test reading with disabled validation for test data
    let config = AedatConfig {
        validate_timestamps: false,
        validate_coordinates: false,
        validate_polarity: false,
        skip_invalid_events: false,
        max_events: None,
        max_resolution: None,
    };
    let reader = AedatReader::with_config(config);
    let (events, metadata) = reader.read_file(&file_path).unwrap();

    assert_eq!(metadata.version, Some(AedatVersion::V1_0));
    assert_eq!(events.len(), test_cases.len());

    // Verify each event
    for (i, (expected_x, expected_y, expected_polarity, expected_timestamp)) in
        test_cases.iter().enumerate()
    {
        let event = &events[i];
        assert_eq!(
            event.x, *expected_x as u16,
            "Event {}: X coordinate mismatch",
            i
        );
        assert_eq!(
            event.y, *expected_y as u16,
            "Event {}: Y coordinate mismatch",
            i
        );
        assert_eq!(
            event.polarity, *expected_polarity,
            "Event {}: Polarity mismatch",
            i
        );
        assert_eq!(
            event.t, *expected_timestamp as f64,
            "Event {}: Timestamp mismatch",
            i
        );
    }
}

/// Test AEDAT 2.0 address decoding with 32-bit addresses
#[test]
fn test_aedat_2_0_address_decoding() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("test_aedat_2_0_address.aedat");

    // Create test AEDAT 2.0 file
    let mut file = File::create(&file_path).unwrap();
    writeln!(file, "#!AER-DAT2.0").unwrap();
    writeln!(file, "# sizeX 640").unwrap();
    writeln!(file, "# sizeY 480").unwrap();
    writeln!(file, "").unwrap(); // Add empty line to mark end of header

    // Test various coordinate combinations
    // AEDAT 2.0 format: polarity (1 bit) + x (15 bits) + y (15 bits) + unused (1 bit)
    let test_cases = vec![
        // (expected_x, expected_y, expected_polarity, timestamp)
        (0, 0, 1, 1000),
        (0, 0, -1, 2000),
        (1, 1, 1, 3000),
        (639, 479, 1, 4000),  // Max resolution
        (320, 240, -1, 5000), // Mid-range coordinates
        (100, 200, 1, 6000),  // Arbitrary coordinates
    ];

    for (x, y, polarity, timestamp) in &test_cases {
        // Create address according to AEDAT 2.0 specification
        let address = ((*y as u32) << 16) | ((*x as u32) << 1) | if *polarity == 1 { 1 } else { 0 };
        file.write_all(&(*timestamp as u32).to_be_bytes()).unwrap(); // Big-endian timestamp
        file.write_all(&address.to_be_bytes()).unwrap(); // Big-endian address
    }

    // Test reading with disabled validation for test data
    let config = AedatConfig {
        validate_timestamps: false,
        validate_coordinates: false,
        validate_polarity: false,
        skip_invalid_events: true,
        max_events: None,
        max_resolution: None,
    };
    let reader = AedatReader::with_config(config);
    let (events, metadata) = reader.read_file(&file_path).unwrap();

    assert_eq!(metadata.version, Some(AedatVersion::V2_0));
    assert_eq!(events.len(), test_cases.len());

    // Verify each event
    for (i, (expected_x, expected_y, expected_polarity, expected_timestamp)) in
        test_cases.iter().enumerate()
    {
        let event = &events[i];
        assert_eq!(
            event.x, *expected_x as u16,
            "Event {}: X coordinate mismatch",
            i
        );
        assert_eq!(
            event.y, *expected_y as u16,
            "Event {}: Y coordinate mismatch",
            i
        );
        assert_eq!(
            event.polarity, *expected_polarity,
            "Event {}: Polarity mismatch",
            i
        );
        assert_eq!(
            event.t, *expected_timestamp as f64,
            "Event {}: Timestamp mismatch",
            i
        );
    }
}

/// Test AEDAT 3.1 address decoding with validity bit
#[test]
fn test_aedat_3_1_address_decoding() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("test_aedat_3_1_address.aedat");

    // Create test AEDAT 3.1 file
    let mut file = File::create(&file_path).unwrap();
    writeln!(file, "#!AER-DAT3.1").unwrap();
    writeln!(file, "# sizeX 346").unwrap();
    writeln!(file, "# sizeY 240").unwrap();
    writeln!(file, "").unwrap(); // Add empty line to mark end of header

    // Test various coordinate combinations
    // AEDAT 3.1 format: validity (1 bit) + polarity (1 bit) + y (15 bits) + x (15 bits)
    let test_cases = vec![
        // (expected_x, expected_y, expected_polarity, timestamp)
        (0, 0, 1, 1000),
        (0, 0, -1, 2000),
        (1, 1, 1, 3000),
        (345, 239, 1, 4000),  // Max resolution
        (173, 120, -1, 5000), // Mid-range coordinates
        (50, 100, 1, 6000),   // Arbitrary coordinates
    ];

    for (x, y, polarity, timestamp) in &test_cases {
        // Create address according to AEDAT 3.1 specification
        let address = ((*x as u32) << 17)
            | ((*y as u32) << 2)
            | (if *polarity == 1 { 1 } else { 0 } << 1)
            | 1; // validity bit = 1
        file.write_all(&(*timestamp as u32).to_le_bytes()).unwrap(); // Little-endian timestamp
        file.write_all(&address.to_le_bytes()).unwrap(); // Little-endian address
    }

    // Test reading with disabled validation for test data
    let config = AedatConfig {
        validate_timestamps: false,
        validate_coordinates: false,
        validate_polarity: false,
        skip_invalid_events: true,
        max_events: None,
        max_resolution: None,
    };
    let reader = AedatReader::with_config(config);
    let (events, metadata) = reader.read_file(&file_path).unwrap();

    assert_eq!(metadata.version, Some(AedatVersion::V3_1));
    assert_eq!(events.len(), test_cases.len());

    // Verify each event
    for (i, (expected_x, expected_y, expected_polarity, expected_timestamp)) in
        test_cases.iter().enumerate()
    {
        let event = &events[i];
        assert_eq!(
            event.x, *expected_x as u16,
            "Event {}: X coordinate mismatch",
            i
        );
        assert_eq!(
            event.y, *expected_y as u16,
            "Event {}: Y coordinate mismatch",
            i
        );
        assert_eq!(
            event.polarity, *expected_polarity,
            "Event {}: Polarity mismatch",
            i
        );
        assert_eq!(
            event.t, *expected_timestamp as f64,
            "Event {}: Timestamp mismatch",
            i
        );
    }
}

/// Test AEDAT 3.1 validity bit handling
#[test]
fn test_aedat_3_1_validity_bit() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("test_aedat_3_1_validity.aedat");

    // Create test AEDAT 3.1 file
    let mut file = File::create(&file_path).unwrap();
    writeln!(file, "#!AER-DAT3.1").unwrap();
    writeln!(file, "").unwrap(); // Add empty line to mark end of header

    // Create events with mixed validity bits
    let test_events = vec![
        // (x, y, polarity, timestamp, valid)
        (10, 20, 1, 1000, true),   // Valid event
        (30, 40, -1, 2000, false), // Invalid event
        (50, 60, 1, 3000, true),   // Valid event
    ];

    for (x, y, polarity, timestamp, valid) in &test_events {
        let address = ((*x as u32) << 17)
            | ((*y as u32) << 2)
            | (if *polarity == 1 { 1 } else { 0 } << 1)
            | if *valid { 1 } else { 0 }; // validity bit
        file.write_all(&(*timestamp as u32).to_le_bytes()).unwrap();
        file.write_all(&address.to_le_bytes()).unwrap();
    }

    // Test with skip_invalid_events = true
    let config = AedatConfig {
        validate_timestamps: false,
        validate_coordinates: false,
        validate_polarity: false,
        skip_invalid_events: true,
        max_events: None,
        max_resolution: None,
    };
    let reader = AedatReader::with_config(config);
    let (events, _) = reader.read_file(&file_path).unwrap();

    // Should only have valid events
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].x, 10);
    assert_eq!(events[0].y, 20);
    assert_eq!(events[1].x, 50);
    assert_eq!(events[1].y, 60);
}

/// Test AEDAT 4.0 address decoding with DV framework format
#[test]
fn test_aedat_4_0_address_decoding() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("test_aedat_4_0_address.aedat");

    // Create test AEDAT 4.0 file
    let mut file = File::create(&file_path).unwrap();
    writeln!(file, "AEDAT4").unwrap();
    writeln!(file, "# sizeX 640").unwrap();
    writeln!(file, "# sizeY 480").unwrap();

    // Write test packet header (28 bytes)
    let packet_type = 1u16; // Polarity events
    let packet_size = 32u32; // 4 events * 8 bytes each

    file.write_all(&packet_type.to_le_bytes()).unwrap();
    file.write_all(&[0u8; 2]).unwrap(); // Reserved
    file.write_all(&packet_size.to_le_bytes()).unwrap();
    file.write_all(&[0u8; 20]).unwrap(); // Rest of header

    // Test various coordinate combinations
    // AEDAT 4.0 format: timestamp (32-bit) + x (16-bit) + y (16-bit with MSB as polarity)
    let test_cases = vec![
        // (expected_x, expected_y, expected_polarity, timestamp)
        (0, 0, 1, 1000),
        (0, 0, -1, 2000),
        (100, 200, 1, 3000),
        (639, 479, -1, 4000), // Max resolution
    ];

    for (x, y, polarity, timestamp) in &test_cases {
        // Create event according to AEDAT 4.0 specification
        let y_with_polarity = (*y as u16) | if *polarity == 1 { 0x8000 } else { 0 };

        file.write_all(&(*timestamp as u32).to_le_bytes()).unwrap();
        file.write_all(&(*x as u16).to_le_bytes()).unwrap();
        file.write_all(&y_with_polarity.to_le_bytes()).unwrap();
    }

    // Test reading
    let reader = AedatReader::new();
    let (events, metadata) = reader.read_file(&file_path).unwrap();

    assert_eq!(metadata.version, Some(AedatVersion::V4_0));
    assert_eq!(events.len(), test_cases.len());

    // Verify each event
    for (i, (expected_x, expected_y, expected_polarity, expected_timestamp)) in
        test_cases.iter().enumerate()
    {
        let event = &events[i];
        assert_eq!(
            event.x, *expected_x as u16,
            "Event {}: X coordinate mismatch",
            i
        );
        assert_eq!(
            event.y, *expected_y as u16,
            "Event {}: Y coordinate mismatch",
            i
        );
        assert_eq!(
            event.polarity, *expected_polarity,
            "Event {}: Polarity mismatch",
            i
        );
        assert_eq!(
            event.t, *expected_timestamp as f64,
            "Event {}: Timestamp mismatch",
            i
        );
    }
}

/// Test coordinate bounds checking
#[test]
fn test_coordinate_bounds_checking() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("test_bounds.aedat");

    // Create test AEDAT 1.0 file with out-of-bounds coordinates
    let mut file = File::create(&file_path).unwrap();
    writeln!(file, "#!AER-DAT1.0").unwrap();

    // Create address with maximum 7-bit coordinates (127, 127)
    let address = (127u16 << 8) | (127u16 << 1) | 1;
    file.write_all(&address.to_le_bytes()).unwrap();
    file.write_all(&1000u32.to_le_bytes()).unwrap();

    // Test with restrictive bounds (should fail)
    let config = AedatConfig {
        max_resolution: Some((100, 100)),
        validate_coordinates: true,
        skip_invalid_events: false,
        ..Default::default()
    };
    let reader = AedatReader::with_config(config);
    let result = reader.read_file(&file_path);
    assert!(result.is_err());

    // Test with sufficient bounds (should succeed)
    let config = AedatConfig {
        max_resolution: Some((128, 128)),
        validate_coordinates: true,
        skip_invalid_events: false,
        ..Default::default()
    };
    let reader = AedatReader::with_config(config);
    let (events, _) = reader.read_file(&file_path).unwrap();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].x, 127);
    assert_eq!(events[0].y, 127);
}

/// Test polarity validation
#[test]
fn test_polarity_validation() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("test_polarity.aedat");

    // Create test AEDAT 1.0 file
    let mut file = File::create(&file_path).unwrap();
    writeln!(file, "#!AER-DAT1.0").unwrap();

    // Create events with both polarities
    let test_events = vec![
        (10, 20, 1, 1000), // ON polarity
        (30, 40, 0, 2000), // OFF polarity (should become -1)
    ];

    for (x, y, polarity, timestamp) in &test_events {
        let address = ((*y as u16) << 8) | ((*x as u16) << 1) | *polarity;
        file.write_all(&address.to_le_bytes()).unwrap();
        file.write_all(&(*timestamp as u32).to_le_bytes()).unwrap();
    }

    let config = AedatConfig {
        validate_polarity: true,
        skip_invalid_events: false,
        ..Default::default()
    };
    let reader = AedatReader::with_config(config);
    let (events, _) = reader.read_file(&file_path).unwrap();

    assert_eq!(events.len(), 2);
    assert_eq!(events[0].polarity, 1); // ON polarity
    assert_eq!(events[1].polarity, -1); // OFF polarity converted to -1
}

/// Test endianness handling
#[test]
fn test_endianness_handling() {
    let temp_dir = TempDir::new().unwrap();

    // Test AEDAT 1.0 (little-endian) vs AEDAT 2.0 (big-endian)
    let file_path_1 = temp_dir.path().join("test_endian_1.aedat");
    let file_path_2 = temp_dir.path().join("test_endian_2.aedat");

    // Create AEDAT 1.0 file (little-endian)
    let mut file1 = File::create(&file_path_1).unwrap();
    writeln!(file1, "#!AER-DAT1.0").unwrap();
    let address1 = (10u16 << 8) | (20u16 << 1) | 1;
    file1.write_all(&address1.to_le_bytes()).unwrap();
    file1.write_all(&1000u32.to_le_bytes()).unwrap();

    // Create AEDAT 2.0 file (big-endian)
    let mut file2 = File::create(&file_path_2).unwrap();
    writeln!(file2, "#!AER-DAT2.0").unwrap();
    writeln!(file2, "").unwrap(); // Add empty line to mark end of header
    let address2 = (10u32 << 16) | (20u32 << 1) | 1;
    file2.write_all(&1000u32.to_be_bytes()).unwrap();
    file2.write_all(&address2.to_be_bytes()).unwrap();

    let config = AedatConfig {
        validate_timestamps: false,
        validate_coordinates: false,
        validate_polarity: false,
        skip_invalid_events: true,
        ..Default::default()
    };
    let reader = AedatReader::with_config(config);

    // Both files should produce the same event
    let (events1, _) = reader.read_file(&file_path_1).unwrap();
    let (events2, _) = reader.read_file(&file_path_2).unwrap();

    assert_eq!(events1.len(), 1);
    assert_eq!(events2.len(), 1);
    assert_eq!(events1[0].x, 20);
    assert_eq!(events1[0].y, 10);
    assert_eq!(events2[0].x, 20);
    assert_eq!(events2[0].y, 10);
    assert_eq!(events1[0].polarity, 1);
    assert_eq!(events2[0].polarity, 1);
}
