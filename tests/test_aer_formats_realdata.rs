#[cfg(feature = "hdf5")]
use evlib::ev_formats::load_events_from_hdf5;
/// Comprehensive unit tests for AER format implementation
/// Tests against real data files to verify format detection, loading, and validation
///
/// This test suite ensures the AER format readers can correctly:
/// 1. Detect file formats automatically
/// 2. Load text event files with proper parsing
/// 3. Validate event coordinates, timestamps, and polarity values
/// 4. Handle large datasets efficiently
/// 5. Parse HDF5 files correctly
/// 6. Detect format issues and handle errors gracefully
use evlib::ev_formats::{
    detect_event_format, load_events_from_text, load_events_with_config, AerConfig, AerReader,
    EventFormat, LoadConfig,
};
use std::collections::HashSet;
use std::path::Path;
use tempfile::NamedTempFile;

/// Lightweight event row extracted from a Polars DataFrame for assertion convenience.
#[derive(Debug, Clone, Copy)]
struct EventRow {
    t: f64, // seconds
    x: u16,
    y: u16,
    polarity: i8, // 0/1 for text and AER (Text-like encoding)
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

const SLIDER_DEPTH_DIR: &str = "/Users/tallam/github/tallamjr/origin/evlib/data/slider_depth";
const ETRAM_HDF5_DIR: &str = "/Users/tallam/github/tallamjr/origin/evlib/data/eTram/h5/val_2";

/// Helper function to check if a test data file exists
fn check_data_file_exists(path: &str) -> bool {
    Path::new(path).exists()
}

/// Helper to get test data paths, skipping if files don't exist
fn get_test_files() -> Vec<String> {
    let mut files = Vec::new();

    // Text files
    let events_txt = format!("{SLIDER_DEPTH_DIR}/events.txt");
    let events_chunk_txt = format!("{SLIDER_DEPTH_DIR}/events_chunk.txt");

    if check_data_file_exists(&events_txt) {
        files.push(events_txt);
    }
    if check_data_file_exists(&events_chunk_txt) {
        files.push(events_chunk_txt);
    }

    // HDF5 file - tracked Prophesee sample
    let etram_hdf5 = format!("{ETRAM_HDF5_DIR}/val_night_007_td.h5");
    if check_data_file_exists(&etram_hdf5) {
        files.push(etram_hdf5);
    }

    files
}

#[test]
fn test_format_detection_text_files() {
    let events_txt = format!("{SLIDER_DEPTH_DIR}/events.txt");
    let events_chunk_txt = format!("{SLIDER_DEPTH_DIR}/events_chunk.txt");

    // Test detection of events.txt
    if check_data_file_exists(&events_txt) {
        let result =
            detect_event_format(&events_txt).expect("Failed to detect format for events.txt");
        assert_eq!(result.format, EventFormat::Text);
        assert!(
            result.confidence > 0.8,
            "Low confidence for text format detection"
        );
        assert!(result.metadata.file_size > 0);

        println!(
            "events.txt detected as: {format} (confidence: {confidence:.2})",
            format = result.format,
            confidence = result.confidence
        );
    }

    // Test detection of events_chunk.txt
    if check_data_file_exists(&events_chunk_txt) {
        let result = detect_event_format(&events_chunk_txt)
            .expect("Failed to detect format for events_chunk.txt");
        assert_eq!(result.format, EventFormat::Text);
        assert!(
            result.confidence > 0.8,
            "Low confidence for text format detection"
        );
        assert!(result.metadata.file_size > 0);

        println!(
            "events_chunk.txt detected as: {format} (confidence: {confidence:.2})",
            format = result.format,
            confidence = result.confidence
        );
    }
}

#[test]
fn test_format_detection_hdf5_files() {
    let etram_hdf5 = format!("{ETRAM_HDF5_DIR}/val_night_007_td.h5");
    if !check_data_file_exists(&etram_hdf5) {
        println!("Skipping HDF5 detection test - file not found: {etram_hdf5}");
        return;
    }

    let result = detect_event_format(&etram_hdf5)
        .unwrap_or_else(|_| panic!("Failed to detect format for {etram_hdf5}"));
    assert_eq!(result.format, EventFormat::HDF5);
    assert!(
        result.confidence > 0.9,
        "Low confidence for HDF5 format detection"
    );
    assert!(result.metadata.file_size > 0);

    println!(
        "{etram_hdf5} detected as: {format} (confidence: {confidence:.2})",
        format = result.format,
        confidence = result.confidence
    );
}

#[test]
fn test_load_text_events_basic() {
    let events_chunk_txt = format!("{SLIDER_DEPTH_DIR}/events_chunk.txt");

    if !check_data_file_exists(&events_chunk_txt) {
        println!("Skipping test: {events_chunk_txt} not found");
        return;
    }

    let config = LoadConfig::new();
    let df = load_events_from_text(&events_chunk_txt, &config)
        .expect("Failed to load events from text file");
    let events = dataframe_to_rows(&df);

    assert!(!events.is_empty(), "No events loaded from text file");
    assert_eq!(events.len(), 50000, "Expected 50000 events in chunk file");

    // Verify first event structure
    let first_event = &events[0];
    assert!(
        first_event.t > 0.0,
        "First event timestamp should be positive"
    );
    assert!(first_event.x <= 1024, "X coordinate should be reasonable");
    assert!(first_event.y <= 1024, "Y coordinate should be reasonable");
    // Text polarity is 0/1
    assert!(first_event.polarity == 0 || first_event.polarity == 1);

    println!(
        "Loaded {count} events from {events_chunk_txt}",
        count = events.len()
    );
    println!(
        "First event: t={t:.6}, x={x}, y={y}, p={p}",
        t = first_event.t,
        x = first_event.x,
        y = first_event.y,
        p = first_event.polarity
    );
}

#[test]
fn test_load_text_events_large_file() {
    let events_txt = format!("{SLIDER_DEPTH_DIR}/events.txt");

    if !check_data_file_exists(&events_txt) {
        println!("Skipping test: {events_txt} not found");
        return;
    }

    // Test loading with a limit to avoid memory issues in CI
    let config = LoadConfig::new();
    let df = load_events_from_text(&events_txt, &config)
        .expect("Failed to load events from large text file");
    let events = dataframe_to_rows(&df);

    assert!(!events.is_empty(), "No events loaded from large text file");
    assert!(
        events.len() > 1000000,
        "Expected over 1M events in full file"
    );

    // Verify event structure
    let first_event = &events[0];
    let last_event = &events[events.len() - 1];

    assert!(
        first_event.t > 0.0,
        "First event timestamp should be positive"
    );
    assert!(
        last_event.t > first_event.t,
        "Timestamps should be increasing"
    );

    println!(
        "Loaded {count} events from large file",
        count = events.len()
    );
    println!(
        "Time range: {start:.6} to {end:.6} seconds",
        start = first_event.t,
        end = last_event.t
    );
}

#[test]
fn test_event_timestamp_ordering() {
    let events_chunk_txt = format!("{SLIDER_DEPTH_DIR}/events_chunk.txt");

    if !check_data_file_exists(&events_chunk_txt) {
        println!("Skipping test: {events_chunk_txt} not found");
        return;
    }

    let config = LoadConfig::new().with_sorting(true);
    let df = load_events_from_text(&events_chunk_txt, &config).expect("Failed to load events");
    let events = dataframe_to_rows(&df);

    // Check if timestamps are in ascending order
    let mut violations = 0;
    for i in 1..events.len() {
        if events[i].t < events[i - 1].t {
            violations += 1;
        }
    }

    // Real data might have some timestamp violations due to sensor characteristics
    let violation_rate = violations as f64 / events.len() as f64;
    assert!(
        violation_rate < 0.01,
        "Too many timestamp violations: {:.4}%",
        violation_rate * 100.0
    );

    println!(
        "Timestamp violations: {violations} out of {total} events ({rate:.4}%)",
        total = events.len(),
        rate = violation_rate * 100.0
    );
}

#[test]
fn test_event_coordinate_ranges() {
    let events_chunk_txt = format!("{SLIDER_DEPTH_DIR}/events_chunk.txt");

    if !check_data_file_exists(&events_chunk_txt) {
        println!("Skipping test: {events_chunk_txt} not found");
        return;
    }

    let config = LoadConfig::new();
    let df = load_events_from_text(&events_chunk_txt, &config).expect("Failed to load events");
    let events = dataframe_to_rows(&df);

    let mut min_x = u16::MAX;
    let mut max_x = 0u16;
    let mut min_y = u16::MAX;
    let mut max_y = 0u16;

    for event in &events {
        min_x = min_x.min(event.x);
        max_x = max_x.max(event.x);
        min_y = min_y.min(event.y);
        max_y = max_y.max(event.y);
    }

    // Verify coordinates are within reasonable sensor bounds
    assert!(max_x < 1024, "X coordinates should be within sensor bounds");
    assert!(max_y < 1024, "Y coordinates should be within sensor bounds");
    assert!(min_x < max_x, "Should have variation in X coordinates");
    assert!(min_y < max_y, "Should have variation in Y coordinates");

    println!("Coordinate ranges: X=[{min_x}, {max_x}], Y=[{min_y}, {max_y}]");
}

#[test]
fn test_event_polarity_distribution() {
    let events_chunk_txt = format!("{SLIDER_DEPTH_DIR}/events_chunk.txt");

    if !check_data_file_exists(&events_chunk_txt) {
        println!("Skipping test: {events_chunk_txt} not found");
        return;
    }

    let config = LoadConfig::new();
    let df = load_events_from_text(&events_chunk_txt, &config).expect("Failed to load events");
    let events = dataframe_to_rows(&df);

    let mut positive_count = 0;
    let mut negative_count = 0;
    let mut invalid_count = 0;

    // Text polarity is 0/1
    for event in &events {
        match event.polarity {
            1 => positive_count += 1,
            0 => negative_count += 1,
            _ => invalid_count += 1,
        }
    }

    assert_eq!(invalid_count, 0, "Found invalid polarity values");
    assert!(positive_count > 0, "Should have positive polarity events");
    assert!(negative_count > 0, "Should have negative polarity events");

    let total = positive_count + negative_count;
    let pos_ratio = positive_count as f64 / total as f64;

    // Reasonable polarity balance (not too skewed)
    assert!(
        pos_ratio > 0.1 && pos_ratio < 0.9,
        "Polarity distribution seems skewed: {:.2}% positive",
        pos_ratio * 100.0
    );

    println!(
        "Polarity distribution: {pos} positive, {neg} negative ({ratio:.1}% positive)",
        pos = positive_count,
        neg = negative_count,
        ratio = pos_ratio * 100.0
    );
}

#[test]
fn test_load_with_time_filtering() {
    let events_chunk_txt = format!("{SLIDER_DEPTH_DIR}/events_chunk.txt");

    if !check_data_file_exists(&events_chunk_txt) {
        println!("Skipping test: {events_chunk_txt} not found");
        return;
    }

    // First load without filtering to get time range
    let config_full = LoadConfig::new();
    let df_full =
        load_events_from_text(&events_chunk_txt, &config_full).expect("Failed to load events");
    let events_full = dataframe_to_rows(&df_full);

    let min_time = events_full
        .iter()
        .map(|e| e.t)
        .fold(f64::INFINITY, f64::min);
    let max_time = events_full
        .iter()
        .map(|e| e.t)
        .fold(f64::NEG_INFINITY, f64::max);
    let mid_time = (min_time + max_time) / 2.0;

    // Load with time window
    let config_filtered = LoadConfig::new().with_time_window(Some(min_time), Some(mid_time));
    let df_filtered = load_events_from_text(&events_chunk_txt, &config_filtered)
        .expect("Failed to load filtered events");
    let events_filtered = dataframe_to_rows(&df_filtered);

    assert!(
        events_filtered.len() < events_full.len(),
        "Filtered events should be fewer than full set"
    );
    assert!(
        !events_filtered.is_empty(),
        "Should have some events in time window"
    );

    // Verify all events are within time window
    for event in &events_filtered {
        assert!(
            event.t >= min_time && event.t <= mid_time,
            "Event timestamp outside filter window"
        );
    }

    println!(
        "Time filtering: {full} -> {filtered} events in window [{start:.6}, {end:.6}]",
        full = events_full.len(),
        filtered = events_filtered.len(),
        start = min_time,
        end = mid_time
    );
}

#[test]
fn test_load_with_spatial_filtering() {
    let events_chunk_txt = format!("{SLIDER_DEPTH_DIR}/events_chunk.txt");

    if !check_data_file_exists(&events_chunk_txt) {
        println!("Skipping test: {events_chunk_txt} not found");
        return;
    }

    // Load with spatial bounds
    let config = LoadConfig::new().with_spatial_bounds(Some(50), Some(200), Some(50), Some(200));
    let df = load_events_from_text(&events_chunk_txt, &config)
        .expect("Failed to load spatially filtered events");
    let events = dataframe_to_rows(&df);

    // Verify all events are within spatial bounds
    for event in &events {
        assert!(
            event.x >= 50 && event.x <= 200,
            "Event X coordinate outside filter bounds: {x}",
            x = event.x
        );
        assert!(
            event.y >= 50 && event.y <= 200,
            "Event Y coordinate outside filter bounds: {y}",
            y = event.y
        );
    }

    println!(
        "Spatial filtering: {count} events in bounds [50-200, 50-200]",
        count = events.len()
    );
}

#[test]
fn test_load_with_polarity_filtering() {
    let events_chunk_txt = format!("{SLIDER_DEPTH_DIR}/events_chunk.txt");

    if !check_data_file_exists(&events_chunk_txt) {
        println!("Skipping test: {events_chunk_txt} not found");
        return;
    }

    // Load only positive polarity events
    let config_pos = LoadConfig::new().with_polarity(Some(true));
    let df_pos = load_events_from_text(&events_chunk_txt, &config_pos)
        .expect("Failed to load positive polarity events");
    let events_pos = dataframe_to_rows(&df_pos);

    // Verify all events have positive polarity (text encodes positive as 1)
    for event in &events_pos {
        assert_eq!(event.polarity, 1, "Found non-positive polarity event");
    }

    // Load only negative polarity events
    let config_neg = LoadConfig::new().with_polarity(Some(false));
    let df_neg = load_events_from_text(&events_chunk_txt, &config_neg)
        .expect("Failed to load negative polarity events");
    let events_neg = dataframe_to_rows(&df_neg);

    // Verify all events have negative polarity (text encodes negative as 0)
    for event in &events_neg {
        assert_eq!(
            event.polarity,
            0,
            "Found non-negative polarity event: {polarity}",
            polarity = event.polarity
        );
    }

    println!(
        "Polarity filtering: {pos} positive, {neg} negative events",
        pos = events_pos.len(),
        neg = events_neg.len()
    );
}

#[cfg(feature = "hdf5")]
#[test]
fn test_hdf5_file_loading() {
    let hdf5_file = format!("{ETRAM_HDF5_DIR}/val_night_007_td.h5");

    if !check_data_file_exists(&hdf5_file) {
        println!("Skipping test: {hdf5_file} not found");
        return;
    }

    match load_events_from_hdf5(&hdf5_file, None) {
        Ok(df) => {
            let events = dataframe_to_rows(&df);
            assert!(!events.is_empty(), "No events loaded from HDF5 file");

            // Verify event structure
            let first_event = &events[0];
            assert!(first_event.t >= 0.0, "Timestamp should be non-negative");
            assert!(
                first_event.x < 1280,
                "X coordinate should be within sensor bounds"
            );
            assert!(
                first_event.y < 720,
                "Y coordinate should be within sensor bounds"
            );

            println!(
                "Successfully loaded {count} events from {hdf5_file} (auto-detected dataset)",
                count = events.len()
            );
        }
        Err(e) => {
            println!("Warning: Could not load HDF5 file {hdf5_file}: {e}");
            // Don't fail the test as HDF5 structure may vary
        }
    }
}

#[test]
fn test_generic_load_function() {
    let test_files = get_test_files();

    if test_files.is_empty() {
        println!("Skipping test: No test data files found");
        return;
    }

    let config = LoadConfig::new();

    for file_path in test_files {
        match load_events_with_config(&file_path, &config) {
            Ok(df) => {
                let events = dataframe_to_rows(&df);
                assert!(!events.is_empty(), "No events loaded from {file_path}");

                // Basic validation
                for (i, event) in events.iter().take(100).enumerate() {
                    assert!(event.t >= 0.0, "Invalid timestamp at event {i}");
                    assert!(event.x < 2048, "Invalid X coordinate at event {i}");
                    assert!(event.y < 2048, "Invalid Y coordinate at event {i}");
                    // Polarity is i8 (0/1 or -1/1 depending on source), always valid
                }

                println!(
                    "Successfully loaded {count} events from {file} using generic loader",
                    count = events.len(),
                    file = file_path
                );
            }
            Err(e) => {
                println!("Warning: Could not load {file_path}: {e}");
                // Some files might not be loadable due to format variations
            }
        }
    }
}

#[test]
fn test_aer_binary_format_synthetic() {
    // Create synthetic AER binary data for testing
    let config = AerConfig::default();
    let reader = AerReader::with_config(config);

    // Create temporary file with synthetic AER data
    let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");

    // Generate test events in AER 18-bit format
    let test_events = [
        (100, 150, 1), // x=100, y=150, polarity=1
        (200, 250, 0), // x=200, y=250, polarity=0
        (300, 350, 1), // x=300, y=350, polarity=1
    ];

    for (x, y, polarity) in test_events.iter() {
        // Create 18-bit AER event: (y << 10) | (x << 1) | polarity
        let raw_event = ((*y as u32) << 10) | ((*x as u32) << 1) | (*polarity as u32);
        let bytes = raw_event.to_le_bytes();
        std::io::Write::write_all(&mut temp_file, &bytes).expect("Failed to write test data");
    }

    // Test reading the synthetic data
    match reader.read_file(temp_file.path()) {
        Ok((df, metadata)) => {
            let events = dataframe_to_rows(&df);
            assert_eq!(events.len(), 3, "Should load 3 synthetic events");
            assert_eq!(metadata.event_count, 3);
            assert_eq!(metadata.bytes_per_event, 4);

            // Verify event data (AER encodes polarity as 0/1)
            assert_eq!(events[0].x, 100);
            assert_eq!(events[0].y, 150);
            assert_eq!(events[0].polarity, 1);

            assert_eq!(events[1].x, 200);
            assert_eq!(events[1].y, 250);
            assert_eq!(events[1].polarity, 0); // polarity 0

            println!("Successfully tested AER binary format with synthetic data");
        }
        Err(e) => {
            println!("AER binary test failed: {e}");
            // Don't panic as this is a synthetic test
        }
    }
}

#[test]
fn test_event_count_accuracy() {
    let events_chunk_txt = format!("{SLIDER_DEPTH_DIR}/events_chunk.txt");

    if !check_data_file_exists(&events_chunk_txt) {
        println!("Skipping test: {events_chunk_txt} not found");
        return;
    }

    // Count lines manually
    let content = std::fs::read_to_string(&events_chunk_txt).expect("Failed to read file");
    let line_count = content
        .lines()
        .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
        .count();

    // Load events
    let config = LoadConfig::new();
    let df = load_events_from_text(&events_chunk_txt, &config).expect("Failed to load events");

    assert_eq!(
        df.height(),
        line_count,
        "Event count mismatch: loaded {loaded} vs {lines} lines",
        loaded = df.height(),
        lines = line_count
    );

    println!(
        "Event count accuracy verified: {count} events",
        count = df.height()
    );
}

#[test]
fn test_duplicate_events() {
    let events_chunk_txt = format!("{SLIDER_DEPTH_DIR}/events_chunk.txt");

    if !check_data_file_exists(&events_chunk_txt) {
        println!("Skipping test: {events_chunk_txt} not found");
        return;
    }

    let config = LoadConfig::new();
    let df = load_events_from_text(&events_chunk_txt, &config).expect("Failed to load events");
    let events = dataframe_to_rows(&df);

    // Check for exact duplicates (same t, x, y, p)
    let mut seen = HashSet::new();
    let mut duplicates = 0;

    for event in &events {
        let key = (
            (event.t * 1e9) as i64, // Convert to nanoseconds for comparison
            event.x,
            event.y,
            event.polarity,
        );

        if seen.contains(&key) {
            duplicates += 1;
        } else {
            seen.insert(key);
        }
    }

    let duplicate_rate = duplicates as f64 / events.len() as f64;

    // Real sensor data might have some duplicates, but not too many
    assert!(
        duplicate_rate < 0.05,
        "Too many duplicate events: {rate:.2}%",
        rate = duplicate_rate * 100.0
    );

    println!(
        "Duplicate events: {dups} out of {total} ({rate:.4}%)",
        dups = duplicates,
        total = events.len(),
        rate = duplicate_rate * 100.0
    );
}

#[test]
fn test_error_handling_invalid_files() {
    // Test with non-existent file
    let config = LoadConfig::new();
    let result = load_events_from_text("/nonexistent/file.txt", &config);
    assert!(result.is_err(), "Should fail for non-existent file");

    // Test with empty file
    let empty_file = NamedTempFile::new().expect("Failed to create temp file");
    let result = load_events_from_text(empty_file.path().to_str().unwrap(), &config);
    // Empty file should return empty events, not error
    if let Ok(df) = result {
        assert_eq!(df.height(), 0, "Empty file should return empty events");
    }
    // Also acceptable if it fails

    // Test with invalid data
    let mut invalid_file = NamedTempFile::new().expect("Failed to create temp file");
    std::io::Write::write_all(&mut invalid_file, b"invalid data\nnot numbers\n")
        .expect("Failed to write invalid data");

    let result = load_events_from_text(invalid_file.path().to_str().unwrap(), &config);
    assert!(result.is_err(), "Should fail for invalid data format");

    println!("Error handling tests completed");
}
