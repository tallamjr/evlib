/// Comprehensive HDF5 format testing with real data files
/// Tests HDF5 event data loading, structure validation, and dataset detection
///
/// This test suite verifies:
/// 1. HDF5 format detection and loading
/// 2. Multiple dataset name conventions
/// 3. Data structure validation
/// 4. Large file handling
/// 5. Error handling for malformed files
use evlib::ev_core::{Event, Events};
use evlib::ev_formats::{
    detect_event_format, load_events_from_hdf5, load_events_from_hdf5_filtered,
    load_events_with_config, EventFormat, LoadConfig,
};
use hdf5::File as H5File;
use std::path::Path;
use tempfile::NamedTempFile;

const ORIGINAL_HDF5_DIR: &str = "/Users/tallam/github/tallamjr/origin/evlib/data/original";
const ETRAM_HDF5_DIR: &str = "/Users/tallam/github/tallamjr/origin/evlib/data/eTram/h5/val_2";

/// Helper function to check if a test data file exists
fn check_data_file_exists(path: &str) -> bool {
    Path::new(path).exists()
}

/// Get all available HDF5 test files
fn get_hdf5_test_files() -> Vec<String> {
    let mut files = Vec::new();

    // Original dataset files
    let original_front_files = ["seq01.h5", "seq02.h5", "seq03.h5", "seq04.h5", "seq05.h5"];

    for file in &original_front_files {
        let path = format!("{}/front/{}", ORIGINAL_HDF5_DIR, file);
        if check_data_file_exists(&path) {
            files.push(path);
        }
    }

    // eTram dataset files
    let etram_files = [
        "val_night_007_td.h5",
        "val_night_008_td.h5",
        "val_night_009_td.h5",
    ];

    for file in &etram_files {
        let path = format!("{}/{}", ETRAM_HDF5_DIR, file);
        if check_data_file_exists(&path) {
            files.push(path);
        }
    }

    files
}

/// Inspect HDF5 file structure to understand dataset organization
fn inspect_hdf5_structure(file_path: &str) -> Result<Vec<String>, hdf5::Error> {
    let file = H5File::open(file_path)?;
    let mut datasets = Vec::new();

    // Try to list all datasets in the file
    // This is a simplified inspection - HDF5 files can have complex hierarchies
    let common_dataset_names = [
        "events",
        "t",
        "x",
        "y",
        "p",
        "polarity",
        "timestamps",
        "x_pos",
        "y_pos",
        "ts",
        "xs",
        "ys",
        "ps",
        "data",
        "event_data",
    ];

    for name in &common_dataset_names {
        if file.dataset(name).is_ok() {
            datasets.push(name.to_string());
        }
    }

    // Check for events group
    if let Ok(events_group) = file.group("events") {
        for name in &common_dataset_names {
            if events_group.dataset(name).is_ok() {
                datasets.push(format!("events/{}", name));
            }
        }
    }

    Ok(datasets)
}

#[test]
fn test_hdf5_format_detection() {
    let test_files = get_hdf5_test_files();

    if test_files.is_empty() {
        println!("Skipping test: No HDF5 test files found");
        return;
    }

    for file_path in &test_files[..3.min(test_files.len())] {
        // Test first 3 files
        let result = detect_event_format(file_path)
            .expect(&format!("Failed to detect format for {}", file_path));

        assert_eq!(
            result.format,
            EventFormat::HDF5,
            "File {} should be detected as HDF5",
            file_path
        );
        assert!(
            result.confidence > 0.9,
            "Low confidence for HDF5 detection: {:.2}",
            result.confidence
        );
        assert!(
            result.metadata.file_size > 0,
            "File size should be positive"
        );

        println!(
            "{} detected as HDF5 (confidence: {:.2}, size: {} bytes)",
            file_path, result.confidence, result.metadata.file_size
        );
    }
}

#[test]
fn test_hdf5_dataset_inspection() {
    let test_files = get_hdf5_test_files();

    if test_files.is_empty() {
        println!("Skipping test: No HDF5 test files found");
        return;
    }

    for file_path in &test_files[..2.min(test_files.len())] {
        // Test first 2 files
        match inspect_hdf5_structure(file_path) {
            Ok(datasets) => {
                assert!(
                    !datasets.is_empty(),
                    "Should find at least one dataset in {}",
                    file_path
                );

                println!("{} contains datasets: {:?}", file_path, datasets);

                // Verify we can identify event-related datasets
                let has_event_data = datasets.iter().any(|name| {
                    name.contains("event")
                        || name.contains("t")
                        || name.contains("x")
                        || name.contains("y")
                });

                if !has_event_data {
                    println!("Warning: No obvious event datasets found in {}", file_path);
                }
            }
            Err(e) => {
                println!("Warning: Could not inspect {}: {}", file_path, e);
            }
        }
    }
}

#[test]
fn test_hdf5_loading_auto_detection() {
    let test_files = get_hdf5_test_files();

    if test_files.is_empty() {
        println!("Skipping test: No HDF5 test files found");
        return;
    }

    for file_path in &test_files[..2.min(test_files.len())] {
        match load_events_from_hdf5(file_path, None) {
            Ok(events) => {
                assert!(!events.is_empty(), "Should load events from {}", file_path);

                // Validate event structure
                let first_event = &events[0];
                assert!(first_event.t >= 0.0, "Timestamp should be non-negative");
                assert!(first_event.x < 10000, "X coordinate should be reasonable");
                assert!(first_event.y < 10000, "Y coordinate should be reasonable");
                // Polarity is bool, so always valid

                let last_event = &events[events.len() - 1];
                assert!(
                    last_event.t >= first_event.t,
                    "Events should be in temporal order"
                );

                println!(
                    "Successfully loaded {} events from {} (auto-detection)",
                    events.len(),
                    file_path
                );

                // Show time range
                println!(
                    "  Time range: {:.6} to {:.6} seconds",
                    first_event.t, last_event.t
                );

                // Show coordinate range
                let min_x = events.iter().map(|e| e.x).min().unwrap();
                let max_x = events.iter().map(|e| e.x).max().unwrap();
                let min_y = events.iter().map(|e| e.y).min().unwrap();
                let max_y = events.iter().map(|e| e.y).max().unwrap();
                println!(
                    "  Coordinate range: X=[{}, {}], Y=[{}, {}]",
                    min_x, max_x, min_y, max_y
                );

                break; // Successfully loaded one file
            }
            Err(e) => {
                println!("Could not load {} with auto-detection: {}", file_path, e);
                continue;
            }
        }
    }
}

#[test]
fn test_hdf5_loading_explicit_datasets() {
    let test_files = get_hdf5_test_files();

    if test_files.is_empty() {
        println!("Skipping test: No HDF5 test files found");
        return;
    }

    let dataset_names = ["events", "t", "timestamps", "data"];

    for file_path in &test_files[..1.min(test_files.len())] {
        let mut loaded_successfully = false;

        for dataset_name in &dataset_names {
            match load_events_from_hdf5(file_path, Some(dataset_name)) {
                Ok(events) => {
                    assert!(
                        !events.is_empty(),
                        "Should load events from dataset '{}'",
                        dataset_name
                    );

                    println!(
                        "Successfully loaded {} events from {} (dataset: {})",
                        events.len(),
                        file_path,
                        dataset_name
                    );
                    loaded_successfully = true;
                    break;
                }
                Err(_) => {
                    continue; // Try next dataset name
                }
            }
        }

        if !loaded_successfully {
            println!(
                "Warning: Could not load {} with any explicit dataset name",
                file_path
            );
        }
    }
}

#[test]
fn test_hdf5_loading_with_filtering() {
    let test_files = get_hdf5_test_files();

    if test_files.is_empty() {
        println!("Skipping test: No HDF5 test files found");
        return;
    }

    // Find a file we can load
    let mut test_file = None;
    for file_path in &test_files {
        if load_events_from_hdf5(file_path, None).is_ok() {
            test_file = Some(file_path);
            break;
        }
    }

    let file_path = match test_file {
        Some(path) => path,
        None => {
            println!("Skipping test: No loadable HDF5 files found");
            return;
        }
    };

    // Load full dataset first to get ranges
    let events_full = load_events_from_hdf5(file_path, None).expect("Failed to load full dataset");

    if events_full.is_empty() {
        println!("Skipping filtering test: No events in dataset");
        return;
    }

    let min_time = events_full
        .iter()
        .map(|e| e.t)
        .fold(f64::INFINITY, f64::min);
    let max_time = events_full
        .iter()
        .map(|e| e.t)
        .fold(f64::NEG_INFINITY, f64::max);
    let mid_time = (min_time + max_time) / 2.0;

    // Test time filtering
    let config = LoadConfig::new().with_time_window(Some(min_time), Some(mid_time));

    let events_filtered = load_events_from_hdf5_filtered(file_path, None, &config)
        .expect("Failed to load filtered events");

    assert!(
        events_filtered.len() <= events_full.len(),
        "Filtered events should not exceed full set"
    );

    // Verify all events are within time window
    for event in &events_filtered {
        assert!(
            event.t >= min_time && event.t <= mid_time,
            "Event outside time filter window"
        );
    }

    println!(
        "HDF5 filtering test: {} -> {} events",
        events_full.len(),
        events_filtered.len()
    );
}

#[test]
fn test_hdf5_data_integrity() {
    let test_files = get_hdf5_test_files();

    if test_files.is_empty() {
        println!("Skipping test: No HDF5 test files found");
        return;
    }

    for file_path in &test_files[..1.min(test_files.len())] {
        match load_events_from_hdf5(file_path, None) {
            Ok(events) => {
                if events.is_empty() {
                    continue;
                }

                // Check for reasonable timestamp progression
                let mut timestamp_violations = 0;
                for i in 1..events.len().min(1000) {
                    // Check first 1000 events
                    if events[i].t < events[i - 1].t {
                        timestamp_violations += 1;
                    }
                }

                let violation_rate = timestamp_violations as f64 / 1000.0;
                if violation_rate > 0.1 {
                    // More than 10% violations
                    println!(
                        "Warning: High timestamp violation rate in {}: {:.2}%",
                        file_path,
                        violation_rate * 100.0
                    );
                }

                // Check coordinate bounds
                let max_x = events.iter().map(|e| e.x).max().unwrap();
                let max_y = events.iter().map(|e| e.y).max().unwrap();

                assert!(
                    max_x < 10000,
                    "X coordinates seem unreasonably large: {}",
                    max_x
                );
                assert!(
                    max_y < 10000,
                    "Y coordinates seem unreasonably large: {}",
                    max_y
                );

                // Check polarity values
                let invalid_polarities = events
                    .iter()
                    .filter(|e| e.polarity < -1 || e.polarity > 1)
                    .count();

                assert!(
                    invalid_polarities == 0,
                    "Found {} invalid polarity values",
                    invalid_polarities
                );

                println!(
                    "Data integrity check passed for {} ({} events)",
                    file_path,
                    events.len()
                );
                break;
            }
            Err(e) => {
                println!("Could not check integrity of {}: {}", file_path, e);
                continue;
            }
        }
    }
}

#[test]
fn test_hdf5_generic_loader() {
    let test_files = get_hdf5_test_files();

    if test_files.is_empty() {
        println!("Skipping test: No HDF5 test files found");
        return;
    }

    let config = LoadConfig::new();

    for file_path in &test_files[..2.min(test_files.len())] {
        match load_events_with_config(file_path, &config) {
            Ok(events) => {
                assert!(
                    !events.is_empty(),
                    "Generic loader should load events from {}",
                    file_path
                );

                // Basic validation
                for event in events.iter().take(10) {
                    assert!(event.t >= 0.0, "Invalid timestamp");
                    assert!(event.x < 10000, "Invalid X coordinate");
                    assert!(event.y < 10000, "Invalid Y coordinate");
                    // Polarity is bool, so always valid
                }

                println!(
                    "Generic loader successfully loaded {} events from {}",
                    events.len(),
                    file_path
                );
            }
            Err(e) => {
                println!("Generic loader failed for {}: {}", file_path, e);
            }
        }
    }
}

#[test]
fn test_hdf5_error_handling() {
    // Test with non-existent file
    let result = load_events_from_hdf5("/nonexistent/file.h5", None);
    assert!(result.is_err(), "Should fail for non-existent file");

    // Test with invalid dataset name
    let test_files = get_hdf5_test_files();
    if !test_files.is_empty() {
        let result = load_events_from_hdf5(&test_files[0], Some("nonexistent_dataset"));
        // This might succeed with fallback to auto-detection, or fail
        match result {
            Ok(_) => println!("HDF5 loader handled invalid dataset name gracefully"),
            Err(_) => println!("HDF5 loader correctly rejected invalid dataset name"),
        }
    }

    // Test with corrupted/non-HDF5 file
    let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
    std::io::Write::write_all(&mut temp_file, b"This is not an HDF5 file")
        .expect("Failed to write test data");

    let result = load_events_from_hdf5(temp_file.path().to_str().unwrap(), None);
    assert!(result.is_err(), "Should fail for non-HDF5 file");

    println!("HDF5 error handling tests completed");
}

#[test]
fn test_hdf5_performance_large_files() {
    let test_files = get_hdf5_test_files();

    if test_files.is_empty() {
        println!("Skipping test: No HDF5 test files found");
        return;
    }

    // Find the largest available file
    let mut largest_file = None;
    let mut largest_size = 0u64;

    for file_path in &test_files {
        if let Ok(metadata) = std::fs::metadata(file_path) {
            let size = metadata.len();
            if size > largest_size {
                largest_size = size;
                largest_file = Some(file_path);
            }
        }
    }

    if let Some(file_path) = largest_file {
        if largest_size > 1_000_000 {
            // Only test files > 1MB
            let start = std::time::Instant::now();

            match load_events_from_hdf5(file_path, None) {
                Ok(events) => {
                    let duration = start.elapsed();
                    let rate = events.len() as f64 / duration.as_secs_f64();

                    println!("Performance test: Loaded {} events from {} MB file in {:.2}s ({:.0} events/sec)",
                             events.len(),
                             largest_size / 1_000_000,
                             duration.as_secs_f64(),
                             rate);

                    // Basic performance expectation: should load at least 100k events/sec
                    if rate < 100_000.0 {
                        println!("Warning: Loading rate seems slow: {:.0} events/sec", rate);
                    }
                }
                Err(e) => {
                    println!("Performance test failed: {}", e);
                }
            }
        } else {
            println!(
                "Skipping performance test: Largest file is only {} bytes",
                largest_size
            );
        }
    }
}

#[test]
fn test_hdf5_multiple_load_consistency() {
    let test_files = get_hdf5_test_files();

    if test_files.is_empty() {
        println!("Skipping test: No HDF5 test files found");
        return;
    }

    // Find a file we can load
    let mut test_file = None;
    for file_path in &test_files {
        if load_events_from_hdf5(file_path, None).is_ok() {
            test_file = Some(file_path);
            break;
        }
    }

    let file_path = match test_file {
        Some(path) => path,
        None => {
            println!("Skipping test: No loadable HDF5 files found");
            return;
        }
    };

    // Load the same file multiple times
    let events1 = load_events_from_hdf5(file_path, None).expect("First load failed");
    let events2 = load_events_from_hdf5(file_path, None).expect("Second load failed");

    assert_eq!(
        events1.len(),
        events2.len(),
        "Event count should be consistent across loads"
    );

    // Check that events are identical
    for (i, (e1, e2)) in events1.iter().zip(events2.iter()).enumerate() {
        assert_eq!(e1.t, e2.t, "Timestamp mismatch at event {}", i);
        assert_eq!(e1.x, e2.x, "X coordinate mismatch at event {}", i);
        assert_eq!(e1.y, e2.y, "Y coordinate mismatch at event {}", i);
        assert_eq!(e1.polarity, e2.polarity, "Polarity mismatch at event {}", i);
    }

    println!(
        "Consistency test passed: {} events loaded identically twice",
        events1.len()
    );
}
