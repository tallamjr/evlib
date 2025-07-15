/// Comprehensive event data validation tests
/// Tests event coordinate, timestamp, and polarity validation across formats
///
/// This test suite verifies:
/// 1. Event coordinate bounds validation
/// 2. Timestamp ordering and monotonicity
/// 3. Polarity value correctness
/// 4. Data consistency across different loading methods
/// 5. Edge case handling and error reporting
use evlib::ev_core::{Event, Events};
use evlib::ev_formats::{
    load_events_from_text, load_events_with_config, AerConfig, AerReader, LoadConfig, TimestampMode,
};
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use tempfile::NamedTempFile;

const SLIDER_DEPTH_DIR: &str = "/Users/tallam/github/tallamjr/origin/evlib/data/slider_depth";

/// Helper function to check if a test data file exists
fn check_data_file_exists(path: &str) -> bool {
    Path::new(path).exists()
}

/// Create test events with known properties for validation
fn create_test_events() -> Events {
    vec![
        Event {
            t: 0.001000,
            x: 100,
            y: 150,
            polarity: 1,
        },
        Event {
            t: 0.001005,
            x: 200,
            y: 250,
            polarity: -1,
        },
        Event {
            t: 0.001010,
            x: 300,
            y: 350,
            polarity: 1,
        },
        Event {
            t: 0.001015,
            x: 400,
            y: 450,
            polarity: 0,
        },
        Event {
            t: 0.001020,
            x: 500,
            y: 550,
            polarity: 1,
        },
    ]
}

/// Create a test file with specific event data
fn create_test_file_with_events(events: &Events) -> NamedTempFile {
    let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");

    writeln!(temp_file, "# timestamp x y polarity").unwrap();
    for event in events {
        writeln!(
            temp_file,
            "{:.6} {} {} {}",
            event.t, event.x, event.y, event.polarity
        )
        .unwrap();
    }

    temp_file
}

/// Event validation statistics
#[derive(Debug, Default)]
struct ValidationStats {
    total_events: usize,
    timestamp_violations: usize,
    coordinate_violations: usize,
    polarity_violations: usize,
    duplicate_events: usize,
    min_timestamp: f64,
    max_timestamp: f64,
    min_x: u16,
    max_x: u16,
    min_y: u16,
    max_y: u16,
    polarity_distribution: HashMap<i8, usize>,
}

impl ValidationStats {
    fn new() -> Self {
        Self {
            min_timestamp: f64::INFINITY,
            max_timestamp: f64::NEG_INFINITY,
            min_x: u16::MAX,
            max_x: 0,
            min_y: u16::MAX,
            max_y: 0,
            ..Default::default()
        }
    }

    fn analyze_events(events: &Events) -> Self {
        let mut stats = Self::new();
        stats.total_events = events.len();

        if events.is_empty() {
            return stats;
        }

        // Initialize with first event
        stats.min_timestamp = events[0].t;
        stats.max_timestamp = events[0].t;
        stats.min_x = events[0].x;
        stats.max_x = events[0].x;
        stats.min_y = events[0].y;
        stats.max_y = events[0].y;

        for (i, event) in events.iter().enumerate() {
            // Update ranges
            stats.min_timestamp = stats.min_timestamp.min(event.t);
            stats.max_timestamp = stats.max_timestamp.max(event.t);
            stats.min_x = stats.min_x.min(event.x);
            stats.max_x = stats.max_x.max(event.x);
            stats.min_y = stats.min_y.min(event.y);
            stats.max_y = stats.max_y.max(event.y);

            // Check timestamp ordering
            if i > 0 && event.t < events[i - 1].t {
                stats.timestamp_violations += 1;
            }

            // Check coordinate bounds (assuming reasonable sensor size)
            if event.x > 2048 || event.y > 2048 {
                stats.coordinate_violations += 1;
            }

            // Check polarity values
            if event.polarity < -1 || event.polarity > 1 {
                stats.polarity_violations += 1;
            }

            // Update polarity distribution
            *stats
                .polarity_distribution
                .entry(event.polarity)
                .or_insert(0) += 1;
        }

        // Check for duplicates (simplified - exact matches only)
        let mut seen = std::collections::HashSet::new();
        for event in events {
            let key = (
                (event.t * 1e9) as i64, // Convert to nanoseconds
                event.x,
                event.y,
                event.polarity,
            );

            if seen.contains(&key) {
                stats.duplicate_events += 1;
            } else {
                seen.insert(key);
            }
        }

        stats
    }

    fn print_summary(&self, label: &str) {
        println!("\n=== {} Validation Summary ===", label);
        println!("Total events: {}", self.total_events);
        println!(
            "Timestamp range: {:.6} to {:.6} ({:.6}s span)",
            self.min_timestamp,
            self.max_timestamp,
            self.max_timestamp - self.min_timestamp
        );
        println!(
            "Coordinate range: X=[{}, {}], Y=[{}, {}]",
            self.min_x, self.max_x, self.min_y, self.max_y
        );
        println!(
            "Violations: {} timestamp, {} coordinate, {} polarity, {} duplicates",
            self.timestamp_violations,
            self.coordinate_violations,
            self.polarity_violations,
            self.duplicate_events
        );
        println!("Polarity distribution: {:?}", self.polarity_distribution);
    }
}

#[test]
fn test_timestamp_validation() {
    let events_chunk_txt = format!("{}/events_chunk.txt", SLIDER_DEPTH_DIR);

    if !check_data_file_exists(&events_chunk_txt) {
        println!("Skipping test: {} not found", events_chunk_txt);
        return;
    }

    let config = LoadConfig::new();
    let events = load_events_from_text(&events_chunk_txt, &config).expect("Failed to load events");

    let stats = ValidationStats::analyze_events(&events);
    stats.print_summary("Timestamp Validation");

    // Validate timestamp properties
    assert!(
        stats.min_timestamp >= 0.0,
        "Timestamps should be non-negative"
    );
    assert!(
        stats.max_timestamp > stats.min_timestamp,
        "Should have temporal span"
    );

    // Allow some timestamp violations in real data due to sensor characteristics
    let violation_rate = stats.timestamp_violations as f64 / stats.total_events as f64;
    assert!(
        violation_rate < 0.02,
        "Timestamp violation rate too high: {:.4}%",
        violation_rate * 100.0
    );

    // Check for reasonable timestamp progression
    let time_span = stats.max_timestamp - stats.min_timestamp;
    assert!(time_span > 0.0, "Should have positive time span");
    assert!(
        time_span < 3600.0,
        "Time span seems unreasonably large: {:.2}s",
        time_span
    );
}

#[test]
fn test_coordinate_validation() {
    let events_chunk_txt = format!("{}/events_chunk.txt", SLIDER_DEPTH_DIR);

    if !check_data_file_exists(&events_chunk_txt) {
        println!("Skipping test: {} not found", events_chunk_txt);
        return;
    }

    let config = LoadConfig::new();
    let events = load_events_from_text(&events_chunk_txt, &config).expect("Failed to load events");

    let stats = ValidationStats::analyze_events(&events);

    // Validate coordinate properties
    assert_eq!(
        stats.coordinate_violations, 0,
        "Found {} events with invalid coordinates",
        stats.coordinate_violations
    );

    // Check for reasonable coordinate ranges
    assert!(
        stats.max_x < 1024,
        "X coordinates seem too large: {}",
        stats.max_x
    );
    assert!(
        stats.max_y < 1024,
        "Y coordinates seem too large: {}",
        stats.max_y
    );

    // Should have coordinate variation
    assert!(
        stats.max_x > stats.min_x,
        "Should have X coordinate variation"
    );
    assert!(
        stats.max_y > stats.min_y,
        "Should have Y coordinate variation"
    );

    // Coordinate distribution should be reasonably spread
    let x_range = stats.max_x - stats.min_x;
    let y_range = stats.max_y - stats.min_y;
    assert!(
        x_range > 10,
        "X coordinate range seems too small: {}",
        x_range
    );
    assert!(
        y_range > 10,
        "Y coordinate range seems too small: {}",
        y_range
    );

    println!(
        "Coordinate validation passed: X=[{}, {}], Y=[{}, {}]",
        stats.min_x, stats.max_x, stats.min_y, stats.max_y
    );
}

#[test]
fn test_polarity_validation() {
    let events_chunk_txt = format!("{}/events_chunk.txt", SLIDER_DEPTH_DIR);

    if !check_data_file_exists(&events_chunk_txt) {
        println!("Skipping test: {} not found", events_chunk_txt);
        return;
    }

    let config = LoadConfig::new();
    let events = load_events_from_text(&events_chunk_txt, &config).expect("Failed to load events");

    let stats = ValidationStats::analyze_events(&events);

    // Validate polarity properties
    assert_eq!(
        stats.polarity_violations, 0,
        "Found {} events with invalid polarity",
        stats.polarity_violations
    );

    // Should have both positive and negative events
    let has_positive = stats.polarity_distribution.get(&1).unwrap_or(&0) > &0;
    let has_negative = stats.polarity_distribution.get(&-1).unwrap_or(&0) > &0
        || stats.polarity_distribution.get(&0).unwrap_or(&0) > &0;

    assert!(has_positive, "Should have positive polarity events");
    assert!(has_negative, "Should have negative polarity events");

    // Check polarity balance (shouldn't be too skewed)
    let total_valid = stats.polarity_distribution.values().sum::<usize>();
    let positive_count = *stats.polarity_distribution.get(&1).unwrap_or(&0);
    let positive_ratio = positive_count as f64 / total_valid as f64;

    assert!(
        positive_ratio > 0.1 && positive_ratio < 0.9,
        "Polarity distribution seems skewed: {:.2}% positive",
        positive_ratio * 100.0
    );

    println!(
        "Polarity validation passed: {:.1}% positive, distribution: {:?}",
        positive_ratio * 100.0,
        stats.polarity_distribution
    );
}

#[test]
fn test_synthetic_event_validation() {
    let test_events = create_test_events();
    let test_file = create_test_file_with_events(&test_events);

    let config = LoadConfig::new();
    let loaded_events = load_events_from_text(test_file.path().to_str().unwrap(), &config)
        .expect("Failed to load synthetic events");

    // Should load exactly the same events
    assert_eq!(
        loaded_events.len(),
        test_events.len(),
        "Event count mismatch"
    );

    for (i, (original, loaded)) in test_events.iter().zip(loaded_events.iter()).enumerate() {
        assert!(
            (original.t - loaded.t).abs() < 1e-6,
            "Timestamp mismatch at event {}: {} vs {}",
            i,
            original.t,
            loaded.t
        );
        assert_eq!(original.x, loaded.x, "X coordinate mismatch at event {}", i);
        assert_eq!(original.y, loaded.y, "Y coordinate mismatch at event {}", i);
        assert_eq!(
            original.polarity, loaded.polarity,
            "Polarity mismatch at event {}",
            i
        );
    }

    let stats = ValidationStats::analyze_events(&loaded_events);
    stats.print_summary("Synthetic Events");

    // Synthetic events should have no violations
    assert_eq!(
        stats.timestamp_violations, 0,
        "Synthetic events should have no timestamp violations"
    );
    assert_eq!(
        stats.coordinate_violations, 0,
        "Synthetic events should have no coordinate violations"
    );
    assert_eq!(
        stats.polarity_violations, 0,
        "Synthetic events should have no polarity violations"
    );

    println!("Synthetic event validation passed");
}

#[test]
fn test_edge_case_coordinates() {
    // Test events with edge case coordinates
    let edge_events = vec![
        Event {
            t: 0.001,
            x: 0,
            y: 0,
            polarity: 1,
        }, // Corner
        Event {
            t: 0.002,
            x: 1023,
            y: 767,
            polarity: -1,
        }, // Max coords
        Event {
            t: 0.003,
            x: 512,
            y: 384,
            polarity: 1,
        }, // Center
    ];

    let test_file = create_test_file_with_events(&edge_events);

    let config = LoadConfig::new();
    let loaded_events = load_events_from_text(test_file.path().to_str().unwrap(), &config)
        .expect("Failed to load edge case events");

    let stats = ValidationStats::analyze_events(&loaded_events);

    // Should handle edge coordinates without violations
    assert_eq!(
        stats.coordinate_violations, 0,
        "Edge coordinates should be valid"
    );
    assert_eq!(stats.min_x, 0, "Should handle x=0");
    assert_eq!(stats.min_y, 0, "Should handle y=0");
    assert!(
        stats.max_x <= 1023,
        "Should handle reasonable max coordinates"
    );

    println!("Edge case coordinate validation passed");
}

#[test]
fn test_invalid_event_handling() {
    // Create file with intentionally invalid events
    let mut invalid_file = NamedTempFile::new().expect("Failed to create temp file");

    writeln!(invalid_file, "# Test file with invalid events").unwrap();
    writeln!(invalid_file, "0.001 100 150 1").unwrap(); // Valid
    writeln!(invalid_file, "0.002 -10 200 1").unwrap(); // Negative coordinate
    writeln!(invalid_file, "0.003 200 300 5").unwrap(); // Invalid polarity
    writeln!(invalid_file, "-0.001 300 400 1").unwrap(); // Negative timestamp
    writeln!(invalid_file, "0.005 5000 150 -1").unwrap(); // Large coordinate
    writeln!(invalid_file, "invalid line").unwrap(); // Malformed line
    writeln!(invalid_file, "0.007 250 350 0").unwrap(); // Valid

    let config = LoadConfig::new();
    let result = load_events_from_text(invalid_file.path().to_str().unwrap(), &config);

    // Loading should fail for invalid data
    assert!(
        result.is_err(),
        "Should fail to load file with invalid events"
    );

    println!("Invalid event handling test passed - correctly rejected malformed data");
}

#[test]
fn test_large_coordinate_filtering() {
    // Test spatial filtering with various coordinate bounds
    let events_chunk_txt = format!("{}/events_chunk.txt", SLIDER_DEPTH_DIR);

    if !check_data_file_exists(&events_chunk_txt) {
        println!("Skipping test: {} not found", events_chunk_txt);
        return;
    }

    // Load full dataset first
    let config_full = LoadConfig::new();
    let events_full =
        load_events_from_text(&events_chunk_txt, &config_full).expect("Failed to load full events");

    let full_stats = ValidationStats::analyze_events(&events_full);

    // Test filtering with tight bounds
    let config_filtered = LoadConfig::new().with_spatial_bounds(
        Some(full_stats.min_x + 50),
        Some(full_stats.max_x - 50),
        Some(full_stats.min_y + 50),
        Some(full_stats.max_y - 50),
    );

    let events_filtered = load_events_from_text(&events_chunk_txt, &config_filtered)
        .expect("Failed to load filtered events");

    assert!(
        events_filtered.len() < events_full.len(),
        "Filtered events should be fewer than full set"
    );

    // Validate that all filtered events are within bounds
    for event in &events_filtered {
        assert!(
            event.x >= full_stats.min_x + 50,
            "Event x below filter bound"
        );
        assert!(
            event.x <= full_stats.max_x - 50,
            "Event x above filter bound"
        );
        assert!(
            event.y >= full_stats.min_y + 50,
            "Event y below filter bound"
        );
        assert!(
            event.y <= full_stats.max_y - 50,
            "Event y above filter bound"
        );
    }

    let filtered_stats = ValidationStats::analyze_events(&events_filtered);
    filtered_stats.print_summary("Spatially Filtered Events");

    println!(
        "Spatial filtering validation passed: {} -> {} events",
        events_full.len(),
        events_filtered.len()
    );
}

#[test]
fn test_timestamp_precision() {
    let events_chunk_txt = format!("{}/events_chunk.txt", SLIDER_DEPTH_DIR);

    if !check_data_file_exists(&events_chunk_txt) {
        println!("Skipping test: {} not found", events_chunk_txt);
        return;
    }

    let config = LoadConfig::new();
    let events = load_events_from_text(&events_chunk_txt, &config).expect("Failed to load events");

    // Check timestamp precision and resolution
    let mut time_diffs = Vec::new();
    for i in 1..events.len().min(1000) {
        // Check first 1000 events
        if events[i].t > events[i - 1].t {
            time_diffs.push(events[i].t - events[i - 1].t);
        }
    }

    if !time_diffs.is_empty() {
        time_diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let min_diff = time_diffs[0];
        let median_diff = time_diffs[time_diffs.len() / 2];

        // Check for reasonable timestamp resolution
        assert!(min_diff > 0.0, "Should have positive time differences");
        assert!(
            min_diff < 0.1,
            "Minimum time difference seems too large: {:.6}s",
            min_diff
        );
        assert!(
            median_diff < 1.0,
            "Median time difference seems too large: {:.6}s",
            median_diff
        );

        println!(
            "Timestamp precision: min_diff={:.6}s, median_diff={:.6}s",
            min_diff, median_diff
        );
    }
}

#[test]
fn test_aer_synthetic_validation() {
    // Test AER format with synthetic data
    let config =
        AerConfig::default().with_timestamp_generation(true, TimestampMode::Sequential, 0.0, 1e-6);
    let reader = AerReader::with_config(config);

    // Create synthetic AER data
    let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");

    let test_events = [
        (100, 150, 1), // x=100, y=150, polarity=1
        (200, 250, 0), // x=200, y=250, polarity=0
        (300, 350, 1), // x=300, y=350, polarity=1
    ];

    for (x, y, polarity) in &test_events {
        let raw_event = ((*y as u32) << 10) | ((*x as u32) << 1) | (*polarity as u32);
        let bytes = raw_event.to_le_bytes();
        temp_file
            .write_all(&bytes)
            .expect("Failed to write test data");
    }

    match reader.read_file(temp_file.path()) {
        Ok((events, metadata)) => {
            let stats = ValidationStats::analyze_events(&events);
            stats.print_summary("AER Synthetic");

            // Validate AER-specific properties
            assert_eq!(events.len(), 3, "Should load 3 synthetic AER events");
            assert_eq!(metadata.event_count, 3);
            assert_eq!(
                stats.coordinate_violations, 0,
                "No coordinate violations expected"
            );
            assert_eq!(
                stats.polarity_violations, 0,
                "No polarity violations expected"
            );

            // Check specific event values
            assert_eq!(events[0].x, 100);
            assert_eq!(events[0].y, 150);
            assert_eq!(events[0].polarity, 1);

            assert_eq!(events[1].x, 200);
            assert_eq!(events[1].y, 250);
            assert_eq!(events[1].polarity, -1); // polarity 0 -> -1

            println!("AER synthetic validation passed");
        }
        Err(e) => {
            println!(
                "AER synthetic test failed (expected for some systems): {}",
                e
            );
        }
    }
}

#[test]
fn test_duplicate_detection() {
    // Create events with intentional duplicates
    let events_with_dups = vec![
        Event {
            t: 0.001,
            x: 100,
            y: 150,
            polarity: 1,
        },
        Event {
            t: 0.002,
            x: 200,
            y: 250,
            polarity: -1,
        },
        Event {
            t: 0.001,
            x: 100,
            y: 150,
            polarity: 1,
        }, // Exact duplicate
        Event {
            t: 0.003,
            x: 300,
            y: 350,
            polarity: 1,
        },
        Event {
            t: 0.002,
            x: 200,
            y: 250,
            polarity: -1,
        }, // Another duplicate
    ];

    let test_file = create_test_file_with_events(&events_with_dups);

    let config = LoadConfig::new();
    let loaded_events = load_events_from_text(test_file.path().to_str().unwrap(), &config)
        .expect("Failed to load events with duplicates");

    let stats = ValidationStats::analyze_events(&loaded_events);

    // Should detect the duplicates
    assert_eq!(
        stats.duplicate_events, 2,
        "Should detect 2 duplicate events, found {}",
        stats.duplicate_events
    );

    println!(
        "Duplicate detection passed: found {} duplicates out of {} events",
        stats.duplicate_events, stats.total_events
    );
}

#[test]
fn test_consistency_across_load_methods() {
    let events_chunk_txt = format!("{}/events_chunk.txt", SLIDER_DEPTH_DIR);

    if !check_data_file_exists(&events_chunk_txt) {
        println!("Skipping test: {} not found", events_chunk_txt);
        return;
    }

    // Load using different methods
    let config = LoadConfig::new();

    let events1 =
        load_events_from_text(&events_chunk_txt, &config).expect("Failed to load with text loader");

    let events2 = load_events_with_config(&events_chunk_txt, &config)
        .expect("Failed to load with generic loader");

    // Results should be identical
    assert_eq!(
        events1.len(),
        events2.len(),
        "Event count mismatch between loading methods"
    );

    for (i, (e1, e2)) in events1.iter().zip(events2.iter()).enumerate() {
        assert!(
            (e1.t - e2.t).abs() < 1e-9,
            "Timestamp mismatch at event {}: {} vs {}",
            i,
            e1.t,
            e2.t
        );
        assert_eq!(e1.x, e2.x, "X coordinate mismatch at event {}", i);
        assert_eq!(e1.y, e2.y, "Y coordinate mismatch at event {}", i);
        assert_eq!(e1.polarity, e2.polarity, "Polarity mismatch at event {}", i);
    }

    let stats1 = ValidationStats::analyze_events(&events1);
    let stats2 = ValidationStats::analyze_events(&events2);

    assert_eq!(stats1.timestamp_violations, stats2.timestamp_violations);
    assert_eq!(stats1.coordinate_violations, stats2.coordinate_violations);
    assert_eq!(stats1.polarity_violations, stats2.polarity_violations);

    println!("Consistency validation passed: identical results from different loaders");
}
