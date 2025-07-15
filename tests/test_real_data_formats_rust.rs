use evlib::ev_core::{Event, Events};
use evlib::ev_formats::{
    detect_event_format,
    evt2_reader::{Evt2Config, Evt2Reader},
    load_events_from_hdf5, load_events_from_text, load_events_with_config, EventFormat, LoadConfig,
    PolarityEncoding,
};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Test configuration for different data files
struct TestFileConfig {
    path: PathBuf,
    expected_format: EventFormat,
    expected_resolution: (u16, u16),
    min_expected_events: usize,
    description: String,
}

/// Validation results for event data
#[derive(Debug)]
struct ValidationResults {
    event_count: usize,
    coordinate_bounds: ((u16, u16), (u16, u16)), // ((x_min, x_max), (y_min, y_max))
    time_range: (f64, f64, f64),                 // (start, end, duration)
    polarity_distribution: (usize, usize),       // (positive, negative)
    data_integrity: DataIntegrity,
    execution_time: f64,
}

#[derive(Debug)]
struct DataIntegrity {
    has_nan: bool,
    has_inf: bool,
    sorted_by_time: bool,
    valid_coordinates: bool,
    valid_polarities: bool,
}

impl TestFileConfig {
    fn new(
        path: &str,
        format: EventFormat,
        resolution: (u16, u16),
        min_events: usize,
        desc: &str,
    ) -> Self {
        Self {
            path: PathBuf::from(path),
            expected_format: format,
            expected_resolution: resolution,
            min_expected_events: min_events,
            description: desc.to_string(),
        }
    }
}

fn validate_events(events: &Events, expected_resolution: (u16, u16)) -> DataIntegrity {
    if events.is_empty() {
        return DataIntegrity {
            has_nan: false,
            has_inf: false,
            sorted_by_time: true,
            valid_coordinates: true,
            valid_polarities: true,
        };
    }

    let has_nan = events.iter().any(|e| e.t.is_nan());
    let has_inf = events.iter().any(|e| e.t.is_infinite());

    let sorted_by_time = events.windows(2).all(|pair| pair[0].t <= pair[1].t);

    let valid_coordinates = events
        .iter()
        .all(|e| e.x < expected_resolution.0 && e.y < expected_resolution.1);

    let valid_polarities = events.iter().all(|e| e.polarity == -1 || e.polarity == 1);

    DataIntegrity {
        has_nan,
        has_inf,
        sorted_by_time,
        valid_coordinates,
        valid_polarities,
    }
}

fn analyze_events(
    events: &Events,
    expected_resolution: (u16, u16),
    execution_time: f64,
) -> ValidationResults {
    if events.is_empty() {
        return ValidationResults {
            event_count: 0,
            coordinate_bounds: ((0, 0), (0, 0)),
            time_range: (0.0, 0.0, 0.0),
            polarity_distribution: (0, 0),
            data_integrity: validate_events(events, expected_resolution),
            execution_time,
        };
    }

    let x_min = events.iter().map(|e| e.x).min().unwrap_or(0);
    let x_max = events.iter().map(|e| e.x).max().unwrap_or(0);
    let y_min = events.iter().map(|e| e.y).min().unwrap_or(0);
    let y_max = events.iter().map(|e| e.y).max().unwrap_or(0);

    let t_min = events.iter().map(|e| e.t).fold(f64::INFINITY, f64::min);
    let t_max = events.iter().map(|e| e.t).fold(f64::NEG_INFINITY, f64::max);
    let duration = t_max - t_min;

    let positive_count = events.iter().filter(|e| e.polarity == 1).count();
    let negative_count = events.iter().filter(|e| e.polarity == -1).count();

    ValidationResults {
        event_count: events.len(),
        coordinate_bounds: ((x_min, x_max), (y_min, y_max)),
        time_range: (t_min, t_max, duration),
        polarity_distribution: (positive_count, negative_count),
        data_integrity: validate_events(events, expected_resolution),
        execution_time,
    }
}

#[test]
fn test_format_detection_comprehensive() {
    let test_files = vec![
        ("data/eTram/raw/val_2/val_night_011.raw", EventFormat::EVT2),
        ("data/eTram/h5/val_2/val_night_011_td.h5", EventFormat::HDF5),
        ("data/slider_depth/events.txt", EventFormat::Text),
    ];

    for (file_path, expected_format) in test_files {
        let path = Path::new(file_path);
        if !path.exists() {
            println!("Skipping {}: file not found", file_path);
            continue;
        }

        println!("Testing format detection for: {}", file_path);

        let detection_result = detect_event_format(file_path);
        assert!(
            detection_result.is_ok(),
            "Format detection failed for {}: {:?}",
            file_path,
            detection_result.err()
        );

        let result = detection_result.unwrap();
        assert_eq!(
            result.format, expected_format,
            "Format mismatch for {}: expected {:?}, got {:?}",
            file_path, expected_format, result.format
        );

        assert!(
            result.confidence > 0.8,
            "Low confidence for {}: {:.2}",
            file_path,
            result.confidence
        );

        println!(
            "✓ Format detection passed: {} -> {:?} (confidence: {:.2})",
            file_path, result.format, result.confidence
        );
    }
}

#[test]
fn test_evt2_reader_comprehensive() {
    let test_files = vec![
        TestFileConfig::new(
            "data/eTram/raw/val_2/val_night_011.raw",
            EventFormat::EVT2,
            (1280, 720),
            100_000,
            "Small EVT2 file",
        ),
        TestFileConfig::new(
            "data/eTram/raw/val_2/val_night_007.raw",
            EventFormat::EVT2,
            (1280, 720),
            1_000_000,
            "Large EVT2 file",
        ),
    ];

    for config in test_files {
        if !config.path.exists() {
            println!("Skipping {}: file not found", config.description);
            continue;
        }

        println!("Testing EVT2 reader: {}", config.description);

        // Test format detection first
        let detection_result = detect_event_format(config.path.to_str().unwrap()).unwrap();
        assert_eq!(detection_result.format, EventFormat::EVT2);

        // Create EVT2 reader configuration
        let evt2_config = Evt2Config {
            validate_coordinates: true,
            skip_invalid_events: false,
            max_events: None,
            sensor_resolution: Some(config.expected_resolution),
            chunk_size: 1_000_000,
            polarity_encoding: Some(PolarityEncoding::ZeroOne),
        };

        let reader = Evt2Reader::with_config(evt2_config);

        // Test basic reading
        let start_time = Instant::now();
        let load_config = LoadConfig::new();
        let events_result = reader.read_with_config(config.path.to_str().unwrap(), &load_config);
        let execution_time = start_time.elapsed().as_secs_f64();

        assert!(
            events_result.is_ok(),
            "EVT2 reading failed: {:?}",
            events_result.err()
        );
        let events = events_result.unwrap();

        // Validate results
        let results = analyze_events(&events, config.expected_resolution, execution_time);

        assert!(
            results.event_count >= config.min_expected_events,
            "Too few events: {} < {}",
            results.event_count,
            config.min_expected_events
        );

        assert!(
            !results.data_integrity.has_nan,
            "NaN values found in timestamps"
        );
        assert!(
            !results.data_integrity.has_inf,
            "Inf values found in timestamps"
        );
        assert!(
            results.data_integrity.valid_coordinates,
            "Invalid coordinates found"
        );
        assert!(
            results.data_integrity.valid_polarities,
            "Invalid polarities found"
        );

        println!(
            "✓ EVT2 test passed: {} events in {:.2}s ({:.0} events/s)",
            results.event_count,
            results.execution_time,
            results.event_count as f64 / results.execution_time
        );

        // Test filtering
        let filter_config = LoadConfig::new()
            .with_time_window(
                Some(results.time_range.0 + 0.1),
                Some(results.time_range.1 - 0.1),
            )
            .with_polarity(Some(1));

        let filtered_events = reader
            .read_with_config(config.path.to_str().unwrap(), &filter_config)
            .unwrap();
        assert!(
            filtered_events.len() < events.len(),
            "Filtering didn't reduce event count"
        );
        assert!(
            filtered_events.iter().all(|e| e.polarity == 1),
            "Polarity filtering failed"
        );

        println!(
            "✓ EVT2 filtering passed: {} -> {} events",
            events.len(),
            filtered_events.len()
        );
    }
}

#[test]
fn test_hdf5_reader_comprehensive() {
    let test_files = vec![
        TestFileConfig::new(
            "data/eTram/h5/val_2/val_night_011_td.h5",
            EventFormat::HDF5,
            (1280, 720),
            100_000,
            "Small HDF5 file",
        ),
        TestFileConfig::new(
            "data/eTram/h5/val_2/val_night_007_td.h5",
            EventFormat::HDF5,
            (1280, 720),
            1_000_000,
            "Large HDF5 file",
        ),
    ];

    for config in test_files {
        if !config.path.exists() {
            println!("Skipping {}: file not found", config.description);
            continue;
        }

        println!("Testing HDF5 reader: {}", config.description);

        // Test format detection
        let detection_result = detect_event_format(config.path.to_str().unwrap()).unwrap();
        assert_eq!(detection_result.format, EventFormat::HDF5);

        // Test basic reading
        let start_time = Instant::now();
        let events_result = load_events_from_hdf5(config.path.to_str().unwrap(), None);
        let execution_time = start_time.elapsed().as_secs_f64();

        assert!(
            events_result.is_ok(),
            "HDF5 reading failed: {:?}",
            events_result.err()
        );
        let events = events_result.unwrap();

        // Validate results
        let results = analyze_events(&events, config.expected_resolution, execution_time);

        assert!(
            results.event_count >= config.min_expected_events,
            "Too few events: {} < {}",
            results.event_count,
            config.min_expected_events
        );

        assert!(
            !results.data_integrity.has_nan,
            "NaN values found in timestamps"
        );
        assert!(
            !results.data_integrity.has_inf,
            "Inf values found in timestamps"
        );
        assert!(
            results.data_integrity.valid_coordinates,
            "Invalid coordinates found"
        );
        assert!(
            results.data_integrity.valid_polarities,
            "Invalid polarities found"
        );

        println!(
            "✓ HDF5 test passed: {} events in {:.2}s ({:.0} events/s)",
            results.event_count,
            results.execution_time,
            results.event_count as f64 / results.execution_time
        );
    }
}

#[test]
fn test_text_reader_comprehensive() {
    let config = TestFileConfig::new(
        "data/slider_depth/events.txt",
        EventFormat::Text,
        (346, 240),
        100_000,
        "Text format file",
    );

    if !config.path.exists() {
        println!("Skipping text reader test: file not found");
        return;
    }

    println!("Testing text reader: {}", config.description);

    // Test format detection
    let detection_result = detect_event_format(config.path.to_str().unwrap()).unwrap();
    assert_eq!(detection_result.format, EventFormat::Text);

    // Test basic reading with polarity encoding conversion
    let start_time = Instant::now();
    let load_config = LoadConfig::new().with_polarity_encoding(PolarityEncoding::ZeroOne);

    let events_result = load_events_from_text(config.path.to_str().unwrap(), &load_config);
    let execution_time = start_time.elapsed().as_secs_f64();

    assert!(
        events_result.is_ok(),
        "Text reading failed: {:?}",
        events_result.err()
    );
    let events = events_result.unwrap();

    // Validate results
    let results = analyze_events(&events, config.expected_resolution, execution_time);

    assert!(
        results.event_count >= config.min_expected_events,
        "Too few events: {} < {}",
        results.event_count,
        config.min_expected_events
    );

    assert!(
        !results.data_integrity.has_nan,
        "NaN values found in timestamps"
    );
    assert!(
        !results.data_integrity.has_inf,
        "Inf values found in timestamps"
    );
    assert!(
        results.data_integrity.valid_coordinates,
        "Invalid coordinates found"
    );
    assert!(
        results.data_integrity.valid_polarities,
        "Invalid polarities found"
    );

    // Check polarity encoding conversion
    let unique_polarities: std::collections::HashSet<i8> =
        events.iter().map(|e| e.polarity).collect();
    assert_eq!(
        unique_polarities,
        [1, -1].iter().cloned().collect(),
        "Polarity encoding conversion failed: {:?}",
        unique_polarities
    );

    println!(
        "✓ Text test passed: {} events in {:.2}s ({:.0} events/s)",
        results.event_count,
        results.execution_time,
        results.event_count as f64 / results.execution_time
    );

    // Test filtering functionality
    let filter_config = LoadConfig::new()
        .with_spatial_bounds(Some(100), Some(200), Some(50), Some(150))
        .with_polarity(Some(1));

    let filtered_events =
        load_events_from_text(config.path.to_str().unwrap(), &filter_config).unwrap();
    assert!(
        filtered_events.len() < events.len(),
        "Filtering didn't reduce event count"
    );
    assert!(
        filtered_events.iter().all(|e| e.polarity == 1),
        "Polarity filtering failed"
    );
    assert!(
        filtered_events.iter().all(|e| e.x >= 100 && e.x <= 200),
        "X coordinate filtering failed"
    );
    assert!(
        filtered_events.iter().all(|e| e.y >= 50 && e.y <= 150),
        "Y coordinate filtering failed"
    );

    println!(
        "✓ Text filtering passed: {} -> {} events",
        events.len(),
        filtered_events.len()
    );
}

#[test]
fn test_data_consistency_evt2_vs_hdf5() {
    let evt2_path = "data/eTram/raw/val_2/val_night_011.raw";
    let hdf5_path = "data/eTram/h5/val_2/val_night_011_td.h5";

    if !Path::new(evt2_path).exists() || !Path::new(hdf5_path).exists() {
        println!("Skipping consistency test: required files not found");
        return;
    }

    println!("Testing data consistency between EVT2 and HDF5 formats");

    // Load EVT2 file
    let evt2_config = Evt2Config {
        validate_coordinates: true,
        skip_invalid_events: false,
        max_events: None,
        sensor_resolution: Some((1280, 720)),
        chunk_size: 1_000_000,
        polarity_encoding: Some(PolarityEncoding::ZeroOne),
    };
    let evt2_reader = Evt2Reader::with_config(evt2_config);
    let evt2_events = evt2_reader
        .read_with_config(evt2_path, &LoadConfig::new())
        .unwrap();

    // Load HDF5 file
    let hdf5_events = load_events_from_hdf5(hdf5_path, None).unwrap();

    // Compare results
    let evt2_count = evt2_events.len();
    let hdf5_count = hdf5_events.len();

    // Allow for small differences due to format conversion (within 1%)
    let count_diff = if evt2_count > hdf5_count {
        evt2_count - hdf5_count
    } else {
        hdf5_count - evt2_count
    };
    let max_allowed_diff = std::cmp::max(evt2_count, hdf5_count) / 100; // 1% tolerance

    assert!(
        count_diff <= max_allowed_diff,
        "Event count difference too large: EVT2 {} vs HDF5 {} (diff: {})",
        evt2_count,
        hdf5_count,
        count_diff
    );

    println!(
        "✓ Consistency test passed: EVT2 {} events, HDF5 {} events (diff: {})",
        evt2_count, hdf5_count, count_diff
    );
}

#[test]
fn test_generic_load_function() {
    let test_files = vec![
        ("data/eTram/raw/val_2/val_night_011.raw", (1280, 720)),
        ("data/eTram/h5/val_2/val_night_011_td.h5", (1280, 720)),
        ("data/slider_depth/events.txt", (346, 240)),
    ];

    for (file_path, expected_resolution) in test_files {
        if !Path::new(file_path).exists() {
            println!(
                "Skipping generic load test for {}: file not found",
                file_path
            );
            continue;
        }

        println!("Testing generic load function: {}", file_path);

        let start_time = Instant::now();
        let config = LoadConfig::new().with_polarity_encoding(PolarityEncoding::ZeroOne);
        let events_result = load_events_with_config(file_path, &config);
        let execution_time = start_time.elapsed().as_secs_f64();

        assert!(
            events_result.is_ok(),
            "Generic load failed for {}: {:?}",
            file_path,
            events_result.err()
        );
        let events = events_result.unwrap();

        let results = analyze_events(&events, expected_resolution, execution_time);

        assert!(
            results.event_count > 0,
            "No events loaded from {}",
            file_path
        );
        assert!(
            !results.data_integrity.has_nan,
            "NaN values found in {}",
            file_path
        );
        assert!(
            !results.data_integrity.has_inf,
            "Inf values found in {}",
            file_path
        );
        assert!(
            results.data_integrity.valid_coordinates,
            "Invalid coordinates in {}",
            file_path
        );
        assert!(
            results.data_integrity.valid_polarities,
            "Invalid polarities in {}",
            file_path
        );

        println!(
            "✓ Generic load passed: {} events in {:.2}s",
            results.event_count, results.execution_time
        );
    }
}

#[test]
fn test_performance_benchmarks() {
    let test_files = vec![
        ("data/eTram/raw/val_2/val_night_007.raw", "Large EVT2 file"),
        ("data/eTram/h5/val_2/val_night_007_td.h5", "Large HDF5 file"),
        ("data/slider_depth/events.txt", "Medium text file"),
    ];

    println!("\n=== PERFORMANCE BENCHMARKS ===");

    for (file_path, description) in test_files {
        if !Path::new(file_path).exists() {
            println!("Skipping benchmark for {}: file not found", description);
            continue;
        }

        println!("\nBenchmarking: {}", description);

        // Get file size
        let file_size = std::fs::metadata(file_path).unwrap().len() as f64 / 1024.0 / 1024.0; // MB

        let start_time = Instant::now();
        let config = LoadConfig::new();
        let events = load_events_with_config(file_path, &config).unwrap();
        let execution_time = start_time.elapsed().as_secs_f64();

        let events_per_second = events.len() as f64 / execution_time;
        let mb_per_second = file_size / execution_time;
        let bytes_per_event = (file_size * 1024.0 * 1024.0) / events.len() as f64;

        println!("  File size: {:.1} MB", file_size);
        println!("  Events: {}", events.len());
        println!("  Time: {:.2} seconds", execution_time);
        println!("  Rate: {:.0} events/second", events_per_second);
        println!("  Throughput: {:.1} MB/second", mb_per_second);
        println!("  Storage efficiency: {:.1} bytes/event", bytes_per_event);

        // Performance assertions
        assert!(
            events_per_second > 100_000.0,
            "Performance too slow: {:.0} events/s",
            events_per_second
        );
        assert!(
            execution_time < file_size,
            "Reading took longer than 1 second per MB"
        );
    }

    println!("\n=== END BENCHMARKS ===\n");
}

#[test]
fn test_memory_efficiency() {
    let file_path = "data/eTram/raw/val_2/val_night_007.raw";

    if !Path::new(file_path).exists() {
        println!("Skipping memory efficiency test: file not found");
        return;
    }

    println!("Testing memory efficiency with large file");

    // Load events
    let config = LoadConfig::new();
    let events = load_events_with_config(file_path, &config).unwrap();

    // Calculate memory usage (rough estimate)
    let event_size = std::mem::size_of::<Event>();
    let expected_memory = events.len() * event_size;
    let expected_memory_mb = expected_memory as f64 / 1024.0 / 1024.0;

    println!("  Events: {}", events.len());
    println!("  Event size: {} bytes", event_size);
    println!("  Expected memory: {:.1} MB", expected_memory_mb);

    // Each event should be exactly 24 bytes (8 + 2 + 2 + 1 + padding)
    assert_eq!(event_size, 24, "Event size changed unexpectedly");

    // Memory usage should be reasonable
    assert!(
        expected_memory_mb < 1000.0,
        "Memory usage too high: {:.1} MB",
        expected_memory_mb
    );

    println!("✓ Memory efficiency test passed");
}
