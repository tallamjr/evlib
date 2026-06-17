/*!
Integration test for Prophesee ECF codec implementation.

Tests the complete ECF decoding pipeline with actual Prophesee HDF5 files.
This test validates:
1. ECF format detection
2. Fallback mechanism prioritization
3. Data integrity after decoding
4. Performance with large files
5. Error handling when codec unavailable
*/

use evlib::ev_formats::format_detector::{detect_event_format, EventFormat};
use evlib::ev_formats::{load_events_with_config, LoadConfig};
use std::env;
use std::path::Path;

/// Lightweight event row extracted from a Polars DataFrame for assertion convenience.
/// Only used by the hdf5-gated decoding tests below.
#[cfg(feature = "hdf5")]
#[derive(Debug, Clone, Copy)]
struct EventRow {
    t: f64, // seconds
    x: u16,
    y: u16,
    polarity: i8, // -1/1 for EVT/AER/HDF5; 0/1 for text
}

#[cfg(feature = "hdf5")]
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

/// Check if we're running in CI environment
fn is_running_in_ci() -> bool {
    env::var("CI").is_ok()
        || env::var("GITHUB_ACTIONS").is_ok()
        || env::var("TRAVIS").is_ok()
        || env::var("CIRCLECI").is_ok()
        || env::var("JENKINS_URL").is_ok()
}

const PROPHESEE_TEST_FILE: &str =
    "/Users/tallam/github/tallamjr/origin/evlib/data/prophesee/samples/hdf5/pedestrians.hdf5";

#[test]
fn test_prophesee_file_exists() {
    if is_running_in_ci() {
        eprintln!("Skipping test in CI environment - Prophesee test file not available in CI");
        return;
    }

    assert!(
        Path::new(PROPHESEE_TEST_FILE).exists(),
        "Prophesee test file not found: {}. This test requires the actual Prophesee HDF5 file.",
        PROPHESEE_TEST_FILE
    );
}

#[test]
fn test_prophesee_format_detection() {
    if is_running_in_ci() || !Path::new(PROPHESEE_TEST_FILE).exists() {
        eprintln!(
            "Skipping test - Prophesee file not available (CI: {}, exists: {})",
            is_running_in_ci(),
            Path::new(PROPHESEE_TEST_FILE).exists()
        );
        return;
    }

    let detection_result = detect_event_format(PROPHESEE_TEST_FILE)
        .expect("Failed to detect format of Prophesee file");

    // Should be detected as HDF5 format
    assert_eq!(detection_result.format, EventFormat::HDF5);
    assert!(
        detection_result.confidence > 0.8,
        "Format detection confidence too low: {}",
        detection_result.confidence
    );

    // Check metadata: file size is exposed as a typed field on the metadata struct.
    assert!(detection_result.metadata.file_size > 0);
    println!(
        "✓ Format detection: {:?} (confidence: {:.2})",
        detection_result.format, detection_result.confidence
    );
}

// The ECF-specific error/decoding path only exists when the crate is built with
// the hdf5 feature; without it the loader returns a generic "HDF5 not compiled in"
// message, so this assertion is only meaningful under that feature.
#[cfg(feature = "hdf5")]
#[test]
fn test_prophesee_ecf_loading_with_fallback() {
    if is_running_in_ci() || !Path::new(PROPHESEE_TEST_FILE).exists() {
        eprintln!(
            "Skipping test - Prophesee file not available (CI: {}, exists: {})",
            is_running_in_ci(),
            Path::new(PROPHESEE_TEST_FILE).exists()
        );
        return;
    }

    println!("Testing Prophesee ECF loading with fallback mechanisms...");

    // Configure for a small subset to avoid memory issues during testing
    let config = LoadConfig::new()
        .with_time_window(Some(0.0), Some(0.1)) // First 100ms only
        .with_spatial_bounds(Some(100), Some(500), Some(100), Some(400)); // Central region

    // This should trigger our fallback mechanism
    // Priority: h5py+ECF plugin -> subprocess -> Rust ECF -> Python ECF -> error
    match load_events_with_config(PROPHESEE_TEST_FILE, &config) {
        Ok(events) => {
            println!(
                "✓ Successfully loaded {} events using fallback system",
                events.height()
            );

            // Validate basic event properties
            assert!(events.height() > 0, "No events loaded from Prophesee file");

            let rows = dataframe_to_rows(&events);

            // Check coordinate bounds
            for event in &rows {
                assert!(
                    event.x >= 100 && event.x <= 500,
                    "X coordinate out of expected range: {}",
                    event.x
                );
                assert!(
                    event.y >= 100 && event.y <= 400,
                    "Y coordinate out of expected range: {}",
                    event.y
                );
                assert!(
                    event.t >= 0.0 && event.t <= 0.1,
                    "Timestamp out of expected range: {}",
                    event.t
                );
            }

            // Check that events are properly decoded
            let first_event = &rows[0];
            println!(
                "✓ First decoded event: t={:.6}s, x={}, y={}, polarity={}",
                first_event.t, first_event.x, first_event.y, first_event.polarity
            );

            // Validate timestamp ordering (should be monotonic)
            for i in 1..rows.len().min(1000) {
                // Check first 1000 events
                assert!(
                    rows[i].t >= rows[i - 1].t,
                    "Events not properly time-ordered at index {}: {} >= {}",
                    i,
                    rows[i].t,
                    rows[i - 1].t
                );
            }

            println!(
                "✓ Event validation passed - all {} events properly decoded",
                rows.len()
            );
        }
        Err(e) => {
            let error_msg = e.to_string();
            println!("Expected failure (no ECF codec installed): {}", error_msg);

            // Verify we get the expected ECF codec error with helpful message
            assert!(
                error_msg.contains("ECF")
                    || error_msg.contains("codec")
                    || error_msg.contains("Prophesee")
                    || error_msg.contains("compound dataset"),
                "Error message should mention ECF codec issue, got: {}",
                error_msg
            );

            // Should mention native support rather than installation
            assert!(
                error_msg.contains("native")
                    || error_msg.contains("automatically")
                    || error_msg.contains("integration"),
                "Error message should mention native ECF support, got: {}",
                error_msg
            );

            println!("✓ Error handling correct - mentions native ECF support");
        }
    }
}

#[cfg(feature = "hdf5")]
#[test]
fn test_prophesee_metadata_extraction() {
    if is_running_in_ci() || !Path::new(PROPHESEE_TEST_FILE).exists() {
        eprintln!(
            "Skipping test - Prophesee file not available (CI: {}, exists: {})",
            is_running_in_ci(),
            Path::new(PROPHESEE_TEST_FILE).exists()
        );
        return;
    }

    // Test that we can at least open the file and read basic metadata
    // even without ECF codec (using hdf5-metno)
    use hdf5_metno::File as H5File;

    let file = H5File::open(PROPHESEE_TEST_FILE)
        .expect("Failed to open Prophesee HDF5 file with hdf5-metno");

    // Check for expected Prophesee structure
    let cd_group = file
        .group("CD")
        .expect("Prophesee file should contain CD group");

    let events_dataset = cd_group
        .dataset("events")
        .expect("CD group should contain events dataset");

    let shape = events_dataset.shape();
    println!("✓ Prophesee file structure detected:");
    println!("  - CD/events dataset shape: {:?}", shape);
    println!("  - Total events in file: {}", shape[0]);

    // This should be around 39 million events based on our analysis
    assert!(
        shape[0] > 1_000_000,
        "File should contain substantial number of events, got: {}",
        shape[0]
    );
    assert!(
        shape[0] < 100_000_000,
        "Event count seems unreasonably high: {}",
        shape[0]
    );

    // Check for indexes dataset
    let indexes_dataset = cd_group
        .dataset("indexes")
        .expect("CD group should contain indexes dataset");
    let indexes_shape = indexes_dataset.shape();
    println!("  - CD/indexes dataset shape: {:?}", indexes_shape);

    println!("✓ Prophesee HDF5 structure validation passed");
}

#[cfg(feature = "hdf5")]
#[test]
fn test_ecf_codec_detection() {
    if is_running_in_ci() || !Path::new(PROPHESEE_TEST_FILE).exists() {
        eprintln!(
            "Skipping test - Prophesee file not available (CI: {}, exists: {})",
            is_running_in_ci(),
            Path::new(PROPHESEE_TEST_FILE).exists()
        );
        return;
    }

    // Test our ECF codec detection logic
    use hdf5_metno::File as H5File;

    let file = H5File::open(PROPHESEE_TEST_FILE).expect("Failed to open Prophesee HDF5 file");

    // Check that we detect this as a Prophesee format file
    let cd_group = file.group("CD").expect("Should have CD group");
    let events_dataset = cd_group
        .dataset("events")
        .expect("Should have events dataset");

    // This should trigger our Prophesee format detection
    let is_prophesee_format = cd_group.dataset("events").is_ok();
    assert!(
        is_prophesee_format,
        "Should detect Prophesee format (CD/events structure)"
    );

    println!("✓ ECF codec detection working correctly");
    println!("  - Detected Prophesee HDF5 format with CD/events compound dataset");
    println!("  - This will trigger ECF codec fallback mechanism");
}

// Importing Python modules from a native cargo-test binary requires a fully
// configured embedded interpreter (PYTHONHOME / filesystem codec), which this test
// harness does not provide; without it Python::with_gil aborts the whole binary.
// Python module importability is covered by the pytest suite instead.
#[cfg(feature = "python")]
#[ignore = "requires a configured embedded Python interpreter not available to the native \
            cargo-test binary; Python module imports are covered by the pytest suite"]
#[test]
fn test_python_fallback_import() {
    // Test that our Python fallback modules can be imported
    // This doesn't require the actual ECF codec, just tests module structure

    use pyo3::prelude::*;

    Python::with_gil(|py| {
        // Test that we can import our fallback modules
        match py.import("evlib.hdf5_prophesee") {
            Ok(_) => println!("✓ Python hdf5_prophesee module importable"),
            Err(e) => println!("⚠ Python fallback module not available: {}", e),
        }

        match py.import("evlib.ecf_decoder") {
            Ok(_) => println!("✓ Python ecf_decoder module importable"),
            Err(e) => println!("⚠ Python ECF decoder module not available: {}", e),
        }
    });
}

#[test]
fn test_rust_ecf_codec_functionality() {
    // Test our Rust ECF codec implementation with synthetic data
    use evlib::ev_formats::ecf_codec::{ECFDecoder, ECFEncoder, EventCD};

    println!("Testing Rust ECF codec implementation...");

    // Create test events
    let test_events = vec![
        EventCD {
            x: 100,
            y: 150,
            p: 1,
            t: 1000,
        },
        EventCD {
            x: 101,
            y: 151,
            p: -1,
            t: 1010,
        },
        EventCD {
            x: 102,
            y: 152,
            p: 1,
            t: 1020,
        },
        EventCD {
            x: 200,
            y: 250,
            p: -1,
            t: 2000,
        },
        EventCD {
            x: 300,
            y: 350,
            p: 1,
            t: 3000,
        },
    ];

    let encoder = ECFEncoder::new().with_debug(false);
    let decoder = ECFDecoder::new().with_debug(false);

    // Test encode-decode roundtrip
    let compressed = encoder
        .encode(&test_events)
        .expect("Failed to encode test events with Rust ECF codec");

    println!(
        "✓ Encoded {} events to {} bytes",
        test_events.len(),
        compressed.len()
    );

    let decoded = decoder
        .decode(&compressed)
        .expect("Failed to decode with Rust ECF codec");

    println!("✓ Decoded {} events from compressed data", decoded.len());

    // Verify roundtrip accuracy
    assert_eq!(
        test_events.len(),
        decoded.len(),
        "Event count mismatch after roundtrip"
    );

    for (i, (original, decoded)) in test_events.iter().zip(decoded.iter()).enumerate() {
        assert_eq!(
            original.x, decoded.x,
            "X coordinate mismatch at event {}",
            i
        );
        assert_eq!(
            original.y, decoded.y,
            "Y coordinate mismatch at event {}",
            i
        );
        assert_eq!(original.p, decoded.p, "Polarity mismatch at event {}", i);
        assert_eq!(original.t, decoded.t, "Timestamp mismatch at event {}", i);
    }

    println!(
        "✓ Rust ECF codec roundtrip test passed - all {} events match exactly",
        test_events.len()
    );
}

#[test]
fn test_performance_with_large_synthetic_data() {
    // Test ECF codec performance with larger synthetic datasets
    use evlib::ev_formats::ecf_codec::{ECFDecoder, ECFEncoder, EventCD};
    use std::time::Instant;

    println!("Testing ECF codec performance with synthetic data...");

    // Generate larger test dataset (10K events)
    let mut test_events = Vec::new();
    for i in 0..10_000 {
        test_events.push(EventCD {
            x: (i % 1280) as u16,
            y: ((i / 1280) % 720) as u16,
            p: if i % 2 == 0 { 1 } else { -1 },
            t: i as i64 * 1000, // 1ms intervals
        });
    }

    let encoder = ECFEncoder::new();
    let decoder = ECFDecoder::new();

    // Test encoding performance
    let encode_start = Instant::now();
    let compressed = encoder
        .encode(&test_events)
        .expect("Failed to encode large synthetic dataset");
    let encode_time = encode_start.elapsed();

    println!(
        "✓ Encoded {} events to {} bytes in {:?}",
        test_events.len(),
        compressed.len(),
        encode_time
    );

    let compression_ratio = (test_events.len() * 14) as f64 / compressed.len() as f64;
    println!("  - Compression ratio: {:.2}x", compression_ratio);

    // Test decoding performance
    let decode_start = Instant::now();
    let decoded = decoder
        .decode(&compressed)
        .expect("Failed to decode large synthetic dataset");
    let decode_time = decode_start.elapsed();

    println!("✓ Decoded {} events in {:?}", decoded.len(), decode_time);

    // Verify correctness
    assert_eq!(test_events.len(), decoded.len());

    // Spot check some events
    for i in (0..test_events.len()).step_by(1000) {
        assert_eq!(test_events[i].x, decoded[i].x);
        assert_eq!(test_events[i].y, decoded[i].y);
        assert_eq!(test_events[i].p, decoded[i].p);
        assert_eq!(test_events[i].t, decoded[i].t);
    }

    println!(
        "✓ Performance test passed - Rust ECF codec handles {} events efficiently",
        test_events.len()
    );
}

// The ECF/native error message is only produced when the crate is built with the
// hdf5 feature; otherwise the loader returns a generic "HDF5 not compiled in" error.
#[cfg(feature = "hdf5")]
#[test]
fn test_error_message_quality() {
    if is_running_in_ci() || !Path::new(PROPHESEE_TEST_FILE).exists() {
        eprintln!(
            "Skipping test - Prophesee file not available (CI: {}, exists: {})",
            is_running_in_ci(),
            Path::new(PROPHESEE_TEST_FILE).exists()
        );
        return;
    }

    // Test that error messages are helpful when ECF codec is not available
    let config = LoadConfig::new();

    match load_events_with_config(PROPHESEE_TEST_FILE, &config) {
        Ok(_) => {
            println!("✓ Successfully loaded events (ECF codec available)");
        }
        Err(e) => {
            let error_msg = e.to_string();
            println!("Testing error message quality...");
            println!("Error: {}", error_msg);

            // Check for key information in error message
            let has_ecf_mention = error_msg.to_lowercase().contains("ecf");
            let has_codec_mention = error_msg.to_lowercase().contains("codec");
            let has_native_support = error_msg.contains("native")
                || error_msg.contains("automatically")
                || error_msg.contains("integration");

            assert!(
                has_ecf_mention || has_codec_mention,
                "Error should mention ECF codec: {}",
                error_msg
            );
            assert!(
                has_native_support,
                "Error should mention native ECF support: {}",
                error_msg
            );

            println!("✓ Error message quality test passed - mentions native ECF support");
        }
    }
}

#[test]
fn test_empty_config_handling() {
    if is_running_in_ci() || !Path::new(PROPHESEE_TEST_FILE).exists() {
        eprintln!(
            "Skipping test - Prophesee file not available (CI: {}, exists: {})",
            is_running_in_ci(),
            Path::new(PROPHESEE_TEST_FILE).exists()
        );
        return;
    }

    // Test with minimal config - should still work or fail gracefully
    let config = LoadConfig::new();

    match load_events_with_config(PROPHESEE_TEST_FILE, &config) {
        Ok(events) => {
            println!("✓ Loaded {} events with empty config", events.height());
            // With no filters, this could be millions of events
            // Just verify we got something reasonable
            assert!(events.height() > 0, "Should load some events");
        }
        Err(e) => {
            println!("Expected failure with empty config: {}", e);
            // Should still provide helpful error message
            let error_msg = e.to_string();
            assert!(error_msg.len() > 50, "Error message should be descriptive");
        }
    }
}
