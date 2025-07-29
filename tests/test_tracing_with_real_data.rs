//! Tests for tracing functionality with real event data loading
//!
//! This test validates that tracing works correctly when actually loading
//! real event data files and doesn't interfere with the loading process.

use evlib::{ev_formats, tracing_config};
use std::path::Path;
use tracing::{debug, error, info, warn};

#[test]
fn test_tracing_with_real_event_loading() {
    // Initialize tracing for testing
    tracing_config::init_test();

    info!("Starting real data loading test with tracing");

    // Test with the slider_depth events.txt file
    let test_file = "data/slider_depth/events.txt";

    if !Path::new(test_file).exists() {
        warn!(
            file_path = test_file,
            "Test file not found, skipping real data test"
        );
        return;
    }

    info!(file_path = test_file, "Loading test event file");

    let config = ev_formats::LoadConfig::new().with_time_window(Some(0.0), Some(1000000.0)); // Load first second of data

    debug!(config = ?config, "Using load configuration");

    // Test format detection with tracing
    let detected_format = ev_formats::detect_event_format(test_file);
    info!(
        file_path = test_file,
        detected_format = ?detected_format,
        "Format detection completed"
    );

    // Test actual event loading with tracing
    match ev_formats::load_events_from_text(test_file, &config) {
        Ok(events) => {
            let event_count = events.len();
            info!(
                file_path = test_file,
                events_loaded = event_count,
                "Successfully loaded events with tracing enabled"
            );

            // Log some statistics with structured logging
            if !events.is_empty() {
                let first_event = &events[0];
                let last_event = &events[event_count - 1];

                debug!(
                    first_timestamp = first_event.t,
                    last_timestamp = last_event.t,
                    first_x = first_event.x,
                    first_y = first_event.y,
                    first_polarity = first_event.polarity,
                    "Event data statistics"
                );
            }

            assert!(event_count > 0, "Should have loaded some events");
        }
        Err(e) => {
            error!(
                file_path = test_file,
                error = %e,
                "Failed to load events"
            );
            panic!("Event loading failed: {}", e);
        }
    }

    info!("Real data loading test completed successfully");
}

#[test]
fn test_tracing_with_chunked_loading() {
    tracing_config::init_test();

    info!("Testing tracing with chunked event loading");

    let test_file = "data/slider_depth/events_chunk.txt";

    if !Path::new(test_file).exists() {
        warn!(file_path = test_file, "Chunk test file not found, skipping");
        return;
    }

    let config = ev_formats::LoadConfig::new();

    match ev_formats::load_events_from_text(test_file, &config) {
        Ok(events) => {
            info!(
                file_path = test_file,
                events_loaded = events.len(),
                "Chunked loading completed"
            );

            // Test that we can continue logging after loading
            debug!(
                memory_usage = "unknown",
                processing_time = "fast",
                "Post-loading metrics"
            );
        }
        Err(e) => {
            warn!(
                file_path = test_file,
                error = %e,
                "Chunked loading failed (may be expected)"
            );
        }
    }
}

#[test]
fn test_tracing_with_format_detection() {
    tracing_config::init_test();

    info!("Testing format detection with tracing");

    // Test format detection on various real files
    let test_files = vec![
        "data/slider_depth/events.txt",
        "data/slider_depth/events_chunk.txt",
        "data/slider_depth/calib.txt",
    ];

    for file_path in test_files {
        if Path::new(file_path).exists() {
            let format = ev_formats::detect_event_format(file_path);
            info!(
                file_path = file_path,
                detected_format = ?format,
                file_exists = true,
                "Format detection result"
            );
        } else {
            debug!(
                file_path = file_path,
                file_exists = false,
                "Skipping non-existent file"
            );
        }
    }
}

#[test]
fn test_error_logging_with_tracing() {
    tracing_config::init_test();

    info!("Testing error handling with tracing");

    // Test error logging when loading non-existent file
    let nonexistent_file = "data/nonexistent/fake_events.txt";

    match ev_formats::load_events_from_text(nonexistent_file, &ev_formats::LoadConfig::new()) {
        Ok(_) => {
            error!("Unexpectedly succeeded loading non-existent file");
            panic!("Should have failed");
        }
        Err(e) => {
            warn!(
                file_path = nonexistent_file,
                error_type = "file_not_found",
                error = %e,
                "Expected error occurred"
            );
            info!("Error handling test completed successfully");
        }
    }
}

#[test]
fn test_performance_logging() {
    tracing_config::init_test();

    info!("Testing performance logging patterns");

    // Simulate performance logging patterns that might be used in evlib
    let start_time = std::time::Instant::now();

    // Simulate some work
    let _config = ev_formats::LoadConfig::new()
        .with_time_window(Some(0.0), Some(1000.0))
        .with_polarity(Some(true));

    let processing_time = start_time.elapsed();

    info!(
        operation = "config_creation",
        duration_us = processing_time.as_micros(),
        "Performance measurement"
    );

    // Test logging of various metrics
    debug!(
        memory_allocated_bytes = 1024 * 1024,
        events_per_second = 50000,
        cache_hit_rate = 0.95,
        "System metrics"
    );

    info!("Performance logging test completed");
}
