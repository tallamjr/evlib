//! Integration tests for tracing functionality with event loading
//!
//! This test file validates that tracing works correctly when integrated
//! with existing evlib functionality like event loading.

use evlib::{ev_formats, tracing_config};
use tracing::{debug, error, info, warn};

#[test]
fn test_tracing_with_event_loading() {
    // Initialize tracing for testing
    tracing_config::init_test();

    // Test that we can emit log messages at different levels
    error!("Starting tracing integration test");
    warn!("This is a warning message");
    info!("This is an info message");
    debug!("This is a debug message");

    // Test structured logging with key-value pairs
    info!(
        test_name = "test_tracing_with_event_loading",
        "Starting event loading test"
    );

    // Test that event loading works with tracing enabled
    let config = ev_formats::LoadConfig::new();
    debug!(config = ?config, "Using configuration");

    // Since we can't guarantee test data files exist, we'll just test that
    // the tracing doesn't break the API calls
    let result = ev_formats::detect_event_format("nonexistent_file.txt");
    debug!(result = ?result, "Format detection result");

    // Test that we can use structured logging after event operations
    info!(
        operations_completed = 1,
        "Integration test completed successfully"
    );
}

#[test]
fn test_different_tracing_configurations() {
    // Test that different tracing configurations don't panic
    // Note: We can't actually test multiple init functions in the same process
    // because tracing can only be initialized once per process.
    // Instead, we test that the configuration objects can be created

    // Test that we can create different EnvFilter configurations
    use tracing_subscriber::filter::EnvFilter;

    let _filter1 = EnvFilter::new("evlib=debug");
    let _filter2 = EnvFilter::new("evlib=info,tokio=warn");
    let _filter3 = EnvFilter::new("warn");

    // Since init_test() was already called in other tests in this process,
    // we can just log some messages to verify tracing is working
    info!("Testing different configuration patterns");
    debug!("This debug message may or may not appear depending on current filter");
    warn!("Warning messages should typically be visible");
}

#[test]
fn test_structured_logging_patterns() {
    tracing_config::init_test();

    // Test common structured logging patterns used throughout evlib

    // File operations
    info!(
        file_path = "/test/path/events.txt",
        file_size = 1024,
        "Loading event file"
    );

    // Event processing
    debug!(
        events_count = 10000,
        processing_time_ms = 150,
        chunk_size = 512,
        "Processing events chunk"
    );

    // Error conditions
    warn!(
        error_type = "validation_error",
        event_index = 42,
        timestamp = 1234567890,
        "Invalid event detected"
    );

    // Performance metrics
    info!(
        memory_usage_mb = 128,
        processing_rate_eps = 50000,
        "Performance metrics"
    );
}

#[test]
fn test_format_detection_with_tracing() {
    tracing_config::init_test();

    info!("Testing format detection with tracing enabled");

    // Test format detection for different file extensions
    let test_files = vec![
        "test.evt2",
        "test.evt3",
        "test.hdf5",
        "test.h5",
        "test.txt",
        "test.raw",
        "unknown.xyz",
    ];

    let num_files = test_files.len();
    for file in &test_files {
        let format = ev_formats::detect_event_format(file);
        debug!(
            filename = file,
            detected_format = ?format,
            "Format detection result"
        );
    }

    info!(
        files_processed = num_files,
        "Format detection test completed"
    );
}

#[test]
fn test_config_creation_with_tracing() {
    tracing_config::init_test();

    info!("Testing configuration creation with tracing");

    // Test different LoadConfig variations
    let default_config = ev_formats::LoadConfig::new();
    debug!(config = ?default_config, "Default configuration");

    let custom_config = ev_formats::LoadConfig::new()
        .with_time_window(Some(0.0), Some(1000000.0))
        .with_polarity(Some(true));
    debug!(config = ?custom_config, "Custom configuration");

    info!("Configuration creation test completed");
}
