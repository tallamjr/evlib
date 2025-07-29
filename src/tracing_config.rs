//! Tracing configuration and initialization for evlib
//!
//! This module provides utilities for configuring structured logging using the `tracing` crate.
//! evlib uses tracing for all internal logging, allowing users to control verbosity and output format.

use tracing_subscriber::{filter::EnvFilter, fmt, prelude::*};

/// Initialize tracing with default configuration
///
/// This sets up structured logging with a reasonable default configuration:
/// - Logs to stderr
/// - Uses compact formatting
/// - Respects RUST_LOG environment variable
/// - Default level: INFO for evlib modules, WARN for others
///
/// # Examples
///
/// ```rust
/// use evlib::tracing_config;
///
/// // Initialize logging with defaults
/// tracing_config::init();
///
/// // Now evlib will log structured messages that can be controlled via RUST_LOG
/// // RUST_LOG=evlib=debug cargo run    # Enable debug logging for evlib
/// // RUST_LOG=evlib=warn cargo run     # Only show warnings and errors
/// ```
pub fn init() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        // Default: INFO level for evlib, WARN for everything else
        EnvFilter::new("warn,evlib=info")
    });

    tracing_subscriber::registry()
        .with(
            fmt::layer()
                .compact()
                .with_target(true)
                .with_thread_ids(false)
                .with_file(false)
                .with_line_number(false),
        )
        .with(filter)
        .init();
}

/// Initialize tracing with debug configuration
///
/// This enables detailed debug logging for development and troubleshooting:
/// - Shows debug messages for evlib modules
/// - Includes file names and line numbers
/// - More verbose output format
///
/// # Examples
///
/// ```rust
/// use evlib::tracing_config;
///
/// // Enable debug logging
/// tracing_config::init_debug();
/// ```
pub fn init_debug() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        // Debug level for evlib, INFO for everything else
        EnvFilter::new("info,evlib=debug")
    });

    tracing_subscriber::registry()
        .with(
            fmt::layer()
                .pretty()
                .with_target(true)
                .with_thread_ids(true)
                .with_file(true)
                .with_line_number(true),
        )
        .with(filter)
        .init();
}

/// Initialize tracing with custom filter
///
/// Allows full control over logging configuration using EnvFilter syntax.
///
/// # Arguments
///
/// * `filter` - EnvFilter string (e.g., "evlib=trace,tokio=info")
///
/// # Examples
///
/// ```rust
/// use evlib::tracing_config;
///
/// // Very verbose logging for ECF decoding
/// tracing_config::init_with_filter("evlib::ev_formats::prophesee_ecf_codec=trace");
///
/// // Only errors from evlib
/// tracing_config::init_with_filter("evlib=error");
///
/// // Specific module debugging
/// tracing_config::init_with_filter("evlib::ev_processing=debug,evlib::ev_formats=info");
/// ```
pub fn init_with_filter(filter: &str) {
    let filter = EnvFilter::new(filter);

    tracing_subscriber::registry()
        .with(
            fmt::layer()
                .compact()
                .with_target(true)
                .with_thread_ids(false)
                .with_file(false)
                .with_line_number(false),
        )
        .with(filter)
        .init();
}

/// Initialize tracing for testing
///
/// Sets up tracing with configuration suitable for tests:
/// - Captures logs for test output
/// - Uses compact format
/// - Default to WARN level to reduce noise
///
/// # Examples
///
/// ```rust
/// #[cfg(test)]
/// mod tests {
///     use evlib::tracing_config;
///
///     #[test]
///     fn test_event_loading() {
///         tracing_config::init_test();
///         // Your test code here - tracing logs will be captured
///     }
/// }
/// ```
pub fn init_test() {
    let _ = tracing_subscriber::registry()
        .with(fmt::layer().compact().with_target(false).with_test_writer())
        .with(EnvFilter::new("warn,evlib=info"))
        .try_init();
}

/// Common logging examples and patterns
pub mod examples {
    /// Example logging configurations for different use cases
    use tracing_subscriber::prelude::*;

    /// Production logging configuration
    ///
    /// Recommended for production deployments:
    /// - JSON formatted logs for structured parsing
    /// - INFO level for application logs
    /// - WARN level for dependencies
    pub fn init_production() {
        use tracing_subscriber::{fmt, EnvFilter};

        let filter =
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn,evlib=info"));

        tracing_subscriber::registry()
            .with(
                fmt::layer()
                    .json()
                    .with_target(true)
                    .with_current_span(false)
                    .with_span_list(false),
            )
            .with(filter)
            .init();
    }

    /// Development logging configuration
    ///
    /// Optimized for development work:
    /// - Pretty-printed output with colors
    /// - DEBUG level for evlib
    /// - File/line information included
    pub fn init_development() {
        use tracing_subscriber::{fmt, EnvFilter};

        let filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("info,evlib=debug"));

        tracing_subscriber::registry()
            .with(
                fmt::layer()
                    .pretty()
                    .with_target(true)
                    .with_file(true)
                    .with_line_number(true)
                    .with_thread_ids(true),
            )
            .with(filter)
            .init();
    }
}

/// Python bindings for tracing configuration
#[cfg(feature = "python")]
pub mod python {
    use pyo3::prelude::*;

    /// Initialize tracing with default configuration from Python
    ///
    /// Examples:
    ///     >>> import evlib
    ///     >>> evlib.tracing_config.init()
    #[pyfunction]
    #[pyo3(name = "init")]
    pub fn init_py() -> PyResult<()> {
        crate::tracing_config::init();
        Ok(())
    }

    /// Initialize tracing with debug configuration from Python
    ///
    /// Examples:
    ///     >>> import evlib
    ///     >>> evlib.tracing_config.init_debug()
    #[pyfunction]
    #[pyo3(name = "init_debug")]
    pub fn init_debug_py() -> PyResult<()> {
        crate::tracing_config::init_debug();
        Ok(())
    }

    /// Initialize tracing with custom filter from Python
    ///
    /// Args:
    ///     filter: Filter string (e.g., "evlib=debug")
    ///
    /// Examples:
    ///     >>> import evlib
    ///     >>> evlib.tracing_config.init_with_filter("evlib=trace")
    #[pyfunction]
    #[pyo3(name = "init_with_filter")]
    pub fn init_with_filter_py(filter: &str) -> PyResult<()> {
        crate::tracing_config::init_with_filter(filter);
        Ok(())
    }

    /// Initialize production logging configuration from Python
    ///
    /// Examples:
    ///     >>> import evlib
    ///     >>> evlib.tracing_config.init_production()
    #[pyfunction]
    #[pyo3(name = "init_production")]
    pub fn init_production_py() -> PyResult<()> {
        crate::tracing_config::examples::init_production();
        Ok(())
    }

    /// Initialize development logging configuration from Python
    ///
    /// Examples:
    ///     >>> import evlib
    ///     >>> evlib.tracing_config.init_development()
    #[pyfunction]
    #[pyo3(name = "init_development")]
    pub fn init_development_py() -> PyResult<()> {
        crate::tracing_config::examples::init_development();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing::{debug, error, info, warn};

    #[test]
    fn test_tracing_initialization() {
        // Test that we can initialize tracing without panicking
        init_test();

        // Test that we can emit log messages
        error!("Test error message");
        warn!("Test warning message");
        info!("Test info message");
        debug!("Test debug message");
    }

    #[test]
    fn test_structured_logging() {
        init_test();

        // Test structured logging patterns used in evlib
        info!(events = 1000, "Events loaded");
        debug!(chunk_size = 512, offset = 1024, "Processing chunk");
        warn!(model = "e2vid", "Model fallback activated");
        error!(
            error = "File not found",
            path = "/test/path",
            "Failed to load file"
        );
    }
}
