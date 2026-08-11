// Data formats module
// Handles reading and writing events from various file formats

// HDF5 is only available on Unix platforms (Linux/macOS) with the `hdf5` feature
#[cfg(all(unix, feature = "hdf5"))]
use hdf5_metno::File as H5File;
use polars::prelude::*;
use pyo3::prelude::*;
// memmap2 removed - no longer using unsafe binary format
use std::fs::File;
use std::io::{BufRead, BufReader, Result as IoResult};

// Format detection module
pub mod format_detector;
pub use format_detector::{
    detect_event_format, EventFormat, FormatDetectionError, FormatDetectionResult, FormatDetector,
    FormatMetadata,
};

// AEDAT format reader module
pub mod aedat_reader;
pub use aedat_reader::{AedatConfig, AedatError, AedatMetadata, AedatReader, AedatVersion};

// AEDAT 4.0 (DV framework) reader module
pub mod aedat4_reader;

// AER format reader module
pub mod aer_reader;
pub use aer_reader::{AerConfig, AerError, AerMetadata, AerReader, TimestampMode};

// EVT2 format reader module
pub mod evt2_reader;
pub use evt2_reader::{Evt2Config, Evt2Error, Evt2Metadata, Evt2Reader};

// EVT2.1 format reader module
pub mod evt21_reader;
pub use evt21_reader::{Evt21Config, Evt21Error, Evt21Metadata, Evt21Reader};

// EVT3 format reader module
pub mod evt3_reader;
pub use evt3_reader::{Evt3Config, Evt3Error, Evt3Metadata, Evt3Reader};

// EVNT TCP live-stream reader
pub mod evnt_tcp_reader;
pub use evnt_tcp_reader::{EvntEvent, EvntTcpReader, EvntTcpReaderError};

// Polarity encoding handler module
pub mod polarity_handler;
pub use polarity_handler::{
    PolarityConfig, PolarityEncoding, PolarityError, PolarityHandler, PolarityStats,
};

// Streaming module for large file processing
pub mod streaming;
pub use streaming::{
    estimate_memory_usage, should_use_streaming, Event, PolarsEventStreamer, StreamingConfig,
};

// ECF (Event Compression Format) codec for Prophesee HDF5 files
pub mod ecf_codec;
pub use ecf_codec::{ECFDecoder, ECFEncoder, EventCD};

// Prophesee ECF codec implementation
pub mod prophesee_ecf_codec;
pub use prophesee_ecf_codec::{PropheseeECFDecoder, PropheseeECFEncoder, PropheseeEvent};

// Native HDF5 reader with ECF support (Unix only, opt-in via `hdf5` feature)
// HDF5 is only available on Unix platforms (Linux/macOS) when the `hdf5`
// feature flag is enabled.
#[cfg(all(unix, feature = "hdf5"))]
pub mod hdf5_reader;
#[cfg(all(unix, feature = "hdf5"))]
pub use hdf5_reader::load_events_from_hdf5;

// DataFrame construction utilities for direct event processing
pub mod dataframe_builder;
pub use dataframe_builder::{
    calculate_optimal_chunk_size, create_empty_events_dataframe, EventDataFrameBuilder,
    EventDataFrameStreamer,
};

// Python bindings for DataFrame-based operations (defined inline below)

// Polars support integrated directly into file readers

/// Unit of `Event.t` as declared by the reader that produced the events.
/// There is deliberately no "guess from magnitude" option (2026-08-08 review, R4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimestampUnit {
    /// `Event.t` is seconds; converted with `(t * 1e6) as i64` (truncation kept
    /// so existing text loads are bit identical).
    Seconds,
    /// `Event.t` is an integer microsecond count stored losslessly in f64.
    Microseconds,
}

/// Configuration for loading events with filtering options
#[derive(Debug, Clone, Default)]
pub struct LoadConfig {
    /// Start time filter (inclusive)
    pub t_start: Option<f64>,
    /// End time filter (inclusive)
    pub t_end: Option<f64>,
    /// Minimum x coordinate (inclusive)
    pub min_x: Option<u16>,
    /// Maximum x coordinate (inclusive)
    pub max_x: Option<u16>,
    /// Minimum y coordinate (inclusive)
    pub min_y: Option<u16>,
    /// Maximum y coordinate (inclusive)
    pub max_y: Option<u16>,
    /// Polarity filter (true for positive, false for negative, None for both)
    pub polarity: Option<bool>,
    /// Sort events by timestamp after loading
    pub sort: bool,
    /// Custom column index for x coordinate (0-based, for text files)
    pub x_col: Option<usize>,
    /// Custom column index for y coordinate (0-based, for text files)
    pub y_col: Option<usize>,
    /// Custom column index for timestamp (0-based, for text files)
    pub t_col: Option<usize>,
    /// Custom column index for polarity (0-based, for text files)
    pub p_col: Option<usize>,
    /// Number of header lines to skip (for text files)
    pub header_lines: usize,
    /// Polarity encoding configuration
    pub polarity_encoding: Option<PolarityEncoding>,
}

impl LoadConfig {
    /// Create a new LoadConfig with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set time window filter
    pub fn with_time_window(mut self, t_start: Option<f64>, t_end: Option<f64>) -> Self {
        self.t_start = t_start;
        self.t_end = t_end;
        self
    }

    /// Set spatial bounds filter
    pub fn with_spatial_bounds(
        mut self,
        min_x: Option<u16>,
        max_x: Option<u16>,
        min_y: Option<u16>,
        max_y: Option<u16>,
    ) -> Self {
        self.min_x = min_x;
        self.max_x = max_x;
        self.min_y = min_y;
        self.max_y = max_y;
        self
    }

    /// Set polarity filter
    pub fn with_polarity(mut self, polarity: Option<bool>) -> Self {
        self.polarity = polarity;
        self
    }

    /// Enable sorting by timestamp
    pub fn with_sorting(mut self, sort: bool) -> Self {
        self.sort = sort;
        self
    }

    /// Set polarity encoding configuration
    pub fn with_polarity_encoding(mut self, encoding: PolarityEncoding) -> Self {
        self.polarity_encoding = Some(encoding);
        self
    }

    /// Set custom column mapping for text files
    pub fn with_custom_columns(
        mut self,
        t_col: Option<usize>,
        x_col: Option<usize>,
        y_col: Option<usize>,
        p_col: Option<usize>,
    ) -> Self {
        self.t_col = t_col;
        self.x_col = x_col;
        self.y_col = y_col;
        self.p_col = p_col;
        self
    }

    /// Set number of header lines to skip
    pub fn with_header_lines(mut self, header_lines: usize) -> Self {
        self.header_lines = header_lines;
        self
    }

    /// Check if an event passes all filters
    pub fn passes_filters(&self, event: &Event) -> bool {
        // Time window filter
        if let Some(t_start) = self.t_start {
            if event.t < t_start {
                return false;
            }
        }
        if let Some(t_end) = self.t_end {
            if event.t > t_end {
                return false;
            }
        }

        // Spatial bounds filter
        if let Some(min_x) = self.min_x {
            if event.x < min_x {
                return false;
            }
        }
        if let Some(max_x) = self.max_x {
            if event.x > max_x {
                return false;
            }
        }
        if let Some(min_y) = self.min_y {
            if event.y < min_y {
                return false;
            }
        }
        if let Some(max_y) = self.max_y {
            if event.y > max_y {
                return false;
            }
        }

        // Polarity filter
        if let Some(polarity) = self.polarity {
            if (event.polarity > 0) != polarity {
                return false;
            }
        }

        true
    }
}

/// # Arguments
/// * `path` - Path to the text file
/// * `config` - Configuration with filtering options
pub fn load_events_from_text(path: &str, config: &LoadConfig) -> IoResult<DataFrame> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut events = Vec::<Event>::new();

    // Estimate capacity if possible (reduce if filtering is likely to remove many events)
    if let Ok(metadata) = std::fs::metadata(path) {
        let file_size = metadata.len() as usize;
        // Assuming average of 20 bytes per line, but reduce estimate if filters are active
        let estimated_capacity = if config.t_start.is_some()
            || config.t_end.is_some()
            || config.min_x.is_some()
            || config.max_x.is_some()
            || config.min_y.is_some()
            || config.max_y.is_some()
            || config.polarity.is_some()
        {
            (file_size / 20) / 2 // Assume filters will remove ~50% of events
        } else {
            file_size / 20
        };
        events.reserve(estimated_capacity);
    } else {
        events.reserve(1000000); // Default pre-allocation
    }

    // Determine column indices
    let t_col = config.t_col.unwrap_or(0);
    let x_col = config.x_col.unwrap_or(1);
    let y_col = config.y_col.unwrap_or(2);
    let p_col = config.p_col.unwrap_or(3);

    let max_col = [t_col, x_col, y_col, p_col].iter().max().unwrap() + 1;

    let mut lines_processed = 0;

    for (line_num, line_res) in reader.lines().enumerate() {
        let line = line_res?;

        // Skip header lines
        if lines_processed < config.header_lines {
            lines_processed += 1;
            continue;
        }

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < max_col {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Line {}: Expected at least {max_col} values for column mapping, got {}",
                    line_num + 1,
                    parts.len()
                ),
            ));
        }

        // Parse values using custom column mapping
        let t = parts[t_col].parse::<f64>().map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Line {}: Invalid timestamp '{}': {e}",
                    line_num + 1,
                    parts[t_col]
                ),
            )
        })?;
        let x = parts[x_col].parse::<u16>().map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Line {}: Invalid x coordinate '{}': {e}",
                    line_num + 1,
                    parts[x_col]
                ),
            )
        })?;
        let y = parts[y_col].parse::<u16>().map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Line {}: Invalid y coordinate '{}': {e}",
                    line_num + 1,
                    parts[y_col]
                ),
            )
        })?;
        let polarity_raw = parts[p_col].parse::<i8>().map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Line {}: Invalid polarity '{}': {e}",
                    line_num + 1,
                    parts[p_col]
                ),
            )
        })?;

        // For text files, preserve original polarity encoding (0/1)
        // Don't convert to -1/1 like other formats
        let polarity = polarity_raw;

        let event = Event { t, x, y, polarity };

        // Apply filters
        if config.passes_filters(&event) {
            events.push(event);
        }

        // Early termination for time-sorted files
        if let Some(t_end) = config.t_end {
            if t > t_end {
                break; // Assume file is sorted by time
            }
        }
    }

    // Sort events if requested
    if config.sort {
        events.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));
    }

    python::build_polars_dataframe(&events, EventFormat::Text, TimestampUnit::Seconds)
        .map_err(|e| std::io::Error::other(format!("DataFrame conversion failed: {}", e)))
}

// Binary format (mmap_events) removed due to safety and reliability issues:
// 1. Unsafe memory operations with no validation
// 2. Assumes files contain raw Event structs (almost never true)
// 3. Produces misleading results with text files
// 4. No error handling for malformed binary data
//
// All file types now use the safe, reliable text parser

/// Generic load function with automatic format detection and filtering
pub fn load_events_with_config(
    path: &str,
    config: &LoadConfig,
) -> Result<DataFrame, Box<dyn std::error::Error>> {
    // When the HDF5 feature is disabled, fail fast with a feature-specific
    // error for .h5/.hdf5 paths before attempting any file I/O so the user
    // sees the rebuild hint rather than a generic "file not found".
    #[cfg(not(all(unix, feature = "hdf5")))]
    {
        let lower = path.to_ascii_lowercase();
        if lower.ends_with(".h5") || lower.ends_with(".hdf5") {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "HDF5 support not compiled in; rebuild with --features hdf5 on Linux or macOS.",
            )));
        }
    }

    // Use format detector to determine the file format
    let detection_result = format_detector::detect_event_format(path)?;

    match detection_result.format {
        #[cfg(all(unix, feature = "hdf5"))]
        EventFormat::HDF5 => {
            let events = load_events_from_hdf5(path, None)?;
            // Apply filters to the loaded events
            // TODO: Apply filters using Polars DataFrame operations
            // Sort if requested
            if config.sort {
                // TODO: Apply sorting using Polars DataFrame.sort() operation\n                // events = events.sort([\"t\"], SortMultipleOptions::new())?;
            }
            Ok(events)
        }
        #[cfg(not(all(unix, feature = "hdf5")))]
        EventFormat::HDF5 => Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "HDF5 support not compiled in; rebuild with --features hdf5 on Linux or macOS.",
        ))),
        EventFormat::Text => Ok(load_events_from_text(path, config)?),
        EventFormat::AEDAT1 | EventFormat::AEDAT2 | EventFormat::AEDAT3 | EventFormat::AEDAT4 => {
            // Use comprehensive AEDAT reader
            let aedat_config = AedatConfig {
                validate_timestamps: true,
                validate_coordinates: true,
                validate_polarity: true,
                skip_invalid_events: false,
                max_events: None,
                max_resolution: Some((1024, 1024)),
            };
            let reader = AedatReader::with_config(aedat_config);
            let (events, _metadata) = reader.read_file(path)?;
            Ok(events)
        }
        EventFormat::AER => {
            // Use comprehensive AER reader
            let aer_config = AerConfig::default()
                .with_validation(true, true) // Validate coordinates and skip invalid events
                .with_timestamp_generation(true, TimestampMode::Sequential, 0.0, 1e-6);

            let reader = AerReader::with_config(aer_config);
            let events = reader
                .read_with_config(path, config)
                .map_err(|e| -> Box<dyn std::error::Error> { Box::new(e) })?;
            Ok(events)
        }
        EventFormat::EVT2 => {
            // Use EVT2 reader
            let evt2_config = Evt2Config {
                validate_coordinates: false,
                skip_invalid_events: false,
                max_events: None,
                sensor_resolution: detection_result.metadata.sensor_resolution,
                chunk_size: 1_000_000,
                polarity_encoding: config.polarity_encoding,
            };
            let reader = Evt2Reader::with_config(evt2_config);
            let events = reader
                .read_with_config(path, config)
                .map_err(|e| -> Box<dyn std::error::Error> { Box::new(e) })?;
            Ok(events)
        }
        EventFormat::EVT21 => {
            // Use EVT2.1 reader
            let evt21_config = Evt21Config {
                validate_coordinates: false,
                skip_invalid_events: false,
                max_events: None,
                sensor_resolution: detection_result.metadata.sensor_resolution,
                chunk_size: 500_000,
                polarity_encoding: config.polarity_encoding,
                decode_vectorized: true,
            };
            let reader = Evt21Reader::with_config(evt21_config);
            let events = reader
                .read_with_config(path, config)
                .map_err(|e| -> Box<dyn std::error::Error> { Box::new(e) })?;
            Ok(events)
        }
        EventFormat::EVT3 => {
            // Use EVT3 reader
            let evt3_config = Evt3Config {
                validate_coordinates: false, // Disable validation for better compatibility
                skip_invalid_events: false,
                max_events: None,
                sensor_resolution: detection_result.metadata.sensor_resolution,
                chunk_size: 1_000_000,
                polarity_encoding: config.polarity_encoding,
            };
            let reader = Evt3Reader::with_config(evt3_config);
            let events = reader
                .read_with_config(path, config)
                .map_err(|e| -> Box<dyn std::error::Error> { Box::new(e) })?;
            Ok(events)
        }
        EventFormat::Binary => {
            // Binary format is not supported for safety reasons
            let format = detection_result.format;
            let confidence = detection_result.confidence;
            Err(format!(
                "Binary format is not supported for safety reasons. Detected format: {format} (confidence: {confidence:.2})"
            ).into())
        }
        EventFormat::Unknown => {
            // Fall back to text format for unknown files
            Ok(load_events_from_text(path, config)?)
        }
    }
}

/// Struct for iterating through a text file of events line by line
/// without loading everything into memory at once
pub struct EventFileIterator {
    reader: BufReader<File>,
}

impl EventFileIterator {
    /// Create a new iterator from a text file path
    pub fn new(path: &str) -> IoResult<Self> {
        let file = File::open(path)?;
        Ok(EventFileIterator {
            reader: BufReader::new(file),
        })
    }
}

impl Iterator for EventFileIterator {
    type Item = IoResult<Event>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut line = String::new();

        // Read the next line
        match self.reader.read_line(&mut line) {
            Ok(0) => None, // EOF
            Ok(_) => {
                // Skip empty lines and comments
                if line.trim().is_empty() || line.starts_with('#') {
                    return self.next();
                }

                // Parse the line
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() < 4 {
                    return self.next(); // Not enough fields
                }

                // Parse values
                let t = match parts[0].parse::<f64>() {
                    Ok(v) => v,
                    Err(e) => {
                        return Some(Err(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))
                    }
                };

                let x = match parts[1].parse::<u16>() {
                    Ok(v) => v,
                    Err(e) => {
                        return Some(Err(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))
                    }
                };

                let y = match parts[2].parse::<u16>() {
                    Ok(v) => v,
                    Err(e) => {
                        return Some(Err(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))
                    }
                };

                let p = match parts[3].parse::<i8>() {
                    Ok(v) => v,
                    Err(e) => {
                        return Some(Err(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))
                    }
                };

                // Create and return event
                Some(Ok(Event {
                    t,
                    x,
                    y,
                    polarity: if p > 0 { 1 } else { -1 },
                }))
            }
            Err(e) => Some(Err(e)),
        }
    }
}

// Window-based event iterator that returns chunks of events based on time windows
pub struct TimeWindowIter<'a> {
    events: &'a Vec<Event>,
    window_duration: f64,
    current_idx: usize,
    start_time: f64,
    end_time: f64,
}

impl<'a> TimeWindowIter<'a> {
    /// Create a new iterator that returns time-windowed chunks of events
    ///
    /// # Arguments
    /// * `events` - Event array to iterate over
    /// * `window_duration` - Duration of each time window in seconds
    pub fn new(events: &'a Vec<Event>, window_duration: f64) -> Self {
        let start_time = if !events.is_empty() { events[0].t } else { 0.0 };

        let end_time = start_time + window_duration;

        TimeWindowIter {
            events,
            window_duration,
            current_idx: 0,
            start_time,
            end_time,
        }
    }
}

impl Iterator for TimeWindowIter<'_> {
    type Item = Vec<Event>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.events.len() {
            return None;
        }

        let mut window = Vec::new();
        let mut idx = self.current_idx;

        // Collect events within current time window
        while idx < self.events.len() && self.events[idx].t < self.end_time {
            window.push(self.events[idx]);
            idx += 1;
        }

        // Update state for next iteration
        self.current_idx = idx;
        self.start_time = self.end_time;
        self.end_time += self.window_duration;

        // Only return Some if we found events in this window
        if window.is_empty() {
            self.next()
        } else {
            Some(window)
        }
    }
}

/// Python bindings for the formats module
pub mod python {
    use super::*;
    use numpy::PyReadonlyArray1;
    use polars::prelude::IntoLazy;
    use std::io::Write;

    // NOTE: convert_polarity function removed - functionality moved to vectorized Polars operations
    // in build_polars_dataframe() for better performance

    /// Build Polars DataFrame directly from events using Series builders for optimal memory efficiency.
    ///
    /// `unit` declares what `Event.t` holds for every event in `events`; there is no
    /// magnitude-based guessing (2026-08-08 review, R4). Callers pass the unit their
    /// reader actually produced.
    pub fn build_polars_dataframe(
        events: &[Event],
        format: EventFormat,
        unit: TimestampUnit,
    ) -> Result<polars::prelude::DataFrame, polars::prelude::PolarsError> {
        use polars::prelude::*;

        let len = events.len();

        if len == 0 {
            // Create empty DataFrame with proper schema
            let empty_x = Series::new("x".into(), Vec::<i16>::new());
            let empty_y = Series::new("y".into(), Vec::<i16>::new());
            let empty_timestamp = Series::new("t".into(), Vec::<i64>::new())
                .cast(&DataType::Duration(TimeUnit::Microseconds))?;
            let empty_polarity = Series::new("polarity".into(), Vec::<i8>::new());

            return DataFrame::new(vec![
                empty_x.into(),
                empty_y.into(),
                empty_timestamp.into(),
                empty_polarity.into(),
            ]);
        }

        // Use optimal data types for memory efficiency
        // x, y: Int16 (sufficient for coordinates, saves 50% memory vs Int32)
        // timestamp: Int64 (required for microsecond precision)
        // polarity: Int8 (sufficient for -1/0/1 values, saves 87.5% memory vs Int64)
        let mut x_builder = PrimitiveChunkedBuilder::<Int16Type>::new("x".into(), len);
        let mut y_builder = PrimitiveChunkedBuilder::<Int16Type>::new("y".into(), len);
        let mut timestamp_builder = PrimitiveChunkedBuilder::<Int64Type>::new("t".into(), len);
        let mut polarity_builder = PrimitiveChunkedBuilder::<Int8Type>::new("polarity".into(), len);

        // Single iteration with direct population - zero intermediate copies
        // Store polarity as raw bool first, convert vectorized later
        for event in events {
            x_builder.append_value(event.x as i16);
            y_builder.append_value(event.y as i16);
            timestamp_builder.append_value(match unit {
                TimestampUnit::Seconds => (event.t * 1_000_000.0) as i64,
                TimestampUnit::Microseconds => event.t as i64,
            });
            // Store raw bool polarity (0/1) - will convert vectorized later
            polarity_builder.append_value(event.polarity);
        }

        // Build Series from builders
        let x_series = x_builder.finish().into_series();
        let y_series = y_builder.finish().into_series();
        let polarity_series_raw = polarity_builder.finish().into_series();

        // Convert timestamp to Duration type
        let timestamp_series = timestamp_builder
            .finish()
            .into_series()
            .cast(&DataType::Duration(TimeUnit::Microseconds))?;

        // Create initial DataFrame with raw polarity
        let df = DataFrame::new(vec![
            x_series.into(),
            y_series.into(),
            timestamp_series.into(),
            polarity_series_raw.into(),
        ])?;

        // VECTORIZED polarity conversion (much faster than per-event)
        let df = match format {
            EventFormat::EVT2 | EventFormat::EVT21 | EventFormat::EVT3 => {
                // EVT2 family: Convert 0/1 to -1/1 using vectorized operations
                df.lazy()
                    .with_column(
                        when(col("polarity").eq(lit(0)))
                            .then(lit(-1i8))
                            .otherwise(lit(1i8))
                            .alias("polarity")
                            .cast(DataType::Int8),
                    )
                    .collect()?
            }
            #[cfg(not(windows))]
            EventFormat::HDF5 => {
                // HDF5: Convert 0/1 to -1/1 for proper polarity encoding
                df.lazy()
                    .with_column(
                        when(col("polarity").eq(lit(0)))
                            .then(lit(-1i8))
                            .otherwise(lit(1i8))
                            .alias("polarity")
                            .cast(DataType::Int8),
                    )
                    .collect()?
            }
            #[cfg(windows)]
            EventFormat::HDF5 => {
                return Err(PolarsError::ComputeError(
                    "HDF5 support is disabled on Windows due to build complexity.".into(),
                ));
            }
            _ => {
                // Text and other formats: Keep 0/1 encoding as-is, but ensure Int8 type
                df.lazy()
                    .with_column(col("polarity").cast(DataType::Int8))
                    .collect()?
            }
        };

        Ok(df)
    }

    /// Convert Polars DataFrame to Python dictionary for LazyFrame creation
    fn return_polars_lazyframe_to_python(
        py: Python<'_>,
        lf: polars::prelude::LazyFrame,
    ) -> PyResult<PyObject> {
        // Convert Polars LazyFrame to Python object directly
        // This leverages polars-python bindings for maximum efficiency
        use pyo3::types::PyModule;

        // Import polars module in Python
        let polars_module = PyModule::import(py, "polars")?;

        // Convert the LazyFrame to Python
        // For now, we'll collect to DataFrame and convert to Python, then make lazy
        let df = lf.collect().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to collect LazyFrame: {e}"
            ))
        })?;

        // Convert DataFrame to Python dict with schema preservation
        let (data_dict, schema_dict) = polars_dataframe_to_python_dict_with_schema(py, df)?;

        // Create Polars DataFrame from dict in Python with explicit schema and return as LazyFrame
        let py_df = polars_module.call_method1("DataFrame", (data_dict, schema_dict))?;
        let py_lazyframe = py_df.call_method0("lazy")?;

        Ok(py_lazyframe.into())
    }

    fn polars_dataframe_to_python_dict_with_schema(
        py: Python<'_>,
        df: polars::prelude::DataFrame,
    ) -> PyResult<(PyObject, PyObject)> {
        use polars::prelude::*;
        use pyo3::types::{PyDict, PyModule};

        let mut data_dict: std::collections::HashMap<String, PyObject> =
            std::collections::HashMap::new();
        let schema_dict = PyDict::new(py);

        // Import polars for Python type creation
        let polars_module = PyModule::import(py, "polars")?;

        for column in df.get_columns() {
            let column_name = column.name();
            let (column_data, py_dtype) = match column.dtype() {
                DataType::Int16 => {
                    let values: Vec<i16> = column
                        .i16()
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                                "Failed to extract i16 column: {e}"
                            ))
                        })?
                        .into_no_null_iter()
                        .collect();
                    let py_type = polars_module.getattr("Int16")?;
                    (values.into_pyobject(py)?.into(), py_type)
                }
                DataType::Int32 => {
                    let values: Vec<i32> = column
                        .i32()
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                                "Failed to extract i32 column: {e}"
                            ))
                        })?
                        .into_no_null_iter()
                        .collect();
                    let py_type = polars_module.getattr("Int32")?;
                    (values.into_pyobject(py)?.into(), py_type)
                }
                DataType::Int8 => {
                    let values: Vec<i8> = column
                        .i8()
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                                "Failed to extract i8 column: {e}"
                            ))
                        })?
                        .into_no_null_iter()
                        .collect();
                    let py_type = polars_module.getattr("Int8")?;
                    (values.into_pyobject(py)?.into(), py_type)
                }
                DataType::Duration(TimeUnit::Microseconds) => {
                    let values: Vec<i64> = column
                        .duration()
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                                "Failed to extract duration column: {e}"
                            ))
                        })?
                        .into_no_null_iter()
                        .collect();
                    let duration_type = polars_module.call_method1("Duration", ("us",))?;
                    (values.into_pyobject(py)?.into(), duration_type)
                }
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                        "Unsupported column type: {:?}",
                        column.dtype()
                    )))
                }
            };

            data_dict.insert(column_name.to_string(), column_data);
            schema_dict.set_item(column_name.as_str(), py_dtype)?;
        }

        Ok((data_dict.into_pyobject(py)?.into(), schema_dict.into()))
    }

    /// Load events from a file and return the full decoded frame.
    ///
    /// Row filters are deliberately not exposed here: they were broken three
    /// independent ways (one-sided bounds ignored, seconds compared against
    /// microsecond values, HDF5/AEDAT paths ignoring them entirely; 2026-08-08
    /// review, R3). Use evlib.load_events, which applies filters as Polars
    /// expressions on the returned LazyFrame.
    #[pyfunction]
    #[pyo3(
        signature = (path, sort=false, x_col=None, y_col=None, t_col=None, p_col=None, header_lines=0),
        name = "load_events"
    )]
    #[allow(clippy::too_many_arguments)]
    pub fn load_events_py(
        py: Python<'_>,
        path: &str,
        sort: bool,
        x_col: Option<usize>,
        y_col: Option<usize>,
        t_col: Option<usize>,
        p_col: Option<usize>,
        header_lines: usize,
    ) -> PyResult<PyObject> {
        let config = LoadConfig::new()
            .with_sorting(sort)
            .with_custom_columns(t_col, x_col, y_col, p_col)
            .with_header_lines(header_lines);

        // When the HDF5 feature is disabled, fail fast with a feature-specific
        // error for .h5/.hdf5 paths so the user sees the rebuild hint rather
        // than a generic format-detection failure.
        #[cfg(not(all(unix, feature = "hdf5")))]
        {
            let lower = path.to_ascii_lowercase();
            if lower.ends_with(".h5") || lower.ends_with(".hdf5") {
                return Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(
                    "HDF5 support not compiled in; rebuild with --features hdf5 on Linux or macOS.",
                ));
            }
        }

        // Detect format for proper polarity encoding
        let _format_result = detect_event_format(path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to detect format: {e}"))
        })?;

        // Load events using existing Rust logic
        let events = load_events_with_config(path, &config).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to load events: {e}"))
        })?;

        // Return Polars DataFrame/LazyFrame directly to Python
        // load_events_with_config already returns a DataFrame, so use it directly
        {
            // Return Polars LazyFrame directly to Python
            // This is much more efficient than converting to dict and back
            return_polars_lazyframe_to_python(py, events.lazy())
        }
    }

    /// Save events to an HDF5 file
    #[pyfunction]
    #[pyo3(name = "save_events_to_hdf5")]
    #[cfg(all(unix, feature = "hdf5"))]
    pub fn save_events_to_hdf5_py(
        xs: PyReadonlyArray1<i64>,
        ys: PyReadonlyArray1<i64>,
        ts: PyReadonlyArray1<f64>,
        ps: PyReadonlyArray1<i64>,
        path: &str,
    ) -> PyResult<()> {
        // Validate array lengths
        let n = ts.len()?;
        if xs.len()? != n || ys.len()? != n || ps.len()? != n {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Arrays must have the same length",
            ));
        }

        // Create HDF5 file
        let file = H5File::create(path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to create HDF5 file: {e}"))
        })?;

        // Create a group to store the data
        let group = file.create_group("events").map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to create group: {e}"))
        })?;

        // Convert arrays to Rust vectors
        let xs_vec: Vec<u16> = xs.as_array().iter().map(|&x| x as u16).collect();
        let ys_vec: Vec<u16> = ys.as_array().iter().map(|&y| y as u16).collect();
        let ts_vec: Vec<f64> = ts.as_slice().unwrap().to_vec();
        let ps_vec: Vec<i8> = ps
            .as_array()
            .iter()
            .map(|&p| {
                if p == -1 {
                    -1i8
                } else if p == 1 {
                    1i8
                } else {
                    0i8
                }
            })
            .collect();

        // Create datasets for each component
        let xs_shape = [n];
        let xs_dataset = group
            .new_dataset::<u16>()
            .shape(xs_shape)
            .create("xs")
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                    "Failed to create xs dataset: {e}"
                ))
            })?;
        xs_dataset.write(&xs_vec).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to write xs data: {e}"))
        })?;

        let ys_dataset = group
            .new_dataset::<u16>()
            .shape(xs_shape)
            .create("ys")
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                    "Failed to create ys dataset: {e}"
                ))
            })?;
        ys_dataset.write(&ys_vec).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to write ys data: {e}"))
        })?;

        let ts_dataset = group
            .new_dataset::<f64>()
            .shape(xs_shape)
            .create("ts")
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                    "Failed to create ts dataset: {e}"
                ))
            })?;
        ts_dataset.write(&ts_vec).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to write ts data: {e}"))
        })?;

        let ps_dataset = group
            .new_dataset::<i8>()
            .shape(xs_shape)
            .create("ps")
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                    "Failed to create ps dataset: {e}"
                ))
            })?;
        ps_dataset.write(&ps_vec).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to write ps data: {e}"))
        })?;

        Ok(())
    }

    /// Save events to a text file, one event per line: "t x y p"
    #[pyfunction]
    #[pyo3(name = "save_events_to_text")]
    pub fn save_events_to_text_py(
        xs: PyReadonlyArray1<i64>,
        ys: PyReadonlyArray1<i64>,
        ts: PyReadonlyArray1<f64>,
        ps: PyReadonlyArray1<i64>,
        path: &str,
    ) -> PyResult<()> {
        // Validate array lengths
        let n = ts.len()?;
        if xs.len()? != n || ys.len()? != n || ps.len()? != n {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Arrays must have the same length",
            ));
        }

        // Create output file
        let mut file = std::fs::File::create(path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to create file: {e}"))
        })?;

        // Write header
        file.write_all(b"# timestamp x y polarity\n").map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to write header: {e}"))
        })?;

        // Write events
        for i in 0..n {
            let line = format!(
                "{:.12} {} {} {}\n",
                ts.get(i).unwrap(),
                xs.get(i).unwrap(),
                ys.get(i).unwrap(),
                ps.get(i).unwrap()
            );
            file.write_all(line.as_bytes()).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to write line: {e}"))
            })?;
        }

        Ok(())
    }

    /// Detect the format of an event data file
    #[pyfunction]
    #[pyo3(name = "detect_format")]
    pub fn detect_format_py(
        path: &str,
    ) -> PyResult<(String, f64, std::collections::HashMap<String, String>)> {
        let result = format_detector::detect_event_format(path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Format detection failed: {e}"))
        })?;

        Ok((
            result.format.to_string(),
            result.confidence,
            result.metadata.properties,
        ))
    }

    /// Get format description
    #[pyfunction]
    #[pyo3(name = "get_format_description")]
    pub fn get_format_description_py(format: &str) -> PyResult<String> {
        let event_format = match format {
            "Text" => EventFormat::Text,
            "HDF5" => EventFormat::HDF5,
            "AER" => EventFormat::AER,
            "AEDAT 1.0" => EventFormat::AEDAT1,
            "AEDAT 2.0" => EventFormat::AEDAT2,
            "AEDAT 3.1" => EventFormat::AEDAT3,
            "AEDAT 4.0" => EventFormat::AEDAT4,
            "EVT2" => EventFormat::EVT2,
            "EVT2.1" => EventFormat::EVT21,
            "EVT3" => EventFormat::EVT3,
            "Binary" => EventFormat::Binary,
            _ => EventFormat::Unknown,
        };

        Ok(FormatDetector::get_format_description(&event_format).to_string())
    }

    /// Test Prophesee ECF decoder with raw compressed data
    #[pyfunction]
    #[pyo3(name = "test_prophesee_ecf_decode")]
    pub fn test_prophesee_ecf_decode_py(
        compressed_data: &[u8],
        debug: Option<bool>,
    ) -> PyResult<Vec<(u16, u16, i16, i64)>> {
        use crate::ev_formats::prophesee_ecf_codec::PropheseeECFDecoder;

        let decoder = PropheseeECFDecoder::new().with_debug(debug.unwrap_or(false));

        match decoder.decode(compressed_data) {
            Ok(events) => {
                // Convert PropheseeEvent to Python-friendly tuple
                let result: Vec<(u16, u16, i16, i64)> =
                    events.into_iter().map(|e| (e.x, e.y, e.p, e.t)).collect();
                Ok(result)
            }
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "ECF decoding failed: {}",
                e
            ))),
        }
    }

    // ============================================================================
    // MIGRATED FROM ev_core::python - Legacy compatibility functions
    // These functions maintain backward compatibility for existing Python code
    // ============================================================================

    /// Convert events to a block representation (direct numpy array processing)
    #[pyfunction]
    #[pyo3(name = "events_to_block")]
    pub fn events_to_block_py(
        py: Python<'_>,
        xs: PyReadonlyArray1<i64>,
        ys: PyReadonlyArray1<i64>,
        ts: PyReadonlyArray1<f64>,
        ps: PyReadonlyArray1<i64>,
    ) -> PyResult<PyObject> {
        use ndarray::Array2;
        use numpy::IntoPyArray;

        let xs_array = xs.as_array();
        let ys_array = ys.as_array();
        let ts_array = ts.as_array();
        let ps_array = ps.as_array();

        let len = xs_array
            .len()
            .min(ys_array.len())
            .min(ts_array.len())
            .min(ps_array.len());

        // Create a 2D array with shape (n, 4) directly from numpy arrays
        let mut block = Array2::<f64>::zeros((len, 4));

        // Fill in the values: [x, y, t, p]
        for i in 0..len {
            block[[i, 0]] = xs_array[i] as f64;
            block[[i, 1]] = ys_array[i] as f64;
            block[[i, 2]] = ts_array[i];
            block[[i, 3]] = if ps_array[i] > 0 { 1.0 } else { 0.0 };
        }

        Ok(block.into_pyarray(py).into())
    }

    /// Merge multiple sets of events into a single chronologically sorted list (direct array processing)
    #[pyfunction]
    #[pyo3(name = "merge_events")]
    pub fn merge_events_py(
        py: Python<'_>,
        event_sets: &Bound<'_, pyo3::types::PyTuple>,
    ) -> PyResult<PyObject> {
        use ndarray::Array1;
        use numpy::IntoPyArray;
        use pyo3::types::PyTuple;

        // Collect all data from input arrays
        let mut all_xs = Vec::new();
        let mut all_ys = Vec::new();
        let mut all_ts = Vec::new();
        let mut all_ps = Vec::new();

        for event_set in event_sets.iter() {
            let tuple = event_set.downcast::<PyTuple>()?;
            if tuple.len() != 4 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Each event set must be a tuple of (xs, ys, ts, ps)",
                ));
            }
            let xs = tuple.get_item(0)?.extract::<PyReadonlyArray1<i64>>()?;
            let ys = tuple.get_item(1)?.extract::<PyReadonlyArray1<i64>>()?;
            let ts = tuple.get_item(2)?.extract::<PyReadonlyArray1<f64>>()?;
            let ps = tuple.get_item(3)?.extract::<PyReadonlyArray1<i64>>()?;

            let xs_array = xs.as_array();
            let ys_array = ys.as_array();
            let ts_array = ts.as_array();
            let ps_array = ps.as_array();

            let len = xs_array
                .len()
                .min(ys_array.len())
                .min(ts_array.len())
                .min(ps_array.len());

            for i in 0..len {
                all_xs.push(xs_array[i]);
                all_ys.push(ys_array[i]);
                all_ts.push(ts_array[i]);
                all_ps.push(ps_array[i]);
            }
        }

        // Create indices for sorting by timestamp
        let mut indices: Vec<usize> = (0..all_ts.len()).collect();
        indices.sort_by(|&a, &b| all_ts[a].partial_cmp(&all_ts[b]).unwrap());

        // Rearrange arrays according to sorted indices
        let xs_sorted: Vec<i64> = indices.iter().map(|&i| all_xs[i]).collect();
        let ys_sorted: Vec<i64> = indices.iter().map(|&i| all_ys[i]).collect();
        let ts_sorted: Vec<f64> = indices.iter().map(|&i| all_ts[i]).collect();
        // Convert polarities from -1/1 to 0/1 for consistency with events_to_block_py
        let ps_sorted: Vec<i64> = indices
            .iter()
            .map(|&i| if all_ps[i] > 0 { 1 } else { 0 })
            .collect();

        // Convert arrays to Python objects
        let xs_py: PyObject = Array1::from(xs_sorted).into_pyarray(py).into();
        let ys_py: PyObject = Array1::from(ys_sorted).into_pyarray(py).into();
        let ts_py: PyObject = Array1::from(ts_sorted).into_pyarray(py).into();
        let ps_py: PyObject = Array1::from(ps_sorted).into_pyarray(py).into();

        // Create result tuple
        let tuple = PyTuple::new(py, [xs_py, ys_py, ts_py, ps_py])?;
        Ok(tuple.into())
    }

    /// Add random events drawn from a uniform distribution (migrated from ev_core)
    #[pyfunction]
    #[pyo3(name = "add_random_events")]
    #[pyo3(signature = (xs, ys, ts, ps, to_add, sensor_resolution=None, sort=true, return_merged=true))]
    #[allow(clippy::too_many_arguments)]
    pub fn add_random_events_py(
        py: Python<'_>,
        xs: PyReadonlyArray1<i64>,
        ys: PyReadonlyArray1<i64>,
        ts: PyReadonlyArray1<f64>,
        ps: PyReadonlyArray1<i64>,
        to_add: usize,
        sensor_resolution: Option<(i64, i64)>,
        sort: bool,
        return_merged: bool,
    ) -> PyResult<PyObject> {
        use ndarray::Array1;
        use numpy::IntoPyArray;
        use pyo3::types::PyTuple;
        use rand::prelude::*;

        let xs_array = xs.as_array();
        let ys_array = ys.as_array();
        let ts_array = ts.as_array();
        let ps_array = ps.as_array();

        // Generate random events
        let max_x = match sensor_resolution {
            Some((w, _)) => w - 1,
            None => xs_array.fold(0, |acc, &x| acc.max(x)),
        };

        let max_y = match sensor_resolution {
            Some((_, h)) => h - 1,
            None => ys_array.fold(0, |acc, &y| acc.max(y)),
        };

        let mut rng = thread_rng();

        let mut xs_new = Vec::with_capacity(to_add);
        let mut ys_new = Vec::with_capacity(to_add);
        let mut ts_new = Vec::with_capacity(to_add);
        let mut ps_new = Vec::with_capacity(to_add);

        let min_ts = ts_array.fold(f64::INFINITY, |acc, &t| acc.min(t));
        let max_ts = ts_array.fold(f64::NEG_INFINITY, |acc, &t| acc.max(t));

        for _ in 0..to_add {
            xs_new.push(rng.gen_range(0..=max_x));
            ys_new.push(rng.gen_range(0..=max_y));
            ts_new.push(rng.gen_range(min_ts..=max_ts));
            ps_new.push(if rng.gen_bool(0.5) { 1 } else { -1 });
        }

        if return_merged {
            // Merge both arrays
            let mut all_xs = Vec::with_capacity(xs_array.len() + xs_new.len());
            let mut all_ys = Vec::with_capacity(ys_array.len() + ys_new.len());
            let mut all_ts: Vec<f64> = Vec::with_capacity(ts_array.len() + ts_new.len());
            let mut all_ps = Vec::with_capacity(ps_array.len() + ps_new.len());

            all_xs.extend(xs_array.iter());
            all_xs.extend(xs_new.iter());

            all_ys.extend(ys_array.iter());
            all_ys.extend(ys_new.iter());

            all_ts.extend(ts_array.iter());
            all_ts.extend(ts_new.iter());

            all_ps.extend(ps_array.iter());
            all_ps.extend(ps_new.iter());

            let merged_xs = Array1::from(all_xs);
            let merged_ys = Array1::from(all_ys);
            let merged_ts = Array1::from(all_ts);
            let merged_ps = Array1::from(all_ps);

            if sort {
                // Sort by timestamp
                let mut indices: Vec<usize> = (0..merged_ts.len()).collect();
                indices.sort_by(|&i, &j| merged_ts[i].partial_cmp(&merged_ts[j]).unwrap());

                let sorted_xs = indices
                    .iter()
                    .map(|&i| merged_xs[i])
                    .collect::<Array1<i64>>();
                let sorted_ys = indices
                    .iter()
                    .map(|&i| merged_ys[i])
                    .collect::<Array1<i64>>();
                let sorted_ts = indices
                    .iter()
                    .map(|&i| merged_ts[i])
                    .collect::<Array1<f64>>();
                let sorted_ps = indices
                    .iter()
                    .map(|&i| merged_ps[i])
                    .collect::<Array1<i64>>();

                // Convert arrays to Python objects
                let xs_py: PyObject = sorted_xs.into_pyarray(py).into();
                let ys_py: PyObject = sorted_ys.into_pyarray(py).into();
                let ts_py: PyObject = sorted_ts.into_pyarray(py).into();
                let ps_py: PyObject = sorted_ps.into_pyarray(py).into();

                // Create result tuple
                let tuple = PyTuple::new(py, [xs_py, ys_py, ts_py, ps_py])?;
                Ok(tuple.into())
            } else {
                // Convert arrays to Python objects without sorting
                let xs_py: PyObject = merged_xs.into_pyarray(py).into();
                let ys_py: PyObject = merged_ys.into_pyarray(py).into();
                let ts_py: PyObject = merged_ts.into_pyarray(py).into();
                let ps_py: PyObject = merged_ps.into_pyarray(py).into();

                // Create result tuple
                let tuple = PyTuple::new(py, [xs_py, ys_py, ts_py, ps_py])?;
                Ok(tuple.into())
            }
        } else {
            // Return only the new events
            let xs_new_array = Array1::from(xs_new);
            let ys_new_array = Array1::from(ys_new);
            let ts_new_array = Array1::from(ts_new);
            let ps_new_array = Array1::from(ps_new);

            if sort {
                // Sort by timestamp
                let mut indices: Vec<usize> = (0..ts_new_array.len()).collect();
                indices.sort_by(|&i, &j| ts_new_array[i].partial_cmp(&ts_new_array[j]).unwrap());

                let sorted_xs = indices
                    .iter()
                    .map(|&i| xs_new_array[i])
                    .collect::<Array1<i64>>();
                let sorted_ys = indices
                    .iter()
                    .map(|&i| ys_new_array[i])
                    .collect::<Array1<i64>>();
                let sorted_ts = indices
                    .iter()
                    .map(|&i| ts_new_array[i])
                    .collect::<Array1<f64>>();
                let sorted_ps = indices
                    .iter()
                    .map(|&i| ps_new_array[i])
                    .collect::<Array1<i64>>();

                // Convert arrays to Python objects
                let xs_py: PyObject = sorted_xs.into_pyarray(py).into();
                let ys_py: PyObject = sorted_ys.into_pyarray(py).into();
                let ts_py: PyObject = sorted_ts.into_pyarray(py).into();
                let ps_py: PyObject = sorted_ps.into_pyarray(py).into();

                // Create result tuple
                let tuple = PyTuple::new(py, [xs_py, ys_py, ts_py, ps_py])?;
                Ok(tuple.into())
            } else {
                // Convert arrays to Python objects
                let xs_py: PyObject = xs_new_array.into_pyarray(py).into();
                let ys_py: PyObject = ys_new_array.into_pyarray(py).into();
                let ts_py: PyObject = ts_new_array.into_pyarray(py).into();
                let ps_py: PyObject = ps_new_array.into_pyarray(py).into();

                // Create result tuple
                let tuple = PyTuple::new(py, [xs_py, ys_py, ts_py, ps_py])?;
                Ok(tuple.into())
            }
        }
    }

    /// Remove events by random selection (migrated from ev_core)
    #[pyfunction]
    #[pyo3(name = "remove_events")]
    #[pyo3(signature = (xs, ys, ts, ps, to_remove, add_noise=0))]
    pub fn remove_events_py(
        py: Python<'_>,
        xs: PyReadonlyArray1<i64>,
        ys: PyReadonlyArray1<i64>,
        ts: PyReadonlyArray1<f64>,
        ps: PyReadonlyArray1<i64>,
        to_remove: usize,
        add_noise: usize,
    ) -> PyResult<PyObject> {
        use ndarray::Array1;
        use numpy::IntoPyArray;
        use pyo3::types::PyTuple;
        use rand::prelude::*;

        let xs_array = xs.as_array();
        let ys_array = ys.as_array();
        let ts_array = ts.as_array();
        let ps_array = ps.as_array();

        let n = xs_array.len();

        if to_remove >= n {
            // Return empty arrays
            let empty_xs = Array1::<i64>::zeros(0);
            let empty_ys = Array1::<i64>::zeros(0);
            let empty_ts = Array1::<f64>::zeros(0);
            let empty_ps = Array1::<i64>::zeros(0);

            // Convert arrays to Python objects
            let xs_py: PyObject = empty_xs.into_pyarray(py).into();
            let ys_py: PyObject = empty_ys.into_pyarray(py).into();
            let ts_py: PyObject = empty_ts.into_pyarray(py).into();
            let ps_py: PyObject = empty_ps.into_pyarray(py).into();

            // Create result tuple
            let tuple = PyTuple::new(py, [xs_py, ys_py, ts_py, ps_py])?;
            return Ok(tuple.into());
        }

        let to_select = n - to_remove;

        // Generate random indices without replacement
        let mut rng = thread_rng();
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);
        indices.truncate(to_select);
        indices.sort();

        // Extract selected events
        let selected_xs = indices
            .iter()
            .map(|&i| xs_array[i])
            .collect::<Array1<i64>>();
        let selected_ys = indices
            .iter()
            .map(|&i| ys_array[i])
            .collect::<Array1<i64>>();
        let selected_ts = indices
            .iter()
            .map(|&i| ts_array[i])
            .collect::<Array1<f64>>();
        let selected_ps = indices
            .iter()
            .map(|&i| ps_array[i])
            .collect::<Array1<i64>>();

        if add_noise == 0 {
            // Convert arrays to Python objects
            let xs_py: PyObject = selected_xs.into_pyarray(py).into();
            let ys_py: PyObject = selected_ys.into_pyarray(py).into();
            let ts_py: PyObject = selected_ts.into_pyarray(py).into();
            let ps_py: PyObject = selected_ps.into_pyarray(py).into();

            // Create result tuple
            let tuple = PyTuple::new(py, [xs_py, ys_py, ts_py, ps_py])?;
            Ok(tuple.into())
        } else {
            // Generate random events for noise
            let max_x = xs_array.fold(0, |acc, &x| acc.max(x));
            let max_y = ys_array.fold(0, |acc, &y| acc.max(y));

            let mut xs_noise = Vec::with_capacity(add_noise);
            let mut ys_noise = Vec::with_capacity(add_noise);
            let mut ts_noise = Vec::with_capacity(add_noise);
            let mut ps_noise = Vec::with_capacity(add_noise);

            let min_ts = ts_array.fold(f64::INFINITY, |acc, &t| acc.min(t));
            let max_ts = ts_array.fold(f64::NEG_INFINITY, |acc, &t| acc.max(t));

            for _ in 0..add_noise {
                xs_noise.push(rng.gen_range(0..=max_x));
                ys_noise.push(rng.gen_range(0..=max_y));
                ts_noise.push(rng.gen_range(min_ts..=max_ts));
                ps_noise.push(if rng.gen_bool(0.5) { 1 } else { -1 });
            }

            // Merge selected events and noise
            let mut all_xs = Vec::with_capacity(selected_xs.len() + add_noise);
            let mut all_ys = Vec::with_capacity(selected_ys.len() + add_noise);
            let mut all_ts: Vec<f64> = Vec::with_capacity(selected_ts.len() + add_noise);
            let mut all_ps = Vec::with_capacity(selected_ps.len() + add_noise);

            all_xs.extend(selected_xs.iter());
            all_xs.extend(xs_noise.iter());

            all_ys.extend(selected_ys.iter());
            all_ys.extend(ys_noise.iter());

            all_ts.extend(selected_ts.iter());
            all_ts.extend(ts_noise.iter());

            all_ps.extend(selected_ps.iter());
            all_ps.extend(ps_noise.iter());

            let merged_xs = Array1::from(all_xs);
            let merged_ys = Array1::from(all_ys);
            let merged_ts = Array1::from(all_ts);
            let merged_ps = Array1::from(all_ps);

            // Sort by timestamp
            let mut indices: Vec<usize> = (0..merged_ts.len()).collect();
            indices.sort_by(|&i, &j| merged_ts[i].partial_cmp(&merged_ts[j]).unwrap());

            let sorted_xs = indices
                .iter()
                .map(|&i| merged_xs[i])
                .collect::<Array1<i64>>();
            let sorted_ys = indices
                .iter()
                .map(|&i| merged_ys[i])
                .collect::<Array1<i64>>();
            let sorted_ts = indices
                .iter()
                .map(|&i| merged_ts[i])
                .collect::<Array1<f64>>();
            let sorted_ps = indices
                .iter()
                .map(|&i| merged_ps[i])
                .collect::<Array1<i64>>();

            // Convert arrays to Python objects
            let xs_py: PyObject = sorted_xs.into_pyarray(py).into();
            let ys_py: PyObject = sorted_ys.into_pyarray(py).into();
            let ts_py: PyObject = sorted_ts.into_pyarray(py).into();
            let ps_py: PyObject = sorted_ps.into_pyarray(py).into();

            // Create result tuple
            let tuple = PyTuple::new(py, [xs_py, ys_py, ts_py, ps_py])?;
            Ok(tuple.into())
        }
    }
}
