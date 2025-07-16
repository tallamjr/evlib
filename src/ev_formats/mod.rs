// Data formats module
// Handles reading and writing events from various file formats

use crate::ev_core::{Event, Events};
use hdf5::File as H5File;
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

// Polarity encoding handler module
pub mod polarity_handler;
pub use polarity_handler::{
    PolarityConfig, PolarityEncoding, PolarityError, PolarityHandler, PolarityStats,
};

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
    /// Polarity filter (1 for positive, -1 for negative, None for both)
    pub polarity: Option<i8>,
    /// Sort events by timestamp after loading
    pub sort: bool,
    /// Chunk size for memory management (not used for filtering, but affects performance)
    pub chunk_size: Option<usize>,
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
    pub fn with_polarity(mut self, polarity: Option<i8>) -> Self {
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
            if event.polarity != polarity {
                return false;
            }
        }

        true
    }
}

/// Load events from an HDF5 file
///
/// Expects a dataset "events" or similar with fields t, x, y, polarity.
///
/// # Arguments
/// * `path` - Path to the HDF5 file
/// * `dataset_name` - Name of the dataset containing events (default: "events")
pub fn load_events_from_hdf5(path: &str, dataset_name: Option<&str>) -> hdf5::Result<Events> {
    let file = H5File::open(path)?;
    let dataset_name = dataset_name.unwrap_or("events");

    // Check if we have a compound dataset
    if let Ok(dataset) = file.dataset(dataset_name) {
        // Try reading as an array of tuples (t,x,y,p)
        let data: Vec<(f64, u16, u16, i8)> = dataset.read_raw()?.to_vec();

        // Convert into our Event struct
        let events: Events = data
            .into_iter()
            .map(|(t, x, y, p)| Event {
                t,
                x,
                y,
                polarity: p,
            })
            .collect();

        return Ok(events);
    }

    // Fallback: check for separate datasets
    let field_names = [
        ("t", "x", "y", "p"),
        ("timestamps", "x_pos", "y_pos", "polarity"),
        ("ts", "xs", "ys", "ps"),
    ];

    for (t_name, x_name, y_name, p_name) in field_names {
        if let (Ok(t_dataset), Ok(x_dataset), Ok(y_dataset), Ok(p_dataset)) = (
            file.dataset(t_name),
            file.dataset(x_name),
            file.dataset(y_name),
            file.dataset(p_name),
        ) {
            let t_arr: Vec<f64> = t_dataset.read_raw()?.to_vec();
            let x_arr: Vec<u16> = x_dataset.read_raw()?.to_vec();
            let y_arr: Vec<u16> = y_dataset.read_raw()?.to_vec();
            let p_arr: Vec<i8> = p_dataset.read_raw()?.to_vec();

            let n = t_arr.len();
            let mut events = Vec::with_capacity(n);

            for i in 0..n {
                events.push(Event {
                    t: t_arr[i],
                    x: x_arr[i],
                    y: y_arr[i],
                    polarity: p_arr[i],
                });
            }

            return Ok(events);
        }
    }

    // Check for datasets inside an "events" group (for files saved by our save function)
    if let Ok(events_group) = file.group("events") {
        for (t_name, x_name, y_name, p_name) in field_names {
            if let (Ok(t_dataset), Ok(x_dataset), Ok(y_dataset), Ok(p_dataset)) = (
                events_group.dataset(t_name),
                events_group.dataset(x_name),
                events_group.dataset(y_name),
                events_group.dataset(p_name),
            ) {
                let t_arr: Vec<f64> = t_dataset.read_raw()?.to_vec();
                let x_arr: Vec<u16> = x_dataset.read_raw()?.to_vec();
                let y_arr: Vec<u16> = y_dataset.read_raw()?.to_vec();
                let p_arr: Vec<i8> = p_dataset.read_raw()?.to_vec();

                let n = t_arr.len();
                let mut events = Vec::with_capacity(n);

                for i in 0..n {
                    events.push(Event {
                        t: t_arr[i],
                        x: x_arr[i],
                        y: y_arr[i],
                        polarity: p_arr[i],
                    });
                }

                return Ok(events);
            }
        }
    }

    // If we get here, we couldn't find the data in any expected format
    Err(hdf5::Error::Internal(format!(
        "Could not find event data in HDF5 file {}",
        path
    )))
}

/// Load events from a plain text file (one event per line)
///
/// Format is expected as: "t x y p" (floating timestamp, int x, int y, int polarity)
/// Each line contains space-separated values for one event.
/// Supports filtering by time, spatial bounds, and polarity.
///
/// # Arguments
/// * `path` - Path to the text file
/// * `config` - Configuration with filtering options
pub fn load_events_from_text(path: &str, config: &LoadConfig) -> IoResult<Events> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut events = Events::new();

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
                    "Line {}: Expected at least {} values for column mapping, got {}",
                    line_num + 1,
                    max_col,
                    parts.len()
                ),
            ));
        }

        // Parse values using custom column mapping
        let t = parts[t_col].parse::<f64>().map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Line {}: Invalid timestamp '{}': {}",
                    line_num + 1,
                    parts[t_col],
                    e
                ),
            )
        })?;
        let x = parts[x_col].parse::<u16>().map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Line {}: Invalid x coordinate '{}': {}",
                    line_num + 1,
                    parts[x_col],
                    e
                ),
            )
        })?;
        let y = parts[y_col].parse::<u16>().map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Line {}: Invalid y coordinate '{}': {}",
                    line_num + 1,
                    parts[y_col],
                    e
                ),
            )
        })?;
        let polarity = parts[p_col].parse::<i8>().map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Line {}: Invalid polarity '{}': {}",
                    line_num + 1,
                    parts[p_col],
                    e
                ),
            )
        })?;

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

    Ok(events)
}

/// Memory-map a binary event file and return events
///
/// This is useful for large files that would not fit in memory.
/// The binary format should be a sequence of Event structures.
///
/// # Arguments
/// * `path` - Path to the binary file
// Binary format (mmap_events) removed due to safety and reliability issues:
// 1. Unsafe memory operations with no validation
// 2. Assumes files contain raw Event structs (almost never true)
// 3. Produces misleading results with text files
// 4. No error handling for malformed binary data
//
// All file types now use the safe, reliable text parser

// Backward compatibility functions (maintain old API)

/// Load events from a text file (backward compatibility)
pub fn load_events_from_text_simple(path: &str) -> IoResult<Events> {
    load_events_from_text(path, &LoadConfig::new())
}

/// Load events from HDF5 with filtering support
pub fn load_events_from_hdf5_filtered(
    path: &str,
    dataset_name: Option<&str>,
    config: &LoadConfig,
) -> hdf5::Result<Events> {
    let mut events = load_events_from_hdf5(path, dataset_name)?;

    // Apply filters to the loaded events
    events.retain(|event| config.passes_filters(event));

    // Sort if requested
    if config.sort {
        events.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));
    }

    Ok(events)
}

/// Generic load function with automatic format detection and filtering
pub fn load_events_with_config(
    path: &str,
    config: &LoadConfig,
) -> Result<Events, Box<dyn std::error::Error>> {
    // Use format detector to determine the file format
    let detection_result = format_detector::detect_event_format(path)?;

    match detection_result.format {
        EventFormat::HDF5 => Ok(load_events_from_hdf5_filtered(path, None, config)?),
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
                polarity_encoding: config.polarity_encoding.clone(),
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
                polarity_encoding: config.polarity_encoding.clone(),
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
                validate_coordinates: true,
                skip_invalid_events: false,
                max_events: None,
                sensor_resolution: detection_result.metadata.sensor_resolution,
                chunk_size: 1_000_000,
                polarity_encoding: config.polarity_encoding.clone(),
            };
            let reader = Evt3Reader::with_config(evt3_config);
            let events = reader
                .read_with_config(path, config)
                .map_err(|e| -> Box<dyn std::error::Error> { Box::new(e) })?;
            Ok(events)
        }
        EventFormat::Binary => {
            // Binary format is not supported for safety reasons
            Err(format!(
                "Binary format is not supported for safety reasons. Detected format: {} (confidence: {:.2})",
                detection_result.format,
                detection_result.confidence
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
                    polarity: p,
                }))
            }
            Err(e) => Some(Err(e)),
        }
    }
}

// Window-based event iterator that returns chunks of events based on time windows
pub struct TimeWindowIter<'a> {
    events: &'a Events,
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
    pub fn new(events: &'a Events, window_duration: f64) -> Self {
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
#[cfg(feature = "python")]
pub mod python {
    use super::*;
    use numpy::{PyArray1, PyReadonlyArray1};
    use pyo3::prelude::*;
    use std::io::Write;

    /// Load events from a file (text, HDF5, or binary)
    ///
    /// Automatically detects the format based on file extension
    #[pyfunction]
    #[pyo3(
        signature = (
            path,
            t_start=None,
            t_end=None,
            min_x=None,
            max_x=None,
            min_y=None,
            max_y=None,
            polarity=None,
            sort=false,
            chunk_size=None,
            x_col=None,
            y_col=None,
            t_col=None,
            p_col=None,
            header_lines=0
        ),
        name = "load_events"
    )]
    pub fn load_events_py(
        py: Python<'_>,
        path: &str,
        t_start: Option<f64>,
        t_end: Option<f64>,
        min_x: Option<u16>,
        max_x: Option<u16>,
        min_y: Option<u16>,
        max_y: Option<u16>,
        polarity: Option<i8>,
        sort: bool,
        chunk_size: Option<usize>,
        x_col: Option<usize>,
        y_col: Option<usize>,
        t_col: Option<usize>,
        p_col: Option<usize>,
        header_lines: usize,
    ) -> PyResult<(PyObject, PyObject, PyObject, PyObject)> {
        // Create LoadConfig from parameters
        let config = LoadConfig {
            t_start,
            t_end,
            min_x,
            max_x,
            min_y,
            max_y,
            polarity,
            sort,
            chunk_size,
            x_col,
            y_col,
            t_col,
            p_col,
            header_lines,
            polarity_encoding: None,
        };

        // Load events with filtering
        let events = load_events_with_config(path, &config).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error loading events: {}", e))
        })?;

        // Separate event fields into arrays
        let n = events.len();

        let mut timestamps = Vec::with_capacity(n);
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        let mut polarities = Vec::with_capacity(n);

        for ev in events {
            timestamps.push(ev.t);
            xs.push(ev.x as i64);
            ys.push(ev.y as i64);
            polarities.push(ev.polarity as i64);
        }

        // Convert to numpy arrays
        let ts_array = PyArray1::from_vec(py, timestamps);
        let xs_array = PyArray1::from_vec(py, xs);
        let ys_array = PyArray1::from_vec(py, ys);
        let ps_array = PyArray1::from_vec(py, polarities);

        Ok((
            xs_array.to_object(py),
            ys_array.to_object(py),
            ts_array.to_object(py),
            ps_array.to_object(py),
        ))
    }

    /// Load events with filtering support
    /// Automatically detects the format based on file extension and applies filters
    #[pyfunction]
    #[pyo3(
        signature = (
            path,
            t_start=None,
            t_end=None,
            min_x=None,
            max_x=None,
            min_y=None,
            max_y=None,
            polarity=None,
            sort=false,
            chunk_size=None,
            x_col=None,
            y_col=None,
            t_col=None,
            p_col=None,
            header_lines=0
        ),
        name = "load_events_filtered"
    )]
    pub fn load_events_filtered_py(
        py: Python<'_>,
        path: &str,
        t_start: Option<f64>,
        t_end: Option<f64>,
        min_x: Option<u16>,
        max_x: Option<u16>,
        min_y: Option<u16>,
        max_y: Option<u16>,
        polarity: Option<i8>,
        sort: bool,
        chunk_size: Option<usize>,
        x_col: Option<usize>,
        y_col: Option<usize>,
        t_col: Option<usize>,
        p_col: Option<usize>,
        header_lines: usize,
    ) -> PyResult<(PyObject, PyObject, PyObject, PyObject)> {
        // Create LoadConfig from parameters
        let config = LoadConfig {
            t_start,
            t_end,
            min_x,
            max_x,
            min_y,
            max_y,
            polarity,
            sort,
            chunk_size,
            x_col,
            y_col,
            t_col,
            p_col,
            header_lines,
            polarity_encoding: None,
        };

        // Load events with filtering
        let events = load_events_with_config(path, &config).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Error loading events: {}", e))
        })?;

        // Separate event fields into arrays
        let n = events.len();

        let mut timestamps = Vec::with_capacity(n);
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        let mut polarities = Vec::with_capacity(n);

        for ev in events {
            timestamps.push(ev.t);
            xs.push(ev.x as i64);
            ys.push(ev.y as i64);
            polarities.push(ev.polarity as i64);
        }

        // Convert to numpy arrays
        let ts_array = PyArray1::from_vec(py, timestamps);
        let xs_array = PyArray1::from_vec(py, xs);
        let ys_array = PyArray1::from_vec(py, ys);
        let ps_array = PyArray1::from_vec(py, polarities);

        Ok((
            xs_array.to_object(py),
            ys_array.to_object(py),
            ts_array.to_object(py),
            ps_array.to_object(py),
        ))
    }

    /// Save events to an HDF5 file
    #[pyfunction]
    #[pyo3(name = "save_events_to_hdf5")]
    pub fn save_events_to_hdf5_py(
        xs: PyReadonlyArray1<i64>,
        ys: PyReadonlyArray1<i64>,
        ts: PyReadonlyArray1<f64>,
        ps: PyReadonlyArray1<i64>,
        path: &str,
    ) -> PyResult<()> {
        // Validate array lengths
        let n = ts.len();
        if xs.len() != n || ys.len() != n || ps.len() != n {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Arrays must have the same length",
            ));
        }

        // Create HDF5 file
        let file = H5File::create(path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                "Failed to create HDF5 file: {}",
                e
            ))
        })?;

        // Create a group to store the data
        let group = file.create_group("events").map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to create group: {}", e))
        })?;

        // Convert arrays to Rust vectors
        let xs_vec: Vec<u16> = xs.as_array().iter().map(|&x| x as u16).collect();
        let ys_vec: Vec<u16> = ys.as_array().iter().map(|&y| y as u16).collect();
        let ts_vec: Vec<f64> = ts.as_slice().unwrap().to_vec();
        let ps_vec: Vec<i8> = ps.as_array().iter().map(|&p| p as i8).collect();

        // Create datasets for each component
        let xs_shape = [n];
        let xs_dataset = group
            .new_dataset::<u16>()
            .shape(xs_shape)
            .create("xs")
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                    "Failed to create xs dataset: {}",
                    e
                ))
            })?;
        xs_dataset.write(&xs_vec).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to write xs data: {}", e))
        })?;

        let ys_dataset = group
            .new_dataset::<u16>()
            .shape(xs_shape)
            .create("ys")
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                    "Failed to create ys dataset: {}",
                    e
                ))
            })?;
        ys_dataset.write(&ys_vec).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to write ys data: {}", e))
        })?;

        let ts_dataset = group
            .new_dataset::<f64>()
            .shape(xs_shape)
            .create("ts")
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                    "Failed to create ts dataset: {}",
                    e
                ))
            })?;
        ts_dataset.write(&ts_vec).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to write ts data: {}", e))
        })?;

        let ps_dataset = group
            .new_dataset::<i8>()
            .shape(xs_shape)
            .create("ps")
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                    "Failed to create ps dataset: {}",
                    e
                ))
            })?;
        ps_dataset.write(&ps_vec).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to write ps data: {}", e))
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
        let n = ts.len();
        if xs.len() != n || ys.len() != n || ps.len() != n {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Arrays must have the same length",
            ));
        }

        // Create output file
        let mut file = std::fs::File::create(path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to create file: {}", e))
        })?;

        // Write header
        file.write_all(b"# timestamp x y polarity\n").map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to write header: {}", e))
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
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to write line: {}", e))
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
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Format detection failed: {}", e))
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

    // Python wrapper for EventFileIterator
    #[pyclass]
    pub struct PyEventFileIterator {
        path: String,
        reader: Option<EventFileIterator>,
    }

    #[pymethods]
    impl PyEventFileIterator {
        #[new]
        fn new(path: String) -> Self {
            PyEventFileIterator { path, reader: None }
        }

        fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
            slf
        }

        fn __next__(&mut self, _py: Python<'_>) -> PyResult<Option<(f64, i64, i64, i64)>> {
            // Initialize reader if needed
            if self.reader.is_none() {
                self.reader = Some(EventFileIterator::new(&self.path).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                        "Failed to open file: {}",
                        e
                    ))
                })?);
            }

            // Read next event
            if let Some(ref mut reader) = self.reader {
                match reader.next() {
                    Some(Ok(event)) => Ok(Some((
                        event.t,
                        event.x as i64,
                        event.y as i64,
                        event.polarity as i64,
                    ))),
                    Some(Err(e)) => Err(PyErr::new::<pyo3::exceptions::PyIOError, _>(format!(
                        "Error reading event: {}",
                        e
                    ))),
                    None => Ok(None),
                }
            } else {
                // This shouldn't happen, but just in case
                Ok(None)
            }
        }
    }

    // Python wrapper for TimeWindowIter
    #[pyclass]
    pub struct PyTimeWindowIter {
        events_xs: Vec<i64>,
        events_ys: Vec<i64>,
        events_ts: Vec<f64>,
        events_ps: Vec<i64>,
        window_duration: f64,
        current_idx: usize,
        start_time: f64,
        end_time: f64,
    }

    #[pymethods]
    impl PyTimeWindowIter {
        #[new]
        fn new(
            xs: PyReadonlyArray1<i64>,
            ys: PyReadonlyArray1<i64>,
            ts: PyReadonlyArray1<f64>,
            ps: PyReadonlyArray1<i64>,
            window_duration: f64,
        ) -> PyResult<Self> {
            // Convert to Rust vectors
            let xs_vec = xs.as_slice().unwrap().to_vec();
            let ys_vec = ys.as_slice().unwrap().to_vec();
            let ts_vec = ts.as_slice().unwrap().to_vec();
            let ps_vec = ps.as_slice().unwrap().to_vec();

            // Validate
            let n = ts_vec.len();
            if xs_vec.len() != n || ys_vec.len() != n || ps_vec.len() != n {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Arrays must have the same length",
                ));
            }

            let start_time = if !ts_vec.is_empty() { ts_vec[0] } else { 0.0 };
            let end_time = start_time + window_duration;

            Ok(PyTimeWindowIter {
                events_xs: xs_vec,
                events_ys: ys_vec,
                events_ts: ts_vec,
                events_ps: ps_vec,
                window_duration,
                current_idx: 0,
                start_time,
                end_time,
            })
        }

        fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
            slf
        }

        fn __next__(
            &mut self,
            py: Python<'_>,
        ) -> PyResult<Option<(PyObject, PyObject, PyObject, PyObject)>> {
            if self.current_idx >= self.events_ts.len() {
                return Ok(None);
            }

            let mut xs_window = Vec::new();
            let mut ys_window = Vec::new();
            let mut ts_window = Vec::new();
            let mut ps_window = Vec::new();
            let mut idx = self.current_idx;

            // Collect events within current time window
            while idx < self.events_ts.len() && self.events_ts[idx] < self.end_time {
                xs_window.push(self.events_xs[idx]);
                ys_window.push(self.events_ys[idx]);
                ts_window.push(self.events_ts[idx]);
                ps_window.push(self.events_ps[idx]);
                idx += 1;
            }

            // Update state for next iteration
            self.current_idx = idx;
            self.start_time = self.end_time;
            self.end_time += self.window_duration;

            // If no events in this window, move to the next one
            if xs_window.is_empty() {
                return self.__next__(py);
            }

            // Convert to numpy arrays
            let xs_array = PyArray1::from_vec(py, xs_window);
            let ys_array = PyArray1::from_vec(py, ys_window);
            let ts_array = PyArray1::from_vec(py, ts_window);
            let ps_array = PyArray1::from_vec(py, ps_window);

            Ok(Some((
                xs_array.to_object(py),
                ys_array.to_object(py),
                ts_array.to_object(py),
                ps_array.to_object(py),
            )))
        }
    }
}
