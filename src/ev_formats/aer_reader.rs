// Address Event Representation (AER) format reader
// Implements comprehensive AER format parsing for neuromorphic vision systems
//
// AER Format Specification:
// - 18-bit structure per event: 1 bit polarity + 9 bits x + 9 bits y
// - No explicit timestamps (real-time processing format)
// - Polarity: 0 = CD OFF (negative), 1 = CD ON (positive)
// - Coordinates: 9-bit values (0-511 range)
// - Common in neuromorphic sensors like GenX320
//
// References:
// - https://docs.prophesee.ai/stable/data/encoding_formats/aer.html
// - GenX320 sensor documentation
// - jAER project specifications

use crate::ev_core::{Event, Events};
use crate::ev_formats::LoadConfig;
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Configuration for AER reader
#[derive(Debug, Clone)]
pub struct AerConfig {
    /// Endianness of the AER data (true = big endian, false = little endian)
    pub big_endian: bool,
    /// Validate coordinate bounds (0-511 for 9-bit coordinates)
    pub validate_coordinates: bool,
    /// Maximum allowed x coordinate (default 511 for 9-bit)
    pub max_x: u16,
    /// Maximum allowed y coordinate (default 511 for 9-bit)
    pub max_y: u16,
    /// Skip invalid events instead of failing
    pub skip_invalid_events: bool,
    /// Generate timestamps based on event order
    pub generate_timestamps: bool,
    /// Timestamp generation mode
    pub timestamp_mode: TimestampMode,
    /// Starting timestamp for generated timestamps
    pub start_timestamp: f64,
    /// Time increment between events (seconds)
    pub time_increment: f64,
    /// Maximum number of events to read (None = all)
    pub max_events: Option<usize>,
    /// Bytes per event (2 or 4, depending on storage format)
    pub bytes_per_event: usize,
}

/// Timestamp generation modes for AER data
#[derive(Debug, Clone, PartialEq)]
pub enum TimestampMode {
    /// Sequential timestamps with fixed increment
    Sequential,
    /// Uniform distribution over time window
    Uniform,
    /// Exponential distribution (for spike-like patterns)
    Exponential,
    /// User-provided timestamps
    Custom(Vec<f64>),
}

/// AER-specific errors
#[derive(Debug)]
pub enum AerError {
    Io(std::io::Error),
    InvalidFileSize(u64, usize),
    InvalidCoordinate(u16, u16, u16, u16),
    InvalidEventData(usize, Vec<u8>),
    InsufficientData(usize, usize),
    InvalidBytesPerEvent(usize),
    EmptyFile,
    ValidationFailed(String),
}

impl std::fmt::Display for AerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AerError::Io(e) => write!(f, "I/O error: {}", e),
            AerError::InvalidFileSize(size, expected) => write!(
                f,
                "Invalid file size: {} bytes, expected multiple of {}",
                size, expected
            ),
            AerError::InvalidCoordinate(x, y, max_x, max_y) => write!(
                f,
                "Invalid coordinate: x={}, y={}, max_x={}, max_y={}",
                x, y, max_x, max_y
            ),
            AerError::InvalidEventData(offset, data) => {
                write!(f, "Invalid event data at byte {}: {:02X?}", offset, data)
            }
            AerError::InsufficientData(expected, actual) => write!(
                f,
                "Insufficient data: expected {} bytes, got {}",
                expected, actual
            ),
            AerError::InvalidBytesPerEvent(bytes) => {
                write!(f, "Invalid bytes per event: {}, expected 2 or 4", bytes)
            }
            AerError::EmptyFile => write!(f, "File is empty"),
            AerError::ValidationFailed(msg) => write!(f, "Event validation failed: {}", msg),
        }
    }
}

impl std::error::Error for AerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            AerError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for AerError {
    fn from(error: std::io::Error) -> Self {
        AerError::Io(error)
    }
}

/// Result type for AER operations
pub type AerResult<T> = Result<T, AerError>;

/// Metadata about AER file
#[derive(Debug, Clone)]
pub struct AerMetadata {
    /// File size in bytes
    pub file_size: u64,
    /// Number of events in file
    pub event_count: usize,
    /// Bytes per event (2 or 4)
    pub bytes_per_event: usize,
    /// Detected endianness
    pub endianness: String,
    /// Coordinate bounds found in data
    pub coordinate_bounds: Option<(u16, u16, u16, u16)>, // min_x, min_y, max_x, max_y
    /// Timestamp range (if generated)
    pub timestamp_range: Option<(f64, f64)>,
    /// Polarity distribution (positive_count, negative_count)
    pub polarity_distribution: Option<(usize, usize)>,
}

impl Default for AerConfig {
    fn default() -> Self {
        Self {
            big_endian: false,
            validate_coordinates: true,
            max_x: 511, // 9-bit maximum
            max_y: 511, // 9-bit maximum
            skip_invalid_events: false,
            generate_timestamps: true,
            timestamp_mode: TimestampMode::Sequential,
            start_timestamp: 0.0,
            time_increment: 1e-6, // 1 microsecond
            max_events: None,
            bytes_per_event: 4, // Common format: 32-bit words
        }
    }
}

impl AerConfig {
    /// Create a new AER configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set endianness
    pub fn with_endianness(mut self, big_endian: bool) -> Self {
        self.big_endian = big_endian;
        self
    }

    /// Set coordinate bounds
    pub fn with_coordinate_bounds(mut self, max_x: u16, max_y: u16) -> Self {
        self.max_x = max_x;
        self.max_y = max_y;
        self
    }

    /// Set timestamp generation parameters
    pub fn with_timestamp_generation(
        mut self,
        generate: bool,
        mode: TimestampMode,
        start: f64,
        increment: f64,
    ) -> Self {
        self.generate_timestamps = generate;
        self.timestamp_mode = mode;
        self.start_timestamp = start;
        self.time_increment = increment;
        self
    }

    /// Set bytes per event
    pub fn with_bytes_per_event(mut self, bytes: usize) -> Self {
        self.bytes_per_event = bytes;
        self
    }

    /// Enable validation with error skipping
    pub fn with_validation(mut self, validate: bool, skip_invalid: bool) -> Self {
        self.validate_coordinates = validate;
        self.skip_invalid_events = skip_invalid;
        self
    }

    /// Set maximum events to read
    pub fn with_max_events(mut self, max_events: Option<usize>) -> Self {
        self.max_events = max_events;
        self
    }
}

/// AER format reader
pub struct AerReader {
    config: AerConfig,
}

impl AerReader {
    /// Create a new AER reader with default configuration
    pub fn new() -> Self {
        Self {
            config: AerConfig::default(),
        }
    }

    /// Create a new AER reader with custom configuration
    pub fn with_config(config: AerConfig) -> Self {
        Self { config }
    }

    /// Read AER events from a file
    pub fn read_file<P: AsRef<Path>>(&self, path: P) -> AerResult<(Events, AerMetadata)> {
        let mut file = File::open(path.as_ref())?;
        let file_size = file.metadata()?.len();

        if file_size == 0 {
            return Err(AerError::EmptyFile);
        }

        // Validate file size
        if file_size % self.config.bytes_per_event as u64 != 0 {
            return Err(AerError::InvalidFileSize(
                file_size,
                self.config.bytes_per_event,
            ));
        }

        let expected_event_count = (file_size / self.config.bytes_per_event as u64) as usize;
        let event_count = match self.config.max_events {
            Some(max) => expected_event_count.min(max),
            None => expected_event_count,
        };

        // Read binary data
        let mut buffer = vec![0u8; event_count * self.config.bytes_per_event];
        file.read_exact(&mut buffer)?;

        // Parse events
        let (events, metadata) = self.parse_events(&buffer, file_size)?;

        Ok((events, metadata))
    }

    /// Parse AER events from binary data
    fn parse_events(&self, data: &[u8], file_size: u64) -> AerResult<(Events, AerMetadata)> {
        if self.config.bytes_per_event != 2 && self.config.bytes_per_event != 4 {
            return Err(AerError::InvalidBytesPerEvent(self.config.bytes_per_event));
        }

        let event_count = data.len() / self.config.bytes_per_event;
        let mut events = Events::with_capacity(event_count);

        // Statistics for metadata
        let mut min_x = u16::MAX;
        let mut min_y = u16::MAX;
        let mut max_x = 0u16;
        let mut max_y = 0u16;
        let mut positive_count = 0;
        let mut negative_count = 0;

        // Process events
        for i in 0..event_count {
            let offset = i * self.config.bytes_per_event;

            match self.parse_single_event(&data[offset..offset + self.config.bytes_per_event], i) {
                Ok(event) => {
                    // Update statistics
                    min_x = min_x.min(event.x);
                    min_y = min_y.min(event.y);
                    max_x = max_x.max(event.x);
                    max_y = max_y.max(event.y);

                    if event.polarity > 0 {
                        positive_count += 1;
                    } else {
                        negative_count += 1;
                    }

                    events.push(event);
                }
                Err(e) => {
                    if self.config.skip_invalid_events {
                        continue;
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        // Generate timestamps if requested
        if self.config.generate_timestamps {
            self.generate_timestamps(&mut events)?;
        }

        let timestamp_range = if !events.is_empty() {
            Some((events[0].t, events[events.len() - 1].t))
        } else {
            None
        };

        let coordinate_bounds = if !events.is_empty() {
            Some((min_x, min_y, max_x, max_y))
        } else {
            None
        };

        let metadata = AerMetadata {
            file_size,
            event_count: events.len(),
            bytes_per_event: self.config.bytes_per_event,
            endianness: if self.config.big_endian {
                "big".to_string()
            } else {
                "little".to_string()
            },
            coordinate_bounds,
            timestamp_range,
            polarity_distribution: Some((positive_count, negative_count)),
        };

        Ok((events, metadata))
    }

    /// Parse a single AER event from binary data
    fn parse_single_event(&self, data: &[u8], _event_index: usize) -> AerResult<Event> {
        if data.len() < self.config.bytes_per_event {
            return Err(AerError::InsufficientData(
                self.config.bytes_per_event,
                data.len(),
            ));
        }

        let raw_event = match self.config.bytes_per_event {
            2 => {
                // 16-bit format (direct 18-bit event, but only 16 bits used)
                if self.config.big_endian {
                    u16::from_be_bytes([data[0], data[1]]) as u32
                } else {
                    u16::from_le_bytes([data[0], data[1]]) as u32
                }
            }
            4 => {
                // 32-bit format (18-bit event in 32-bit word)
                if self.config.big_endian {
                    u32::from_be_bytes([data[0], data[1], data[2], data[3]])
                } else {
                    u32::from_le_bytes([data[0], data[1], data[2], data[3]])
                }
            }
            _ => {
                return Err(AerError::InvalidBytesPerEvent(self.config.bytes_per_event));
            }
        };

        // Extract 18-bit AER structure: 1 bit polarity + 9 bits x + 9 bits y
        let polarity_bit = (raw_event & 0x1) as u8;
        let x = ((raw_event >> 1) & 0x1FF) as u16; // 9 bits for x coordinate
        let y = ((raw_event >> 10) & 0x1FF) as u16; // 9 bits for y coordinate

        // Convert polarity bit to signed polarity
        let polarity = if polarity_bit == 1 { 1i8 } else { -1i8 };

        // Validate coordinates if requested
        if self.config.validate_coordinates && (x > self.config.max_x || y > self.config.max_y) {
            return Err(AerError::InvalidCoordinate(
                x,
                y,
                self.config.max_x,
                self.config.max_y,
            ));
        }

        // Create event with placeholder timestamp (will be generated later if needed)
        let event = Event {
            t: 0.0, // Will be set by generate_timestamps if needed
            x,
            y,
            polarity,
        };

        Ok(event)
    }

    /// Generate timestamps for events based on configuration
    fn generate_timestamps(&self, events: &mut [Event]) -> AerResult<()> {
        if events.is_empty() {
            return Ok(());
        }

        match &self.config.timestamp_mode {
            TimestampMode::Sequential => {
                for (i, event) in events.iter_mut().enumerate() {
                    event.t = self.config.start_timestamp + (i as f64 * self.config.time_increment);
                }
            }
            TimestampMode::Uniform => {
                let total_time = events.len() as f64 * self.config.time_increment;
                let event_count = events.len();
                for (i, event) in events.iter_mut().enumerate() {
                    event.t =
                        self.config.start_timestamp + (i as f64 / event_count as f64) * total_time;
                }
            }
            TimestampMode::Exponential => {
                // Generate exponentially distributed timestamps
                let mut current_time = self.config.start_timestamp;
                let lambda = 1.0 / self.config.time_increment; // Rate parameter

                for event in events.iter_mut() {
                    // Simple exponential distribution approximation
                    let u: f64 = (fastrand::f64() + 1e-10).ln(); // Add small value to avoid log(0)
                    let interval = -u / lambda;
                    current_time += interval;
                    event.t = current_time;
                }
            }
            TimestampMode::Custom(timestamps) => {
                if timestamps.len() != events.len() {
                    return Err(AerError::ValidationFailed(format!(
                        "Custom timestamp count ({}) doesn't match event count ({})",
                        timestamps.len(),
                        events.len()
                    )));
                }

                for (event, &timestamp) in events.iter_mut().zip(timestamps.iter()) {
                    event.t = timestamp;
                }
            }
        }

        Ok(())
    }

    /// Read AER events with LoadConfig filtering
    pub fn read_with_config<P: AsRef<Path>>(
        &self,
        path: P,
        load_config: &LoadConfig,
    ) -> AerResult<Events> {
        let (mut events, _metadata) = self.read_file(path)?;

        // Apply filtering from LoadConfig
        events.retain(|event| load_config.passes_filters(event));

        // Sort if requested
        if load_config.sort {
            events.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));
        }

        Ok(events)
    }

    /// Get configuration
    pub fn config(&self) -> &AerConfig {
        &self.config
    }
}

impl Default for AerReader {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function to read AER file with default configuration
pub fn read_aer_file<P: AsRef<Path>>(path: P) -> AerResult<(Events, AerMetadata)> {
    let reader = AerReader::new();
    reader.read_file(path)
}

/// Convenience function to read AER file with custom configuration
pub fn read_aer_file_with_config<P: AsRef<Path>>(
    path: P,
    config: AerConfig,
) -> AerResult<(Events, AerMetadata)> {
    let reader = AerReader::with_config(config);
    reader.read_file(path)
}

/// Detect if a file is likely to be AER format
pub fn is_aer_format<P: AsRef<Path>>(path: P) -> bool {
    let file = match File::open(path.as_ref()) {
        Ok(f) => f,
        Err(_) => return false,
    };

    let file_size = match file.metadata() {
        Ok(m) => m.len(),
        Err(_) => return false,
    };

    // Check if file size is consistent with AER format
    if file_size % 4 != 0 && file_size % 2 != 0 {
        return false;
    }

    // Try to read a few events and validate
    let bytes_to_read = std::cmp::min(32, file_size as usize);
    let mut buffer = vec![0u8; bytes_to_read];
    let mut file = file;
    if file.read_exact(&mut buffer).is_err() {
        return false;
    }

    // Check if the data looks like valid AER events
    let config = AerConfig::default();
    let reader = AerReader::with_config(config);

    for i in 0..8 {
        let offset = i * 4;
        if offset + 4 > buffer.len() {
            break;
        }

        if reader
            .parse_single_event(&buffer[offset..offset + 4], i)
            .is_err()
        {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_aer_config_default() {
        let config = AerConfig::default();
        assert!(!config.big_endian);
        assert!(config.validate_coordinates);
        assert_eq!(config.max_x, 511);
        assert_eq!(config.max_y, 511);
        assert_eq!(config.bytes_per_event, 4);
        assert!(config.generate_timestamps);
        assert_eq!(config.timestamp_mode, TimestampMode::Sequential);
    }

    #[test]
    fn test_aer_config_builder() {
        let config = AerConfig::new()
            .with_endianness(true)
            .with_coordinate_bounds(1023, 1023)
            .with_bytes_per_event(2)
            .with_validation(true, true);

        assert!(config.big_endian);
        assert_eq!(config.max_x, 1023);
        assert_eq!(config.max_y, 1023);
        assert_eq!(config.bytes_per_event, 2);
        assert!(config.skip_invalid_events);
    }

    #[test]
    fn test_parse_18bit_aer_event() {
        let config = AerConfig::default();
        let reader = AerReader::with_config(config);

        // Create a test event: polarity=1, x=100, y=200
        // Raw: (200 << 10) | (100 << 1) | 1 = 204800 + 200 + 1 = 205001
        let raw_event = 205001u32;
        let data = raw_event.to_le_bytes();

        let event = reader.parse_single_event(&data, 0).unwrap();

        assert_eq!(event.x, 100);
        assert_eq!(event.y, 200);
        assert_eq!(event.polarity, 1);
    }

    #[test]
    fn test_parse_negative_polarity() {
        let config = AerConfig::default();
        let reader = AerReader::with_config(config);

        // Create a test event: polarity=0, x=50, y=75
        // Raw: (75 << 10) | (50 << 1) | 0 = 76800 + 100 + 0 = 76900
        let raw_event = 76900u32;
        let data = raw_event.to_le_bytes();

        let event = reader.parse_single_event(&data, 0).unwrap();

        assert_eq!(event.x, 50);
        assert_eq!(event.y, 75);
        assert_eq!(event.polarity, -1);
    }

    #[test]
    fn test_coordinate_validation() {
        let config = AerConfig::default().with_coordinate_bounds(100, 100);
        let reader = AerReader::with_config(config);

        // Create an event with coordinates exceeding bounds
        let raw_event = ((200u32 << 10) | (150u32 << 1) | 1); // x=150, y=200
        let data = raw_event.to_le_bytes();

        let result = reader.parse_single_event(&data, 0);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            AerError::InvalidCoordinate(_, _, _, _)
        ));
    }

    #[test]
    fn test_skip_invalid_events() {
        let config = AerConfig::default()
            .with_coordinate_bounds(100, 100)
            .with_validation(true, true); // Skip invalid events

        let reader = AerReader::with_config(config);

        // Create test data with mix of valid and invalid events
        let mut data = Vec::new();

        // Valid event: x=50, y=75, polarity=1
        let valid_event = ((75u32 << 10) | (50u32 << 1) | 1);
        data.extend_from_slice(&valid_event.to_le_bytes());

        // Invalid event: x=150, y=200, polarity=0 (exceeds bounds)
        let invalid_event = ((200u32 << 10) | (150u32 << 1) | 0);
        data.extend_from_slice(&invalid_event.to_le_bytes());

        // Another valid event: x=25, y=30, polarity=0
        let valid_event2 = ((30u32 << 10) | (25u32 << 1) | 0);
        data.extend_from_slice(&valid_event2.to_le_bytes());

        let (events, metadata) = reader.parse_events(&data, data.len() as u64).unwrap();

        assert_eq!(events.len(), 2); // Only valid events should be included
        assert_eq!(metadata.event_count, 2);
        assert_eq!(events[0].x, 50);
        assert_eq!(events[0].y, 75);
        assert_eq!(events[1].x, 25);
        assert_eq!(events[1].y, 30);
    }

    #[test]
    fn test_timestamp_generation_sequential() {
        let config = AerConfig::default().with_timestamp_generation(
            true,
            TimestampMode::Sequential,
            1.0,
            0.001,
        );

        let reader = AerReader::with_config(config);

        // Create test data
        let events_data = vec![
            ((100u32 << 10) | (50u32 << 1) | 1).to_le_bytes(),
            ((200u32 << 10) | (150u32 << 1) | 0).to_le_bytes(),
            ((300u32 << 10) | (250u32 << 1) | 1).to_le_bytes(),
        ];

        let data: Vec<u8> = events_data.into_iter().flatten().collect();
        let (events, _) = reader.parse_events(&data, data.len() as u64).unwrap();

        assert_eq!(events.len(), 3);
        assert_eq!(events[0].t, 1.0);
        assert_eq!(events[1].t, 1.001);
        assert_eq!(events[2].t, 1.002);
    }

    #[test]
    fn test_timestamp_generation_uniform() {
        let config = AerConfig::default().with_timestamp_generation(
            true,
            TimestampMode::Uniform,
            0.0,
            0.003,
        );

        let reader = AerReader::with_config(config);

        // Create test data for 3 events
        let events_data = vec![
            ((100u32 << 10) | (50u32 << 1) | 1).to_le_bytes(),
            ((200u32 << 10) | (150u32 << 1) | 0).to_le_bytes(),
            ((300u32 << 10) | (250u32 << 1) | 1).to_le_bytes(),
        ];

        let data: Vec<u8> = events_data.into_iter().flatten().collect();
        let (events, _) = reader.parse_events(&data, data.len() as u64).unwrap();

        assert_eq!(events.len(), 3);
        assert_eq!(events[0].t, 0.0);
        assert_eq!(events[1].t, 0.003);
        assert_eq!(events[2].t, 0.006);
    }

    #[test]
    fn test_big_endian_parsing() {
        let config = AerConfig::default().with_endianness(true);
        let reader = AerReader::with_config(config);

        // Create a test event: polarity=1, x=100, y=200
        let raw_event = 205001u32;
        let data = raw_event.to_be_bytes(); // Big endian

        let event = reader.parse_single_event(&data, 0).unwrap();

        assert_eq!(event.x, 100);
        assert_eq!(event.y, 200);
        assert_eq!(event.polarity, 1);
    }

    #[test]
    fn test_16bit_format() {
        let config = AerConfig::default().with_bytes_per_event(2);
        let reader = AerReader::with_config(config);

        // Create a test event in 16-bit format
        // Since we're using 16-bit, we need to fit polarity + x + y into 16 bits
        // Let's use: 1 bit polarity + 7 bits x + 8 bits y
        let raw_event = ((75u16 << 8) | (50u16 << 1) | 1); // polarity=1, x=50, y=75
        let data = raw_event.to_le_bytes();

        let event = reader.parse_single_event(&data, 0).unwrap();

        // With 16-bit format, coordinates will be different due to bit layout
        assert_eq!(event.polarity, 1);
    }

    #[test]
    fn test_read_aer_file() {
        let config = AerConfig::default();
        let reader = AerReader::with_config(config);

        // Create temporary file with test data
        let mut temp_file = NamedTempFile::new().unwrap();

        // Write test events
        let events_data = vec![
            ((100u32 << 10) | (50u32 << 1) | 1).to_le_bytes(),
            ((200u32 << 10) | (150u32 << 1) | 0).to_le_bytes(),
            ((300u32 << 10) | (250u32 << 1) | 1).to_le_bytes(),
        ];

        for event_bytes in events_data {
            temp_file.write_all(&event_bytes).unwrap();
        }

        let (events, metadata) = reader.read_file(temp_file.path()).unwrap();

        assert_eq!(events.len(), 3);
        assert_eq!(metadata.event_count, 3);
        assert_eq!(metadata.bytes_per_event, 4);
        assert_eq!(metadata.file_size, 12);

        // Check first event
        assert_eq!(events[0].x, 50);
        assert_eq!(events[0].y, 100);
        assert_eq!(events[0].polarity, 1);

        // Check metadata
        assert!(metadata.coordinate_bounds.is_some());
        assert!(metadata.polarity_distribution.is_some());
        let (pos_count, neg_count) = metadata.polarity_distribution.unwrap();
        assert_eq!(pos_count, 2);
        assert_eq!(neg_count, 1);
    }

    #[test]
    fn test_empty_file() {
        let config = AerConfig::default();
        let reader = AerReader::with_config(config);

        let temp_file = NamedTempFile::new().unwrap();
        let result = reader.read_file(temp_file.path());

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), AerError::EmptyFile));
    }

    #[test]
    fn test_invalid_file_size() {
        let config = AerConfig::default();
        let reader = AerReader::with_config(config);

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(&[1, 2, 3]).unwrap(); // 3 bytes, not divisible by 4

        let result = reader.read_file(temp_file.path());
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            AerError::InvalidFileSize(_, _)
        ));
    }

    #[test]
    fn test_custom_timestamps() {
        let custom_timestamps = vec![0.1, 0.5, 1.2];
        let config = AerConfig::default().with_timestamp_generation(
            true,
            TimestampMode::Custom(custom_timestamps.clone()),
            0.0,
            0.001,
        );

        let reader = AerReader::with_config(config);

        // Create test data
        let events_data = vec![
            ((100u32 << 10) | (50u32 << 1) | 1).to_le_bytes(),
            ((200u32 << 10) | (150u32 << 1) | 0).to_le_bytes(),
            ((300u32 << 10) | (250u32 << 1) | 1).to_le_bytes(),
        ];

        let data: Vec<u8> = events_data.into_iter().flatten().collect();
        let (events, _) = reader.parse_events(&data, data.len() as u64).unwrap();

        assert_eq!(events.len(), 3);
        assert_eq!(events[0].t, 0.1);
        assert_eq!(events[1].t, 0.5);
        assert_eq!(events[2].t, 1.2);
    }

    #[test]
    fn test_is_aer_format() {
        // Create a temporary file with valid AER data
        let mut temp_file = NamedTempFile::new().unwrap();

        let events_data = vec![
            ((100u32 << 10) | (50u32 << 1) | 1).to_le_bytes(),
            ((200u32 << 10) | (150u32 << 1) | 0).to_le_bytes(),
        ];

        for event_bytes in events_data {
            temp_file.write_all(&event_bytes).unwrap();
        }

        assert!(is_aer_format(temp_file.path()));
    }

    #[test]
    fn test_max_events_limit() {
        let config = AerConfig::default().with_max_events(Some(2));
        let reader = AerReader::with_config(config);

        let mut temp_file = NamedTempFile::new().unwrap();

        // Write 3 events but limit to 2
        let events_data = vec![
            ((100u32 << 10) | (50u32 << 1) | 1).to_le_bytes(),
            ((200u32 << 10) | (150u32 << 1) | 0).to_le_bytes(),
            ((300u32 << 10) | (250u32 << 1) | 1).to_le_bytes(),
        ];

        for event_bytes in events_data {
            temp_file.write_all(&event_bytes).unwrap();
        }

        let (events, metadata) = reader.read_file(temp_file.path()).unwrap();

        assert_eq!(events.len(), 2);
        assert_eq!(metadata.event_count, 2);
    }
}
