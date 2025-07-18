/// EVT2 binary event reader for Prophesee event camera data
///
/// This module provides a reader for EVT2 (Event Data 2.0) format used by Prophesee event cameras.
/// The format consists of a text header followed by binary event data.
///
/// EVT2 Format Structure:
/// - Text header starting with "% evt 2.0" and ending with "% end"
/// - Binary event data with 32-bit words encoding different event types
/// - Events include CD (Change Detection), Time High, and External Trigger events
///
/// References:
/// - Prophesee EVT2 specification
/// - https://docs.prophesee.ai/stable/data/encoding_formats/evt2.html
/// - OpenEB standalone samples
use crate::ev_core::{Event, Events};
use crate::ev_formats::{polarity_handler::PolarityHandler, LoadConfig, PolarityEncoding};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// EVT2 event types encoded in 4-bit field
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum Evt2EventType {
    /// CD OFF event - decrease in illumination (polarity 0)
    CdOff = 0x00,
    /// CD ON event - increase in illumination (polarity 1)
    CdOn = 0x01,
    /// Vendor-specific event type 2
    VendorType2 = 0x02,
    /// Vendor-specific event type 3
    VendorType3 = 0x03,
    /// Vendor-specific event type 4
    VendorType4 = 0x04,
    /// Vendor-specific event type 5
    VendorType5 = 0x05,
    /// Vendor-specific event type 6
    VendorType6 = 0x06,
    /// Vendor-specific event type 7
    VendorType7 = 0x07,
    /// Time High event - encodes higher portion of timebase (bits 33-6)
    TimeHigh = 0x08,
    /// Vendor-specific event type 9
    VendorType9 = 0x09,
    /// External trigger event
    ExtTrigger = 0x0A,
    /// Vendor-specific event type 11
    VendorType11 = 0x0B,
    /// Vendor-specific event type 12
    VendorType12 = 0x0C,
    /// Vendor-specific event type 13
    VendorType13 = 0x0D,
    /// OTHERS event type - vendor specific
    Others = 0x0E,
    /// CONTINUED event type - extra data for events arriving in multiple words
    Continued = 0x0F,
}

impl TryFrom<u8> for Evt2EventType {
    type Error = Evt2Error;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0x00 => Ok(Evt2EventType::CdOff),
            0x01 => Ok(Evt2EventType::CdOn),
            0x02 => Ok(Evt2EventType::VendorType2),
            0x03 => Ok(Evt2EventType::VendorType3),
            0x04 => Ok(Evt2EventType::VendorType4),
            0x05 => Ok(Evt2EventType::VendorType5),
            0x06 => Ok(Evt2EventType::VendorType6),
            0x07 => Ok(Evt2EventType::VendorType7),
            0x08 => Ok(Evt2EventType::TimeHigh),
            0x09 => Ok(Evt2EventType::VendorType9),
            0x0A => Ok(Evt2EventType::ExtTrigger),
            0x0B => Ok(Evt2EventType::VendorType11),
            0x0C => Ok(Evt2EventType::VendorType12),
            0x0D => Ok(Evt2EventType::VendorType13),
            0x0E => Ok(Evt2EventType::Others),
            0x0F => Ok(Evt2EventType::Continued),
            _ => Err(Evt2Error::InvalidEventType {
                type_value: value,
                offset: 0, // Will be set by caller
            }),
        }
    }
}

/// Raw EVT2 event structure (32-bit word)
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct RawEvt2Event {
    pub data: u32,
}

impl RawEvt2Event {
    /// Extract event type from raw data
    pub fn event_type(&self) -> Result<Evt2EventType, Evt2Error> {
        let type_bits = ((self.data >> 28) & 0x0F) as u8;
        Evt2EventType::try_from(type_bits)
    }

    /// Parse as CD event
    pub fn as_cd_event(&self) -> Result<CdEvent, Evt2Error> {
        let event_type = self.event_type()?;
        if !matches!(
            event_type,
            Evt2EventType::CdOff
                | Evt2EventType::CdOn
                | Evt2EventType::VendorType2
                | Evt2EventType::VendorType3
                | Evt2EventType::VendorType4
                | Evt2EventType::VendorType5
                | Evt2EventType::VendorType6
                | Evt2EventType::VendorType7
                | Evt2EventType::VendorType9
                | Evt2EventType::VendorType11
                | Evt2EventType::VendorType12
                | Evt2EventType::VendorType13
        ) {
            return Err(Evt2Error::InvalidEventType {
                type_value: event_type as u8,
                offset: 0,
            });
        }

        // For vendor-specific types, we try to parse as CD events
        // The polarity is inferred from the event type
        let polarity = match event_type {
            Evt2EventType::CdOn => true,
            Evt2EventType::CdOff => false,
            // For vendor types, we'll try to infer polarity from the data
            // or assume the pattern follows CD_OFF/CD_ON
            Evt2EventType::VendorType2 => false, // Assume OFF-like
            Evt2EventType::VendorType3 => true,  // Assume ON-like
            Evt2EventType::VendorType4 => false, // Assume OFF-like
            Evt2EventType::VendorType5 => true,  // Assume ON-like
            Evt2EventType::VendorType6 => false, // Assume OFF-like
            Evt2EventType::VendorType7 => true,  // Assume ON-like
            Evt2EventType::VendorType9 => true,  // Assume ON-like
            Evt2EventType::VendorType11 => true, // Assume ON-like
            Evt2EventType::VendorType12 => false, // Assume OFF-like
            Evt2EventType::VendorType13 => true, // Assume ON-like
            _ => false,
        };

        Ok(CdEvent {
            x: (self.data & 0x7FF) as u16,
            y: ((self.data >> 11) & 0x7FF) as u16,
            timestamp: ((self.data >> 22) & 0x3F) as u8,
            polarity,
        })
    }

    /// Parse as Time High event
    pub fn as_time_high_event(&self) -> Result<TimeHighEvent, Evt2Error> {
        if self.event_type()? != Evt2EventType::TimeHigh {
            return Err(Evt2Error::InvalidEventType {
                type_value: self.event_type()? as u8,
                offset: 0,
            });
        }

        Ok(TimeHighEvent {
            timestamp: self.data & 0x0FFFFFFF,
        })
    }

    /// Parse as External Trigger event
    pub fn as_ext_trigger_event(&self) -> Result<ExtTriggerEvent, Evt2Error> {
        if self.event_type()? != Evt2EventType::ExtTrigger {
            return Err(Evt2Error::InvalidEventType {
                type_value: self.event_type()? as u8,
                offset: 0,
            });
        }

        Ok(ExtTriggerEvent {
            value: ((self.data >> 4) & 0x1) != 0,
            id: ((self.data >> 12) & 0x1F) as u8,
            timestamp: ((self.data >> 26) & 0x3F) as u8,
        })
    }
}

/// CD (Change Detection) event structure
#[derive(Debug, Clone, Copy)]
pub struct CdEvent {
    pub x: u16,
    pub y: u16,
    pub timestamp: u8, // 6-bit timestamp (LSB of full timestamp)
    pub polarity: bool,
}

/// Time High event structure
#[derive(Debug, Clone, Copy)]
pub struct TimeHighEvent {
    pub timestamp: u32, // 28-bit timestamp (MSB of full timestamp)
}

/// External Trigger event structure
#[derive(Debug, Clone, Copy)]
pub struct ExtTriggerEvent {
    pub value: bool,   // Trigger edge polarity
    pub id: u8,        // Trigger channel ID
    pub timestamp: u8, // 6-bit timestamp (LSB of full timestamp)
}

/// Errors that can occur during EVT2 reading
#[derive(Debug)]
pub enum Evt2Error {
    Io(std::io::Error),
    InvalidHeader(String),
    InvalidEventType {
        type_value: u8,
        offset: u64,
    },
    InvalidBinaryData {
        offset: u64,
        message: String,
    },
    InsufficientData {
        expected: usize,
        actual: usize,
    },
    CoordinateOutOfBounds {
        x: u16,
        y: u16,
        max_x: u16,
        max_y: u16,
    },
    TimestampError(String),
    PolarityError(Box<dyn std::error::Error + Send + Sync>),
}

impl std::fmt::Display for Evt2Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Evt2Error::Io(e) => write!(f, "I/O error: {}", e),
            Evt2Error::InvalidHeader(msg) => write!(f, "Invalid header: {}", msg),
            Evt2Error::InvalidEventType { type_value, offset } => {
                write!(f, "Invalid event type {} at offset {}", type_value, offset)
            }
            Evt2Error::InvalidBinaryData { offset, message } => {
                write!(f, "Invalid binary data at offset {}: {}", offset, message)
            }
            Evt2Error::InsufficientData { expected, actual } => {
                write!(
                    f,
                    "Insufficient data: expected {} bytes, got {} bytes",
                    expected, actual
                )
            }
            Evt2Error::CoordinateOutOfBounds { x, y, max_x, max_y } => {
                write!(
                    f,
                    "Coordinate out of bounds: ({}, {}) exceeds ({}, {})",
                    x, y, max_x, max_y
                )
            }
            Evt2Error::TimestampError(msg) => write!(f, "Timestamp error: {}", msg),
            Evt2Error::PolarityError(e) => write!(f, "Polarity error: {}", e),
        }
    }
}

impl std::error::Error for Evt2Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Evt2Error::Io(e) => Some(e),
            Evt2Error::PolarityError(e) => Some(e.as_ref()),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Evt2Error {
    fn from(error: std::io::Error) -> Self {
        Evt2Error::Io(error)
    }
}

/// Configuration for EVT2 reader
#[derive(Debug, Clone)]
pub struct Evt2Config {
    /// Validate event coordinates against sensor resolution
    pub validate_coordinates: bool,
    /// Skip invalid events instead of returning errors
    pub skip_invalid_events: bool,
    /// Maximum number of events to read (None for unlimited)
    pub max_events: Option<usize>,
    /// Expected sensor resolution (width, height)
    pub sensor_resolution: Option<(u16, u16)>,
    /// Chunk size for reading binary data
    pub chunk_size: usize,
    /// Polarity encoding configuration
    pub polarity_encoding: Option<PolarityEncoding>,
}

impl Default for Evt2Config {
    fn default() -> Self {
        Self {
            validate_coordinates: true,
            skip_invalid_events: false,
            max_events: None,
            sensor_resolution: None,
            chunk_size: 1_000_000, // 1M events per chunk
            polarity_encoding: None,
        }
    }
}

/// Metadata extracted from EVT2 header
#[derive(Debug, Clone, Default)]
pub struct Evt2Metadata {
    /// Sensor resolution (width, height)
    pub sensor_resolution: Option<(u16, u16)>,
    /// Header properties
    pub properties: HashMap<String, String>,
    /// File size in bytes
    pub file_size: u64,
    /// Header size in bytes
    pub header_size: u64,
    /// Data size in bytes
    pub data_size: u64,
    /// Estimated event count
    pub estimated_event_count: Option<u64>,
}

/// EVT2 reader implementation
pub struct Evt2Reader {
    config: Evt2Config,
    polarity_handler: Option<PolarityHandler>,
}

impl Evt2Reader {
    /// Create new EVT2 reader with default configuration
    pub fn new() -> Self {
        Self {
            config: Evt2Config::default(),
            polarity_handler: None,
        }
    }

    /// Create new EVT2 reader with custom configuration
    pub fn with_config(config: Evt2Config) -> Self {
        let polarity_handler = config
            .polarity_encoding
            .as_ref()
            .map(|_encoding| PolarityHandler::new());

        Self {
            config,
            polarity_handler,
        }
    }

    /// Read EVT2 file and return events with metadata
    pub fn read_file<P: AsRef<Path>>(&self, path: P) -> Result<(Events, Evt2Metadata), Evt2Error> {
        let path = path.as_ref();
        let mut file = File::open(path)?;
        let file_size = file.metadata()?.len();

        // Parse header
        let (metadata, header_size) = self.parse_header(&mut file)?;

        // Read binary data
        let events = self.read_binary_data(&mut file, header_size, &metadata)?;

        // Apply polarity encoding if configured
        if let Some(ref _handler) = self.polarity_handler {
            // For now, we'll skip polarity conversion as the implementation needs adjustment
            // The events already use the standard -1/1 encoding
        }

        let final_metadata = Evt2Metadata {
            file_size,
            header_size,
            data_size: file_size - header_size,
            estimated_event_count: Some(events.len() as u64),
            ..metadata
        };

        Ok((events, final_metadata))
    }

    /// Read EVT2 file with LoadConfig filtering
    pub fn read_with_config<P: AsRef<Path>>(
        &self,
        path: P,
        load_config: &LoadConfig,
    ) -> Result<Events, Evt2Error> {
        let (mut events, _) = self.read_file(path)?;

        // Apply LoadConfig filters
        events.retain(|event| load_config.passes_filters(event));

        // Sort if requested
        if load_config.sort {
            events.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));
        }

        Ok(events)
    }

    /// Parse EVT2 header
    fn parse_header(&self, file: &mut File) -> Result<(Evt2Metadata, u64), Evt2Error> {
        let mut reader = BufReader::new(file);
        let mut metadata = Evt2Metadata::default();
        let mut header_size = 0u64;

        // Read header line by line
        loop {
            let mut line = String::new();
            let bytes_read = reader.read_line(&mut line)?;
            if bytes_read == 0 {
                return Err(Evt2Error::InvalidHeader(
                    "Unexpected end of file".to_string(),
                ));
            }

            header_size += bytes_read as u64;
            let line = line.trim_end();

            if line == "% end" {
                break;
            }

            // Parse header fields
            if let Some(stripped) = line.strip_prefix("% ") {
                if let Some((key, value)) = stripped.split_once(' ') {
                    match key {
                        "evt" => {
                            if value != "2.0" {
                                return Err(Evt2Error::InvalidHeader(format!(
                                    "Expected EVT 2.0, got: {}",
                                    value
                                )));
                            }
                        }
                        "format" => {
                            self.parse_format_line(value, &mut metadata)?;
                        }
                        "geometry" => {
                            self.parse_geometry_line(value, &mut metadata)?;
                        }
                        _ => {
                            metadata
                                .properties
                                .insert(key.to_string(), value.to_string());
                        }
                    }
                }
            }
        }

        // Validate required fields
        if metadata.sensor_resolution.is_none() {
            return Err(Evt2Error::InvalidHeader(
                "Missing sensor resolution".to_string(),
            ));
        }

        Ok((metadata, header_size))
    }

    /// Parse format line (e.g., "EVT2;height=720;width=1280")
    fn parse_format_line(&self, line: &str, metadata: &mut Evt2Metadata) -> Result<(), Evt2Error> {
        let parts: Vec<&str> = line.split(';').collect();

        if parts.is_empty() || parts[0] != "EVT2" {
            return Err(Evt2Error::InvalidHeader(format!(
                "Expected EVT2 format, got: {}",
                line
            )));
        }

        let mut width = None;
        let mut height = None;

        for part in parts.iter().skip(1) {
            if let Some((key, value)) = part.split_once('=') {
                match key {
                    "width" => {
                        width = Some(value.parse().map_err(|_| {
                            Evt2Error::InvalidHeader(format!("Invalid width: {}", value))
                        })?);
                    }
                    "height" => {
                        height = Some(value.parse().map_err(|_| {
                            Evt2Error::InvalidHeader(format!("Invalid height: {}", value))
                        })?);
                    }
                    _ => {
                        metadata
                            .properties
                            .insert(key.to_string(), value.to_string());
                    }
                }
            }
        }

        if let (Some(w), Some(h)) = (width, height) {
            metadata.sensor_resolution = Some((w, h));
        }

        Ok(())
    }

    /// Parse geometry line (e.g., "1280x720")
    fn parse_geometry_line(
        &self,
        line: &str,
        metadata: &mut Evt2Metadata,
    ) -> Result<(), Evt2Error> {
        if let Some((width_str, height_str)) = line.split_once('x') {
            let width = width_str.parse().map_err(|_| {
                Evt2Error::InvalidHeader(format!("Invalid width in geometry: {}", width_str))
            })?;
            let height = height_str.parse().map_err(|_| {
                Evt2Error::InvalidHeader(format!("Invalid height in geometry: {}", height_str))
            })?;

            metadata.sensor_resolution = Some((width, height));
        } else {
            return Err(Evt2Error::InvalidHeader(format!(
                "Invalid geometry format: {}",
                line
            )));
        }

        Ok(())
    }

    /// Read binary event data
    fn read_binary_data(
        &self,
        file: &mut File,
        header_size: u64,
        metadata: &Evt2Metadata,
    ) -> Result<Events, Evt2Error> {
        // Seek to binary data start
        file.seek(SeekFrom::Start(header_size))?;

        let mut events = Events::new();
        let mut buffer = vec![0u8; self.config.chunk_size * 4]; // 4 bytes per event

        // State for timestamp reconstruction
        let mut current_time_base: u64 = 0;
        let mut first_time_base_set = false;
        let mut time_high_loop_count = 0u64;

        // Constants for timestamp handling
        const MAX_TIMESTAMP_BASE: u64 = ((1u64 << 28) - 1) << 6; // 17179869120us
        const TIME_LOOP: u64 = MAX_TIMESTAMP_BASE + (1 << 6); // 17179869184us
        const LOOP_THRESHOLD: u64 = 10 << 6; // Threshold for loop detection

        let mut bytes_read_total = 0;

        loop {
            let bytes_read = file.read(&mut buffer)?;
            if bytes_read == 0 {
                break; // End of file
            }

            bytes_read_total += bytes_read;
            let events_in_chunk = bytes_read / 4;

            // Process events in chunks
            for i in 0..events_in_chunk {
                let event_offset =
                    header_size + (bytes_read_total - bytes_read) as u64 + (i * 4) as u64;
                let raw_bytes = &buffer[i * 4..(i + 1) * 4];

                // Parse raw event (little-endian)
                let raw_data =
                    u32::from_le_bytes([raw_bytes[0], raw_bytes[1], raw_bytes[2], raw_bytes[3]]);

                let raw_event = RawEvt2Event { data: raw_data };

                match raw_event.event_type() {
                    Ok(event_type) => {
                        match event_type {
                            Evt2EventType::TimeHigh => {
                                let time_event =
                                    raw_event.as_time_high_event().map_err(|mut e| {
                                        if let Evt2Error::InvalidEventType { offset, .. } = &mut e {
                                            *offset = event_offset;
                                        }
                                        e
                                    })?;

                                let new_time_base = (time_event.timestamp as u64) << 6;
                                let new_time_base_with_loops =
                                    new_time_base + time_high_loop_count * TIME_LOOP;

                                // Handle time loop detection
                                if current_time_base > new_time_base_with_loops
                                    && current_time_base - new_time_base_with_loops
                                        >= MAX_TIMESTAMP_BASE - LOOP_THRESHOLD
                                {
                                    time_high_loop_count += 1;
                                    current_time_base =
                                        new_time_base + time_high_loop_count * TIME_LOOP;
                                } else {
                                    current_time_base = new_time_base_with_loops;
                                }

                                first_time_base_set = true;
                            }
                            Evt2EventType::CdOff
                            | Evt2EventType::CdOn
                            | Evt2EventType::VendorType2
                            | Evt2EventType::VendorType3
                            | Evt2EventType::VendorType4
                            | Evt2EventType::VendorType5
                            | Evt2EventType::VendorType6
                            | Evt2EventType::VendorType7
                            | Evt2EventType::VendorType9
                            | Evt2EventType::VendorType11
                            | Evt2EventType::VendorType12
                            | Evt2EventType::VendorType13 => {
                                // Skip CD events until we have a time base
                                if !first_time_base_set {
                                    continue;
                                }

                                let cd_event = raw_event.as_cd_event().map_err(|mut e| {
                                    if let Evt2Error::InvalidEventType { offset, .. } = &mut e {
                                        *offset = event_offset;
                                    }
                                    e
                                })?;

                                // Validate coordinates if configured
                                if self.config.validate_coordinates {
                                    if let Some((max_x, max_y)) = metadata.sensor_resolution {
                                        if cd_event.x >= max_x || cd_event.y >= max_y {
                                            let error = Evt2Error::CoordinateOutOfBounds {
                                                x: cd_event.x,
                                                y: cd_event.y,
                                                max_x,
                                                max_y,
                                            };

                                            if self.config.skip_invalid_events {
                                                continue;
                                            } else {
                                                return Err(error);
                                            }
                                        }
                                    }
                                }

                                // Reconstruct full timestamp
                                let timestamp = current_time_base + cd_event.timestamp as u64;

                                // Convert to Event struct
                                let event = Event {
                                    t: timestamp as f64 / 1_000_000.0, // Convert μs to seconds
                                    x: cd_event.x,
                                    y: cd_event.y,
                                    polarity: cd_event.polarity,
                                };

                                events.push(event);

                                // Check max events limit
                                if let Some(max_events) = self.config.max_events {
                                    if events.len() >= max_events {
                                        return Ok(events);
                                    }
                                }
                            }
                            Evt2EventType::ExtTrigger => {
                                // Skip external trigger events for now
                                // Could be implemented if needed
                            }
                            Evt2EventType::Others | Evt2EventType::Continued => {
                                // Skip vendor-specific OTHERS and CONTINUED events
                                // These are documented as vendor-specific in the EVT2 spec
                            }
                        }
                    }
                    Err(e) => {
                        if self.config.skip_invalid_events {
                            continue;
                        } else {
                            let mut error = e;
                            if let Evt2Error::InvalidEventType { offset, .. } = &mut error {
                                *offset = event_offset;
                            }
                            return Err(error);
                        }
                    }
                }
            }
        }

        Ok(events)
    }
}

impl Default for Evt2Reader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_evt2_event_type_parsing() {
        // Test CD OFF event - event type at bits 31-28
        let raw_event = RawEvt2Event { data: 0x00000000 };
        assert_eq!(raw_event.event_type().unwrap(), Evt2EventType::CdOff);

        // Test CD ON event
        let raw_event = RawEvt2Event { data: 0x10000000 };
        assert_eq!(raw_event.event_type().unwrap(), Evt2EventType::CdOn);

        // Test Time High event
        let raw_event = RawEvt2Event { data: 0x80000000 };
        assert_eq!(raw_event.event_type().unwrap(), Evt2EventType::TimeHigh);

        // Test External Trigger event
        let raw_event = RawEvt2Event { data: 0xA0000000 };
        assert_eq!(raw_event.event_type().unwrap(), Evt2EventType::ExtTrigger);

        // Test Continued event type (0xF at bits 31-28 is valid)
        let raw_event = RawEvt2Event { data: 0xF0000000 };
        assert_eq!(raw_event.event_type().unwrap(), Evt2EventType::Continued);
    }

    #[test]
    fn test_cd_event_parsing() {
        // Test CD ON event at (100, 200) with timestamp 30
        // Using correct EVT2.0 bit layout: [31-28: type] [27-22: timestamp] [21-11: Y] [10-0: X]
        let raw_data = (0x1u32 << 28) | (30u32 << 22) | (200u32 << 11) | 100u32;
        let raw_event = RawEvt2Event { data: raw_data };

        let cd_event = raw_event.as_cd_event().unwrap();
        assert_eq!(cd_event.x, 100);
        assert_eq!(cd_event.y, 200);
        assert_eq!(cd_event.timestamp, 30);
        assert_eq!(cd_event.polarity, true);
    }

    #[test]
    fn test_time_high_event_parsing() {
        // Test Time High event with timestamp 0x1234567 (28 bits)
        // Using correct EVT2.0 bit layout: [31-28: type (0x8)] [27-0: timestamp]
        let raw_data = (0x8u32 << 28) | 0x1234567u32;
        let raw_event = RawEvt2Event { data: raw_data };

        let time_event = raw_event.as_time_high_event().unwrap();
        assert_eq!(time_event.timestamp, 0x1234567);
    }

    #[test]
    fn test_header_parsing() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.raw");

        let mut file = File::create(&file_path).unwrap();
        writeln!(file, "% evt 2.0").unwrap();
        writeln!(file, "% format EVT2;height=720;width=1280").unwrap();
        writeln!(file, "% geometry 1280x720").unwrap();
        writeln!(file, "% end").unwrap();
        file.write_all(&[0u8; 16]).unwrap(); // Some dummy binary data

        let reader = Evt2Reader::new();
        let mut file = File::open(&file_path).unwrap();
        let (metadata, header_size) = reader.parse_header(&mut file).unwrap();

        assert_eq!(metadata.sensor_resolution, Some((1280, 720)));
        assert!(header_size > 0);
    }

    #[test]
    fn test_evt2_config_default() {
        let config = Evt2Config::default();
        assert_eq!(config.validate_coordinates, true);
        assert_eq!(config.skip_invalid_events, false);
        assert_eq!(config.max_events, None);
        assert_eq!(config.chunk_size, 1_000_000);
    }
}
