/// EVT2.1 binary event reader for Prophesee event camera data
///
/// This module provides a reader for EVT2.1 (Event Data 2.1) format used by Prophesee event cameras.
/// EVT2.1 is a 64-bit vectorized data format designed for high-speed event processing with improved
/// compression through vectorized events that can encode up to 32 pixels per event word.
///
/// EVT2.1 Format Structure:
/// - Text header starting with "% evt 2.1" and ending with "% end"
/// - Binary event data with 64-bit words encoding different event types
/// - Vectorized events for efficient transmission of spatially correlated events
/// - Events include CD (Change Detection), Time High, Vector events, and External Trigger events
///
/// Key differences from EVT2:
/// - 64-bit data words instead of 32-bit
/// - Vectorized events support up to 32 pixels per word
/// - Improved timestamp resolution and range
/// - Enhanced data compression for dense event streams
///
/// References:
/// - Prophesee EVT2.1 specification
/// - https://docs.prophesee.ai/stable/data/encoding_formats/evt21.html
/// - OpenEB standalone samples
use crate::ev_core::{Event, Events};
use crate::ev_formats::{polarity_handler::PolarityHandler, LoadConfig, PolarityEncoding};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// EVT2.1 event types encoded in 4-bit field
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum Evt21EventType {
    /// EVT_NEG - Negative polarity vectorized event (32-pixel group)
    EvtNeg = 0x00,
    /// EVT_POS - Positive polarity vectorized event (32-pixel group)
    EvtPos = 0x01,
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
    /// Time High event - encodes higher portion of timebase (bits 63-6)
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

impl TryFrom<u8> for Evt21EventType {
    type Error = Evt21Error;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0x00 => Ok(Evt21EventType::EvtNeg),
            0x01 => Ok(Evt21EventType::EvtPos),
            0x02 => Ok(Evt21EventType::VendorType2),
            0x03 => Ok(Evt21EventType::VendorType3),
            0x04 => Ok(Evt21EventType::VendorType4),
            0x05 => Ok(Evt21EventType::VendorType5),
            0x06 => Ok(Evt21EventType::VendorType6),
            0x07 => Ok(Evt21EventType::VendorType7),
            0x08 => Ok(Evt21EventType::TimeHigh),
            0x09 => Ok(Evt21EventType::VendorType9),
            0x0A => Ok(Evt21EventType::ExtTrigger),
            0x0B => Ok(Evt21EventType::VendorType11),
            0x0C => Ok(Evt21EventType::VendorType12),
            0x0D => Ok(Evt21EventType::VendorType13),
            0x0E => Ok(Evt21EventType::Others),
            0x0F => Ok(Evt21EventType::Continued),
            _ => Err(Evt21Error::InvalidEventType {
                type_value: value,
                offset: 0, // Will be set by caller
            }),
        }
    }
}

/// Raw EVT2.1 event structure (64-bit word)
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct RawEvt21Event {
    pub data: u64,
}

impl RawEvt21Event {
    /// Extract event type from raw data
    pub fn event_type(&self) -> Result<Evt21EventType, Evt21Error> {
        let type_bits = ((self.data >> 60) & 0x0F) as u8;
        Evt21EventType::try_from(type_bits)
    }

    /// Parse as vectorized event (EVT_NEG or EVT_POS)
    pub fn as_vectorized_event(&self) -> Result<VectorizedEvent, Evt21Error> {
        let event_type = self.event_type()?;
        if !matches!(event_type, Evt21EventType::EvtNeg | Evt21EventType::EvtPos) {
            return Err(Evt21Error::InvalidEventType {
                type_value: event_type as u8,
                offset: 0,
            });
        }

        let polarity = matches!(event_type, Evt21EventType::EvtPos);

        // Extract fields from 64-bit word per official EVT2.1 specification
        // Bits 63-60: 4-bit event type (already extracted above)
        // Bits 59-54: 6-bit timestamp (least significant bits)
        let timestamp = ((self.data >> 54) & 0x3F) as u16;

        // Bits 53-43: 11-bit X coordinate (aligned on 32)
        let x_base = ((self.data >> 43) & 0x7FF) as u16;

        // Bits 42-32: 11-bit Y coordinate
        let y = ((self.data >> 32) & 0x7FF) as u16;

        // Bits 31-0: 32-bit validity mask (representing 32 pixel events)
        let validity_mask = (self.data & 0xFFFFFFFF) as u32;

        Ok(VectorizedEvent {
            x_base,
            y,
            timestamp,
            polarity,
            validity_mask,
        })
    }

    /// Parse as Time High event
    pub fn as_time_high_event(&self) -> Result<TimeHighEvent, Evt21Error> {
        if self.event_type()? != Evt21EventType::TimeHigh {
            return Err(Evt21Error::InvalidEventType {
                type_value: self.event_type()? as u8,
                offset: 0,
            });
        }

        Ok(TimeHighEvent {
            timestamp: (self.data >> 32) & 0x0FFFFFFF,
        })
    }

    /// Parse as External Trigger event
    pub fn as_ext_trigger_event(&self) -> Result<ExtTriggerEvent, Evt21Error> {
        if self.event_type()? != Evt21EventType::ExtTrigger {
            return Err(Evt21Error::InvalidEventType {
                type_value: self.event_type()? as u8,
                offset: 0,
            });
        }

        Ok(ExtTriggerEvent {
            value: ((self.data >> 32) & 0x1) != 0,
            id: ((self.data >> 40) & 0x1F) as u8,
            timestamp: ((self.data >> 54) & 0x3F) as u16,
        })
    }

    /// Parse as vendor-specific event
    pub fn as_vendor_event(&self) -> Result<VendorEvent, Evt21Error> {
        let event_type = self.event_type()?;
        if !matches!(
            event_type,
            Evt21EventType::VendorType2
                | Evt21EventType::VendorType3
                | Evt21EventType::VendorType4
                | Evt21EventType::VendorType5
                | Evt21EventType::VendorType6
                | Evt21EventType::VendorType7
                | Evt21EventType::VendorType9
                | Evt21EventType::VendorType11
                | Evt21EventType::VendorType12
                | Evt21EventType::VendorType13
        ) {
            return Err(Evt21Error::InvalidEventType {
                type_value: event_type as u8,
                offset: 0,
            });
        }

        Ok(VendorEvent {
            event_type,
            data: (self.data >> 4) & 0x0FFFFFFFFFFFFFFF,
        })
    }
}

/// Vectorized event structure for EVT_NEG and EVT_POS
#[derive(Debug, Clone, Copy)]
pub struct VectorizedEvent {
    pub x_base: u16,        // Base X coordinate (12 bits)
    pub y: u16,             // Y coordinate (12 bits)
    pub timestamp: u16,     // Timestamp lower bits (10 bits)
    pub polarity: bool,     // Event polarity (true for EVT_POS, false for EVT_NEG)
    pub validity_mask: u32, // 32-bit mask indicating valid pixels
}

/// Time High event structure for 64-bit timestamps
#[derive(Debug, Clone, Copy)]
pub struct TimeHighEvent {
    pub timestamp: u64, // 60-bit timestamp (MSB of full timestamp)
}

/// External Trigger event structure
#[derive(Debug, Clone, Copy)]
pub struct ExtTriggerEvent {
    pub value: bool,    // Trigger edge polarity
    pub id: u8,         // Trigger channel ID (5 bits)
    pub timestamp: u16, // 10-bit timestamp (LSB of full timestamp)
}

/// Vendor-specific event structure
#[derive(Debug, Clone, Copy)]
pub struct VendorEvent {
    pub event_type: Evt21EventType,
    pub data: u64, // 60-bit vendor-specific data
}

/// Errors that can occur during EVT2.1 reading
#[derive(Debug)]
pub enum Evt21Error {
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
    VectorizedDecodingError(String),
}

impl std::fmt::Display for Evt21Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Evt21Error::Io(e) => write!(f, "I/O error: {}", e),
            Evt21Error::InvalidHeader(msg) => write!(f, "Invalid header: {}", msg),
            Evt21Error::InvalidEventType { type_value, offset } => {
                write!(f, "Invalid event type {} at offset {}", type_value, offset)
            }
            Evt21Error::InvalidBinaryData { offset, message } => {
                write!(f, "Invalid binary data at offset {}: {}", offset, message)
            }
            Evt21Error::InsufficientData { expected, actual } => {
                write!(
                    f,
                    "Insufficient data: expected {} bytes, got {} bytes",
                    expected, actual
                )
            }
            Evt21Error::CoordinateOutOfBounds { x, y, max_x, max_y } => {
                write!(
                    f,
                    "Coordinate out of bounds: ({}, {}) exceeds ({}, {})",
                    x, y, max_x, max_y
                )
            }
            Evt21Error::TimestampError(msg) => write!(f, "Timestamp error: {}", msg),
            Evt21Error::PolarityError(e) => write!(f, "Polarity error: {}", e),
            Evt21Error::VectorizedDecodingError(msg) => {
                write!(f, "Vectorized decoding error: {}", msg)
            }
        }
    }
}

impl std::error::Error for Evt21Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Evt21Error::Io(e) => Some(e),
            Evt21Error::PolarityError(e) => Some(e.as_ref()),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Evt21Error {
    fn from(error: std::io::Error) -> Self {
        Evt21Error::Io(error)
    }
}

/// Configuration for EVT2.1 reader
#[derive(Debug, Clone)]
pub struct Evt21Config {
    /// Validate event coordinates against sensor resolution
    pub validate_coordinates: bool,
    /// Skip invalid events instead of returning errors
    pub skip_invalid_events: bool,
    /// Maximum number of events to read (None for unlimited)
    pub max_events: Option<usize>,
    /// Expected sensor resolution (width, height)
    pub sensor_resolution: Option<(u16, u16)>,
    /// Chunk size for reading binary data (in number of 64-bit words)
    pub chunk_size: usize,
    /// Polarity encoding configuration
    pub polarity_encoding: Option<PolarityEncoding>,
    /// Whether to decode vectorized events (if false, only individual events)
    pub decode_vectorized: bool,
}

impl Default for Evt21Config {
    fn default() -> Self {
        Self {
            validate_coordinates: false,
            skip_invalid_events: false,
            max_events: None,
            sensor_resolution: None,
            chunk_size: 500_000, // 500K 64-bit words per chunk
            polarity_encoding: None,
            decode_vectorized: true,
        }
    }
}

/// Metadata extracted from EVT2.1 header
#[derive(Debug, Clone, Default)]
pub struct Evt21Metadata {
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

/// EVT2.1 reader implementation
pub struct Evt21Reader {
    config: Evt21Config,
    polarity_handler: Option<PolarityHandler>,
}

impl Evt21Reader {
    /// Create new EVT2.1 reader with default configuration
    pub fn new() -> Self {
        Self {
            config: Evt21Config::default(),
            polarity_handler: None,
        }
    }

    /// Create new EVT2.1 reader with custom configuration
    pub fn with_config(config: Evt21Config) -> Self {
        let polarity_handler = config
            .polarity_encoding
            .as_ref()
            .map(|_encoding| PolarityHandler::new());

        Self {
            config,
            polarity_handler,
        }
    }

    /// Read EVT2.1 file and return events with metadata
    pub fn read_file<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<(Events, Evt21Metadata), Evt21Error> {
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

        let final_metadata = Evt21Metadata {
            file_size,
            header_size,
            data_size: file_size - header_size,
            estimated_event_count: Some(events.len() as u64),
            ..metadata
        };

        Ok((events, final_metadata))
    }

    /// Read EVT2.1 file with LoadConfig filtering
    pub fn read_with_config<P: AsRef<Path>>(
        &self,
        path: P,
        load_config: &LoadConfig,
    ) -> Result<Events, Evt21Error> {
        let (mut events, _) = self.read_file(path)?;

        // Apply LoadConfig filters
        events.retain(|event| load_config.passes_filters(event));

        // Sort if requested
        if load_config.sort {
            events.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));
        }

        Ok(events)
    }

    /// Parse EVT2.1 header
    fn parse_header(&self, file: &mut File) -> Result<(Evt21Metadata, u64), Evt21Error> {
        let mut reader = BufReader::new(file);
        let mut metadata = Evt21Metadata::default();
        let mut header_size = 0u64;

        // Read header line by line
        loop {
            let mut line = String::new();
            let bytes_read = reader.read_line(&mut line)?;
            if bytes_read == 0 {
                return Err(Evt21Error::InvalidHeader(
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
                            if value != "2.1" {
                                return Err(Evt21Error::InvalidHeader(format!(
                                    "Expected EVT 2.1, got: {}",
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
            return Err(Evt21Error::InvalidHeader(
                "Missing sensor resolution".to_string(),
            ));
        }

        Ok((metadata, header_size))
    }

    /// Parse format line (e.g., "EVT21;height=720;width=1280")
    fn parse_format_line(
        &self,
        line: &str,
        metadata: &mut Evt21Metadata,
    ) -> Result<(), Evt21Error> {
        let parts: Vec<&str> = line.split(';').collect();

        if parts.is_empty() || !parts[0].starts_with("EVT2") {
            return Err(Evt21Error::InvalidHeader(format!(
                "Expected EVT2.1 format, got: {}",
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
                            Evt21Error::InvalidHeader(format!("Invalid width: {}", value))
                        })?);
                    }
                    "height" => {
                        height = Some(value.parse().map_err(|_| {
                            Evt21Error::InvalidHeader(format!("Invalid height: {}", value))
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
        metadata: &mut Evt21Metadata,
    ) -> Result<(), Evt21Error> {
        if let Some((width_str, height_str)) = line.split_once('x') {
            let width = width_str.parse().map_err(|_| {
                Evt21Error::InvalidHeader(format!("Invalid width in geometry: {}", width_str))
            })?;
            let height = height_str.parse().map_err(|_| {
                Evt21Error::InvalidHeader(format!("Invalid height in geometry: {}", height_str))
            })?;

            metadata.sensor_resolution = Some((width, height));
        } else {
            return Err(Evt21Error::InvalidHeader(format!(
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
        metadata: &Evt21Metadata,
    ) -> Result<Events, Evt21Error> {
        // Seek to binary data start
        file.seek(SeekFrom::Start(header_size))?;

        let mut events = Events::new();
        let mut buffer = vec![0u8; self.config.chunk_size * 8]; // 8 bytes per 64-bit word

        // State for timestamp reconstruction
        let mut current_time_base: u64 = 0;
        let mut first_time_base_set = false;
        let mut time_high_loop_count = 0u64;

        // Constants for 64-bit timestamp handling
        const MAX_TIMESTAMP_BASE: u64 = ((1u64 << 50) - 1) << 10; // 50-bit time base to avoid overflow
        const TIME_LOOP: u64 = MAX_TIMESTAMP_BASE + (1 << 10);
        const LOOP_THRESHOLD: u64 = 10 << 10; // Threshold for loop detection

        let mut bytes_read_total = 0;

        loop {
            let bytes_read = file.read(&mut buffer)?;
            if bytes_read == 0 {
                break; // End of file
            }

            bytes_read_total += bytes_read;
            let words_in_chunk = bytes_read / 8;

            // Process events in chunks
            for i in 0..words_in_chunk {
                let word_offset =
                    header_size + (bytes_read_total - bytes_read) as u64 + (i * 8) as u64;
                let raw_bytes = &buffer[i * 8..(i + 1) * 8];

                // Parse raw event (little-endian)
                let raw_data = u64::from_le_bytes([
                    raw_bytes[0],
                    raw_bytes[1],
                    raw_bytes[2],
                    raw_bytes[3],
                    raw_bytes[4],
                    raw_bytes[5],
                    raw_bytes[6],
                    raw_bytes[7],
                ]);

                let raw_event = RawEvt21Event { data: raw_data };

                match raw_event.event_type() {
                    Ok(event_type) => {
                        match event_type {
                            Evt21EventType::TimeHigh => {
                                let time_event =
                                    raw_event.as_time_high_event().map_err(|mut e| {
                                        if let Evt21Error::InvalidEventType { offset, .. } = &mut e
                                        {
                                            *offset = word_offset;
                                        }
                                        e
                                    })?;

                                let new_time_base = time_event.timestamp << 10;
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
                            Evt21EventType::EvtNeg | Evt21EventType::EvtPos => {
                                // Skip vectorized events until we have a time base
                                if !first_time_base_set {
                                    continue;
                                }

                                let vectorized_event =
                                    raw_event.as_vectorized_event().map_err(|mut e| {
                                        if let Evt21Error::InvalidEventType { offset, .. } = &mut e
                                        {
                                            *offset = word_offset;
                                        }
                                        e
                                    })?;

                                // Decode vectorized event into individual events
                                if self.config.decode_vectorized {
                                    let decoded_events = self.decode_vectorized_event(
                                        &vectorized_event,
                                        current_time_base,
                                        metadata,
                                    )?;

                                    for event in decoded_events {
                                        events.push(event);

                                        // Check max events limit
                                        if let Some(max_events) = self.config.max_events {
                                            if events.len() >= max_events {
                                                return Ok(events);
                                            }
                                        }
                                    }
                                }
                            }
                            Evt21EventType::ExtTrigger => {
                                // Skip external trigger events for now
                                // Could be implemented if needed
                            }
                            Evt21EventType::VendorType2
                            | Evt21EventType::VendorType3
                            | Evt21EventType::VendorType4
                            | Evt21EventType::VendorType5
                            | Evt21EventType::VendorType6
                            | Evt21EventType::VendorType7
                            | Evt21EventType::VendorType9
                            | Evt21EventType::VendorType11
                            | Evt21EventType::VendorType12
                            | Evt21EventType::VendorType13 => {
                                // Skip vendor-specific events for now
                                // Could be implemented if needed
                            }
                            Evt21EventType::Others | Evt21EventType::Continued => {
                                // Skip vendor-specific OTHERS and CONTINUED events
                                // These are documented as vendor-specific in the EVT2.1 spec
                            }
                        }
                    }
                    Err(e) => {
                        if self.config.skip_invalid_events {
                            continue;
                        } else {
                            let mut error = e;
                            if let Evt21Error::InvalidEventType { offset, .. } = &mut error {
                                *offset = word_offset;
                            }
                            return Err(error);
                        }
                    }
                }
            }
        }

        Ok(events)
    }

    /// Decode a vectorized event into individual events
    fn decode_vectorized_event(
        &self,
        vectorized_event: &VectorizedEvent,
        current_time_base: u64,
        metadata: &Evt21Metadata,
    ) -> Result<Vec<Event>, Evt21Error> {
        let mut events = Vec::new();
        let full_timestamp = current_time_base + vectorized_event.timestamp as u64;

        // Iterate through the 32-bit validity mask
        for bit_index in 0..32 {
            if (vectorized_event.validity_mask >> bit_index) & 1 != 0 {
                let x = vectorized_event.x_base + bit_index as u16;
                let y = vectorized_event.y;

                // Validate coordinates if configured
                if self.config.validate_coordinates {
                    if let Some((max_x, max_y)) = metadata.sensor_resolution {
                        if x >= max_x || y >= max_y {
                            let error = Evt21Error::CoordinateOutOfBounds { x, y, max_x, max_y };

                            if self.config.skip_invalid_events {
                                continue;
                            } else {
                                return Err(error);
                            }
                        }
                    }
                }

                // Create event
                let event = Event {
                    t: full_timestamp as f64 / 1_000_000.0, // Convert μs to seconds
                    x,
                    y,
                    polarity: vectorized_event.polarity,
                };

                events.push(event);
            }
        }

        Ok(events)
    }
}

impl Default for Evt21Reader {
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
    fn test_evt21_event_type_parsing() {
        // Test EVT_NEG event
        let raw_event = RawEvt21Event {
            data: 0x0000000000000000, // Event type 0x0 (EvtNeg) at bits 63-60
        };
        assert_eq!(raw_event.event_type().unwrap(), Evt21EventType::EvtNeg);

        // Test EVT_POS event
        let raw_event = RawEvt21Event {
            data: 0x1000000000000000, // Event type 0x1 (EvtPos) at bits 63-60
        };
        assert_eq!(raw_event.event_type().unwrap(), Evt21EventType::EvtPos);

        // Test Time High event
        let raw_event = RawEvt21Event {
            data: 0x8000000000000000, // Event type 0x8 (TimeHigh) at bits 63-60
        };
        assert_eq!(raw_event.event_type().unwrap(), Evt21EventType::TimeHigh);

        // Test External Trigger event
        let raw_event = RawEvt21Event {
            data: 0xA000000000000000, // Event type 0xA (ExtTrigger) at bits 63-60
        };
        assert_eq!(raw_event.event_type().unwrap(), Evt21EventType::ExtTrigger);

        // Test that all 4-bit values are accepted (event type now at bits 63-60)
        let raw_event = RawEvt21Event {
            data: 0x0000000000000000, // Event type 0x0 (EVT_NEG) at bits 63-60
        };
        assert_eq!(raw_event.event_type().unwrap(), Evt21EventType::EvtNeg);
    }

    #[test]
    fn test_vectorized_event_parsing() {
        // Test EVT_POS event at (100, 200) with timestamp 30 and validity mask 0x0000000F
        // Using correct EVT2.1 bit layout from official specification

        // Bits 63-60: Event type (0x1 for EVT_POS)
        // Bits 59-54: Timestamp (30)
        // Bits 53-43: X coordinate (100)
        // Bits 42-32: Y coordinate (200)
        // Bits 31-0: Validity mask (0x0000000F)

        let raw_data = (0x1u64 << 60) |        // Event type: EVT_POS
                      (30u64 << 54) |          // Timestamp: 30
                      (100u64 << 43) |         // X coordinate: 100
                      (200u64 << 32) |         // Y coordinate: 200
                      0x0000000F; // Validity mask: 0x0000000F

        let raw_event = RawEvt21Event { data: raw_data };

        let vec_event = raw_event.as_vectorized_event().unwrap();
        assert_eq!(vec_event.x_base, 100);
        assert_eq!(vec_event.y, 200);
        assert_eq!(vec_event.timestamp, 30);
        assert_eq!(vec_event.polarity, true);
        assert_eq!(vec_event.validity_mask, 0x0000000F);
    }

    #[test]
    fn test_time_high_event_parsing() {
        // Test Time High event with timestamp 0x2345678 (28 bits max)
        let raw_data = (0x8u64 << 60) | (0x02345678u64 << 32);
        let raw_event = RawEvt21Event { data: raw_data };

        let time_event = raw_event.as_time_high_event().unwrap();
        assert_eq!(time_event.timestamp, 0x02345678);
    }

    #[test]
    fn test_ext_trigger_event_parsing() {
        // Test External Trigger event with value=true, id=15, timestamp=30
        // Event type 0xA at bits 63-60, timestamp at bits 59-54, id at bits 44-40, value at bit 32
        let raw_data = (0xAu64 << 60) | (30u64 << 54) | (15u64 << 40) | (1u64 << 32);
        let raw_event = RawEvt21Event { data: raw_data };

        let trigger_event = raw_event.as_ext_trigger_event().unwrap();
        assert_eq!(trigger_event.value, true);
        assert_eq!(trigger_event.id, 15);
        assert_eq!(trigger_event.timestamp, 30);
    }

    #[test]
    fn test_header_parsing() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.raw");

        let mut file = File::create(&file_path).unwrap();
        writeln!(file, "% evt 2.1").unwrap();
        writeln!(file, "% format EVT21;height=720;width=1280").unwrap();
        writeln!(file, "% geometry 1280x720").unwrap();
        writeln!(file, "% end").unwrap();
        file.write_all(&[0u8; 64]).unwrap(); // Some dummy binary data

        let reader = Evt21Reader::new();
        let mut file = File::open(&file_path).unwrap();
        let (metadata, header_size) = reader.parse_header(&mut file).unwrap();

        assert_eq!(metadata.sensor_resolution, Some((1280, 720)));
        assert!(header_size > 0);
    }

    #[test]
    fn test_evt21_config_default() {
        let config = Evt21Config::default();
        assert_eq!(config.validate_coordinates, false);
        assert_eq!(config.skip_invalid_events, false);
        assert_eq!(config.max_events, None);
        assert_eq!(config.chunk_size, 500_000);
        assert_eq!(config.decode_vectorized, true);
    }

    #[test]
    fn test_decode_vectorized_event() {
        let reader = Evt21Reader::new();
        let metadata = Evt21Metadata {
            sensor_resolution: Some((1280, 720)),
            ..Default::default()
        };

        let vectorized_event = VectorizedEvent {
            x_base: 100,
            y: 200,
            timestamp: 30,
            polarity: true,
            validity_mask: 0x0000000F, // First 4 bits set
        };

        let events = reader
            .decode_vectorized_event(&vectorized_event, 1000000, &metadata)
            .unwrap();
        assert_eq!(events.len(), 4); // 4 bits set in validity mask

        // Check first event
        assert_eq!(events[0].x, 100);
        assert_eq!(events[0].y, 200);
        assert_eq!(events[0].polarity, true);
        assert_eq!(events[0].t, 1.000030); // (1000000 + 30) / 1_000_000.0

        // Check last event
        assert_eq!(events[3].x, 103);
        assert_eq!(events[3].y, 200);
        assert_eq!(events[3].polarity, true);
    }

    #[test]
    fn test_coordinate_validation() {
        let config = Evt21Config {
            validate_coordinates: true,
            skip_invalid_events: false,
            ..Default::default()
        };
        let reader = Evt21Reader::with_config(config);
        let metadata = Evt21Metadata {
            sensor_resolution: Some((100, 100)),
            ..Default::default()
        };

        let vectorized_event = VectorizedEvent {
            x_base: 98,
            y: 200, // Out of bounds
            timestamp: 30,
            polarity: true,
            validity_mask: 0x00000001,
        };

        let result = reader.decode_vectorized_event(&vectorized_event, 1000000, &metadata);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            Evt21Error::CoordinateOutOfBounds { .. }
        ));
    }

    #[test]
    fn test_coordinate_validation_with_skip() {
        let config = Evt21Config {
            validate_coordinates: true,
            skip_invalid_events: true,
            ..Default::default()
        };
        let reader = Evt21Reader::with_config(config);
        let metadata = Evt21Metadata {
            sensor_resolution: Some((100, 100)),
            ..Default::default()
        };

        let vectorized_event = VectorizedEvent {
            x_base: 98,
            y: 200, // Out of bounds
            timestamp: 30,
            polarity: true,
            validity_mask: 0x00000001,
        };

        let events = reader
            .decode_vectorized_event(&vectorized_event, 1000000, &metadata)
            .unwrap();
        assert_eq!(events.len(), 0); // Event should be skipped
    }
}
