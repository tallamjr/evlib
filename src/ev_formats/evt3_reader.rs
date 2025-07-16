/// EVT3 binary event reader for Prophesee event camera data
///
/// This module provides a reader for EVT3 (Event Data 3.0) format used by Prophesee event cameras.
/// EVT3 is a 16-bit vectorized data format designed for data compactness and vector event data support.
/// It avoids transmitting redundant event data for time, y, and x values.
///
/// EVT3 Format Structure:
/// - Text header starting with "% evt 3.0" and ending with "% end"
/// - Binary event data with 16-bit words encoding different event types
/// - Vectorized events to reduce data redundancy
///
/// References:
/// - Prophesee EVT3 specification
/// - https://docs.prophesee.ai/stable/data/encoding_formats/evt3.html
/// - OpenEB standalone samples
use crate::ev_core::{Event, Events};
use crate::ev_formats::{polarity_handler::PolarityHandler, LoadConfig, PolarityEncoding};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// EVT3 event types encoded in 4-bit field
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum Evt3EventType {
    /// Y address event - identifies CD event and Y coordinate
    AddrY = 0x0,
    /// X address event - marks valid single event with polarity and X coordinate
    AddrX = 0x2,
    /// Vector base X - transmits base address for subsequent vector events
    VectBaseX = 0x3,
    /// Vector event with 12 valid bits
    Vect12 = 0x4,
    /// Vector event with 8 valid bits
    Vect8 = 0x5,
    /// Time Low event - encodes lower 12 bits of timebase (bits 11-0)
    TimeLow = 0x6,
    /// Time High event - encodes higher portion of timebase (bits 23-12)
    TimeHigh = 0x8,
    /// External trigger event
    ExtTrigger = 0xA,
}

impl TryFrom<u8> for Evt3EventType {
    type Error = Evt3Error;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0x0 => Ok(Evt3EventType::AddrY),
            0x2 => Ok(Evt3EventType::AddrX),
            0x3 => Ok(Evt3EventType::VectBaseX),
            0x4 => Ok(Evt3EventType::Vect12),
            0x5 => Ok(Evt3EventType::Vect8),
            0x6 => Ok(Evt3EventType::TimeLow),
            0x8 => Ok(Evt3EventType::TimeHigh),
            0xA => Ok(Evt3EventType::ExtTrigger),
            _ => Err(Evt3Error::InvalidEventType {
                type_value: value,
                offset: 0,
            }),
        }
    }
}

/// Raw EVT3 event structure (16-bit word)
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct RawEvt3Event {
    pub data: u16,
}

impl RawEvt3Event {
    /// Extract event type from raw data
    pub fn event_type(&self) -> Result<Evt3EventType, Evt3Error> {
        let type_bits = (self.data & 0x000F) as u8;
        Evt3EventType::try_from(type_bits)
    }

    /// Parse as Y address event
    pub fn as_y_addr_event(&self) -> Result<YAddrEvent, Evt3Error> {
        if self.event_type()? != Evt3EventType::AddrY {
            return Err(Evt3Error::InvalidEventType {
                type_value: self.event_type()? as u8,
                offset: 0,
            });
        }

        Ok(YAddrEvent {
            y: (self.data >> 4) & 0x7FF,
            orig: ((self.data >> 15) & 0x1) != 0,
        })
    }

    /// Parse as X address event
    pub fn as_x_addr_event(&self) -> Result<XAddrEvent, Evt3Error> {
        if self.event_type()? != Evt3EventType::AddrX {
            return Err(Evt3Error::InvalidEventType {
                type_value: self.event_type()? as u8,
                offset: 0,
            });
        }

        Ok(XAddrEvent {
            x: (self.data >> 4) & 0x7FF,
            polarity: ((self.data >> 15) & 0x1) != 0,
        })
    }

    /// Parse as vector base X event
    pub fn as_vect_base_x_event(&self) -> Result<VectBaseXEvent, Evt3Error> {
        if self.event_type()? != Evt3EventType::VectBaseX {
            return Err(Evt3Error::InvalidEventType {
                type_value: self.event_type()? as u8,
                offset: 0,
            });
        }

        Ok(VectBaseXEvent {
            x: (self.data >> 4) & 0x7FF,
            polarity: ((self.data >> 15) & 0x1) != 0,
        })
    }

    /// Parse as vector 12 event
    pub fn as_vect12_event(&self) -> Result<Vect12Event, Evt3Error> {
        if self.event_type()? != Evt3EventType::Vect12 {
            return Err(Evt3Error::InvalidEventType {
                type_value: self.event_type()? as u8,
                offset: 0,
            });
        }

        Ok(Vect12Event {
            valid: (self.data >> 4) & 0xFFF,
        })
    }

    /// Parse as vector 8 event
    pub fn as_vect8_event(&self) -> Result<Vect8Event, Evt3Error> {
        if self.event_type()? != Evt3EventType::Vect8 {
            return Err(Evt3Error::InvalidEventType {
                type_value: self.event_type()? as u8,
                offset: 0,
            });
        }

        Ok(Vect8Event {
            valid: ((self.data >> 4) & 0xFF) as u8,
        })
    }

    /// Parse as time event (low or high)
    pub fn as_time_event(&self) -> Result<TimeEvent, Evt3Error> {
        let event_type = self.event_type()?;
        if !matches!(event_type, Evt3EventType::TimeLow | Evt3EventType::TimeHigh) {
            return Err(Evt3Error::InvalidEventType {
                type_value: event_type as u8,
                offset: 0,
            });
        }

        Ok(TimeEvent {
            time: (self.data >> 4) & 0xFFF,
            is_high: event_type == Evt3EventType::TimeHigh,
        })
    }

    /// Parse as external trigger event
    pub fn as_ext_trigger_event(&self) -> Result<ExtTriggerEvent, Evt3Error> {
        if self.event_type()? != Evt3EventType::ExtTrigger {
            return Err(Evt3Error::InvalidEventType {
                type_value: self.event_type()? as u8,
                offset: 0,
            });
        }

        Ok(ExtTriggerEvent {
            value: ((self.data >> 4) & 0x1) != 0,
            id: ((self.data >> 5) & 0x1F) as u8,
        })
    }
}

/// Y address event structure
#[derive(Debug, Clone, Copy)]
pub struct YAddrEvent {
    pub y: u16,
    pub orig: bool, // System type: false = Master, true = Slave
}

/// X address event structure
#[derive(Debug, Clone, Copy)]
pub struct XAddrEvent {
    pub x: u16,
    pub polarity: bool,
}

/// Vector base X event structure
#[derive(Debug, Clone, Copy)]
pub struct VectBaseXEvent {
    pub x: u16,
    pub polarity: bool,
}

/// Vector 12 event structure
#[derive(Debug, Clone, Copy)]
pub struct Vect12Event {
    pub valid: u16, // 12-bit validity mask
}

/// Vector 8 event structure
#[derive(Debug, Clone, Copy)]
pub struct Vect8Event {
    pub valid: u8, // 8-bit validity mask
}

/// Time event structure (low or high)
#[derive(Debug, Clone, Copy)]
pub struct TimeEvent {
    pub time: u16,
    pub is_high: bool,
}

/// External trigger event structure
#[derive(Debug, Clone, Copy)]
pub struct ExtTriggerEvent {
    pub value: bool,
    pub id: u8,
}

/// Errors that can occur during EVT3 reading
#[derive(Debug)]
pub enum Evt3Error {
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
    DecodingError(String),
}

impl std::fmt::Display for Evt3Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Evt3Error::Io(e) => write!(f, "I/O error: {}", e),
            Evt3Error::InvalidHeader(msg) => write!(f, "Invalid header: {}", msg),
            Evt3Error::InvalidEventType { type_value, offset } => {
                write!(f, "Invalid event type {} at offset {}", type_value, offset)
            }
            Evt3Error::InvalidBinaryData { offset, message } => {
                write!(f, "Invalid binary data at offset {}: {}", offset, message)
            }
            Evt3Error::InsufficientData { expected, actual } => {
                write!(
                    f,
                    "Insufficient data: expected {} bytes, got {} bytes",
                    expected, actual
                )
            }
            Evt3Error::CoordinateOutOfBounds { x, y, max_x, max_y } => {
                write!(
                    f,
                    "Coordinate out of bounds: ({}, {}) exceeds ({}, {})",
                    x, y, max_x, max_y
                )
            }
            Evt3Error::TimestampError(msg) => write!(f, "Timestamp error: {}", msg),
            Evt3Error::PolarityError(e) => write!(f, "Polarity error: {}", e),
            Evt3Error::DecodingError(msg) => write!(f, "Decoding error: {}", msg),
        }
    }
}

impl std::error::Error for Evt3Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Evt3Error::Io(e) => Some(e),
            Evt3Error::PolarityError(e) => Some(e.as_ref()),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Evt3Error {
    fn from(error: std::io::Error) -> Self {
        Evt3Error::Io(error)
    }
}

/// Configuration for EVT3 reader
#[derive(Debug, Clone)]
pub struct Evt3Config {
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

impl Default for Evt3Config {
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

/// Metadata extracted from EVT3 header
#[derive(Debug, Clone, Default)]
pub struct Evt3Metadata {
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

/// Decoder state for EVT3 events
#[derive(Debug, Clone, Default)]
struct DecoderState {
    /// Current timestamp (24-bit)
    current_timestamp: u32,
    /// Current Y coordinate
    current_y: u16,
    /// Current polarity
    #[allow(dead_code)]
    current_polarity: bool,
    /// Current vector base X coordinate
    vect_base_x: u16,
    /// Current vector base polarity
    vect_base_polarity: bool,
    /// Whether we have a valid Y coordinate
    has_y: bool,
    /// Whether we have a valid timestamp
    has_timestamp: bool,
}

/// EVT3 reader implementation
pub struct Evt3Reader {
    config: Evt3Config,
    polarity_handler: Option<PolarityHandler>,
}

impl Evt3Reader {
    /// Create new EVT3 reader with default configuration
    pub fn new() -> Self {
        Self {
            config: Evt3Config::default(),
            polarity_handler: None,
        }
    }

    /// Create new EVT3 reader with custom configuration
    pub fn with_config(config: Evt3Config) -> Self {
        let polarity_handler = config
            .polarity_encoding
            .as_ref()
            .map(|_encoding| PolarityHandler::new());

        Self {
            config,
            polarity_handler,
        }
    }

    /// Read EVT3 file and return events with metadata
    pub fn read_file<P: AsRef<Path>>(&self, path: P) -> Result<(Events, Evt3Metadata), Evt3Error> {
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

        let final_metadata = Evt3Metadata {
            file_size,
            header_size,
            data_size: file_size - header_size,
            estimated_event_count: Some(events.len() as u64),
            ..metadata
        };

        Ok((events, final_metadata))
    }

    /// Read EVT3 file with LoadConfig filtering
    pub fn read_with_config<P: AsRef<Path>>(
        &self,
        path: P,
        load_config: &LoadConfig,
    ) -> Result<Events, Evt3Error> {
        let (mut events, _) = self.read_file(path)?;

        // Apply LoadConfig filters
        events.retain(|event| load_config.passes_filters(event));

        // Sort if requested
        if load_config.sort {
            events.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));
        }

        Ok(events)
    }

    /// Parse EVT3 header
    pub fn parse_header(&self, file: &mut File) -> Result<(Evt3Metadata, u64), Evt3Error> {
        let mut reader = BufReader::new(file);
        let mut metadata = Evt3Metadata::default();
        let mut header_size = 0u64;

        // Read header line by line
        loop {
            let mut line = String::new();
            let bytes_read = reader.read_line(&mut line)?;
            if bytes_read == 0 {
                return Err(Evt3Error::InvalidHeader(
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
                            if value != "3.0" {
                                return Err(Evt3Error::InvalidHeader(format!(
                                    "Expected EVT 3.0, got: {}",
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
            return Err(Evt3Error::InvalidHeader(
                "Missing sensor resolution".to_string(),
            ));
        }

        Ok((metadata, header_size))
    }

    /// Parse format line (e.g., "EVT3;height=720;width=1280")
    fn parse_format_line(&self, line: &str, metadata: &mut Evt3Metadata) -> Result<(), Evt3Error> {
        let parts: Vec<&str> = line.split(';').collect();

        if parts.is_empty() || parts[0] != "EVT3" {
            return Err(Evt3Error::InvalidHeader(format!(
                "Expected EVT3 format, got: {}",
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
                            Evt3Error::InvalidHeader(format!("Invalid width: {}", value))
                        })?);
                    }
                    "height" => {
                        height = Some(value.parse().map_err(|_| {
                            Evt3Error::InvalidHeader(format!("Invalid height: {}", value))
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
        metadata: &mut Evt3Metadata,
    ) -> Result<(), Evt3Error> {
        if let Some((width_str, height_str)) = line.split_once('x') {
            let width = width_str.parse().map_err(|_| {
                Evt3Error::InvalidHeader(format!("Invalid width in geometry: {}", width_str))
            })?;
            let height = height_str.parse().map_err(|_| {
                Evt3Error::InvalidHeader(format!("Invalid height in geometry: {}", height_str))
            })?;

            metadata.sensor_resolution = Some((width, height));
        } else {
            return Err(Evt3Error::InvalidHeader(format!(
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
        metadata: &Evt3Metadata,
    ) -> Result<Events, Evt3Error> {
        // Seek to binary data start
        file.seek(SeekFrom::Start(header_size))?;

        let mut events = Events::new();
        let mut buffer = vec![0u8; self.config.chunk_size * 2]; // 2 bytes per event
        let mut decoder_state = DecoderState::default();

        let mut bytes_read_total = 0;

        loop {
            let bytes_read = file.read(&mut buffer)?;
            if bytes_read == 0 {
                break; // End of file
            }

            bytes_read_total += bytes_read;
            let events_in_chunk = bytes_read / 2;

            // Process events in chunks
            for i in 0..events_in_chunk {
                let event_offset =
                    header_size + (bytes_read_total - bytes_read) as u64 + (i * 2) as u64;
                let raw_bytes = &buffer[i * 2..(i + 1) * 2];

                // Parse raw event (little-endian)
                let raw_data = u16::from_le_bytes([raw_bytes[0], raw_bytes[1]]);
                let raw_event = RawEvt3Event { data: raw_data };

                match raw_event.event_type() {
                    Ok(event_type) => {
                        match event_type {
                            Evt3EventType::TimeLow => {
                                let time_event = raw_event.as_time_event().map_err(|mut e| {
                                    if let Evt3Error::InvalidEventType { offset, .. } = &mut e {
                                        *offset = event_offset;
                                    }
                                    e
                                })?;

                                // Update lower 12 bits of timestamp
                                decoder_state.current_timestamp = (decoder_state.current_timestamp
                                    & 0xFFF000)
                                    | time_event.time as u32;
                                decoder_state.has_timestamp = true;
                            }
                            Evt3EventType::TimeHigh => {
                                let time_event = raw_event.as_time_event().map_err(|mut e| {
                                    if let Evt3Error::InvalidEventType { offset, .. } = &mut e {
                                        *offset = event_offset;
                                    }
                                    e
                                })?;

                                // Update upper 12 bits of timestamp
                                decoder_state.current_timestamp = (decoder_state.current_timestamp
                                    & 0x000FFF)
                                    | ((time_event.time as u32) << 12);
                                decoder_state.has_timestamp = true;
                            }
                            Evt3EventType::AddrY => {
                                let y_event = raw_event.as_y_addr_event().map_err(|mut e| {
                                    if let Evt3Error::InvalidEventType { offset, .. } = &mut e {
                                        *offset = event_offset;
                                    }
                                    e
                                })?;

                                decoder_state.current_y = y_event.y;
                                decoder_state.has_y = true;
                            }
                            Evt3EventType::AddrX => {
                                let x_event = raw_event.as_x_addr_event().map_err(|mut e| {
                                    if let Evt3Error::InvalidEventType { offset, .. } = &mut e {
                                        *offset = event_offset;
                                    }
                                    e
                                })?;

                                // Generate single event
                                if let Some(event) = self.generate_event(
                                    x_event.x,
                                    decoder_state.current_y,
                                    decoder_state.current_timestamp,
                                    x_event.polarity,
                                    &decoder_state,
                                    metadata,
                                )? {
                                    events.push(event);

                                    // Check max events limit
                                    if let Some(max_events) = self.config.max_events {
                                        if events.len() >= max_events {
                                            return Ok(events);
                                        }
                                    }
                                }
                            }
                            Evt3EventType::VectBaseX => {
                                let vect_base_event =
                                    raw_event.as_vect_base_x_event().map_err(|mut e| {
                                        if let Evt3Error::InvalidEventType { offset, .. } = &mut e {
                                            *offset = event_offset;
                                        }
                                        e
                                    })?;

                                decoder_state.vect_base_x = vect_base_event.x;
                                decoder_state.vect_base_polarity = vect_base_event.polarity;
                            }
                            Evt3EventType::Vect12 => {
                                let vect12_event =
                                    raw_event.as_vect12_event().map_err(|mut e| {
                                        if let Evt3Error::InvalidEventType { offset, .. } = &mut e {
                                            *offset = event_offset;
                                        }
                                        e
                                    })?;

                                // Generate events from 12-bit validity mask
                                for bit in 0..12 {
                                    if (vect12_event.valid >> bit) & 1 != 0 {
                                        let x = decoder_state.vect_base_x + bit;
                                        if let Some(event) = self.generate_event(
                                            x,
                                            decoder_state.current_y,
                                            decoder_state.current_timestamp,
                                            decoder_state.vect_base_polarity,
                                            &decoder_state,
                                            metadata,
                                        )? {
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

                                // Update vector base X
                                decoder_state.vect_base_x += 12;
                            }
                            Evt3EventType::Vect8 => {
                                let vect8_event = raw_event.as_vect8_event().map_err(|mut e| {
                                    if let Evt3Error::InvalidEventType { offset, .. } = &mut e {
                                        *offset = event_offset;
                                    }
                                    e
                                })?;

                                // Generate events from 8-bit validity mask
                                for bit in 0..8 {
                                    if (vect8_event.valid >> bit) & 1 != 0 {
                                        let x = decoder_state.vect_base_x + bit as u16;
                                        if let Some(event) = self.generate_event(
                                            x,
                                            decoder_state.current_y,
                                            decoder_state.current_timestamp,
                                            decoder_state.vect_base_polarity,
                                            &decoder_state,
                                            metadata,
                                        )? {
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

                                // Update vector base X
                                decoder_state.vect_base_x += 8;
                            }
                            Evt3EventType::ExtTrigger => {
                                // Skip external trigger events for now
                            }
                        }
                    }
                    Err(e) => {
                        if self.config.skip_invalid_events {
                            continue;
                        } else {
                            let mut error = e;
                            if let Evt3Error::InvalidEventType { offset, .. } = &mut error {
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

    /// Generate an event from coordinates, timestamp, and polarity
    fn generate_event(
        &self,
        x: u16,
        y: u16,
        timestamp: u32,
        polarity: bool,
        decoder_state: &DecoderState,
        metadata: &Evt3Metadata,
    ) -> Result<Option<Event>, Evt3Error> {
        // Check if we have valid state
        if !decoder_state.has_y || !decoder_state.has_timestamp {
            return Ok(None);
        }

        // Validate coordinates if configured
        if self.config.validate_coordinates {
            if let Some((max_x, max_y)) = metadata.sensor_resolution {
                if x >= max_x || y >= max_y {
                    let error = Evt3Error::CoordinateOutOfBounds { x, y, max_x, max_y };

                    if self.config.skip_invalid_events {
                        return Ok(None);
                    } else {
                        return Err(error);
                    }
                }
            }
        }

        // Create event
        let event = Event {
            t: timestamp as f64 / 1_000_000.0, // Convert μs to seconds
            x,
            y,
            polarity: if polarity { 1 } else { -1 },
        };

        Ok(Some(event))
    }
}

impl Default for Evt3Reader {
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
    fn test_evt3_event_type_parsing() {
        // Test Y address event
        let raw_event = RawEvt3Event { data: 0x0000 };
        assert_eq!(raw_event.event_type().unwrap(), Evt3EventType::AddrY);

        // Test X address event
        let raw_event = RawEvt3Event { data: 0x0002 };
        assert_eq!(raw_event.event_type().unwrap(), Evt3EventType::AddrX);

        // Test Vector Base X event
        let raw_event = RawEvt3Event { data: 0x0003 };
        assert_eq!(raw_event.event_type().unwrap(), Evt3EventType::VectBaseX);

        // Test Vector 12 event
        let raw_event = RawEvt3Event { data: 0x0004 };
        assert_eq!(raw_event.event_type().unwrap(), Evt3EventType::Vect12);

        // Test Vector 8 event
        let raw_event = RawEvt3Event { data: 0x0005 };
        assert_eq!(raw_event.event_type().unwrap(), Evt3EventType::Vect8);

        // Test Time Low event
        let raw_event = RawEvt3Event { data: 0x0006 };
        assert_eq!(raw_event.event_type().unwrap(), Evt3EventType::TimeLow);

        // Test Time High event
        let raw_event = RawEvt3Event { data: 0x0008 };
        assert_eq!(raw_event.event_type().unwrap(), Evt3EventType::TimeHigh);

        // Test External Trigger event
        let raw_event = RawEvt3Event { data: 0x000A };
        assert_eq!(raw_event.event_type().unwrap(), Evt3EventType::ExtTrigger);
    }

    #[test]
    fn test_y_addr_event_parsing() {
        // Test Y address event at y=300, orig=true
        let raw_data = (1u16 << 15) | (300u16 << 4) | 0x0;
        let raw_event = RawEvt3Event { data: raw_data };

        let y_event = raw_event.as_y_addr_event().unwrap();
        assert_eq!(y_event.y, 300);
        assert_eq!(y_event.orig, true);
    }

    #[test]
    fn test_x_addr_event_parsing() {
        // Test X address event at x=500, polarity=true
        let raw_data = (1u16 << 15) | (500u16 << 4) | 0x2;
        let raw_event = RawEvt3Event { data: raw_data };

        let x_event = raw_event.as_x_addr_event().unwrap();
        assert_eq!(x_event.x, 500);
        assert_eq!(x_event.polarity, true);
    }

    #[test]
    fn test_vect12_event_parsing() {
        // Test Vector 12 event with validity mask 0xABC
        let raw_data = (0xABCu16 << 4) | 0x4;
        let raw_event = RawEvt3Event { data: raw_data };

        let vect12_event = raw_event.as_vect12_event().unwrap();
        assert_eq!(vect12_event.valid, 0xABC);
    }

    #[test]
    fn test_time_event_parsing() {
        // Test Time Low event with time=0x123
        let raw_data = (0x123u16 << 4) | 0x6;
        let raw_event = RawEvt3Event { data: raw_data };

        let time_event = raw_event.as_time_event().unwrap();
        assert_eq!(time_event.time, 0x123);
        assert_eq!(time_event.is_high, false);

        // Test Time High event with time=0x456
        let raw_data = (0x456u16 << 4) | 0x8;
        let raw_event = RawEvt3Event { data: raw_data };

        let time_event = raw_event.as_time_event().unwrap();
        assert_eq!(time_event.time, 0x456);
        assert_eq!(time_event.is_high, true);
    }

    #[test]
    fn test_decoder_state_default() {
        let state = DecoderState::default();
        assert_eq!(state.current_timestamp, 0);
        assert_eq!(state.current_y, 0);
        assert_eq!(state.current_polarity, false);
        assert_eq!(state.vect_base_x, 0);
        assert_eq!(state.vect_base_polarity, false);
        assert_eq!(state.has_y, false);
        assert_eq!(state.has_timestamp, false);
    }

    #[test]
    fn test_evt3_config_default() {
        let config = Evt3Config::default();
        assert_eq!(config.validate_coordinates, true);
        assert_eq!(config.skip_invalid_events, false);
        assert_eq!(config.max_events, None);
        assert_eq!(config.chunk_size, 1_000_000);
    }
}
