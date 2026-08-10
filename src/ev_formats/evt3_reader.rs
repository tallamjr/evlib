/// EVT3 binary event reader for Prophesee event camera data
///
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
use crate::ev_formats::dataframe_builder::EventDataFrameBuilder;
use crate::ev_formats::EventFormat;
use crate::ev_formats::{polarity_handler::PolarityHandler, LoadConfig, PolarityEncoding};
use polars::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
/// EVT3 event types encoded in 4-bit field
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum Evt3EventType {
    /// Y address event - identifies CD event and Y coordinate
    AddrY = 0x0,
    /// Reserved/Unknown event type (found in some EVT3 files)
    Reserved1 = 0x1,
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
    /// Continued 4 event (reserved)
    Continued4 = 0x7,
    /// Time High event - encodes higher portion of timebase (bits 23-12)
    TimeHigh = 0x8,
    /// Reserved/Unknown event type 9 (found in some EVT3 files)
    Reserved9 = 0x9,
    /// External trigger event
    ExtTrigger = 0xA,
    /// Reserved/Unknown event type B (found in some EVT3 files)
    ReservedB = 0xB,
    /// Reserved/Unknown event type C (found in some EVT3 files)
    ReservedC = 0xC,
    /// Reserved/Unknown event type D (found in some EVT3 files)
    ReservedD = 0xD,
    /// Others event (reserved)
    Others = 0xE,
    /// Continued 12 event (reserved)
    Continued12 = 0xF,
}
impl TryFrom<u8> for Evt3EventType {
    type Error = Evt3Error;
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0x0 => Ok(Evt3EventType::AddrY),
            0x1 => Ok(Evt3EventType::Reserved1),
            0x2 => Ok(Evt3EventType::AddrX),
            0x3 => Ok(Evt3EventType::VectBaseX),
            0x4 => Ok(Evt3EventType::Vect12),
            0x5 => Ok(Evt3EventType::Vect8),
            0x6 => Ok(Evt3EventType::TimeLow),
            0x7 => Ok(Evt3EventType::Continued4),
            0x8 => Ok(Evt3EventType::TimeHigh),
            0x9 => Ok(Evt3EventType::Reserved9),
            0xA => Ok(Evt3EventType::ExtTrigger),
            0xB => Ok(Evt3EventType::ReservedB),
            0xC => Ok(Evt3EventType::ReservedC),
            0xD => Ok(Evt3EventType::ReservedD),
            0xE => Ok(Evt3EventType::Others),
            0xF => Ok(Evt3EventType::Continued12),
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
    // EVT3 wire-format bit layout per OpenEB's C bitfield structs
    // (`hal/cpp/include/metavision/hal/decoders/evt3/evt3_event_types.h`)
    // with first-field-is-LSB packing (GCC/Clang default on ARM64/x86-64):
    //
    //   bit:   15 14 13 12 | 11 | 10 9 8 7 6 5 4 3 2 1 0
    //          <-- type -->  pol <----------- x (or y) ----------->
    //
    // So: type = (data >> 12) & 0xF, pol = (data >> 11) & 1, x = data & 0x7FF,
    // and the generic 12-bit `content` field used for ADDR_Y / time / mask /
    // etc. is `data & 0x0FFF` (bits 0-11). This matches OpenEB end-to-end.
    //
    // The previous layout in this file (type in the low nibble, payload in
    // bits 4-15) was self-consistent in unit tests but produced garbage when
    // fed real Prophesee `.raw` files or V4L2 buffers from a GenX320 — those
    // use the OpenEB layout.
    /// Extract event type from raw data
    pub fn event_type(&self) -> Result<Evt3EventType, Evt3Error> {
        let type_bits = ((self.data >> 12) & 0x000F) as u8;
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
            y: self.data & 0x7FF,
            orig: ((self.data >> 11) & 0x1) != 0,
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
            x: self.data & 0x7FF,
            polarity: ((self.data >> 11) & 0x1) != 0,
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
            x: self.data & 0x7FF,
            polarity: ((self.data >> 11) & 0x1) != 0,
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
            valid: self.data & 0xFFF,
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
            valid: (self.data & 0xFF) as u8,
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
            time: self.data & 0xFFF,
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
        // Ext trigger layout: bits 0 = value, bits 1-5 = id (5 bits)
        Ok(ExtTriggerEvent {
            value: (self.data & 0x1) != 0,
            id: ((self.data >> 1) & 0x1F) as u8,
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
            Evt3Error::Io(e) => write!(f, "I/O error: {e}"),
            Evt3Error::InvalidHeader(msg) => write!(f, "Invalid header: {msg}"),
            Evt3Error::InvalidEventType { type_value, offset } => {
                write!(f, "Invalid event type {type_value} at offset {offset}")
            }
            Evt3Error::InvalidBinaryData { offset, message } => {
                write!(f, "Invalid binary data at offset {offset}: {message}")
            }
            Evt3Error::InsufficientData { expected, actual } => {
                write!(
                    f,
                    "Insufficient data: expected {expected} bytes, got {actual} bytes"
                )
            }
            Evt3Error::CoordinateOutOfBounds { x, y, max_x, max_y } => {
                write!(
                    f,
                    "Coordinate out of bounds: ({x}, {y}) exceeds ({max_x}, {max_y})"
                )
            }
            Evt3Error::TimestampError(msg) => write!(f, "Timestamp error: {msg}"),
            Evt3Error::PolarityError(e) => write!(f, "Polarity error: {e}"),
            Evt3Error::DecodingError(msg) => write!(f, "Decoding error: {msg}"),
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
            validate_coordinates: false, // Disable by default for better compatibility
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
    /// Full reconstructed timestamp in microseconds (time base including
    /// TIME_HIGH loops, plus the current 12-bit TIME_LOW).
    current_timestamp: u64,
    /// Time base from TIME_HIGH, including accumulated loop offsets, in
    /// microseconds (always a multiple of 2^12).
    current_time_base: u64,
    /// Current 12-bit TIME_LOW value in microseconds.
    current_time_low: u64,
    /// Number of TIME_HIGH wraparounds (loops) observed so far.
    time_high_loop_count: u64,
    /// Whether the first TIME_HIGH has been seen (initialises the time base).
    first_time_base_set: bool,
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
    pub fn read_file<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<(DataFrame, Evt3Metadata), Evt3Error> {
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
            estimated_event_count: Some(events.height() as u64),
            ..metadata
        };
        Ok((events, final_metadata))
    }
    /// Read EVT3 file with LoadConfig filtering
    pub fn read_with_config<P: AsRef<Path>>(
        &self,
        path: P,
        load_config: &LoadConfig,
    ) -> Result<DataFrame, Evt3Error> {
        let (df, _) = self.read_file(path)?;
        {
            use polars::prelude::*;
            let mut df = df;
            // Apply time window filter if specified
            if let (Some(start), Some(end)) = (load_config.t_start, load_config.t_end) {
                df = df
                    .lazy()
                    .filter(col("t").gt_eq(lit(start)).and(col("t").lt_eq(lit(end))))
                    .collect()
                    .map_err(|e| Evt3Error::InvalidBinaryData {
                        offset: 0,
                        message: format!("Time window filter failed: {}", e),
                    })?;
            }
            // Apply bounding box filter if specified
            if let (Some(x_min), Some(x_max), Some(y_min), Some(y_max)) = (
                load_config.min_x,
                load_config.max_x,
                load_config.min_y,
                load_config.max_y,
            ) {
                df = df
                    .lazy()
                    .filter(
                        col("x")
                            .gt_eq(lit(x_min))
                            .and(col("x").lt_eq(lit(x_max)))
                            .and(col("y").gt_eq(lit(y_min)))
                            .and(col("y").lt_eq(lit(y_max))),
                    )
                    .collect()
                    .map_err(|e| Evt3Error::InvalidBinaryData {
                        offset: 0,
                        message: format!("Bounding box filter failed: {}", e),
                    })?;
            }
            // Sort if requested
            if load_config.sort {
                df = df.sort(["t"], Default::default()).map_err(|e| {
                    Evt3Error::InvalidBinaryData {
                        offset: 0,
                        message: format!("Sort failed: {}", e),
                    }
                })?;
            }
            Ok(df)
        }
    }
    /// Parse EVT3 header
    pub fn parse_header(&self, file: &mut File) -> Result<(Evt3Metadata, u64), Evt3Error> {
        let mut metadata = Evt3Metadata::default();
        let mut byte_buffer = [0u8; 1];
        let mut current_line = Vec::new();
        let mut header_size = 0u64;
        loop {
            let bytes_read = file.read(&mut byte_buffer)?;
            if bytes_read == 0 {
                // End of file reached - process any remaining line buffer
                if !current_line.is_empty() {
                    let line_str = String::from_utf8_lossy(&current_line);
                    let line = line_str.trim_end();
                    // Check if this line starts with "%" - if not, it's the start of binary data
                    if !line.starts_with("% ") {
                        // This line doesn't start with "% " so binary data has started
                        // Back up to the start of this line
                        header_size -= current_line.len() as u64;
                        break;
                    }
                    if line == "% end" {
                        break;
                    }
                    // Process header line
                    self.process_header_line(line, &mut metadata)?;
                }
                // End of file reached - this is OK if we have some valid header data
                if !metadata.properties.is_empty() || metadata.sensor_resolution.is_some() {
                    break;
                } else {
                    return Err(Evt3Error::InvalidHeader(
                        "Unexpected end of file".to_string(),
                    ));
                }
            }
            let byte = byte_buffer[0];
            header_size += 1;
            if byte == b'\n' {
                // End of line - process the line
                let line_str = String::from_utf8_lossy(&current_line);
                let line = line_str.trim_end();
                // Check if this line starts with "%" - if not, it's the start of binary data
                if !line.starts_with("% ") && !line.is_empty() {
                    // This line doesn't start with "% " so binary data has started
                    // Back up to the start of this line (including the newline)
                    header_size -= current_line.len() as u64 + 1; // +1 for the newline
                    break;
                }
                if line == "% end" {
                    break;
                }
                // Process header line
                if !line.is_empty() {
                    self.process_header_line(line, &mut metadata)?;
                }
                // Clear current line for next iteration
                current_line.clear();
            } else {
                // Add byte to current line
                current_line.push(byte);
            }
        }
        // For Prophesee EVT3 files, sensor resolution might not be in header
        // Set a default resolution if missing (will be auto-detected from events)
        if metadata.sensor_resolution.is_none() {
            // Common resolutions for Prophesee Gen4 cameras
            metadata.sensor_resolution = Some((1280, 720)); // Default, will be updated during event parsing
        }
        Ok((metadata, header_size))
    }
    /// Process a single header line
    fn process_header_line(
        &self,
        line: &str,
        metadata: &mut Evt3Metadata,
    ) -> Result<(), Evt3Error> {
        if let Some(stripped) = line.strip_prefix("% ") {
            if let Some((key, value)) = stripped.split_once(' ') {
                match key {
                    "evt" => {
                        if value != "3.0" {
                            return Err(Evt3Error::InvalidHeader(format!(
                                "Expected EVT 3.0, got: {value}"
                            )));
                        }
                    }
                    "format" => {
                        self.parse_format_line(value, metadata)?;
                    }
                    "geometry" => {
                        self.parse_geometry_line(value, metadata)?;
                    }
                    _ => {
                        metadata
                            .properties
                            .insert(key.to_string(), value.to_string());
                    }
                }
            }
        }
        Ok(())
    }
    /// Parse format line (e.g., "EVT3;height=720;width=1280")
    fn parse_format_line(&self, line: &str, metadata: &mut Evt3Metadata) -> Result<(), Evt3Error> {
        let parts: Vec<&str> = line.split(';').collect();
        if parts.is_empty() || parts[0] != "EVT3" {
            return Err(Evt3Error::InvalidHeader(format!(
                "Expected EVT3 format, got: {line}"
            )));
        }
        let mut width = None;
        let mut height = None;
        for part in parts.iter().skip(1) {
            if let Some((key, value)) = part.split_once('=') {
                match key {
                    "width" => {
                        width = Some(value.parse().map_err(|_| {
                            Evt3Error::InvalidHeader(format!("Invalid width: {value}"))
                        })?);
                    }
                    "height" => {
                        height = Some(value.parse().map_err(|_| {
                            Evt3Error::InvalidHeader(format!("Invalid height: {value}"))
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
                Evt3Error::InvalidHeader(format!("Invalid width in geometry: {width_str}"))
            })?;
            let height = height_str.parse().map_err(|_| {
                Evt3Error::InvalidHeader(format!("Invalid height in geometry: {height_str}"))
            })?;
            metadata.sensor_resolution = Some((width, height));
        } else {
            return Err(Evt3Error::InvalidHeader(format!(
                "Invalid geometry format: {line}"
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
    ) -> Result<DataFrame, Evt3Error> {
        // Use DataFrame-based implementation
        {
            // Seek to binary data start
            file.seek(SeekFrom::Start(header_size))?;
            // Estimate total events for builder capacity
            let estimated_events = ((metadata.data_size) / 2) as usize; // 2 bytes per event
            let mut builder = EventDataFrameBuilder::new(EventFormat::EVT3, estimated_events);
            let mut buffer = vec![0u8; self.config.chunk_size * 2]; // 2 bytes per event
            let mut decoder_state = DecoderState::default();
            // EVT3 timebase loop constants (12-bit TIME_HIGH << 12, mirrors OpenEB).
            const MAX_TIMESTAMP_BASE: u64 = ((1u64 << 12) - 1) << 12; // 16_773_120
            const TIME_LOOP: u64 = MAX_TIMESTAMP_BASE + (1 << 12); // 16_777_216
            const LOOP_THRESHOLD: u64 = 10 << 12;
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
                                    if let Ok(time_event) = raw_event.as_time_event() {
                                        // Update the lower 12 bits (TIME_LOW) of the timestamp.
                                        decoder_state.current_time_low = time_event.time as u64;
                                        decoder_state.current_timestamp = decoder_state
                                            .current_time_base
                                            + decoder_state.current_time_low;
                                        decoder_state.has_timestamp = true;
                                    }
                                }
                                Evt3EventType::TimeHigh => {
                                    if let Ok(time_event) = raw_event.as_time_event() {
                                        // Update the upper 12 bits (TIME_HIGH) of the timestamp,
                                        // tracking 24-bit timebase loops so timestamps keep
                                        // increasing past the ~16.78s wraparound (matches OpenEB).
                                        let new_time_base = (time_event.time as u64) << 12;

                                        if !decoder_state.first_time_base_set {
                                            decoder_state.current_time_base = new_time_base;
                                            decoder_state.first_time_base_set = true;
                                        } else {
                                            let candidate_time_base = new_time_base
                                                + decoder_state.time_high_loop_count * TIME_LOOP;

                                            // Detect a TIME_HIGH wraparound: the new base appears
                                            // to jump backwards by almost the full 24-bit range.
                                            let candidate_time_base = if decoder_state
                                                .current_time_base
                                                > candidate_time_base
                                                && decoder_state.current_time_base
                                                    - candidate_time_base
                                                    >= MAX_TIMESTAMP_BASE - LOOP_THRESHOLD
                                            {
                                                decoder_state.time_high_loop_count += 1;
                                                candidate_time_base + TIME_LOOP
                                            } else {
                                                candidate_time_base
                                            };

                                            decoder_state.current_time_base = candidate_time_base;
                                        }

                                        decoder_state.current_timestamp = decoder_state
                                            .current_time_base
                                            + decoder_state.current_time_low;
                                        decoder_state.has_timestamp = true;
                                    }
                                }
                                Evt3EventType::AddrY => {
                                    if let Ok(y_event) = raw_event.as_y_addr_event() {
                                        decoder_state.current_y = y_event.y;
                                        decoder_state.has_y = true;
                                    }
                                }
                                Evt3EventType::AddrX => {
                                    if let Ok(x_event) = raw_event.as_x_addr_event() {
                                        // Generate single event
                                        if decoder_state.has_y && decoder_state.has_timestamp {
                                            let x = x_event.x;
                                            let y = decoder_state.current_y;
                                            let timestamp = decoder_state.current_timestamp;
                                            let polarity = x_event.polarity;
                                            // Validate coordinates if configured
                                            if self.config.validate_coordinates {
                                                if let Some((max_x, max_y)) =
                                                    metadata.sensor_resolution
                                                {
                                                    if x >= max_x || y >= max_y {
                                                        if self.config.skip_invalid_events {
                                                            continue;
                                                        } else {
                                                            return Err(
                                                                Evt3Error::CoordinateOutOfBounds {
                                                                    x,
                                                                    y,
                                                                    max_x,
                                                                    max_y,
                                                                },
                                                            );
                                                        }
                                                    }
                                                }
                                            }
                                            // Add event directly to DataFrame builder
                                            builder.add_event_microseconds(
                                                x,
                                                y,
                                                timestamp as i64,
                                                polarity,
                                            );
                                            // Check max events limit
                                            if let Some(max_events) = self.config.max_events {
                                                if builder.len() >= max_events {
                                                    return builder.build().map_err(|e| {
                                                        Evt3Error::InvalidBinaryData {
                                                            offset: event_offset,
                                                            message: format!(
                                                                "DataFrame build failed: {}",
                                                                e
                                                            ),
                                                        }
                                                    });
                                                }
                                            }
                                        }
                                    }
                                }
                                Evt3EventType::VectBaseX => {
                                    if let Ok(vect_base_event) = raw_event.as_vect_base_x_event() {
                                        decoder_state.vect_base_x = vect_base_event.x;
                                        decoder_state.vect_base_polarity = vect_base_event.polarity;
                                    }
                                }
                                Evt3EventType::Vect12 => {
                                    if let Ok(vect12_event) = raw_event.as_vect12_event() {
                                        // Generate events from 12-bit validity mask
                                        for bit in 0..12 {
                                            if (vect12_event.valid >> bit) & 1 != 0
                                                && decoder_state.has_y
                                                && decoder_state.has_timestamp
                                            {
                                                let x = decoder_state.vect_base_x + bit;
                                                let y = decoder_state.current_y;
                                                let timestamp = decoder_state.current_timestamp;
                                                let polarity = decoder_state.vect_base_polarity;
                                                // Validate coordinates if configured
                                                if self.config.validate_coordinates {
                                                    if let Some((max_x, max_y)) =
                                                        metadata.sensor_resolution
                                                    {
                                                        if x >= max_x || y >= max_y {
                                                            if self.config.skip_invalid_events {
                                                                continue;
                                                            } else {
                                                                return Err(Evt3Error::CoordinateOutOfBounds { x, y, max_x, max_y });
                                                            }
                                                        }
                                                    }
                                                }
                                                // Add event directly to DataFrame builder
                                                builder.add_event_microseconds(
                                                    x,
                                                    y,
                                                    timestamp as i64,
                                                    polarity,
                                                );
                                                // Check max events limit
                                                if let Some(max_events) = self.config.max_events {
                                                    if builder.len() >= max_events {
                                                        return builder.build().map_err(|e| {
                                                            Evt3Error::InvalidBinaryData {
                                                                offset: event_offset,
                                                                message: format!(
                                                                    "DataFrame build failed: {}",
                                                                    e
                                                                ),
                                                            }
                                                        });
                                                    }
                                                }
                                            }
                                        }
                                        // Update vector base X
                                        decoder_state.vect_base_x += 12;
                                    }
                                }
                                Evt3EventType::Vect8 => {
                                    if let Ok(vect8_event) = raw_event.as_vect8_event() {
                                        // Generate events from 8-bit validity mask
                                        for bit in 0..8 {
                                            if (vect8_event.valid >> bit) & 1 != 0
                                                && decoder_state.has_y
                                                && decoder_state.has_timestamp
                                            {
                                                let x = decoder_state.vect_base_x + bit as u16;
                                                let y = decoder_state.current_y;
                                                let timestamp = decoder_state.current_timestamp;
                                                let polarity = decoder_state.vect_base_polarity;
                                                // Validate coordinates if configured
                                                if self.config.validate_coordinates {
                                                    if let Some((max_x, max_y)) =
                                                        metadata.sensor_resolution
                                                    {
                                                        if x >= max_x || y >= max_y {
                                                            if self.config.skip_invalid_events {
                                                                continue;
                                                            } else {
                                                                return Err(Evt3Error::CoordinateOutOfBounds { x, y, max_x, max_y });
                                                            }
                                                        }
                                                    }
                                                }
                                                // Add event directly to DataFrame builder
                                                builder.add_event_microseconds(
                                                    x,
                                                    y,
                                                    timestamp as i64,
                                                    polarity,
                                                );
                                                // Check max events limit
                                                if let Some(max_events) = self.config.max_events {
                                                    if builder.len() >= max_events {
                                                        return builder.build().map_err(|e| {
                                                            Evt3Error::InvalidBinaryData {
                                                                offset: event_offset,
                                                                message: format!(
                                                                    "DataFrame build failed: {}",
                                                                    e
                                                                ),
                                                            }
                                                        });
                                                    }
                                                }
                                            }
                                        }
                                        // Update vector base X
                                        decoder_state.vect_base_x += 8;
                                    }
                                }
                                _ => {
                                    // Skip other event types (ExtTrigger, Reserved, etc.)
                                    continue;
                                }
                            }
                        }
                        Err(_) => {
                            if !self.config.skip_invalid_events {
                                return Err(Evt3Error::InvalidEventType {
                                    type_value: 0,
                                    offset: event_offset,
                                });
                            }
                        }
                    }
                }
            }
            builder.build().map_err(|e| Evt3Error::InvalidBinaryData {
                offset: 0,
                message: format!("DataFrame build failed: {}", e),
            })
        }
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
    /// Build a 16-bit EVT3 word from type (bits 12-15) + 12-bit content.
    fn evt3_word(ty: u16, content: u16) -> u16 {
        ((ty & 0xF) << 12) | (content & 0xFFF)
    }

    #[test]
    fn test_evt3_event_type_parsing() {
        // Type lives in bits 12-15 of the 16-bit word.
        let raw_event = RawEvt3Event {
            data: evt3_word(0x0, 0),
        };
        assert_eq!(raw_event.event_type().unwrap(), Evt3EventType::AddrY);
        let raw_event = RawEvt3Event {
            data: evt3_word(0x2, 0),
        };
        assert_eq!(raw_event.event_type().unwrap(), Evt3EventType::AddrX);
        let raw_event = RawEvt3Event {
            data: evt3_word(0x3, 0),
        };
        assert_eq!(raw_event.event_type().unwrap(), Evt3EventType::VectBaseX);
        let raw_event = RawEvt3Event {
            data: evt3_word(0x4, 0),
        };
        assert_eq!(raw_event.event_type().unwrap(), Evt3EventType::Vect12);
        let raw_event = RawEvt3Event {
            data: evt3_word(0x5, 0),
        };
        assert_eq!(raw_event.event_type().unwrap(), Evt3EventType::Vect8);
        let raw_event = RawEvt3Event {
            data: evt3_word(0x6, 0),
        };
        assert_eq!(raw_event.event_type().unwrap(), Evt3EventType::TimeLow);
        let raw_event = RawEvt3Event {
            data: evt3_word(0x8, 0),
        };
        assert_eq!(raw_event.event_type().unwrap(), Evt3EventType::TimeHigh);
        let raw_event = RawEvt3Event {
            data: evt3_word(0xA, 0),
        };
        assert_eq!(raw_event.event_type().unwrap(), Evt3EventType::ExtTrigger);
        let raw_event = RawEvt3Event {
            data: evt3_word(0x1, 0),
        };
        assert_eq!(raw_event.event_type().unwrap(), Evt3EventType::Reserved1);
        let raw_event = RawEvt3Event {
            data: evt3_word(0x7, 0),
        };
        assert_eq!(raw_event.event_type().unwrap(), Evt3EventType::Continued4);
        let raw_event = RawEvt3Event {
            data: evt3_word(0xE, 0),
        };
        assert_eq!(raw_event.event_type().unwrap(), Evt3EventType::Others);
        let raw_event = RawEvt3Event {
            data: evt3_word(0xF, 0),
        };
        assert_eq!(raw_event.event_type().unwrap(), Evt3EventType::Continued12);
    }
    #[test]
    fn test_y_addr_event_parsing() {
        // Y=300 (bits 0-10), orig=true (bit 11), type=AddrY (bits 12-15)
        let raw_data = evt3_word(0x0, (1 << 11) | 300);
        let raw_event = RawEvt3Event { data: raw_data };
        let y_event = raw_event.as_y_addr_event().unwrap();
        assert_eq!(y_event.y, 300);
        assert!(y_event.orig);
    }
    #[test]
    fn test_x_addr_event_parsing() {
        // X=500 (bits 0-10), polarity=true (bit 11), type=AddrX (bits 12-15)
        let raw_data = evt3_word(0x2, (1 << 11) | 500);
        let raw_event = RawEvt3Event { data: raw_data };
        let x_event = raw_event.as_x_addr_event().unwrap();
        assert_eq!(x_event.x, 500);
        assert!(x_event.polarity);
    }
    #[test]
    fn test_vect12_event_parsing() {
        // Vector 12 event with validity mask 0xABC in bits 0-11, type in bits 12-15
        let raw_data = evt3_word(0x4, 0xABC);
        let raw_event = RawEvt3Event { data: raw_data };
        let vect12_event = raw_event.as_vect12_event().unwrap();
        assert_eq!(vect12_event.valid, 0xABC);
    }
    #[test]
    fn test_time_event_parsing() {
        let raw_data = evt3_word(0x6, 0x123);
        let raw_event = RawEvt3Event { data: raw_data };
        let time_event = raw_event.as_time_event().unwrap();
        assert_eq!(time_event.time, 0x123);
        assert!(!time_event.is_high);
        let raw_data = evt3_word(0x8, 0x456);
        let raw_event = RawEvt3Event { data: raw_data };
        let time_event = raw_event.as_time_event().unwrap();
        assert_eq!(time_event.time, 0x456);
        assert!(time_event.is_high);
    }
    #[test]
    fn test_decoder_state_default() {
        let state = DecoderState::default();
        assert_eq!(state.current_timestamp, 0);
        assert_eq!(state.current_y, 0);
        assert!(!state.current_polarity);
        assert_eq!(state.vect_base_x, 0);
        assert!(!state.vect_base_polarity);
        assert!(!state.has_y);
        assert!(!state.has_timestamp);
    }
    #[test]
    fn test_evt3_config_default() {
        let config = Evt3Config::default();
        assert!(!config.validate_coordinates); // Changed to false for better compatibility
        assert!(!config.skip_invalid_events);
        assert_eq!(config.max_events, None);
        assert_eq!(config.chunk_size, 1_000_000);
    }
}
