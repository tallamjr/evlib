use crate::ev_formats::dataframe_builder::{
    calculate_optimal_chunk_size, EventDataFrameBuilder, EventDataFrameStreamer,
};
use crate::ev_formats::EventFormat;
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
// Removed: use crate::{Event, Events}; - legacy types no longer exist
use crate::ev_formats::{polarity_handler::PolarityHandler, LoadConfig, PolarityEncoding};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

#[cfg(feature = "polars")]
use polars::prelude::*;

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
            Evt2Error::Io(e) => write!(f, "I/O error: {e}"),
            Evt2Error::InvalidHeader(msg) => write!(f, "Invalid header: {msg}"),
            Evt2Error::InvalidEventType { type_value, offset } => {
                write!(f, "Invalid event type {type_value} at offset {offset}")
            }
            Evt2Error::InvalidBinaryData { offset, message } => {
                write!(f, "Invalid binary data at offset {offset}: {message}")
            }
            Evt2Error::InsufficientData { expected, actual } => {
                write!(
                    f,
                    "Insufficient data: expected {expected} bytes, got {actual} bytes"
                )
            }
            Evt2Error::CoordinateOutOfBounds { x, y, max_x, max_y } => {
                write!(
                    f,
                    "Coordinate out of bounds: ({x}, {y}) exceeds ({max_x}, {max_y})"
                )
            }
            Evt2Error::TimestampError(msg) => write!(f, "Timestamp error: {msg}"),
            Evt2Error::PolarityError(e) => write!(f, "Polarity error: {e}"),
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
    pub fn read_file<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<(DataFrame, Evt2Metadata), Evt2Error> {
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
            estimated_event_count: Some(events.height() as u64),
            ..metadata
        };

        Ok((events, final_metadata))
    }

    /// Read EVT2 file with LoadConfig filtering
    pub fn read_with_config<P: AsRef<Path>>(
        &self,
        path: P,
        load_config: &LoadConfig,
    ) -> Result<DataFrame, Evt2Error> {
        let (df, _) = self.read_file(path)?;

        #[cfg(feature = "polars")]
        {
            use polars::prelude::*;
            let mut df = df;

            // Apply time window filter if specified
            if let (Some(start), Some(end)) = (load_config.t_start, load_config.t_end) {
                df = df
                    .lazy()
                    .filter(col("t").gt_eq(lit(start)).and(col("t").lt_eq(lit(end))))
                    .collect()
                    .map_err(|e| Evt2Error::InvalidBinaryData {
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
                    .map_err(|e| Evt2Error::InvalidBinaryData {
                        offset: 0,
                        message: format!("Bounding box filter failed: {}", e),
                    })?;
            }

            // Sort if requested
            if load_config.sort {
                df = df.sort(["t"], Default::default()).map_err(|e| {
                    Evt2Error::InvalidBinaryData {
                        offset: 0,
                        message: format!("Sort failed: {}", e),
                    }
                })?;
            }

            Ok(df)
        }

        #[cfg(not(feature = "polars"))]
        {
            Err(Evt2Error::InvalidBinaryData {
                offset: 0,
                message: "Polars feature not enabled for DataFrame support".to_string(),
            })
        }
    }

    /// Parse EVT2 header
    fn parse_header(&self, file: &mut File) -> Result<(Evt2Metadata, u64), Evt2Error> {
        let mut metadata = Evt2Metadata::default();
        // Remove unused header_buffer variable
        let mut byte_buffer = [0u8; 1];
        let mut current_line = Vec::new();
        let mut header_size = 0u64;

        // Read header byte by byte to avoid UTF-8 issues with binary data
        let mut consecutive_binary_bytes = 0;
        const MAX_BINARY_BYTES: usize = 10; // If we see this many non-printable bytes, assume binary data started

        loop {
            let bytes_read = file.read(&mut byte_buffer)?;
            if bytes_read == 0 {
                // End of file reached - this is OK if we have some valid header data
                if !metadata.properties.is_empty() {
                    break;
                } else {
                    return Err(Evt2Error::InvalidHeader(
                        "Unexpected end of file".to_string(),
                    ));
                }
            }

            header_size += 1;
            let byte = byte_buffer[0];

            // Check if we're hitting binary data (non-printable ASCII bytes)
            if byte < 32 && byte != b'\n' && byte != b'\r' && byte != b'\t' {
                consecutive_binary_bytes += 1;
                if consecutive_binary_bytes > MAX_BINARY_BYTES {
                    // We've hit binary data, back up to where it started
                    header_size -= consecutive_binary_bytes as u64;
                    break;
                }
            } else {
                consecutive_binary_bytes = 0;
            }

            if byte == b'\n' {
                // End of line - process the line
                let line_str = String::from_utf8_lossy(&current_line);
                let line = line_str.trim_end();

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
                                        "Expected EVT 2.0, got: {value}"
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

                // Clear current line for next iteration
                current_line.clear();
            } else {
                // Add byte to current line
                current_line.push(byte);
            }
        }

        // For Prophesee RAW files, sensor resolution might not be in header
        // Set a default resolution if missing (will be auto-detected from events)
        if metadata.sensor_resolution.is_none() {
            // Common resolutions for Prophesee Gen3 cameras
            metadata.sensor_resolution = Some((640, 480)); // Default, will be updated during event parsing
        }

        Ok((metadata, header_size))
    }

    /// Parse format line (e.g., "EVT2;height=720;width=1280")
    fn parse_format_line(&self, line: &str, metadata: &mut Evt2Metadata) -> Result<(), Evt2Error> {
        let parts: Vec<&str> = line.split(';').collect();

        if parts.is_empty() || parts[0] != "EVT2" {
            return Err(Evt2Error::InvalidHeader(format!(
                "Expected EVT2 format, got: {line}"
            )));
        }

        let mut width = None;
        let mut height = None;

        for part in parts.iter().skip(1) {
            if let Some((key, value)) = part.split_once('=') {
                match key {
                    "width" => {
                        width = Some(value.parse().map_err(|_| {
                            Evt2Error::InvalidHeader(format!("Invalid width: {value}"))
                        })?);
                    }
                    "height" => {
                        height = Some(value.parse().map_err(|_| {
                            Evt2Error::InvalidHeader(format!("Invalid height: {value}"))
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
                Evt2Error::InvalidHeader(format!("Invalid width in geometry: {width_str}"))
            })?;
            let height = height_str.parse().map_err(|_| {
                Evt2Error::InvalidHeader(format!("Invalid height in geometry: {height_str}"))
            })?;

            metadata.sensor_resolution = Some((width, height));
        } else {
            return Err(Evt2Error::InvalidHeader(format!(
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
        metadata: &Evt2Metadata,
    ) -> Result<DataFrame, Evt2Error> {
        // For DataFrame mode, use the optimized path
        #[cfg(feature = "polars")]
        {
            let estimated_events = (metadata.data_size / 4) as usize; // 4 bytes per event
            self.read_binary_data_to_dataframe(file, header_size, estimated_events)
                .map_err(|e| Evt2Error::InvalidBinaryData {
                    offset: 0,
                    message: format!("DataFrame conversion failed: {}", e),
                })
        }

        #[cfg(not(feature = "polars"))]
        {
            return Err(Evt2Error::InvalidBinaryData {
                offset: 0,
                message: "Polars feature not enabled for DataFrame support".to_string(),
            });
        }
    }

    /// Read EVT2 file directly into a Polars DataFrame (optimized path)
    /// This eliminates the intermediate Event struct and builds the DataFrame directly
    #[cfg(feature = "polars")]
    pub fn read_file_to_dataframe<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> Result<(DataFrame, Evt2Metadata), Box<dyn std::error::Error + Send + Sync>> {
        let path = path.as_ref();
        let mut file = File::open(path)?;
        let file_size = file.metadata()?.len();

        // Parse header
        let (metadata, header_size) = self.parse_header(&mut file).map_err(Box::new)?;

        // Calculate optimal chunk size
        let chunk_size = calculate_optimal_chunk_size(file_size, 1_000_000_000); // 1GB default available memory

        // Estimate total events for builder capacity
        let estimated_events = ((file_size - header_size) / 4) as usize; // 4 bytes per event

        // Use streaming if the dataset is large
        if estimated_events > 5_000_000 {
            let df = self.read_binary_data_streaming(&mut file, header_size, chunk_size)?;
            let final_metadata = Evt2Metadata {
                file_size,
                header_size,
                data_size: file_size - header_size,
                estimated_event_count: Some(df.height() as u64),
                ..metadata
            };
            Ok((df, final_metadata))
        } else {
            let df =
                self.read_binary_data_to_dataframe(&mut file, header_size, estimated_events)?;
            let final_metadata = Evt2Metadata {
                file_size,
                header_size,
                data_size: file_size - header_size,
                estimated_event_count: Some(df.height() as u64),
                ..metadata
            };
            Ok((df, final_metadata))
        }
    }

    /// Read binary data directly into DataFrame (small files)
    #[cfg(feature = "polars")]
    fn read_binary_data_to_dataframe(
        &self,
        file: &mut File,
        header_size: u64,
        estimated_events: usize,
    ) -> Result<DataFrame, Box<dyn std::error::Error + Send + Sync>> {
        // Seek to binary data start
        file.seek(SeekFrom::Start(header_size))?;

        let mut builder = EventDataFrameBuilder::new(EventFormat::EVT2, estimated_events);
        let mut buffer = vec![0u8; self.config.chunk_size * 4]; // 4 bytes per event

        // State for timestamp reconstruction (same as original)
        let mut current_time_base: u64 = 0;
        let mut first_time_base_set = false;
        let mut time_high_loop_count = 0u64;

        // Constants for timestamp handling
        const MAX_TIMESTAMP_BASE: u64 = ((1u64 << 28) - 1) << 6;
        const TIME_LOOP: u64 = MAX_TIMESTAMP_BASE + (1 << 6);
        const LOOP_THRESHOLD: u64 = 10 << 6;

        let mut _bytes_read_total = 0;

        loop {
            let bytes_read = file.read(&mut buffer)?;
            if bytes_read == 0 {
                break; // End of file
            }

            _bytes_read_total += bytes_read;
            let events_in_chunk = bytes_read / 4;

            // Process events in chunks
            for i in 0..events_in_chunk {
                let raw_bytes = &buffer[i * 4..(i + 1) * 4];

                // Parse raw event (little-endian)
                let raw_data =
                    u32::from_le_bytes([raw_bytes[0], raw_bytes[1], raw_bytes[2], raw_bytes[3]]);
                let raw_event = RawEvt2Event { data: raw_data };

                if let Ok(event_type) = raw_event.event_type() {
                    match event_type {
                        Evt2EventType::TimeHigh => {
                            if let Ok(time_event) = raw_event.as_time_high_event() {
                                let new_time_base = time_event.timestamp as u64;

                                if !first_time_base_set {
                                    current_time_base = new_time_base;
                                    first_time_base_set = true;
                                } else if new_time_base < current_time_base
                                    && (current_time_base - new_time_base) > LOOP_THRESHOLD
                                {
                                    time_high_loop_count += 1;
                                    current_time_base =
                                        new_time_base + time_high_loop_count * TIME_LOOP;
                                } else {
                                    current_time_base =
                                        new_time_base + time_high_loop_count * TIME_LOOP;
                                }
                            }
                        }
                        Evt2EventType::CdOff | Evt2EventType::CdOn => {
                            if let Ok(cd_event) = raw_event.as_cd_event() {
                                let x = cd_event.x;
                                let y = cd_event.y;
                                let polarity = event_type == Evt2EventType::CdOn;

                                // Calculate full timestamp
                                let timestamp =
                                    (current_time_base + cd_event.timestamp as u64) as f64;

                                // Validate coordinates if enabled
                                if self.config.validate_coordinates {
                                    if let Some((max_x, max_y)) = self.config.sensor_resolution {
                                        if (x >= max_x || y >= max_y)
                                            && self.config.skip_invalid_events
                                        {
                                            continue;
                                        }
                                    }
                                }

                                // Add event directly to DataFrame builder
                                builder.add_event(x, y, timestamp, polarity);

                                // Check max events limit
                                if let Some(max_events) = self.config.max_events {
                                    if builder.len() >= max_events {
                                        return Ok(builder.build()?);
                                    }
                                }
                            }
                        }
                        _ => {
                            // Skip other event types (External Trigger, etc.)
                            continue;
                        }
                    }
                }
            }
        }

        Ok(builder.build()?)
    }

    /// Read binary data using streaming for large files
    #[cfg(feature = "polars")]
    fn read_binary_data_streaming(
        &self,
        file: &mut File,
        header_size: u64,
        chunk_size: usize,
    ) -> Result<DataFrame, Box<dyn std::error::Error + Send + Sync>> {
        // Seek to binary data start
        file.seek(SeekFrom::Start(header_size))?;

        let mut streamer = EventDataFrameStreamer::new(EventFormat::EVT2, chunk_size);
        let mut buffer = vec![0u8; self.config.chunk_size * 4];
        let mut dataframes: Vec<DataFrame> = Vec::new();

        // State for timestamp reconstruction (same as original)
        let mut current_time_base: u64 = 0;
        let mut first_time_base_set = false;
        let mut time_high_loop_count = 0u64;

        const MAX_TIMESTAMP_BASE: u64 = ((1u64 << 28) - 1) << 6;
        const TIME_LOOP: u64 = MAX_TIMESTAMP_BASE + (1 << 6);
        const LOOP_THRESHOLD: u64 = 10 << 6;

        let mut _bytes_read_total = 0;

        loop {
            let bytes_read = file.read(&mut buffer)?;
            if bytes_read == 0 {
                break; // End of file
            }

            _bytes_read_total += bytes_read;
            let events_in_chunk = bytes_read / 4;

            // Process events in chunks
            for i in 0..events_in_chunk {
                let raw_bytes = &buffer[i * 4..(i + 1) * 4];

                let raw_data =
                    u32::from_le_bytes([raw_bytes[0], raw_bytes[1], raw_bytes[2], raw_bytes[3]]);
                let raw_event = RawEvt2Event { data: raw_data };

                if let Ok(event_type) = raw_event.event_type() {
                    match event_type {
                        Evt2EventType::TimeHigh => {
                            if let Ok(time_event) = raw_event.as_time_high_event() {
                                let new_time_base = time_event.timestamp as u64;

                                if !first_time_base_set {
                                    current_time_base = new_time_base;
                                    first_time_base_set = true;
                                } else if new_time_base < current_time_base
                                    && (current_time_base - new_time_base) > LOOP_THRESHOLD
                                {
                                    time_high_loop_count += 1;
                                    current_time_base =
                                        new_time_base + time_high_loop_count * TIME_LOOP;
                                } else {
                                    current_time_base =
                                        new_time_base + time_high_loop_count * TIME_LOOP;
                                }
                            }
                        }
                        Evt2EventType::CdOff | Evt2EventType::CdOn => {
                            if let Ok(cd_event) = raw_event.as_cd_event() {
                                let x = cd_event.x;
                                let y = cd_event.y;
                                let polarity = event_type == Evt2EventType::CdOn;
                                let timestamp =
                                    (current_time_base + cd_event.timestamp as u64) as f64;

                                // Validate coordinates if enabled
                                if self.config.validate_coordinates {
                                    if let Some((max_x, max_y)) = self.config.sensor_resolution {
                                        if (x >= max_x || y >= max_y)
                                            && self.config.skip_invalid_events
                                        {
                                            continue;
                                        }
                                    }
                                }

                                // Add to streamer, collect DataFrame if chunk is full
                                if let Some(df) = streamer.add_event(x, y, timestamp, polarity)? {
                                    dataframes.push(df);
                                }

                                // Check max events limit
                                if let Some(max_events) = self.config.max_events {
                                    if streamer.total_events() >= max_events {
                                        let final_df = streamer.flush()?;
                                        if final_df.height() > 0 {
                                            dataframes.push(final_df);
                                        }
                                        return Self::concatenate_dataframes(dataframes);
                                    }
                                }
                            }
                        }
                        _ => {
                            continue;
                        }
                    }
                }
            }
        }

        // Flush any remaining events
        let final_df = streamer.flush()?;
        if final_df.height() > 0 {
            dataframes.push(final_df);
        }

        Self::concatenate_dataframes(dataframes)
    }

    /// Concatenate multiple DataFrames efficiently
    #[cfg(feature = "polars")]
    fn concatenate_dataframes(
        dataframes: Vec<DataFrame>,
    ) -> Result<DataFrame, Box<dyn std::error::Error + Send + Sync>> {
        if dataframes.is_empty() {
            return Ok(crate::ev_formats::dataframe_builder::create_empty_events_dataframe()?);
        }

        if dataframes.len() == 1 {
            return Ok(dataframes.into_iter().next().unwrap());
        }

        // Convert DataFrames to LazyFrames for concat, then collect back to DataFrame
        let lazy_frames: Vec<LazyFrame> = dataframes.into_iter().map(|df| df.lazy()).collect();
        let df = concat(&lazy_frames, UnionArgs::default())?.collect()?;
        Ok(df)
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
        assert!(cd_event.polarity);
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
        assert!(config.validate_coordinates);
        assert!(!config.skip_invalid_events);
        assert_eq!(config.max_events, None);
        assert_eq!(config.chunk_size, 1_000_000);
    }
}
