// Format detection system for event camera data formats
// Automatically detects the format of event data files using file extension and content analysis
//
// Supported formats:
// - Text: Plain text files with space-separated values
// - HDF5: Hierarchical Data Format 5 files
// - AER: Address Event Representation format (18-bit structure)
// - AEDAT: Multiple versions (1.0-4.0) with different structures
// - Binary: Raw binary event data
//
// References:
// - https://docs.prophesee.ai/stable/data/encoding_formats/aer.html
// - https://docs.inivation.com/software/software-advanced-usage/file-formats/
// - jAER documentation

use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

/// Supported event data formats
#[derive(Debug, Clone, PartialEq)]
pub enum EventFormat {
    /// Plain text format with space-separated values
    Text,
    /// HDF5 hierarchical data format
    HDF5,
    /// Address Event Representation (18-bit structure: 1 bit polarity + 9 bits x + 9 bits y)
    AER,
    /// AEDAT version 1.0: Optional header + [address, timestamp] pairs (6 bytes per event)
    AEDAT1,
    /// AEDAT version 2.0: Header line + 32-bit big-endian timestamp + address pairs
    AEDAT2,
    /// AEDAT version 3.1: Signed little-endian format
    AEDAT3,
    /// AEDAT version 4.0: DV framework format with packet structure
    AEDAT4,
    /// EVT2 format (Prophesee): Header + binary events
    EVT2,
    /// EVT3 format (Prophesee): Header + binary events (16-bit vectorized)
    EVT3,
    /// Raw binary format
    Binary,
    /// Unknown or unsupported format
    Unknown,
}

impl std::fmt::Display for EventFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EventFormat::Text => write!(f, "Text"),
            EventFormat::HDF5 => write!(f, "HDF5"),
            EventFormat::AER => write!(f, "AER"),
            EventFormat::AEDAT1 => write!(f, "AEDAT 1.0"),
            EventFormat::AEDAT2 => write!(f, "AEDAT 2.0"),
            EventFormat::AEDAT3 => write!(f, "AEDAT 3.1"),
            EventFormat::AEDAT4 => write!(f, "AEDAT 4.0"),
            EventFormat::EVT2 => write!(f, "EVT2"),
            EventFormat::EVT3 => write!(f, "EVT3"),
            EventFormat::Binary => write!(f, "Binary"),
            EventFormat::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Detection result with confidence score
#[derive(Debug, Clone)]
pub struct FormatDetectionResult {
    /// Detected format
    pub format: EventFormat,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Additional metadata about the detection
    pub metadata: FormatMetadata,
}

/// Metadata extracted during format detection
#[derive(Debug, Clone, Default)]
pub struct FormatMetadata {
    /// File size in bytes
    pub file_size: u64,
    /// Detected magic bytes or header signature
    pub magic_bytes: Option<Vec<u8>>,
    /// Header information for AEDAT files
    pub header_info: Option<String>,
    /// Estimated number of events (if determinable)
    pub estimated_event_count: Option<u64>,
    /// Sensor resolution (width, height) if found in header
    pub sensor_resolution: Option<(u16, u16)>,
    /// Additional format-specific properties
    pub properties: std::collections::HashMap<String, String>,
}

/// Errors that can occur during format detection
#[derive(Debug)]
pub enum FormatDetectionError {
    Io(std::io::Error),
    FileNotFound(String),
    EmptyFile,
    InsufficientData,
    InvalidPath,
}

impl std::fmt::Display for FormatDetectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FormatDetectionError::Io(e) => write!(f, "I/O error: {}", e),
            FormatDetectionError::FileNotFound(path) => write!(f, "File not found: {}", path),
            FormatDetectionError::EmptyFile => write!(f, "File is empty"),
            FormatDetectionError::InsufficientData => {
                write!(f, "Insufficient data to determine format")
            }
            FormatDetectionError::InvalidPath => write!(f, "Invalid file path"),
        }
    }
}

impl std::error::Error for FormatDetectionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            FormatDetectionError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for FormatDetectionError {
    fn from(error: std::io::Error) -> Self {
        FormatDetectionError::Io(error)
    }
}

/// Magic bytes for different formats
const HDF5_MAGIC: &[u8] = b"\x89HDF\r\n\x1a\n";
const AEDAT4_MAGIC: &[u8] = b"AEDAT4";
const AEDAT3_MAGIC: &[u8] = b"#!AER-DAT";
const AEDAT2_MAGIC: &[u8] = b"#!AER-DAT2.0";
const AEDAT1_MAGIC: &[u8] = b"#!AER-DAT1.0";
const EVT2_MAGIC: &[u8] = b"% evt 2.0";
const EVT3_MAGIC: &[u8] = b"% evt 3.0";

/// Main format detector struct
pub struct FormatDetector;

impl FormatDetector {
    /// Detect the format of an event data file
    ///
    /// # Arguments
    /// * `file_path` - Path to the file to analyze
    ///
    /// # Returns
    /// * `FormatDetectionResult` containing the detected format and confidence
    pub fn detect_format<P: AsRef<Path>>(
        file_path: P,
    ) -> Result<FormatDetectionResult, FormatDetectionError> {
        let path = file_path.as_ref();

        // Check if file exists
        if !path.exists() {
            return Err(FormatDetectionError::FileNotFound(
                path.to_string_lossy().to_string(),
            ));
        }

        // Get file metadata
        let metadata = std::fs::metadata(path)?;
        let file_size = metadata.len();

        // Check for empty file
        if file_size == 0 {
            return Err(FormatDetectionError::EmptyFile);
        }

        // Start with extension-based detection
        let extension_result = Self::detect_from_extension(path);

        // Perform content-based detection
        let content_result = Self::detect_from_content(path, file_size)?;

        // Combine results with weighted confidence
        let final_result = Self::combine_detection_results(extension_result, content_result);

        Ok(final_result)
    }

    /// Detect format from file extension
    fn detect_from_extension(path: &Path) -> FormatDetectionResult {
        let mut metadata = FormatMetadata::default();

        let (format, confidence) = match path.extension().and_then(|ext| ext.to_str()) {
            Some("txt") | Some("dat") => (EventFormat::Text, 0.7),
            Some("h5") | Some("hdf5") => (EventFormat::HDF5, 0.8),
            Some("aer") => (EventFormat::AER, 0.6),
            Some("aedat") => (EventFormat::AEDAT2, 0.5), // Default to AEDAT2, will be refined by content
            Some("raw") => (EventFormat::EVT2, 0.5), // Default to EVT2, will be refined by content
            Some("bin") => (EventFormat::Binary, 0.6),
            _ => (EventFormat::Unknown, 0.0),
        };

        metadata
            .properties
            .insert("detection_method".to_string(), "extension".to_string());

        FormatDetectionResult {
            format,
            confidence,
            metadata,
        }
    }

    /// Detect format from file content analysis
    fn detect_from_content(
        path: &Path,
        file_size: u64,
    ) -> Result<FormatDetectionResult, FormatDetectionError> {
        let mut file = File::open(path)?;
        let mut buffer = [0u8; 512]; // Read first 512 bytes for analysis
        let bytes_read = file.read(&mut buffer)?;

        if bytes_read < 8 {
            return Err(FormatDetectionError::InsufficientData);
        }

        let mut metadata = FormatMetadata {
            file_size,
            ..Default::default()
        };

        // Check for magic bytes
        if Self::starts_with(&buffer, HDF5_MAGIC) {
            metadata.magic_bytes = Some(HDF5_MAGIC.to_vec());
            metadata
                .properties
                .insert("detection_method".to_string(), "magic_bytes".to_string());
            return Ok(FormatDetectionResult {
                format: EventFormat::HDF5,
                confidence: 0.95,
                metadata,
            });
        }

        // Check for AEDAT formats
        if Self::starts_with(&buffer, AEDAT4_MAGIC) {
            return Self::analyze_aedat4_format(path, metadata);
        }

        if Self::starts_with(&buffer, AEDAT3_MAGIC) {
            return Self::analyze_aedat3_format(path, metadata);
        }

        if Self::starts_with(&buffer, AEDAT2_MAGIC) {
            return Self::analyze_aedat2_format(path, metadata);
        }

        if Self::starts_with(&buffer, AEDAT1_MAGIC) {
            return Self::analyze_aedat1_format(path, metadata);
        }

        // Check for EVT2 format
        if Self::contains_evt2_header(&buffer[..bytes_read]) {
            return Self::analyze_evt2_format(path, metadata);
        }

        // Check for EVT3 format - prioritize this over text detection
        if Self::contains_evt3_header(&buffer[..bytes_read]) {
            return Self::analyze_evt3_format(path, metadata);
        }

        // Check for text format - only after binary formats have been ruled out
        if Self::is_text_format(&buffer[..bytes_read]) {
            return Self::analyze_text_format(path, metadata);
        }

        // Check for AER format (no magic bytes, need to analyze structure)
        if Self::could_be_aer_format(&buffer[..bytes_read], file_size) {
            return Self::analyze_aer_format(path, metadata);
        }

        // Check for binary format
        if Self::could_be_binary_format(&buffer[..bytes_read], file_size) {
            return Self::analyze_binary_format(path, metadata);
        }

        // Unknown format
        metadata
            .properties
            .insert("detection_method".to_string(), "unknown".to_string());
        Ok(FormatDetectionResult {
            format: EventFormat::Unknown,
            confidence: 0.0,
            metadata,
        })
    }

    /// Check if buffer starts with given bytes
    fn starts_with(buffer: &[u8], pattern: &[u8]) -> bool {
        if buffer.len() < pattern.len() {
            return false;
        }
        &buffer[..pattern.len()] == pattern
    }

    /// Analyze AEDAT 4.0 format (DV framework)
    fn analyze_aedat4_format(
        path: &Path,
        mut metadata: FormatMetadata,
    ) -> Result<FormatDetectionResult, FormatDetectionError> {
        let mut file = File::open(path)?;
        let mut header_buffer = [0u8; 1024];
        let bytes_read = file.read(&mut header_buffer)?;

        metadata.magic_bytes = Some(AEDAT4_MAGIC.to_vec());
        metadata
            .properties
            .insert("detection_method".to_string(), "aedat4_header".to_string());

        // Parse DV framework header
        let header_str = String::from_utf8_lossy(&header_buffer[..bytes_read]);
        metadata.header_info = Some(header_str.lines().take(10).collect::<Vec<_>>().join("\n"));

        // Look for resolution information
        if let Some(resolution) = Self::extract_resolution_from_header(&header_str) {
            metadata.sensor_resolution = Some(resolution);
        }

        Ok(FormatDetectionResult {
            format: EventFormat::AEDAT4,
            confidence: 0.9,
            metadata,
        })
    }

    /// Analyze AEDAT 3.1 format (signed little-endian)
    fn analyze_aedat3_format(
        path: &Path,
        mut metadata: FormatMetadata,
    ) -> Result<FormatDetectionResult, FormatDetectionError> {
        let mut file = File::open(path)?;
        let mut header_buffer = [0u8; 1024];
        let bytes_read = file.read(&mut header_buffer)?;

        metadata.magic_bytes = Some(AEDAT3_MAGIC.to_vec());
        metadata
            .properties
            .insert("detection_method".to_string(), "aedat3_header".to_string());

        let header_str = String::from_utf8_lossy(&header_buffer[..bytes_read]);
        metadata.header_info = Some(header_str.lines().take(10).collect::<Vec<_>>().join("\n"));

        // Look for resolution information
        if let Some(resolution) = Self::extract_resolution_from_header(&header_str) {
            metadata.sensor_resolution = Some(resolution);
        }

        Ok(FormatDetectionResult {
            format: EventFormat::AEDAT3,
            confidence: 0.9,
            metadata,
        })
    }

    /// Analyze AEDAT 2.0 format (32-bit big-endian)
    fn analyze_aedat2_format(
        path: &Path,
        mut metadata: FormatMetadata,
    ) -> Result<FormatDetectionResult, FormatDetectionError> {
        let mut file = File::open(path)?;
        let mut header_buffer = [0u8; 1024];
        let bytes_read = file.read(&mut header_buffer)?;

        metadata.magic_bytes = Some(AEDAT2_MAGIC.to_vec());
        metadata
            .properties
            .insert("detection_method".to_string(), "aedat2_header".to_string());

        let header_str = String::from_utf8_lossy(&header_buffer[..bytes_read]);
        metadata.header_info = Some(header_str.lines().take(10).collect::<Vec<_>>().join("\n"));

        // Look for resolution information
        if let Some(resolution) = Self::extract_resolution_from_header(&header_str) {
            metadata.sensor_resolution = Some(resolution);
        }

        Ok(FormatDetectionResult {
            format: EventFormat::AEDAT2,
            confidence: 0.9,
            metadata,
        })
    }

    /// Analyze AEDAT 1.0 format (6 bytes per event)
    fn analyze_aedat1_format(
        path: &Path,
        mut metadata: FormatMetadata,
    ) -> Result<FormatDetectionResult, FormatDetectionError> {
        let mut file = File::open(path)?;
        let mut header_buffer = [0u8; 1024];
        let bytes_read = file.read(&mut header_buffer)?;

        metadata.magic_bytes = Some(AEDAT1_MAGIC.to_vec());
        metadata
            .properties
            .insert("detection_method".to_string(), "aedat1_header".to_string());

        let header_str = String::from_utf8_lossy(&header_buffer[..bytes_read]);
        metadata.header_info = Some(header_str.lines().take(10).collect::<Vec<_>>().join("\n"));

        // AEDAT 1.0 uses 6 bytes per event (4 bytes address + 2 bytes timestamp)
        let header_end = Self::find_header_end(&header_str);
        let data_size = metadata.file_size - header_end as u64;
        metadata.estimated_event_count = Some(data_size / 6);

        Ok(FormatDetectionResult {
            format: EventFormat::AEDAT1,
            confidence: 0.9,
            metadata,
        })
    }

    /// Check if content appears to be text format
    fn is_text_format(buffer: &[u8]) -> bool {
        // Don't treat files with binary format headers as text
        // Check for binary format headers first
        if Self::starts_with(buffer, EVT2_MAGIC) || Self::starts_with(buffer, EVT3_MAGIC) {
            return false;
        }

        // Check for other binary format indicators
        let content = String::from_utf8_lossy(buffer);

        // Explicitly exclude EVT format files
        if content.contains("% evt ") || content.contains("% format EVT") {
            return false; // This is likely a binary format with ASCII header
        }

        // Exclude files that contain the "% end" marker (typical of binary formats)
        if content.contains("% end") {
            return false;
        }

        // Check if most bytes are ASCII printable or whitespace
        let printable_count = buffer
            .iter()
            .filter(|&&b| b.is_ascii_graphic() || b.is_ascii_whitespace())
            .count();

        let ratio = printable_count as f64 / buffer.len() as f64;

        // Use a higher threshold for files that might be binary formats
        // This prevents EVT3 files from being detected as text
        ratio > 0.98
    }

    /// Analyze text format
    fn analyze_text_format(
        path: &Path,
        mut metadata: FormatMetadata,
    ) -> Result<FormatDetectionResult, FormatDetectionError> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut line = String::new();
        let mut line_count = 0;
        let mut valid_event_lines = 0;

        metadata
            .properties
            .insert("detection_method".to_string(), "text_analysis".to_string());

        // Analyze first few lines to determine format
        while line_count < 10 && reader.read_line(&mut line)? > 0 {
            line_count += 1;

            if line.trim().is_empty() || line.starts_with('#') {
                continue;
            }

            // Check if line contains space-separated numeric values
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                let mut valid_parts = 0;

                // Try to parse as numbers (t, x, y, p)
                if parts[0].parse::<f64>().is_ok() {
                    valid_parts += 1;
                }
                if parts[1].parse::<u16>().is_ok() {
                    valid_parts += 1;
                }
                if parts[2].parse::<u16>().is_ok() {
                    valid_parts += 1;
                }
                if parts[3].parse::<i8>().is_ok() {
                    valid_parts += 1;
                }

                if valid_parts >= 3 {
                    valid_event_lines += 1;
                }
            }

            line.clear();
        }

        let confidence = if valid_event_lines > 0 {
            (valid_event_lines as f64 / line_count as f64) * 0.8
        } else {
            0.3
        };

        Ok(FormatDetectionResult {
            format: EventFormat::Text,
            confidence,
            metadata,
        })
    }

    /// Check if content could be AER format
    fn could_be_aer_format(buffer: &[u8], file_size: u64) -> bool {
        // AER format typically has 18-bit events, often stored as 32-bit values
        // Check if file size is consistent with 4-byte events
        if file_size % 4 != 0 {
            return false;
        }

        // Check for patterns typical of AER data
        // Events should have reasonable coordinate values
        if buffer.len() >= 8 {
            let event1 = u32::from_le_bytes([buffer[0], buffer[1], buffer[2], buffer[3]]);
            let event2 = u32::from_le_bytes([buffer[4], buffer[5], buffer[6], buffer[7]]);

            // Extract x, y coordinates (assuming 9-bit each)
            let x1 = (event1 >> 1) & 0x1FF;
            let y1 = (event1 >> 10) & 0x1FF;
            let x2 = (event2 >> 1) & 0x1FF;
            let y2 = (event2 >> 10) & 0x1FF;

            // Check if coordinates are reasonable (< 1024 for typical sensors)
            return x1 < 1024 && y1 < 1024 && x2 < 1024 && y2 < 1024;
        }

        false
    }

    /// Analyze AER format
    fn analyze_aer_format(
        _path: &Path,
        mut metadata: FormatMetadata,
    ) -> Result<FormatDetectionResult, FormatDetectionError> {
        metadata
            .properties
            .insert("detection_method".to_string(), "aer_analysis".to_string());
        metadata
            .properties
            .insert("event_size".to_string(), "4".to_string());

        // Estimate event count (4 bytes per event)
        metadata.estimated_event_count = Some(metadata.file_size / 4);

        Ok(FormatDetectionResult {
            format: EventFormat::AER,
            confidence: 0.7,
            metadata,
        })
    }

    /// Check if content could be binary format
    fn could_be_binary_format(_buffer: &[u8], file_size: u64) -> bool {
        // Check if file size is consistent with Event struct size
        const EVENT_SIZE: u64 = 17; // f64(8) + u16(2) + u16(2) + i8(1) + padding

        if file_size % EVENT_SIZE == 0 {
            return true;
        }

        // Check for other common binary event sizes
        let common_sizes = [8, 12, 16, 20, 24];
        for &size in &common_sizes {
            if file_size % size == 0 {
                return true;
            }
        }

        false
    }

    /// Analyze binary format
    fn analyze_binary_format(
        _path: &Path,
        mut metadata: FormatMetadata,
    ) -> Result<FormatDetectionResult, FormatDetectionError> {
        metadata.properties.insert(
            "detection_method".to_string(),
            "binary_analysis".to_string(),
        );

        // Try to estimate event count based on common event sizes
        const EVENT_SIZE: u64 = 17; // Most likely size for Event struct
        metadata.estimated_event_count = Some(metadata.file_size / EVENT_SIZE);

        Ok(FormatDetectionResult {
            format: EventFormat::Binary,
            confidence: 0.6,
            metadata,
        })
    }

    /// Extract resolution from AEDAT header
    fn extract_resolution_from_header(header: &str) -> Option<(u16, u16)> {
        for line in header.lines() {
            if line.contains("sizeX") || line.contains("width") {
                if let Some(width) = Self::extract_number_from_line(line) {
                    // Look for height in subsequent lines
                    for height_line in header.lines() {
                        if height_line.contains("sizeY") || height_line.contains("height") {
                            if let Some(height) = Self::extract_number_from_line(height_line) {
                                return Some((width, height));
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Extract numeric value from a line
    fn extract_number_from_line(line: &str) -> Option<u16> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        for part in parts {
            if let Ok(num) = part.parse::<u16>() {
                return Some(num);
            }
        }
        None
    }

    /// Find the end of header section in AEDAT files
    fn find_header_end(header: &str) -> usize {
        let mut offset = 0;
        for line in header.lines() {
            offset += line.len() + 1; // +1 for newline
            if !line.starts_with('#') && !line.trim().is_empty() {
                break;
            }
        }
        offset
    }

    /// Combine extension and content detection results
    fn combine_detection_results(
        extension_result: FormatDetectionResult,
        content_result: FormatDetectionResult,
    ) -> FormatDetectionResult {
        // If content detection is confident, use it
        if content_result.confidence > 0.8 {
            return content_result;
        }

        // If extension and content agree, boost confidence
        if extension_result.format == content_result.format {
            let combined_confidence =
                (extension_result.confidence + content_result.confidence) / 2.0 * 1.2;
            let combined_confidence = combined_confidence.min(1.0);

            let mut combined_metadata = content_result.metadata.clone();
            combined_metadata.properties.insert(
                "detection_method".to_string(),
                "extension_and_content".to_string(),
            );

            return FormatDetectionResult {
                format: content_result.format,
                confidence: combined_confidence,
                metadata: combined_metadata,
            };
        }

        // If they disagree, prefer content detection if it's more confident
        if content_result.confidence > extension_result.confidence {
            content_result
        } else {
            extension_result
        }
    }

    /// Check if buffer contains EVT2 header
    fn contains_evt2_header(buffer: &[u8]) -> bool {
        let content = String::from_utf8_lossy(buffer);
        content.contains("% evt 2.0") || content.contains("% format EVT2")
    }

    /// Check if buffer contains EVT3 header
    fn contains_evt3_header(buffer: &[u8]) -> bool {
        // Check for EVT3 magic bytes first - this is the most reliable method
        if buffer.len() >= EVT3_MAGIC.len() && Self::starts_with(buffer, EVT3_MAGIC) {
            return true;
        }

        // Also check for EVT3 format string in the content
        let content = String::from_utf8_lossy(buffer);

        // Check for the version string
        if content.contains("% evt 3.0") {
            return true;
        }

        // Check for the format declaration
        if content.contains("% format EVT3") {
            return true;
        }

        false
    }

    /// Analyze EVT2 format (Prophesee)
    fn analyze_evt2_format(
        path: &Path,
        mut metadata: FormatMetadata,
    ) -> Result<FormatDetectionResult, FormatDetectionError> {
        let mut file = File::open(path)?;
        let mut header_buffer = [0u8; 2048]; // EVT2 headers can be longer
        let bytes_read = file.read(&mut header_buffer)?;

        metadata.magic_bytes = Some(EVT2_MAGIC.to_vec());
        metadata
            .properties
            .insert("detection_method".to_string(), "evt2_header".to_string());

        let header_str = String::from_utf8_lossy(&header_buffer[..bytes_read]);

        // Extract header information
        let header_end = header_str.find("% end").unwrap_or(header_str.len());
        let header_lines: Vec<&str> = header_str[..header_end].lines().collect();
        metadata.header_info = Some(header_lines.join("\n"));

        // Extract resolution from format line
        for line in header_lines {
            if line.contains("% format EVT2") {
                if let Some(width) = Self::extract_evt2_parameter(line, "width") {
                    if let Some(height) = Self::extract_evt2_parameter(line, "height") {
                        metadata.sensor_resolution = Some((width, height));
                    }
                }
            }
        }

        // Estimate event count based on remaining data
        let header_size = header_str.find("% end").map(|pos| pos + 5).unwrap_or(0); // +5 for "% end"
        let data_size = metadata.file_size - header_size as u64;
        metadata.estimated_event_count = Some(data_size / 8); // EVT2 typically uses 8 bytes per event

        Ok(FormatDetectionResult {
            format: EventFormat::EVT2,
            confidence: 0.95,
            metadata,
        })
    }

    /// Analyze EVT3 format (Prophesee)
    fn analyze_evt3_format(
        path: &Path,
        mut metadata: FormatMetadata,
    ) -> Result<FormatDetectionResult, FormatDetectionError> {
        let mut file = File::open(path)?;
        let mut header_buffer = [0u8; 2048]; // EVT3 headers can be similar to EVT2
        let bytes_read = file.read(&mut header_buffer)?;

        metadata.magic_bytes = Some(EVT3_MAGIC.to_vec());
        metadata
            .properties
            .insert("detection_method".to_string(), "evt3_header".to_string());

        let header_str = String::from_utf8_lossy(&header_buffer[..bytes_read]);

        // Extract header information
        let header_end = header_str.find("% end").unwrap_or(header_str.len());
        let header_lines: Vec<&str> = header_str[..header_end].lines().collect();
        metadata.header_info = Some(header_lines.join("\n"));

        // Extract resolution from format line
        for line in header_lines {
            if line.contains("% format EVT3") {
                if let Some(width) = Self::extract_evt2_parameter(line, "width") {
                    if let Some(height) = Self::extract_evt2_parameter(line, "height") {
                        metadata.sensor_resolution = Some((width, height));
                    }
                }
            }
        }

        // Estimate event count based on remaining data
        let header_size = header_str.find("% end").map(|pos| pos + 5).unwrap_or(0); // +5 for "% end"
        let data_size = metadata.file_size - header_size as u64;
        metadata.estimated_event_count = Some(data_size / 8); // EVT3 uses 8 bytes per event (4 words x 2 bytes)

        Ok(FormatDetectionResult {
            format: EventFormat::EVT3,
            confidence: 0.95,
            metadata,
        })
    }

    /// Extract parameter from EVT2 format line
    fn extract_evt2_parameter(line: &str, param: &str) -> Option<u16> {
        if let Some(start) = line.find(&format!("{}=", param)) {
            let value_start = start + param.len() + 1;
            if let Some(end) = line[value_start..]
                .find(';')
                .or_else(|| line[value_start..].find(' '))
            {
                let value_str = &line[value_start..value_start + end];
                return value_str.parse::<u16>().ok();
            } else {
                let value_str = &line[value_start..];
                return value_str.parse::<u16>().ok();
            }
        }
        None
    }

    /// Get format description for user display
    pub fn get_format_description(format: &EventFormat) -> &'static str {
        match format {
            EventFormat::Text => "Plain text format with space-separated values (t x y p)",
            EventFormat::HDF5 => "HDF5 hierarchical data format",
            EventFormat::AER => "Address Event Representation (18-bit structure)",
            EventFormat::AEDAT1 => "AEDAT 1.0 format (6 bytes per event)",
            EventFormat::AEDAT2 => "AEDAT 2.0 format (32-bit big-endian)",
            EventFormat::AEDAT3 => "AEDAT 3.1 format (signed little-endian)",
            EventFormat::AEDAT4 => "AEDAT 4.0 format (DV framework)",
            EventFormat::EVT2 => "EVT2 format (Prophesee binary events)",
            EventFormat::EVT3 => "EVT3 format (Prophesee vectorized binary events)",
            EventFormat::Binary => "Raw binary event data",
            EventFormat::Unknown => "Unknown or unsupported format",
        }
    }
}

/// Convenience function for detecting file format
pub fn detect_event_format<P: AsRef<Path>>(
    file_path: P,
) -> Result<FormatDetectionResult, FormatDetectionError> {
    FormatDetector::detect_format(file_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_detect_text_format() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.txt");

        let mut file = File::create(&file_path).unwrap();
        writeln!(file, "# timestamp x y polarity").unwrap();
        writeln!(file, "1.23456 100 200 1").unwrap();
        writeln!(file, "2.34567 150 250 -1").unwrap();

        let result = FormatDetector::detect_format(&file_path).unwrap();
        assert_eq!(result.format, EventFormat::Text);
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn test_detect_hdf5_format() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.h5");

        let mut file = File::create(&file_path).unwrap();
        file.write_all(HDF5_MAGIC).unwrap();
        file.write_all(&[0; 100]).unwrap(); // Some dummy data

        let result = FormatDetector::detect_format(&file_path).unwrap();
        assert_eq!(result.format, EventFormat::HDF5);
        assert!(result.confidence > 0.9);
    }

    #[test]
    fn test_detect_aedat4_format() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.aedat");

        let mut file = File::create(&file_path).unwrap();
        file.write_all(AEDAT4_MAGIC).unwrap();
        file.write_all(b"\n# DV format\n").unwrap();

        let result = FormatDetector::detect_format(&file_path).unwrap();
        assert_eq!(result.format, EventFormat::AEDAT4);
        assert!(result.confidence > 0.8);
    }

    #[test]
    fn test_empty_file() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("empty.txt");
        File::create(&file_path).unwrap();

        let result = FormatDetector::detect_format(&file_path);
        assert!(matches!(result, Err(FormatDetectionError::EmptyFile)));
    }

    #[test]
    fn test_nonexistent_file() {
        let result = FormatDetector::detect_format("nonexistent.txt");
        assert!(matches!(result, Err(FormatDetectionError::FileNotFound(_))));
    }

    #[test]
    fn test_extension_detection() {
        let temp_dir = TempDir::new().unwrap();

        // Test various extensions
        let test_cases = vec![
            ("test.h5", EventFormat::HDF5),
            ("test.hdf5", EventFormat::HDF5),
            ("test.txt", EventFormat::Text),
            ("test.aer", EventFormat::AER),
            ("test.bin", EventFormat::Binary),
        ];

        for (filename, expected_format) in test_cases {
            let file_path = temp_dir.path().join(filename);
            let mut file = File::create(&file_path).unwrap();
            file.write_all(b"dummy data").unwrap();

            let result = FormatDetector::detect_format(&file_path).unwrap();
            // Extension detection should at least suggest the right format
            assert_eq!(result.format, expected_format);
        }
    }
}
