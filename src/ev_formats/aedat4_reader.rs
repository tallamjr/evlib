//! AEDAT 4.0 (iniVation DV framework) reader.
//!
//! Implements the real on-disk format produced by the DV software framework, as
//! opposed to the invented "28-byte packet header" layout that the legacy
//! `aedat_reader` carried. The wire format is, in order:
//!
//! 1. A fixed 14-byte version line: `#!AER-DAT4.0\r\n`.
//! 2. A little-endian `u32` size prefix followed by an `IOHeader` FlatBuffer
//!    (file identifier `IOHE`). The header carries the compression type, the
//!    byte position of the trailing `FileDataTable`, and an XML `infoNode`
//!    string describing every stream (id, type identifier, output name,
//!    sensor resolution).
//! 3. A sequence of packets. Each packet is an 8-byte `PacketHeader`
//!    (`StreamID: i32`, `Size: i32`, both little-endian) followed by `Size`
//!    bytes of body. The body is a size-prefixed `EventPacket` FlatBuffer
//!    (file identifier `EVTS`), optionally LZ4-frame compressed.
//!
//! The `EventPacket` contains a vector of fixed 16-byte `Event` structs:
//! `timestamp: i64` (microseconds) at byte 0, `x: i16` at byte 8, `y: i16` at
//! byte 10, `polarity: bool` at byte 12 (remaining bytes are struct padding).
//!
//! Reference: `lib/dv-processing/include/dv-processing/io/reader.hpp` and the
//! FlatBuffer schemas under `lib/dv-processing/include/dv-processing/`.
//!
//! Only the FlatBuffer fields evlib actually consumes are parsed; the parsing is
//! hand-written against the (fixed, simple) schemas rather than relying on
//! `flatc`-generated code, to avoid coupling the crate to a specific
//! `flatbuffers` runtime version.

use crate::ev_formats::aedat_reader::{AedatConfig, AedatError, AedatMetadata, AedatVersion};
use crate::ev_formats::dataframe_builder::EventDataFrameBuilder;
use crate::ev_formats::EventFormat;
use polars::prelude::DataFrame;
use std::io::Read;

/// Length of the AEDAT 4.0 version line `#!AER-DAT4.0\r\n`.
const AEDAT4_VERSION_LENGTH: usize = 14;
/// The exact version line expected at the start of an AEDAT 4.0 file.
pub const AEDAT4_VERSION_LINE: &[u8] = b"#!AER-DAT4.0\r\n";
/// FlatBuffer file identifier for the `EventPacket` type.
const EVENT_PACKET_IDENTIFIER: &[u8; 4] = b"EVTS";
/// Size in bytes of a single `Event` struct (8 + 2 + 2 + 1 + 3 padding).
const EVENT_STRUCT_SIZE: usize = 16;
/// Size in bytes of a `PacketHeader` (`StreamID: i32` + `Size: i32`).
const PACKET_HEADER_SIZE: usize = 8;

/// Compression type used for packet bodies (mirrors the DV `CompressionType` enum).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompressionType {
    None,
    Lz4,
    Zstd,
}

impl CompressionType {
    fn from_enum(value: i32) -> Result<Self, AedatError> {
        match value {
            // NONE
            0 => Ok(CompressionType::None),
            // LZ4 and LZ4_HIGH both decode with the same LZ4 frame decoder.
            1 | 2 => Ok(CompressionType::Lz4),
            // ZSTD and ZSTD_HIGH both decode with the same Zstd decoder.
            3 | 4 => Ok(CompressionType::Zstd),
            other => Err(AedatError::CorruptedHeader(format!(
                "Unknown AEDAT 4.0 compression type: {other}"
            ))),
        }
    }
}

/// Description of one stream extracted from the IOHeader's XML info node.
#[derive(Debug, Clone)]
struct StreamInfo {
    id: i32,
    type_identifier: String,
    size_x: Option<u16>,
    size_y: Option<u16>,
}

/// Parsed IOHeader contents.
#[derive(Debug)]
struct IoHeader {
    compression: CompressionType,
    /// Byte position of the trailing FileDataTable, or `-1` if unknown. Packets
    /// occupy the bytes between the IOHeader and this position.
    data_table_position: i64,
    streams: Vec<StreamInfo>,
}

/// Read the entire AEDAT 4.0 file into events using the DV framework layout.
///
/// `data` must be the full file contents (the version line is re-validated here).
pub fn read_aedat_4_0(
    data: &[u8],
    config: &AedatConfig,
) -> Result<(DataFrame, AedatMetadata), AedatError> {
    if data.len() < AEDAT4_VERSION_LENGTH || &data[..AEDAT4_VERSION_LENGTH] != AEDAT4_VERSION_LINE {
        return Err(AedatError::InvalidVersion(
            "AEDAT 4.0: missing or invalid `#!AER-DAT4.0` version line".to_string(),
        ));
    }

    // IOHeader: u32 LE size prefix at offset 14, then that many bytes of FlatBuffer.
    let header_size_offset = AEDAT4_VERSION_LENGTH;
    let io_header_size = read_u32_le(data, header_size_offset)? as usize;
    let io_header_start = header_size_offset + 4;
    let io_header_end = io_header_start
        .checked_add(io_header_size)
        .ok_or_else(|| AedatError::CorruptedHeader("IOHeader size overflow".to_string()))?;
    if io_header_end > data.len() {
        return Err(AedatError::InsufficientData {
            expected: io_header_end,
            actual: data.len(),
        });
    }
    let io_header = parse_io_header(&data[io_header_start..io_header_end])?;

    // Determine which stream ids carry event data (type identifier "EVTS").
    let event_streams: Vec<&StreamInfo> = io_header
        .streams
        .iter()
        .filter(|s| s.type_identifier == "EVTS")
        .collect();

    let mut metadata = AedatMetadata {
        version: Some(AedatVersion::V4_0),
        ..Default::default()
    };
    metadata.header_size = io_header_end as u64;
    // Record sensor resolution from the first event stream that declares one.
    for stream in &event_streams {
        if let (Some(w), Some(h)) = (stream.size_x, stream.size_y) {
            metadata.sensor_resolution = Some((w, h));
            break;
        }
    }

    let event_stream_ids: std::collections::HashSet<i32> =
        event_streams.iter().map(|s| s.id).collect();

    let mut builder = EventDataFrameBuilder::new(EventFormat::AEDAT4, 100_000);
    let mut total_events = 0usize;

    // Packets occupy the bytes between the IOHeader and the trailing
    // FileDataTable. Use dataTablePosition as the upper bound when present so we
    // never misread the FileDataTable region as a packet header.
    let packet_region_end = match io_header.data_table_position {
        pos if pos >= 0 && (pos as usize) <= data.len() => pos as usize,
        _ => data.len(),
    };

    // Iterate packets sequentially from the end of the IOHeader to the data table.
    let mut cursor = io_header_end;
    while cursor + PACKET_HEADER_SIZE <= packet_region_end {
        let stream_id = read_i32_le(data, cursor)?;
        let packet_size = read_i32_le(data, cursor + 4)?;
        if packet_size < 0 {
            return Err(AedatError::InvalidBinaryData {
                offset: cursor as u64,
                message: format!("Negative packet size {packet_size}"),
            });
        }
        let body_start = cursor + PACKET_HEADER_SIZE;
        let body_end = body_start
            .checked_add(packet_size as usize)
            .ok_or_else(|| AedatError::CorruptedHeader("Packet size overflow".to_string()))?;
        if body_end > packet_region_end {
            // Truncated trailing packet (or we hit the FileDataTable region). Stop.
            break;
        }

        if event_stream_ids.contains(&stream_id) {
            let raw_body = &data[body_start..body_end];
            let decompressed = decompress(raw_body, io_header.compression)?;
            let added = parse_event_packet(&decompressed, config, &mut builder, total_events)?;
            total_events += added;
        }

        if let Some(max_events) = config.max_events {
            if total_events >= max_events {
                break;
            }
        }

        cursor = body_end;
    }

    metadata.event_count = Some(total_events);

    let events = builder.build().map_err(|e| AedatError::InvalidBinaryData {
        offset: 0,
        message: format!("Failed to build DataFrame: {e}"),
    })?;

    Ok((events, metadata))
}

/// Decompress a packet body according to the file's compression type.
fn decompress(body: &[u8], compression: CompressionType) -> Result<Vec<u8>, AedatError> {
    match compression {
        CompressionType::None => Ok(body.to_vec()),
        CompressionType::Lz4 => {
            let mut decoder = lz4_flex::frame::FrameDecoder::new(body);
            let mut out = Vec::new();
            decoder
                .read_to_end(&mut out)
                .map_err(|e| AedatError::InvalidBinaryData {
                    offset: 0,
                    message: format!("LZ4 frame decompression failed: {e}"),
                })?;
            Ok(out)
        }
        CompressionType::Zstd => Err(AedatError::CorruptedHeader(
            "AEDAT 4.0 Zstd-compressed packets are not yet supported".to_string(),
        )),
    }
}

/// Parse a size-prefixed `EventPacket` FlatBuffer, appending events to the builder.
///
/// Returns the number of events appended.
fn parse_event_packet(
    buffer: &[u8],
    config: &AedatConfig,
    builder: &mut EventDataFrameBuilder,
    events_so_far: usize,
) -> Result<usize, AedatError> {
    // Size-prefixed root: first 4 bytes are a u32 prefix; the FlatBuffer root
    // begins immediately after it.
    if buffer.len() < 8 {
        return Err(AedatError::InvalidBinaryData {
            offset: 0,
            message: "EventPacket buffer too small".to_string(),
        });
    }
    let root_base = 4usize;

    // The file identifier sits at bytes [root_base + 4 .. root_base + 8].
    if buffer.len() >= root_base + 8 {
        let ident = &buffer[root_base + 4..root_base + 8];
        if ident != EVENT_PACKET_IDENTIFIER {
            return Err(AedatError::InvalidBinaryData {
                offset: 0,
                message: format!("Unexpected FlatBuffer identifier {ident:?}, expected EVTS"),
            });
        }
    }

    // Root table offset (u32 LE relative to root_base).
    let root_table_offset = read_u32_le(buffer, root_base)? as usize;
    let root_table = root_base + root_table_offset;

    // vtable: signed offset stored at the start of the table; vtable = table - soffset.
    let soffset = read_i32_le(buffer, root_table)?;
    let vtable = (root_table as i64 - soffset as i64) as usize;
    let vtable_size = read_u16_le(buffer, vtable)? as usize;

    // EventPacket has a single field `elements` at vtable slot offset 4.
    // The vtable layout: [vtable_size(u16), table_size(u16), field0_voffset(u16), ...].
    let elements_slot = vtable + 4;
    if elements_slot + 2 > vtable + vtable_size {
        // Field not present; empty packet.
        return Ok(0);
    }
    let field_voffset = read_u16_le(buffer, elements_slot)? as usize;
    if field_voffset == 0 {
        return Ok(0);
    }

    // Field stores a u32 offset (relative to its own position) to the vector.
    let field_pos = root_table + field_voffset;
    let vector_offset = read_u32_le(buffer, field_pos)? as usize;
    let vector_pos = field_pos + vector_offset;

    // Vector layout: u32 length, then `length` inline Event structs.
    let count = read_u32_le(buffer, vector_pos)? as usize;
    let elements_start = vector_pos + 4;
    let elements_end = elements_start
        .checked_add(count.checked_mul(EVENT_STRUCT_SIZE).ok_or_else(|| {
            AedatError::CorruptedHeader("EventPacket element count overflow".to_string())
        })?)
        .ok_or_else(|| {
            AedatError::CorruptedHeader("EventPacket vector range overflow".to_string())
        })?;
    if elements_end > buffer.len() {
        return Err(AedatError::InsufficientData {
            expected: elements_end,
            actual: buffer.len(),
        });
    }

    let mut appended = 0usize;
    for i in 0..count {
        let base = elements_start + i * EVENT_STRUCT_SIZE;
        let timestamp = read_i64_le(buffer, base)?;
        let x = read_i16_le(buffer, base + 8)?;
        let y = read_i16_le(buffer, base + 10)?;
        let polarity = buffer[base + 12] != 0;

        if x < 0 || y < 0 {
            if config.skip_invalid_events {
                continue;
            }
            return Err(AedatError::InvalidBinaryData {
                offset: base as u64,
                message: format!("Negative event coordinate x={x}, y={y}"),
            });
        }
        let xu = x as u16;
        let yu = y as u16;

        if config.validate_coordinates {
            if let Some((max_x, max_y)) = config.max_resolution {
                if xu >= max_x || yu >= max_y {
                    if config.skip_invalid_events {
                        continue;
                    }
                    return Err(AedatError::CoordinateOutOfBounds {
                        event_index: events_so_far + appended,
                        x: xu,
                        y: yu,
                        max_x,
                        max_y,
                    });
                }
            }
        }

        // DV timestamps are exact integer microseconds (frequently absolute epoch
        // microseconds), so store them verbatim and bypass the magnitude heuristic.
        builder.add_event_microseconds(xu, yu, timestamp, polarity);
        appended += 1;
    }

    Ok(appended)
}

/// Parse the IOHeader FlatBuffer, extracting compression and the stream info.
fn parse_io_header(buffer: &[u8]) -> Result<IoHeader, AedatError> {
    // Non-size-prefixed root: u32 offset at byte 0.
    let root_table_offset = read_u32_le(buffer, 0)? as usize;
    let root_table = root_table_offset;

    let soffset = read_i32_le(buffer, root_table)?;
    let vtable = (root_table as i64 - soffset as i64) as usize;
    let vtable_size = read_u16_le(buffer, vtable)? as usize;

    // IOHeader fields: 0 compression (i32), 1 dataTablePosition (i64), 2 infoNode (string).
    // vtable slots start at vtable+4.
    let compression = {
        let slot = vtable + 4;
        if slot + 2 <= vtable + vtable_size {
            let voffset = read_u16_le(buffer, slot)? as usize;
            if voffset != 0 {
                CompressionType::from_enum(read_i32_le(buffer, root_table + voffset)?)?
            } else {
                CompressionType::None
            }
        } else {
            CompressionType::None
        }
    };

    let data_table_position = {
        let slot = vtable + 6; // field index 1 -> vtable offset 4 + 1*2
        if slot + 2 <= vtable + vtable_size {
            let voffset = read_u16_le(buffer, slot)? as usize;
            if voffset != 0 {
                read_i64_le(buffer, root_table + voffset)?
            } else {
                -1
            }
        } else {
            -1
        }
    };

    let info_node = {
        let slot = vtable + 8; // field index 2 -> vtable offset 4 + 2*2
        if slot + 2 <= vtable + vtable_size {
            let voffset = read_u16_le(buffer, slot)? as usize;
            if voffset != 0 {
                let field_pos = root_table + voffset;
                let str_offset = read_u32_le(buffer, field_pos)? as usize;
                let str_pos = field_pos + str_offset;
                let str_len = read_u32_le(buffer, str_pos)? as usize;
                let str_start = str_pos + 4;
                let str_end = str_start.checked_add(str_len).ok_or_else(|| {
                    AedatError::CorruptedHeader("infoNode string range overflow".to_string())
                })?;
                if str_end > buffer.len() {
                    return Err(AedatError::InsufficientData {
                        expected: str_end,
                        actual: buffer.len(),
                    });
                }
                String::from_utf8_lossy(&buffer[str_start..str_end]).into_owned()
            } else {
                String::new()
            }
        } else {
            String::new()
        }
    };

    let streams = parse_info_node(&info_node);

    Ok(IoHeader {
        compression,
        data_table_position,
        streams,
    })
}

/// Extract per-stream metadata from the IOHeader's XML `infoNode`.
///
/// The XML is a DV configuration tree. Each stream is a `<node name="<id>">`
/// directly under the `outInfo` node, carrying `typeIdentifier` and
/// `originalOutputName` attributes plus an `info` child with `sizeX`/`sizeY`.
/// A lightweight scan is sufficient and avoids pulling in an XML dependency.
fn parse_info_node(xml: &str) -> Vec<StreamInfo> {
    let mut streams = Vec::new();

    // Split into <node ...> opening tags and walk them in order. We track the
    // most recent numeric stream id so that the `info` child's sizeX/sizeY can
    // be attributed to it.
    let mut current: Option<StreamInfo> = None;

    for segment in xml.split('<') {
        let segment = segment.trim_start();
        if let Some(rest) = segment.strip_prefix("node ") {
            // A node opening tag. Extract its name attribute.
            if let Some(name) = extract_attr_value(rest, "name") {
                if let Ok(id) = name.parse::<i32>() {
                    // Numeric node name == stream id. Push any pending stream first.
                    if let Some(stream) = current.take() {
                        streams.push(stream);
                    }
                    current = Some(StreamInfo {
                        id,
                        type_identifier: String::new(),
                        size_x: None,
                        size_y: None,
                    });
                }
            }
        } else if let Some(rest) = segment.strip_prefix("attr ") {
            // An attribute entry: <attr key="..." type="...">value</attr>.
            if let (Some(key), Some(value)) =
                (extract_attr_value(rest, "key"), extract_tag_text(segment))
            {
                if let Some(stream) = current.as_mut() {
                    match key.as_str() {
                        "typeIdentifier" => stream.type_identifier = value,
                        "sizeX" => stream.size_x = value.parse().ok(),
                        "sizeY" => stream.size_y = value.parse().ok(),
                        _ => {}
                    }
                }
            }
        }
    }
    if let Some(stream) = current.take() {
        streams.push(stream);
    }

    streams
}

/// Extract the value of `key="..."` from an attribute fragment.
fn extract_attr_value(fragment: &str, key: &str) -> Option<String> {
    let needle = format!("{key}=\"");
    let start = fragment.find(&needle)? + needle.len();
    let end = fragment[start..].find('"')? + start;
    Some(fragment[start..end].to_string())
}

/// Extract the text content of a `<attr ...>text</attr>` fragment.
fn extract_tag_text(segment: &str) -> Option<String> {
    let close = segment.find('>')?;
    let after = &segment[close + 1..];
    // Text runs until the start of the closing tag (the next `<`, which split
    // already removed, so the remainder of this segment is the text).
    Some(after.trim().to_string())
}

// --- Little-endian primitive readers with bounds checks -------------------

fn read_u16_le(data: &[u8], offset: usize) -> Result<u16, AedatError> {
    let end = offset + 2;
    if end > data.len() {
        return Err(AedatError::InsufficientData {
            expected: end,
            actual: data.len(),
        });
    }
    Ok(u16::from_le_bytes([data[offset], data[offset + 1]]))
}

fn read_i16_le(data: &[u8], offset: usize) -> Result<i16, AedatError> {
    Ok(read_u16_le(data, offset)? as i16)
}

fn read_u32_le(data: &[u8], offset: usize) -> Result<u32, AedatError> {
    let end = offset + 4;
    if end > data.len() {
        return Err(AedatError::InsufficientData {
            expected: end,
            actual: data.len(),
        });
    }
    Ok(u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]))
}

fn read_i32_le(data: &[u8], offset: usize) -> Result<i32, AedatError> {
    Ok(read_u32_le(data, offset)? as i32)
}

fn read_i64_le(data: &[u8], offset: usize) -> Result<i64, AedatError> {
    let end = offset + 8;
    if end > data.len() {
        return Err(AedatError::InsufficientData {
            expected: end,
            actual: data.len(),
        });
    }
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&data[offset..end]);
    Ok(i64::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// Reference event counts and first events, cross-checked against
    /// dv-processing 2.0.3 (`MonoCameraRecording`).
    struct Sample {
        path: &'static str,
        events: usize,
        first_x: u16,
        first_y: u16,
        first_polarity: bool,
        first_timestamp: i64,
        max_x: u16,
        max_y: u16,
    }

    const SAMPLES: &[Sample] = &[
        Sample {
            path: "lib/dv-processing/tests/io/test_files/sample_data.aedat4",
            events: 9193,
            first_x: 56,
            first_y: 16,
            first_polarity: true,
            first_timestamp: 1_663_249_605_734_020,
            max_x: 346,
            max_y: 260,
        },
        Sample {
            path: "lib/dv-processing/python/tests/data/sample_data.aedat4",
            events: 9408,
            first_x: 64,
            first_y: 48,
            first_polarity: true,
            first_timestamp: 1,
            max_x: 640,
            max_y: 480,
        },
        Sample {
            path: "lib/dv-processing/tests/io/test_files/test-minimal.aedat4",
            events: 255_283,
            first_x: 185,
            first_y: 168,
            first_polarity: false,
            first_timestamp: 1_631_717_221_674_515,
            max_x: 640,
            max_y: 480,
        },
    ];

    fn read_first_event(df: &DataFrame) -> (u16, u16, i64, i8) {
        let x = df.column("x").unwrap().i16().unwrap();
        let y = df.column("y").unwrap().i16().unwrap();
        let t = df.column("t").unwrap().duration().unwrap();
        let p = df.column("polarity").unwrap().i8().unwrap();
        (
            x.get(0).unwrap() as u16,
            y.get(0).unwrap() as u16,
            t.get(0).unwrap(),
            p.get(0).unwrap(),
        )
    }

    #[test]
    fn reads_real_dv_samples() {
        for sample in SAMPLES {
            if !Path::new(sample.path).exists() {
                eprintln!("skipping absent sample {}", sample.path);
                continue;
            }
            let data = std::fs::read(sample.path).unwrap();
            // Use generous bounds so the loosely-bounded resolution does not
            // reject legitimate events; the per-sample assertions below verify
            // coordinates stay within sensor bounds.
            let config = AedatConfig {
                validate_timestamps: false,
                validate_coordinates: true,
                validate_polarity: false,
                skip_invalid_events: false,
                max_events: None,
                max_resolution: Some((sample.max_x, sample.max_y)),
            };
            let (df, metadata) = read_aedat_4_0(&data, &config).unwrap();

            assert_eq!(metadata.version, Some(AedatVersion::V4_0));
            assert_eq!(
                df.height(),
                sample.events,
                "event count mismatch for {}",
                sample.path
            );

            let (fx, fy, ft, fp) = read_first_event(&df);
            assert_eq!(fx, sample.first_x, "first x for {}", sample.path);
            assert_eq!(fy, sample.first_y, "first y for {}", sample.path);
            assert_eq!(ft, sample.first_timestamp, "first ts for {}", sample.path);
            let expected_pol: i8 = if sample.first_polarity { 1 } else { -1 };
            assert_eq!(fp, expected_pol, "first polarity for {}", sample.path);

            // All coordinates within sensor bounds; polarity strictly in {-1, 1}.
            let xs = df.column("x").unwrap().i16().unwrap();
            let ys = df.column("y").unwrap().i16().unwrap();
            let ps = df.column("polarity").unwrap().i8().unwrap();
            for i in 0..df.height() {
                let xv = xs.get(i).unwrap();
                let yv = ys.get(i).unwrap();
                let pv = ps.get(i).unwrap();
                assert!(xv >= 0 && (xv as u16) < sample.max_x);
                assert!(yv >= 0 && (yv as u16) < sample.max_y);
                assert!(pv == 1 || pv == -1, "polarity {pv} not in {{-1,1}}");
            }
        }
    }
}
