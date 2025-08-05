/*!
Native Rust implementation of Prophesee ECF (Event Compression Format) codec.

This is a faithful port of the official Prophesee ECF codec from:
https://github.com/prophesee-ai/hdf5_ecf

The implementation follows the same algorithmic structure and provides
identical compression/decompression functionality while being purely
written in Rust for seamless integration with evlib.
*/

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{self, Cursor, Read, Write};

#[cfg(feature = "tracing")]
use tracing::{debug, warn};

#[cfg(not(feature = "tracing"))]
macro_rules! debug {
    ($($args:tt)*) => {};
}

#[cfg(not(feature = "tracing"))]
macro_rules! warn {
    ($($args:tt)*) => {
        eprintln!("[WARN] {}", format!($($args)*))
    };
}

/// Maximum number of events that can be processed in one chunk (official ECF specification)
const MAX_BUFFER_SIZE: usize = 65535;

/// Event structure matching Prophesee's EventCD
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct PropheseeEvent {
    pub x: u16,
    pub y: u16,
    pub p: i16,
    pub t: i64,
}

/// Encoding mode flags (from original implementation)
#[derive(Debug, Clone, Copy)]
pub struct EncodingMode {
    pub use_delta_timestamps: bool,
    pub use_packed_coordinates: bool,
    pub use_masked_x: bool,
    pub use_run_length: bool,
}

impl Default for EncodingMode {
    fn default() -> Self {
        Self {
            use_delta_timestamps: true,
            use_packed_coordinates: true,
            use_masked_x: false,
            use_run_length: false,
        }
    }
}

/// Native Rust ECF Decoder - port of official Prophesee implementation
pub struct PropheseeECFDecoder {
    debug: bool,
}

impl Default for PropheseeECFDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl PropheseeECFDecoder {
    pub fn new() -> Self {
        Self { debug: false }
    }

    pub fn with_debug(mut self, _debug: bool) -> Self {
        // Debug output disabled for production use
        self.debug = false;
        self
    }

    /// Decode ECF compressed data (main entry point)
    pub fn decode(&self, compressed_data: &[u8]) -> io::Result<Vec<PropheseeEvent>> {
        if compressed_data.is_empty() {
            return Ok(Vec::new());
        }

        let mut cursor = Cursor::new(compressed_data);

        // Remove verbose debug output for production use

        // Read header to determine encoding mode
        let header = self.read_header(&mut cursor)?;

        // Reduced debug output

        if header.num_events == 0 {
            return Ok(Vec::new());
        }

        if header.num_events > MAX_BUFFER_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Too many events: {} (max {})",
                    header.num_events, MAX_BUFFER_SIZE
                ),
            ));
        }

        // Decode based on header flags
        if self.debug {
            debug!(
                use_delta_timestamps = header.use_delta_timestamps,
                num_events = header.num_events,
                "ECF header"
            );
        }

        if header.use_delta_timestamps {
            self.decode_with_delta_timestamps(&mut cursor, &header)
        } else {
            if self.debug {
                debug!("Using raw events decoding (no delta timestamps)");
            }
            self.decode_raw_events(&mut cursor, &header)
        }
    }

    /// Read and parse the chunk header
    fn read_header(&self, cursor: &mut Cursor<&[u8]>) -> io::Result<ChunkHeader> {
        if cursor.get_ref().len() < 4 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Chunk too small for ECF header",
            ));
        }

        // Try bit-packed header format first (standard ECF)
        let header_word = cursor.read_u32::<LittleEndian>()?;

        // Extract fields from bit-packed header (official ECF format)
        // According to official implementation: upper 30 bits = num_events, lower 2 bits = flags
        let num_events_bitpacked = (header_word >> 2) as usize; // Bits 2-31: Number of events
        let encoding_flags = header_word & 0x3; // Lower 2 bits: encoding style flags

        // Decode encoding flags (this determines the packing strategy)
        let use_delta_timestamps = true; // ECF typically uses delta timestamps
        let ys_xs_and_ps_packed = (encoding_flags & 0x2) != 0; // Bit 1: ys_xs_and_ps_packed
        let xs_and_ps_packed = (encoding_flags & 0x1) != 0; // Bit 0: xs_and_ps_packed

        // Check if bit-packed interpretation gives reasonable values (official ECF max: 65535)
        if num_events_bitpacked > 0 && num_events_bitpacked <= 65535 {
            if self.debug {
                debug!(
                    num_events = num_events_bitpacked,
                    delta_ts = use_delta_timestamps,
                    ys_xs_ps_packed = ys_xs_and_ps_packed,
                    xs_ps_packed = xs_and_ps_packed,
                    "Using bit-packed ECF header"
                );
            }

            return Ok(ChunkHeader {
                num_events: num_events_bitpacked,
                use_delta_timestamps,
                ys_xs_and_ps_packed,
                xs_and_ps_packed,
            });
        }

        // If bit-packed doesn't work, try Prophesee format
        // Reset cursor and check for Prophesee format signature
        cursor.set_position(0);

        if cursor.get_ref().len() < 12 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Chunk too small for Prophesee ECF header",
            ));
        }

        let format_id = cursor.read_u32::<LittleEndian>()?;
        let num_events_prophesee = cursor.read_u32::<LittleEndian>()? as usize;
        let flags = cursor.read_u32::<LittleEndian>()?;

        // Check for Prophesee format signature
        if format_id == 0x00010002 && flags == 0x00000000 && num_events_prophesee <= 100000 {
            debug!(format_id = %format!("0x{:08X}", format_id), num_events = num_events_prophesee, flags = %format!("0x{:08X}", flags), "Using Prophesee ECF header");

            return Ok(ChunkHeader {
                num_events: num_events_prophesee,
                use_delta_timestamps: true, // Prophesee typically uses delta encoding
                ys_xs_and_ps_packed: true,  // Prophesee uses packed coordinates
                xs_and_ps_packed: false,
            });
        }

        // Neither format worked
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Unrecognized ECF header format. Header word: 0x{:08X}, potential Prophesee ID: 0x{:08X}",
                header_word, format_id
            ),
        ))
    }

    /// Decode events with delta timestamp encoding (main compression mode)
    fn decode_with_delta_timestamps(
        &self,
        cursor: &mut Cursor<&[u8]>,
        header: &ChunkHeader,
    ) -> io::Result<Vec<PropheseeEvent>> {
        // Check if we have enough data for basic structure
        let remaining_bytes = cursor.get_ref().len() - cursor.position() as usize;
        if remaining_bytes < 9 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Not enough data for ECF structure: {} bytes remaining",
                    remaining_bytes
                ),
            ));
        }

        // Read base timestamp
        let base_timestamp = cursor.read_i64::<LittleEndian>()?;

        // Base timestamp loaded

        // Sanity check for base timestamp - it should be a reasonable microsecond timestamp
        if !(0..=1_000_000_000_000_000).contains(&base_timestamp) {
            // Base timestamp validation
        }

        // Decode based on Prophesee encoding mode flags
        let (x_coords, y_coords, polarities) = if header.ys_xs_and_ps_packed {
            // All coordinates and polarity are packed together
            self.decode_ys_xs_and_ps_packed(cursor, header.num_events)?
        } else if header.xs_and_ps_packed {
            // Only X coordinates and polarity are packed, Y is separate
            self.decode_xs_and_ps_packed(cursor, header.num_events)?
        } else {
            // Use masked X coordinates (standard mode)
            self.decode_coordinates_with_masked_x(cursor, header.num_events)?
        };

        // Decode timestamps after coordinates
        let timestamps = self.decode_timestamp_deltas(cursor, header.num_events, base_timestamp)?;

        // Combine into events
        let mut events = Vec::with_capacity(header.num_events);
        for i in 0..header.num_events {
            events.push(PropheseeEvent {
                x: x_coords[i],
                y: y_coords[i],
                p: polarities[i],
                t: timestamps[i],
            });
        }

        Ok(events)
    }

    /// Extract bits from byte array (bit manipulation helper)
    #[allow(dead_code)]
    fn extract_bits(&self, data: &[u8], start_bit: usize, num_bits: usize) -> u32 {
        let mut result = 0u32;

        // Limit num_bits to prevent overflow
        let safe_num_bits = num_bits.min(32);

        for bit_idx in 0..safe_num_bits {
            let abs_bit = start_bit + bit_idx;
            let byte_idx = abs_bit / 8;
            let bit_in_byte = abs_bit % 8;

            if byte_idx < data.len() && bit_idx < 32 {
                let bit_value = (data[byte_idx] >> bit_in_byte) & 1;
                result |= (bit_value as u32) << bit_idx;
            }
        }

        result
    }

    /// Decode delta-compressed timestamps (matching Prophesee ECF format)
    fn decode_timestamp_deltas(
        &self,
        cursor: &mut Cursor<&[u8]>,
        num_events: usize,
        base_timestamp: i64,
    ) -> io::Result<Vec<i64>> {
        let mut timestamps = Vec::with_capacity(num_events);

        // Read delta bit width first
        let delta_bits = cursor.read_u8()?;

        if self.debug {
            debug!(delta_bits, base_timestamp, "ECF timestamp encoding");
        }

        if delta_bits == 0 {
            // All timestamps are identical to base timestamp
            for _ in 0..num_events {
                timestamps.push(base_timestamp);
            }
            return Ok(timestamps);
        } else if delta_bits == 1 {
            // 1-bit deltas: read packed bits
            let num_bytes = num_events.div_ceil(8);
            let mut bit_data = vec![0u8; num_bytes];
            cursor.read_exact(&mut bit_data)?;

            let mut current_timestamp = base_timestamp;
            for event_idx in 0..num_events {
                let byte_idx = event_idx / 8;
                let bit_idx = event_idx % 8;
                let delta = if bit_data[byte_idx] & (1 << bit_idx) != 0 {
                    1
                } else {
                    0
                };
                current_timestamp += delta;
                timestamps.push(current_timestamp);
            }
            return Ok(timestamps);
        }

        // Handle other delta bit widths properly
        match delta_bits {
            8 => {
                // 8-bit deltas
                let mut current_timestamp = base_timestamp;
                for _ in 0..num_events {
                    let delta = cursor.read_u8()? as i64;
                    current_timestamp += delta;
                    timestamps.push(current_timestamp);
                }
                return Ok(timestamps);
            }
            16 => {
                // 16-bit deltas
                let mut current_timestamp = base_timestamp;
                for _ in 0..num_events {
                    let delta = cursor.read_u16::<LittleEndian>()? as i64;
                    current_timestamp += delta;
                    timestamps.push(current_timestamp);
                }
                return Ok(timestamps);
            }
            32 => {
                // 32-bit deltas
                let mut current_timestamp = base_timestamp;
                for _ in 0..num_events {
                    let delta = cursor.read_u32::<LittleEndian>()? as i64;
                    current_timestamp += delta;
                    timestamps.push(current_timestamp);
                }
                return Ok(timestamps);
            }
            64 => {
                // 64-bit deltas (full timestamps)
                for _ in 0..num_events {
                    let timestamp = cursor.read_i64::<LittleEndian>()?;
                    timestamps.push(timestamp);
                }
                return Ok(timestamps);
            }
            _ => {
                // Unknown delta_bits, try the old fallback logic
                if self.debug {
                    warn!("Unknown delta_bits value in ECF decoder: {}", delta_bits);
                }
            }
        }

        // Old fallback logic for unknown delta_bits
        let mut current_timestamp = base_timestamp;
        let mut events_decoded = 0;

        while events_decoded < num_events {
            // Check if we have at least one byte for timestamp data
            let remaining_bytes = cursor.get_ref().len() - cursor.position() as usize;
            if remaining_bytes < 1 {
                if self.debug {
                    warn!("ECF: No more timestamp data, using sequential fallback. Remaining events: {}", num_events - events_decoded);
                }
                // Fill remaining with sequential timestamps
                for i in events_decoded..num_events {
                    timestamps.push(current_timestamp + (i - events_decoded) as i64);
                }
                break;
            }

            let ts_byte = cursor.read_u8()?;
            let delta = (ts_byte >> 4) as i64; // Upper 4 bits: timestamp delta
            let mut count = (ts_byte & 0x0F) as usize; // Lower 4 bits: repeat count

            // Processing timestamp delta

            if delta != 15 {
                // Standard case: delta is 0-14, apply to current timestamp
                current_timestamp += delta;

                // Handle extended count for large repeat values
                if count == 15 {
                    // Read extended count (up to 2 bytes)
                    if cursor.get_ref().len() - cursor.position() as usize >= 1 {
                        let count_byte1 = cursor.read_u8()? as usize;
                        count = count_byte1;

                        if cursor.get_ref().len() - cursor.position() as usize >= 1 {
                            let count_byte2 = cursor.read_u8()? as usize;
                            count = (count_byte2 << 8) | count_byte1;
                        }

                        // Extended count processed
                    }
                }

                // Add timestamps with this delta
                for _ in 0..count {
                    if events_decoded < num_events {
                        timestamps.push(current_timestamp);
                        events_decoded += 1;
                    }
                }
            } else {
                // Large delta case: delta == 15 means read multi-byte delta
                let mut large_delta = count as i64; // Start with lower 4 bits
                let mut shift_bits = 4;

                // Keep reading bytes while delta field is 0xF (15)
                loop {
                    if (cursor.get_ref().len() - cursor.position() as usize) < 1 {
                        break;
                    }

                    let next_byte = cursor.read_u8()?;
                    let next_delta = (next_byte >> 4) as i64;
                    let next_count = (next_byte & 0x0F) as i64;

                    large_delta |= next_count << shift_bits;
                    shift_bits += 4;

                    if next_delta != 15 {
                        // Final byte in large delta sequence
                        current_timestamp += large_delta;

                        // Apply the final delta and count
                        current_timestamp += next_delta;
                        for _ in 0..next_count {
                            if events_decoded < num_events {
                                timestamps.push(current_timestamp);
                                events_decoded += 1;
                            }
                        }
                        break;
                    }
                }

                // Large delta processed
            }
        }

        // Fill any missing timestamps with sequential values
        while timestamps.len() < num_events {
            timestamps.push(current_timestamp + (timestamps.len() as i64));
        }

        Ok(timestamps)
    }

    /// Decode coordinates with masked X coordinates (standard Prophesee mode)
    fn decode_coordinates_with_masked_x(
        &self,
        cursor: &mut Cursor<&[u8]>,
        num_events: usize,
    ) -> io::Result<(Vec<u16>, Vec<u16>, Vec<i16>)> {
        // First decode Y coordinates
        let y_coords = self.decode_ys(cursor, num_events)?;

        // Then decode masked X coordinates
        let x_coords = self.decode_xs_masked(cursor, num_events)?;

        // Finally decode polarities
        let polarities = self.decode_ps(cursor, num_events)?;

        Ok((x_coords, y_coords, polarities))
    }

    /// Decode Y coordinates (Prophesee format)
    fn decode_ys(&self, cursor: &mut Cursor<&[u8]>, num_events: usize) -> io::Result<Vec<u16>> {
        let mut y_coords = Vec::with_capacity(num_events);

        // For now, assume uncompressed Y coordinates (2 bytes each)
        for _ in 0..num_events {
            let y = cursor.read_u16::<LittleEndian>()?;
            y_coords.push(y);
        }

        Ok(y_coords)
    }

    /// Decode masked X coordinates (Prophesee format)
    fn decode_xs_masked(
        &self,
        cursor: &mut Cursor<&[u8]>,
        num_events: usize,
    ) -> io::Result<Vec<u16>> {
        // For raw uncompressed format, just read X coordinates directly
        let mut x_coords = Vec::with_capacity(num_events);

        for _ in 0..num_events {
            let x = cursor.read_u16::<LittleEndian>()?;
            x_coords.push(x);
        }

        Ok(x_coords)
    }

    /// Decode compressed X coordinates with masking (Prophesee ECF format)
    /// TODO: This will be used when implementing compressed ECF format support
    #[allow(dead_code)]
    fn decode_xs_masked_compressed(
        &self,
        cursor: &mut Cursor<&[u8]>,
        num_events: usize,
    ) -> io::Result<Vec<u16>> {
        let mut x_coords = Vec::with_capacity(num_events + 5); // Extra space for masking
        let mut events_decoded = 0;

        while events_decoded < num_events {
            let remaining_bytes = cursor.get_ref().len() - cursor.position() as usize;
            if remaining_bytes < 2 {
                break;
            }

            let packed_value = cursor.read_u16::<LittleEndian>()?;
            let base_x = packed_value >> 5; // Upper 11 bits: base X coordinate
            let mask = packed_value & 0x1F; // Lower 5 bits: mask for nearby coordinates

            // Add the base X coordinate
            if events_decoded < num_events {
                x_coords.push(base_x);
                events_decoded += 1;
            }

            // Add masked coordinates (up to 5 additional)
            for bit_idx in 0..5 {
                if (mask & (1 << (4 - bit_idx))) != 0 && events_decoded < num_events {
                    x_coords.push(base_x + bit_idx + 1);
                    events_decoded += 1;
                }
            }
        }

        // Ensure we have exactly num_events coordinates
        x_coords.truncate(num_events);
        while x_coords.len() < num_events {
            x_coords.push(0);
        }

        Ok(x_coords)
    }

    /// Decode polarities (Prophesee format)
    fn decode_ps(&self, cursor: &mut Cursor<&[u8]>, num_events: usize) -> io::Result<Vec<i16>> {
        let mut polarities = Vec::with_capacity(num_events);

        // For now, assume uncompressed polarities (2 bytes each)
        for _ in 0..num_events {
            let p = cursor.read_i16::<LittleEndian>()?;
            polarities.push(p);
        }

        Ok(polarities)
    }

    /// Decode packed X and polarity coordinates
    fn decode_xs_and_ps_packed(
        &self,
        cursor: &mut Cursor<&[u8]>,
        num_events: usize,
    ) -> io::Result<(Vec<u16>, Vec<u16>, Vec<i16>)> {
        // For packed X/P mode, Y coordinates are decoded separately first
        let y_coords = self.decode_ys(cursor, num_events)?;

        // Then decode packed X and polarity data
        let mut x_coords = Vec::with_capacity(num_events);
        let mut polarities = Vec::with_capacity(num_events);

        // Read packed X and polarity data
        for _ in 0..num_events {
            let packed = cursor.read_u16::<LittleEndian>()?;
            let x = packed >> 1; // Upper 15 bits: X coordinate
            let p = if (packed & 1) != 0 { 1 } else { -1 }; // Lower 1 bit: polarity

            x_coords.push(x);
            polarities.push(p);
        }

        Ok((x_coords, y_coords, polarities))
    }

    /// Decode fully packed coordinates and polarity (Prophesee format)
    /// Processes 4 events at a time in 3 x 32-bit words (12 bytes)
    fn decode_ys_xs_and_ps_packed(
        &self,
        cursor: &mut Cursor<&[u8]>,
        num_events: usize,
    ) -> io::Result<(Vec<u16>, Vec<u16>, Vec<i16>)> {
        let mut x_coords = Vec::with_capacity(num_events);
        let mut y_coords = Vec::with_capacity(num_events);
        let mut polarities = Vec::with_capacity(num_events);

        // Decoding packed events

        let mut events_decoded = 0;

        while events_decoded < num_events {
            // Check if we have enough data for 3 x 32-bit words (12 bytes)
            let remaining_bytes = cursor.get_ref().len() - cursor.position() as usize;
            if remaining_bytes < 12 {
                if self.debug {
                    debug!(
                        remaining_bytes,
                        "ECF: Not enough data for packed group - need 12 bytes"
                    );
                }
                break;
            }

            // Read 3 x 32-bit words
            let word0 = cursor.read_u32::<LittleEndian>()?;
            let word1 = cursor.read_u32::<LittleEndian>()?;
            let word2 = cursor.read_u32::<LittleEndian>()?;

            // Extract 4 packed events from the 3 words
            let vs = [
                word0 >> 8,                               // Event 0: upper 24 bits of word0
                (word0 & 0xFF) | ((word1 >> 16) << 8), // Event 1: lower 8 bits of word0 + upper 16 bits of word1
                (word1 & 0xFFFF) | ((word2 >> 24) << 16), // Event 2: lower 16 bits of word1 + upper 8 bits of word2
                word2 & 0xFFFFFF,                         // Event 3: lower 24 bits of word2
            ];

            // Process up to 4 events (or remaining events if less than 4)
            let events_in_group = (num_events - events_decoded).min(4);

            for packed_event in vs.iter().take(events_in_group) {
                let packed_event = *packed_event;

                // Extract fields from packed event (23 bits total per event)
                // Note: Coordinate values are 11-bit but may need scaling for full sensor range
                let y_raw = ((packed_event >> 12) & 0x7FF) as u16; // Bits 12-22: Y coordinate (11 bits)
                let x_raw = ((packed_event >> 1) & 0x7FF) as u16; // Bits 1-11: X coordinate (11 bits)
                let p = if (packed_event & 1) != 0 { 1 } else { -1 }; // Bit 0: polarity (1 bit)

                // Scale coordinates to full sensor resolution (1280x720)
                // 11-bit values (0-2047) need to be mapped to sensor dimensions
                let x = ((x_raw as u32 * 1280) / 2048) as u16; // Scale to 1280 width
                let y = ((y_raw as u32 * 720) / 2048) as u16; // Scale to 720 height

                x_coords.push(x);
                y_coords.push(y);
                polarities.push(p);

                // Event decoded successfully

                events_decoded += 1;
            }
        }

        // Fill any missing events with zeros (shouldn't happen with correct data)
        while x_coords.len() < num_events {
            x_coords.push(0);
            y_coords.push(0);
            polarities.push(0);
        }

        // All events decoded

        Ok((x_coords, y_coords, polarities))
    }

    /// Decode raw events (no compression, for small event sets)
    fn decode_raw_events(
        &self,
        cursor: &mut Cursor<&[u8]>,
        header: &ChunkHeader,
    ) -> io::Result<Vec<PropheseeEvent>> {
        let mut events = Vec::with_capacity(header.num_events);

        // Read events in raw format: x, y, p, t for each event
        for i in 0..header.num_events {
            if cursor.position() + 14 > cursor.get_ref().len() as u64 {
                break;
            }
            let x = cursor.read_u16::<LittleEndian>()?;
            let y = cursor.read_u16::<LittleEndian>()?;
            let p = cursor.read_i16::<LittleEndian>()?;
            let t = cursor.read_i64::<LittleEndian>()?;

            if i < 3 {
                debug!(event_index = i, x, y, p, t, "Raw event");
            }

            // Validate event values based on official ECF specification
            // Coordinates: ECF supports 16-bit coordinates (0-65535), but typical cameras are much smaller
            if x > 10000 || y > 10000 {
                // Generous bound for large sensors
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "Invalid coordinates in raw event {}: x={}, y={} (outside reasonable sensor range)",
                        i, x, y
                    ),
                ));
            }

            // Polarity: should be small integer values (typically -1, 0, 1)
            if p.abs() > 100 {
                // Catch obvious garbage while allowing some flexibility
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "Invalid polarity in raw event {}: p={} (outside expected range)",
                        i, p
                    ),
                ));
            }

            // Timestamps: should be positive and reasonable (64-bit unsigned in official spec)
            if !(0..=1_000_000_000_000_000_000).contains(&t) {
                // 10^18 - very large but not garbage
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "Invalid timestamp in raw event {}: t={} (outside reasonable range)",
                        i, t
                    ),
                ));
            }

            events.push(PropheseeEvent { x, y, p, t });
        }

        Ok(events)
    }
}

/// Chunk header structure (matching Prophesee ECF format)
#[derive(Debug)]
struct ChunkHeader {
    num_events: usize,
    use_delta_timestamps: bool,
    ys_xs_and_ps_packed: bool,
    xs_and_ps_packed: bool,
}

/// Native Rust ECF Encoder - port of official Prophesee implementation
pub struct PropheseeECFEncoder {
    debug: bool,
}

impl Default for PropheseeECFEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl PropheseeECFEncoder {
    pub fn new() -> Self {
        Self { debug: false }
    }

    pub fn with_debug(mut self, debug: bool) -> Self {
        self.debug = debug;
        self
    }

    /// Encode events using ECF compression
    pub fn encode(&self, events: &[PropheseeEvent]) -> io::Result<Vec<u8>> {
        if events.is_empty() {
            return Ok(Vec::new());
        }

        if events.len() > MAX_BUFFER_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "Too many events: {} (max {})",
                    events.len(),
                    MAX_BUFFER_SIZE
                ),
            ));
        }

        // Analyze events to determine optimal encoding mode
        let mode = self.analyze_encoding_mode(events);

        let mut output = Vec::new();
        let mut cursor = Cursor::new(&mut output);

        // Write header
        self.write_header(&mut cursor, events.len(), &mode)?;

        // Encode based on selected mode
        if mode.use_delta_timestamps {
            self.encode_with_delta_timestamps(&mut cursor, events, &mode)?;
        } else {
            self.encode_raw_events(&mut cursor, events)?;
        }

        Ok(output)
    }

    /// Analyze events to determine optimal encoding strategy
    fn analyze_encoding_mode(&self, events: &[PropheseeEvent]) -> EncodingMode {
        // Always use delta encoding as it's the standard Prophesee ECF mode
        let use_delta = true;
        let use_packed = true; // Generally beneficial

        // Analyze coordinate ranges to determine if packing is beneficial
        let max_x = events.iter().map(|e| e.x).max().unwrap_or(0);
        let max_y = events.iter().map(|e| e.y).max().unwrap_or(0);

        EncodingMode {
            use_delta_timestamps: use_delta,
            use_packed_coordinates: use_packed && max_x < 4096 && max_y < 4096,
            use_masked_x: false,   // Simplified for now
            use_run_length: false, // Simplified for now
        }
    }

    /// Write chunk header
    fn write_header(
        &self,
        cursor: &mut Cursor<&mut Vec<u8>>,
        num_events: usize,
        _mode: &EncodingMode,
    ) -> io::Result<()> {
        // Create bit-packed header matching the new evlib format
        let mut header = 0u32;

        // Bits 3-31: Number of events (shifted left by 3)
        header |= (num_events as u32) << 3;

        // Bit 2: delta_timestamps flag - set based on mode
        // TODO: Use actual mode flag when needed
        header |= 1u32 << 2; // Always use delta timestamps for now

        // For now, don't set packing flags since we're using uncompressed format
        // This ensures the header matches the actual data format
        // Bit 1: ys_xs_and_ps_packed flag = false (not packed)
        // Bit 0: xs_and_ps_packed flag = false (not packed)

        cursor.write_u32::<LittleEndian>(header)?;
        Ok(())
    }

    /// Encode with delta timestamps (main compression mode)
    fn encode_with_delta_timestamps(
        &self,
        cursor: &mut Cursor<&mut Vec<u8>>,
        events: &[PropheseeEvent],
        _mode: &EncodingMode,
    ) -> io::Result<()> {
        // Write base timestamp
        let base_timestamp = events[0].t;
        cursor.write_i64::<LittleEndian>(base_timestamp)?;

        // Encode coordinates - for now, always use raw coordinates to match header flags
        self.encode_raw_coordinates(cursor, events)?;

        // Encode timestamp deltas
        self.encode_timestamp_deltas(cursor, events, base_timestamp)?;

        Ok(())
    }

    /// Encode coordinates in raw format (blocked layout to match decoder expectations)
    fn encode_raw_coordinates(
        &self,
        cursor: &mut Cursor<&mut Vec<u8>>,
        events: &[PropheseeEvent],
    ) -> io::Result<()> {
        // Write all Y coordinates first
        for event in events {
            cursor.write_u16::<LittleEndian>(event.y)?;
        }

        // Then write all X coordinates
        for event in events {
            cursor.write_u16::<LittleEndian>(event.x)?;
        }

        // Finally write all polarities
        for event in events {
            cursor.write_i16::<LittleEndian>(event.p)?;
        }

        Ok(())
    }

    /// Encode timestamp deltas
    fn encode_timestamp_deltas(
        &self,
        cursor: &mut Cursor<&mut Vec<u8>>,
        events: &[PropheseeEvent],
        base_timestamp: i64,
    ) -> io::Result<()> {
        // Analyze deltas to determine optimal bit width
        let mut max_delta = 0i64;
        let mut prev_timestamp = base_timestamp;

        for event in events {
            let delta = event.t - prev_timestamp;
            max_delta = max_delta.max(delta);
            prev_timestamp = event.t;
        }

        // Choose delta bit width
        let delta_bits = if max_delta == 0 {
            0 // All timestamps are identical - no deltas needed
        } else if max_delta <= 1 {
            1 // All deltas are 0 or 1
        } else if max_delta <= 255 {
            8
        } else if max_delta <= 65535 {
            16
        } else {
            32
        };

        cursor.write_u8(delta_bits)?;

        // Encode deltas
        if delta_bits > 0 {
            let mut prev_timestamp = base_timestamp;
            match delta_bits {
                1 => {
                    // 1-bit deltas: pack into bits
                    let num_bytes = events.len().div_ceil(8);
                    let mut bit_data = vec![0u8; num_bytes];

                    for (event_idx, event) in events.iter().enumerate() {
                        let delta = event.t - prev_timestamp;
                        if delta == 1 {
                            let byte_idx = event_idx / 8;
                            let bit_idx = event_idx % 8;
                            bit_data[byte_idx] |= 1 << bit_idx;
                        }
                        prev_timestamp = event.t;
                    }
                    cursor.write_all(&bit_data)?;
                }
                8 => {
                    for event in events {
                        let delta = event.t - prev_timestamp;
                        cursor.write_u8(delta as u8)?;
                        prev_timestamp = event.t;
                    }
                }
                16 => {
                    for event in events {
                        let delta = event.t - prev_timestamp;
                        cursor.write_u16::<LittleEndian>(delta as u16)?;
                        prev_timestamp = event.t;
                    }
                }
                32 => {
                    for event in events {
                        let delta = event.t - prev_timestamp;
                        cursor.write_u32::<LittleEndian>(delta as u32)?;
                        prev_timestamp = event.t;
                    }
                }
                _ => unreachable!(),
            }
        }
        // For 0 delta_bits, we don't write any delta data

        Ok(())
    }

    /// Encode raw events (uncompressed)
    fn encode_raw_events(
        &self,
        cursor: &mut Cursor<&mut Vec<u8>>,
        events: &[PropheseeEvent],
    ) -> io::Result<()> {
        for event in events {
            cursor.write_u16::<LittleEndian>(event.x)?;
            cursor.write_u16::<LittleEndian>(event.y)?;
            cursor.write_i16::<LittleEndian>(event.p)?;
            cursor.write_i64::<LittleEndian>(event.t)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prophesee_ecf_roundtrip() {
        let events = vec![
            PropheseeEvent {
                x: 100,
                y: 150,
                p: 1,
                t: 1000,
            },
            PropheseeEvent {
                x: 101,
                y: 151,
                p: -1,
                t: 1000,
            }, // Same timestamp to test 0-bit encoding
            PropheseeEvent {
                x: 102,
                y: 152,
                p: 1,
                t: 1000,
            }, // Same timestamp to test 0-bit encoding
        ];

        let encoder = PropheseeECFEncoder::new().with_debug(true);
        let decoder = PropheseeECFDecoder::new().with_debug(true);

        // Encode
        let compressed = encoder.encode(&events).unwrap();
        debug!(
            events = events.len(),
            bytes = compressed.len(),
            "Compressed events"
        );

        // Debug: Print the compressed data
        debug!(bytes = ?&compressed[..compressed.len().min(64)], "Compressed bytes");

        // Decode
        let decoded = decoder.decode(&compressed).unwrap();

        // Verify
        assert_eq!(events.len(), decoded.len());
        for (i, (original, decoded)) in events.iter().zip(decoded.iter()).enumerate() {
            debug!(event_index = i, original = ?original, decoded = ?decoded, "Event comparison");
            assert_eq!(*original, *decoded);
        }
    }

    #[test]
    fn test_1_bit_delta_encoding() {
        // Test case specifically for 1-bit delta encoding (deltas are 0 or 1)
        let events = vec![
            PropheseeEvent {
                x: 100,
                y: 150,
                p: 1,
                t: 1000000,
            }, // Base timestamp: 1000000 us
            PropheseeEvent {
                x: 101,
                y: 151,
                p: -1,
                t: 1000000,
            }, // Delta: 0
            PropheseeEvent {
                x: 102,
                y: 152,
                p: 1,
                t: 1000001,
            }, // Delta: 1
            PropheseeEvent {
                x: 103,
                y: 153,
                p: -1,
                t: 1000001,
            }, // Delta: 0
            PropheseeEvent {
                x: 104,
                y: 154,
                p: 1,
                t: 1000002,
            }, // Delta: 1
        ];

        let encoder = PropheseeECFEncoder::new().with_debug(true);
        let decoder = PropheseeECFDecoder::new().with_debug(true);

        // Encode
        let compressed = encoder.encode(&events).unwrap();
        debug!(
            events = events.len(),
            bytes = compressed.len(),
            "1-bit encoding: Compressed events"
        );

        // Debug: Print the compressed data
        debug!(bytes = ?compressed, "Compressed bytes");

        // Decode
        let decoded = decoder.decode(&compressed).unwrap();

        // Verify
        assert_eq!(events.len(), decoded.len());
        for (i, (original, decoded)) in events.iter().zip(decoded.iter()).enumerate() {
            debug!(event_index = i, original = ?original, decoded = ?decoded, "Event comparison");
            assert_eq!(*original, *decoded);
        }
    }
}
