/*!
Native Rust implementation of Prophesee ECF (Event Compression Format) codec.

This is a faithful port of the official Prophesee ECF codec from:
https://github.com/prophesee-ai/hdf5_ecf (see `lib/hdf5_ecf/ecf_codec.cpp`).

A chunk is laid out as:

```text
[u32 header] [timestamp section] [coordinate section]
```

The header packs the event count in bits 2-31, the `ys_xs_and_ps_packed` flag
in bit 1 and the `xs_and_ps_packed` flag in bit 0. The decoder always reads the
timestamp section first (an absolute origin followed by a nibble run-length
delta stream) and only then the coordinate section, whose layout depends on the
two packing flags. Getting this order wrong misaligns the byte cursor and
produces garbage coordinates and timestamps, so the ordering below mirrors the
reference decoder exactly.
*/

use std::io;

/// Maximum number of events that can be processed in one chunk (official ECF specification)
const MAX_BUFFER_SIZE: usize = 65535;

/// Decoded X, Y and polarity columns plus the number of bytes consumed.
type PackedColumns = (Vec<u16>, Vec<u16>, Vec<i16>, usize);

/// Event structure matching Prophesee's EventCD
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct PropheseeEvent {
    pub x: u16,
    pub y: u16,
    pub p: i16,
    pub t: i64,
}

fn unexpected_eof(what: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::UnexpectedEof,
        format!("ECF chunk truncated while reading {}", what),
    )
}

#[inline]
fn rd_u8(data: &[u8], pos: usize) -> io::Result<u8> {
    data.get(pos).copied().ok_or_else(|| unexpected_eof("byte"))
}

#[inline]
fn rd_u16(data: &[u8], pos: usize) -> io::Result<u16> {
    if pos + 2 > data.len() {
        return Err(unexpected_eof("u16"));
    }
    Ok(u16::from_le_bytes([data[pos], data[pos + 1]]))
}

#[inline]
fn rd_u32(data: &[u8], pos: usize) -> io::Result<u32> {
    if pos + 4 > data.len() {
        return Err(unexpected_eof("u32"));
    }
    Ok(u32::from_le_bytes([
        data[pos],
        data[pos + 1],
        data[pos + 2],
        data[pos + 3],
    ]))
}

#[inline]
fn rd_u64(data: &[u8], pos: usize) -> io::Result<u64> {
    if pos + 8 > data.len() {
        return Err(unexpected_eof("u64"));
    }
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&data[pos..pos + 8]);
    Ok(u64::from_le_bytes(buf))
}

/// Native Rust ECF Decoder: faithful port of the official Prophesee decoder.
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

    pub fn with_debug(mut self, debug: bool) -> Self {
        self.debug = debug;
        self
    }

    /// Decode ECF compressed data (main entry point).
    pub fn decode(&self, compressed_data: &[u8]) -> io::Result<Vec<PropheseeEvent>> {
        if compressed_data.is_empty() {
            return Ok(Vec::new());
        }
        if compressed_data.len() < 4 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "ECF chunk too small for header",
            ));
        }

        let header = rd_u32(compressed_data, 0)?;
        let num_events = (header >> 2) as usize;
        let ys_xs_and_ps_packed = (header >> 1) & 1 != 0;
        let xs_and_ps_packed = header & 1 != 0;

        if num_events == 0 {
            return Ok(Vec::new());
        }
        if num_events > MAX_BUFFER_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Too many events: {} (max {})", num_events, MAX_BUFFER_SIZE),
            ));
        }

        // The chunk body starts after the 4-byte header.
        let body = &compressed_data[4..];
        let mut pos = 0usize;

        // Timestamps are decoded first, exactly as in the reference decoder.
        let (timestamps, consumed) = Self::decode_ts(&body[pos..], num_events)?;
        pos += consumed;

        // Coordinate section, dispatched on the header packing flags.
        let (x_coords, y_coords, polarities) = if ys_xs_and_ps_packed {
            let (xs, ys, ps, _consumed) =
                Self::decode_ys_xs_and_ps_packed(&body[pos..], num_events)?;
            (xs, ys, ps)
        } else {
            let (ys, consumed_y) = Self::decode_ys(&body[pos..], num_events)?;
            pos += consumed_y;
            if xs_and_ps_packed {
                let (xs, ps, _consumed) = Self::decode_xs_and_ps_packed(&body[pos..], num_events)?;
                (xs, ys, ps)
            } else {
                let (xs, consumed_x) = Self::decode_xs_masked(&body[pos..], num_events)?;
                pos += consumed_x;
                let (ps, _consumed_p) = Self::decode_ps(&body[pos..], num_events)?;
                (xs, ys, ps)
            }
        };

        let events = (0..num_events)
            .map(|i| PropheseeEvent {
                x: x_coords[i],
                y: y_coords[i],
                p: polarities[i],
                t: timestamps[i],
            })
            .collect();

        Ok(events)
    }

    /// Decode the timestamp section: an 8-byte absolute origin followed by a
    /// nibble-based run-length delta stream. Returns the timestamps and the
    /// number of bytes consumed.
    fn decode_ts(data: &[u8], num_events: usize) -> io::Result<(Vec<i64>, usize)> {
        let t0 = rd_u64(data, 0)?;
        let mut timestamps = Vec::with_capacity(num_events);
        let mut cur_t = t0;
        let mut pos = 8usize;
        let mut i = 0usize;

        while i < num_events {
            let v = rd_u8(data, pos)?;
            let t = v >> 4;
            let mut c = (v & 0x0f) as usize;

            if t != 0x0f {
                cur_t = cur_t.wrapping_add(t as u64);
                if c == 0x0f {
                    // Extended repeat count: two following bytes, little-endian.
                    let c0 = rd_u8(data, pos + 1)? as usize;
                    let c1 = rd_u8(data, pos + 2)? as usize;
                    c = (c1 << 8) | c0;
                    pos += 2;
                }
                for _ in 0..c {
                    if i < num_events {
                        timestamps.push(cur_t as i64);
                        i += 1;
                    }
                }
                pos += 1;
            } else {
                // Multi-nibble large delta: accumulate nibbles from consecutive
                // 0xF-prefixed bytes until a non-0xF byte terminates the run. The
                // terminating byte is left in place to be re-read as a normal
                // (delta, count) entry by the next loop iteration.
                let mut dt: u64 = 0;
                let mut shift = 0u32;
                let mut cur_c = c;
                loop {
                    dt |= (cur_c as u64) << (4 * shift);
                    shift += 1;
                    pos += 1;
                    let nv = rd_u8(data, pos)?;
                    let nt = nv >> 4;
                    cur_c = (nv & 0x0f) as usize;
                    if nt != 0x0f {
                        break;
                    }
                }
                cur_t = cur_t.wrapping_add(dt);
            }
        }

        Ok((timestamps, pos))
    }

    /// Decode run-length encoded Y coordinates.
    fn decode_ys(data: &[u8], num_events: usize) -> io::Result<(Vec<u16>, usize)> {
        let mut ys = vec![0u16; num_events];
        let mut pos = 0usize;
        let mut i = 0usize;

        while i < num_events {
            let v = rd_u16(data, pos)?;
            pos += 2;
            let y = v >> 5;
            let mut c = (v & 0b11111) as usize;
            if c == 0b11111 {
                c = rd_u16(data, pos)? as usize;
                pos += 2;
            }
            for _ in 0..c {
                if i < num_events {
                    ys[i] = y;
                }
                i += 1;
            }
        }

        Ok((ys, pos))
    }

    /// Decode run-length encoded polarities.
    fn decode_ps(data: &[u8], num_events: usize) -> io::Result<(Vec<i16>, usize)> {
        let mut ps = vec![0i16; num_events];
        let mut pos = 0usize;
        let mut i = 0usize;

        while i < num_events {
            let v = rd_u8(data, pos)?;
            pos += 1;
            let p_bit = v >> 7;
            let mut c = (v & 0b1111111) as usize;
            if c == 0b1111111 {
                let c0 = rd_u8(data, pos)? as usize;
                let c1 = rd_u8(data, pos + 1)? as usize;
                c = (c1 << 8) | c0;
                pos += 2;
            }
            let polarity: i16 = if p_bit != 0 { 1 } else { -1 };
            for _ in 0..c {
                if i < num_events {
                    ps[i] = polarity;
                }
                i += 1;
            }
        }

        Ok((ps, pos))
    }

    /// Decode X coordinates in the masked layout, where each 16-bit word stores a
    /// base X (11 bits) plus a 5-bit mask selecting nearby follow-on coordinates.
    fn decode_xs_masked(data: &[u8], num_events: usize) -> io::Result<(Vec<u16>, usize)> {
        // Allow a few extra slots so the last group can overshoot without a
        // bounds check, matching the reference which over-allocates by 5.
        let mut xs = vec![0u16; num_events + 5];
        let mut pos = 0usize;
        let mut i = 0usize;

        while i < num_events {
            let v = rd_u16(data, pos)?;
            pos += 2;
            let x = v >> 5;
            let mask = v & 0b11111;

            if i < xs.len() {
                xs[i] = x;
            }
            i += 1;

            for j in 0..5u16 {
                if (mask & (1 << (4 - j))) != 0 {
                    if i < xs.len() {
                        xs[i] = x + j + 1;
                    }
                    i += 1;
                }
            }
        }

        xs.truncate(num_events);
        Ok((xs, pos))
    }

    /// Decode X coordinates and polarity packed four events into three 16-bit
    /// words (12 bits per event: 11-bit X plus 1-bit polarity).
    fn decode_xs_and_ps_packed(
        data: &[u8],
        num_events: usize,
    ) -> io::Result<(Vec<u16>, Vec<i16>, usize)> {
        let mut xs = vec![0u16; num_events];
        let mut ps = vec![0i16; num_events];
        let mut pos = 0usize;
        let mut i = 0usize;

        while i < num_events {
            let a = rd_u16(data, pos)?;
            let b = rd_u16(data, pos + 2)?;
            let c = rd_u16(data, pos + 4)?;
            let vs = [
                a >> 4,
                (a & 0b1111) | ((b >> 8) << 4),
                (b & 0b11111111) | ((c >> 12) << 8),
                c & 0b111111111111,
            ];

            for (j, &packed) in vs.iter().enumerate() {
                let idx = i + j;
                if idx < num_events {
                    xs[idx] = packed >> 1;
                    ps[idx] = if packed & 1 != 0 { 1 } else { -1 };
                }
            }

            pos += 6;
            i += 4;
        }

        Ok((xs, ps, pos))
    }

    /// Decode fully packed coordinates and polarity: four events span three
    /// 32-bit words, each event holding an 11-bit Y, an 11-bit X and a 1-bit
    /// polarity. The 11-bit fields are the raw sensor coordinates (no rescaling),
    /// matching the OpenEB reference (ecf_codec.cpp:199-201).
    fn decode_ys_xs_and_ps_packed(data: &[u8], num_events: usize) -> io::Result<PackedColumns> {
        let mut xs = vec![0u16; num_events];
        let mut ys = vec![0u16; num_events];
        let mut ps = vec![0i16; num_events];
        let mut pos = 0usize;
        let mut i = 0usize;

        while i < num_events {
            let word0 = rd_u32(data, pos)?;
            let word1 = rd_u32(data, pos + 4)?;
            let word2 = rd_u32(data, pos + 8)?;

            let vs = [
                word0 >> 8,
                (word0 & 0xff) | ((word1 >> 16) << 8),
                (word1 & 0xffff) | ((word2 >> 24) << 16),
                word2 & 0xffffff,
            ];

            for (j, &packed) in vs.iter().enumerate() {
                let idx = i + j;
                if idx < num_events {
                    ys[idx] = ((packed >> 12) & 0x7ff) as u16;
                    xs[idx] = ((packed >> 1) & 0x7ff) as u16;
                    ps[idx] = if packed & 1 != 0 { 1 } else { -1 };
                }
            }

            pos += 12;
            i += 4;
        }

        Ok((xs, ys, ps, pos))
    }
}

/// Native Rust ECF Encoder: faithful port of the official Prophesee encoder.
///
/// The encoder always selects the fully packed `ys_xs_and_ps_packed` mode, which
/// is a valid ECF encoding the reference decoder (and this decoder) reads back
/// exactly. It exists mainly to round-trip synthetic data in tests.
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

    /// Encode events using ECF compression (fully packed mode).
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

        let mut output = Vec::new();

        // Header: count in bits 2-31, ys_xs_and_ps_packed flag in bit 1.
        let header = ((events.len() as u32) << 2) | 0x2;
        output.extend_from_slice(&header.to_le_bytes());

        Self::encode_ts(&mut output, events);
        Self::encode_ys_xs_and_ps_packed(&mut output, events);

        Ok(output)
    }

    /// Encode the timestamp section (origin plus nibble run-length delta stream).
    fn encode_ts(output: &mut Vec<u8>, events: &[PropheseeEvent]) {
        let count = events.len();
        let t0 = events[0].t as u64;
        output.extend_from_slice(&t0.to_le_bytes());

        let mut cur_t = t0;
        let mut i = 0usize;
        while i < count {
            let ti = events[i].t as u64;

            if ti >= cur_t + 0b1111 {
                let mut dt = ti - cur_t;
                while dt > 0 {
                    output.push(0b1111_0000 | ((dt & 0b1111) as u8));
                    dt >>= 4;
                }
                cur_t = ti;
            }

            let t: u8 = if ti < cur_t {
                0
            } else {
                let d = (ti - cur_t) as u8;
                cur_t = ti;
                d
            };

            let mut c = 1usize;
            while c < count - i {
                if events[i + c].t as u64 != cur_t {
                    break;
                }
                c += 1;
            }

            if c >= 0b1111 {
                output.push((t << 4) | 0b1111);
                output.push((c & 0xff) as u8);
                output.push(((c >> 8) & 0xff) as u8);
            } else {
                output.push((t << 4) | (c as u8));
            }

            i += c;
        }
    }

    /// Encode fully packed coordinates and polarity.
    fn encode_ys_xs_and_ps_packed(output: &mut Vec<u8>, events: &[PropheseeEvent]) {
        let count = events.len();
        let mut i = 0usize;
        while i < count {
            let mut vs = [0u32; 4];
            for (j, slot) in vs.iter_mut().enumerate() {
                let idx = i + j;
                if idx < count {
                    let e = &events[idx];
                    let x = e.x as u32 & 0x7ff;
                    let y = e.y as u32 & 0x7ff;
                    let p_bit = if e.p > 0 { 1u32 } else { 0 };
                    *slot = (y << 12) | (x << 1) | p_bit;
                }
            }

            let word0 = (vs[0] << 8) | (vs[1] & 0xff);
            let word1 = ((vs[1] >> 8) << 16) | (vs[2] & 0xffff);
            let word2 = ((vs[2] >> 16) << 24) | (vs[3] & 0xffffff);

            output.extend_from_slice(&word0.to_le_bytes());
            output.extend_from_slice(&word1.to_le_bytes());
            output.extend_from_slice(&word2.to_le_bytes());

            i += 4;
        }
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
            },
            PropheseeEvent {
                x: 102,
                y: 152,
                p: 1,
                t: 1000,
            },
        ];

        let encoder = PropheseeECFEncoder::new();
        let decoder = PropheseeECFDecoder::new();

        let compressed = encoder.encode(&events).unwrap();
        let decoded = decoder.decode(&compressed).unwrap();

        assert_eq!(events.len(), decoded.len());
        for (original, decoded) in events.iter().zip(decoded.iter()) {
            assert_eq!(*original, *decoded);
        }
    }

    #[test]
    fn test_1_bit_delta_encoding() {
        let events = vec![
            PropheseeEvent {
                x: 100,
                y: 150,
                p: 1,
                t: 1000000,
            },
            PropheseeEvent {
                x: 101,
                y: 151,
                p: -1,
                t: 1000000,
            },
            PropheseeEvent {
                x: 102,
                y: 152,
                p: 1,
                t: 1000001,
            },
            PropheseeEvent {
                x: 103,
                y: 153,
                p: -1,
                t: 1000001,
            },
            PropheseeEvent {
                x: 104,
                y: 154,
                p: 1,
                t: 1000002,
            },
        ];

        let encoder = PropheseeECFEncoder::new();
        let decoder = PropheseeECFDecoder::new();

        let compressed = encoder.encode(&events).unwrap();
        let decoded = decoder.decode(&compressed).unwrap();

        assert_eq!(events.len(), decoded.len());
        for (original, decoded) in events.iter().zip(decoded.iter()) {
            assert_eq!(*original, *decoded);
        }
    }

    #[test]
    fn test_large_delta_and_run_length_timestamps() {
        // A jump larger than 15 forces the multi-nibble large-delta path, and
        // long runs of equal timestamps exercise the run-length count. This is a
        // regression guard for the timestamp section being decoded before the
        // coordinate section.
        let mut events = Vec::new();
        for i in 0..12u16 {
            events.push(PropheseeEvent {
                x: 200 + i,
                y: 300 + i,
                p: if i % 2 == 0 { 1 } else { -1 },
                t: 1000,
            });
        }
        // Large jump then more events.
        for i in 0..8u16 {
            events.push(PropheseeEvent {
                x: 400 + i,
                y: 100 + i,
                p: if i % 2 == 0 { 1 } else { -1 },
                t: 5_000_000 + i as i64,
            });
        }

        let encoder = PropheseeECFEncoder::new();
        let decoder = PropheseeECFDecoder::new();

        let compressed = encoder.encode(&events).unwrap();
        let decoded = decoder.decode(&compressed).unwrap();

        assert_eq!(events.len(), decoded.len());
        for (original, decoded) in events.iter().zip(decoded.iter()) {
            assert_eq!(*original, *decoded);
        }
    }

    #[test]
    fn test_decode_hand_built_reference_chunk() {
        // Build a fully packed chunk BY HAND in the exact reference byte layout
        // (header, then timestamp section, then coordinate words), independent of
        // our encoder, so it pins the decoder to the reference format. The events
        // carry distinct timestamps so a decoder that reads coordinates before
        // timestamps (the historical bug) cannot pass.
        let events = [
            (100u16, 200u16, 1i16, 1000i64),
            (101, 201, -1, 1000),
            (102, 202, 1, 1005),
            (103, 203, -1, 1005),
        ];

        let mut chunk = Vec::new();
        // Header: count in bits 2-31, bit 1 = ys_xs_and_ps_packed.
        chunk.extend_from_slice(&(((events.len() as u32) << 2) | 0x2).to_le_bytes());
        // Timestamp origin.
        chunk.extend_from_slice(&1000u64.to_le_bytes());
        // Timestamp RLE nibbles, packed as (delta << 4) | count:
        // (delta 0, count 2) then (delta 5, count 2).
        chunk.push(0x02);
        chunk.push(0x52);
        // Packed coordinate words.
        let vs: Vec<u32> = events
            .iter()
            .map(|&(x, y, p, _)| {
                let p_bit = if p > 0 { 1u32 } else { 0 };
                ((y as u32) << 12) | ((x as u32) << 1) | p_bit
            })
            .collect();
        let word0 = (vs[0] << 8) | (vs[1] & 0xff);
        let word1 = ((vs[1] >> 8) << 16) | (vs[2] & 0xffff);
        let word2 = ((vs[2] >> 16) << 24) | (vs[3] & 0xffffff);
        chunk.extend_from_slice(&word0.to_le_bytes());
        chunk.extend_from_slice(&word1.to_le_bytes());
        chunk.extend_from_slice(&word2.to_le_bytes());

        let decoder = PropheseeECFDecoder::new();
        let decoded = decoder
            .decode(&chunk)
            .expect("hand-built chunk should decode");

        assert_eq!(decoded.len(), events.len());
        for (i, &(x, y, p, t)) in events.iter().enumerate() {
            assert_eq!(decoded[i].x, x, "event {i}: x");
            assert_eq!(decoded[i].y, y, "event {i}: y");
            assert_eq!(decoded[i].p, p, "event {i}: polarity");
            assert_eq!(decoded[i].t, t, "event {i}: timestamp");
        }
    }
}
