//! EVNT-framed TCP event stream reader.
//!
//! Frame format (big-endian):
//!   [4 bytes "EVNT"][u32 payload_len][u32 num_events][payload][u32 crc32]
//!   payload = num_events * 13 bytes (u64 t, u16 x, u16 y, i8 p)
//!
//! CRC32 covers payload only. This is the same protocol used by
//! `uep/serialstreamer.py` and `boundingbox-rs/src/socket_input.rs`.

use byteorder::{BigEndian, ByteOrder};
use thiserror::Error;
use tokio::io::{AsyncReadExt, BufReader};
use tokio::net::TcpStream;

const MAGIC: [u8; 4] = *b"EVNT";
const HEADER_LEN: usize = 12; // magic + payload_len + num_events
const CRC_LEN: usize = 4;
const EVENT_LEN: usize = 13;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvntEvent {
    pub t: u64,
    pub x: u16,
    pub y: u16,
    pub p: i8,
}

#[derive(Debug, Error)]
pub enum EvntTcpReaderError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("bad magic: expected EVNT, got {got:?}")]
    BadMagic { got: [u8; 4] },
    #[error("CRC mismatch: expected {expected:#x}, got {got:#x}")]
    CrcMismatch { expected: u32, got: u32 },
    #[error("payload length {len} not a multiple of event size ({event_len})")]
    BadPayloadLength { len: u32, event_len: usize },
}

pub struct EvntTcpReader {
    reader: BufReader<TcpStream>,
}

impl EvntTcpReader {
    pub async fn connect(addr: &str) -> Result<Self, EvntTcpReaderError> {
        let stream = TcpStream::connect(addr).await?;
        Ok(Self { reader: BufReader::new(stream) })
    }

    /// Read the next complete frame and return its decoded events.
    pub async fn next_batch(&mut self) -> Result<Vec<EvntEvent>, EvntTcpReaderError> {
        let mut header = [0u8; HEADER_LEN];
        self.reader.read_exact(&mut header).await?;
        let magic = [header[0], header[1], header[2], header[3]];
        if magic != MAGIC {
            return Err(EvntTcpReaderError::BadMagic { got: magic });
        }
        let payload_len = BigEndian::read_u32(&header[4..8]);
        let num_events = BigEndian::read_u32(&header[8..12]);

        if payload_len as usize % EVENT_LEN != 0 {
            return Err(EvntTcpReaderError::BadPayloadLength {
                len: payload_len,
                event_len: EVENT_LEN,
            });
        }

        let mut payload = vec![0u8; payload_len as usize];
        self.reader.read_exact(&mut payload).await?;

        let mut crc_buf = [0u8; CRC_LEN];
        self.reader.read_exact(&mut crc_buf).await?;
        let crc_got = BigEndian::read_u32(&crc_buf);
        let crc_expected = crc32fast::hash(&payload);
        if crc_got != crc_expected {
            return Err(EvntTcpReaderError::CrcMismatch {
                expected: crc_expected,
                got: crc_got,
            });
        }

        let mut events = Vec::with_capacity(num_events as usize);
        for i in 0..num_events as usize {
            let off = i * EVENT_LEN;
            events.push(EvntEvent {
                t: BigEndian::read_u64(&payload[off..off + 8]),
                x: BigEndian::read_u16(&payload[off + 8..off + 10]),
                y: BigEndian::read_u16(&payload[off + 10..off + 12]),
                p: payload[off + 12] as i8,
            });
        }
        Ok(events)
    }
}
