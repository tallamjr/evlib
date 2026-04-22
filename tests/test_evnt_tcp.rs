//! Round-trip test: pack EVNT frames in-process, read them through EvntTcpReader,
//! assert events decode identically.

use std::io::Write;
use std::net::TcpListener;
use std::thread;

use evlib::ev_formats::{EvntEvent, EvntTcpReader};

fn pack_frame(events: &[EvntEvent]) -> Vec<u8> {
    use byteorder::{BigEndian, WriteBytesExt};
    let mut payload = Vec::with_capacity(events.len() * 13);
    for e in events {
        payload.write_u64::<BigEndian>(e.t).unwrap();
        payload.write_u16::<BigEndian>(e.x).unwrap();
        payload.write_u16::<BigEndian>(e.y).unwrap();
        payload.write_i8(e.p).unwrap();
    }
    let mut out = Vec::new();
    out.extend_from_slice(b"EVNT");
    out.write_u32::<BigEndian>(payload.len() as u32).unwrap();
    out.write_u32::<BigEndian>(events.len() as u32).unwrap();
    out.extend_from_slice(&payload);
    let crc = crc32fast::hash(&payload);
    out.write_u32::<BigEndian>(crc).unwrap();
    out
}

#[tokio::test]
async fn tcp_reader_receives_single_frame() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let sent = vec![
        EvntEvent { t: 100, x: 10, y: 20, p: 1 },
        EvntEvent { t: 200, x: 30, y: 40, p: -1 },
        EvntEvent { t: 300, x: 50, y: 60, p: 1 },
    ];
    let frame = pack_frame(&sent);

    thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream.write_all(&frame).unwrap();
    });

    let mut reader = EvntTcpReader::connect(&addr.to_string()).await.unwrap();
    let received = reader.next_batch().await.unwrap();
    assert_eq!(received, sent);
}

#[tokio::test]
async fn tcp_reader_receives_multiple_frames() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let batch_a: Vec<EvntEvent> = (0..5)
        .map(|i| EvntEvent { t: i * 1000, x: i as u16, y: (i + 1) as u16, p: 1 })
        .collect();
    let batch_b: Vec<EvntEvent> = (5..10)
        .map(|i| EvntEvent { t: i * 1000, x: i as u16, y: (i + 1) as u16, p: -1 })
        .collect();
    let mut frame_bytes = pack_frame(&batch_a);
    frame_bytes.extend_from_slice(&pack_frame(&batch_b));

    thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream.write_all(&frame_bytes).unwrap();
    });

    let mut reader = EvntTcpReader::connect(&addr.to_string()).await.unwrap();
    let first = reader.next_batch().await.unwrap();
    let second = reader.next_batch().await.unwrap();
    assert_eq!(first, batch_a);
    assert_eq!(second, batch_b);
}

#[tokio::test]
async fn tcp_reader_errors_on_bad_crc() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let sent = vec![EvntEvent { t: 1, x: 2, y: 3, p: 1 }];
    let mut frame = pack_frame(&sent);
    // Flip CRC byte
    let len = frame.len();
    frame[len - 1] ^= 0xFF;

    thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream.write_all(&frame).unwrap();
    });

    let mut reader = EvntTcpReader::connect(&addr.to_string()).await.unwrap();
    let result = reader.next_batch().await;
    assert!(matches!(result, Err(evlib::ev_formats::EvntTcpReaderError::CrcMismatch { .. })));
}
