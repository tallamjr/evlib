//! Regression tests for the 2026-08-08 R4 finding: the magnitude guessing
//! convert_timestamp heuristic multiplied any timestamp below 1000 us by 1e6
//! and divided any timestamp at or above 1e9 us by 1e3. Readers now declare
//! their unit explicitly; these tests pin exact microsecond round trips for
//! events inside the first millisecond of a recording.

use std::fs::File;
use std::io::Write;
use tempfile::TempDir;

fn t_us(df: &polars::prelude::DataFrame) -> Vec<i64> {
    let t = df.column("t").unwrap().duration().unwrap();
    (0..df.height()).map(|i| t.get(i).unwrap()).collect()
}

#[test]
fn evt2_first_millisecond_round_trips_exactly() {
    use evlib::ev_formats::evt2_reader::{Evt2Config, Evt2Reader};
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("first_ms.raw");
    let mut f = File::create(&path).unwrap();
    writeln!(f, "% evt 2.0").unwrap();
    writeln!(f, "% format EVT2;height=720;width=1280").unwrap();
    writeln!(f, "% geometry 1280x720").unwrap();
    writeln!(f, "% end").unwrap();
    // EVT_TIME_HIGH = 0, then CD_ON at ts6 = 5 and CD_OFF at ts6 = 63:
    // full timestamps 5 us and 63 us (OpenEB layout, see test_evt2_bitlayout.rs).
    let words: [u32; 3] = [
        0x8u32 << 28,
        (0x1u32 << 28) | (5u32 << 22) | (10u32 << 11) | 20,
        (63u32 << 22) | (11u32 << 11) | 21,
    ];
    for w in words {
        f.write_all(&w.to_le_bytes()).unwrap();
    }
    drop(f);
    let (df, _) = Evt2Reader::with_config(Evt2Config::default())
        .read_file(&path)
        .unwrap();
    assert_eq!(t_us(&df), vec![5, 63]);
}

#[test]
fn evt3_first_millisecond_round_trips_exactly() {
    use evlib::ev_formats::evt3_reader::{Evt3Config, Evt3Reader};
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("first_ms_evt3.raw");
    let mut f = File::create(&path).unwrap();
    writeln!(f, "% evt 3.0").unwrap();
    writeln!(f, "% format EVT3;height=480;width=640").unwrap();
    writeln!(f, "% end").unwrap();
    // TIME_HIGH 0, TIME_LOW 7, Y = 100, X = 100 positive: t = 7 us.
    let words: [u16; 4] = [
        0x8u16 << 12,
        (0x6u16 << 12) | 7,
        100,
        (0x2u16 << 12) | (1u16 << 11) | 100,
    ];
    for w in words {
        f.write_all(&w.to_le_bytes()).unwrap();
    }
    drop(f);
    let (df, _) = Evt3Reader::with_config(Evt3Config {
        validate_coordinates: false,
        skip_invalid_events: false,
        max_events: None,
        sensor_resolution: Some((640, 480)),
        chunk_size: 1000,
        polarity_encoding: None,
    })
    .read_file(&path)
    .unwrap();
    assert_eq!(t_us(&df), vec![7]);
}

#[test]
fn aedat1_first_millisecond_round_trips_exactly() {
    use evlib::ev_formats::{AedatConfig, AedatReader};
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("first_ms.aedat");
    let mut f = File::create(&path).unwrap();
    writeln!(f, "#!AER-DAT1.0").unwrap();
    writeln!(f, "# sizeX 240").unwrap();
    writeln!(f, "# sizeY 180").unwrap();
    // address 0x0103 (x=1, y=1, on), timestamp 500 us (< 1000: the heuristic
    // multiplied this by 1e6).
    f.write_all(&0x0103u16.to_le_bytes()).unwrap();
    f.write_all(&500u32.to_le_bytes()).unwrap();
    drop(f);
    let reader = AedatReader::with_config(AedatConfig {
        validate_timestamps: false,
        validate_coordinates: false,
        validate_polarity: false,
        skip_invalid_events: false,
        max_events: None,
        max_resolution: None,
    });
    let (df, _) = reader.read_file(&path).unwrap();
    assert_eq!(t_us(&df), vec![500]);
}

#[test]
fn text_seconds_convert_exactly_for_first_millisecond() {
    use evlib::ev_formats::{load_events_from_text, LoadConfig};
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("first_ms.txt");
    std::fs::write(&path, "0.000005 10 20 1\n0.000250 11 21 0\n").unwrap();
    let df = load_events_from_text(path.to_str().unwrap(), &LoadConfig::new()).unwrap();
    // Unchanged behaviour: text is seconds, converted with the same truncating
    // expression as before, pinned here as the explicit contract.
    assert_eq!(t_us(&df), vec![5, 250]);
}
