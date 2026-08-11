//! Synthetic EVT2.1 conformance tests (no real EVT2.1 sample exists in the
//! repo). Pins the timestamp reconstruction: the event word's timestamp field
//! is the 6 LSBs (bits 59-54) and TIME_HIGH carries the 28 MSBs (bits 59-32),
//! so full_t = (time_high << 6) + ts6 with a 2^34 us rollover period. The old
//! code shifted by 10, stretching every timestamp roughly 16x and breaking
//! rollover detection (2026-08-08 review, R5).

use evlib::ev_formats::evt21_reader::{Evt21Config, Evt21Error, Evt21Reader};
use std::fs::File;
use std::io::Write;
use tempfile::TempDir;

fn th_word(value28: u64) -> u64 {
    (0x8u64 << 60) | ((value28 & 0x0FFF_FFFF) << 32)
}

fn evt_pos_word(ts6: u64, x_base: u64, y: u64, mask: u32) -> u64 {
    (0x1u64 << 60)
        | ((ts6 & 0x3F) << 54)
        | ((x_base & 0x7FF) << 43)
        | ((y & 0x7FF) << 32)
        | mask as u64
}

fn write_evt21(words: &[u64]) -> (TempDir, std::path::PathBuf) {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("synthetic.raw");
    let mut f = File::create(&path).unwrap();
    writeln!(f, "% evt 2.1").unwrap();
    writeln!(f, "% format EVT21;height=720;width=1280").unwrap();
    writeln!(f, "% geometry 1280x720").unwrap();
    writeln!(f, "% end").unwrap();
    for w in words {
        f.write_all(&w.to_le_bytes()).unwrap();
    }
    (dir, path)
}

#[test]
fn evt21_timestamps_use_six_bit_shift_and_survive_rollover() {
    let words = [
        // base = 0: vectorised event at ts6 = 5, two valid bits -> t = 5 us
        th_word(0),
        evt_pos_word(5, 100, 50, 0b11),
        // base = 16 << 6 = 1024 us (the wrong shift gives 16 << 10 = 16384)
        th_word(16),
        evt_pos_word(5, 100, 50, 0b1),
        // just below the 34-bit ceiling: t = (2^28 - 1) * 64 + 63 = 2^34 - 1
        th_word(0x0FFF_FFFF),
        evt_pos_word(63, 200, 60, 0b1),
        // TIME_HIGH wraps to 0: rollover must add one 2^34 us loop
        th_word(0),
        evt_pos_word(1, 300, 70, 0b1),
    ];
    let (_dir, path) = write_evt21(&words);
    let (df, _) = Evt21Reader::with_config(Evt21Config::default())
        .read_file(&path)
        .unwrap();
    let t = df.column("t").unwrap().duration().unwrap();
    let got: Vec<i64> = (0..df.height()).map(|i| t.get(i).unwrap()).collect();
    assert_eq!(
        got,
        vec![5, 5, 1029, 17_179_869_183, 17_179_869_185],
        "EVT2.1 reconstruction must be (time_high << 6) + ts6 with 2^34 us loops"
    );
    // The vectorised pair shares one timestamp but two x positions.
    let x = df.column("x").unwrap().i16().unwrap();
    assert_eq!((x.get(0).unwrap(), x.get(1).unwrap()), (100, 101));
    assert!(
        got.windows(2).all(|w| w[0] <= w[1]),
        "stream must be monotonic across the TIME_HIGH rollover"
    );
}

// Replacement coverage for the deleted `decode_vectorized_event` unit tests
// (2026-08-10 review): the vectorised event below has x_base = 1270 with
// validity bits 0 and 10 set, producing pixels x = 1270 (in bounds, the
// synthetic header declares width 1280) and x = 1280 (out of bounds).

#[test]
fn evt21_coordinate_validation_errors_when_out_of_bounds() {
    let words = [th_word(0), evt_pos_word(5, 1270, 50, 0b100_0000_0001)];
    let (_dir, path) = write_evt21(&words);
    let config = Evt21Config {
        validate_coordinates: true,
        skip_invalid_events: false,
        ..Default::default()
    };
    let result = Evt21Reader::with_config(config).read_file(&path);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        Evt21Error::CoordinateOutOfBounds {
            x: 1280,
            y: 50,
            max_x: 1280,
            max_y: 720
        }
    ));
}

#[test]
fn evt21_coordinate_validation_skips_out_of_bounds_when_configured() {
    let words = [th_word(0), evt_pos_word(5, 1270, 50, 0b100_0000_0001)];
    let (_dir, path) = write_evt21(&words);
    let config = Evt21Config {
        validate_coordinates: true,
        skip_invalid_events: true,
        ..Default::default()
    };
    let (df, _) = Evt21Reader::with_config(config).read_file(&path).unwrap();
    assert_eq!(df.height(), 1);
    let x = df.column("x").unwrap().i16().unwrap();
    assert_eq!(x.get(0).unwrap(), 1270);
}
