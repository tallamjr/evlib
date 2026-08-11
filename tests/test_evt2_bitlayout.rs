/// EVT2 wire-format conformance tests against the OpenEB reference bit layout.
///
/// OpenEB defines the EVT2 CD event (32-bit little-endian word) as:
///   bits [10:0]  = y          (11 bits)
///   bits [21:11] = x          (11 bits)
///   bits [27:22] = timestamp  (6 bits, low bits of the timebase)
///   bits [31:28] = type       (4 bits; 0x00 CD_OFF, 0x01 CD_ON)
/// and the EVT_TIME_HIGH event (type 0x08) carries a 28-bit value that is the
/// high part of the 34-bit timebase, so the full timestamp in microseconds is
///   (time_high << 6) | cd_timestamp6
/// Only types 0x00 and 0x01 are CD events; all other type codes are not CD.
///
/// See lib/openeb/standalone_samples/metavision_evt2_raw_file_decoder.
#[cfg(test)]
mod evt2_bitlayout_tests {
    use evlib::ev_formats::evt2_reader::{Evt2Config, Evt2Reader};
    use std::fs::File;
    use std::io::Write;
    use tempfile::TempDir;

    #[derive(Debug, Clone, Copy)]
    struct EventRow {
        t_us: i64,
        x: u16,
        y: u16,
        polarity: i8,
    }

    fn dataframe_to_rows(df: &polars::prelude::DataFrame) -> Vec<EventRow> {
        let x = df.column("x").unwrap().i16().unwrap();
        let y = df.column("y").unwrap().i16().unwrap();
        let t = df.column("t").unwrap().duration().unwrap();
        let p = df.column("polarity").unwrap().i8().unwrap();
        (0..df.height())
            .map(|i| EventRow {
                t_us: t.get(i).unwrap(),
                x: x.get(i).unwrap() as u16,
                y: y.get(i).unwrap() as u16,
                polarity: p.get(i).unwrap(),
            })
            .collect()
    }

    /// Build an EVT2 CD word per the OpenEB layout (y low, x next, ts, type).
    fn cd_word(ty: u32, x: u32, y: u32, ts6: u32) -> u32 {
        ((ty & 0xF) << 28) | ((ts6 & 0x3F) << 22) | ((x & 0x7FF) << 11) | (y & 0x7FF)
    }

    /// Build an EVT_TIME_HIGH word (type 0x08, 28-bit value).
    fn time_high_word(value: u32) -> u32 {
        (0x8u32 << 28) | (value & 0x0FFF_FFFF)
    }

    fn write_evt2(words: &[u32]) -> (TempDir, std::path::PathBuf) {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("synthetic.raw");
        let mut file = File::create(&path).unwrap();
        writeln!(file, "% evt 2.0").unwrap();
        writeln!(file, "% format EVT2;height=720;width=1280").unwrap();
        writeln!(file, "% geometry 1280x720").unwrap();
        writeln!(file, "% end").unwrap();
        for w in words {
            file.write_all(&w.to_le_bytes()).unwrap();
        }
        (dir, path)
    }

    /// Write an EVT2 file whose header lines are given verbatim (so a caller can
    /// omit the `% end` terminator, as Gen3 recordings do).
    fn write_evt2_with_header(
        header_lines: &[&str],
        words: &[u32],
    ) -> (TempDir, std::path::PathBuf) {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("synthetic.raw");
        let mut file = File::create(&path).unwrap();
        for line in header_lines {
            writeln!(file, "{line}").unwrap();
        }
        for w in words {
            file.write_all(&w.to_le_bytes()).unwrap();
        }
        (dir, path)
    }

    /// A Gen3-style EVT2 header has no `% end` terminator: the binary data begins
    /// at the first line that does not start with `%`. The reader must locate that
    /// boundary by the line prefix (as OpenEB does), not by scanning for a run of
    /// non-printable bytes, otherwise it reads the binary section as header text
    /// and the event stream is misaligned.
    #[test]
    fn test_evt2_header_without_end_terminator_is_aligned() {
        let header = ["% evt 2.0", "% integrator_name Prophesee"]; // no "% end"
        let words = [
            time_high_word(100),
            cd_word(0x01, 600, 400, 5), // CD_ON  -> x=600 y=400 t=6405
            cd_word(0x00, 10, 20, 1),   // CD_OFF -> x=10  y=20  t=6401
        ];
        let (_dir, path) = write_evt2_with_header(&header, &words);

        let config = Evt2Config {
            validate_coordinates: false,
            skip_invalid_events: false,
            max_events: None,
            sensor_resolution: Some((1280, 720)),
            chunk_size: 1000,
        };
        let reader = Evt2Reader::with_config(config);
        let (df, _) = reader.read_file(&path).unwrap();
        let rows = dataframe_to_rows(&df);

        assert_eq!(rows.len(), 2, "binary section misaligned, got {rows:?}");
        assert_eq!(
            (rows[0].x, rows[0].y, rows[0].polarity, rows[0].t_us),
            (600, 400, 1, 6405)
        );
        assert_eq!(
            (rows[1].x, rows[1].y, rows[1].polarity, rows[1].t_us),
            (10, 20, -1, 6401)
        );
    }

    /// A CD event that appears before the first EVT_TIME_HIGH has no valid time
    /// base, so its timestamp is meaningless. OpenEB skips all events until the
    /// first TIME_HIGH; evlib must do the same, otherwise it emits a spurious
    /// leading event with a garbage timestamp.
    #[test]
    fn test_evt2_skips_cd_events_before_first_time_high() {
        let words = [
            cd_word(0x01, 795, 176, 43), // CD before any TIME_HIGH -> must be skipped
            time_high_word(100),         // first time base = 100 << 6 = 6400
            cd_word(0x01, 600, 400, 5),  // emitted at t = 6405
        ];
        let (_dir, path) = write_evt2(&words);

        let config = Evt2Config {
            validate_coordinates: false,
            skip_invalid_events: false,
            max_events: None,
            sensor_resolution: Some((1280, 720)),
            chunk_size: 1000,
        };
        let reader = Evt2Reader::with_config(config);
        let (df, _) = reader.read_file(&path).unwrap();
        let rows = dataframe_to_rows(&df);

        assert_eq!(
            rows.len(),
            1,
            "pre-TIME_HIGH event not skipped, got {rows:?}"
        );
        assert_eq!(
            (rows[0].x, rows[0].y, rows[0].polarity, rows[0].t_us),
            (600, 400, 1, 6405)
        );
    }

    #[test]
    fn test_evt2_cd_event_matches_openeb_layout() {
        // TIME_HIGH = 100 -> base timestamp = 100 << 6 = 6400 us.
        // CD_ON at x=600, y=400, ts_low=5 -> full t = 6400 + 5 = 6405 us.
        // A vendor type (0x04) word must NOT produce a CD event.
        let words = [
            time_high_word(100),
            cd_word(0x01, 600, 400, 5),
            cd_word(0x04, 1234, 1234, 0), // vendor/non-CD: must be ignored
        ];
        let (_dir, path) = write_evt2(&words);

        let config = Evt2Config {
            validate_coordinates: false,
            skip_invalid_events: false,
            max_events: None,
            sensor_resolution: Some((1280, 720)),
            chunk_size: 1000,
        };
        let reader = Evt2Reader::with_config(config);
        let (df, _) = reader.read_file(&path).unwrap();
        let rows = dataframe_to_rows(&df);

        // Only the single real CD event should be decoded (vendor word ignored).
        assert_eq!(rows.len(), 1, "expected exactly one CD event, got {rows:?}");
        let e = rows[0];
        // x and y must NOT be transposed: x from bits[21:11], y from bits[10:0].
        assert_eq!(e.x, 600, "x decoded wrong (transposed?)");
        assert_eq!(e.y, 400, "y decoded wrong (transposed?)");
        assert_eq!(e.polarity, 1, "CD_ON must be +1 polarity");
        // Timestamp: TIME_HIGH must be shifted left by 6 before adding the low bits.
        assert_eq!(e.t_us, 6405, "timestamp wrong: TIME_HIGH not shifted by 6?");
    }
}
