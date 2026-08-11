//! Pins build_polars_dataframe's exact output before and after the S2 builder
//! unification, including the raw i8 polarity passthrough semantics, so the
//! refactor is provably behaviour preserving.

use evlib::ev_formats::python::build_polars_dataframe;
use evlib::ev_formats::{Event, EventFormat, TimestampUnit};
use polars::prelude::*;

fn ev(t: f64, x: u16, y: u16, polarity: i8) -> Event {
    Event { t, x, y, polarity }
}

fn t_us(df: &DataFrame) -> Vec<i64> {
    let t = df.column("t").unwrap().duration().unwrap();
    (0..df.height()).map(|i| t.get(i).unwrap()).collect()
}

fn pol(df: &DataFrame) -> Vec<i8> {
    let p = df.column("polarity").unwrap().i8().unwrap();
    (0..df.height()).map(|i| p.get(i).unwrap()).collect()
}

#[test]
fn text_seconds_frame_is_pinned() {
    // Text: seconds truncate with (t * 1e6) as i64 and 0/1 polarity is kept raw.
    let events = [ev(0.000005, 10, 20, 1), ev(0.000250, 11, 21, 0)];
    let df = build_polars_dataframe(&events, EventFormat::Text, TimestampUnit::Seconds).unwrap();
    assert_eq!(
        df.dtypes(),
        vec![
            DataType::Int16,
            DataType::Int16,
            DataType::Duration(TimeUnit::Microseconds),
            DataType::Int8
        ]
    );
    assert_eq!(t_us(&df), vec![5, 250]);
    assert_eq!(pol(&df), vec![1, 0]);
}

#[test]
fn hdf5_microseconds_polarity_mapping_is_pinned() {
    // HDF5: integer microseconds pass through verbatim; the 0 -> -1 / else -> 1
    // vectorised conversion applies to the raw appended i8. An input of -1 maps
    // to +1 today (the "otherwise" arm); this pins that pre-existing behaviour
    // so the unification cannot silently change it.
    let events = [ev(249.0, 1, 2, 0), ev(250.0, 3, 4, 1), ev(251.0, 5, 6, -1)];
    let df =
        build_polars_dataframe(&events, EventFormat::HDF5, TimestampUnit::Microseconds).unwrap();
    assert_eq!(t_us(&df), vec![249, 250, 251]);
    assert_eq!(pol(&df), vec![-1, 1, 1]);
}

#[test]
fn empty_input_keeps_the_schema() {
    let df = build_polars_dataframe(&[], EventFormat::Text, TimestampUnit::Seconds).unwrap();
    assert_eq!(df.height(), 0);
    assert_eq!(
        df.get_column_names()
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>(),
        vec!["x", "y", "t", "polarity"]
    );
    assert_eq!(
        df.column("t").unwrap().dtype(),
        &DataType::Duration(TimeUnit::Microseconds)
    );
}
