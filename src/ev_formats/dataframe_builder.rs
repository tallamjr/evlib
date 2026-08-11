// DataFrame construction utilities for direct event processing
// This module provides optimized DataFrame builders that eliminate the need for intermediate Event structs

use crate::ev_formats::EventFormat;
use polars::prelude::*;

#[cfg(unix)]
use tracing::debug;

#[cfg(not(unix))]
macro_rules! debug {
    ($($args:tt)*) => {};
}

#[cfg(not(unix))]
macro_rules! info {
    ($($args:tt)*) => {};
}

/// Direct DataFrame builder for event data
/// This eliminates the intermediate Event struct and builds DataFrames directly from raw event data
pub struct EventDataFrameBuilder {
    x_builder: PrimitiveChunkedBuilder<Int16Type>,
    y_builder: PrimitiveChunkedBuilder<Int16Type>,
    timestamp_builder: PrimitiveChunkedBuilder<Int64Type>,
    polarity_builder: PrimitiveChunkedBuilder<Int8Type>,
    format: EventFormat,
    event_count: usize,
}

impl EventDataFrameBuilder {
    /// Create a new builder with estimated capacity
    pub fn new(format: EventFormat, estimated_capacity: usize) -> Self {
        Self {
            x_builder: PrimitiveChunkedBuilder::<Int16Type>::new("x".into(), estimated_capacity),
            y_builder: PrimitiveChunkedBuilder::<Int16Type>::new("y".into(), estimated_capacity),
            timestamp_builder: PrimitiveChunkedBuilder::<Int64Type>::new(
                "t".into(),
                estimated_capacity,
            ),
            polarity_builder: PrimitiveChunkedBuilder::<Int8Type>::new(
                "polarity".into(),
                estimated_capacity,
            ),
            format,
            event_count: 0,
        }
    }

    /// Add a single event with an integer microsecond timestamp.
    ///
    /// This is the sole per-event entry point: there is no magnitude-based
    /// guessing (2026-08-08 review, R4). Callers must already hold an integer
    /// microsecond value.
    pub fn add_event_microseconds(&mut self, x: u16, y: u16, timestamp_us: i64, polarity: bool) {
        self.x_builder.append_value(x as i16);
        self.y_builder.append_value(y as i16);
        self.timestamp_builder.append_value(timestamp_us);
        self.polarity_builder
            .append_value(if polarity { 1i8 } else { 0i8 });
        self.event_count += 1;
    }

    /// Get the current number of events in the builder
    pub fn len(&self) -> usize {
        self.event_count
    }

    /// Check if the builder is empty
    pub fn is_empty(&self) -> bool {
        self.event_count == 0
    }

    /// Build the final DataFrame with format-specific polarity conversion
    pub fn build(self) -> PolarsResult<DataFrame> {
        if self.event_count == 0 {
            // Create empty DataFrame with proper schema
            let empty_x = Series::new("x".into(), Vec::<i16>::new());
            let empty_y = Series::new("y".into(), Vec::<i16>::new());
            let empty_timestamp = Series::new("t".into(), Vec::<i64>::new())
                .cast(&DataType::Duration(TimeUnit::Microseconds))?;
            let empty_polarity = Series::new("polarity".into(), Vec::<i8>::new());

            return DataFrame::new(vec![
                empty_x.into(),
                empty_y.into(),
                empty_timestamp.into(),
                empty_polarity.into(),
            ]);
        }

        // Build Series from builders
        let x_series = self.x_builder.finish().into_series();
        let y_series = self.y_builder.finish().into_series();
        let polarity_series_raw = self.polarity_builder.finish().into_series();

        // Convert timestamp to Duration type
        let timestamp_series = self
            .timestamp_builder
            .finish()
            .into_series()
            .cast(&DataType::Duration(TimeUnit::Microseconds))?;

        // Create initial DataFrame with raw polarity
        let df = DataFrame::new(vec![
            x_series.into(),
            y_series.into(),
            timestamp_series.into(),
            polarity_series_raw.into(),
        ])?;

        // VECTORIZED polarity conversion (much faster than per-event)
        let df = match self.format {
            EventFormat::EVT2 | EventFormat::EVT21 | EventFormat::EVT3 | EventFormat::AEDAT4 => {
                // EVT2 family and AEDAT 4.0 (DV): Convert 0/1 to -1/1 using vectorized operations
                df.lazy()
                    .with_column(
                        when(col("polarity").eq(lit(0)))
                            .then(lit(-1i8))
                            .otherwise(lit(1i8))
                            .alias("polarity")
                            .cast(DataType::Int8),
                    )
                    .collect()?
            }
            #[cfg(not(windows))]
            EventFormat::HDF5 => {
                // HDF5: Convert 0/1 to -1/1 for proper polarity encoding
                df.lazy()
                    .with_column(
                        when(col("polarity").eq(lit(0)))
                            .then(lit(-1i8))
                            .otherwise(lit(1i8))
                            .alias("polarity")
                            .cast(DataType::Int8),
                    )
                    .collect()?
            }
            #[cfg(windows)]
            EventFormat::HDF5 => {
                return Err(PolarsError::ComputeError(
                    "HDF5 support is disabled on Windows due to build complexity.".into(),
                ));
            }
            _ => {
                // Text and other formats: Keep 0/1 encoding as-is, but ensure Int8 type
                df.lazy()
                    .with_column(col("polarity").cast(DataType::Int8))
                    .collect()?
            }
        };

        debug!(events = self.event_count, format = ?self.format, "Built DataFrame directly");
        Ok(df)
    }
}

/// Create an empty DataFrame with the correct schema
pub fn create_empty_events_dataframe() -> PolarsResult<DataFrame> {
    let empty_x = Series::new("x".into(), Vec::<i16>::new());
    let empty_y = Series::new("y".into(), Vec::<i16>::new());
    let empty_timestamp = Series::new("t".into(), Vec::<i64>::new())
        .cast(&DataType::Duration(TimeUnit::Microseconds))?;
    let empty_polarity = Series::new("polarity".into(), Vec::<i8>::new());

    DataFrame::new(vec![
        empty_x.into(),
        empty_y.into(),
        empty_timestamp.into(),
        empty_polarity.into(),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataframe_builder() {
        let mut builder = EventDataFrameBuilder::new(EventFormat::Text, 10);

        // Add some test events
        builder.add_event_microseconds(100, 200, 1_500_000, true);
        builder.add_event_microseconds(150, 250, 2_000_000, false);
        builder.add_event_microseconds(200, 300, 2_500_000, true);

        assert_eq!(builder.len(), 3);

        let df = builder.build().unwrap();
        assert_eq!(df.height(), 3);
        assert_eq!(df.width(), 4);

        let columns = df.get_column_names();
        let column_names: Vec<String> = columns.iter().map(|s| s.to_string()).collect();
        assert!(column_names.contains(&"x".to_string()));
        assert!(column_names.contains(&"y".to_string()));
        assert!(column_names.contains(&"t".to_string()));
        assert!(column_names.contains(&"polarity".to_string()));
    }

    #[test]
    fn test_add_event_microseconds_stores_value_verbatim() {
        let mut builder = EventDataFrameBuilder::new(EventFormat::AER, 4);

        // Values that the magnitude heuristic would misclassify if routed through
        // convert_timestamp: 1 us (< 1000 => seconds) and 1_001_000 us (>= 1000).
        builder.add_event_microseconds(10, 20, 1, true);
        builder.add_event_microseconds(30, 40, 1_001_000, false);

        let df = builder.build().unwrap();
        let t = df.column("t").unwrap().duration().unwrap();
        assert_eq!(t.get(0).unwrap(), 1);
        assert_eq!(t.get(1).unwrap(), 1_001_000);
    }
}
