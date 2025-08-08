// DataFrame construction utilities for direct event processing
// This module provides optimized DataFrame builders that eliminate the need for intermediate Event structs

use crate::ev_formats::EventFormat;
#[cfg(feature = "polars")]
use polars::prelude::*;

#[cfg(feature = "tracing")]
use tracing::debug;

#[cfg(not(feature = "tracing"))]
macro_rules! debug {
    ($($args:tt)*) => {};
}

#[cfg(not(feature = "tracing"))]
macro_rules! info {
    ($($args:tt)*) => {};
}

/// Convert timestamp to microseconds for Polars Duration type
pub fn convert_timestamp(timestamp: f64) -> i64 {
    if timestamp >= 1_000_000_000.0 {
        // Likely nanoseconds, convert to microseconds
        (timestamp / 1_000.0) as i64
    } else if timestamp >= 1_000.0 {
        // Likely already in microseconds
        timestamp as i64
    } else {
        // Likely in seconds, convert to microseconds
        (timestamp * 1_000_000.0) as i64
    }
}

/// Direct DataFrame builder for event data
/// This eliminates the intermediate Event struct and builds DataFrames directly from raw event data
#[cfg(feature = "polars")]
pub struct EventDataFrameBuilder {
    x_builder: PrimitiveChunkedBuilder<Int16Type>,
    y_builder: PrimitiveChunkedBuilder<Int16Type>,
    timestamp_builder: PrimitiveChunkedBuilder<Int64Type>,
    polarity_builder: PrimitiveChunkedBuilder<Int8Type>,
    format: EventFormat,
    event_count: usize,
}

#[cfg(feature = "polars")]
impl EventDataFrameBuilder {
    /// Create a new builder with estimated capacity
    pub fn new(format: EventFormat, estimated_capacity: usize) -> Self {
        Self {
            x_builder: PrimitiveChunkedBuilder::<Int16Type>::new("x".into(), estimated_capacity),
            y_builder: PrimitiveChunkedBuilder::<Int16Type>::new("y".into(), estimated_capacity),
            timestamp_builder: PrimitiveChunkedBuilder::<Int64Type>::new(
                "timestamp".into(),
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

    /// Add a single event directly to the DataFrame builder
    pub fn add_event(&mut self, x: u16, y: u16, timestamp: f64, polarity: bool) {
        self.x_builder.append_value(x as i16);
        self.y_builder.append_value(y as i16);
        self.timestamp_builder
            .append_value(convert_timestamp(timestamp));
        // Store raw bool polarity (0/1) - will convert vectorized later
        self.polarity_builder
            .append_value(if polarity { 1i8 } else { 0i8 });
        self.event_count += 1;
    }

    /// Add multiple events in batch
    pub fn add_events_batch(&mut self, events: &[(u16, u16, f64, bool)]) {
        for &(x, y, timestamp, polarity) in events {
            self.add_event(x, y, timestamp, polarity);
        }
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
            let empty_timestamp = Series::new("timestamp".into(), Vec::<i64>::new())
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
            EventFormat::EVT2 | EventFormat::EVT21 | EventFormat::EVT3 => {
                // EVT2 family: Convert 0/1 to -1/1 using vectorized operations
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
#[cfg(feature = "polars")]
pub fn create_empty_events_dataframe() -> PolarsResult<DataFrame> {
    let empty_x = Series::new("x".into(), Vec::<i16>::new());
    let empty_y = Series::new("y".into(), Vec::<i16>::new());
    let empty_timestamp = Series::new("timestamp".into(), Vec::<i64>::new())
        .cast(&DataType::Duration(TimeUnit::Microseconds))?;
    let empty_polarity = Series::new("polarity".into(), Vec::<i8>::new());

    DataFrame::new(vec![
        empty_x.into(),
        empty_y.into(),
        empty_timestamp.into(),
        empty_polarity.into(),
    ])
}

/// Streaming builder for very large datasets
/// Processes events in chunks and yields DataFrames incrementally
#[cfg(feature = "polars")]
pub struct EventDataFrameStreamer {
    builder: EventDataFrameBuilder,
    chunk_size: usize,
    total_events: usize,
}

#[cfg(feature = "polars")]
impl EventDataFrameStreamer {
    /// Create a new streaming builder
    pub fn new(format: EventFormat, chunk_size: usize) -> Self {
        Self {
            builder: EventDataFrameBuilder::new(format, chunk_size),
            chunk_size,
            total_events: 0,
        }
    }

    /// Add an event to the stream, returning a DataFrame if chunk is full
    pub fn add_event(
        &mut self,
        x: u16,
        y: u16,
        timestamp: f64,
        polarity: bool,
    ) -> PolarsResult<Option<DataFrame>> {
        self.builder.add_event(x, y, timestamp, polarity);
        self.total_events += 1;

        if self.builder.len() >= self.chunk_size {
            let df = self.flush()?;
            Ok(Some(df))
        } else {
            Ok(None)
        }
    }

    /// Flush remaining events to a DataFrame
    pub fn flush(&mut self) -> PolarsResult<DataFrame> {
        if self.builder.is_empty() {
            return create_empty_events_dataframe();
        }

        let format = self.builder.format;
        let old_builder = std::mem::replace(
            &mut self.builder,
            EventDataFrameBuilder::new(format, self.chunk_size),
        );
        old_builder.build()
    }

    /// Get total events processed
    pub fn total_events(&self) -> usize {
        self.total_events
    }
}

/// Calculate optimal chunk size based on available memory and file size
pub fn calculate_optimal_chunk_size(file_size: u64, available_memory_bytes: usize) -> usize {
    // Use 25% of available memory for chunk processing
    let target_memory_usage = available_memory_bytes / 4;

    // Estimate bytes per event in DataFrame (approximately 16 bytes per event)
    let estimated_event_size = 16;

    let memory_based_chunk = target_memory_usage / estimated_event_size;

    // Also consider file size - for small files, don't over-chunk
    let file_based_chunk = if file_size < 10_000_000 {
        // < 10MB
        100_000 // Small chunks for small files
    } else if file_size < 100_000_000 {
        // < 100MB
        500_000 // Medium chunks for medium files
    } else {
        2_000_000 // Large chunks for large files
    };

    // Use the smaller of the two, but ensure reasonable bounds
    memory_based_chunk
        .min(file_based_chunk)
        .clamp(100_000, 5_000_000)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "polars")]
    #[test]
    fn test_dataframe_builder() {
        let mut builder = EventDataFrameBuilder::new(EventFormat::Text, 10);

        // Add some test events
        builder.add_event(100, 200, 1.5, true);
        builder.add_event(150, 250, 2.0, false);
        builder.add_event(200, 300, 2.5, true);

        assert_eq!(builder.len(), 3);

        let df = builder.build().unwrap();
        assert_eq!(df.height(), 3);
        assert_eq!(df.width(), 4);

        let columns: Vec<&str> = df.get_column_names();
        assert!(columns.contains(&"x"));
        assert!(columns.contains(&"y"));
        assert!(columns.contains(&"timestamp"));
        assert!(columns.contains(&"polarity"));
    }

    #[test]
    fn test_chunk_size_calculation() {
        // Test small file
        let chunk = calculate_optimal_chunk_size(1_000_000, 1_000_000_000); // 1MB file, 1GB memory
        assert!(chunk >= 100_000 && chunk <= 5_000_000);

        // Test large file
        let chunk = calculate_optimal_chunk_size(1_000_000_000, 1_000_000_000); // 1GB file, 1GB memory
        assert!(chunk >= 100_000 && chunk <= 5_000_000);
    }

    #[test]
    fn test_timestamp_conversion() {
        // Test seconds to microseconds
        assert_eq!(convert_timestamp(1.5), 1_500_000);

        // Test microseconds (no conversion)
        assert_eq!(convert_timestamp(1_500_000.0), 1_500_000);

        // Test nanoseconds to microseconds
        assert_eq!(convert_timestamp(1_500_000_000.0), 1_500_000);
    }
}
