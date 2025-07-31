use crate::ev_core::Event;
use crate::ev_formats::EventFormat;

#[cfg(feature = "polars")]
use polars::prelude::*;

/// Configuration for streaming event processing
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Number of events to process per chunk
    pub chunk_size: usize,
    /// Maximum memory usage in MB (reserved for future use)
    pub _memory_limit_mb: usize,
    /// Progress reporting interval (in events)
    pub progress_interval: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1_000_000,
            _memory_limit_mb: 512,
            progress_interval: 10_000_000,
        }
    }
}

/// High-performance event streamer for large datasets with chunked processing
///
/// This streamer processes events in chunks to handle files >100M events efficiently
/// while maintaining memory constraints and producing identical results to direct loading.
pub struct PolarsEventStreamer {
    chunk_size: usize,
    format: EventFormat,
    _memory_limit_mb: usize,
    progress_interval: usize,
}

impl PolarsEventStreamer {
    /// Create a new PolarsEventStreamer with specified parameters
    ///
    /// # Arguments
    /// * `chunk_size` - Number of events to process per chunk
    /// * `format` - Event format for proper polarity encoding
    ///
    /// # Returns
    /// A new PolarsEventStreamer instance
    pub fn new(chunk_size: usize, format: EventFormat) -> Self {
        Self {
            chunk_size,
            format,
            _memory_limit_mb: 512,
            progress_interval: 10_000_000,
        }
    }

    /// Create a new PolarsEventStreamer with full configuration
    ///
    /// # Arguments
    /// * `config` - Streaming configuration
    /// * `format` - Event format for proper polarity encoding
    ///
    /// # Returns
    /// A new PolarsEventStreamer instance
    pub fn with_config(config: StreamingConfig, format: EventFormat) -> Self {
        Self {
            chunk_size: config.chunk_size,
            format,
            _memory_limit_mb: config._memory_limit_mb,
            progress_interval: config.progress_interval,
        }
    }

    /// Calculate optimal chunk size based on available memory and event count
    ///
    /// # Arguments
    /// * `total_events` - Total number of events to process
    /// * `available_memory_mb` - Available memory in MB
    ///
    /// # Returns
    /// Optimal chunk size for memory efficiency
    pub fn calculate_optimal_chunk_size(total_events: usize, available_memory_mb: usize) -> usize {
        // Conservative estimate: 15 bytes per event in memory
        const BYTES_PER_EVENT: usize = 15;

        // Use 25% of available memory for chunk processing
        let target_memory_bytes = (available_memory_mb * 1024 * 1024) / 4;
        let memory_based_chunk_size = target_memory_bytes / BYTES_PER_EVENT;

        // Clamp to reasonable bounds
        let chunk_size = memory_based_chunk_size.clamp(100_000, 10_000_000);

        // Optimize chunk size based on file size
        if total_events > 500_000_000 {
            // Very large files (500M+): Use larger chunks for efficiency
            chunk_size.clamp(5_000_000, 10_000_000)
        } else if total_events > 100_000_000 {
            // Large files (100M+): Use medium-large chunks
            chunk_size.clamp(2_000_000, 5_000_000)
        } else if total_events > 10_000_000 {
            // Medium files (10M+): Use medium chunks
            chunk_size.clamp(1_000_000, 2_000_000)
        } else {
            // Smaller files: Use smaller chunks
            chunk_size.clamp(100_000, 1_000_000)
        }
    }

    /// Main streaming function that processes events in chunks and returns a Polars DataFrame
    ///
    /// # Arguments
    /// * `events` - Iterator of Event objects to process
    ///
    /// # Returns
    /// Result containing a Polars DataFrame with all events
    #[cfg(feature = "polars")]
    #[allow(unused_assignments)]
    pub fn stream_to_polars<I>(&self, events: I) -> PolarsResult<DataFrame>
    where
        I: Iterator<Item = Event>,
    {
        let mut dataframes = Vec::new();
        let mut chunk_buffer = Vec::with_capacity(self.chunk_size);
        let mut total_processed = 0;
        let mut _chunk_count = 0;

        for event in events {
            chunk_buffer.push(event);

            // Process chunk when it's full
            if chunk_buffer.len() >= self.chunk_size {
                let chunk_df = self.build_chunk(&chunk_buffer)?;
                if !chunk_df.is_empty() {
                    dataframes.push(chunk_df);
                }

                total_processed += chunk_buffer.len();
                _chunk_count += 1;

                // Progress reporting
                if total_processed % self.progress_interval == 0 {
                    // Progress reporting was removed
                }

                chunk_buffer.clear();
            }
        }

        // Process remaining events in the buffer
        if !chunk_buffer.is_empty() {
            let chunk_df = self.build_chunk(&chunk_buffer)?;
            if !chunk_df.is_empty() {
                dataframes.push(chunk_df);
            }
            total_processed += chunk_buffer.len();
            _chunk_count += 1;
        }

        // Handle empty case
        if dataframes.is_empty() {
            return self.create_empty_dataframe();
        }

        // Concatenate all chunks into final DataFrame
        let final_df = if dataframes.len() == 1 {
            dataframes.into_iter().next().unwrap()
        } else {
            // Convert DataFrames to LazyFrames for concatenation with explicit schema preservation
            let lazy_frames: Vec<LazyFrame> = dataframes
                .into_iter()
                .map(|df| {
                    // Ensure consistent schema before concatenation
                    df.lazy().with_columns([
                        col("x").cast(DataType::Int16),
                        col("y").cast(DataType::Int16),
                        col("polarity").cast(DataType::Int8),
                        col("timestamp").cast(DataType::Duration(TimeUnit::Microseconds)),
                    ])
                })
                .collect();
            concat(&lazy_frames, UnionArgs::default())?.collect()?
        };

        // Final schema enforcement to ensure correct types
        let final_df_with_schema = final_df
            .lazy()
            .with_columns([
                col("x").cast(DataType::Int16),
                col("y").cast(DataType::Int16),
                col("polarity").cast(DataType::Int8),
                col("timestamp").cast(DataType::Duration(TimeUnit::Microseconds)),
            ])
            .collect()?;

        Ok(final_df_with_schema)
    }

    /// Build a Polars DataFrame from a chunk of events
    ///
    /// This method reuses the existing optimized DataFrame construction logic
    /// to ensure consistency with direct loading.
    ///
    /// # Arguments
    /// * `events` - Slice of Event objects to process
    ///
    /// # Returns
    /// Result containing a Polars DataFrame for this chunk
    #[cfg(feature = "polars")]
    pub fn build_chunk(&self, events: &[Event]) -> PolarsResult<DataFrame> {
        use polars::prelude::*;

        let len = events.len();

        if len == 0 {
            return self.create_empty_dataframe();
        }

        // Use optimal data types for memory efficiency
        let mut x_builder = PrimitiveChunkedBuilder::<Int16Type>::new("x".into(), len);
        let mut y_builder = PrimitiveChunkedBuilder::<Int16Type>::new("y".into(), len);
        let mut timestamp_builder =
            PrimitiveChunkedBuilder::<Int64Type>::new("timestamp".into(), len);
        let mut polarity_builder = PrimitiveChunkedBuilder::<Int8Type>::new("polarity".into(), len);

        // Single iteration with direct population - zero intermediate copies
        // Store polarity as raw bool first, convert vectorized later
        for event in events {
            x_builder.append_value(event.x as i16);
            y_builder.append_value(event.y as i16);
            timestamp_builder.append_value(self.convert_timestamp(event.t));
            // Store raw bool polarity (0/1) - will convert vectorized later
            polarity_builder.append_value(if event.polarity { 1i8 } else { 0i8 });
        }

        // Build Series from builders
        let x_series = x_builder.finish().into_series();
        let y_series = y_builder.finish().into_series();
        let polarity_series_raw = polarity_builder.finish().into_series();

        // Convert timestamp to Duration type
        let timestamp_series = timestamp_builder
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
                            .alias("polarity"),
                    )
                    .collect()?
            }
            EventFormat::HDF5 => {
                // HDF5 format: Convert 0/1 to -1/1 for consistency
                df.lazy()
                    .with_column(
                        when(col("polarity").eq(lit(0)))
                            .then(lit(-1i8))
                            .otherwise(lit(1i8))
                            .alias("polarity"),
                    )
                    .collect()?
            }
            _ => {
                // Text and other formats: use 0/1 encoding directly
                df
            }
        };

        Ok(df)
    }

    /// Create an empty DataFrame with the correct schema
    #[cfg(feature = "polars")]
    fn create_empty_dataframe(&self) -> PolarsResult<DataFrame> {
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

    /// Convert event polarity based on format-specific encoding requirements
    ///
    /// Reuses the existing polarity conversion logic to ensure consistency
    pub fn convert_polarity(&self, polarity: bool) -> i8 {
        match self.format {
            EventFormat::EVT2 | EventFormat::EVT21 | EventFormat::EVT3 => {
                // EVT2 family uses -1/1 encoding
                if polarity {
                    1i8
                } else {
                    -1i8
                }
            }
            EventFormat::HDF5 => {
                // HDF5 format converts 0/1 to -1/1 for consistency
                if polarity {
                    1i8
                } else {
                    -1i8
                }
            }
            _ => {
                // Text and other formats use 0/1 encoding
                if polarity {
                    1i8
                } else {
                    0i8
                }
            }
        }
    }

    /// Convert timestamp to microseconds for Polars Duration type
    ///
    /// Reuses the existing timestamp conversion logic to ensure consistency
    pub fn convert_timestamp(&self, timestamp: f64) -> i64 {
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
}

/// Helper function to determine if streaming should be used based on event count
///
/// # Arguments
/// * `event_count` - Number of events to process
/// * `threshold` - Threshold for switching to streaming mode
///
/// # Returns
/// True if streaming should be used, false for direct loading
pub fn should_use_streaming(event_count: usize, threshold: Option<usize>) -> bool {
    let default_threshold = 5_000_000; // 5M events
    let actual_threshold = threshold.unwrap_or(default_threshold);
    event_count > actual_threshold
}

/// Memory usage estimation for event processing
///
/// # Arguments
/// * `event_count` - Number of events
///
/// # Returns
/// Estimated memory usage in bytes
pub fn estimate_memory_usage(event_count: usize) -> usize {
    // Conservative estimate including overhead
    // Each event: 24 bytes for Event struct + polars overhead
    const BYTES_PER_EVENT: usize = 30;
    event_count * BYTES_PER_EVENT
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ev_core::Event;

    #[test]
    fn test_calculate_optimal_chunk_size() {
        // Test with different memory sizes
        let chunk_size_256mb = PolarsEventStreamer::calculate_optimal_chunk_size(10_000_000, 256);
        let chunk_size_1gb = PolarsEventStreamer::calculate_optimal_chunk_size(10_000_000, 1024);

        assert!(chunk_size_256mb >= 100_000);
        assert!(chunk_size_256mb <= 10_000_000);
        assert!(chunk_size_1gb >= chunk_size_256mb);
    }

    #[test]
    fn test_should_use_streaming() {
        assert!(!should_use_streaming(1_000_000, None));
        assert!(should_use_streaming(10_000_000, None));
        assert!(should_use_streaming(1_000_000, Some(500_000)));
        assert!(!should_use_streaming(1_000_000, Some(2_000_000)));
    }

    #[test]
    fn test_estimate_memory_usage() {
        let usage_1m = estimate_memory_usage(1_000_000);
        let usage_10m = estimate_memory_usage(10_000_000);

        assert_eq!(usage_10m, usage_1m * 10);
        assert!(usage_1m > 0);
    }

    #[cfg(feature = "polars")]
    #[test]
    fn test_polars_event_streamer_empty() {
        let streamer = PolarsEventStreamer::new(1000, EventFormat::HDF5);
        let empty_events = Vec::<Event>::new();
        let result = streamer.stream_to_polars(empty_events.into_iter());

        assert!(result.is_ok());
        let df = result.unwrap();
        assert_eq!(df.height(), 0);
        assert_eq!(df.width(), 4);
    }

    #[cfg(feature = "polars")]
    #[test]
    fn test_polars_event_streamer_small_chunk() {
        let streamer = PolarsEventStreamer::new(2, EventFormat::HDF5);
        let events = vec![
            Event {
                t: 0.001,
                x: 100,
                y: 200,
                polarity: true,
            },
            Event {
                t: 0.002,
                x: 101,
                y: 201,
                polarity: false,
            },
            Event {
                t: 0.003,
                x: 102,
                y: 202,
                polarity: true,
            },
        ];

        let result = streamer.stream_to_polars(events.into_iter());
        assert!(result.is_ok());

        let df = result.unwrap();
        assert_eq!(df.height(), 3);
        assert_eq!(df.width(), 4);

        // Verify column names
        let columns: Vec<&str> = df.get_column_names().iter().map(|s| s.as_str()).collect();
        assert_eq!(columns, vec!["x", "y", "timestamp", "polarity"]);
    }

    #[cfg(feature = "polars")]
    #[test]
    fn test_polarity_conversion() {
        let streamer_evt2 = PolarsEventStreamer::new(1000, EventFormat::EVT2);
        let streamer_hdf5 = PolarsEventStreamer::new(1000, EventFormat::HDF5);

        assert_eq!(streamer_evt2.convert_polarity(true), 1i8);
        assert_eq!(streamer_evt2.convert_polarity(false), -1i8);

        assert_eq!(streamer_hdf5.convert_polarity(true), 1i8);
        assert_eq!(streamer_hdf5.convert_polarity(false), -1i8);
    }

    #[cfg(feature = "polars")]
    #[test]
    fn test_timestamp_conversion() {
        let streamer = PolarsEventStreamer::new(1000, EventFormat::HDF5);

        // Test seconds to microseconds conversion
        assert_eq!(streamer.convert_timestamp(1.0), 1_000_000);
        assert_eq!(streamer.convert_timestamp(0.001), 1_000);

        // Test microseconds passthrough
        assert_eq!(streamer.convert_timestamp(1_000_000.0), 1_000_000);
        assert_eq!(streamer.convert_timestamp(2_000_000.0), 2_000_000);
    }
}
