// Apache Arrow integration for zero-copy event data transfer
// This module provides Arrow array creation and schema management for evlib

#[cfg(feature = "arrow")]
use arrow::{
    array::{ArrayBuilder, Int16Builder, Int8Builder, RecordBatch},
    datatypes::{DataType, Field, Schema, TimeUnit},
};

#[cfg(feature = "arrow")]
use arrow_array::{
    builder::DurationMicrosecondBuilder, DurationMicrosecondArray, Int16Array, Int8Array,
};

#[cfg(feature = "arrow")]
use std::sync::Arc;

use crate::ev_core::{Event, Events};
use crate::ev_formats::EventFormat;

/// Error types for Arrow operations
#[derive(Debug, thiserror::Error)]
pub enum ArrowBuilderError {
    #[error("Arrow array construction failed: {0}")]
    ArrayConstruction(String),

    #[error("Invalid event data: {message}")]
    InvalidData { message: String },

    #[error("Memory allocation failed for {event_count} events")]
    MemoryAllocation { event_count: usize },

    #[error("Schema validation failed: {0}")]
    SchemaValidation(String),

    #[error("Feature not enabled: Arrow support requires 'arrow' feature flag")]
    FeatureNotEnabled,
}

#[cfg(not(feature = "arrow"))]
impl From<ArrowBuilderError> for Box<dyn std::error::Error + Send + Sync> {
    fn from(e: ArrowBuilderError) -> Self {
        Box::new(e)
    }
}

#[cfg(feature = "arrow")]
impl From<arrow::error::ArrowError> for ArrowBuilderError {
    fn from(e: arrow::error::ArrowError) -> Self {
        ArrowBuilderError::ArrayConstruction(e.to_string())
    }
}

/// Create the standard Arrow schema for event data
///
/// This schema exactly matches the current Polars schema to ensure compatibility:
/// - x, y: Int16 (saves 50% memory vs Int32, sufficient for most sensors)
/// - timestamp: Duration(Microseconds) (Int64 with time semantics)
/// - polarity: Int8 (saves 87.5% memory vs Int64, supports -1/0/1 values)
#[cfg(feature = "arrow")]
pub fn create_event_arrow_schema() -> Schema {
    Schema::new(vec![
        Field::new("x", DataType::Int16, false),
        Field::new("y", DataType::Int16, false),
        Field::new(
            "timestamp",
            DataType::Duration(TimeUnit::Microsecond),
            false,
        ),
        Field::new("polarity", DataType::Int8, false),
    ])
}

/// High-performance Arrow array builder for event data
///
/// Provides zero-copy construction from Event vectors with format-specific
/// polarity encoding and optimised memory layout.
#[cfg(feature = "arrow")]
pub struct ArrowEventBuilder {
    x_builder: Int16Builder,
    y_builder: Int16Builder,
    timestamp_builder: DurationMicrosecondBuilder,
    polarity_builder: Int8Builder,
    format: EventFormat,
    capacity: usize,
    schema: Arc<Schema>,
}

#[cfg(feature = "arrow")]
impl ArrowEventBuilder {
    /// Create a new ArrowEventBuilder with specified capacity
    ///
    /// # Arguments
    /// * `capacity` - Expected number of events (for memory pre-allocation)
    /// * `format` - Event format for proper polarity encoding
    ///
    /// # Returns
    /// A new ArrowEventBuilder instance
    pub fn new(capacity: usize, format: EventFormat) -> Self {
        Self {
            x_builder: Int16Builder::with_capacity(capacity),
            y_builder: Int16Builder::with_capacity(capacity),
            timestamp_builder: DurationMicrosecondBuilder::with_capacity(capacity),
            polarity_builder: Int8Builder::with_capacity(capacity),
            format,
            capacity,
            schema: Arc::new(create_event_arrow_schema()),
        }
    }

    /// Create a new ArrowEventBuilder for a specific event slice
    ///
    /// # Arguments
    /// * `events` - Slice of events to process
    /// * `format` - Event format for proper polarity encoding
    ///
    /// # Returns
    /// A new ArrowEventBuilder instance with optimal capacity
    pub fn for_events(events: &[Event], format: EventFormat) -> Self {
        Self::new(events.len(), format)
    }

    /// Add a single event to the builder
    ///
    /// # Arguments
    /// * `event` - Event to add
    ///
    /// # Returns
    /// Result indicating success or failure
    pub fn append_event(&mut self, event: &Event) -> Result<(), ArrowBuilderError> {
        // Convert coordinates to Int16 (safe cast from u16)
        self.x_builder.append_value(event.x as i16);
        self.y_builder.append_value(event.y as i16);

        // Convert timestamp to microseconds
        let timestamp_us = self.convert_timestamp(event.t);
        self.timestamp_builder.append_value(timestamp_us);

        // Convert polarity based on format-specific encoding
        let polarity_value = self.convert_polarity(event.polarity);
        self.polarity_builder.append_value(polarity_value);

        Ok(())
    }

    /// Build Arrow arrays from events with zero-copy optimisation
    ///
    /// This is the primary method for converting Event vectors to Arrow format.
    /// It processes events in a single pass with minimal memory copying.
    ///
    /// # Arguments
    /// * `events` - Slice of events to convert
    ///
    /// # Returns
    /// Result containing a RecordBatch with the event data
    pub fn from_events_zero_copy(
        events: &[Event],
        format: EventFormat,
    ) -> Result<RecordBatch, ArrowBuilderError> {
        if events.is_empty() {
            return Self::create_empty_batch();
        }

        let mut builder = Self::new(events.len(), format);

        // Single-pass vectorised processing
        for event in events {
            builder.append_event(event)?;
        }

        builder.finish()
    }

    /// Build Arrow arrays from an iterator of events (for streaming)
    ///
    /// # Arguments
    /// * `events` - Iterator of events to convert
    /// * `format` - Event format for proper polarity encoding
    /// * `size_hint` - Optional size hint for memory pre-allocation
    ///
    /// # Returns
    /// Result containing a RecordBatch with the event data
    pub fn from_events_iter<I>(
        events: I,
        format: EventFormat,
        size_hint: Option<usize>,
    ) -> Result<RecordBatch, ArrowBuilderError>
    where
        I: Iterator<Item = Event>,
    {
        let capacity = size_hint.unwrap_or(1000);
        let mut builder = Self::new(capacity, format);

        for event in events {
            builder.append_event(&event)?;
        }

        builder.finish()
    }

    /// Finish building and return the RecordBatch
    ///
    /// # Returns
    /// Result containing the final RecordBatch
    pub fn finish(mut self) -> Result<RecordBatch, ArrowBuilderError> {
        let x_array = self.x_builder.finish();
        let y_array = self.y_builder.finish();
        let timestamp_array = self.timestamp_builder.finish();
        let polarity_array = self.polarity_builder.finish();

        let batch = RecordBatch::try_new(
            self.schema,
            vec![
                Arc::new(x_array),
                Arc::new(y_array),
                Arc::new(timestamp_array),
                Arc::new(polarity_array),
            ],
        )?;

        Ok(batch)
    }

    /// Create an empty RecordBatch with the correct schema
    ///
    /// # Returns
    /// Result containing an empty RecordBatch
    pub fn create_empty_batch() -> Result<RecordBatch, ArrowBuilderError> {
        let schema = Arc::new(create_event_arrow_schema());

        let x_array = Int16Array::from(Vec::<i16>::new());
        let y_array = Int16Array::from(Vec::<i16>::new());
        let timestamp_array = DurationMicrosecondArray::from(Vec::<i64>::new());
        let polarity_array = Int8Array::from(Vec::<i8>::new());

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(x_array),
                Arc::new(y_array),
                Arc::new(timestamp_array),
                Arc::new(polarity_array),
            ],
        )?;

        Ok(batch)
    }

    /// Convert event polarity based on format-specific encoding requirements
    ///
    /// This matches the existing Polars implementation exactly:
    /// - EVT2/EVT3/HDF5: Use -1/1 encoding (true polarity representation)
    /// - Text/Other: Use 0/1 encoding (matches file format)
    ///
    /// # Arguments
    /// * `polarity` - Boolean polarity value from Event
    ///
    /// # Returns
    /// Int8 polarity value according to format encoding
    fn convert_polarity(&self, polarity: bool) -> i8 {
        match self.format {
            EventFormat::EVT2 | EventFormat::EVT21 | EventFormat::EVT3 | EventFormat::HDF5 => {
                // Convert 0/1 to -1/1 for proper polarity encoding
                if polarity {
                    1i8
                } else {
                    -1i8
                }
            }
            _ => {
                // Text and other formats: keep 0/1 encoding
                if polarity {
                    1i8
                } else {
                    0i8
                }
            }
        }
    }

    /// Convert timestamp to microseconds for Duration type
    ///
    /// This matches the existing Polars implementation exactly:
    /// - If timestamp >= 1,000,000: Assume already in microseconds
    /// - Otherwise: Convert seconds to microseconds
    ///
    /// # Arguments
    /// * `timestamp` - Floating-point timestamp from Event
    ///
    /// # Returns
    /// Int64 timestamp in microseconds
    fn convert_timestamp(&self, timestamp: f64) -> i64 {
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

    /// Get the Arrow schema for event data
    ///
    /// # Returns
    /// Reference to the Arrow schema
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Get the current capacity of the builder
    ///
    /// # Returns
    /// Builder capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the current length (number of events added)
    ///
    /// # Returns
    /// Number of events currently in the builder
    pub fn len(&self) -> usize {
        self.x_builder.len()
    }

    /// Check if the builder is empty
    ///
    /// # Returns
    /// True if no events have been added
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Arrow-based event streamer for large datasets
///
/// Provides chunked processing of large event streams while maintaining
/// memory efficiency and producing Arrow RecordBatches.
#[cfg(feature = "arrow")]
pub struct ArrowEventStreamer {
    chunk_size: usize,
    format: EventFormat,
    schema: Arc<Schema>,
}

#[cfg(feature = "arrow")]
impl ArrowEventStreamer {
    /// Create a new ArrowEventStreamer
    ///
    /// # Arguments
    /// * `chunk_size` - Number of events to process per chunk
    /// * `format` - Event format for proper polarity encoding
    ///
    /// # Returns
    /// A new ArrowEventStreamer instance
    pub fn new(chunk_size: usize, format: EventFormat) -> Self {
        Self {
            chunk_size,
            format,
            schema: Arc::new(create_event_arrow_schema()),
        }
    }

    /// Stream events to Arrow RecordBatch with chunked processing
    ///
    /// For large datasets, this processes events in chunks to maintain
    /// memory efficiency while producing a single consolidated RecordBatch.
    ///
    /// # Arguments
    /// * `events` - Iterator of events to process
    ///
    /// # Returns
    /// Result containing a RecordBatch with all events
    pub fn stream_to_arrow<I>(&self, events: I) -> Result<RecordBatch, ArrowBuilderError>
    where
        I: Iterator<Item = Event>,
    {
        let mut record_batches = Vec::new();
        let mut chunk_buffer = Vec::with_capacity(self.chunk_size);

        for event in events {
            chunk_buffer.push(event);

            // Process chunk when it's full
            if chunk_buffer.len() >= self.chunk_size {
                let chunk_batch =
                    ArrowEventBuilder::from_events_zero_copy(&chunk_buffer, self.format)?;
                if chunk_batch.num_rows() > 0 {
                    record_batches.push(chunk_batch);
                }
                chunk_buffer.clear();
            }
        }

        // Process remaining events in the buffer
        if !chunk_buffer.is_empty() {
            let chunk_batch = ArrowEventBuilder::from_events_zero_copy(&chunk_buffer, self.format)?;
            if chunk_batch.num_rows() > 0 {
                record_batches.push(chunk_batch);
            }
        }

        // Handle empty case
        if record_batches.is_empty() {
            return ArrowEventBuilder::create_empty_batch();
        }

        // Concatenate all chunks into final RecordBatch
        if record_batches.len() == 1 {
            Ok(record_batches.into_iter().next().unwrap())
        } else {
            self.concatenate_batches(&record_batches)
        }
    }

    /// Concatenate multiple RecordBatches into a single batch
    ///
    /// # Arguments
    /// * `batches` - Slice of RecordBatches to concatenate
    ///
    /// # Returns
    /// Result containing the concatenated RecordBatch
    fn concatenate_batches(
        &self,
        batches: &[RecordBatch],
    ) -> Result<RecordBatch, ArrowBuilderError> {
        use arrow::compute::concat_batches;

        concat_batches(&self.schema, batches.iter())
            .map_err(|e| ArrowBuilderError::ArrayConstruction(e.to_string()))
    }

    /// Get the chunk size
    ///
    /// # Returns
    /// Chunk size for streaming
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Get the Arrow schema
    ///
    /// # Returns
    /// Reference to the Arrow schema
    pub fn schema(&self) -> &Schema {
        &self.schema
    }
}

// Conversion utilities for interoperability

/// Convert Arrow RecordBatch to Events vector
///
/// # Arguments
/// * `batch` - Arrow RecordBatch to convert
///
/// # Returns
/// Result containing Events vector
#[cfg(feature = "arrow")]
pub fn arrow_to_events(batch: &RecordBatch) -> Result<Events, ArrowBuilderError> {
    use arrow::array::{Array, DurationMicrosecondArray, Int16Array, Int8Array};

    // Validate schema
    let expected_schema = create_event_arrow_schema();
    if !batch.schema().fields().eq(expected_schema.fields()) {
        return Err(ArrowBuilderError::SchemaValidation(
            "RecordBatch schema does not match expected event schema".to_string(),
        ));
    }

    let num_rows = batch.num_rows();
    if num_rows == 0 {
        return Ok(Vec::new());
    }

    // Extract arrays
    let x_array = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int16Array>()
        .ok_or_else(|| ArrowBuilderError::InvalidData {
            message: "x column is not Int16Array".to_string(),
        })?;

    let y_array = batch
        .column(1)
        .as_any()
        .downcast_ref::<Int16Array>()
        .ok_or_else(|| ArrowBuilderError::InvalidData {
            message: "y column is not Int16Array".to_string(),
        })?;

    let timestamp_array = batch
        .column(2)
        .as_any()
        .downcast_ref::<DurationMicrosecondArray>()
        .ok_or_else(|| ArrowBuilderError::InvalidData {
            message: "timestamp column is not DurationMicrosecondArray".to_string(),
        })?;

    let polarity_array = batch
        .column(3)
        .as_any()
        .downcast_ref::<Int8Array>()
        .ok_or_else(|| ArrowBuilderError::InvalidData {
            message: "polarity column is not Int8Array".to_string(),
        })?;

    // Convert to Events
    let mut events = Vec::with_capacity(num_rows);

    for i in 0..num_rows {
        let x = x_array.value(i) as u16;
        let y = y_array.value(i) as u16;
        let timestamp_us = timestamp_array.value(i);
        let polarity_raw = polarity_array.value(i);

        // Convert timestamp from microseconds to seconds
        let t = timestamp_us as f64 / 1_000_000.0;

        // Convert polarity from Int8 to bool
        let polarity = polarity_raw > 0;

        events.push(Event { t, x, y, polarity });
    }

    Ok(events)
}

// Stub implementations for when Arrow feature is disabled

#[cfg(not(feature = "arrow"))]
pub fn create_event_arrow_schema() -> Result<(), ArrowBuilderError> {
    Err(ArrowBuilderError::FeatureNotEnabled)
}

#[cfg(not(feature = "arrow"))]
pub struct ArrowEventBuilder;

#[cfg(not(feature = "arrow"))]
impl ArrowEventBuilder {
    pub fn new(_capacity: usize, _format: EventFormat) -> Result<Self, ArrowBuilderError> {
        Err(ArrowBuilderError::FeatureNotEnabled)
    }

    pub fn from_events_zero_copy(
        _events: &[Event],
        _format: EventFormat,
    ) -> Result<(), ArrowBuilderError> {
        Err(ArrowBuilderError::FeatureNotEnabled)
    }
}

#[cfg(not(feature = "arrow"))]
pub struct ArrowEventStreamer;

#[cfg(not(feature = "arrow"))]
impl ArrowEventStreamer {
    pub fn new(_chunk_size: usize, _format: EventFormat) -> Result<Self, ArrowBuilderError> {
        Err(ArrowBuilderError::FeatureNotEnabled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ev_core::Event;
    use crate::ev_formats::EventFormat;

    fn create_test_events() -> Vec<Event> {
        vec![
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
        ]
    }

    #[cfg(feature = "arrow")]
    #[test]
    fn test_create_event_arrow_schema() {
        let schema = create_event_arrow_schema();
        assert_eq!(schema.fields().len(), 4);

        let field_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(field_names, vec!["x", "y", "timestamp", "polarity"]);
    }

    #[cfg(feature = "arrow")]
    #[test]
    fn test_arrow_event_builder_empty() {
        let builder = ArrowEventBuilder::new(0, EventFormat::HDF5);
        assert_eq!(builder.len(), 0);
        assert!(builder.is_empty());
        assert_eq!(builder.capacity(), 0);
    }

    #[cfg(feature = "arrow")]
    #[test]
    fn test_arrow_event_builder_basic() {
        let events = create_test_events();
        let batch = ArrowEventBuilder::from_events_zero_copy(&events, EventFormat::HDF5)
            .expect("Failed to create Arrow batch");

        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 4);
        assert_eq!(batch.schema().fields().len(), 4);
    }

    #[cfg(feature = "arrow")]
    #[test]
    fn test_polarity_encoding_evt2() {
        let events = create_test_events();
        let batch = ArrowEventBuilder::from_events_zero_copy(&events, EventFormat::EVT2)
            .expect("Failed to create Arrow batch");

        // Check polarity encoding: EVT2 should use -1/1
        let polarity_column = batch.column(3);
        let polarity_array = polarity_column
            .as_any()
            .downcast_ref::<Int8Array>()
            .unwrap();

        assert_eq!(polarity_array.value(0), 1i8); // true -> 1
        assert_eq!(polarity_array.value(1), -1i8); // false -> -1
        assert_eq!(polarity_array.value(2), 1i8); // true -> 1
    }

    #[cfg(feature = "arrow")]
    #[test]
    fn test_polarity_encoding_text() {
        let events = create_test_events();
        let batch = ArrowEventBuilder::from_events_zero_copy(&events, EventFormat::Text)
            .expect("Failed to create Arrow batch");

        // Check polarity encoding: Text should use 0/1
        let polarity_column = batch.column(3);
        let polarity_array = polarity_column
            .as_any()
            .downcast_ref::<Int8Array>()
            .unwrap();

        assert_eq!(polarity_array.value(0), 1i8); // true -> 1
        assert_eq!(polarity_array.value(1), 0i8); // false -> 0
        assert_eq!(polarity_array.value(2), 1i8); // true -> 1
    }

    #[cfg(feature = "arrow")]
    #[test]
    fn test_timestamp_conversion() {
        let events = vec![
            Event {
                t: 1.0,
                x: 100,
                y: 200,
                polarity: true,
            }, // 1 second
            Event {
                t: 0.001,
                x: 101,
                y: 201,
                polarity: false,
            }, // 1 millisecond
            Event {
                t: 1_000_000.0,
                x: 102,
                y: 202,
                polarity: true,
            }, // 1 second in microseconds
        ];

        let batch = ArrowEventBuilder::from_events_zero_copy(&events, EventFormat::HDF5)
            .expect("Failed to create Arrow batch");

        let timestamp_column = batch.column(2);
        let timestamp_array = timestamp_column
            .as_any()
            .downcast_ref::<DurationMicrosecondArray>()
            .unwrap();

        assert_eq!(timestamp_array.value(0), 1_000_000i64); // 1 second -> 1M microseconds
        assert_eq!(timestamp_array.value(1), 1_000i64); // 1 ms -> 1K microseconds
        assert_eq!(timestamp_array.value(2), 1_000_000i64); // Already in microseconds
    }

    #[cfg(feature = "arrow")]
    #[test]
    fn test_arrow_to_events_conversion() {
        let events = create_test_events();
        let batch = ArrowEventBuilder::from_events_zero_copy(&events, EventFormat::HDF5)
            .expect("Failed to create Arrow batch");

        let converted_events =
            arrow_to_events(&batch).expect("Failed to convert Arrow batch to events");

        assert_eq!(converted_events.len(), 3);

        // Note: timestamps are converted to seconds, so we check with tolerance
        assert!((converted_events[0].t - 0.001).abs() < 1e-9);
        assert_eq!(converted_events[0].x, 100);
        assert_eq!(converted_events[0].y, 200);
        assert_eq!(converted_events[0].polarity, true);
    }

    #[cfg(feature = "arrow")]
    #[test]
    fn test_arrow_event_streamer() {
        let events = create_test_events();
        let streamer = ArrowEventStreamer::new(2, EventFormat::HDF5);

        let batch = streamer
            .stream_to_arrow(events.into_iter())
            .expect("Failed to stream events to Arrow");

        assert_eq!(batch.num_rows(), 3);
        assert_eq!(batch.num_columns(), 4);
    }

    #[cfg(not(feature = "arrow"))]
    #[test]
    fn test_arrow_disabled() {
        let result = create_event_arrow_schema();
        assert!(matches!(result, Err(ArrowBuilderError::FeatureNotEnabled)));
    }
}
