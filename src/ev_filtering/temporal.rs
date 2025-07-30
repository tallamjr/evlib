//! Polars-first temporal filtering operations for event camera data
//!
//! This module provides time-based filtering functionality using Polars DataFrames
//! and LazyFrames for maximum performance and memory efficiency. All operations
//! work directly with Polars expressions and avoid unnecessary conversions.
//!
//! # Philosophy
//!
//! This implementation follows a strict Polars-first approach:
//! - Input: LazyFrame (from events_to_dataframe)
//! - Processing: Polars expressions and transformations
//! - Output: LazyFrame (convertible to Vec<Event>/numpy only when needed)
//!
//! # Performance Benefits
//!
//! - Lazy evaluation: Operations are optimized and executed only when needed
//! - Vectorized operations: All filtering uses SIMD-optimized Polars operations
//! - Memory efficiency: No intermediate Vec<Event> allocations
//! - Query optimization: Polars optimizes the entire filtering pipeline
//!
//! # Example
//!
//! ```rust
//! use polars::prelude::*;
//! use evlib::ev_filtering::temporal::*;
//!
//! // Convert events to LazyFrame once
//! let events_df = events_to_dataframe(&events)?.lazy();
//!
//! // Apply temporal filtering with Polars expressions
//! let filtered = apply_temporal_filter(events_df, &TemporalFilter::time_window(1.0, 5.0))?;
//! ```

use crate::ev_core::{Event, Events};
use crate::ev_filtering::config::Validatable;
use crate::ev_filtering::{FilterError, FilterResult, SingleFilter};
use polars::prelude::*;
use tracing::{debug, info, instrument, warn};

/// Polars column names for event data
pub const COL_X: &str = "x";
pub const COL_Y: &str = "y";
pub const COL_T: &str = "t";
pub const COL_POLARITY: &str = "polarity";

/// Temporal filtering configuration optimized for Polars operations
#[derive(Debug, Clone, PartialEq)]
pub struct TemporalFilter {
    /// Start time (inclusive) in seconds
    pub t_start: Option<f64>,
    /// End time (inclusive) in seconds
    pub t_end: Option<f64>,
    /// Duration from start time (alternative to t_end)
    pub duration: Option<f64>,
    /// Fraction of middle portion to keep (0.0 to 1.0)
    pub middle_fraction: Option<f64>,
    /// Whether to use microsecond precision for filtering
    pub microsecond_precision: bool,
}

impl TemporalFilter {
    /// Create a new temporal filter with time window
    ///
    /// # Arguments
    ///
    /// * `t_start` - Start time in seconds (inclusive)
    /// * `t_end` - End time in seconds (inclusive)
    ///
    /// # Example
    ///
    /// ```rust
    /// use evlib::ev_filtering::temporal::TemporalFilter;
    ///
    /// // Filter events between 1.0 and 5.0 seconds
    /// let filter = TemporalFilter::time_window(1.0, 5.0);
    /// ```
    pub fn time_window(t_start: f64, t_end: f64) -> Self {
        Self {
            t_start: Some(t_start),
            t_end: Some(t_end),
            duration: None,
            middle_fraction: None,
            microsecond_precision: false,
        }
    }

    /// Create a temporal filter with start time and duration
    pub fn duration(t_start: f64, duration: f64) -> Self {
        Self {
            t_start: Some(t_start),
            t_end: None,
            duration: Some(duration),
            middle_fraction: None,
            microsecond_precision: false,
        }
    }

    /// Create a temporal filter for the middle fraction of data
    ///
    /// # Arguments
    ///
    /// * `fraction` - Fraction of the middle portion to keep (0.0 to 1.0)
    ///
    /// This is useful for removing startup/shutdown artifacts from recordings.
    pub fn middle_fraction(fraction: f64) -> Self {
        Self {
            t_start: None,
            t_end: None,
            duration: None,
            middle_fraction: Some(fraction),
            microsecond_precision: false,
        }
    }

    /// Create a filter that keeps events from a specific time onward
    pub fn from_time(t_start: f64) -> Self {
        Self {
            t_start: Some(t_start),
            t_end: None,
            duration: None,
            middle_fraction: None,
            microsecond_precision: false,
        }
    }

    /// Create a filter that keeps events until a specific time
    pub fn until_time(t_end: f64) -> Self {
        Self {
            t_start: None,
            t_end: Some(t_end),
            duration: None,
            middle_fraction: None,
            microsecond_precision: false,
        }
    }

    /// Enable microsecond precision filtering (for high-precision timestamps)
    pub fn with_microsecond_precision(mut self) -> Self {
        self.microsecond_precision = true;
        self
    }

    /// Convert this filter to Polars expressions
    ///
    /// This is the core of the Polars-first approach - we build Polars expressions
    /// that can be optimized and executed efficiently by the Polars query engine.
    pub fn to_polars_expr(&self, df: &LazyFrame) -> PolarsResult<Option<Expr>> {
        let mut conditions = Vec::new();

        // Handle middle fraction calculation (requires data inspection)
        if let Some(fraction) = self.middle_fraction {
            return self.build_middle_fraction_expr(df, fraction);
        }

        // Handle start time condition
        if let Some(t_start) = self.t_start {
            let start_expr = col(COL_T).gt_eq(lit(t_start));
            conditions.push(start_expr);
        }

        // Handle end time condition (either explicit or calculated from duration)
        let t_end = if let Some(duration) = self.duration {
            self.t_start.map(|start| start + duration)
        } else {
            self.t_end
        };

        if let Some(end_time) = t_end {
            let end_expr = col(COL_T).lt_eq(lit(end_time));
            conditions.push(end_expr);
        }

        // Combine all conditions with AND
        match conditions.len() {
            0 => Ok(None), // No filtering needed
            1 => Ok(Some(conditions.into_iter().next().unwrap())),
            _ => {
                let combined = conditions
                    .into_iter()
                    .reduce(|acc, cond| acc.and(cond))
                    .unwrap();
                Ok(Some(combined))
            }
        }
    }

    /// Build expression for middle fraction filtering
    fn build_middle_fraction_expr(
        &self,
        df: &LazyFrame,
        fraction: f64,
    ) -> PolarsResult<Option<Expr>> {
        if fraction <= 0.0 || fraction > 1.0 {
            warn!(
                "Middle fraction should be between 0.0 and 1.0, got {}",
                fraction
            );
            return Ok(None);
        }

        // We need to calculate the time bounds based on actual data
        // This requires collecting min/max which we'll do lazily
        let time_bounds = df
            .clone()
            .select([
                col(COL_T).min().alias("t_min"),
                col(COL_T).max().alias("t_max"),
            ])
            .collect()?;

        let t_min = time_bounds.column("t_min")?.get(0)?.try_extract::<f64>()?;
        let t_max = time_bounds.column("t_max")?.get(0)?.try_extract::<f64>()?;

        let duration = t_max - t_min;
        let margin = duration * (1.0 - fraction) / 2.0;
        let t_start = t_min + margin;
        let t_end = t_max - margin;

        info!(
            "Middle fraction {}: filtering to [{:.6}, {:.6}] from [{:.6}, {:.6}]",
            fraction, t_start, t_end, t_min, t_max
        );

        Ok(Some(
            col(COL_T)
                .gt_eq(lit(t_start))
                .and(col(COL_T).lt_eq(lit(t_end))),
        ))
    }
}

impl SingleFilter for TemporalFilter {
    fn apply(&self, events: &Events) -> FilterResult<Events> {
        // Legacy Vec<Event> interface - convert to DataFrame and back
        // This is for backward compatibility only
        warn!("Using legacy Vec<Event> interface - consider using LazyFrame directly for better performance");

        let df = crate::ev_core::events_to_dataframe(events)
            .map_err(|e| {
                FilterError::ProcessingError(format!("DataFrame conversion failed: {}", e))
            })?
            .lazy();

        let filtered_df = apply_temporal_filter(df, self)
            .map_err(|e| FilterError::ProcessingError(format!("Polars filtering failed: {}", e)))?;

        // Convert back to Vec<Event> - this is inefficient but maintains compatibility
        let result_df = filtered_df.collect().map_err(|e| {
            FilterError::ProcessingError(format!("LazyFrame collection failed: {}", e))
        })?;

        // Convert DataFrame back to Events
        dataframe_to_events(&result_df)
    }

    fn description(&self) -> String {
        match (
            &self.t_start,
            &self.t_end,
            &self.duration,
            &self.middle_fraction,
        ) {
            (Some(start), Some(end), _, _) => {
                format!("Temporal filter: [{:.3}, {:.3}]s", start, end)
            }
            (Some(start), _, Some(dur), _) => {
                format!("Temporal filter: {:.3}s + {:.3}s duration", start, dur)
            }
            (Some(start), _, _, _) => format!("Temporal filter: from {:.3}s", start),
            (_, Some(end), _, _) => format!("Temporal filter: until {:.3}s", end),
            (_, _, _, Some(frac)) => format!("Temporal filter: middle {:.1}%", frac * 100.0),
            _ => "Temporal filter: no constraints".to_string(),
        }
    }

    fn is_enabled(&self) -> bool {
        self.t_start.is_some()
            || self.t_end.is_some()
            || self.duration.is_some()
            || self.middle_fraction.is_some()
    }
}

impl Validatable for TemporalFilter {
    fn validate(&self) -> Result<(), FilterError> {
        // Validate time bounds
        if let (Some(start), Some(end)) = (self.t_start, self.t_end) {
            if start >= end {
                return Err(FilterError::InvalidConfig(format!(
                    "Start time ({}) must be less than end time ({})",
                    start, end
                )));
            }
        }

        // Validate duration
        if let Some(duration) = self.duration {
            if duration <= 0.0 {
                return Err(FilterError::InvalidConfig(format!(
                    "Duration must be positive, got {}",
                    duration
                )));
            }
        }

        // Validate middle fraction
        if let Some(fraction) = self.middle_fraction {
            if fraction <= 0.0 || fraction > 1.0 {
                return Err(FilterError::InvalidConfig(format!(
                    "Middle fraction must be between 0.0 and 1.0, got {}",
                    fraction
                )));
            }
        }

        Ok(())
    }
}

/// Apply temporal filtering using Polars expressions
///
/// This is the main temporal filtering function that works entirely with Polars
/// operations for maximum performance.
///
/// # Arguments
///
/// * `df` - Input LazyFrame containing event data
/// * `filter` - Temporal filter configuration
///
/// # Returns
///
/// Filtered LazyFrame with temporal constraints applied
///
/// # Example
///
/// ```rust
/// use polars::prelude::*;
/// use evlib::ev_filtering::temporal::*;
///
/// let events_df = events_to_dataframe(&events)?.lazy();
/// let filter = TemporalFilter::time_window(1.0, 5.0);
/// let filtered = apply_temporal_filter(events_df, &filter)?;
/// ```
#[instrument(skip(df), fields(filter = ?filter))]
pub fn apply_temporal_filter(df: LazyFrame, filter: &TemporalFilter) -> PolarsResult<LazyFrame> {
    debug!("Applying temporal filter: {:?}", filter);

    match filter.to_polars_expr(&df)? {
        Some(expr) => {
            debug!("Temporal filter expression: {:?}", expr);
            Ok(df.filter(expr))
        }
        None => {
            debug!("No temporal filtering needed");
            Ok(df)
        }
    }
}

/// Filter events by time window using Polars expressions
///
/// # Arguments
///
/// * `df` - Input LazyFrame
/// * `t_start` - Start time in seconds (None for no lower bound)
/// * `t_end` - End time in seconds (None for no upper bound)
///
/// # Returns
///
/// Filtered LazyFrame
#[instrument(skip(df))]
pub fn filter_time_window(
    df: LazyFrame,
    t_start: Option<f64>,
    t_end: Option<f64>,
) -> PolarsResult<LazyFrame> {
    let mut conditions = Vec::new();

    if let Some(start) = t_start {
        conditions.push(col(COL_T).gt_eq(lit(start)));
    }

    if let Some(end) = t_end {
        conditions.push(col(COL_T).lt_eq(lit(end)));
    }

    match conditions.len() {
        0 => Ok(df),
        1 => Ok(df.filter(conditions.into_iter().next().unwrap())),
        _ => {
            let combined = conditions
                .into_iter()
                .reduce(|acc, cond| acc.and(cond))
                .unwrap();
            Ok(df.filter(combined))
        }
    }
}

/// Legacy function for backward compatibility - delegates to Polars implementation
pub fn filter_by_time(events: &Events, t_start: f64, t_end: f64) -> FilterResult<Events> {
    let filter = TemporalFilter::time_window(t_start, t_end);
    filter.apply(events)
}

/// Get temporal statistics using Polars aggregations
///
/// This function computes comprehensive temporal statistics efficiently
/// using Polars' built-in aggregation functions.
///
/// # Arguments
///
/// * `df` - Input LazyFrame
///
/// # Returns
///
/// DataFrame containing temporal statistics
#[instrument(skip(df))]
pub fn get_temporal_statistics(df: LazyFrame) -> PolarsResult<DataFrame> {
    df.select([
        col(COL_T).min().alias("t_min"),
        col(COL_T).max().alias("t_max"),
        col(COL_T).mean().alias("t_mean"),
        col(COL_T).median().alias("t_median"),
        col(COL_T).std(1).alias("t_std"),
        (col(COL_T).max() - col(COL_T).min()).alias("duration"),
        len().alias("total_events"),
        (len().cast(DataType::Float64) / (col(COL_T).max() - col(COL_T).min())).alias("event_rate"),
    ])
    .collect()
}

/// Calculate event rates over time windows using Polars
///
/// This function bins events into time windows and calculates the event rate
/// for each window, which is useful for analyzing temporal patterns.
///
/// # Arguments
///
/// * `df` - Input LazyFrame
/// * `window_size` - Size of each time window in seconds
///
/// # Returns
///
/// DataFrame with time windows and corresponding event rates
#[instrument(skip(df))]
pub fn calculate_event_rates(df: LazyFrame, window_size: f64) -> PolarsResult<DataFrame> {
    // Create time bins using Polars expressions
    df.with_columns([((col(COL_T) / lit(window_size))
        .cast(DataType::Int64)
        .cast(DataType::Float64)
        * lit(window_size))
    .alias("time_bin")])
        .group_by([col("time_bin")])
        .agg([
            len().alias("event_count"),
            col(COL_T).min().alias("window_start"),
            col(COL_T).max().alias("window_end"),
        ])
        .with_columns([
            (col("event_count").cast(DataType::Float64) / lit(window_size)).alias("event_rate"),
        ])
        .sort(["time_bin"], SortMultipleOptions::default())
        .collect()
}

/// Helper function to convert DataFrame back to Events (for legacy compatibility)
fn dataframe_to_events(df: &DataFrame) -> FilterResult<Events> {
    let height = df.height();
    let mut events = Vec::with_capacity(height);

    let x_series = df
        .column(COL_X)
        .map_err(|e| FilterError::ProcessingError(format!("Missing x column: {}", e)))?;
    let y_series = df
        .column(COL_Y)
        .map_err(|e| FilterError::ProcessingError(format!("Missing y column: {}", e)))?;
    let t_series = df
        .column(COL_T)
        .map_err(|e| FilterError::ProcessingError(format!("Missing t column: {}", e)))?;
    let p_series = df
        .column(COL_POLARITY)
        .map_err(|e| FilterError::ProcessingError(format!("Missing polarity column: {}", e)))?;

    let x_values = x_series
        .i64()
        .map_err(|e| FilterError::ProcessingError(format!("X column type error: {}", e)))?;
    let y_values = y_series
        .i64()
        .map_err(|e| FilterError::ProcessingError(format!("Y column type error: {}", e)))?;
    let t_values = t_series
        .f64()
        .map_err(|e| FilterError::ProcessingError(format!("T column type error: {}", e)))?;
    let p_values = p_series
        .i64()
        .map_err(|e| FilterError::ProcessingError(format!("Polarity column type error: {}", e)))?;

    for i in 0..height {
        let x = x_values
            .get(i)
            .ok_or_else(|| FilterError::ProcessingError("Missing x value".to_string()))?
            as u16;
        let y = y_values
            .get(i)
            .ok_or_else(|| FilterError::ProcessingError("Missing y value".to_string()))?
            as u16;
        let t = t_values
            .get(i)
            .ok_or_else(|| FilterError::ProcessingError("Missing t value".to_string()))?;
        let p = p_values
            .get(i)
            .ok_or_else(|| FilterError::ProcessingError("Missing polarity value".to_string()))?
            > 0;

        events.push(Event {
            x,
            y,
            t,
            polarity: p,
        });
    }

    Ok(events)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ev_core::{events_to_dataframe, Event};

    fn create_test_events() -> Vec<Event> {
        vec![
            Event {
                x: 100,
                y: 200,
                t: 1.0,
                polarity: true,
            },
            Event {
                x: 150,
                y: 250,
                t: 2.0,
                polarity: false,
            },
            Event {
                x: 200,
                y: 300,
                t: 3.0,
                polarity: true,
            },
            Event {
                x: 250,
                y: 350,
                t: 4.0,
                polarity: false,
            },
            Event {
                x: 300,
                y: 400,
                t: 5.0,
                polarity: true,
            },
        ]
    }

    #[test]
    fn test_time_window_filtering_polars() -> PolarsResult<()> {
        let events = create_test_events();
        let df = events_to_dataframe(&events)?.lazy();

        let filtered = filter_time_window(df, Some(2.0), Some(4.0))?;
        let result = filtered.collect()?;

        assert_eq!(result.height(), 3); // Events at t=2,3,4

        let times: Vec<f64> = result.column(COL_T)?.f64()?.into_no_null_iter().collect();
        assert_eq!(times, vec![2.0, 3.0, 4.0]);

        Ok(())
    }

    #[test]
    fn test_temporal_filter_legacy_compatibility() {
        let events = create_test_events();
        let filter = TemporalFilter::time_window(2.0, 4.0);

        let filtered = filter.apply(&events).unwrap();
        assert_eq!(filtered.len(), 3); // Events at t=2,3,4
    }

    #[test]
    fn test_filter_by_time_legacy() {
        let events = create_test_events();
        let filtered = filter_by_time(&events, 2.0, 4.0).unwrap();
        assert_eq!(filtered.len(), 3); // Events at t=2,3,4
    }

    #[test]
    fn test_temporal_statistics() -> PolarsResult<()> {
        let events = create_test_events();
        let df = events_to_dataframe(&events)?.lazy();

        let stats = get_temporal_statistics(df)?;

        assert_eq!(stats.height(), 1);
        assert_eq!(stats.width(), 8); // All statistics columns

        let t_min = stats.column("t_min")?.get(0)?.try_extract::<f64>()?;
        let t_max = stats.column("t_max")?.get(0)?.try_extract::<f64>()?;
        let duration = stats.column("duration")?.get(0)?.try_extract::<f64>()?;
        let total_events = stats.column("total_events")?.get(0)?.try_extract::<u32>()?;

        assert_eq!(t_min, 1.0);
        assert_eq!(t_max, 5.0);
        assert_eq!(duration, 4.0);
        assert_eq!(total_events, 5);

        Ok(())
    }

    #[test]
    fn test_middle_fraction_filtering() -> PolarsResult<()> {
        let events = create_test_events();
        let df = events_to_dataframe(&events)?.lazy();

        let filter = TemporalFilter::middle_fraction(0.6);
        let filtered = apply_temporal_filter(df, &filter)?;
        let result = filtered.collect()?;

        // Should filter out some events from start/end
        assert!(result.height() < events.len());
        assert!(result.height() > 0);

        Ok(())
    }

    #[test]
    fn test_filter_validation() {
        let valid_filter = TemporalFilter::time_window(1.0, 5.0);
        assert!(valid_filter.validate().is_ok());

        let invalid_filter = TemporalFilter::time_window(5.0, 1.0); // Start > end
        assert!(invalid_filter.validate().is_err());

        let invalid_fraction = TemporalFilter::middle_fraction(1.5); // > 1.0
        assert!(invalid_fraction.validate().is_err());
    }

    #[test]
    fn test_polars_expressions() -> PolarsResult<()> {
        let events = create_test_events();
        let df = events_to_dataframe(&events)?.lazy();

        let filter = TemporalFilter::time_window(2.0, 4.0);
        let expr = filter.to_polars_expr(&df)?;

        assert!(expr.is_some());

        // Test that the expression can be applied
        let filtered = df.filter(expr.unwrap());
        let result = filtered.collect()?;
        assert_eq!(result.height(), 3);

        Ok(())
    }
}
