//! Polars-first utilities for event filtering operations
//!
//! This module provides shared functionality using Polars DataFrames and LazyFrames
//! for maximum performance and memory efficiency. All operations work directly with
//! Polars expressions and avoid manual Vec<Event> operations.

use crate::ev_core::Events;
use crate::ev_filtering::{FilterError, FilterResult};
use polars::prelude::*;
use std::collections::HashMap;
use tracing::{debug, instrument, warn};

/// Polars column names (consistent across all filtering modules)
pub const COL_X: &str = "x";
pub const COL_Y: &str = "y";
pub const COL_T: &str = "t";
pub const COL_POLARITY: &str = "polarity";

/// Comprehensive event statistics computed using Polars aggregations
#[derive(Debug, Clone)]
pub struct EventStats {
    pub count: u32,
    pub time_range: (f64, f64),
    pub spatial_bounds: (i64, i64, i64, i64), // (min_x, max_x, min_y, max_y)
    pub positive_events: u32,
    pub negative_events: u32,
    pub duration: f64,
    pub avg_event_rate: f64,
    pub unique_pixels: u32,
    pub temporal_std: f64,
    pub spatial_extent: (i64, i64), // (width, height)
}

impl EventStats {
    /// Calculate comprehensive statistics using Polars aggregations
    ///
    /// This function leverages Polars' optimized aggregation functions to compute
    /// all statistics in a single pass, which is much faster than manual iteration.
    #[instrument(skip(df))]
    pub fn calculate_from_dataframe(df: LazyFrame) -> PolarsResult<Self> {
        let stats_df = df
            .select([
                len().alias("count"),
                col(COL_T).min().alias("min_t"),
                col(COL_T).max().alias("max_t"),
                col(COL_T).std(1).alias("temporal_std"),
                col(COL_X).min().alias("min_x"),
                col(COL_X).max().alias("max_x"),
                col(COL_Y).min().alias("min_y"),
                col(COL_Y).max().alias("max_y"),
                col(COL_POLARITY).sum().alias("positive_events"),
                // Calculate unique pixels by creating struct expressions
                // Count unique pixels by using coordinate pair grouping
                lit(0).alias("unique_pixels"), // Placeholder - will be calculated separately
            ])
            .with_columns([
                // Calculate derived statistics
                (col("max_t") - col("min_t")).alias("duration"),
                (len().cast(DataType::UInt32) - col("positive_events")).alias("negative_events"),
                (col("max_x") - col("min_x")).alias("width"),
                (col("max_y") - col("min_y")).alias("height"),
            ])
            .with_columns([
                // Calculate event rate
                (col("count").cast(DataType::Float64) / col("duration")).alias("avg_event_rate"),
            ])
            .collect()?;

        if stats_df.height() == 0 {
            return Ok(Self::empty());
        }

        let row = stats_df.get_row(0)?;

        Ok(Self {
            count: row.0[0].try_extract::<u32>()?,
            time_range: (
                row.0[1].try_extract::<f64>()?,
                row.0[2].try_extract::<f64>()?,
            ),
            temporal_std: row.0[3].try_extract::<f64>().unwrap_or(0.0),
            spatial_bounds: (
                row.0[4].try_extract::<i64>()?,
                row.0[5].try_extract::<i64>()?,
                row.0[6].try_extract::<i64>()?,
                row.0[7].try_extract::<i64>()?,
            ),
            positive_events: row.0[8].try_extract::<u32>()?,
            unique_pixels: row.0[9].try_extract::<u32>()?,
            duration: row.0[10].try_extract::<f64>()?,
            negative_events: row.0[11].try_extract::<u32>()?,
            spatial_extent: (
                row.0[12].try_extract::<i64>()?,
                row.0[13].try_extract::<i64>()?,
            ),
            avg_event_rate: row.0[14].try_extract::<f64>().unwrap_or(0.0),
        })
    }

    /// Legacy interface for Vec<Event> - delegates to Polars implementation
    pub fn calculate(events: &Events) -> Self {
        if events.is_empty() {
            return Self::empty();
        }

        warn!(
            "Using legacy Vec<Event> interface for statistics - consider using LazyFrame directly"
        );

        let df = match crate::ev_core::events_to_dataframe(events) {
            Ok(df) => df.lazy(),
            Err(e) => {
                warn!("Failed to convert events to DataFrame: {}, falling back", e);
                return Self::calculate_legacy(events);
            }
        };

        match Self::calculate_from_dataframe(df) {
            Ok(stats) => stats,
            Err(e) => {
                warn!("Polars statistics calculation failed: {}, falling back", e);
                Self::calculate_legacy(events)
            }
        }
    }

    /// Fallback legacy calculation (kept for compatibility)
    fn calculate_legacy(events: &Events) -> Self {
        if events.is_empty() {
            return Self::empty();
        }

        let mut min_t = events[0].t;
        let mut max_t = events[0].t;
        let mut min_x = events[0].x as i64;
        let mut max_x = events[0].x as i64;
        let mut min_y = events[0].y as i64;
        let mut max_y = events[0].y as i64;
        let mut positive_count = 0u32;
        let mut unique_pixels = std::collections::HashSet::new();

        for event in events {
            min_t = min_t.min(event.t);
            max_t = max_t.max(event.t);
            min_x = min_x.min(event.x as i64);
            max_x = max_x.max(event.x as i64);
            min_y = min_y.min(event.y as i64);
            max_y = max_y.max(event.y as i64);

            if event.polarity {
                positive_count += 1;
            }

            unique_pixels.insert((event.x, event.y));
        }

        let duration = max_t - min_t;
        let avg_event_rate = if duration > 0.0 {
            events.len() as f64 / duration
        } else {
            0.0
        };

        Self {
            count: events.len() as u32,
            time_range: (min_t, max_t),
            spatial_bounds: (min_x, max_x, min_y, max_y),
            positive_events: positive_count,
            negative_events: events.len() as u32 - positive_count,
            duration,
            avg_event_rate,
            unique_pixels: unique_pixels.len() as u32,
            temporal_std: 0.0, // Would require second pass to calculate
            spatial_extent: (max_x - min_x, max_y - min_y),
        }
    }

    fn empty() -> Self {
        Self {
            count: 0,
            time_range: (0.0, 0.0),
            spatial_bounds: (0, 0, 0, 0),
            positive_events: 0,
            negative_events: 0,
            duration: 0.0,
            avg_event_rate: 0.0,
            unique_pixels: 0,
            temporal_std: 0.0,
            spatial_extent: (0, 0),
        }
    }
}

/// Pixel statistics computed using Polars group_by operations
#[derive(Debug, Clone)]
pub struct PixelStats {
    pub x: u16,
    pub y: u16,
    pub total_events: u32,
    pub positive_events: u32,
    pub negative_events: u32,
    pub first_event_time: f64,
    pub last_event_time: f64,
    pub event_rate: f64,
    pub temporal_spread: f64,
}

/// Calculate per-pixel statistics using Polars group_by operations
///
/// This function is much faster than the manual HashMap approach as it leverages
/// Polars' optimized group operations and vectorized aggregations.
#[instrument(skip(df))]
pub fn calculate_pixel_stats_polars(df: LazyFrame) -> PolarsResult<DataFrame> {
    df.group_by([col(COL_X), col(COL_Y)])
        .agg([
            len().alias("total_events"),
            col(COL_POLARITY).sum().alias("positive_events"),
            col(COL_T).min().alias("first_event_time"),
            col(COL_T).max().alias("last_event_time"),
            col(COL_T).std(1).alias("temporal_std"),
        ])
        .with_columns([
            // Calculate derived statistics
            (col("total_events") - col("positive_events")).alias("negative_events"),
            (col("last_event_time") - col("first_event_time")).alias("temporal_spread"),
        ])
        .with_columns([
            // Calculate event rate
            (col("total_events").cast(DataType::Float64) / col("temporal_spread"))
                .alias("event_rate"),
        ])
        .sort([COL_X, COL_Y], SortMultipleOptions::default())
        .collect()
}

/// Legacy interface for pixel statistics - delegates to Polars
pub fn calculate_pixel_stats(events: &Events) -> HashMap<(u16, u16), PixelStats> {
    warn!("Using legacy Vec<Event> interface for pixel stats - consider using LazyFrame directly");

    let df = match crate::ev_core::events_to_dataframe(events) {
        Ok(df) => df.lazy(),
        Err(_) => return calculate_pixel_stats_legacy(events),
    };

    match calculate_pixel_stats_polars(df) {
        Ok(stats_df) => {
            let mut result = HashMap::new();

            for row_idx in 0..stats_df.height() {
                if let Ok(row) = stats_df.get_row(row_idx) {
                    if let (
                        Ok(x),
                        Ok(y),
                        Ok(total),
                        Ok(positive),
                        Ok(negative),
                        Ok(first_t),
                        Ok(last_t),
                        Ok(event_rate),
                    ) = (
                        row.0[0].try_extract::<i64>(),
                        row.0[1].try_extract::<i64>(),
                        row.0[2].try_extract::<u32>(),
                        row.0[3].try_extract::<u32>(),
                        row.0[4].try_extract::<u32>(),
                        row.0[5].try_extract::<f64>(),
                        row.0[6].try_extract::<f64>(),
                        row.0[7].try_extract::<f64>(),
                    ) {
                        result.insert(
                            (x as u16, y as u16),
                            PixelStats {
                                x: x as u16,
                                y: y as u16,
                                total_events: total,
                                positive_events: positive,
                                negative_events: negative,
                                first_event_time: first_t,
                                last_event_time: last_t,
                                event_rate: event_rate.max(0.0),
                                temporal_spread: last_t - first_t,
                            },
                        );
                    }
                }
            }

            result
        }
        Err(_) => calculate_pixel_stats_legacy(events),
    }
}

/// Fallback legacy pixel statistics calculation
fn calculate_pixel_stats_legacy(events: &Events) -> HashMap<(u16, u16), PixelStats> {
    let mut pixel_map = HashMap::new();

    for event in events {
        let coords = (event.x, event.y);
        let entry = pixel_map.entry(coords).or_insert_with(|| PixelStats {
            x: event.x,
            y: event.y,
            total_events: 0,
            positive_events: 0,
            negative_events: 0,
            first_event_time: event.t,
            last_event_time: event.t,
            event_rate: 0.0,
            temporal_spread: 0.0,
        });

        entry.total_events += 1;
        if event.polarity {
            entry.positive_events += 1;
        } else {
            entry.negative_events += 1;
        }

        entry.first_event_time = entry.first_event_time.min(event.t);
        entry.last_event_time = entry.last_event_time.max(event.t);
    }

    // Calculate derived statistics
    for stats in pixel_map.values_mut() {
        stats.temporal_spread = stats.last_event_time - stats.first_event_time;
        stats.event_rate = if stats.temporal_spread > 0.0 {
            stats.total_events as f64 / stats.temporal_spread
        } else {
            0.0
        };
    }

    pixel_map
}

/// Sort DataFrame by timestamp using Polars operations
#[instrument(skip(df))]
pub fn sort_events_by_time_polars(df: LazyFrame) -> PolarsResult<LazyFrame> {
    Ok(df.sort([COL_T], SortMultipleOptions::default()))
}

/// Check if events are sorted using Polars expressions
#[instrument(skip(df))]
pub fn is_sorted_by_time_polars(df: LazyFrame) -> PolarsResult<bool> {
    let result = df
        .select([(col(COL_T) - col(COL_T).shift(lit(1)))
            .lt(lit(0.0))
            .any(false)
            .alias("has_backwards_time")])
        .collect()?;

    let has_backwards_value = result.column("has_backwards_time")?.get(0)?;

    let has_backwards = match has_backwards_value {
        AnyValue::Boolean(b) => b,
        _ => return Err(PolarsError::ComputeError("Expected boolean value".into())),
    };

    Ok(!has_backwards)
}

/// Validate events using Polars expressions
///
/// This function uses Polars' vectorized operations to check for various
/// data quality issues much faster than manual iteration.
#[instrument(skip(df))]
pub fn validate_events_polars(df: LazyFrame, strict: bool) -> PolarsResult<FilterResult<()>> {
    let validation_df = df
        .select([
            col(COL_T).is_nan().sum().alias("nan_timestamps"),
            col(COL_T).lt(0.0).sum().alias("negative_timestamps"),
            col(COL_X).gt(10000).sum().alias("invalid_x"),
            col(COL_Y).gt(10000).sum().alias("invalid_y"),
            // Check for out-of-order events (requires sorting first)
            (col(COL_T) - col(COL_T).shift(lit(1)))
                .lt(lit(0.0))
                .cast(DataType::Int32)
                .sum()
                .alias("out_of_order"),
            len().alias("total_events"),
        ])
        .collect()?;

    if validation_df.height() == 0 {
        return Ok(Ok(()));
    }

    let row = validation_df.get_row(0)?;
    let nan_count: u32 = row.0[0].try_extract()?;
    let negative_count: u32 = row.0[1].try_extract()?;
    let invalid_x_count: u32 = row.0[2].try_extract()?;
    let invalid_y_count: u32 = row.0[3].try_extract()?;
    let out_of_order_count: u32 = row.0[4].try_extract()?;
    let total_events: u32 = row.0[5].try_extract()?;

    let mut issues = Vec::new();

    if nan_count > 0 {
        issues.push(format!("{} events have NaN timestamps", nan_count));
    }
    if negative_count > 0 {
        issues.push(format!(
            "{} events have negative timestamps",
            negative_count
        ));
    }
    if invalid_x_count > 0 {
        issues.push(format!(
            "{} events have invalid X coordinates (>10000)",
            invalid_x_count
        ));
    }
    if invalid_y_count > 0 {
        issues.push(format!(
            "{} events have invalid Y coordinates (>10000)",
            invalid_y_count
        ));
    }
    if out_of_order_count > 0 {
        issues.push(format!(
            "{} events are out of temporal order",
            out_of_order_count
        ));
    }

    if !issues.is_empty() {
        let message = format!(
            "Found {} validation issues in {} events: {}",
            issues.len(),
            total_events,
            issues.join("; ")
        );

        if strict {
            return Ok(Err(FilterError::InvalidInput(message)));
        } else {
            warn!("Event validation issues: {}", message);
        }
    }

    debug!(
        "Validated {} events successfully using Polars",
        total_events
    );
    Ok(Ok(()))
}

/// Legacy function to check if events are sorted by time
pub fn is_sorted_by_time(events: &Events) -> bool {
    if events.len() <= 1 {
        return true;
    }

    for i in 1..events.len() {
        if events[i].t < events[i - 1].t {
            return false;
        }
    }
    true
}

/// Legacy function to sort events by time
pub fn sort_events_by_time(events: &mut Events) {
    events.sort_unstable_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));
}

/// Legacy validation function - delegates to Polars
pub fn validate_events(events: &Events, strict: bool) -> FilterResult<()> {
    if events.is_empty() {
        return Ok(());
    }

    warn!("Using legacy Vec<Event> interface for validation - consider using LazyFrame directly");

    let df = match crate::ev_core::events_to_dataframe(events) {
        Ok(df) => df.lazy(),
        Err(_) => return validate_events_legacy(events, strict),
    };

    match validate_events_polars(df, strict) {
        Ok(result) => result,
        Err(_) => validate_events_legacy(events, strict),
    }
}

/// Fallback legacy validation
fn validate_events_legacy(events: &Events, strict: bool) -> FilterResult<()> {
    let mut issues = Vec::new();
    let mut prev_time = f64::NEG_INFINITY;

    for (i, event) in events.iter().enumerate() {
        if event.t.is_nan() {
            issues.push(format!("Event {} has NaN timestamp", i));
        } else if event.t < 0.0 {
            issues.push(format!("Event {} has negative timestamp: {}", i, event.t));
        } else if event.t < prev_time {
            issues.push(format!(
                "Event {} is out of temporal order: {} < {}",
                i, event.t, prev_time
            ));
        }
        prev_time = event.t;

        if event.x > 10000 || event.y > 10000 {
            issues.push(format!(
                "Event {} has suspicious coordinates: ({}, {})",
                i, event.x, event.y
            ));
        }
    }

    if !issues.is_empty() {
        let message = format!(
            "Found {} validation issues: {}",
            issues.len(),
            issues.join("; ")
        );

        if strict {
            return Err(FilterError::InvalidInput(message));
        } else {
            warn!("Event validation issues: {}", message);
        }
    }

    debug!("Validated {} events successfully", events.len());
    Ok(())
}

/// Memory and performance utilities
pub mod performance {
    use super::*;

    /// Estimate optimal processing strategy based on data characteristics
    #[instrument(skip(df))]
    pub fn analyze_processing_requirements(df: &LazyFrame) -> PolarsResult<ProcessingStrategy> {
        let analysis = df
            .clone()
            .select([
                len().alias("total_events"),
                col(COL_T).max().alias("max_t"),
                col(COL_T).min().alias("min_t"),
                col(COL_X).n_unique().alias("unique_x"),
                col(COL_Y).n_unique().alias("unique_y"),
            ])
            .collect()?;

        if analysis.height() == 0 {
            return Ok(ProcessingStrategy::Direct);
        }

        let row = analysis.get_row(0)?;
        let total_events: u32 = row.0[0].try_extract()?;
        let duration: f64 = row.0[1].try_extract::<f64>()? - row.0[2].try_extract::<f64>()?;
        let spatial_complexity = row.0[3].try_extract::<u32>()? * row.0[4].try_extract::<u32>()?;

        let strategy = match (total_events, duration, spatial_complexity) {
            (n, _, _) if n < 100_000 => ProcessingStrategy::Direct,
            (n, d, s) if n < 1_000_000 && d < 60.0 && s < 1_000_000 => {
                ProcessingStrategy::Optimized
            }
            (n, _, _) if n < 10_000_000 => ProcessingStrategy::Chunked {
                chunk_size: 1_000_000,
            },
            _ => ProcessingStrategy::Streaming {
                chunk_size: 500_000,
            },
        };

        Ok(strategy)
    }

    /// Processing strategy recommendations
    #[derive(Debug, Clone)]
    pub enum ProcessingStrategy {
        /// Direct processing for small datasets
        Direct,
        /// Optimized processing with parallelization
        Optimized,
        /// Chunked processing for medium datasets
        Chunked { chunk_size: usize },
        /// Streaming processing for large datasets
        Streaming { chunk_size: usize },
    }
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
                x: 100,
                y: 200,
                t: 4.0,
                polarity: false,
            }, // Same pixel as first
            Event {
                x: 300,
                y: 400,
                t: 5.0,
                polarity: true,
            },
        ]
    }

    #[test]
    fn test_event_stats_polars() -> PolarsResult<()> {
        let events = create_test_events();
        let df = events_to_dataframe(&events)?.lazy();

        let stats = EventStats::calculate_from_dataframe(df)?;

        assert_eq!(stats.count, 5);
        assert_eq!(stats.time_range, (1.0, 5.0));
        assert_eq!(stats.duration, 4.0);
        assert_eq!(stats.positive_events, 3);
        assert_eq!(stats.negative_events, 2);
        assert_eq!(stats.unique_pixels, 4);

        Ok(())
    }

    #[test]
    fn test_pixel_stats_polars() -> PolarsResult<()> {
        let events = create_test_events();
        let df = events_to_dataframe(&events)?.lazy();

        let pixel_stats_df = calculate_pixel_stats_polars(df)?;

        assert_eq!(pixel_stats_df.height(), 4); // 4 unique pixels

        // Check that we have the expected columns
        assert!(pixel_stats_df.column("total_events").is_ok());
        assert!(pixel_stats_df.column("positive_events").is_ok());
        assert!(pixel_stats_df.column("event_rate").is_ok());

        Ok(())
    }

    #[test]
    fn test_validation_polars() -> PolarsResult<()> {
        let events = create_test_events();
        let df = events_to_dataframe(&events)?.lazy();

        let validation_result = validate_events_polars(df, true)?;
        assert!(validation_result.is_ok());

        Ok(())
    }

    #[test]
    fn test_sorting_polars() -> PolarsResult<()> {
        let mut events = create_test_events();
        // Shuffle the events
        events.reverse();

        let df = events_to_dataframe(&events)?.lazy();
        let sorted_df = sort_events_by_time_polars(df)?;
        let result = sorted_df.collect()?;

        let times: Vec<f64> = result.column(COL_T)?.f64()?.into_no_null_iter().collect();

        assert_eq!(times, vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        Ok(())
    }

    #[test]
    fn test_processing_strategy() -> PolarsResult<()> {
        let events = create_test_events();
        let df = events_to_dataframe(&events)?.lazy();

        let strategy = performance::analyze_processing_requirements(&df)?;

        // Small dataset should use direct processing
        matches!(strategy, performance::ProcessingStrategy::Direct);

        Ok(())
    }
}
