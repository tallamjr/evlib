//! Utility functions and statistics for event filtering using Polars DataFrames
//!
//! This module provides various utility functions that work primarily with Polars
//! DataFrames for efficient event data processing and analysis.

use crate::ev_filtering::{FilterError, FilterResult};
// Removed: use crate::Events; - legacy type no longer exists
use polars::prelude::*;
#[cfg(feature = "tracing")]
use tracing::{debug, instrument, warn};

#[cfg(not(feature = "tracing"))]
macro_rules! debug {
    ($($args:tt)*) => {};
}

#[cfg(not(feature = "tracing"))]
macro_rules! info {
    ($($args:tt)*) => {};
}

#[cfg(not(feature = "tracing"))]
macro_rules! warn {
    ($($args:tt)*) => {
        eprintln!("[WARN] {}", format!($($args)*))
    };
}

#[cfg(not(feature = "tracing"))]
macro_rules! trace {
    ($($args:tt)*) => {};
}

#[cfg(not(feature = "tracing"))]
macro_rules! error {
    ($($args:tt)*) => {
        eprintln!("[ERROR] {}", format!($($args)*))
    };
}

#[cfg(not(feature = "tracing"))]
macro_rules! instrument {
    ($($args:tt)*) => {};
}

/// Polars column names for consistent DataFrame operations
pub const COL_X: &str = "x";
pub const COL_Y: &str = "y";
pub const COL_T: &str = "timestamp";
pub const COL_POLARITY: &str = "polarity";

/// Comprehensive event statistics calculated using Polars aggregations
#[derive(Debug, Clone)]
pub struct EventStats {
    pub count: u32,
    pub time_range: (f64, f64),
    pub spatial_bounds: (i64, i64, i64, i64),
    pub positive_events: u32,
    pub negative_events: u32,
    pub duration: f64,
    pub avg_event_rate: f64,
    pub unique_pixels: u32,
    pub temporal_std: f64,
    pub spatial_extent: (i64, i64),
}

impl EventStats {
    /// Calculate comprehensive event statistics using Polars operations
    ///
    /// This is the preferred method for calculating statistics as it leverages
    /// Polars' optimized aggregation functions.
    #[cfg_attr(feature = "tracing", instrument(skip(df)))]
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
                // TODO: Fix unique pixel calculation once struct function is available
                lit(0).alias("unique_pixels"), // Placeholder
            ])
            .with_columns([
                // Calculate derived columns
                (col("max_t") - col("min_t")).alias("duration"),
                (col("count") - col("positive_events")).alias("negative_events"),
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
            duration: row.0[10].try_extract::<f64>().unwrap_or(0.0),
            negative_events: row.0[11].try_extract::<u32>()?,
            spatial_extent: (
                row.0[12].try_extract::<i64>().unwrap_or(0),
                row.0[13].try_extract::<i64>().unwrap_or(0),
            ),
            avg_event_rate: row.0[14].try_extract::<f64>().unwrap_or(0.0),
        })
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
#[cfg_attr(feature = "tracing", instrument(skip(df)))]
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

/// Sort DataFrame by timestamp using Polars operations
#[cfg_attr(feature = "tracing", instrument(skip(df)))]
pub fn sort_events_by_time_polars(df: LazyFrame) -> PolarsResult<LazyFrame> {
    Ok(df.sort([COL_T], SortMultipleOptions::default()))
}

/// Check if events are sorted using Polars expressions
#[cfg_attr(feature = "tracing", instrument(skip(df)))]
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
#[cfg_attr(feature = "tracing", instrument(skip(df)))]
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

/// Memory and performance utilities
pub mod performance {
    use super::*;

    /// Estimate optimal processing strategy based on data characteristics
    #[cfg_attr(feature = "tracing", instrument(skip(df)))]
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

    impl ProcessingStrategy {
        /// Get recommended memory usage based on strategy
        pub fn estimated_memory_usage(&self, events_per_mb: f64) -> f64 {
            match self {
                ProcessingStrategy::Direct => 50.0,
                ProcessingStrategy::Optimized => 100.0,
                ProcessingStrategy::Chunked { chunk_size } => {
                    (*chunk_size as f64 / events_per_mb).max(50.0)
                }
                ProcessingStrategy::Streaming { chunk_size } => {
                    (*chunk_size as f64 / events_per_mb).max(20.0)
                }
            }
        }

        /// Get processing description
        pub fn description(&self) -> String {
            match self {
                ProcessingStrategy::Direct => "Direct processing".to_string(),
                ProcessingStrategy::Optimized => "Optimized parallel processing".to_string(),
                ProcessingStrategy::Chunked { chunk_size } => {
                    format!("Chunked processing ({}k events/chunk)", chunk_size / 1000)
                }
                ProcessingStrategy::Streaming { chunk_size } => {
                    format!("Streaming processing ({}k events/chunk)", chunk_size / 1000)
                }
            }
        }
    }

    /// Calculate memory requirements for a filtering operation
    pub fn estimate_memory_requirements(
        total_events: usize,
        processing_strategy: &ProcessingStrategy,
    ) -> MemoryRequirements {
        let base_event_size = 24; // bytes per event (struct Event)
        let polars_overhead = 1.5; // Polars DataFrame overhead factor
        let intermediate_buffer = 2.0; // Factor for intermediate results

        let base_memory = (total_events * base_event_size) as f64 * polars_overhead;

        match processing_strategy {
            ProcessingStrategy::Direct => MemoryRequirements {
                minimum: base_memory,
                recommended: base_memory * intermediate_buffer,
                maximum: base_memory * 3.0,
            },
            ProcessingStrategy::Optimized => MemoryRequirements {
                minimum: base_memory,
                recommended: base_memory * intermediate_buffer * 1.5,
                maximum: base_memory * 4.0,
            },
            ProcessingStrategy::Chunked { chunk_size } => {
                let chunk_memory = (*chunk_size * base_event_size) as f64 * polars_overhead;
                MemoryRequirements {
                    minimum: chunk_memory,
                    recommended: chunk_memory * intermediate_buffer,
                    maximum: chunk_memory * 3.0,
                }
            }
            ProcessingStrategy::Streaming { chunk_size } => {
                let chunk_memory = (*chunk_size * base_event_size) as f64 * polars_overhead;
                MemoryRequirements {
                    minimum: chunk_memory,
                    recommended: chunk_memory * 1.2,
                    maximum: chunk_memory * 2.0,
                }
            }
        }
    }

    /// Memory requirements for processing
    #[derive(Debug, Clone)]
    pub struct MemoryRequirements {
        /// Minimum memory required (MB)
        pub minimum: f64,
        /// Recommended memory for good performance (MB)
        pub recommended: f64,
        /// Maximum memory that could be used (MB)
        pub maximum: f64,
    }

    impl MemoryRequirements {
        /// Convert to megabytes
        pub fn to_mb(&self) -> (f64, f64, f64) {
            (
                self.minimum / 1_000_000.0,
                self.recommended / 1_000_000.0,
                self.maximum / 1_000_000.0,
            )
        }

        /// Get a human-readable description
        pub fn description(&self) -> String {
            let (min_mb, rec_mb, max_mb) = self.to_mb();
            format!(
                "Memory: {:.1}MB min, {:.1}MB recommended, {:.1}MB max",
                min_mb, rec_mb, max_mb
            )
        }
    }
}
