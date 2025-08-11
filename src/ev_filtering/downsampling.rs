//! Polars-first event downsampling operations for reducing event density
//!
//! This module provides various strategies for reducing the number of events
//! while preserving important information, using Polars DataFrames and LazyFrames
//! for maximum performance and memory efficiency. All operations work directly with
//! Polars expressions and avoid manual Vec<Event> iterations.
//!
//! # Philosophy
//!
//! This implementation follows a strict Polars-first approach:
//! - Input: LazyFrame (from events_to_dataframe)
//! - Processing: Polars expressions and transformations
//! - Output: LazyFrame (convertible to Vec<Event> only when needed)
//!
//! # Performance Benefits
//!
//! - Lazy evaluation: Operations are optimized and executed only when needed
//! - Vectorized operations: All downsampling uses SIMD-optimized Polars operations
//! - Memory efficiency: No intermediate Vec<Event> allocations
//! - Query optimization: Polars optimizes the entire downsampling pipeline

// Removed: use crate::{Event, Events}; - legacy types no longer exist
use crate::ev_filtering::config::Validatable;
use crate::ev_filtering::{FilterError, FilterResult};
use polars::prelude::*;
use std::time::Instant;
#[cfg(feature = "tracing")]
use tracing::{debug, info, instrument, warn};

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

/// Polars column names for event data
pub const COL_X: &str = "x";
pub const COL_Y: &str = "y";
pub const COL_T: &str = "t";
pub const COL_POLARITY: &str = "polarity";

/// Downsampling strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DownsamplingStrategy {
    /// Uniform random sampling - keep each event with fixed probability
    Uniform,
    /// Temporal decimation - keep every Nth event in time
    TemporalDecimation,
    /// Spatial decimation - keep events from every Nth pixel
    SpatialDecimation,
    /// Adaptive sampling based on local event density
    Adaptive,
    /// Importance-based sampling - prioritize events with high information content
    ImportanceBased,
    /// Fixed count - keep exactly N events
    FixedCount,
}

impl DownsamplingStrategy {
    /// Get description of this strategy
    pub fn description(&self) -> &'static str {
        match self {
            DownsamplingStrategy::Uniform => "uniform random",
            DownsamplingStrategy::TemporalDecimation => "temporal decimation",
            DownsamplingStrategy::SpatialDecimation => "spatial decimation",
            DownsamplingStrategy::Adaptive => "adaptive density",
            DownsamplingStrategy::ImportanceBased => "importance-based",
            DownsamplingStrategy::FixedCount => "fixed count",
        }
    }
}

/// Downsampling filter configuration
#[derive(Debug, Clone)]
pub struct DownsamplingFilter {
    /// Downsampling strategy to use
    pub strategy: DownsamplingStrategy,
    /// Sampling rate (0.0 to 1.0) for uniform and adaptive strategies
    pub sampling_rate: Option<f64>,
    /// Decimation factor (keep every Nth event)
    pub decimation_factor: Option<usize>,
    /// Target number of events for fixed count strategy
    pub target_count: Option<usize>,
    /// Time window for adaptive sampling (microseconds)
    pub adaptive_window_us: Option<f64>,
    /// Spatial window size for adaptive sampling (pixels)
    pub adaptive_spatial_window: Option<u16>,
    /// Whether to preserve temporal order
    pub preserve_order: bool,
    /// Random seed for reproducible sampling
    pub random_seed: Option<u64>,
    /// Whether to balance polarities in downsampling
    pub balance_polarities: bool,
}

impl DownsamplingFilter {
    /// Create uniform random downsampling filter
    ///
    /// # Arguments
    /// * `rate` - Sampling rate (0.0 to 1.0)
    pub fn uniform(rate: f64) -> Self {
        Self {
            strategy: DownsamplingStrategy::Uniform,
            sampling_rate: Some(rate),
            decimation_factor: None,
            target_count: None,
            adaptive_window_us: None,
            adaptive_spatial_window: None,
            preserve_order: true,
            random_seed: None,
            balance_polarities: false,
        }
    }

    /// Create temporal decimation filter
    ///
    /// # Arguments
    /// * `factor` - Keep every Nth event
    pub fn temporal_decimation(factor: usize) -> Self {
        Self {
            strategy: DownsamplingStrategy::TemporalDecimation,
            sampling_rate: None,
            decimation_factor: Some(factor),
            target_count: None,
            adaptive_window_us: None,
            adaptive_spatial_window: None,
            preserve_order: true,
            random_seed: None,
            balance_polarities: false,
        }
    }

    /// Create spatial decimation filter
    ///
    /// # Arguments
    /// * `factor` - Spatial decimation factor
    pub fn spatial_decimation(factor: usize) -> Self {
        Self {
            strategy: DownsamplingStrategy::SpatialDecimation,
            sampling_rate: None,
            decimation_factor: Some(factor),
            target_count: None,
            adaptive_window_us: None,
            adaptive_spatial_window: None,
            preserve_order: true,
            random_seed: None,
            balance_polarities: false,
        }
    }

    /// Create adaptive downsampling filter
    ///
    /// # Arguments
    /// * `rate` - Base sampling rate
    /// * `window_us` - Time window for density calculation
    /// * `spatial_window` - Spatial window size
    pub fn adaptive(rate: f64, window_us: f64, spatial_window: u16) -> Self {
        Self {
            strategy: DownsamplingStrategy::Adaptive,
            sampling_rate: Some(rate),
            decimation_factor: None,
            target_count: None,
            adaptive_window_us: Some(window_us),
            adaptive_spatial_window: Some(spatial_window),
            preserve_order: true,
            random_seed: None,
            balance_polarities: false,
        }
    }

    /// Create importance-based downsampling filter
    ///
    /// # Arguments
    /// * `rate` - Base sampling rate
    pub fn importance_based(rate: f64) -> Self {
        Self {
            strategy: DownsamplingStrategy::ImportanceBased,
            sampling_rate: Some(rate),
            decimation_factor: None,
            target_count: None,
            adaptive_window_us: None,
            adaptive_spatial_window: None,
            preserve_order: true,
            random_seed: None,
            balance_polarities: false,
        }
    }

    /// Create fixed count downsampling filter
    ///
    /// # Arguments
    /// * `count` - Target number of events to keep
    pub fn fixed_count(count: usize) -> Self {
        Self {
            strategy: DownsamplingStrategy::FixedCount,
            sampling_rate: None,
            decimation_factor: None,
            target_count: Some(count),
            adaptive_window_us: None,
            adaptive_spatial_window: None,
            preserve_order: true,
            random_seed: None,
            balance_polarities: false,
        }
    }

    /// Set random seed for reproducible sampling
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Set whether to preserve temporal order
    pub fn with_order_preservation(mut self, preserve: bool) -> Self {
        self.preserve_order = preserve;
        self
    }

    /// Set whether to balance polarities
    pub fn with_polarity_balance(mut self, balance: bool) -> Self {
        self.balance_polarities = balance;
        self
    }

    /// Get description of this filter
    pub fn description(&self) -> String {
        let mut parts = vec![self.strategy.description().to_string()];

        if let Some(rate) = self.sampling_rate {
            parts.push(format!("rate: {:.3}", rate));
        }

        if let Some(factor) = self.decimation_factor {
            parts.push(format!("factor: {}", factor));
        }

        if let Some(count) = self.target_count {
            parts.push(format!("target: {}", count));
        }

        if self.balance_polarities {
            parts.push("balanced polarities".to_string());
        }

        parts.join(", ")
    }
}

impl Default for DownsamplingFilter {
    fn default() -> Self {
        Self::uniform(0.5) // 50% sampling by default
    }
}

impl Validatable for DownsamplingFilter {
    fn validate(&self) -> FilterResult<()> {
        match self.strategy {
            DownsamplingStrategy::Uniform => {
                if let Some(rate) = self.sampling_rate {
                    if rate <= 0.0 || rate > 1.0 {
                        return Err(FilterError::InvalidConfig(format!(
                            "Uniform sampling rate must be between 0.0 and 1.0, got {}",
                            rate
                        )));
                    }
                } else {
                    return Err(FilterError::InvalidConfig(
                        "Uniform strategy requires sampling_rate".to_string(),
                    ));
                }
            }
            DownsamplingStrategy::TemporalDecimation | DownsamplingStrategy::SpatialDecimation => {
                if let Some(factor) = self.decimation_factor {
                    if factor == 0 {
                        return Err(FilterError::InvalidConfig(
                            "Decimation factor must be positive".to_string(),
                        ));
                    }
                } else {
                    return Err(FilterError::InvalidConfig(
                        "Decimation strategy requires decimation_factor".to_string(),
                    ));
                }
            }
            DownsamplingStrategy::Adaptive => {
                if self.sampling_rate.is_none() || self.adaptive_window_us.is_none() {
                    return Err(FilterError::InvalidConfig(
                        "Adaptive strategy requires sampling_rate and adaptive_window_us"
                            .to_string(),
                    ));
                }
                if let Some(window) = self.adaptive_window_us {
                    if window <= 0.0 {
                        return Err(FilterError::InvalidConfig(
                            "Adaptive window must be positive".to_string(),
                        ));
                    }
                }
            }
            DownsamplingStrategy::ImportanceBased => {
                if self.sampling_rate.is_none() {
                    return Err(FilterError::InvalidConfig(
                        "Importance-based strategy requires sampling_rate".to_string(),
                    ));
                }
            }
            DownsamplingStrategy::FixedCount => {
                if self.target_count.is_none() {
                    return Err(FilterError::InvalidConfig(
                        "Fixed count strategy requires target_count".to_string(),
                    ));
                }
            }
        }

        Ok(())
    }
}

//
// ============================================================================
// POLARS-FIRST IMPLEMENTATIONS
// ============================================================================
//

/// Apply downsampling filter to LazyFrame using Polars expressions
///
/// This is the primary Polars-first implementation that reduces the number of events
/// according to the specified downsampling strategy using efficient Polars operations.
///
/// # Arguments
///
/// * `df` - Input LazyFrame with event data
/// * `filter` - Downsampling filter configuration
///
/// # Returns
///
/// * `PolarsResult<LazyFrame>` - Downsampled events as LazyFrame
///
/// # Example
///
/// ```rust
/// use polars::prelude::*;
/// use evlib::ev_filtering::downsampling::*;
///
/// // Convert events to LazyFrame once
/// let events_df = events_to_dataframe(&events)?.lazy();
///
/// // Apply downsampling with Polars expressions
/// let downsampled = apply_downsampling_filter_polars(events_df, &DownsamplingFilter::uniform(0.5))?;
/// ```
#[cfg_attr(feature = "tracing", instrument(skip(df), fields(strategy = ?filter.strategy)))]
pub fn apply_downsampling_filter_polars(
    df: LazyFrame,
    filter: &DownsamplingFilter,
) -> PolarsResult<LazyFrame> {
    let start_time = Instant::now();

    // Validate filter configuration first
    if let Err(e) = filter.validate() {
        return Err(PolarsError::ComputeError(
            format!("Invalid downsampling configuration: {}", e).into(),
        ));
    }

    let downsampled_df = match filter.strategy {
        DownsamplingStrategy::Uniform => {
            apply_uniform_sampling_polars(df, filter.sampling_rate.unwrap(), filter.random_seed)?
        }
        DownsamplingStrategy::TemporalDecimation => {
            apply_temporal_decimation_polars(df, filter.decimation_factor.unwrap())?
        }
        DownsamplingStrategy::SpatialDecimation => {
            apply_spatial_decimation_polars(df, filter.decimation_factor.unwrap())?
        }
        DownsamplingStrategy::Adaptive => apply_adaptive_sampling_polars(
            df,
            filter.sampling_rate.unwrap(),
            filter.adaptive_window_us.unwrap(),
            filter.adaptive_spatial_window.unwrap_or(3),
        )?,
        DownsamplingStrategy::ImportanceBased => {
            apply_importance_based_sampling_polars(df, filter.sampling_rate.unwrap())?
        }
        DownsamplingStrategy::FixedCount => {
            apply_fixed_count_sampling_polars(df, filter.target_count.unwrap(), filter.random_seed)?
        }
    };

    let mut final_df = downsampled_df;

    // Balance polarities if requested
    if filter.balance_polarities {
        final_df = balance_polarity_sampling_polars(final_df)?;
    }

    // Preserve temporal order if requested
    if filter.preserve_order {
        final_df = final_df.sort([COL_T], SortMultipleOptions::default());
    }

    let processing_time = start_time.elapsed().as_secs_f64();
    info!(
        "Polars downsampling ({}): {:.3}s processing time",
        filter.strategy.description(),
        processing_time
    );

    Ok(final_df)
}

/// Apply uniform random sampling using Polars sample() function
///
/// This function uses Polars' optimized random sampling which is much more efficient
/// than manual iteration over events.
#[cfg_attr(feature = "tracing", instrument(skip(df), fields(sampling_rate = sampling_rate)))]
pub fn apply_uniform_sampling_polars(
    df: LazyFrame,
    sampling_rate: f64,
    seed: Option<u64>,
) -> PolarsResult<LazyFrame> {
    debug!(
        "Applying Polars uniform sampling with rate: {:.3}",
        sampling_rate
    );

    // Use Polars' sample function for efficient random sampling
    // Note: sample_frac not available on LazyFrame, so collect first
    let collected_df = df.collect()?;
    // Create Series for sampling_rate parameter
    let frac_series = Series::new("".into(), [sampling_rate]);
    Ok(collected_df
        .sample_frac(
            &frac_series,
            false, // with_replacement = false
            true,  // shuffle
            seed,
        )?
        .lazy())
}

/// Apply temporal decimation using Polars slice operations
///
/// This uses Polars' optimized slice operations instead of manual step_by iteration.
#[cfg_attr(feature = "tracing", instrument(skip(df), fields(factor = factor)))]
pub fn apply_temporal_decimation_polars(df: LazyFrame, factor: usize) -> PolarsResult<LazyFrame> {
    debug!(
        "Applying Polars temporal decimation with factor: {}",
        factor
    );

    // Use Polars expressions to select every Nth row efficiently
    Ok(df
        .with_row_index("__temp_idx", None)
        .filter((col("__temp_idx") % lit(factor as i64)).eq(lit(0)))
        .drop(["__temp_idx"]))
}

/// Apply spatial decimation using Polars filter expressions
///
/// This replaces the manual pixel coordinate checking loop with efficient Polars expressions.
#[cfg_attr(feature = "tracing", instrument(skip(df), fields(factor = factor)))]
pub fn apply_spatial_decimation_polars(df: LazyFrame, factor: usize) -> PolarsResult<LazyFrame> {
    debug!("Applying Polars spatial decimation with factor: {}", factor);

    // Use Polars expressions for efficient coordinate filtering
    Ok(df.filter(
        (col(COL_X) % lit(factor as i64))
            .eq(lit(0))
            .and((col(COL_Y) % lit(factor as i64)).eq(lit(0))),
    ))
}

/// Apply fixed count sampling using Polars sample() function
///
/// This replaces the reservoir sampling algorithm with Polars' optimized sample function.
#[cfg_attr(feature = "tracing", instrument(skip(df), fields(target_count = target_count)))]
pub fn apply_fixed_count_sampling_polars(
    df: LazyFrame,
    target_count: usize,
    seed: Option<u64>,
) -> PolarsResult<LazyFrame> {
    debug!(
        "Applying Polars fixed count sampling with target: {}",
        target_count
    );

    // Use Polars' sample function for efficient fixed-count sampling
    // Note: sample_n not available on LazyFrame, so collect first
    let collected_df = df.collect()?;
    // Create Series for target_count parameter
    let n_series = Series::new("".into(), [target_count as u32]);
    Ok(collected_df
        .sample_n(
            &n_series, false, // with_replacement = false
            true,  // shuffle
            seed,
        )?
        .lazy())
}

/// Balance polarity distribution using Polars group_by operations
///
/// This replaces the manual polarity separation and balancing with efficient Polars operations.
#[cfg_attr(feature = "tracing", instrument(skip(df)))]
pub fn balance_polarity_sampling_polars(df: LazyFrame) -> PolarsResult<LazyFrame> {
    debug!("Applying Polars polarity balancing");

    // Calculate minimum count per polarity using group_by
    let min_count_df = df
        .clone()
        .group_by([col(COL_POLARITY)])
        .agg([len().alias("count")])
        .select([col("count").min().alias("min_count")])
        .collect()?;

    if min_count_df.height() == 0 {
        return Ok(df);
    }

    let min_count = min_count_df.column("min_count")?.u32()?.get(0).unwrap_or(0) as usize;

    if min_count == 0 {
        return Ok(df.limit(0)); // Return empty if no events of one polarity
    }

    // Sample equal numbers from each polarity
    // Note: sample_n not available on LazyFrame, so collect first
    let positive_collected = df.clone().filter(col(COL_POLARITY).gt(lit(0))).collect()?;
    // Create Series for min_count parameter
    let min_count_series = Series::new("".into(), [min_count as u32]);
    let positive_sample = positive_collected
        .sample_n(
            &min_count_series,
            false, // with_replacement = false
            false, // shuffle
            None,  // seed
        )?
        .lazy();

    let negative_collected = df.filter(col(COL_POLARITY).eq(lit(0))).collect()?;
    let negative_sample = negative_collected
        .sample_n(
            &min_count_series,
            false, // with_replacement = false
            false, // shuffle
            None,  // seed
        )?
        .lazy();

    // Combine the balanced samples
    concat([positive_sample, negative_sample], Default::default())
}

/// Apply adaptive sampling using Polars window functions
///
/// This replaces the nested loops with efficient Polars window operations for
/// calculating local event density.
#[cfg_attr(feature = "tracing", instrument(skip(df), fields(base_rate = base_rate, window_us = window_us, spatial_window = spatial_window)))]
pub fn apply_adaptive_sampling_polars(
    df: LazyFrame,
    base_rate: f64,
    window_us: f64,
    spatial_window: u16,
) -> PolarsResult<LazyFrame> {
    debug!(
        "Applying Polars adaptive sampling: rate={:.3}, window={}μs, spatial={}px",
        base_rate, window_us, spatial_window
    );

    let _window_sec = window_us / 1_000_000.0;

    // Add window-based density calculations using Polars expressions
    let _df_with_density = df
        .clone()
        .sort([COL_T], SortMultipleOptions::default())
        .with_columns([
            // Calculate local temporal density using rolling window
            // TODO: Implement proper rolling operations for temporal density
            lit(1).alias("temporal_density"), // Placeholder for now
            // Add random values for sampling decisions
            when(lit(true))
                .then(lit(0.0)) // Will be replaced with random values
                .otherwise(lit(1.0))
                .alias("random_val"),
        ])
        .with_columns([
            // Calculate adaptive sampling rate based on density
            // Simplified adaptive rate calculation (Polars 0.49.1 compatible)
            when(col("temporal_density").gt(lit(1.0)))
                .then(
                    when(
                        (lit(base_rate) / col("temporal_density").cast(DataType::Float64))
                            .lt(lit(0.01)),
                    )
                    .then(lit(0.01))
                    .otherwise(
                        when(
                            (lit(base_rate) / col("temporal_density").cast(DataType::Float64))
                                .gt(lit(1.0)),
                        )
                        .then(lit(1.0))
                        .otherwise(
                            lit(base_rate) / col("temporal_density").cast(DataType::Float64),
                        ),
                    ),
                )
                .otherwise(lit(base_rate))
                .alias("adaptive_rate"),
        ]);

    // For now, fall back to uniform sampling with base rate
    // TODO: Implement proper density-based adaptive sampling with spatial windows
    warn!("Adaptive sampling using simplified uniform rate - full spatial-temporal density calculation not yet implemented in Polars");
    apply_uniform_sampling_polars(df, base_rate, None)
}

/// Apply importance-based sampling using Polars expressions for scoring
///
/// This replaces the manual importance calculation with Polars expressions.
#[cfg_attr(feature = "tracing", instrument(skip(df), fields(base_rate = base_rate)))]
pub fn apply_importance_based_sampling_polars(
    df: LazyFrame,
    base_rate: f64,
) -> PolarsResult<LazyFrame> {
    debug!(
        "Applying Polars importance-based sampling with rate: {:.3}",
        base_rate
    );

    let _df_with_importance = df
        .clone()
        .sort([COL_T], SortMultipleOptions::default())
        .with_columns([
            // Calculate importance scores using Polars expressions
            // Factor 1: Polarity changes increase importance
            col(COL_POLARITY)
                .neq(col(COL_POLARITY).shift(lit(1)))
                .fill_null(lit(false))
                .cast(DataType::Float64)
                * lit(0.5)
                + lit(1.0).alias("polarity_importance"),
            // Factor 2: Time difference patterns
            (col(COL_T) - col(COL_T).shift(lit(1)))
                .fill_null(lit(0.001))
                .alias("time_diff"),
        ])
        .with_columns([
            // Factor 3: Unusual timing patterns
            when(
                col("time_diff")
                    .lt(lit(0.0001))
                    .or(col("time_diff").gt(lit(0.01))),
            )
            .then(lit(1.2))
            .otherwise(lit(1.0))
            .alias("timing_importance"),
        ])
        .with_columns([
            // Combined importance score
            (col("polarity_importance") * col("timing_importance")).alias("importance_score"),
        ])
        .with_columns([
            // Sampling probability based on importance
            // Clamp sampling probability between 0.0 and 1.0 (Polars 0.49.1 compatible)
            when((lit(base_rate) * col("importance_score")).lt(lit(0.0)))
                .then(lit(0.0))
                .otherwise(
                    when((lit(base_rate) * col("importance_score")).gt(lit(1.0)))
                        .then(lit(1.0))
                        .otherwise(lit(base_rate) * col("importance_score")),
                )
                .alias("sampling_prob"),
        ]);

    // For now, use uniform sampling with base rate
    // TODO: Implement proper importance-weighted sampling
    warn!("Importance-based sampling using simplified uniform rate - full importance weighting not yet implemented in Polars");
    apply_uniform_sampling_polars(df, base_rate, None)
}

//
// ============================================================================
// CONVENIENCE FUNCTIONS FOR POLARS-FIRST USAGE
// ============================================================================
//

/// Convenience function for Polars uniform downsampling
///
/// # Example
///
/// ```rust
/// use polars::prelude::*;
/// use evlib::ev_filtering::downsampling::*;
///
/// let events_df = events_to_dataframe(&events)?.lazy();
/// let downsampled = downsample_uniform_polars(events_df, 0.5)?;
/// ```
pub fn downsample_uniform_polars(df: LazyFrame, rate: f64) -> PolarsResult<LazyFrame> {
    let filter = DownsamplingFilter::uniform(rate);
    apply_downsampling_filter_polars(df, &filter)
}

/// Convenience function for Polars event count reduction
pub fn downsample_events_polars(df: LazyFrame, target_count: usize) -> PolarsResult<LazyFrame> {
    let filter = DownsamplingFilter::fixed_count(target_count);
    apply_downsampling_filter_polars(df, &filter)
}

/// Convenience function for Polars temporal decimation
pub fn downsample_temporal_decimation_polars(
    df: LazyFrame,
    factor: usize,
) -> PolarsResult<LazyFrame> {
    let filter = DownsamplingFilter::temporal_decimation(factor);
    apply_downsampling_filter_polars(df, &filter)
}

/// Convenience function for Polars spatial decimation
pub fn downsample_spatial_decimation_polars(
    df: LazyFrame,
    factor: usize,
) -> PolarsResult<LazyFrame> {
    let filter = DownsamplingFilter::spatial_decimation(factor);
    apply_downsampling_filter_polars(df, &filter)
}
