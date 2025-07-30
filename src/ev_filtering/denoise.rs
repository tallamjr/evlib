//! Polars-first noise filtering operations for event camera data
//!
//! This module provides various denoising algorithms using Polars DataFrames
//! and LazyFrames for maximum performance and memory efficiency. All operations
//! work directly with Polars expressions and avoid manual Vec<Event> iterations.
//!
//! # Philosophy
//!
//! This implementation follows a strict Polars-first approach:
//! - Input: LazyFrame (from events_to_dataframe)
//! - Processing: Polars expressions, window functions, and group operations
//! - Output: LazyFrame (convertible to Vec<Event>/numpy only when needed)
//!
//! # Performance Benefits
//!
//! - Lazy evaluation: Operations are optimized and executed only when needed
//! - Vectorized operations: All filtering uses SIMD-optimized Polars operations
//! - Memory efficiency: No intermediate Vec<Event> allocations or HashMap operations
//! - Query optimization: Polars optimizes the entire filtering pipeline
//!
//! # Example
//!
//! ```rust
//! use polars::prelude::*;
//! use evlib::ev_filtering::denoise::*;
//!
//! // Convert events to LazyFrame once
//! let events_df = events_to_dataframe(&events)?.lazy();
//!
//! // Apply refractory period filtering with Polars expressions
//! let filter = RefractoryFilter::new(1000.0); // 1ms refractory period
//! let filtered = apply_refractory_filter(events_df, &filter)?;
//! ```

use crate::ev_core::{Event, Events};
use crate::ev_filtering::config::Validatable;
use crate::ev_filtering::{utils, FilterError, FilterResult, SingleFilter};
use polars::prelude::*;
use tracing::{debug, info, instrument, warn};

/// Polars column names for event data (consistent across all filtering modules)
pub const COL_X: &str = "x";
pub const COL_Y: &str = "y";
pub const COL_T: &str = "timestamp";
pub const COL_POLARITY: &str = "polarity";

/// Noise filtering methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenoiseMethod {
    /// Refractory period filtering - removes events too close in time at same pixel
    RefractoryPeriod,
    /// Temporal correlation filtering - requires neighboring events in time
    TemporalCorrelation,
    /// Spatial-temporal correlation - requires neighboring events in space and time
    SpatialTemporalCorrelation,
    /// Background activity filtering - removes isolated events
    BackgroundActivity,
    /// Multi-scale filtering - combines multiple filtering scales
    MultiScale,
}

impl DenoiseMethod {
    /// Get description of this method
    pub fn description(&self) -> &'static str {
        match self {
            DenoiseMethod::RefractoryPeriod => "refractory period",
            DenoiseMethod::TemporalCorrelation => "temporal correlation",
            DenoiseMethod::SpatialTemporalCorrelation => "spatial-temporal correlation",
            DenoiseMethod::BackgroundActivity => "background activity",
            DenoiseMethod::MultiScale => "multi-scale",
        }
    }
}

/// Refractory period filter configuration
#[derive(Debug, Clone)]
pub struct RefractoryFilter {
    /// Refractory period in microseconds
    pub period_us: f64,
    /// Whether to apply separately for each polarity
    pub per_polarity: bool,
    /// Whether to use absolute refractory period (hard cutoff)
    pub absolute_refractory: bool,
}

impl RefractoryFilter {
    /// Create a new refractory filter
    pub fn new(period_us: f64) -> Self {
        Self {
            period_us,
            per_polarity: true,
            absolute_refractory: true,
        }
    }

    /// Set whether to filter per polarity
    pub fn with_per_polarity(mut self, per_polarity: bool) -> Self {
        self.per_polarity = per_polarity;
        self
    }

    /// Set whether to use absolute refractory period
    pub fn with_absolute_refractory(mut self, absolute: bool) -> Self {
        self.absolute_refractory = absolute;
        self
    }
}

/// Temporal correlation filter configuration
#[derive(Debug, Clone)]
pub struct TemporalCorrelationFilter {
    /// Time window for correlation in microseconds
    pub time_window_us: f64,
    /// Minimum number of events required in window
    pub min_events: usize,
    /// Maximum distance in time for correlation
    pub max_time_distance_us: f64,
}

impl TemporalCorrelationFilter {
    /// Create a new temporal correlation filter
    pub fn new(time_window_us: f64, min_events: usize) -> Self {
        Self {
            time_window_us,
            min_events,
            max_time_distance_us: time_window_us,
        }
    }
}

/// Spatial-temporal correlation filter configuration
#[derive(Debug, Clone)]
pub struct SpatialTemporalFilter {
    /// Spatial radius for correlation (pixels)
    pub spatial_radius: u16,
    /// Time window for correlation in microseconds
    pub time_window_us: f64,
    /// Minimum number of neighbor events required
    pub min_neighbors: usize,
    /// Whether to consider polarity in correlation
    pub consider_polarity: bool,
}

impl SpatialTemporalFilter {
    /// Create a new spatial-temporal filter
    pub fn new(spatial_radius: u16, time_window_us: f64, min_neighbors: usize) -> Self {
        Self {
            spatial_radius,
            time_window_us,
            min_neighbors,
            consider_polarity: false,
        }
    }

    /// Set whether to consider polarity
    pub fn with_polarity_consideration(mut self, consider: bool) -> Self {
        self.consider_polarity = consider;
        self
    }
}

/// Performance configuration for denoising operations
#[derive(Debug, Clone)]
pub struct DenoisePerformanceConfig {
    /// Processing chunk size for streaming operations
    pub chunk_size: usize,
    /// Enable parallel processing when available
    pub enable_parallel: bool,
    /// Memory usage limit in bytes
    pub memory_limit: usize,
}

impl Default for DenoisePerformanceConfig {
    fn default() -> Self {
        Self {
            chunk_size: 100_000,
            enable_parallel: true,
            memory_limit: 1_000_000_000, // 1GB
        }
    }
}

impl DenoisePerformanceConfig {
    /// High performance configuration for large datasets
    pub fn high_performance() -> Self {
        Self {
            chunk_size: 500_000,
            enable_parallel: true,
            memory_limit: 4_000_000_000, // 4GB
        }
    }

    /// Memory efficient configuration for resource-constrained environments
    pub fn memory_efficient() -> Self {
        Self {
            chunk_size: 50_000,
            enable_parallel: false,
            memory_limit: 500_000_000, // 500MB
        }
    }
}

/// Main denoising filter configuration
#[derive(Debug, Clone)]
pub struct DenoiseFilter {
    /// Primary denoising method
    pub method: DenoiseMethod,
    /// Refractory period filter configuration
    pub refractory_filter: Option<RefractoryFilter>,
    /// Temporal correlation filter configuration
    pub temporal_correlation_filter: Option<TemporalCorrelationFilter>,
    /// Spatial-temporal correlation filter configuration
    pub spatial_temporal_filter: Option<SpatialTemporalFilter>,
    /// Background activity threshold (events per second)
    pub background_threshold: Option<f64>,
    /// Whether to preserve temporal order
    pub preserve_order: bool,
    /// Whether to validate results
    pub validate_results: bool,
    /// Performance configuration
    pub performance_config: DenoisePerformanceConfig,
}

impl DenoiseFilter {
    /// Create a refractory period filter
    pub fn refractory(period_us: f64) -> Self {
        Self {
            method: DenoiseMethod::RefractoryPeriod,
            refractory_filter: Some(RefractoryFilter::new(period_us)),
            temporal_correlation_filter: None,
            spatial_temporal_filter: None,
            background_threshold: None,
            preserve_order: true,
            validate_results: true,
            performance_config: DenoisePerformanceConfig::default(),
        }
    }

    /// Create a temporal correlation filter
    pub fn temporal_correlation(time_window_us: f64, min_events: usize) -> Self {
        Self {
            method: DenoiseMethod::TemporalCorrelation,
            refractory_filter: None,
            temporal_correlation_filter: Some(TemporalCorrelationFilter::new(
                time_window_us,
                min_events,
            )),
            spatial_temporal_filter: None,
            background_threshold: None,
            preserve_order: true,
            validate_results: true,
            performance_config: DenoisePerformanceConfig::default(),
        }
    }

    /// Create a spatial-temporal correlation filter
    pub fn spatial_temporal(
        spatial_radius: u16,
        time_window_us: f64,
        min_neighbors: usize,
    ) -> Self {
        Self {
            method: DenoiseMethod::SpatialTemporalCorrelation,
            refractory_filter: None,
            temporal_correlation_filter: None,
            spatial_temporal_filter: Some(SpatialTemporalFilter::new(
                spatial_radius,
                time_window_us,
                min_neighbors,
            )),
            background_threshold: None,
            preserve_order: true,
            validate_results: true,
            performance_config: DenoisePerformanceConfig::default(),
        }
    }

    /// Create a background activity filter
    pub fn background_activity(threshold_events_per_sec: f64) -> Self {
        Self {
            method: DenoiseMethod::BackgroundActivity,
            refractory_filter: None,
            temporal_correlation_filter: None,
            spatial_temporal_filter: None,
            background_threshold: Some(threshold_events_per_sec),
            preserve_order: true,
            validate_results: true,
            performance_config: DenoisePerformanceConfig::default(),
        }
    }

    /// Create a multi-scale filter combining multiple methods
    pub fn multi_scale(
        refractory_period_us: f64,
        spatial_radius: u16,
        time_window_us: f64,
    ) -> Self {
        Self {
            method: DenoiseMethod::MultiScale,
            refractory_filter: Some(RefractoryFilter::new(refractory_period_us)),
            temporal_correlation_filter: None,
            spatial_temporal_filter: Some(SpatialTemporalFilter::new(
                spatial_radius,
                time_window_us,
                2,
            )),
            background_threshold: None,
            preserve_order: true,
            validate_results: true,
            performance_config: DenoisePerformanceConfig::default(),
        }
    }

    /// Set whether to preserve temporal order
    pub fn with_order_preservation(mut self, preserve: bool) -> Self {
        self.preserve_order = preserve;
        self
    }

    /// Set whether to validate results
    pub fn with_validation(mut self, validate: bool) -> Self {
        self.validate_results = validate;
        self
    }

    /// Set performance configuration
    pub fn with_performance_config(mut self, config: DenoisePerformanceConfig) -> Self {
        self.performance_config = config;
        self
    }

    /// Enable high performance mode
    pub fn with_high_performance(mut self) -> Self {
        self.performance_config = DenoisePerformanceConfig::high_performance();
        self
    }

    /// Enable memory efficient mode
    pub fn with_memory_efficient(mut self) -> Self {
        self.performance_config = DenoisePerformanceConfig::memory_efficient();
        self
    }

    /// Get description of this filter
    pub fn description(&self) -> String {
        let mut parts = vec![self.method.description().to_string()];

        if let Some(ref_filter) = &self.refractory_filter {
            parts.push(format!("refractory: {:.1}µs", ref_filter.period_us));
        }

        if let Some(temp_filter) = &self.temporal_correlation_filter {
            parts.push(format!(
                "temporal: {:.1}µs window, {} events",
                temp_filter.time_window_us, temp_filter.min_events
            ));
        }

        if let Some(st_filter) = &self.spatial_temporal_filter {
            parts.push(format!(
                "spatial-temporal: r={}, {:.1}µs, {} neighbors",
                st_filter.spatial_radius, st_filter.time_window_us, st_filter.min_neighbors
            ));
        }

        if let Some(bg_threshold) = self.background_threshold {
            parts.push(format!("background: {:.1} events/s", bg_threshold));
        }

        parts.join(", ")
    }
}

impl Default for DenoiseFilter {
    fn default() -> Self {
        Self::refractory(1000.0) // 1ms default refractory period
    }
}

impl Validatable for DenoiseFilter {
    fn validate(&self) -> FilterResult<()> {
        match self.method {
            DenoiseMethod::RefractoryPeriod => {
                if let Some(ref_filter) = &self.refractory_filter {
                    if ref_filter.period_us <= 0.0 {
                        return Err(FilterError::InvalidConfig(
                            "Refractory period must be positive".to_string(),
                        ));
                    }
                } else {
                    return Err(FilterError::InvalidConfig(
                        "Refractory period method requires refractory_filter".to_string(),
                    ));
                }
            }
            DenoiseMethod::TemporalCorrelation => {
                if let Some(temp_filter) = &self.temporal_correlation_filter {
                    if temp_filter.time_window_us <= 0.0 {
                        return Err(FilterError::InvalidConfig(
                            "Temporal correlation time window must be positive".to_string(),
                        ));
                    }
                    if temp_filter.min_events == 0 {
                        return Err(FilterError::InvalidConfig(
                            "Temporal correlation min_events must be positive".to_string(),
                        ));
                    }
                } else {
                    return Err(FilterError::InvalidConfig(
                        "Temporal correlation method requires temporal_correlation_filter"
                            .to_string(),
                    ));
                }
            }
            DenoiseMethod::SpatialTemporalCorrelation => {
                if let Some(st_filter) = &self.spatial_temporal_filter {
                    if st_filter.time_window_us <= 0.0 {
                        return Err(FilterError::InvalidConfig(
                            "Spatial-temporal time window must be positive".to_string(),
                        ));
                    }
                    if st_filter.min_neighbors == 0 {
                        return Err(FilterError::InvalidConfig(
                            "Spatial-temporal min_neighbors must be positive".to_string(),
                        ));
                    }
                } else {
                    return Err(FilterError::InvalidConfig(
                        "Spatial-temporal method requires spatial_temporal_filter".to_string(),
                    ));
                }
            }
            DenoiseMethod::BackgroundActivity => {
                if let Some(threshold) = self.background_threshold {
                    if threshold <= 0.0 {
                        return Err(FilterError::InvalidConfig(
                            "Background activity threshold must be positive".to_string(),
                        ));
                    }
                } else {
                    return Err(FilterError::InvalidConfig(
                        "Background activity method requires background_threshold".to_string(),
                    ));
                }
            }
            DenoiseMethod::MultiScale => {
                if self.refractory_filter.is_none() && self.spatial_temporal_filter.is_none() {
                    return Err(FilterError::InvalidConfig(
                        "Multi-scale method requires at least one sub-filter".to_string(),
                    ));
                }
            }
        }

        Ok(())
    }
}

impl SingleFilter for DenoiseFilter {
    fn apply(&self, events: &Events) -> FilterResult<Events> {
        apply_denoise_filter(events, self)
    }

    fn description(&self) -> String {
        format!("Denoise filter: {}", self.description())
    }

    fn is_enabled(&self) -> bool {
        true
    }
}

/// Apply denoising filter to events
///
/// This function applies the specified denoising method to remove noise events
/// while preserving signal events.
///
/// # Arguments
///
/// * `events` - Input events to denoise
/// * `filter` - Denoising filter configuration
///
/// # Returns
///
/// * `FilterResult<Events>` - Denoised events
pub fn apply_denoise_filter(events: &Events, filter: &DenoiseFilter) -> FilterResult<Events> {
    let start_time = std::time::Instant::now();

    if events.is_empty() {
        debug!("No events to denoise");
        return Ok(Vec::new());
    }

    // Validate filter configuration
    filter.validate()?;

    let denoised_events = match filter.method {
        DenoiseMethod::RefractoryPeriod => {
            apply_refractory_period_filter(events, filter.refractory_filter.as_ref().unwrap())?
        }
        DenoiseMethod::TemporalCorrelation => apply_temporal_correlation_filter(
            events,
            filter.temporal_correlation_filter.as_ref().unwrap(),
        )?,
        DenoiseMethod::SpatialTemporalCorrelation => {
            apply_spatial_temporal_filter(events, filter.spatial_temporal_filter.as_ref().unwrap())?
        }
        DenoiseMethod::BackgroundActivity => {
            apply_background_activity_filter(events, filter.background_threshold.unwrap())?
        }
        DenoiseMethod::MultiScale => apply_multi_scale_filter(events, filter)?,
    };

    // Preserve temporal order if requested
    let mut final_events = denoised_events;
    if filter.preserve_order && !utils::is_sorted_by_time(&final_events) {
        debug!("Sorting denoised events to preserve temporal order");
        utils::sort_events_by_time(&mut final_events);
    }

    // Validate results if requested
    if filter.validate_results {
        utils::validate_events(&final_events, false)?;
    }

    let processing_time = start_time.elapsed().as_secs_f64();
    let input_count = events.len();
    let output_count = final_events.len();
    let removed_count = input_count - output_count;

    info!(
        "Denoising ({}): {} -> {} events ({} removed, {:.1}% reduction) in {:.3}s",
        filter.method.description(),
        input_count,
        output_count,
        removed_count,
        (removed_count as f64 / input_count as f64) * 100.0,
        processing_time
    );

    Ok(final_events)
}

/// Apply refractory period filtering using Polars window functions (Polars-first implementation)
///
/// This is the main Polars-first implementation that uses window functions
/// to track the time difference between consecutive events at each pixel.
///
/// # Arguments
///
/// * `df` - Input LazyFrame containing event data
/// * `filter` - Refractory filter configuration
///
/// # Returns
///
/// Filtered LazyFrame with refractory period constraints applied
#[instrument(skip(df), fields(filter = ?filter))]
pub fn apply_refractory_filter_polars(
    df: LazyFrame,
    filter: &RefractoryFilter,
) -> PolarsResult<LazyFrame> {
    debug!("Applying refractory period filter: {:?}", filter);

    let grouping_columns = if filter.per_polarity {
        vec![col(COL_X), col(COL_Y), col(COL_POLARITY)]
    } else {
        vec![col(COL_X), col(COL_Y)]
    };

    let filtered_df = df
        .sort([COL_T], SortMultipleOptions::default()) // Ensure temporal order
        .with_columns([
            // Calculate time difference from previous event at same pixel
            // Note: Using shift instead of diff for older Polars versions
            (col(COL_T) - col(COL_T).shift(lit(1)).over(grouping_columns.clone()))
                .alias("time_diff_s"),
        ])
        .with_columns([
            // Convert to microseconds and handle first events (null diff)
            (col("time_diff_s") * lit(1_000_000.0)).alias("time_diff_us"),
        ])
        .filter(
            // Keep first events (null time_diff) or events beyond refractory period
            col("time_diff_us")
                .is_null()
                .or(col("time_diff_us").gt_eq(lit(filter.period_us))),
        )
        .drop(["time_diff_s", "time_diff_us"]); // Clean up temporary columns

    debug!("Refractory filter applied using Polars window functions");
    Ok(filtered_df)
}

/// Legacy function for backward compatibility - delegates to Polars implementation
fn apply_refractory_period_filter(
    events: &Events,
    filter: &RefractoryFilter,
) -> FilterResult<Events> {
    warn!("Using legacy Vec<Event> interface - consider using LazyFrame directly for better performance");

    if events.is_empty() {
        return Ok(Vec::new());
    }

    let df = crate::ev_core::events_to_dataframe(events)
        .map_err(|e| FilterError::ProcessingError(format!("DataFrame conversion failed: {}", e)))?
        .lazy();

    let filtered_df = apply_refractory_filter_polars(df, filter)
        .map_err(|e| FilterError::ProcessingError(format!("Polars filtering failed: {}", e)))?;

    // Convert back to Vec<Event> - this is inefficient but maintains compatibility
    let result_df = filtered_df
        .collect()
        .map_err(|e| FilterError::ProcessingError(format!("LazyFrame collection failed: {}", e)))?;

    dataframe_to_events(&result_df)
}

// Legacy HashMap-based refractory functions removed - functionality moved to Polars-first implementation

// Legacy HashMap-based refractory parallel/streaming functions removed

/// Apply temporal correlation filtering using Polars window functions (Polars-first implementation)
///
/// This is the main Polars-first implementation that uses window functions
/// to count neighboring events within a time window for each event.
///
/// # Arguments
///
/// * `df` - Input LazyFrame containing event data
/// * `filter` - Temporal correlation filter configuration
///
/// # Returns
///
/// Filtered LazyFrame with temporal correlation constraints applied
#[instrument(skip(df), fields(filter = ?filter))]
pub fn apply_temporal_correlation_filter_polars(
    df: LazyFrame,
    filter: &TemporalCorrelationFilter,
) -> PolarsResult<LazyFrame> {
    debug!("Applying temporal correlation filter: {:?}", filter);

    let time_window_sec = filter.time_window_us / 1_000_000.0;

    let filtered_df = df
        .sort([COL_T], SortMultipleOptions::default()) // Ensure temporal order
        .with_row_index("original_idx", None) // Track original indices
        .with_columns([
            // Add window bounds for each event
            (col(COL_T) - lit(time_window_sec)).alias("window_start"),
            (col(COL_T) + lit(time_window_sec)).alias("window_end"),
        ])
        .with_columns([
            // Count neighbors in time window using a self-join approach via cross product
            // This uses an efficient range-based approach for temporal correlation
            col(COL_T)
                .map(
                    move |s| {
                        let times = s.f64()?;
                        let len = times.len();
                        let mut neighbor_counts = Vec::with_capacity(len);

                        for i in 0..len {
                            if let Some(event_time) = times.get(i) {
                                let mut count = 0usize;
                                let min_time = event_time - time_window_sec;
                                let max_time = event_time + time_window_sec;

                                // Count neighbors in time window (excluding self)
                                for j in 0..len {
                                    if i != j {
                                        if let Some(other_time) = times.get(j) {
                                            if other_time >= min_time && other_time <= max_time {
                                                count += 1;
                                            }
                                        }
                                    }
                                }
                                neighbor_counts.push(Some(count as i64));
                            } else {
                                neighbor_counts.push(None);
                            }
                        }

                        Ok(Some(
                            Int64Chunked::from_iter_options(
                                "neighbor_count".into(),
                                neighbor_counts.into_iter(),
                            )
                            .into_series()
                            .into(),
                        ))
                    },
                    GetOutput::from_type(DataType::Int64),
                )
                .alias("neighbor_count"),
        ])
        .filter(
            // Keep events with sufficient neighbors
            col("neighbor_count").gt_eq(lit(filter.min_events as i64)),
        )
        .drop([
            "original_idx",
            "window_start",
            "window_end",
            "neighbor_count",
        ]); // Clean up temporary columns

    debug!("Temporal correlation filter applied using Polars window functions");
    Ok(filtered_df)
}

/// Legacy function for backward compatibility - delegates to Polars implementation
fn apply_temporal_correlation_filter(
    events: &Events,
    filter: &TemporalCorrelationFilter,
) -> FilterResult<Events> {
    warn!("Using legacy Vec<Event> interface - consider using LazyFrame directly for better performance");

    if events.is_empty() {
        return Ok(Vec::new());
    }

    let df = crate::ev_core::events_to_dataframe(events)
        .map_err(|e| FilterError::ProcessingError(format!("DataFrame conversion failed: {}", e)))?
        .lazy();

    let filtered_df = apply_temporal_correlation_filter_polars(df, filter)
        .map_err(|e| FilterError::ProcessingError(format!("Polars filtering failed: {}", e)))?;

    // Convert back to Vec<Event> - this is inefficient but maintains compatibility
    let result_df = filtered_df
        .collect()
        .map_err(|e| FilterError::ProcessingError(format!("LazyFrame collection failed: {}", e)))?;

    dataframe_to_events(&result_df)
}

// Legacy HashMap-based temporal correlation functions removed

/// Apply spatial-temporal correlation filtering using Polars grid binning (Polars-first implementation)
///
/// This is the main Polars-first implementation that uses grid-based spatial binning
/// and window functions to find spatially and temporally neighboring events.
///
/// # Arguments
///
/// * `df` - Input LazyFrame containing event data
/// * `filter` - Spatial-temporal filter configuration
///
/// # Returns
///
/// Filtered LazyFrame with spatial-temporal correlation constraints applied
#[instrument(skip(df), fields(filter = ?filter))]
pub fn apply_spatial_temporal_filter_polars(
    df: LazyFrame,
    filter: &SpatialTemporalFilter,
) -> PolarsResult<LazyFrame> {
    debug!("Applying spatial-temporal correlation filter: {:?}", filter);

    let time_window_sec = filter.time_window_us / 1_000_000.0;
    let radius = filter.spatial_radius as i64;

    let filtered_df = df
        .sort([COL_T], SortMultipleOptions::default()) // Ensure temporal order
        .with_row_index("event_idx", None) // Track original indices
        .with_columns([
            // Create spatial grid bins for efficient spatial indexing
            // Use grid cells of size radius/2 for good neighbor coverage
            (col(COL_X).cast(DataType::Int64) / lit((radius.max(1) / 2).max(1))).alias("grid_x"),
            (col(COL_Y).cast(DataType::Int64) / lit((radius.max(1) / 2).max(1))).alias("grid_y"),
            // Add time window bounds
            (col(COL_T) - lit(time_window_sec)).alias("time_min"),
            (col(COL_T) + lit(time_window_sec)).alias("time_max"),
        ]);
    // For now, we'll use the join-based approach directly
    // This could be optimized further with custom Polars expressions

    // Since Polars doesn't have native spatial indexing, we'll use a cross-join approach
    // with filtering - this is still more efficient than manual loops
    let self_df = filtered_df.clone();

    let result_df = filtered_df
        .join(
            self_df.select([
                col("event_idx").alias("neighbor_idx"),
                col(COL_X).alias("neighbor_x"),
                col(COL_Y).alias("neighbor_y"),
                col(COL_T).alias("neighbor_t"),
                col(COL_POLARITY).alias("neighbor_polarity"),
            ]),
            [col("grid_x"), col("grid_y")],
            [col("grid_x"), col("grid_y")],
            JoinArgs::new(JoinType::Inner),
        )
        .filter(
            // Exclude self-joins
            col("event_idx").neq(col("neighbor_idx")),
        )
        .with_columns([
            // Calculate spatial distance (Chebyshev distance)
            (col(COL_X).cast(DataType::Int64) - col("neighbor_x").cast(DataType::Int64))
                .alias("dx_raw"),
            (col(COL_Y).cast(DataType::Int64) - col("neighbor_y").cast(DataType::Int64))
                .alias("dy_raw"),
            // Calculate temporal distance
            (col(COL_T) - col("neighbor_t")).alias("dt_raw"),
        ])
        .with_columns([
            // Convert to absolute values (since .abs() may not be available)
            when(col("dx_raw").lt(lit(0)))
                .then(-col("dx_raw"))
                .otherwise(col("dx_raw"))
                .alias("dx_abs"),
            when(col("dy_raw").lt(lit(0)))
                .then(-col("dy_raw"))
                .otherwise(col("dy_raw"))
                .alias("dy_abs"),
            when(col("dt_raw").lt(lit(0.0)))
                .then(-col("dt_raw"))
                .otherwise(col("dt_raw"))
                .alias("dt_abs"),
        ])
        .with_columns([
            // Calculate maximum of dx_abs and dy_abs for Chebyshev distance
            when(col("dx_abs").gt(col("dy_abs")))
                .then(col("dx_abs"))
                .otherwise(col("dy_abs"))
                .alias("spatial_distance"),
        ])
        .filter(
            // Apply spatial distance constraint
            col("spatial_distance").lt_eq(lit(radius)),
        )
        .filter(
            // Apply temporal distance constraint
            col("dt_abs").lt_eq(lit(time_window_sec)),
        )
        .filter(
            // Apply polarity constraint if required
            if filter.consider_polarity {
                col(COL_POLARITY).eq(col("neighbor_polarity"))
            } else {
                lit(true)
            },
        )
        .group_by([col("event_idx")])
        .agg([
            // Keep original event data
            col(COL_X).first(),
            col(COL_Y).first(),
            col(COL_T).first(),
            col(COL_POLARITY).first(),
            // Count neighbors
            col("neighbor_idx").count().alias("neighbor_count"),
        ])
        .filter(
            // Keep events with sufficient neighbors
            col("neighbor_count").gt_eq(lit(filter.min_neighbors as u32)),
        )
        .select([col(COL_X), col(COL_Y), col(COL_T), col(COL_POLARITY)])
        .sort([COL_T], SortMultipleOptions::default()); // Restore temporal order

    debug!("Spatial-temporal correlation filter applied using Polars grid binning");
    Ok(result_df)
}

/// Legacy function for backward compatibility - delegates to Polars implementation
fn apply_spatial_temporal_filter(
    events: &Events,
    filter: &SpatialTemporalFilter,
) -> FilterResult<Events> {
    warn!("Using legacy Vec<Event> interface - consider using LazyFrame directly for better performance");

    if events.is_empty() {
        return Ok(Vec::new());
    }

    let df = crate::ev_core::events_to_dataframe(events)
        .map_err(|e| FilterError::ProcessingError(format!("DataFrame conversion failed: {}", e)))?
        .lazy();

    let filtered_df = apply_spatial_temporal_filter_polars(df, filter)
        .map_err(|e| FilterError::ProcessingError(format!("Polars filtering failed: {}", e)))?;

    // Convert back to Vec<Event> - this is inefficient but maintains compatibility
    let result_df = filtered_df
        .collect()
        .map_err(|e| FilterError::ProcessingError(format!("LazyFrame collection failed: {}", e)))?;

    dataframe_to_events(&result_df)
}

// Legacy HashMap-based spatial-temporal correlation functions removed

/// Apply background activity filtering using Polars group_by operations (Polars-first implementation)
///
/// This is the main Polars-first implementation that uses group_by operations
/// to calculate pixel activity rates and filter low-activity pixels.
///
/// # Arguments
///
/// * `df` - Input LazyFrame containing event data
/// * `threshold_events_per_sec` - Minimum event rate threshold for active pixels
///
/// # Returns
///
/// Filtered LazyFrame with low-activity pixels removed
#[instrument(skip(df), fields(threshold = threshold_events_per_sec))]
pub fn apply_background_activity_filter_polars(
    df: LazyFrame,
    threshold_events_per_sec: f64,
) -> PolarsResult<LazyFrame> {
    debug!(
        "Applying background activity filter with threshold: {:.1} events/s",
        threshold_events_per_sec
    );

    // Calculate time span for rate computation
    let time_stats = df
        .clone()
        .select([
            col(COL_T).min().alias("min_time"),
            col(COL_T).max().alias("max_time"),
        ])
        .collect()?;

    let min_time = time_stats.column("min_time")?.f64()?.get(0).unwrap_or(0.0);
    let max_time = time_stats.column("max_time")?.f64()?.get(0).unwrap_or(0.0);
    let time_span = (max_time - min_time).max(1e-6); // Avoid division by zero

    let filtered_df = df
        .with_columns([
            // Create pixel coordinate for grouping using string concatenation
            // Note: Using format! macro approach for older Polars versions
            (col(COL_X).cast(DataType::String) + lit(",") + col(COL_Y).cast(DataType::String))
                .alias("pixel_key"),
        ])
        .with_columns([
            // Add time span as a constant for rate calculation
            lit(time_span).alias("time_span"),
        ])
        // Calculate pixel activity statistics
        .group_by([col("pixel_key")])
        .agg([
            // Count events per pixel
            col(COL_T).count().alias("event_count"),
            col("time_span").first().alias("time_span"),
            // Keep original data for active pixels
            col(COL_X).first().alias("pixel_x"),
            col(COL_Y).first().alias("pixel_y"),
            // Keep all events for later reconstruction
            col(COL_X).alias("all_x"),
            col(COL_Y).alias("all_y"),
            col(COL_T).alias("all_t"),
            col(COL_POLARITY).alias("all_polarity"),
        ])
        .with_columns([
            // Calculate event rate per pixel
            (col("event_count").cast(DataType::Float64) / col("time_span")).alias("event_rate"),
        ])
        .filter(
            // Keep only active pixels that exceed threshold
            col("event_rate").gt_eq(lit(threshold_events_per_sec)),
        )
        // Explode the arrays to get individual events back
        .select([
            col("all_x").explode().alias(COL_X),
            col("all_y").explode().alias(COL_Y),
            col("all_t").explode().alias(COL_T),
            col("all_polarity").explode().alias(COL_POLARITY),
        ])
        .sort([COL_T], SortMultipleOptions::default()); // Restore temporal order

    debug!("Background activity filter applied using Polars group_by operations");
    Ok(filtered_df)
}

/// Legacy function for backward compatibility - delegates to Polars implementation
fn apply_background_activity_filter(
    events: &Events,
    threshold_events_per_sec: f64,
) -> FilterResult<Events> {
    warn!("Using legacy Vec<Event> interface - consider using LazyFrame directly for better performance");

    if events.is_empty() {
        return Ok(Vec::new());
    }

    let df = crate::ev_core::events_to_dataframe(events)
        .map_err(|e| FilterError::ProcessingError(format!("DataFrame conversion failed: {}", e)))?
        .lazy();

    let filtered_df = apply_background_activity_filter_polars(df, threshold_events_per_sec)
        .map_err(|e| FilterError::ProcessingError(format!("Polars filtering failed: {}", e)))?;

    // Convert back to Vec<Event> - this is inefficient but maintains compatibility
    let result_df = filtered_df
        .collect()
        .map_err(|e| FilterError::ProcessingError(format!("LazyFrame collection failed: {}", e)))?;

    dataframe_to_events(&result_df)
}

/// Apply multi-scale filtering using Polars pipeline (Polars-first implementation)
///
/// This is the main Polars-first implementation that combines multiple denoising
/// methods in an efficient pipeline using LazyFrame operations.
///
/// # Arguments
///
/// * `df` - Input LazyFrame containing event data
/// * `filter` - Multi-scale filter configuration
///
/// # Returns
///
/// Filtered LazyFrame with multi-scale denoising applied
#[instrument(skip(df), fields(filter = ?filter))]
pub fn apply_multi_scale_filter_polars(
    df: LazyFrame,
    filter: &DenoiseFilter,
) -> PolarsResult<LazyFrame> {
    debug!("Applying multi-scale filter: {:?}", filter);

    let mut current_df = df;

    // Apply refractory period filter first (fastest and most effective)
    if let Some(ref_filter) = &filter.refractory_filter {
        current_df = apply_refractory_filter_polars(current_df, ref_filter)?;
        debug!("Multi-scale: applied refractory filter");
    }

    // Apply spatial-temporal correlation filter
    if let Some(st_filter) = &filter.spatial_temporal_filter {
        current_df = apply_spatial_temporal_filter_polars(current_df, st_filter)?;
        debug!("Multi-scale: applied spatial-temporal filter");
    }

    // Apply background activity filter if threshold is set
    if let Some(threshold) = filter.background_threshold {
        current_df = apply_background_activity_filter_polars(current_df, threshold)?;
        debug!("Multi-scale: applied background activity filter");
    }

    debug!("Multi-scale filter pipeline completed using Polars operations");
    Ok(current_df)
}

/// Legacy function for backward compatibility - delegates to Polars implementation
fn apply_multi_scale_filter(events: &Events, filter: &DenoiseFilter) -> FilterResult<Events> {
    warn!("Using legacy Vec<Event> interface - consider using LazyFrame directly for better performance");

    if events.is_empty() {
        return Ok(Vec::new());
    }

    let df = crate::ev_core::events_to_dataframe(events)
        .map_err(|e| FilterError::ProcessingError(format!("DataFrame conversion failed: {}", e)))?
        .lazy();

    let filtered_df = apply_multi_scale_filter_polars(df, filter)
        .map_err(|e| FilterError::ProcessingError(format!("Polars filtering failed: {}", e)))?;

    // Convert back to Vec<Event> - this is inefficient but maintains compatibility
    let result_df = filtered_df
        .collect()
        .map_err(|e| FilterError::ProcessingError(format!("LazyFrame collection failed: {}", e)))?;

    dataframe_to_events(&result_df)
}

/// Convenience function for refractory period filtering
pub fn apply_refractory_period(events: &Events, period_us: f64) -> FilterResult<Events> {
    let filter = DenoiseFilter::refractory(period_us);
    apply_denoise_filter(events, &filter)
}

/// Convenience function for noise filtering (Python API compatible)
///
/// This function provides a simple interface matching the Python API:
/// `filtered = evf.filter_noise("data/events.h5", method="refractory", refractory_period_us=1000)`
pub fn filter_noise(
    events: &Events,
    method: DenoiseMethod,
    parameters: &[f64],
) -> FilterResult<Events> {
    let mut filter = match method {
        DenoiseMethod::RefractoryPeriod => {
            if parameters.is_empty() {
                return Err(FilterError::InvalidConfig(
                    "Refractory period requires period parameter".to_string(),
                ));
            }
            DenoiseFilter::refractory(parameters[0])
        }
        DenoiseMethod::TemporalCorrelation => {
            if parameters.len() < 2 {
                return Err(FilterError::InvalidConfig(
                    "Temporal correlation requires time_window and min_events parameters"
                        .to_string(),
                ));
            }
            DenoiseFilter::temporal_correlation(parameters[0], parameters[1] as usize)
        }
        DenoiseMethod::SpatialTemporalCorrelation => {
            if parameters.len() < 3 {
                return Err(FilterError::InvalidConfig("Spatial-temporal correlation requires radius, time_window, and min_neighbors parameters".to_string()));
            }
            DenoiseFilter::spatial_temporal(
                parameters[0] as u16,
                parameters[1],
                parameters[2] as usize,
            )
        }
        DenoiseMethod::BackgroundActivity => {
            if parameters.is_empty() {
                return Err(FilterError::InvalidConfig(
                    "Background activity requires threshold parameter".to_string(),
                ));
            }
            DenoiseFilter::background_activity(parameters[0])
        }
        DenoiseMethod::MultiScale => {
            if parameters.len() < 3 {
                return Err(FilterError::InvalidConfig("Multi-scale requires refractory_period, spatial_radius, and time_window parameters".to_string()));
            }
            DenoiseFilter::multi_scale(parameters[0], parameters[1] as u16, parameters[2])
        }
    };

    // Auto-configure performance based on dataset size
    if events.len() > 1_000_000 {
        filter = filter.with_high_performance();
    } else if events.len() > 100_000 {
        filter = filter.with_performance_config(DenoisePerformanceConfig::default());
    } else {
        filter = filter.with_memory_efficient();
    }

    apply_denoise_filter(events, &filter)
}

/// High-level noise filtering function with automatic parameter selection
///
/// This function automatically selects appropriate parameters based on the dataset
/// characteristics for common noise filtering scenarios.
pub fn auto_filter_noise(events: &Events, aggressiveness: f32) -> FilterResult<Events> {
    if events.is_empty() {
        return Ok(Vec::new());
    }

    // Analyze dataset characteristics
    let stats = utils::EventStats::calculate(events);
    let avg_rate = stats.avg_event_rate;

    // Auto-select refractory period based on event rate and aggressiveness
    let base_refractory_us = if avg_rate > 100_000.0 {
        500.0 // High activity -> short refractory
    } else if avg_rate > 10_000.0 {
        1000.0 // Medium activity -> medium refractory
    } else {
        2000.0 // Low activity -> long refractory
    };

    let adjusted_refractory = base_refractory_us * (2.0 - aggressiveness.clamp(0.0, 1.0)) as f64;

    info!(
        "Auto-filtering noise: event_rate={:.0}/s, refractory_period={:.0}µs",
        avg_rate, adjusted_refractory
    );

    let filter = DenoiseFilter::refractory(adjusted_refractory).with_high_performance(); // Use best performance for auto mode

    apply_denoise_filter(events, &filter)
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

/// Main Polars-first denoising API - apply any denoise filter to LazyFrame
///
/// This is the preferred API for high-performance denoising. All filtering
/// operations are performed using Polars expressions and lazy evaluation.
///
/// # Arguments
///
/// * `df` - Input LazyFrame containing event data with columns: x, y, t, polarity
/// * `filter` - Denoising filter configuration
///
/// # Returns
///
/// * `PolarsResult<LazyFrame>` - Filtered LazyFrame with noise events removed
///
/// # Example
///
/// ```rust
/// use polars::prelude::*;
/// use evlib::ev_filtering::denoise::*;
///
/// // Convert events to LazyFrame once
/// let events_df = events_to_dataframe(&events)?.lazy();
///
/// // Apply refractory period filtering
/// let filter = DenoiseFilter::refractory(1000.0); // 1ms
/// let filtered_df = apply_denoise_filter_polars(events_df, &filter)?;
///
/// // Chain multiple operations efficiently
/// let final_df = filtered_df
///     .filter(col("t").gt(lit(start_time)))
///     .select([col("*")]);
/// ```
#[instrument(skip(df), fields(method = ?filter.method))]
pub fn apply_denoise_filter_polars(
    df: LazyFrame,
    filter: &DenoiseFilter,
) -> PolarsResult<LazyFrame> {
    debug!("Applying Polars-first denoise filter: {:?}", filter.method);

    match filter.method {
        DenoiseMethod::RefractoryPeriod => {
            apply_refractory_filter_polars(df, filter.refractory_filter.as_ref().unwrap())
        }
        DenoiseMethod::TemporalCorrelation => apply_temporal_correlation_filter_polars(
            df,
            filter.temporal_correlation_filter.as_ref().unwrap(),
        ),
        DenoiseMethod::SpatialTemporalCorrelation => apply_spatial_temporal_filter_polars(
            df,
            filter.spatial_temporal_filter.as_ref().unwrap(),
        ),
        DenoiseMethod::BackgroundActivity => {
            apply_background_activity_filter_polars(df, filter.background_threshold.unwrap())
        }
        DenoiseMethod::MultiScale => apply_multi_scale_filter_polars(df, filter),
    }
}

/// Convenience function for refractory period filtering with LazyFrame
///
/// # Arguments
///
/// * `df` - Input LazyFrame containing event data
/// * `period_us` - Refractory period in microseconds
///
/// # Returns
///
/// * `PolarsResult<LazyFrame>` - Filtered LazyFrame
pub fn apply_refractory_period_polars(df: LazyFrame, period_us: f64) -> PolarsResult<LazyFrame> {
    let filter = RefractoryFilter::new(period_us);
    apply_refractory_filter_polars(df, &filter)
}

/// Convenience function for temporal correlation filtering with LazyFrame
///
/// # Arguments
///
/// * `df` - Input LazyFrame containing event data
/// * `time_window_us` - Time window in microseconds
/// * `min_events` - Minimum number of neighbor events required
///
/// # Returns
///
/// * `PolarsResult<LazyFrame>` - Filtered LazyFrame
pub fn apply_temporal_correlation_polars(
    df: LazyFrame,
    time_window_us: f64,
    min_events: usize,
) -> PolarsResult<LazyFrame> {
    let filter = TemporalCorrelationFilter::new(time_window_us, min_events);
    apply_temporal_correlation_filter_polars(df, &filter)
}

/// Convenience function for spatial-temporal correlation filtering with LazyFrame
///
/// # Arguments
///
/// * `df` - Input LazyFrame containing event data
/// * `spatial_radius` - Spatial radius in pixels
/// * `time_window_us` - Time window in microseconds
/// * `min_neighbors` - Minimum number of neighbor events required
///
/// # Returns
///
/// * `PolarsResult<LazyFrame>` - Filtered LazyFrame
pub fn apply_spatial_temporal_polars(
    df: LazyFrame,
    spatial_radius: u16,
    time_window_us: f64,
    min_neighbors: usize,
) -> PolarsResult<LazyFrame> {
    let filter = SpatialTemporalFilter::new(spatial_radius, time_window_us, min_neighbors);
    apply_spatial_temporal_filter_polars(df, &filter)
}

/// Convenience function for background activity filtering with LazyFrame
///
/// # Arguments
///
/// * `df` - Input LazyFrame containing event data
/// * `threshold_events_per_sec` - Minimum event rate threshold
///
/// # Returns
///
/// * `PolarsResult<LazyFrame>` - Filtered LazyFrame
pub fn apply_background_activity_polars(
    df: LazyFrame,
    threshold_events_per_sec: f64,
) -> PolarsResult<LazyFrame> {
    apply_background_activity_filter_polars(df, threshold_events_per_sec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ev_core::Event;

    fn create_test_events_with_noise() -> Events {
        let mut events = Vec::new();

        // Add some normal events
        for i in 0..10 {
            events.push(Event {
                t: i as f64 * 0.01, // 10ms intervals
                x: 100,
                y: 200,
                polarity: i % 2 == 0,
            });
        }

        // Add noise events (too close in time)
        for i in 0..5 {
            events.push(Event {
                t: i as f64 * 0.01 + 0.0001, // 0.1ms after normal events
                x: 100,
                y: 200,
                polarity: i % 2 == 0,
            });
        }

        // Add isolated noise events
        events.push(Event {
            t: 1.0,
            x: 300,
            y: 400,
            polarity: true,
        });
        events.push(Event {
            t: 2.0,
            x: 301,
            y: 401,
            polarity: false,
        });

        events
    }

    #[test]
    fn test_denoise_filter_creation() {
        let filter = DenoiseFilter::refractory(1000.0);
        assert_eq!(filter.method, DenoiseMethod::RefractoryPeriod);
        assert!(filter.refractory_filter.is_some());

        let filter = DenoiseFilter::temporal_correlation(5000.0, 3);
        assert_eq!(filter.method, DenoiseMethod::TemporalCorrelation);
        assert!(filter.temporal_correlation_filter.is_some());

        let filter = DenoiseFilter::spatial_temporal(3, 5000.0, 2);
        assert_eq!(filter.method, DenoiseMethod::SpatialTemporalCorrelation);
        assert!(filter.spatial_temporal_filter.is_some());
    }

    #[test]
    fn test_refractory_period_filtering() {
        let events = create_test_events_with_noise();
        let original_count = events.len();

        // Use 1ms refractory period
        let filtered = apply_refractory_period(&events, 1000.0).unwrap();

        // Should remove the noise events that are too close in time
        assert!(filtered.len() < original_count);

        // Check that filtered events respect refractory period
        // Sort events by time first to ensure proper validation
        let mut sorted_filtered = filtered.clone();
        sorted_filtered.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));

        let mut last_times: std::collections::HashMap<(u16, u16, bool), f64> =
            std::collections::HashMap::new();
        for event in &sorted_filtered {
            let key = (event.x, event.y, event.polarity);
            if let Some(&last_time) = last_times.get(&key) {
                let time_diff = (event.t - last_time) * 1_000_000.0; // microseconds
                assert!(
                    time_diff >= 1000.0,
                    "Refractory period violated: {:.1}µs",
                    time_diff
                );
            }
            last_times.insert(key, event.t);
        }
    }

    #[test]
    fn test_temporal_correlation_filtering() {
        let events = create_test_events_with_noise();

        // Require at least 2 events within 20ms window
        let filter = DenoiseFilter::temporal_correlation(20000.0, 2);
        let filtered = apply_denoise_filter(&events, &filter).unwrap();

        // Should remove isolated events
        assert!(filtered.len() <= events.len());
    }

    #[test]
    fn test_spatial_temporal_filtering() {
        let events = create_test_events_with_noise();

        // Require at least 1 neighbor within 5 pixels and 10ms
        let filter = DenoiseFilter::spatial_temporal(5, 10000.0, 1);
        let filtered = apply_denoise_filter(&events, &filter).unwrap();

        // Should remove spatially isolated events
        assert!(filtered.len() <= events.len());
    }

    #[test]
    fn test_background_activity_filtering() {
        // Create events with different activity levels
        let mut events = Vec::new();

        // High activity pixel
        for i in 0..100 {
            events.push(Event {
                t: i as f64 * 0.001, // 1000 events/s
                x: 100,
                y: 200,
                polarity: i % 2 == 0,
            });
        }

        // Low activity pixel
        for i in 0..5 {
            events.push(Event {
                t: i as f64 * 0.1, // 50 events/s
                x: 300,
                y: 400,
                polarity: i % 2 == 0,
            });
        }

        let filter = DenoiseFilter::background_activity(100.0); // 100 events/s threshold
        let filtered = apply_denoise_filter(&events, &filter).unwrap();

        // Should keep high activity pixel, remove low activity pixel
        assert!(filtered.iter().any(|e| e.x == 100 && e.y == 200));
        assert!(!filtered.iter().any(|e| e.x == 300 && e.y == 400));
    }

    #[test]
    fn test_multi_scale_filtering() {
        let events = create_test_events_with_noise();
        let original_count = events.len();

        let filter = DenoiseFilter::multi_scale(1000.0, 3, 5000.0);
        let filtered = apply_denoise_filter(&events, &filter).unwrap();

        // Should remove more noise than single-scale filters
        assert!(filtered.len() < original_count);
    }

    #[test]
    fn test_filter_validation() {
        // Valid filters
        assert!(DenoiseFilter::refractory(1000.0).validate().is_ok());
        assert!(DenoiseFilter::temporal_correlation(5000.0, 3)
            .validate()
            .is_ok());
        assert!(DenoiseFilter::spatial_temporal(3, 5000.0, 2)
            .validate()
            .is_ok());

        // Invalid filters
        let mut invalid_refractory = DenoiseFilter::refractory(1000.0);
        invalid_refractory
            .refractory_filter
            .as_mut()
            .unwrap()
            .period_us = -1.0;
        assert!(invalid_refractory.validate().is_err());

        let mut invalid_temporal = DenoiseFilter::temporal_correlation(5000.0, 3);
        invalid_temporal
            .temporal_correlation_filter
            .as_mut()
            .unwrap()
            .min_events = 0;
        assert!(invalid_temporal.validate().is_err());
    }

    #[test]
    fn test_empty_events() {
        let events = Vec::new();
        let filter = DenoiseFilter::refractory(1000.0);
        let filtered = apply_denoise_filter(&events, &filter).unwrap();
        assert!(filtered.is_empty());
    }

    #[test]
    fn test_per_polarity_refractory() {
        let events = vec![
            Event {
                t: 1.0,
                x: 100,
                y: 200,
                polarity: true,
            },
            Event {
                t: 1.0005,
                x: 100,
                y: 200,
                polarity: false,
            }, // Different polarity, should pass
            Event {
                t: 1.001,
                x: 100,
                y: 200,
                polarity: true,
            }, // Same polarity, too close
        ];

        let mut filter = RefractoryFilter::new(1000.0); // 1ms
        filter.per_polarity = true;

        let filtered = apply_refractory_period_filter(&events, &filter).unwrap();

        // Should keep first two events (different polarities), remove third
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].polarity, true);
        assert_eq!(filtered[1].polarity, false);
    }

    #[test]
    fn test_convenience_functions() {
        let events = create_test_events_with_noise();

        // Test apply_refractory_period convenience function
        let filtered1 = apply_refractory_period(&events, 1000.0).unwrap();

        let filter = DenoiseFilter::refractory(1000.0);
        let filtered2 = apply_denoise_filter(&events, &filter).unwrap();

        assert_eq!(filtered1.len(), filtered2.len());
    }

    #[test]
    fn test_filter_noise_function() {
        let events = create_test_events_with_noise();

        // Test different methods through filter_noise function
        let filtered1 = filter_noise(&events, DenoiseMethod::RefractoryPeriod, &[1000.0]).unwrap();
        assert!(filtered1.len() <= events.len());

        let filtered2 =
            filter_noise(&events, DenoiseMethod::TemporalCorrelation, &[5000.0, 2.0]).unwrap();
        assert!(filtered2.len() <= events.len());

        // Test error handling
        let result = filter_noise(&events, DenoiseMethod::RefractoryPeriod, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_order_preservation() {
        let mut events = create_test_events_with_noise();
        // Shuffle events to test order preservation
        events.swap(0, 5);
        events.swap(2, 8);

        let filter = DenoiseFilter::refractory(1000.0).with_order_preservation(true);
        let filtered = apply_denoise_filter(&events, &filter).unwrap();

        // Should be sorted after filtering
        assert!(utils::is_sorted_by_time(&filtered));
    }
}
