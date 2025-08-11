//! Polars-first hot pixel detection and removal for event camera data
//!
//! This module provides functionality for detecting and filtering out hot pixels
//! (pixels that generate an unusually high number of events) which are common
//! artifacts in event cameras due to sensor defects or environmental factors.
//!
//! # Philosophy
//!
//! This implementation follows a strict Polars-first approach:
//! - Input: LazyFrame (from events_to_dataframe)
//! - Processing: Polars expressions and group_by operations
//! - Output: LazyFrame (convertible to Vec<Event> only when needed)
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
//! use evlib::ev_filtering::hot_pixel::*;
//!
//! // Convert events to LazyFrame once
//! let events_df = events_to_dataframe(&events)?.lazy();
//!
//! // Apply hot pixel filtering with Polars expressions
//! let filter = HotPixelFilter::percentile(99.5);
//! let filtered = apply_hot_pixel_filter(events_df, &filter)?;
//! ```

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

/// Polars column names for event data (consistent with temporal.rs)
pub const COL_X: &str = "x";
pub const COL_Y: &str = "y";
pub const COL_T: &str = "t";
pub const COL_POLARITY: &str = "polarity";

/// Default percentile threshold for Python compatibility (99.9th percentile)
const DEFAULT_PERCENTILE_THRESHOLD: f64 = 99.9;

/// Minimum events required for hot pixel detection
const MIN_EVENTS_FOR_DETECTION: u32 = 10;

/// Threshold for large dataset parallel processing
#[allow(dead_code)]
const LARGE_DATASET_THRESHOLD: usize = 100_000;

/// Methods for detecting hot pixels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HotPixelDetectionMethod {
    /// Statistical outlier detection based on event count percentiles
    Percentile,
    /// Standard deviation-based outlier detection
    StandardDeviation,
    /// Fixed threshold based on absolute event count
    FixedThreshold,
    /// Dynamic threshold based on local neighborhood statistics
    LocalAdaptive,
    /// Frequency domain analysis using time windows
    FrequencyAnalysis,
}

impl HotPixelDetectionMethod {
    /// Get a description of this detection method
    pub fn description(&self) -> &'static str {
        match self {
            HotPixelDetectionMethod::Percentile => "percentile threshold",
            HotPixelDetectionMethod::StandardDeviation => "standard deviation",
            HotPixelDetectionMethod::FixedThreshold => "fixed threshold",
            HotPixelDetectionMethod::LocalAdaptive => "local adaptive",
            HotPixelDetectionMethod::FrequencyAnalysis => "frequency analysis",
        }
    }
}

/// Configuration for hot pixel detection and removal optimized for Polars operations
#[derive(Debug, Clone)]
pub struct HotPixelFilter {
    /// Detection method to use
    pub method: HotPixelDetectionMethod,
    /// Percentile threshold for percentile method (e.g., 99.5)
    pub percentile_threshold: Option<f64>,
    /// Standard deviation multiplier for std dev method (e.g., 3.0)
    pub std_dev_multiplier: Option<f64>,
    /// Fixed event count threshold
    pub fixed_threshold: Option<u32>,
    /// Radius for local adaptive method
    pub adaptive_radius: Option<u16>,
    /// Minimum time span for frequency analysis (seconds)
    pub frequency_time_span: Option<f64>,
    /// Whether to use temporal information in detection
    pub use_temporal_info: bool,
    /// Minimum events required for a pixel to be considered for hot pixel detection
    pub min_events_for_detection: u32,
    /// Whether to validate detected hot pixels
    pub validate_detections: bool,
}

impl HotPixelFilter {
    /// Create a percentile-based hot pixel filter
    ///
    /// # Arguments
    /// * `percentile` - Percentile threshold (e.g., 99.5 for 99.5th percentile)
    pub fn percentile(percentile: f64) -> Self {
        Self {
            method: HotPixelDetectionMethod::Percentile,
            percentile_threshold: Some(percentile),
            std_dev_multiplier: None,
            fixed_threshold: None,
            adaptive_radius: None,
            frequency_time_span: None,
            use_temporal_info: false,
            min_events_for_detection: MIN_EVENTS_FOR_DETECTION,
            validate_detections: true,
        }
    }

    /// Create a standard deviation-based hot pixel filter
    ///
    /// # Arguments
    /// * `multiplier` - Standard deviation multiplier (e.g., 3.0 for 3-sigma rule)
    pub fn standard_deviation(multiplier: f64) -> Self {
        Self {
            method: HotPixelDetectionMethod::StandardDeviation,
            percentile_threshold: None,
            std_dev_multiplier: Some(multiplier),
            fixed_threshold: None,
            adaptive_radius: None,
            frequency_time_span: None,
            use_temporal_info: false,
            min_events_for_detection: MIN_EVENTS_FOR_DETECTION,
            validate_detections: true,
        }
    }

    /// Create a fixed threshold hot pixel filter
    ///
    /// # Arguments
    /// * `threshold` - Minimum event count to be considered a hot pixel
    pub fn fixed_threshold(threshold: u32) -> Self {
        Self {
            method: HotPixelDetectionMethod::FixedThreshold,
            percentile_threshold: None,
            std_dev_multiplier: None,
            fixed_threshold: Some(threshold),
            adaptive_radius: None,
            frequency_time_span: None,
            use_temporal_info: false,
            min_events_for_detection: MIN_EVENTS_FOR_DETECTION,
            validate_detections: true,
        }
    }

    /// Create a local adaptive hot pixel filter
    ///
    /// # Arguments
    /// * `radius` - Radius for local neighborhood analysis
    pub fn local_adaptive(radius: u16) -> Self {
        Self {
            method: HotPixelDetectionMethod::LocalAdaptive,
            percentile_threshold: None,
            std_dev_multiplier: None,
            fixed_threshold: None,
            adaptive_radius: Some(radius),
            frequency_time_span: None,
            use_temporal_info: true,
            min_events_for_detection: MIN_EVENTS_FOR_DETECTION,
            validate_detections: true,
        }
    }

    /// Create a frequency analysis-based hot pixel filter
    ///
    /// # Arguments
    /// * `time_span` - Time span for frequency analysis (seconds)
    pub fn frequency_analysis(time_span: f64) -> Self {
        Self {
            method: HotPixelDetectionMethod::FrequencyAnalysis,
            percentile_threshold: None,
            std_dev_multiplier: None,
            fixed_threshold: None,
            adaptive_radius: None,
            frequency_time_span: Some(time_span),
            use_temporal_info: true,
            min_events_for_detection: 50,
            validate_detections: true,
        }
    }

    /// Enable temporal information usage
    pub fn with_temporal_info(mut self, use_temporal: bool) -> Self {
        self.use_temporal_info = use_temporal;
        self
    }

    /// Set minimum events required for detection
    pub fn with_min_events(mut self, min_events: u32) -> Self {
        self.min_events_for_detection = min_events;
        self
    }

    /// Enable or disable detection validation
    pub fn with_validation(mut self, validate: bool) -> Self {
        self.validate_detections = validate;
        self
    }

    /// Get description of this filter
    pub fn description(&self) -> String {
        let mut parts = vec![self.method.description().to_string()];

        match self.method {
            HotPixelDetectionMethod::Percentile => {
                if let Some(p) = self.percentile_threshold {
                    parts.push(format!("{:.1}th percentile", p));
                }
            }
            HotPixelDetectionMethod::StandardDeviation => {
                if let Some(m) = self.std_dev_multiplier {
                    parts.push(format!("{:.1}σ", m));
                }
            }
            HotPixelDetectionMethod::FixedThreshold => {
                if let Some(t) = self.fixed_threshold {
                    parts.push(format!(">={} events", t));
                }
            }
            HotPixelDetectionMethod::LocalAdaptive => {
                if let Some(r) = self.adaptive_radius {
                    parts.push(format!("radius {}", r));
                }
            }
            HotPixelDetectionMethod::FrequencyAnalysis => {
                if let Some(t) = self.frequency_time_span {
                    parts.push(format!("{:.3}s window", t));
                }
            }
        }

        parts.join(", ")
    }

    /// Convert filter to Polars expression for hot pixel detection
    ///
    /// This creates a Polars expression that identifies hot pixels based on
    /// the configured detection method and parameters.
    #[cfg_attr(feature = "tracing", instrument(skip(pixel_stats_df)))]
    pub fn to_hot_pixel_expr(&self, pixel_stats_df: &LazyFrame) -> PolarsResult<Option<Expr>> {
        match self.method {
            HotPixelDetectionMethod::Percentile => {
                if let Some(percentile) = self.percentile_threshold {
                    self.build_percentile_expr(pixel_stats_df, percentile)
                } else {
                    Ok(None)
                }
            }
            HotPixelDetectionMethod::StandardDeviation => {
                if let Some(multiplier) = self.std_dev_multiplier {
                    self.build_std_dev_expr(pixel_stats_df, multiplier)
                } else {
                    Ok(None)
                }
            }
            HotPixelDetectionMethod::FixedThreshold => {
                if let Some(threshold) = self.fixed_threshold {
                    Ok(Some(col("total_events").gt(lit(threshold))))
                } else {
                    Ok(None)
                }
            }
            HotPixelDetectionMethod::LocalAdaptive => {
                if let Some(radius) = self.adaptive_radius {
                    self.build_local_adaptive_expr(pixel_stats_df, radius)
                } else {
                    Ok(None)
                }
            }
            HotPixelDetectionMethod::FrequencyAnalysis => {
                if let Some(time_span) = self.frequency_time_span {
                    self.build_frequency_expr(pixel_stats_df, time_span)
                } else {
                    Ok(None)
                }
            }
        }
    }

    /// Build percentile-based detection expression
    fn build_percentile_expr(
        &self,
        pixel_stats_df: &LazyFrame,
        percentile: f64,
    ) -> PolarsResult<Option<Expr>> {
        // Calculate the percentile threshold using Polars quantile function
        let threshold_df = pixel_stats_df
            .clone()
            .select([col("total_events")
                .quantile(lit(percentile / 100.0), QuantileMethod::Linear)
                .alias("threshold")])
            .collect()?;

        let threshold = threshold_df
            .column("threshold")?
            .get(0)?
            .try_extract::<f64>()?;

        debug!("Percentile threshold ({}%): {}", percentile, threshold);

        Ok(Some(col("total_events").gt(lit(threshold))))
    }

    /// Build standard deviation-based detection expression
    fn build_std_dev_expr(
        &self,
        pixel_stats_df: &LazyFrame,
        multiplier: f64,
    ) -> PolarsResult<Option<Expr>> {
        // Calculate mean and std using Polars aggregations
        let stats_df = pixel_stats_df
            .clone()
            .select([
                col("total_events").mean().alias("mean_events"),
                col("total_events").std(1).alias("std_events"),
            ])
            .collect()?;

        let mean = stats_df
            .column("mean_events")?
            .get(0)?
            .try_extract::<f64>()?;
        let std_dev = stats_df
            .column("std_events")?
            .get(0)?
            .try_extract::<f64>()?;
        let threshold = mean + multiplier * std_dev;

        debug!(
            "Standard deviation threshold ({}σ): {} (mean: {}, std: {})",
            multiplier, threshold, mean, std_dev
        );

        Ok(Some(col("total_events").gt(lit(threshold))))
    }

    /// Build local adaptive detection expression (simplified for Polars)
    fn build_local_adaptive_expr(
        &self,
        _pixel_stats_df: &LazyFrame,
        _radius: u16,
    ) -> PolarsResult<Option<Expr>> {
        // Local adaptive requires complex spatial operations
        // For now, fall back to percentile-based detection
        warn!("Local adaptive method not fully implemented in Polars-first approach, using percentile fallback");
        self.build_percentile_expr(_pixel_stats_df, 95.0)
    }

    /// Build frequency analysis detection expression
    fn build_frequency_expr(
        &self,
        _pixel_stats_df: &LazyFrame,
        time_span: f64,
    ) -> PolarsResult<Option<Expr>> {
        // Use event rate as a proxy for frequency analysis
        let expected_rate = 1.0 / time_span;
        let threshold = expected_rate * 10.0; // 10x higher than expected

        debug!("Frequency analysis threshold: {} events/s", threshold);

        Ok(Some(col("event_rate").gt(lit(threshold))))
    }
}

impl Default for HotPixelFilter {
    fn default() -> Self {
        Self::percentile(DEFAULT_PERCENTILE_THRESHOLD)
    }
}

impl Validatable for HotPixelFilter {
    fn validate(&self) -> FilterResult<()> {
        match self.method {
            HotPixelDetectionMethod::Percentile => {
                if let Some(p) = self.percentile_threshold {
                    if !(0.0..=100.0).contains(&p) {
                        return Err(FilterError::InvalidConfig(format!(
                            "Percentile must be between 0.0 and 100.0, got {}",
                            p
                        )));
                    }
                    if p < 50.0 {
                        warn!(
                            "Low percentile threshold ({:.1}) may remove many normal pixels",
                            p
                        );
                    }
                } else {
                    return Err(FilterError::InvalidConfig(
                        "Percentile method requires percentile_threshold".to_string(),
                    ));
                }
            }
            HotPixelDetectionMethod::StandardDeviation => {
                if let Some(m) = self.std_dev_multiplier {
                    if m <= 0.0 {
                        return Err(FilterError::InvalidConfig(
                            "Standard deviation multiplier must be positive".to_string(),
                        ));
                    }
                    if m < 2.0 {
                        warn!(
                            "Low std dev multiplier ({:.1}) may remove many normal pixels",
                            m
                        );
                    }
                } else {
                    return Err(FilterError::InvalidConfig(
                        "Standard deviation method requires std_dev_multiplier".to_string(),
                    ));
                }
            }
            HotPixelDetectionMethod::FixedThreshold => {
                if self.fixed_threshold.is_none() {
                    return Err(FilterError::InvalidConfig(
                        "Fixed threshold method requires fixed_threshold".to_string(),
                    ));
                }
            }
            HotPixelDetectionMethod::LocalAdaptive => {
                if self.adaptive_radius.is_none() {
                    return Err(FilterError::InvalidConfig(
                        "Local adaptive method requires adaptive_radius".to_string(),
                    ));
                }
            }
            HotPixelDetectionMethod::FrequencyAnalysis => {
                if let Some(t) = self.frequency_time_span {
                    if t <= 0.0 {
                        return Err(FilterError::InvalidConfig(
                            "Frequency analysis time span must be positive".to_string(),
                        ));
                    }
                } else {
                    return Err(FilterError::InvalidConfig(
                        "Frequency analysis method requires frequency_time_span".to_string(),
                    ));
                }
            }
        }

        if self.min_events_for_detection == 0 {
            return Err(FilterError::InvalidConfig(
                "Minimum events for detection must be positive".to_string(),
            ));
        }

        Ok(())
    }
}

/// Hot pixel detection results using Polars analysis
#[derive(Debug, Clone)]
pub struct HotPixelDetector {
    /// Detected hot pixel coordinates
    pub hot_pixels: Vec<(u16, u16)>,
    /// Hot pixel statistics as Polars DataFrame
    pub hot_pixel_stats: Option<DataFrame>,
    /// Detection threshold used
    pub threshold_used: f64,
    /// Total pixels analyzed
    pub total_pixels: usize,
    /// Detection method used
    pub method: HotPixelDetectionMethod,
    /// Comprehensive pixel analysis results
    pub pixel_analysis: Option<PixelActivityAnalysis>,
}

/// Comprehensive pixel activity analysis results
#[derive(Debug, Clone)]
pub struct PixelActivityAnalysis {
    pub total_active_pixels: usize,
    pub event_count_statistics: PixelStatistics,
    pub spatial_distribution: SpatialDistribution,
    pub temporal_patterns: TemporalPatterns,
    pub clustering_results: Option<ClusteringResults>,
}

/// Pixel statistics calculated using Polars aggregations
#[derive(Debug, Clone)]
pub struct PixelStatistics {
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub min: u32,
    pub max: u32,
    /// Percentiles as Polars DataFrame for efficient access
    pub percentiles_df: Option<DataFrame>,
    pub skewness: f64,
    pub kurtosis: f64,
}

/// Spatial distribution analysis of hot pixels
#[derive(Debug, Clone)]
pub struct SpatialDistribution {
    pub bounds: (u16, u16, u16, u16), // (min_x, max_x, min_y, max_y)
    pub avg_distance: f64,
    pub clustering_coefficient: f64,
    /// Density map as Polars DataFrame with x, y, density columns
    pub density_df: Option<DataFrame>,
}

/// Temporal patterns analysis of hot pixels
#[derive(Debug, Clone)]
pub struct TemporalPatterns {
    pub avg_firing_rate: f64,
    pub regularity_score: f64,
    pub burst_patterns: Vec<BurstPattern>,
    pub stability_score: f64,
}

/// Burst pattern detection result
#[derive(Debug, Clone)]
pub struct BurstPattern {
    pub pixel: (u16, u16),
    pub start_time: f64,
    pub end_time: f64,
    pub event_count: usize,
    pub peak_rate: f64,
}

/// Clustering analysis results
#[derive(Debug, Clone)]
pub struct ClusteringResults {
    pub num_clusters: usize,
    pub clusters: Vec<Vec<(u16, u16)>>,
    pub centroids: Vec<(f64, f64)>,
    pub silhouette_score: f64,
}

impl HotPixelDetector {
    /// Detect hot pixels using Polars operations
    ///
    /// This function uses Polars group_by and aggregation operations to efficiently
    /// identify hot pixels without manual iteration through events.
    #[cfg_attr(feature = "tracing", instrument(skip(df), fields(method = ?filter.method)))]
    pub fn detect_polars(df: LazyFrame, filter: &HotPixelFilter) -> PolarsResult<Self> {
        let start_time = Instant::now();

        // Validate filter configuration
        filter.validate().map_err(|e| {
            PolarsError::ComputeError(format!("Filter validation failed: {}", e).into())
        })?;

        // Calculate per-pixel statistics using Polars group_by
        let pixel_stats_df = calculate_pixel_statistics_polars(df.clone(), filter)?;

        if pixel_stats_df.height() == 0 {
            info!("No pixels meet minimum event count requirement");
            return Ok(Self::empty(filter.method));
        }

        // Detect hot pixels using configured method
        let hot_pixel_expr = filter
            .to_hot_pixel_expr(&pixel_stats_df.clone().lazy())?
            .ok_or_else(|| {
                PolarsError::ComputeError("Failed to build hot pixel detection expression".into())
            })?;

        let hot_pixel_df = pixel_stats_df
            .clone()
            .lazy()
            .filter(hot_pixel_expr)
            .collect()?;

        // Extract hot pixel coordinates using Polars operations
        let hot_pixels = extract_hot_pixel_coordinates_polars(&hot_pixel_df)?;

        // Keep the hot pixel statistics DataFrame for efficient access
        let hot_pixel_stats = if hot_pixel_df.height() > 0 {
            Some(hot_pixel_df.clone())
        } else {
            None
        };

        // Calculate detection threshold (simplified)
        let threshold_used = calculate_detection_threshold(&pixel_stats_df, filter)?;

        let processing_time = start_time.elapsed().as_secs_f64();
        info!(
            "Hot pixel detection ({}): {} hot pixels found from {} candidate pixels in {:.3}s",
            filter.method.description(),
            hot_pixels.len(),
            pixel_stats_df.height(),
            processing_time
        );

        // Perform comprehensive analysis if requested
        let pixel_analysis = if filter.validate_detections {
            Some(analyze_pixel_activity_polars(
                df,
                &pixel_stats_df,
                &hot_pixels,
            )?)
        } else {
            None
        };

        Ok(Self {
            hot_pixels,
            hot_pixel_stats,
            threshold_used,
            total_pixels: pixel_stats_df.height(),
            method: filter.method,
            pixel_analysis,
        })
    }

    /// Legacy interface for Vec<Event> - delegates to Polars implementation
    ///
    /// Create empty detector result
    fn empty(method: HotPixelDetectionMethod) -> Self {
        Self {
            hot_pixels: Vec::new(),
            hot_pixel_stats: None,
            threshold_used: 0.0,
            total_pixels: 0,
            method,
            pixel_analysis: None,
        }
    }

    /// Get the hot pixel locations
    pub fn hot_pixel_locations(&self) -> &[(u16, u16)] {
        &self.hot_pixels
    }

    /// Check if a pixel is detected as hot
    pub fn is_hot_pixel(&self, x: u16, y: u16) -> bool {
        self.hot_pixels.contains(&(x, y))
    }

    /// Get event count for a specific hot pixel using Polars DataFrame lookup
    pub fn get_pixel_event_count(&self, x: u16, y: u16) -> Option<u32> {
        if let Some(ref stats_df) = self.hot_pixel_stats {
            // Use Polars filter to find the pixel's statistics
            if let Ok(pixel_row_df) = stats_df
                .clone()
                .lazy()
                .filter(
                    col(COL_X)
                        .eq(lit(x as i64))
                        .and(col(COL_Y).eq(lit(y as i64))),
                )
                .select([col("total_events")])
                .collect()
            {
                if pixel_row_df.height() > 0 {
                    if let Ok(event_count) = pixel_row_df
                        .column("total_events")
                        .and_then(|col| col.get(0))
                        .and_then(|val| val.try_extract::<u32>())
                    {
                        return Some(event_count);
                    }
                }
            }
        }
        None
    }

    /// Get confidence score for a specific hot pixel from DataFrame
    pub fn get_confidence_score(&self, x: u16, y: u16) -> Option<f64> {
        if let Some(ref stats_df) = self.hot_pixel_stats {
            // Use confidence based on how far above threshold the pixel is
            if let Ok(pixel_row_df) = stats_df
                .clone()
                .lazy()
                .filter(
                    col(COL_X)
                        .eq(lit(x as i64))
                        .and(col(COL_Y).eq(lit(y as i64))),
                )
                .select([col("total_events")])
                .collect()
            {
                if pixel_row_df.height() > 0 {
                    if let Ok(event_count) = pixel_row_df
                        .column("total_events")
                        .and_then(|col| col.get(0))
                        .and_then(|val| val.try_extract::<u32>())
                    {
                        // Simple confidence: normalized by threshold
                        let confidence = if self.threshold_used > 0.0 {
                            ((event_count as f64) / self.threshold_used).min(1.0)
                        } else {
                            1.0
                        };
                        return Some(confidence);
                    }
                }
            }
        }
        None
    }

    /// Get statistics about the detection
    pub fn detection_stats(&self) -> String {
        format!(
            "Detected {} hot pixels from {} total pixels ({:.2}% hot) using {} (threshold: {:.1})",
            self.hot_pixels.len(),
            self.total_pixels,
            (self.hot_pixels.len() as f64 / self.total_pixels as f64) * 100.0,
            self.method.description(),
            self.threshold_used
        )
    }
}

/// Extract hot pixel coordinates from DataFrame using Polars operations
fn extract_hot_pixel_coordinates_polars(hot_pixel_df: &DataFrame) -> PolarsResult<Vec<(u16, u16)>> {
    let mut hot_pixels = Vec::with_capacity(hot_pixel_df.height());

    let x_series = hot_pixel_df.column(COL_X)?;
    let y_series = hot_pixel_df.column(COL_Y)?;

    let x_values = x_series.i64()?;
    let y_values = y_series.i64()?;

    for i in 0..hot_pixel_df.height() {
        if let (Some(x), Some(y)) = (x_values.get(i), y_values.get(i)) {
            hot_pixels.push((x as u16, y as u16));
        }
    }

    Ok(hot_pixels)
}

/// Apply hot pixel filtering using Polars operations
///
/// This is the main hot pixel filtering function that works entirely with Polars
/// operations for maximum performance.
///
/// # Arguments
///
/// * `df` - Input LazyFrame containing event data
/// * `filter` - Hot pixel filter configuration
///
/// # Returns
///
/// Filtered LazyFrame with hot pixel events removed
///
/// # Example
///
/// ```rust
/// use polars::prelude::*;
/// use evlib::ev_filtering::hot_pixel::*;
///
/// let events_df = events_to_dataframe(&events)?.lazy();
/// let filter = HotPixelFilter::percentile(99.5);
/// let filtered = apply_hot_pixel_filter(events_df, &filter)?;
/// ```
#[cfg_attr(feature = "tracing", instrument(skip(df), fields(method = ?filter.method)))]
pub fn apply_hot_pixel_filter(df: LazyFrame, filter: &HotPixelFilter) -> PolarsResult<LazyFrame> {
    let start_time = Instant::now();

    debug!("Applying hot pixel filter: {:?}", filter.method);

    // Calculate per-pixel statistics
    let pixel_stats_df = calculate_pixel_statistics_polars(df.clone(), filter)?;

    if pixel_stats_df.height() == 0 {
        debug!("No pixels meet minimum event count requirement");
        return Ok(df);
    }

    // Build hot pixel detection expression
    let hot_pixel_expr = match filter.to_hot_pixel_expr(&pixel_stats_df.clone().lazy())? {
        Some(expr) => expr,
        None => {
            debug!("No hot pixel filtering expression generated");
            return Ok(df);
        }
    };

    // Get hot pixel coordinates
    let hot_pixel_coords_df = pixel_stats_df
        .lazy()
        .filter(hot_pixel_expr)
        .select([col(COL_X), col(COL_Y)])
        .collect()?;

    if hot_pixel_coords_df.height() == 0 {
        debug!("No hot pixels detected");
        return Ok(df);
    }

    debug!("Detected {} hot pixels", hot_pixel_coords_df.height());

    // Create anti-join expression to filter out hot pixel events
    // Convert hot pixel coordinates to a set for filtering
    let hot_pixel_coords: Vec<(i64, i64)> = hot_pixel_coords_df
        .iter()
        .filter_map(|row| {
            if row.len() < 2 {
                warn!(
                    "Hot pixel row has {} columns, expected 2 - skipping",
                    row.len()
                );
                return None;
            }
            match (row.get(0), row.get(1)) {
                (Ok(x_val), Ok(y_val)) => {
                    match (x_val.try_extract::<i64>(), y_val.try_extract::<i64>()) {
                        (Ok(x), Ok(y)) => Some((x, y)),
                        _ => {
                            warn!("Failed to extract x,y coordinates as i64 - skipping pixel");
                            None
                        }
                    }
                }
                _ => {
                    warn!("Failed to get x,y values from hot pixel row - skipping");
                    None
                }
            }
        })
        .collect();

    // Use direct filtering instead of problematic join
    let filtered_df = if hot_pixel_coords.len() < 1000 {
        // For small sets, use direct expression filtering
        let mut filter_expr: Option<Expr> = None;
        for (x, y) in hot_pixel_coords {
            let pixel_expr = col(COL_X).eq(lit(x)).and(col(COL_Y).eq(lit(y)));
            filter_expr = match filter_expr {
                None => Some(pixel_expr.not()),
                Some(existing) => Some(existing.and(pixel_expr.not())),
            };
        }
        match filter_expr {
            Some(expr) => df.filter(expr),
            None => df, // No hot pixels to filter
        }
    } else {
        // For large sets, fall back to the join approach
        df.join(
            hot_pixel_coords_df.lazy(),
            [col(COL_X), col(COL_Y)],
            [col(COL_X), col(COL_Y)],
            JoinArgs::new(JoinType::Left).with_suffix(Some("_right".into())),
        )
        .filter(col(format!("{}_right", COL_X)).is_null())
    };

    let processing_time = start_time.elapsed().as_secs_f64();
    debug!("Hot pixel filtering completed in {:.3}s", processing_time);

    Ok(filtered_df)
}

/// Calculate per-pixel statistics using Polars group_by operations
///
/// This function efficiently computes pixel-level statistics using Polars'
/// optimized group operations instead of manual HashMap operations.
#[cfg_attr(feature = "tracing", instrument(skip(df)))]
fn calculate_pixel_statistics_polars(
    df: LazyFrame,
    filter: &HotPixelFilter,
) -> PolarsResult<DataFrame> {
    let mut agg_exprs = vec![
        len().alias("total_events"),
        col(COL_POLARITY).sum().alias("positive_events"),
    ];

    // Add temporal aggregations if temporal info is used
    if filter.use_temporal_info {
        agg_exprs.extend([
            col(COL_T).min().alias("first_event_time"),
            col(COL_T).max().alias("last_event_time"),
            col(COL_T).std(1).alias("temporal_std"),
        ]);
    }

    let mut pixel_stats_df = df
        .group_by([col(COL_X), col(COL_Y)])
        .agg(&agg_exprs)
        .with_columns([
            // Calculate derived statistics
            (col("total_events") - col("positive_events")).alias("negative_events"),
        ]);

    // Add temporal derived columns if temporal info is used
    if filter.use_temporal_info {
        pixel_stats_df = pixel_stats_df.with_columns([(col("last_event_time")
            - col("first_event_time"))
        .alias("temporal_spread")]);

        pixel_stats_df = pixel_stats_df.with_columns([
            // Calculate event rate (handle division by zero)
            when(col("temporal_spread").gt(0.0))
                .then(col("total_events").cast(DataType::Float64) / col("temporal_spread"))
                .otherwise(lit(0.0))
                .alias("event_rate"),
        ]);
    } else {
        // Add placeholder event_rate column for consistency
        pixel_stats_df = pixel_stats_df.with_columns([lit(0.0).alias("event_rate")]);
    }

    // Filter pixels based on minimum events requirement
    pixel_stats_df =
        pixel_stats_df.filter(col("total_events").gt_eq(lit(filter.min_events_for_detection)));

    pixel_stats_df
        .sort([COL_X, COL_Y], SortMultipleOptions::default())
        .collect()
}

/// Calculate detection threshold based on method and pixel statistics
fn calculate_detection_threshold(
    pixel_stats_df: &DataFrame,
    filter: &HotPixelFilter,
) -> PolarsResult<f64> {
    match filter.method {
        HotPixelDetectionMethod::Percentile => {
            if let Some(percentile) = filter.percentile_threshold {
                let threshold_df = pixel_stats_df
                    .clone()
                    .lazy()
                    .select([col("total_events")
                        .quantile(lit(percentile / 100.0), QuantileMethod::Linear)
                        .alias("threshold")])
                    .collect()?;
                Ok(threshold_df
                    .column("threshold")?
                    .get(0)?
                    .try_extract::<f64>()?)
            } else {
                Ok(0.0)
            }
        }
        HotPixelDetectionMethod::StandardDeviation => {
            if let Some(multiplier) = filter.std_dev_multiplier {
                let stats_df = pixel_stats_df
                    .clone()
                    .lazy()
                    .select([
                        col("total_events").mean().alias("mean_events"),
                        col("total_events").std(1).alias("std_events"),
                    ])
                    .collect()?;
                let mean = stats_df
                    .column("mean_events")?
                    .get(0)?
                    .try_extract::<f64>()?;
                let std_dev = stats_df
                    .column("std_events")?
                    .get(0)?
                    .try_extract::<f64>()?;
                Ok(mean + multiplier * std_dev)
            } else {
                Ok(0.0)
            }
        }
        HotPixelDetectionMethod::FixedThreshold => Ok(filter.fixed_threshold.unwrap_or(0) as f64),
        HotPixelDetectionMethod::LocalAdaptive => Ok(0.0), // Placeholder
        HotPixelDetectionMethod::FrequencyAnalysis => {
            if let Some(time_span) = filter.frequency_time_span {
                Ok(10.0 / time_span) // 10x higher than expected rate
            } else {
                Ok(0.0)
            }
        }
    }
}

/// Perform comprehensive pixel activity analysis using Polars operations
fn analyze_pixel_activity_polars(
    _df: LazyFrame,
    pixel_stats_df: &DataFrame,
    hot_pixels: &[(u16, u16)],
) -> PolarsResult<PixelActivityAnalysis> {
    let start_time = Instant::now();

    // Calculate pixel statistics using Polars aggregations
    let event_count_statistics = calculate_pixel_statistics_from_df(pixel_stats_df)?;

    // Analyze spatial distribution
    let spatial_distribution = analyze_spatial_distribution_simple(hot_pixels);

    // Create simplified temporal patterns
    let temporal_patterns = TemporalPatterns {
        avg_firing_rate: 0.0,
        regularity_score: 0.0,
        burst_patterns: Vec::new(),
        stability_score: 0.0,
    };

    // Skip clustering for simplicity in this version
    let clustering_results = None;

    let processing_time = start_time.elapsed().as_secs_f64();
    debug!(
        "Pixel activity analysis completed in {:.3}s",
        processing_time
    );

    Ok(PixelActivityAnalysis {
        total_active_pixels: pixel_stats_df.height(),
        event_count_statistics,
        spatial_distribution,
        temporal_patterns,
        clustering_results,
    })
}

/// Calculate pixel statistics from DataFrame using Polars aggregations
fn calculate_pixel_statistics_from_df(pixel_stats_df: &DataFrame) -> PolarsResult<PixelStatistics> {
    let stats_df = pixel_stats_df
        .clone()
        .lazy()
        .select([
            col("total_events").mean().alias("mean"),
            col("total_events").median().alias("median"),
            col("total_events").std(1).alias("std_dev"),
            col("total_events").min().alias("min"),
            col("total_events").max().alias("max"),
        ])
        .collect()?;

    // We'll create the percentiles DataFrame after basic stats

    if stats_df.height() == 0 {
        return Ok(PixelStatistics {
            mean: 0.0,
            median: 0.0,
            std_dev: 0.0,
            min: 0,
            max: 0,
            percentiles_df: None,
            skewness: 0.0,
            kurtosis: 0.0,
        });
    }

    let row = stats_df.get_row(0)?;

    // Create percentiles DataFrame using a simpler approach (skip for now to avoid compilation issues)
    let percentiles_df: Option<DataFrame> = None;

    Ok(PixelStatistics {
        mean: row.0[0].try_extract::<f64>()?,
        median: row.0[1].try_extract::<f64>()?,
        std_dev: row.0[2].try_extract::<f64>()?,
        min: row.0[3].try_extract::<u32>()?,
        max: row.0[4].try_extract::<u32>()?,
        percentiles_df,
        skewness: 0.0, // Would require additional computation
        kurtosis: 0.0, // Would require additional computation
    })
}

/// Simple spatial distribution analysis using Polars for density calculation
fn analyze_spatial_distribution_simple(hot_pixels: &[(u16, u16)]) -> SpatialDistribution {
    if hot_pixels.is_empty() {
        return SpatialDistribution {
            bounds: (0, 0, 0, 0),
            avg_distance: 0.0,
            clustering_coefficient: 0.0,
            density_df: None,
        };
    }

    let min_x = hot_pixels.iter().map(|p| p.0).min().unwrap_or(0);
    let max_x = hot_pixels.iter().map(|p| p.0).max().unwrap_or(0);
    let min_y = hot_pixels.iter().map(|p| p.1).min().unwrap_or(0);
    let max_y = hot_pixels.iter().map(|p| p.1).max().unwrap_or(0);

    // Calculate average distance between hot pixels
    let mut total_distance = 0.0;
    let mut pair_count = 0;

    for (i, &(x1, y1)) in hot_pixels.iter().enumerate() {
        for &(x2, y2) in hot_pixels.iter().skip(i + 1) {
            let dx = (x2 as f64) - (x1 as f64);
            let dy = (y2 as f64) - (y1 as f64);
            total_distance += (dx * dx + dy * dy).sqrt();
            pair_count += 1;
        }
    }

    let avg_distance = if pair_count > 0 {
        total_distance / pair_count as f64
    } else {
        0.0
    };

    // Create density DataFrame using Polars - simplified grid-based density
    let density_df = create_density_map_polars(hot_pixels, min_x, max_x, min_y, max_y);

    SpatialDistribution {
        bounds: (min_x, max_x, min_y, max_y),
        avg_distance,
        clustering_coefficient: 0.0, // Simplified
        density_df,
    }
}

/// Create density map using Polars DataFrame operations
fn create_density_map_polars(
    hot_pixels: &[(u16, u16)],
    min_x: u16,
    max_x: u16,
    min_y: u16,
    max_y: u16,
) -> Option<DataFrame> {
    if hot_pixels.is_empty() {
        return None;
    }

    // Create DataFrame from hot pixel coordinates
    let x_coords: Vec<i64> = hot_pixels.iter().map(|p| p.0 as i64).collect();
    let y_coords: Vec<i64> = hot_pixels.iter().map(|p| p.1 as i64).collect();

    let result = df!(
        COL_X => x_coords,
        COL_Y => y_coords
    )
    .map(|df| {
        // Add density calculation - for now just count occurrences per pixel
        df.lazy()
            .group_by([col(COL_X), col(COL_Y)])
            .agg([len().alias("density")])
            .with_columns([
                // Normalize density to 0-1 range based on grid size
                col("density").cast(DataType::Float64)
                    / lit((max_x - min_x + 1) as f64 * (max_y - min_y + 1) as f64),
            ])
            .sort([COL_X, COL_Y], SortMultipleOptions::default())
            .collect()
    });

    result.ok().and_then(|r| r.ok())
}

/// Detect hot pixels and return their locations - Polars version
pub fn detect_hot_pixels_polars(
    df: LazyFrame,
    percentile_threshold: f64,
) -> PolarsResult<Vec<(u16, u16)>> {
    let filter = HotPixelFilter::percentile(percentile_threshold);
    let detector = HotPixelDetector::detect_polars(df, &filter)?;
    Ok(detector.hot_pixels)
}
