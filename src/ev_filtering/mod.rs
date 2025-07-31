//! Event filtering module for evlib
//!
//! This module provides comprehensive event filtering functionality for event camera data.
//! It supports both streaming and batch processing with a focus on performance and memory efficiency.
//!
//! # Features
//!
//! - **Temporal filtering**: Time-based event selection with microsecond precision
//! - **Spatial filtering**: Region of interest (ROI) and coordinate-based filtering
//! - **Polarity filtering**: Positive/negative event separation
//! - **Hot pixel removal**: Statistical outlier detection and removal
//! - **Noise filtering**: Refractory period and distance-based denoising
//! - **Event downsampling**: Intelligent event reduction strategies
//! - **Pixel dropping**: Selective pixel masking and exclusion
//!
//! # Performance
//!
//! All filtering operations are optimized for:
//! - Vectorized processing using SIMD instructions
//! - Memory-efficient streaming for large datasets
//! - Parallel processing using Rayon
//! - Zero-copy operations where possible
//!
//! # Usage
//!
//! ```rust
//! use evlib::ev_filtering::{FilterConfig, filter_events, TemporalFilter, SpatialFilter};
//! use evlib::{Event, Events};
//!
//! // Create sample events
//! let events = vec![
//!     Event { t: 1.0, x: 100, y: 200, polarity: true },
//!     Event { t: 2.0, x: 150, y: 250, polarity: false },
//!     Event { t: 3.0, x: 200, y: 300, polarity: true },
//! ];
//!
//! // Configure filters
//! let config = FilterConfig::new()
//!     .with_temporal_filter(TemporalFilter::new(1.5, 2.5))
//!     .with_spatial_filter(SpatialFilter::roi(120, 180, 220, 280));
//!
//! // Apply filters
//! let filtered_events = filter_events(&events, &config)?;
//! ```

use crate::ev_core::{Event, Events};
use std::fmt;

#[cfg(feature = "polars")]
use polars::prelude::*;

// Column names for DataFrame consistency
#[cfg(feature = "polars")]
pub const COL_X: &str = "x";
#[cfg(feature = "polars")]
pub const COL_Y: &str = "y";
#[cfg(feature = "polars")]
pub const COL_T: &str = "t";
#[cfg(feature = "polars")]
pub const COL_POLARITY: &str = "polarity";

/// Convert DataFrame back to Events vector for legacy compatibility
#[cfg(feature = "polars")]
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
            .ok_or_else(|| FilterError::ProcessingError("X value missing".to_string()))?
            as u16;
        let y = y_values
            .get(i)
            .ok_or_else(|| FilterError::ProcessingError("Y value missing".to_string()))?
            as u16;
        let t = t_values
            .get(i)
            .ok_or_else(|| FilterError::ProcessingError("T value missing".to_string()))?;
        let polarity = p_values
            .get(i)
            .ok_or_else(|| FilterError::ProcessingError("Polarity value missing".to_string()))?
            > 0;

        events.push(Event { t, x, y, polarity });
    }

    Ok(events)
}

// Sub-modules
pub mod config;
pub mod denoise;
pub mod downsampling;
pub mod drop_pixel;
pub mod hot_pixel;
pub mod polarity;
pub mod python;
pub mod spatial;
pub mod temporal; // Now Polars-first implementation
pub mod utils;

// Re-export core types and functions for convenience
pub use config::{FilterConfig, FilterError, FilterResult};
pub use denoise::{apply_refractory_period, filter_noise, DenoiseFilter, RefractoryFilter};
pub use downsampling::{
    downsample_events, downsample_uniform, DownsamplingFilter, DownsamplingStrategy,
};
pub use drop_pixel::{drop_pixels, DropPixelFilter, PixelMask};
pub use hot_pixel::{detect_hot_pixels, filter_hot_pixels, HotPixelDetector, HotPixelFilter};
pub use polarity::{filter_by_polarity, separate_polarities, PolarityFilter};
pub use spatial::{
    create_pixel_mask, filter_by_circle, filter_by_circular_roi, filter_by_coordinates,
    filter_by_multiple_rois, filter_by_pixel_mask, filter_by_polygon, filter_by_roi,
    find_spatial_clusters, split_by_spatial_grid, CircularROI, MultipleROIs, Point, PolygonROI,
    ROICombination, RegionOfInterest, SpatialFilter,
};
pub use temporal::{filter_by_time, TemporalFilter};

/// Polars-first filtering function that applies filters entirely using LazyFrame operations
///
/// This function processes events through a pipeline of filters based on the provided
/// configuration using Polars LazyFrame operations throughout. It provides significant
/// performance improvements over the Vec<Event> approach.
///
/// # Arguments
///
/// * `events` - Input events to filter
/// * `config` - Filter configuration specifying which filters to apply
///
/// # Returns
///
/// * `FilterResult<Events>` - Filtered events or error
///
/// # Example
///
/// ```rust
/// use evlib::ev_filtering::{FilterConfig, filter_events_polars, TemporalFilter};
/// use evlib::{Event, Events};
///
/// let events = vec![
///     Event { t: 1.0, x: 100, y: 200, polarity: true },
///     Event { t: 2.0, x: 150, y: 250, polarity: false },
/// ];
///
/// let config = FilterConfig::new()
///     .with_temporal_filter(TemporalFilter::new(0.5, 1.5));
///
/// let filtered = filter_events_polars(&events, &config)?;
/// ```
#[cfg(feature = "polars")]
pub fn filter_events_polars(events: &Events, config: &FilterConfig) -> FilterResult<Events> {
    // Convert input Vec<Event> to LazyFrame once at the beginning
    let df = crate::ev_core::events_to_dataframe(events)
        .map_err(|e| FilterError::ProcessingError(format!("DataFrame conversion error: {}", e)))?
        .lazy();

    // Apply filters in order of efficiency (fastest first) using Polars-first implementations
    let mut filtered_df = df;

    // 1. Temporal filtering - very fast with sorted data
    if let Some(temporal_filter) = &config.temporal_filter {
        filtered_df = temporal::apply_temporal_filter(filtered_df, temporal_filter)
            .map_err(|e| FilterError::ProcessingError(format!("Temporal filter error: {}", e)))?;
    }

    // 2. Spatial filtering - fast coordinate bounds checking
    if let Some(spatial_filter) = &config.spatial_filter {
        filtered_df = spatial::apply_spatial_filter(filtered_df, spatial_filter)
            .map_err(|e| FilterError::ProcessingError(format!("Spatial filter error: {}", e)))?;
    }

    // 3. Polarity filtering - simple integer comparison
    if let Some(polarity_filter) = &config.polarity_filter {
        filtered_df = polarity::apply_polarity_filter(filtered_df, polarity_filter)
            .map_err(|e| FilterError::ProcessingError(format!("Polarity filter error: {}", e)))?;
    }

    // 4. Drop pixel filtering - requires pixel lookup
    if let Some(drop_pixel_filter) = &config.drop_pixel_filter {
        filtered_df = drop_pixel::apply_drop_pixel_filter(filtered_df, drop_pixel_filter)
            .map_err(|e| FilterError::ProcessingError(format!("Drop pixel filter error: {}", e)))?;
    }

    // 5. Hot pixel filtering - requires statistical analysis
    if let Some(hot_pixel_filter) = &config.hot_pixel_filter {
        filtered_df = hot_pixel::apply_hot_pixel_filter(filtered_df, hot_pixel_filter)
            .map_err(|e| FilterError::ProcessingError(format!("Hot pixel filter error: {}", e)))?;
    }

    // 6. Denoising - most expensive, requires sorting and grouping
    if let Some(denoise_filter) = &config.denoise_filter {
        filtered_df = denoise::apply_denoise_filter_polars(filtered_df, denoise_filter)
            .map_err(|e| FilterError::ProcessingError(format!("Denoise filter error: {}", e)))?;
    }

    // 7. Downsampling - final step to reduce event count
    if let Some(downsampling_filter) = &config.downsampling_filter {
        filtered_df =
            downsampling::apply_downsampling_filter_polars(filtered_df, downsampling_filter)
                .map_err(|e| {
                    FilterError::ProcessingError(format!("Downsampling filter error: {}", e))
                })?;
    }

    // Only convert back to Vec<Event> at the very end for legacy compatibility
    let result_df = filtered_df
        .collect()
        .map_err(|e| FilterError::ProcessingError(format!("LazyFrame collection error: {}", e)))?;

    dataframe_to_events(&result_df)
}

/// Main filtering function that applies a comprehensive filter configuration
///
/// This function processes events through a pipeline of filters based on the provided
/// configuration. It supports both streaming and batch processing modes.
///
/// # Arguments
///
/// * `events` - Input events to filter
/// * `config` - Filter configuration specifying which filters to apply
///
/// # Returns
///
/// * `FilterResult<Events>` - Filtered events or error
///
/// # Example
///
/// ```rust
/// use evlib::ev_filtering::{FilterConfig, filter_events, TemporalFilter};
/// use evlib::{Event, Events};
///
/// let events = vec![
///     Event { t: 1.0, x: 100, y: 200, polarity: true },
///     Event { t: 2.0, x: 150, y: 250, polarity: false },
/// ];
///
/// let config = FilterConfig::new()
///     .with_temporal_filter(TemporalFilter::new(0.5, 1.5));
///
/// let filtered = filter_events(&events, &config)?;
/// ```
pub fn filter_events(events: &Events, config: &FilterConfig) -> FilterResult<Events> {
    // Use Polars-first implementation when available for better performance
    #[cfg(feature = "polars")]
    {
        filter_events_polars(events, config)
    }

    // Fallback to Vec<Event> implementation when Polars is not available
    #[cfg(not(feature = "polars"))]
    {
        let mut filtered_events = events.clone();

        // Apply filters in order of efficiency (fastest first)
        // 1. Temporal filtering - very fast with sorted data
        if let Some(temporal_filter) = &config.temporal_filter {
            filtered_events = temporal_filter.apply(&filtered_events)?;
        }

        // 2. Spatial filtering - fast coordinate bounds checking
        if let Some(spatial_filter) = &config.spatial_filter {
            filtered_events = spatial_filter.apply(&filtered_events)?;
        }

        // 3. Polarity filtering - simple integer comparison
        if let Some(polarity_filter) = &config.polarity_filter {
            filtered_events = polarity::apply_polarity_filter(&filtered_events, polarity_filter)?;
        }

        // 4. Drop pixel filtering - requires pixel lookup
        if let Some(drop_pixel_filter) = &config.drop_pixel_filter {
            filtered_events =
                drop_pixel::apply_drop_pixel_filter(&filtered_events, drop_pixel_filter)?;
        }

        // 5. Hot pixel filtering - requires statistical analysis
        if let Some(hot_pixel_filter) = &config.hot_pixel_filter {
            filtered_events =
                hot_pixel::apply_hot_pixel_filter(&filtered_events, hot_pixel_filter)?;
        }

        // 6. Denoising - most expensive, requires sorting and grouping
        if let Some(denoise_filter) = &config.denoise_filter {
            filtered_events = denoise::apply_denoise_filter(&filtered_events, denoise_filter)?;
        }

        // 7. Downsampling - final step to reduce event count
        if let Some(downsampling_filter) = &config.downsampling_filter {
            filtered_events =
                downsampling::apply_downsampling_filter(&filtered_events, downsampling_filter)?;
        }

        Ok(filtered_events)
    }
}

/// Streaming version of filter_events for large datasets
///
/// This function processes events in chunks to maintain memory efficiency
/// while applying the same filtering pipeline.
///
/// # Arguments
///
/// * `events` - Input events to filter
/// * `config` - Filter configuration
/// * `chunk_size` - Number of events to process at once
///
/// # Returns
///
/// * `FilterResult<Events>` - Filtered events or error
pub fn filter_events_streaming(
    events: &Events,
    config: &FilterConfig,
    chunk_size: usize,
) -> FilterResult<Events> {
    if events.len() <= chunk_size {
        // Use regular filtering for small datasets
        return filter_events(events, config);
    }

    let mut filtered_events = Events::with_capacity(events.len() / 2); // Estimate 50% reduction

    // Process in chunks
    for chunk in events.chunks(chunk_size) {
        let chunk_vec = chunk.to_vec();
        let chunk_filtered = filter_events(&chunk_vec, config)?;
        filtered_events.extend(chunk_filtered);
    }

    // Sort the final result if needed
    if config.ensure_temporal_order {
        utils::sort_events_by_time(&mut filtered_events);
    }

    Ok(filtered_events)
}

/// Apply a single filter type to events
///
/// This is a convenience function for applying individual filters without
/// creating a full FilterConfig.
pub fn apply_single_filter<F>(events: &Events, filter: F) -> FilterResult<Events>
where
    F: SingleFilter,
{
    filter.apply(events)
}

/// Trait for individual filter implementations
pub trait SingleFilter {
    /// Apply this filter to a set of events
    fn apply(&self, events: &Events) -> FilterResult<Events>;

    /// Get a description of this filter
    fn description(&self) -> String;

    /// Check if this filter is enabled/configured
    fn is_enabled(&self) -> bool {
        true
    }
}

/// Statistics about filtering operations
#[derive(Debug, Clone)]
pub struct FilterStats {
    /// Original number of events
    pub input_count: usize,
    /// Final number of events after filtering
    pub output_count: usize,
    /// Number of events removed
    pub removed_count: usize,
    /// Fraction of events removed (0.0 to 1.0)
    pub removal_fraction: f64,
    /// Processing time in seconds
    pub processing_time: f64,
    /// Events processed per second
    pub throughput: f64,
}

impl FilterStats {
    /// Create new filter statistics
    pub fn new(input_count: usize, output_count: usize, processing_time: f64) -> Self {
        let removed_count = input_count.saturating_sub(output_count);
        let removal_fraction = if input_count > 0 {
            removed_count as f64 / input_count as f64
        } else {
            0.0
        };
        let throughput = if processing_time > 0.0 {
            input_count as f64 / processing_time
        } else {
            0.0
        };

        Self {
            input_count,
            output_count,
            removed_count,
            removal_fraction,
            processing_time,
            throughput,
        }
    }
}

impl fmt::Display for FilterStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Filter Stats: {} -> {} events ({:.1}% removed) in {:.3}s ({:.0} events/s)",
            self.input_count,
            self.output_count,
            self.removal_fraction * 100.0,
            self.processing_time,
            self.throughput
        )
    }
}

/// Apply filtering with statistics collection
pub fn filter_events_with_stats(
    events: &Events,
    config: &FilterConfig,
) -> FilterResult<(Events, FilterStats)> {
    let start_time = std::time::Instant::now();
    let input_count = events.len();

    let filtered_events = filter_events(events, config)?;

    let processing_time = start_time.elapsed().as_secs_f64();
    let output_count = filtered_events.len();
    let stats = FilterStats::new(input_count, output_count, processing_time);

    Ok((filtered_events, stats))
}

/// Convenience function for common filtering operations
pub mod presets {
    use super::*;

    /// Create a configuration for basic noise removal
    pub fn noise_removal() -> FilterConfig {
        FilterConfig::new()
            .with_hot_pixel_filter(HotPixelFilter::percentile(99.5))
            .with_denoise_filter(DenoiseFilter::refractory(1000.0)) // 1ms refractory period
    }

    /// Create a configuration for aggressive noise removal
    pub fn aggressive_noise_removal() -> FilterConfig {
        FilterConfig::new()
            .with_hot_pixel_filter(HotPixelFilter::percentile(99.0))
            .with_denoise_filter(DenoiseFilter::refractory(500.0)) // 0.5ms refractory period
    }

    /// Create a configuration for minimal processing (quality preservation)
    pub fn minimal_processing() -> FilterConfig {
        FilterConfig::new().with_hot_pixel_filter(HotPixelFilter::percentile(99.9))
    }

    /// Create a configuration for high throughput processing (aggressive filtering)
    pub fn high_throughput() -> FilterConfig {
        FilterConfig::new()
            .with_hot_pixel_filter(HotPixelFilter::percentile(98.0))
            .with_denoise_filter(DenoiseFilter::refractory(2000.0)) // 2ms refractory period
            .with_downsampling_filter(DownsamplingFilter::uniform(0.5)) // Keep 50% of events
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_events() -> Events {
        vec![
            Event {
                t: 1.0,
                x: 100,
                y: 200,
                polarity: true,
            },
            Event {
                t: 2.0,
                x: 150,
                y: 250,
                polarity: false,
            },
            Event {
                t: 3.0,
                x: 200,
                y: 300,
                polarity: true,
            },
            Event {
                t: 4.0,
                x: 250,
                y: 350,
                polarity: false,
            },
            Event {
                t: 5.0,
                x: 300,
                y: 400,
                polarity: true,
            },
        ]
    }

    #[test]
    fn test_empty_config() {
        let events = create_test_events();
        let config = FilterConfig::new();
        let filtered = filter_events(&events, &config).unwrap();
        assert_eq!(filtered.len(), events.len());
    }

    #[test]
    fn test_temporal_filter() {
        let events = create_test_events();
        let config = FilterConfig::new().with_temporal_filter(TemporalFilter::new(2.0, 4.0));
        let filtered = filter_events(&events, &config).unwrap();
        assert_eq!(filtered.len(), 3); // Events at t=2, 3, 4
    }

    #[test]
    fn test_spatial_filter() {
        let events = create_test_events();
        let config =
            FilterConfig::new().with_spatial_filter(SpatialFilter::roi(120, 220, 220, 320));
        let filtered = filter_events(&events, &config).unwrap();
        assert_eq!(filtered.len(), 2); // Events at (150,250) and (200,300)
    }

    #[test]
    fn test_polarity_filter() {
        let events = create_test_events();
        let config = FilterConfig::new().with_polarity_filter(PolarityFilter::positive_only());
        let filtered = filter_events(&events, &config).unwrap();
        assert_eq!(filtered.len(), 3); // Only positive events
    }

    #[test]
    fn test_combined_filters() {
        let events = create_test_events();
        let config = FilterConfig::new()
            .with_temporal_filter(TemporalFilter::new(2.0, 4.0))
            .with_polarity_filter(PolarityFilter::positive_only());
        let filtered = filter_events(&events, &config).unwrap();
        assert_eq!(filtered.len(), 1); // Only event at t=3 (positive and in time range)
    }

    #[cfg(feature = "polars")]
    #[test]
    fn test_polars_first_filtering() {
        let events = create_test_events();
        let config = FilterConfig::new()
            .with_temporal_filter(TemporalFilter::new(2.0, 4.0))
            .with_polarity_filter(PolarityFilter::positive_only());

        // Test the new Polars-first implementation directly
        let filtered_polars = filter_events_polars(&events, &config).unwrap();

        // Should produce the same results as the regular implementation
        assert_eq!(filtered_polars.len(), 1); // Only event at t=3 (positive and in time range)

        // Verify the event content is correct
        assert_eq!(filtered_polars[0].t, 3.0);
        assert!(filtered_polars[0].polarity);
        assert_eq!(filtered_polars[0].x, 200);
        assert_eq!(filtered_polars[0].y, 300);
    }

    #[test]
    fn test_filter_stats() {
        let events = create_test_events();
        let config = FilterConfig::new().with_temporal_filter(TemporalFilter::new(2.0, 4.0));
        let (filtered, stats) = filter_events_with_stats(&events, &config).unwrap();

        assert_eq!(stats.input_count, 5);
        assert_eq!(stats.output_count, 3);
        assert_eq!(stats.removed_count, 2);
        assert!((stats.removal_fraction - 0.4).abs() < 0.001);
    }

    #[test]
    fn test_streaming() {
        let mut events = Vec::new();
        for i in 0..1000 {
            events.push(Event {
                t: i as f64,
                x: (i % 640) as u16,
                y: (i % 480) as u16,
                polarity: i % 2 == 0,
            });
        }

        let config = FilterConfig::new().with_temporal_filter(TemporalFilter::new(100.0, 200.0));

        let filtered = filter_events_streaming(&events, &config, 100).unwrap();
        assert_eq!(filtered.len(), 101); // Events 100-200 inclusive
    }
}
