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

// Removed: use crate::{Event, Events}; - legacy types no longer exist
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
pub use denoise::{DenoiseFilter, RefractoryFilter};
pub use downsampling::{DownsamplingFilter, DownsamplingStrategy};
pub use drop_pixel::{DropPixelFilter, PixelMask};
pub use hot_pixel::{HotPixelDetector, HotPixelFilter};
pub use polarity::PolarityFilter;
pub use spatial::{
    CircularROI, MultipleROIs, Point, PolygonROI, ROICombination, RegionOfInterest, SpatialFilter,
};
pub use temporal::TemporalFilter;

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
/// Filter events using DataFrame - high-performance DataFrame-native approach (RECOMMENDED)
///
/// This is the recommended high-performance filtering function that works entirely with
/// Polars LazyFrames, avoiding the overhead of converting to/from Vec<Event>.
/// Use this when you already have your data in DataFrame format or when you need maximum
/// performance.
///
/// # Arguments
///
/// * `df` - Input LazyFrame containing event data
/// * `config` - Filter configuration specifying which filters to apply
///
/// # Returns
///
/// Filtered LazyFrame
///
/// # Example
///
/// ```rust
/// use evlib::ev_filtering::{FilterConfig, temporal::TemporalFilter, spatial::SpatialFilter};
/// use evlib::ev_core::events_to_dataframe;
///
/// // Convert events to DataFrame once
/// let df = events_to_dataframe(&events)?.lazy();
///
/// let config = FilterConfig::new()
///     .with_temporal_filter(TemporalFilter::time_window(1.0, 5.0))
///     .with_spatial_filter(SpatialFilter::roi(100, 200, 150, 250));
///
/// let filtered_df = filter_events_dataframe(df, &config)?;
/// // Use filtered_df directly or collect() to DataFrame if needed
/// ```
#[cfg(feature = "polars")]
pub fn filter_events_dataframe(df: LazyFrame, config: &FilterConfig) -> PolarsResult<LazyFrame> {
    let mut filtered_df = df;

    // Apply filters in order of efficiency (fastest first) using DataFrame-native methods

    // 1. Temporal filtering - very fast with sorted data
    if let Some(temporal_filter) = &config.temporal_filter {
        filtered_df = temporal_filter.apply_to_dataframe(filtered_df)?;
    }

    // 2. Spatial filtering - fast coordinate bounds checking
    if let Some(spatial_filter) = &config.spatial_filter {
        filtered_df = spatial_filter.apply_to_dataframe(filtered_df)?;
    }

    // 3. Polarity filtering - simple integer comparison
    if let Some(polarity_filter) = &config.polarity_filter {
        filtered_df = polarity_filter.apply_to_dataframe(filtered_df)?;
    }

    // Additional filters can be chained here as they get DataFrame-native support
    // TODO: Add denoise, hot_pixel, drop_pixel, downsampling filters as they are updated

    Ok(filtered_df)
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
}
