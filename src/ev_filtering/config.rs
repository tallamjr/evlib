//! Filter configuration and error handling
//!
//! This module defines the configuration structures for event filtering operations
//! and provides comprehensive error handling for filtering operations.

use std::error::Error;
use std::fmt;

#[cfg(feature = "tracing")]
use tracing::warn;

#[cfg(not(feature = "tracing"))]
macro_rules! warn {
    ($($args:tt)*) => {
        eprintln!("[WARN] {}", format!($($args)*))
    };
}

use crate::ev_filtering::{
    DenoiseFilter, DownsamplingFilter, DropPixelFilter, HotPixelFilter, PolarityFilter,
    SpatialFilter, TemporalFilter,
};

/// Result type for filtering operations
pub type FilterResult<T> = Result<T, FilterError>;

/// Comprehensive error types for filtering operations
#[derive(Debug, Clone)]
pub enum FilterError {
    /// Invalid configuration parameters
    InvalidConfig(String),
    /// Invalid input data
    InvalidInput(String),
    /// Insufficient memory for operation
    OutOfMemory(String),
    /// Processing timeout
    Timeout(String),
    /// Internal processing error
    ProcessingError(String),
    /// I/O related error
    IoError(String),
    /// Mathematical computation error (e.g., division by zero, overflow)
    MathError(String),
    /// Polars DataFrame operation error
    #[cfg(feature = "polars")]
    PolarsError(String),
    /// Parallel processing error
    ParallelError(String),
}

impl fmt::Display for FilterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FilterError::InvalidConfig(msg) => write!(f, "Invalid filter configuration: {}", msg),
            FilterError::InvalidInput(msg) => write!(f, "Invalid input data: {}", msg),
            FilterError::OutOfMemory(msg) => write!(f, "Out of memory: {}", msg),
            FilterError::Timeout(msg) => write!(f, "Operation timed out: {}", msg),
            FilterError::ProcessingError(msg) => write!(f, "Processing error: {}", msg),
            FilterError::IoError(msg) => write!(f, "I/O error: {}", msg),
            FilterError::MathError(msg) => write!(f, "Mathematical error: {}", msg),
            #[cfg(feature = "polars")]
            FilterError::PolarsError(msg) => write!(f, "Polars error: {}", msg),
            FilterError::ParallelError(msg) => write!(f, "Parallel processing error: {}", msg),
        }
    }
}

impl Error for FilterError {}

/// Convert from standard I/O errors
impl From<std::io::Error> for FilterError {
    fn from(error: std::io::Error) -> Self {
        FilterError::IoError(error.to_string())
    }
}

/// Convert from Polars errors if feature is enabled
#[cfg(feature = "polars")]
impl From<polars::error::PolarsError> for FilterError {
    fn from(error: polars::error::PolarsError) -> Self {
        FilterError::PolarsError(error.to_string())
    }
}

/// Main configuration structure for event filtering
///
/// This structure allows you to configure multiple types of filters that will be
/// applied in sequence. Filters are applied in order of computational efficiency
/// to maximize performance.
///
/// # Example
///
/// ```rust
/// use evlib::ev_filtering::{FilterConfig, TemporalFilter, SpatialFilter, PolarityFilter};
///
/// let config = FilterConfig::new()
///     .with_temporal_filter(TemporalFilter::new(0.1, 0.5))
///     .with_spatial_filter(SpatialFilter::roi(100, 500, 100, 400))
///     .with_polarity_filter(PolarityFilter::positive_only())
///     .with_ensure_temporal_order(true);
/// ```
#[derive(Debug, Clone, Default)]
pub struct FilterConfig {
    /// Temporal filtering configuration
    pub temporal_filter: Option<TemporalFilter>,

    /// Spatial filtering configuration
    pub spatial_filter: Option<SpatialFilter>,

    /// Polarity filtering configuration
    pub polarity_filter: Option<PolarityFilter>,

    /// Hot pixel removal configuration
    pub hot_pixel_filter: Option<HotPixelFilter>,

    /// Noise removal configuration
    pub denoise_filter: Option<DenoiseFilter>,

    /// Pixel dropping configuration
    pub drop_pixel_filter: Option<DropPixelFilter>,

    /// Event downsampling configuration
    pub downsampling_filter: Option<DownsamplingFilter>,

    /// Whether to ensure events remain sorted by timestamp after filtering
    pub ensure_temporal_order: bool,

    /// Maximum processing time in seconds (0.0 for no limit)
    pub timeout_seconds: f64,

    /// Whether to use parallel processing when available
    pub enable_parallel: bool,

    /// Chunk size for streaming processing (0 for automatic)
    pub chunk_size: usize,

    /// Whether to collect detailed statistics during processing
    pub collect_stats: bool,

    /// Whether to validate input data before processing
    pub validate_input: bool,

    /// Memory limit in bytes (0 for no limit)
    pub memory_limit: usize,
}

impl FilterConfig {
    /// Create a new, empty filter configuration
    pub fn new() -> Self {
        Self {
            temporal_filter: None,
            spatial_filter: None,
            polarity_filter: None,
            hot_pixel_filter: None,
            denoise_filter: None,
            drop_pixel_filter: None,
            downsampling_filter: None,
            ensure_temporal_order: false,
            timeout_seconds: 0.0,
            enable_parallel: true,
            chunk_size: 0,
            collect_stats: false,
            validate_input: true,
            memory_limit: 0,
        }
    }

    // =================
    // Convenience builders for common filter configurations
    // =================

    /// Add temporal filtering with time range (Python API compatible)
    ///
    /// # Arguments
    ///
    /// * `t_start` - Start time in seconds (None for no lower bound)
    /// * `t_end` - End time in seconds (None for no upper bound)
    ///
    /// # Example
    ///
    /// ```rust
    /// let config = FilterConfig::new()
    ///     .with_time_range(Some(0.1), Some(0.5));
    /// ```
    pub fn with_time_range(mut self, t_start: Option<f64>, t_end: Option<f64>) -> Self {
        let filter = match (t_start, t_end) {
            (Some(start), Some(end)) => TemporalFilter::time_window(start, end),
            (Some(start), None) => TemporalFilter::from_time(start),
            (None, Some(end)) => TemporalFilter::until_time(end),
            (None, None) => return self, // No filtering needed
        };
        self.temporal_filter = Some(filter);
        self
    }

    /// Add spatial filtering with region of interest (Python API compatible)
    ///
    /// # Arguments
    ///
    /// * `x_min` - Minimum x coordinate (inclusive)
    /// * `x_max` - Maximum x coordinate (inclusive)
    /// * `y_min` - Minimum y coordinate (inclusive)
    /// * `y_max` - Maximum y coordinate (inclusive)
    ///
    /// # Example
    ///
    /// ```rust
    /// let config = FilterConfig::new()
    ///     .with_roi(100, 500, 100, 400);
    /// ```
    pub fn with_roi(mut self, x_min: u16, x_max: u16, y_min: u16, y_max: u16) -> Self {
        self.spatial_filter = Some(SpatialFilter::roi(x_min, x_max, y_min, y_max));
        self
    }

    /// Add polarity filtering (Python API compatible)
    ///
    /// # Arguments
    ///
    /// * `polarities` - Vector of polarity values to keep (supports 0/1 and -1/1 encodings)
    ///
    /// # Example
    ///
    /// ```rust
    /// // Keep only positive events
    /// let config = FilterConfig::new()
    ///     .with_polarity(vec![1]);
    ///
    /// // Keep both positive and negative (for -1/1 encoding)
    /// let config = FilterConfig::new()
    ///     .with_polarity(vec![-1, 1]);
    /// ```
    pub fn with_polarity(mut self, polarities: Vec<i8>) -> Self {
        self.polarity_filter = Some(PolarityFilter::from_values(polarities));
        self
    }

    /// Add hot pixel removal (Python API compatible)
    ///
    /// # Arguments
    ///
    /// * `threshold_percentile` - Percentile threshold for hot pixel detection (e.g., 99.9)
    ///
    /// # Example
    ///
    /// ```rust
    /// let config = FilterConfig::new()
    ///     .with_hot_pixel_removal(99.9);
    /// ```
    pub fn with_hot_pixel_removal(mut self, threshold_percentile: f64) -> Self {
        self.hot_pixel_filter = Some(HotPixelFilter::percentile(threshold_percentile));
        self
    }

    /// Add refractory period noise filtering (Python API compatible)
    ///
    /// # Arguments
    ///
    /// * `refractory_period_us` - Refractory period in microseconds
    ///
    /// # Example
    ///
    /// ```rust
    /// let config = FilterConfig::new()
    ///     .with_refractory_period(1000.0); // 1ms refractory period
    /// ```
    pub fn with_refractory_period(mut self, refractory_period_us: f64) -> Self {
        self.denoise_filter = Some(DenoiseFilter::refractory(refractory_period_us));
        self
    }

    /// Add uniform event downsampling
    ///
    /// # Arguments
    ///
    /// * `fraction` - Fraction of events to keep (0.0 to 1.0)
    ///
    /// # Example
    ///
    /// ```rust
    /// let config = FilterConfig::new()
    ///     .with_downsampling(0.5); // Keep 50% of events
    /// ```
    pub fn with_downsampling(mut self, fraction: f64) -> Self {
        self.downsampling_filter = Some(DownsamplingFilter::uniform(fraction));
        self
    }

    /// Add pixel exclusion filter
    ///
    /// # Arguments
    ///
    /// * `excluded_pixels` - Vector of (x, y) coordinates to exclude
    ///
    /// # Example
    ///
    /// ```rust
    /// let config = FilterConfig::new()
    ///     .with_excluded_pixels(vec![(100, 200), (150, 250)]);
    /// ```
    pub fn with_excluded_pixels(mut self, excluded_pixels: Vec<(u16, u16)>) -> Self {
        use std::collections::HashSet;
        let pixel_set: HashSet<(u16, u16)> = excluded_pixels.into_iter().collect();
        self.drop_pixel_filter = Some(DropPixelFilter::exclude(pixel_set));
        self
    }

    /// Add temporal filtering
    pub fn with_temporal_filter(mut self, filter: TemporalFilter) -> Self {
        self.temporal_filter = Some(filter);
        self
    }

    /// Add spatial filtering
    pub fn with_spatial_filter(mut self, filter: SpatialFilter) -> Self {
        self.spatial_filter = Some(filter);
        self
    }

    /// Add polarity filtering
    pub fn with_polarity_filter(mut self, filter: PolarityFilter) -> Self {
        self.polarity_filter = Some(filter);
        self
    }

    /// Add hot pixel removal
    pub fn with_hot_pixel_filter(mut self, filter: HotPixelFilter) -> Self {
        self.hot_pixel_filter = Some(filter);
        self
    }

    /// Add noise filtering
    pub fn with_denoise_filter(mut self, filter: DenoiseFilter) -> Self {
        self.denoise_filter = Some(filter);
        self
    }

    /// Add pixel dropping
    pub fn with_drop_pixel_filter(mut self, filter: DropPixelFilter) -> Self {
        self.drop_pixel_filter = Some(filter);
        self
    }

    /// Add event downsampling
    pub fn with_downsampling_filter(mut self, filter: DownsamplingFilter) -> Self {
        self.downsampling_filter = Some(filter);
        self
    }

    /// Set whether to ensure temporal ordering
    pub fn with_ensure_temporal_order(mut self, ensure: bool) -> Self {
        self.ensure_temporal_order = ensure;
        self
    }

    /// Set processing timeout in seconds
    pub fn with_timeout(mut self, seconds: f64) -> Self {
        self.timeout_seconds = seconds;
        self
    }

    /// Enable or disable parallel processing
    pub fn with_parallel_processing(mut self, enable: bool) -> Self {
        self.enable_parallel = enable;
        self
    }

    /// Set chunk size for streaming processing
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    /// Enable or disable statistics collection
    pub fn with_stats_collection(mut self, collect: bool) -> Self {
        self.collect_stats = collect;
        self
    }

    /// Enable or disable input validation
    pub fn with_input_validation(mut self, validate: bool) -> Self {
        self.validate_input = validate;
        self
    }

    /// Set memory limit in bytes
    pub fn with_memory_limit(mut self, limit: usize) -> Self {
        self.memory_limit = limit;
        self
    }

    // =================
    // Preset configurations for common use cases
    // =================

    /// Create a configuration optimised for noise removal
    ///
    /// Applies:
    /// - Hot pixel removal (99.5th percentile)
    /// - Refractory period filtering (1ms)
    /// - Input validation enabled
    ///
    /// # Example
    ///
    /// ```rust
    /// let config = FilterConfig::noise_removal();
    /// ```
    pub fn noise_removal() -> Self {
        Self::new()
            .with_hot_pixel_removal(99.5)
            .with_refractory_period(1000.0)
            .with_input_validation(true)
            .with_stats_collection(true)
    }

    /// Create a configuration for aggressive noise removal
    ///
    /// Applies:
    /// - Aggressive hot pixel removal (99.0th percentile)
    /// - Short refractory period (0.5ms)
    /// - Statistics collection enabled
    ///
    /// # Example
    ///
    /// ```rust
    /// let config = FilterConfig::aggressive_noise_removal();
    /// ```
    pub fn aggressive_noise_removal() -> Self {
        Self::new()
            .with_hot_pixel_removal(99.0)
            .with_refractory_period(500.0)
            .with_input_validation(true)
            .with_stats_collection(true)
    }

    /// Create a configuration for high-throughput processing
    ///
    /// Applies:
    /// - Moderate hot pixel removal (98.0th percentile)
    /// - Longer refractory period (2ms)
    /// - 50% downsampling
    /// - Parallel processing enabled
    /// - Larger chunk sizes
    ///
    /// # Example
    ///
    /// ```rust
    /// let config = FilterConfig::high_throughput();
    /// ```
    pub fn high_throughput() -> Self {
        Self::new()
            .with_hot_pixel_removal(98.0)
            .with_refractory_period(2000.0)
            .with_downsampling(0.5)
            .with_parallel_processing(true)
            .with_chunk_size(1_000_000)
            .with_input_validation(false) // Skip validation for speed
    }

    /// Create a configuration for quality preservation (minimal filtering)
    ///
    /// Applies:
    /// - Conservative hot pixel removal (99.9th percentile)
    /// - No refractory filtering
    /// - Full validation enabled
    /// - Statistics collection enabled
    ///
    /// # Example
    ///
    /// ```rust
    /// let config = FilterConfig::quality_preservation();
    /// ```
    pub fn quality_preservation() -> Self {
        Self::new()
            .with_hot_pixel_removal(99.9)
            .with_input_validation(true)
            .with_stats_collection(true)
            .with_ensure_temporal_order(true)
    }

    /// Create a configuration for debugging purposes
    ///
    /// Applies:
    /// - Small chunk sizes for easier debugging
    /// - Single-threaded processing
    /// - Full validation and statistics
    /// - Conservative filtering
    ///
    /// # Example
    ///
    /// ```rust
    /// let config = FilterConfig::debug();
    /// ```
    pub fn debug() -> Self {
        Self::new()
            .with_hot_pixel_removal(99.5)
            .with_refractory_period(1000.0)
            .with_parallel_processing(false)
            .with_chunk_size(10_000)
            .with_input_validation(true)
            .with_stats_collection(true)
            .with_ensure_temporal_order(true)
    }

    /// Check if any filters are configured
    pub fn has_filters(&self) -> bool {
        self.temporal_filter.is_some()
            || self.spatial_filter.is_some()
            || self.polarity_filter.is_some()
            || self.hot_pixel_filter.is_some()
            || self.denoise_filter.is_some()
            || self.drop_pixel_filter.is_some()
            || self.downsampling_filter.is_some()
    }

    /// Count the number of active filters
    pub fn filter_count(&self) -> usize {
        let mut count = 0;
        if self.temporal_filter.is_some() {
            count += 1;
        }
        if self.spatial_filter.is_some() {
            count += 1;
        }
        if self.polarity_filter.is_some() {
            count += 1;
        }
        if self.hot_pixel_filter.is_some() {
            count += 1;
        }
        if self.denoise_filter.is_some() {
            count += 1;
        }
        if self.drop_pixel_filter.is_some() {
            count += 1;
        }
        if self.downsampling_filter.is_some() {
            count += 1;
        }
        count
    }

    /// Get a description of the configured filters
    pub fn description(&self) -> String {
        let mut parts = Vec::new();

        if let Some(filter) = &self.temporal_filter {
            parts.push(format!("Temporal({})", filter.description()));
        }
        if let Some(filter) = &self.spatial_filter {
            parts.push(format!("Spatial({})", filter.description()));
        }
        if let Some(filter) = &self.polarity_filter {
            parts.push(format!("Polarity({})", filter.description()));
        }
        if let Some(filter) = &self.hot_pixel_filter {
            parts.push(format!("HotPixel({})", filter.description()));
        }
        if let Some(filter) = &self.denoise_filter {
            parts.push(format!("Denoise({})", filter.description()));
        }
        if let Some(filter) = &self.drop_pixel_filter {
            parts.push(format!("DropPixel({})", filter.description()));
        }
        if let Some(filter) = &self.downsampling_filter {
            parts.push(format!("Downsample({})", filter.description()));
        }

        if parts.is_empty() {
            "No filters configured".to_string()
        } else {
            format!("Filters: [{}]", parts.join(", "))
        }
    }

    /// Validate the configuration for consistency and correctness
    pub fn validate(&self) -> FilterResult<()> {
        // Check timeout
        if self.timeout_seconds < 0.0 {
            return Err(FilterError::InvalidConfig(
                "Timeout cannot be negative".to_string(),
            ));
        }

        // Check chunk size
        if self.chunk_size == 1 {
            return Err(FilterError::InvalidConfig(
                "Chunk size of 1 is inefficient, use 0 for automatic or larger value".to_string(),
            ));
        }

        // Validate individual filters
        if let Some(filter) = &self.temporal_filter {
            filter.validate()?;
        }
        if let Some(filter) = &self.spatial_filter {
            filter.validate()?;
        }
        if let Some(filter) = &self.polarity_filter {
            filter.validate()?;
        }
        if let Some(filter) = &self.hot_pixel_filter {
            filter.validate()?;
        }
        if let Some(filter) = &self.denoise_filter {
            filter.validate()?;
        }
        if let Some(filter) = &self.drop_pixel_filter {
            filter.validate()?;
        }
        if let Some(filter) = &self.downsampling_filter {
            filter.validate()?;
        }

        // Check for conflicting configurations
        if let (Some(_temporal), Some(_downsampling)) =
            (&self.temporal_filter, &self.downsampling_filter)
        {
            // Warn about potential issues, but don't fail validation
            warn!("Temporal filtering followed by downsampling may produce unexpected results");
        }

        // Validate memory limit is reasonable
        if self.memory_limit > 0 && self.memory_limit < 1024 * 1024 {
            // Less than 1MB
            warn!(
                "Memory limit of {} bytes is very low and may cause performance issues",
                self.memory_limit
            );
        }

        // Check for potentially conflicting parallel and chunk settings
        if !self.enable_parallel && self.chunk_size > 1_000_000 {
            warn!(
                "Large chunk size ({}) with parallel processing disabled may cause memory issues",
                self.chunk_size
            );
        }

        Ok(())
    }
}

/// Configuration for processing behavior
#[derive(Debug, Clone)]
pub struct ProcessingConfig {
    /// Use streaming processing for large datasets
    pub use_streaming: bool,
    /// Streaming chunk size
    pub stream_chunk_size: usize,
    /// Number of parallel threads (0 for automatic)
    pub num_threads: usize,
    /// Enable SIMD optimizations
    pub enable_simd: bool,
    /// Memory usage target in bytes
    pub memory_target: usize,
    /// Progress reporting interval (events)
    pub progress_interval: usize,
    /// Enable detailed logging
    pub verbose_logging: bool,
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            use_streaming: true,
            stream_chunk_size: 1_000_000, // 1M events per chunk
            num_threads: 0,               // Automatic
            enable_simd: true,
            memory_target: 1024 * 1024 * 1024, // 1GB target
            progress_interval: 10_000_000,     // Report every 10M events
            verbose_logging: false,
        }
    }
}

impl ProcessingConfig {
    /// Create a new processing configuration with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder method to set streaming mode
    pub fn with_streaming(mut self, enable: bool) -> Self {
        self.use_streaming = enable;
        self
    }

    /// Builder method to set chunk size
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.stream_chunk_size = size;
        self
    }

    /// Builder method to set number of threads
    pub fn with_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = num_threads;
        self
    }

    /// Builder method to enable/disable SIMD
    pub fn with_simd(mut self, enable: bool) -> Self {
        self.enable_simd = enable;
        self
    }

    /// Builder method to set memory target
    pub fn with_memory_target(mut self, target: usize) -> Self {
        self.memory_target = target;
        self
    }

    /// Builder method to set progress reporting interval
    pub fn with_progress_interval(mut self, interval: usize) -> Self {
        self.progress_interval = interval;
        self
    }

    /// Builder method to enable/disable verbose logging
    pub fn with_verbose_logging(mut self, enable: bool) -> Self {
        self.verbose_logging = enable;
        self
    }

    /// Configure for maximum performance (higher memory usage)
    pub fn high_performance() -> Self {
        Self {
            use_streaming: false,          // Process all in memory
            stream_chunk_size: 10_000_000, // Large chunks if streaming
            num_threads: 0,                // Use all cores
            enable_simd: true,
            memory_target: 4 * 1024 * 1024 * 1024, // 4GB target
            progress_interval: 50_000_000,         // Less frequent reporting
            verbose_logging: false,
        }
    }

    /// Configure for memory efficiency (lower memory usage)
    pub fn memory_efficient() -> Self {
        Self {
            use_streaming: true,
            stream_chunk_size: 100_000, // Small chunks
            num_threads: 2,             // Limit parallelism
            enable_simd: true,
            memory_target: 256 * 1024 * 1024, // 256MB target
            progress_interval: 1_000_000,     // Frequent reporting
            verbose_logging: false,
        }
    }

    /// Configure for debugging (verbose logging, validation)
    pub fn debug() -> Self {
        Self {
            use_streaming: true,
            stream_chunk_size: 10_000, // Very small chunks for debugging
            num_threads: 1,            // Single thread for easier debugging
            enable_simd: false,        // Disable SIMD for clearer debugging
            memory_target: 100 * 1024 * 1024, // 100MB target
            progress_interval: 10_000, // Very frequent reporting
            verbose_logging: true,
        }
    }
}

/// Trait for validating filter configurations
pub trait Validatable {
    /// Validate this configuration
    fn validate(&self) -> FilterResult<()>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ev_filtering::{PolarityFilter, SpatialFilter, TemporalFilter};

    #[test]
    fn test_empty_config() {
        let config = FilterConfig::new();
        assert!(!config.has_filters());
        assert_eq!(config.filter_count(), 0);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_builder() {
        let config = FilterConfig::new()
            .with_temporal_filter(TemporalFilter::new(1.0, 2.0))
            .with_spatial_filter(SpatialFilter::roi(0, 100, 0, 100))
            .with_polarity_filter(PolarityFilter::positive_only())
            .with_ensure_temporal_order(true)
            .with_timeout(30.0)
            .with_parallel_processing(true);

        assert!(config.has_filters());
        assert_eq!(config.filter_count(), 3);
        assert!(config.ensure_temporal_order);
        assert_eq!(config.timeout_seconds, 30.0);
        assert!(config.enable_parallel);
    }

    #[test]
    fn test_convenience_builders() {
        // Test Python API compatible builders
        let config = FilterConfig::new()
            .with_time_range(Some(0.1), Some(0.5))
            .with_roi(100, 500, 100, 400)
            .with_polarity(vec![1])
            .with_hot_pixel_removal(99.9)
            .with_refractory_period(1000.0);

        assert!(config.has_filters());
        assert_eq!(config.filter_count(), 5);
        assert!(config.temporal_filter.is_some());
        assert!(config.spatial_filter.is_some());
        assert!(config.polarity_filter.is_some());
        assert!(config.hot_pixel_filter.is_some());
        assert!(config.denoise_filter.is_some());
    }

    #[test]
    fn test_preset_configurations() {
        // Test noise removal preset
        let noise_config = FilterConfig::noise_removal();
        assert!(noise_config.hot_pixel_filter.is_some());
        assert!(noise_config.denoise_filter.is_some());
        assert!(noise_config.validate_input);
        assert!(noise_config.collect_stats);

        // Test high throughput preset
        let throughput_config = FilterConfig::high_throughput();
        assert!(throughput_config.hot_pixel_filter.is_some());
        assert!(throughput_config.denoise_filter.is_some());
        assert!(throughput_config.downsampling_filter.is_some());
        assert!(throughput_config.enable_parallel);
        assert!(!throughput_config.validate_input); // Disabled for speed

        // Test quality preservation preset
        let quality_config = FilterConfig::quality_preservation();
        assert!(quality_config.hot_pixel_filter.is_some());
        assert!(quality_config.denoise_filter.is_none()); // No refractory filtering
        assert!(quality_config.validate_input);
        assert!(quality_config.ensure_temporal_order);

        // Test debug preset
        let debug_config = FilterConfig::debug();
        assert!(!debug_config.enable_parallel);
        assert_eq!(debug_config.chunk_size, 10_000);
        assert!(debug_config.validate_input);
        assert!(debug_config.collect_stats);
    }

    #[test]
    fn test_config_validation() {
        // Valid config
        let config = FilterConfig::new().with_temporal_filter(TemporalFilter::new(1.0, 2.0));
        assert!(config.validate().is_ok());

        // Invalid timeout
        let config = FilterConfig::new().with_timeout(-1.0);
        assert!(config.validate().is_err());

        // Invalid chunk size
        let config = FilterConfig::new().with_chunk_size(1);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_description() {
        let config = FilterConfig::new()
            .with_temporal_filter(TemporalFilter::new(1.0, 2.0))
            .with_polarity_filter(PolarityFilter::positive_only());

        let description = config.description();
        assert!(description.contains("Temporal"));
        assert!(description.contains("Polarity"));
    }

    #[test]
    fn test_processing_config_presets() {
        let high_perf = ProcessingConfig::high_performance();
        assert!(!high_perf.use_streaming);
        assert!(high_perf.enable_simd);

        let mem_efficient = ProcessingConfig::memory_efficient();
        assert!(mem_efficient.use_streaming);
        assert_eq!(mem_efficient.stream_chunk_size, 100_000);

        let debug = ProcessingConfig::debug();
        assert!(debug.verbose_logging);
        assert_eq!(debug.num_threads, 1);
    }

    #[test]
    fn test_processing_config_builder() {
        let config = ProcessingConfig::new()
            .with_streaming(false)
            .with_chunk_size(500_000)
            .with_threads(4)
            .with_simd(true)
            .with_memory_target(2 * 1024 * 1024 * 1024) // 2GB
            .with_progress_interval(1_000_000)
            .with_verbose_logging(true);

        assert!(!config.use_streaming);
        assert_eq!(config.stream_chunk_size, 500_000);
        assert_eq!(config.num_threads, 4);
        assert!(config.enable_simd);
        assert_eq!(config.memory_target, 2 * 1024 * 1024 * 1024);
        assert_eq!(config.progress_interval, 1_000_000);
        assert!(config.verbose_logging);
    }

    #[test]
    fn test_error_display() {
        let error = FilterError::InvalidConfig("test error".to_string());
        assert_eq!(
            format!("{}", error),
            "Invalid filter configuration: test error"
        );

        let error = FilterError::ProcessingError("processing failed".to_string());
        assert_eq!(format!("{}", error), "Processing error: processing failed");
    }
}
