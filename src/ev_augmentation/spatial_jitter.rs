//! Polars-first spatial jitter augmentation for event camera data
//!
//! This module provides spatial jittering functionality using Polars DataFrames
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
//! - Vectorized operations: All jittering uses SIMD-optimized Polars operations
//! - Memory efficiency: No intermediate Vec<Event> allocations
//! - Query optimization: Polars optimizes the entire augmentation pipeline
//!
//! # Example
//!
//! ```rust
//! use polars::prelude::*;
//! use evlib::ev_augmentation::spatial_jitter::*;
//!
//! // Convert events to LazyFrame once
//! let events_df = events_to_dataframe(&events)?.lazy();
//!
//! // Apply spatial jittering with Polars expressions
//! let jittered = apply_spatial_jitter(events_df, &SpatialJitterAugmentation::new(1.0, 1.0))?;
//! ```

use crate::ev_augmentation::{AugmentationError, AugmentationResult, Validatable};

// Removed: use crate::{Event, Events}; - legacy types no longer exist
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

#[cfg(feature = "tracing")]
use tracing::{debug, info, instrument};

#[cfg(not(feature = "tracing"))]
macro_rules! debug {
    ($($args:tt)*) => {};
}

#[cfg(not(feature = "tracing"))]
macro_rules! info {
    ($($args:tt)*) => {};
}

#[cfg(feature = "polars")]
use polars::prelude::*;

// Polars column names for event data consistency
#[cfg(feature = "polars")]
pub const COL_X: &str = "x";
#[cfg(feature = "polars")]
pub const COL_Y: &str = "y";
#[cfg(feature = "polars")]
pub const COL_T: &str = "t";
#[cfg(feature = "polars")]
pub const COL_POLARITY: &str = "polarity";

/// Spatial jitter augmentation configuration
///
/// Changes x/y coordinates for each event by adding samples from a multivariate
/// Gaussian distribution with the following properties:
///
/// mean = [x, y]
/// Σ = [[var_x, sigma_xy], [sigma_xy, var_y]]
///
/// # Example
///
/// ```rust
/// use evlib::ev_augmentation::SpatialJitterAugmentation;
///
/// // Create jitter with independent x,y noise
/// let jitter = SpatialJitterAugmentation::new(1.0, 1.0);
///
/// // Create jitter with correlated noise
/// let jitter = SpatialJitterAugmentation::new(2.0, 2.0)
///     .with_correlation(0.5);
/// ```
#[derive(Debug, Clone)]
pub struct SpatialJitterAugmentation {
    /// Variance for x-coordinate jitter
    pub var_x: f64,
    /// Variance for y-coordinate jitter
    pub var_y: f64,
    /// Covariance between x and y (creates diagonal jitter)
    pub sigma_xy: f64,
    /// Whether to clip events that fall outside sensor bounds
    pub clip_outliers: bool,
    /// Optional sensor size for clipping [width, height]
    pub sensor_size: Option<(u16, u16)>,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl SpatialJitterAugmentation {
    /// Create a new spatial jitter augmentation
    ///
    /// # Arguments
    ///
    /// * `var_x` - Variance for x-coordinate jitter (squared standard deviation)
    /// * `var_y` - Variance for y-coordinate jitter (squared standard deviation)
    pub fn new(var_x: f64, var_y: f64) -> Self {
        Self {
            var_x,
            var_y,
            sigma_xy: 0.0,
            clip_outliers: false,
            sensor_size: None,
            seed: None,
        }
    }

    /// Set the covariance between x and y coordinates
    ///
    /// This creates correlated jitter along diagonal axes
    pub fn with_correlation(mut self, sigma_xy: f64) -> Self {
        self.sigma_xy = sigma_xy;
        self
    }

    /// Enable clipping of events that fall outside sensor bounds
    pub fn with_clipping(mut self, sensor_width: u16, sensor_height: u16) -> Self {
        self.clip_outliers = true;
        self.sensor_size = Some((sensor_width, sensor_height));
        self
    }

    /// Set random seed for reproducibility
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Check if the covariance matrix is valid (positive semi-definite)
    fn is_valid_covariance(&self) -> bool {
        // For a 2x2 covariance matrix [[var_x, sigma_xy], [sigma_xy, var_y]]
        // Positive semi-definite requires:
        // 1. var_x >= 0 and var_y >= 0 (diagonal elements non-negative)
        // 2. determinant = var_x * var_y - sigma_xy^2 >= 0
        let det = self.var_x * self.var_y - self.sigma_xy * self.sigma_xy;
        let trace = self.var_x + self.var_y;

        // Positive semi-definite check: det >= 0 and trace >= 0
        det >= 0.0 && trace >= 0.0
    }

    /// Generate jitter for a single coordinate
    fn generate_jitter(&self, rng: &mut impl Rng) -> (f64, f64) {
        // For uncorrelated jitter, use simple normal distributions
        if self.sigma_xy.abs() < 1e-10 {
            let dist_x = Normal::new(0.0, self.var_x.sqrt()).unwrap();
            let dist_y = Normal::new(0.0, self.var_y.sqrt()).unwrap();
            return (dist_x.sample(rng), dist_y.sample(rng));
        }

        // For correlated jitter, use multivariate normal
        // First generate two independent standard normal samples
        let std_normal = Normal::new(0.0, 1.0).unwrap();
        let z1 = std_normal.sample(rng);
        let z2 = std_normal.sample(rng);

        // Cholesky decomposition of covariance matrix to generate correlated samples
        let a = self.var_x.sqrt();
        let b = self.sigma_xy / a;
        let c = (self.var_y - b * b).max(0.0).sqrt();

        let x_jitter = a * z1;
        let y_jitter = b * z1 + c * z2;

        (x_jitter, y_jitter)
    }

    /// Apply spatial jitter directly to DataFrame (recommended approach)
    ///
    /// This is the high-performance DataFrame-native method that should be used
    /// instead of the legacy Vec<Event> approach when possible.
    ///
    /// # Arguments
    ///
    /// * `df` - Input LazyFrame containing event data
    ///
    /// # Returns
    ///
    /// Filtered LazyFrame with spatial jitter applied
    #[cfg(feature = "polars")]
    pub fn apply_to_dataframe(&self, df: LazyFrame) -> PolarsResult<LazyFrame> {
        apply_spatial_jitter(df, self)
    }

    /// Apply spatial jitter directly to DataFrame and return DataFrame
    ///
    /// Convenience method that applies jittering and collects the result.
    ///
    /// # Arguments
    ///
    /// * `df` - Input DataFrame containing event data
    ///
    /// # Returns
    ///
    /// Jittered DataFrame with spatial noise applied
    #[cfg(feature = "polars")]
    pub fn apply_to_dataframe_eager(&self, df: DataFrame) -> PolarsResult<DataFrame> {
        apply_spatial_jitter(df.lazy(), self)?.collect()
    }

    /// Get description of this augmentation
    pub fn description(&self) -> String {
        if self.sigma_xy.abs() < 1e-10 {
            format!("σx²={:.2}, σy²={:.2}", self.var_x, self.var_y)
        } else {
            format!(
                "σx²={:.2}, σy²={:.2}, σxy={:.2}",
                self.var_x, self.var_y, self.sigma_xy
            )
        }
    }
}

impl Validatable for SpatialJitterAugmentation {
    fn validate(&self) -> AugmentationResult<()> {
        if self.var_x < 0.0 {
            return Err(AugmentationError::InvalidConfig(
                "X variance must be non-negative".to_string(),
            ));
        }
        if self.var_y < 0.0 {
            return Err(AugmentationError::InvalidConfig(
                "Y variance must be non-negative".to_string(),
            ));
        }
        if !self.is_valid_covariance() {
            return Err(AugmentationError::InvalidConfig(
                "Covariance matrix must be positive semi-definite".to_string(),
            ));
        }
        if self.clip_outliers && self.sensor_size.is_none() {
            return Err(AugmentationError::InvalidConfig(
                "Sensor size must be specified when clipping is enabled".to_string(),
            ));
        }
        Ok(())
    }
}

/// Apply spatial jitter using Polars expressions
///
/// This is the main spatial jitter function that works entirely with Polars
/// operations for maximum performance. For now, this function generates
/// random jitter using a deterministic seeded approach and applies it
/// using vectorized Polars operations.
///
/// # Arguments
///
/// * `df` - Input LazyFrame containing event data
/// * `config` - Spatial jitter configuration
///
/// # Returns
///
/// Jittered LazyFrame with spatial noise applied
///
/// # Example
///
/// ```rust
/// use polars::prelude::*;
/// use evlib::ev_augmentation::spatial_jitter::*;
///
/// let events_df = events_to_dataframe(&events)?.lazy();
/// let config = SpatialJitterAugmentation::new(1.0, 1.0);
/// let jittered = apply_spatial_jitter(events_df, &config)?;
/// ```
#[cfg(feature = "polars")]
#[cfg_attr(feature = "tracing", instrument(skip(df), fields(config = ?config)))]
pub fn apply_spatial_jitter(
    df: LazyFrame,
    config: &SpatialJitterAugmentation,
) -> PolarsResult<LazyFrame> {
    debug!("Applying spatial jitter with Polars: {:?}", config);

    // For complex transformations with random generation, we'll collect and use Vec operations
    // This provides better control and compatibility across Polars versions
    let collected_df = df.collect()?;

    // TODO: Implement native Polars spatial jitter without Events type
    return Err(PolarsError::ComputeError(
        "Spatial jitter temporarily disabled - Events type removed".into(),
    ));
}

/// Legacy Polars function for backward compatibility
#[cfg(feature = "polars")]
pub fn apply_spatial_jitter_polars(
    df: LazyFrame,
    config: &SpatialJitterAugmentation,
) -> PolarsResult<LazyFrame> {
    apply_spatial_jitter(df, config)
}

/// Apply spatial jitter directly to LazyFrame - DataFrame-native version (recommended)
///
/// This function applies spatial jittering directly to a LazyFrame for optimal performance.
/// Use this instead of the legacy Vec<Event> version when possible.
///
/// # Arguments
///
/// * `df` - Input LazyFrame containing event data
/// * `std_x` - Standard deviation for x-coordinate jitter
/// * `std_y` - Standard deviation for y-coordinate jitter
///
/// # Returns
///
/// Jittered LazyFrame
#[cfg(feature = "polars")]
pub fn apply_spatial_jitter_df(df: LazyFrame, std_x: f64, std_y: f64) -> PolarsResult<LazyFrame> {
    let config = SpatialJitterAugmentation::new(std_x * std_x, std_y * std_y);
    apply_spatial_jitter(df, &config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_jitter_creation() {
        let jitter = SpatialJitterAugmentation::new(1.0, 2.0);
        assert_eq!(jitter.var_x, 1.0);
        assert_eq!(jitter.var_y, 2.0);
        assert_eq!(jitter.sigma_xy, 0.0);
        assert!(!jitter.clip_outliers);
    }

    #[test]
    fn test_spatial_jitter_validation() {
        // Valid configuration
        let valid = SpatialJitterAugmentation::new(1.0, 1.0);
        assert!(valid.validate().is_ok());

        // Invalid: negative variance
        let invalid = SpatialJitterAugmentation::new(-1.0, 1.0);
        assert!(invalid.validate().is_err());

        // Invalid: clipping without sensor size
        let invalid = SpatialJitterAugmentation::new(1.0, 1.0)
            .with_clipping(640, 480)
            .with_clipping(0, 0); // This would set clip_outliers=true but sensor_size=Some((0,0))
        let mut invalid_modified = invalid;
        invalid_modified.sensor_size = None;
        assert!(invalid_modified.validate().is_err());
    }
}
