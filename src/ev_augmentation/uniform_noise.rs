//! Polars-first uniform noise augmentation for event camera data
//!
//! This module provides noise injection functionality using Polars DataFrames
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
//! - Vectorized operations: All noise generation uses SIMD-optimized Polars operations
//! - Memory efficiency: No intermediate Vec<Event> allocations
//! - Query optimization: Polars optimizes the entire augmentation pipeline
//!
//! # Noise Generation
//!
//! This module implements noise injection by adding uniformly distributed
//! synthetic events across the sensor dimensions and time range.
//!
//! # Example
//!
//! ```rust
//! use polars::prelude::*;
//! use evlib::ev_augmentation::uniform_noise::*;
//!
//! // Convert events to LazyFrame once
//! let events_df = events_to_dataframe(&events)?.lazy();
//!
//! // Apply uniform noise with Polars expressions
//! let config = UniformNoiseAugmentation::new(1000, 640, 480);
//! let noisy = apply_uniform_noise(events_df, &config)?;
//! ```

use crate::ev_augmentation::{AugmentationError, AugmentationResult, Validatable};

// Polars column names for event data consistency
#[cfg(feature = "polars")]
pub const COL_X: &str = "x";
#[cfg(feature = "polars")]
pub const COL_Y: &str = "y";
#[cfg(feature = "polars")]
pub const COL_T: &str = "t";
#[cfg(feature = "polars")]
pub const COL_POLARITY: &str = "polarity";
// Removed: use crate::{Event, Events}; - legacy types no longer exist
use rand::SeedableRng;
use rand_distr::Distribution;
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

#[cfg(feature = "polars")]
use polars::prelude::*;

/// Uniform noise augmentation configuration
///
/// Adds a fixed number of noise events that are uniformly distributed
/// across sensor dimensions and the time range of existing events.
/// This simulates background activity and sensor noise.
///
/// # Example
///
/// ```rust
/// use evlib::ev_augmentation::UniformNoiseAugmentation;
///
/// // Add 1000 noise events to a 640x480 sensor
/// let noise = UniformNoiseAugmentation::new(1000, 640, 480);
///
/// // Add noise with balanced polarities
/// let noise = UniformNoiseAugmentation::new(500, 640, 480)
///     .with_polarity_balance(true);
/// ```
#[derive(Debug, Clone)]
pub struct UniformNoiseAugmentation {
    /// Number of noise events to add
    pub n_events: usize,
    /// Sensor width in pixels
    pub sensor_width: u16,
    /// Sensor height in pixels
    pub sensor_height: u16,
    /// Whether to balance positive and negative polarities
    pub balance_polarities: bool,
    /// Whether to sort events after adding noise
    pub sort_timestamps: bool,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl UniformNoiseAugmentation {
    /// Create a new uniform noise augmentation
    ///
    /// # Arguments
    ///
    /// * `n_events` - Number of noise events to add
    /// * `sensor_width` - Sensor width in pixels
    /// * `sensor_height` - Sensor height in pixels
    pub fn new(n_events: usize, sensor_width: u16, sensor_height: u16) -> Self {
        Self {
            n_events,
            sensor_width,
            sensor_height,
            balance_polarities: false,
            sort_timestamps: true,
            seed: None,
        }
    }

    /// Set whether to balance positive and negative polarities
    pub fn with_polarity_balance(mut self, balance: bool) -> Self {
        self.balance_polarities = balance;
        self
    }

    /// Set whether to sort events after adding noise
    pub fn with_sorting(mut self, sort: bool) -> Self {
        self.sort_timestamps = sort;
        self
    }

    /// Set random seed for reproducibility
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Get description of this augmentation
    pub fn description(&self) -> String {
        format!(
            "n={}, sensor={}x{}",
            self.n_events, self.sensor_width, self.sensor_height
        )
    }

    /// Apply uniform noise directly to DataFrame (recommended approach)
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
    /// Combined LazyFrame with original events plus uniform noise
    #[cfg(feature = "polars")]
    pub fn apply_to_dataframe(&self, df: LazyFrame) -> PolarsResult<LazyFrame> {
        apply_uniform_noise(df, self)
    }

    /// Apply uniform noise directly to DataFrame and return DataFrame
    ///
    /// Convenience method that applies noise and collects the result.
    ///
    /// # Arguments
    ///
    /// * `df` - Input DataFrame containing event data
    ///
    /// # Returns
    ///
    /// Combined DataFrame with original events plus uniform noise
    #[cfg(feature = "polars")]
    pub fn apply_to_dataframe_eager(&self, df: DataFrame) -> PolarsResult<DataFrame> {
        apply_uniform_noise(df.lazy(), self)?.collect()
    }
}

impl Validatable for UniformNoiseAugmentation {
    fn validate(&self) -> AugmentationResult<()> {
        if self.sensor_width == 0 || self.sensor_height == 0 {
            return Err(AugmentationError::InvalidSensorSize(
                self.sensor_width,
                self.sensor_height,
            ));
        }
        Ok(())
    }
}

/// Apply uniform noise using Polars expressions
///
/// This is the main uniform noise function that works entirely with Polars
/// operations for maximum performance. It creates synthetic noise events
/// and combines them with the input data using efficient vectorized operations.
///
/// # Arguments
///
/// * `df` - Input LazyFrame containing event data
/// * `config` - Uniform noise configuration
///
/// # Returns
///
/// Combined LazyFrame with original events plus uniform noise
///
/// # Example
///
/// ```rust
/// use polars::prelude::*;
/// use evlib::ev_augmentation::uniform_noise::*;
///
/// let events_df = events_to_dataframe(&events)?.lazy();
/// let config = UniformNoiseAugmentation::new(1000, 640, 480);
/// let noisy = apply_uniform_noise(events_df, &config)?;
/// ```
#[cfg(feature = "polars")]
#[cfg_attr(feature = "tracing", instrument(skip(df), fields(config = ?config)))]
pub fn apply_uniform_noise(
    df: LazyFrame,
    config: &UniformNoiseAugmentation,
) -> PolarsResult<LazyFrame> {
    debug!("Applying uniform noise with Polars: {:?}", config);

    if config.n_events == 0 {
        debug!("No noise events to add");
        return Ok(df);
    }

    // Get time range from existing events
    let time_bounds = df
        .clone()
        .select([
            col(COL_T).min().alias("t_min"),
            col(COL_T).max().alias("t_max"),
        ])
        .collect()?;

    let t_min = time_bounds
        .column("t_min")?
        .get(0)?
        .try_extract::<f64>()
        .unwrap_or(0.0);
    let t_max = time_bounds
        .column("t_max")?
        .get(0)?
        .try_extract::<f64>()
        .unwrap_or(1.0);

    // Ensure we have a valid time range
    let (time_min, time_max) = if t_max > t_min {
        (t_min, t_max)
    } else {
        (0.0, 1.0)
    };

    info!(
        "Adding {} noise events in time range [{:.6}, {:.6}]",
        config.n_events, time_min, time_max
    );

    // Generate random data for noise events using deterministic seed if provided
    let mut rng = if let Some(seed) = config.seed {
        rand::rngs::StdRng::seed_from_u64(seed)
    } else {
        rand::rngs::StdRng::from_entropy()
    };

    use rand::Rng;

    // Generate noise event data
    let mut noise_x = Vec::with_capacity(config.n_events);
    let mut noise_y = Vec::with_capacity(config.n_events);
    let mut noise_t = Vec::with_capacity(config.n_events);
    let mut noise_polarity = Vec::with_capacity(config.n_events);

    let x_dist = rand_distr::Uniform::new(0, config.sensor_width);
    let y_dist = rand_distr::Uniform::new(0, config.sensor_height);
    let t_dist = rand_distr::Uniform::new(time_min, time_max);

    for i in 0..config.n_events {
        noise_x.push(x_dist.sample(&mut rng) as i64);
        noise_y.push(y_dist.sample(&mut rng) as i64);
        noise_t.push(t_dist.sample(&mut rng));

        let polarity = if config.balance_polarities {
            i % 2 == 0
        } else {
            rng.gen_bool(0.5)
        };
        noise_polarity.push(if polarity { 1i64 } else { 0i64 });
    }

    // Create noise DataFrame
    let noise_df = df! {
        COL_X => noise_x,
        COL_Y => noise_y,
        COL_T => noise_t,
        COL_POLARITY => noise_polarity,
    }?;

    // Combine original events with noise events
    let combined_df = df.collect()?.vstack(&noise_df)?;

    let mut result = combined_df.lazy();

    // Sort by timestamp if requested
    if config.sort_timestamps {
        result = result.sort([COL_T], SortMultipleOptions::default());
    }

    Ok(result)
}

/// Legacy Polars function for backward compatibility
#[cfg(feature = "polars")]
pub fn apply_uniform_noise_polars(
    df: LazyFrame,
    config: &UniformNoiseAugmentation,
) -> PolarsResult<LazyFrame> {
    apply_uniform_noise(df, config)
}

/// Apply uniform noise directly to LazyFrame - DataFrame-native version (recommended)
///
/// This function applies noise injection directly to a LazyFrame for optimal performance.
/// Use this instead of the legacy Vec<Event> version when possible.
///
/// # Arguments
///
/// * `df` - Input LazyFrame containing event data
/// * `n_events` - Number of noise events to add
/// * `sensor_width` - Sensor width in pixels
/// * `sensor_height` - Sensor height in pixels
///
/// # Returns
///
/// Combined LazyFrame with original events plus noise
#[cfg(feature = "polars")]
pub fn add_uniform_noise_df(
    df: LazyFrame,
    n_events: usize,
    sensor_width: u16,
    sensor_height: u16,
) -> PolarsResult<LazyFrame> {
    let config = UniformNoiseAugmentation::new(n_events, sensor_width, sensor_height);
    apply_uniform_noise(df, &config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_noise_creation() {
        let noise = UniformNoiseAugmentation::new(100, 640, 480);
        assert_eq!(noise.n_events, 100);
        assert_eq!(noise.sensor_width, 640);
        assert_eq!(noise.sensor_height, 480);
        assert!(!noise.balance_polarities);
        assert!(noise.sort_timestamps);
    }

    #[test]
    fn test_uniform_noise_validation() {
        // Valid configuration
        let valid = UniformNoiseAugmentation::new(100, 640, 480);
        assert!(valid.validate().is_ok());

        // Invalid: zero sensor size
        let invalid = UniformNoiseAugmentation::new(100, 0, 480);
        assert!(invalid.validate().is_err());

        let invalid = UniformNoiseAugmentation::new(100, 640, 0);
        assert!(invalid.validate().is_err());
    }
}
