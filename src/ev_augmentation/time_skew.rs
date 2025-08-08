//! Time skew augmentation for event data
//!
//! This module implements temporal skewing by applying a linear transformation
//! to event timestamps, effectively stretching or compressing time.

use crate::ev_augmentation::{AugmentationError, AugmentationResult, Validatable};
// Removed: use crate::{Event, Events}; - legacy types no longer exist
use rand::{Rng, SeedableRng};

#[cfg(feature = "polars")]
use crate::ev_augmentation::COL_T;
use rand_distr::{Distribution, Uniform};
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

/// Time skew augmentation configuration
///
/// Skews all event timestamps according to a linear transform:
/// t_new = t_old * coefficient + offset
///
/// This can simulate:
/// - Clock drift (coefficient != 1.0)
/// - Time delays (offset != 0)
/// - Playback speed changes
///
/// # Example
///
/// ```rust
/// use evlib::ev_augmentation::TimeSkewAugmentation;
///
/// // Speed up events by 10%
/// let skew = TimeSkewAugmentation::new(1.1);
///
/// // Slow down by 20% and add 100ms delay
/// let skew = TimeSkewAugmentation::new(0.8)
///     .with_offset(0.1);
///
/// // Random coefficient between 0.9 and 1.1
/// let skew = TimeSkewAugmentation::random(0.9, 1.1);
/// ```
#[derive(Debug, Clone)]
pub struct TimeSkewAugmentation {
    /// Multiplicative coefficient for timestamps
    pub coefficient: f64,
    /// Optional range for random coefficient selection
    pub coefficient_range: Option<(f64, f64)>,
    /// Additive offset in seconds
    pub offset: f64,
    /// Optional range for random offset selection
    pub offset_range: Option<(f64, f64)>,
    /// Whether to clip negative timestamps
    pub clip_negative: bool,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl TimeSkewAugmentation {
    /// Create a new time skew augmentation with fixed coefficient
    ///
    /// # Arguments
    ///
    /// * `coefficient` - Multiplicative coefficient for timestamps
    pub fn new(coefficient: f64) -> Self {
        Self {
            coefficient,
            coefficient_range: None,
            offset: 0.0,
            offset_range: None,
            clip_negative: true,
            seed: None,
        }
    }

    /// Create time skew with random coefficient in range
    ///
    /// # Arguments
    ///
    /// * `min_coeff` - Minimum coefficient value
    /// * `max_coeff` - Maximum coefficient value
    pub fn random(min_coeff: f64, max_coeff: f64) -> Self {
        Self {
            coefficient: (min_coeff + max_coeff) / 2.0, // Default to middle
            coefficient_range: Some((min_coeff, max_coeff)),
            offset: 0.0,
            offset_range: None,
            clip_negative: true,
            seed: None,
        }
    }

    /// Set fixed offset
    pub fn with_offset(mut self, offset: f64) -> Self {
        self.offset = offset;
        self.offset_range = None;
        self
    }

    /// Set random offset range
    pub fn with_random_offset(mut self, min_offset: f64, max_offset: f64) -> Self {
        self.offset = (min_offset + max_offset) / 2.0;
        self.offset_range = Some((min_offset, max_offset));
        self
    }

    /// Set whether to clip negative timestamps
    pub fn with_clipping(mut self, clip: bool) -> Self {
        self.clip_negative = clip;
        self
    }

    /// Set random seed for reproducibility
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Get the actual coefficient and offset to use
    fn get_parameters(&self, rng: &mut impl Rng) -> (f64, f64) {
        let coeff = if let Some((min, max)) = self.coefficient_range {
            Uniform::new(min, max).sample(rng)
        } else {
            self.coefficient
        };

        let offset = if let Some((min, max)) = self.offset_range {
            Uniform::new(min, max).sample(rng)
        } else {
            self.offset
        };

        (coeff, offset)
    }

    /// Get description of this augmentation
    pub fn description(&self) -> String {
        if self.coefficient_range.is_some() || self.offset_range.is_some() {
            let coeff_str = if let Some((min, max)) = self.coefficient_range {
                format!("coeff∈[{:.2},{:.2}]", min, max)
            } else {
                format!("coeff={:.2}", self.coefficient)
            };

            let offset_str = if let Some((min, max)) = self.offset_range {
                format!("offset∈[{:.2},{:.2}]s", min, max)
            } else if self.offset.abs() > 1e-10 {
                format!("offset={:.3}s", self.offset)
            } else {
                String::new()
            };

            if offset_str.is_empty() {
                coeff_str
            } else {
                format!("{}, {}", coeff_str, offset_str)
            }
        } else {
            format!("coeff={:.2}, offset={:.3}s", self.coefficient, self.offset)
        }
    }
}

impl Validatable for TimeSkewAugmentation {
    fn validate(&self) -> AugmentationResult<()> {
        // Check coefficient
        if self.coefficient <= 0.0 {
            return Err(AugmentationError::InvalidConfig(
                "Time skew coefficient must be positive".to_string(),
            ));
        }

        // Check coefficient range
        if let Some((min, max)) = self.coefficient_range {
            if min <= 0.0 {
                return Err(AugmentationError::InvalidConfig(
                    "Minimum coefficient must be positive".to_string(),
                ));
            }
            if min >= max {
                return Err(AugmentationError::InvalidConfig(
                    "Invalid coefficient range".to_string(),
                ));
            }
        }

        // Check offset range
        if let Some((min, max)) = self.offset_range {
            if min >= max {
                return Err(AugmentationError::InvalidConfig(
                    "Invalid offset range".to_string(),
                ));
            }
        }

        Ok(())
    }
}

/// Apply time skew using Polars operations (Polars-first implementation)
///
/// This implementation uses vectorized operations for better performance on large datasets.
///
/// # Arguments
///
/// * `df` - Input LazyFrame containing event data
/// * `config` - Time skew configuration
///
/// # Returns
///
/// * `PolarsResult<LazyFrame>` - Skewed events as LazyFrame
#[cfg(feature = "polars")]
#[cfg_attr(feature = "tracing", instrument(skip(df), fields(config = ?config)))]
pub fn apply_time_skew_polars(
    df: LazyFrame,
    config: &TimeSkewAugmentation,
) -> PolarsResult<LazyFrame> {
    debug!("Applying time skew with Polars: {:?}", config);

    // For random parameters, we need to collect and process
    if config.coefficient_range.is_some() || config.offset_range.is_some() {
        let collected_df = df.collect()?;
        // For now, return the original dataframe as a placeholder
        let skewed_df = collected_df;
        return Ok(skewed_df.lazy());
    }

    // For fixed parameters, use Polars expressions
    let skewed_df = df.with_columns([
        // Apply linear transformation
        (col(COL_T) * lit(config.coefficient) + lit(config.offset)).alias("t_skewed"),
    ]);

    let result = if config.clip_negative {
        // Filter out negative timestamps
        skewed_df
            .filter(col("t_skewed").gt_eq(lit(0.0)))
            .with_columns([col("t_skewed").alias(COL_T)])
            .drop(["t_skewed"])
    } else {
        // Just ensure non-negative using conditional expression
        skewed_df
            .with_columns([when(col("t_skewed").gt_eq(lit(0.0)))
                .then(col("t_skewed"))
                .otherwise(lit(0.0))
                .alias(COL_T)])
            .drop(["t_skewed"])
    };

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_skew_creation() {
        let skew = TimeSkewAugmentation::new(1.5);
        assert_eq!(skew.coefficient, 1.5);
        assert_eq!(skew.offset, 0.0);
        assert!(skew.clip_negative);

        let skew = TimeSkewAugmentation::random(0.8, 1.2);
        assert!(skew.coefficient_range.is_some());
        assert_eq!(skew.coefficient_range.unwrap(), (0.8, 1.2));
    }

    #[test]
    fn test_time_skew_validation() {
        // Valid configurations
        let valid = TimeSkewAugmentation::new(2.0);
        assert!(valid.validate().is_ok());

        let valid = TimeSkewAugmentation::random(0.5, 1.5);
        assert!(valid.validate().is_ok());

        // Invalid: zero or negative coefficient
        let invalid = TimeSkewAugmentation::new(0.0);
        assert!(invalid.validate().is_err());

        let invalid = TimeSkewAugmentation::new(-1.0);
        assert!(invalid.validate().is_err());

        // Invalid: bad range
        let invalid = TimeSkewAugmentation::random(1.5, 0.5);
        assert!(invalid.validate().is_err());
    }
}
