//! Spatial jitter augmentation for event data
//!
//! This module implements spatial jittering by adding samples from a multivariate
//! Gaussian distribution to event coordinates, simulating sensor noise and small
//! movements.

use crate::ev_augmentation::{
    AugmentationError, AugmentationResult, SingleAugmentation, Validatable,
};

use crate::ev_core::{Event, Events};
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

impl SingleAugmentation for SpatialJitterAugmentation {
    fn apply(&self, events: &Events) -> AugmentationResult<Events> {
        spatial_jitter(events, self)
    }

    fn description(&self) -> String {
        format!("Spatial jitter: {}", self.description())
    }
}

/// Apply spatial jitter to events
///
/// This function adds Gaussian noise to event coordinates, simulating sensor noise
/// and small movements. Events can optionally be clipped if they fall outside sensor bounds.
///
/// # Arguments
///
/// * `events` - Input events to augment
/// * `config` - Spatial jitter configuration
///
/// # Returns
///
/// * `AugmentationResult<Events>` - Jittered events
#[cfg_attr(feature = "tracing", instrument(skip(events), fields(n_events = events.len())))]
pub fn spatial_jitter(
    events: &Events,
    config: &SpatialJitterAugmentation,
) -> AugmentationResult<Events> {
    let start_time = std::time::Instant::now();

    if events.is_empty() {
        debug!("No events to jitter");
        return Ok(Vec::new());
    }

    // Validate configuration
    config.validate()?;

    // Initialize RNG
    let mut rng = if let Some(seed) = config.seed {
        rand::rngs::StdRng::seed_from_u64(seed)
    } else {
        rand::rngs::StdRng::from_entropy()
    };

    // Apply jitter to each event
    let mut jittered_events = Vec::with_capacity(events.len());
    let mut clipped_count = 0;

    for event in events {
        let (dx, dy) = config.generate_jitter(&mut rng);

        // Apply jitter
        let new_x = event.x as f64 + dx;
        let new_y = event.y as f64 + dy;

        // Check bounds if clipping is enabled
        if config.clip_outliers {
            if let Some((width, height)) = config.sensor_size {
                if new_x < 0.0 || new_x >= width as f64 || new_y < 0.0 || new_y >= height as f64 {
                    clipped_count += 1;
                    continue;
                }
            }
        }

        // Create jittered event, ensuring coordinates stay within u16 bounds
        let jittered_event = Event {
            t: event.t,
            x: new_x.round().max(0.0).min(u16::MAX as f64) as u16,
            y: new_y.round().max(0.0).min(u16::MAX as f64) as u16,
            polarity: event.polarity,
        };

        jittered_events.push(jittered_event);
    }

    let processing_time = start_time.elapsed().as_secs_f64();
    let output_count = jittered_events.len();

    info!(
        "Spatial jitter applied: {} -> {} events ({} clipped) in {:.3}s",
        events.len(),
        output_count,
        clipped_count,
        processing_time
    );

    Ok(jittered_events)
}

/// Apply spatial jitter using Polars operations (Polars-first implementation)
///
/// This implementation uses vectorized operations for better performance on large datasets.
///
/// # Arguments
///
/// * `df` - Input LazyFrame containing event data
/// * `config` - Spatial jitter configuration
///
/// # Returns
///
/// * `PolarsResult<LazyFrame>` - Jittered events as LazyFrame
#[cfg(feature = "polars")]
#[cfg_attr(feature = "tracing", instrument(skip(df), fields(config = ?config)))]
pub fn apply_spatial_jitter_polars(
    df: LazyFrame,
    config: &SpatialJitterAugmentation,
) -> PolarsResult<LazyFrame> {
    debug!("Applying spatial jitter with Polars: {:?}", config);

    // For now, we'll collect and use the Vec implementation
    // A fully vectorized Polars implementation would require custom expressions
    // for multivariate normal sampling

    let collected_df = df.collect()?;

    // Convert to Events
    let events = crate::ev_augmentation::dataframe_to_events(&collected_df)
        .map_err(|e| PolarsError::ComputeError(format!("Conversion error: {}", e).into()))?;

    // Apply jitter
    let jittered = spatial_jitter(&events, config)
        .map_err(|e| PolarsError::ComputeError(format!("Jitter error: {}", e).into()))?;

    // Convert back to DataFrame
    let jittered_df = crate::ev_core::events_to_dataframe(&jittered)?;

    Ok(jittered_df.lazy())
}

/// Convenience function for simple spatial jitter
///
/// # Arguments
///
/// * `events` - Input events
/// * `std_x` - Standard deviation for x-coordinate jitter
/// * `std_y` - Standard deviation for y-coordinate jitter
pub fn apply_spatial_jitter_simple(
    events: &Events,
    std_x: f64,
    std_y: f64,
) -> AugmentationResult<Events> {
    let config = SpatialJitterAugmentation::new(std_x * std_x, std_y * std_y);
    spatial_jitter(events, &config)
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
        ]
    }

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

    #[test]
    fn test_spatial_jitter_application() {
        let events = create_test_events();
        let config = SpatialJitterAugmentation::new(1.0, 1.0).with_seed(42);

        let jittered = spatial_jitter(&events, &config).unwrap();

        // Should have same number of events
        assert_eq!(jittered.len(), events.len());

        // Timestamps and polarities should be unchanged
        for (orig, jit) in events.iter().zip(jittered.iter()) {
            assert_eq!(orig.t, jit.t);
            assert_eq!(orig.polarity, jit.polarity);
        }

        // Coordinates should be different (with very high probability)
        let coords_changed = events
            .iter()
            .zip(jittered.iter())
            .any(|(orig, jit)| orig.x != jit.x || orig.y != jit.y);
        assert!(coords_changed);
    }

    #[test]
    fn test_spatial_jitter_with_clipping() {
        let events = vec![
            Event {
                t: 1.0,
                x: 5,
                y: 5,
                polarity: true,
            },
            Event {
                t: 2.0,
                x: 635,
                y: 475,
                polarity: false,
            },
        ];

        // Large jitter with clipping
        let config = SpatialJitterAugmentation::new(100.0, 100.0)
            .with_clipping(640, 480)
            .with_seed(42);

        let jittered = spatial_jitter(&events, &config).unwrap();

        // Some events may be clipped
        assert!(jittered.len() <= events.len());

        // All remaining events should be within bounds
        for event in &jittered {
            assert!(event.x < 640);
            assert!(event.y < 480);
        }
    }

    #[test]
    fn test_spatial_jitter_reproducibility() {
        let events = create_test_events();
        let config = SpatialJitterAugmentation::new(1.0, 1.0).with_seed(12345);

        let jittered1 = spatial_jitter(&events, &config).unwrap();
        let jittered2 = spatial_jitter(&events, &config).unwrap();

        // With same seed, results should be identical
        assert_eq!(jittered1.len(), jittered2.len());
        for (e1, e2) in jittered1.iter().zip(jittered2.iter()) {
            assert_eq!(e1.x, e2.x);
            assert_eq!(e1.y, e2.y);
        }
    }

    #[test]
    fn test_spatial_jitter_empty_events() {
        let events = Vec::new();
        let config = SpatialJitterAugmentation::new(1.0, 1.0);
        let jittered = spatial_jitter(&events, &config).unwrap();
        assert!(jittered.is_empty());
    }
}
