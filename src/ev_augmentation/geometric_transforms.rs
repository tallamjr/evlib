//! Geometric transformation augmentations for event data
//!
//! This module implements geometric transformations including horizontal flip,
//! vertical flip, and polarity flip operations on event data.

use crate::ev_augmentation::{
    AugmentationError, AugmentationResult, SingleAugmentation, Validatable,
};
use crate::ev_core::{Event, Events};
use rand::{Rng, SeedableRng};
use tracing::{debug, info, instrument};

#[cfg(feature = "polars")]
use crate::ev_augmentation::{COL_POLARITY, COL_X, COL_Y};

#[cfg(feature = "polars")]
use polars::prelude::*;

/// Geometric transform augmentation configuration
///
/// Applies geometric transformations to events with specified probabilities:
/// - Horizontal flip (left-right)
/// - Vertical flip (up-down)
/// - Polarity flip
///
/// # Example
///
/// ```rust
/// use evlib::ev_augmentation::GeometricTransformAugmentation;
///
/// // Create geometric transforms with 50% probability each
/// let transforms = GeometricTransformAugmentation::new(0.5, 0.5, 0.3, 640, 480);
/// ```
#[derive(Debug, Clone)]
pub struct GeometricTransformAugmentation {
    /// Probability of horizontal flip (0-1)
    pub flip_lr_prob: f64,
    /// Probability of vertical flip (0-1)
    pub flip_ud_prob: f64,
    /// Probability of polarity flip (0-1)
    pub flip_polarity_prob: f64,
    /// Sensor width in pixels
    pub sensor_width: u16,
    /// Sensor height in pixels
    pub sensor_height: u16,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl GeometricTransformAugmentation {
    /// Create new geometric transform augmentation with default probabilities (all 0.0)
    ///
    /// # Arguments
    ///
    /// * `sensor_width` - Sensor width in pixels
    /// * `sensor_height` - Sensor height in pixels
    pub fn new(sensor_width: u16, sensor_height: u16) -> Self {
        Self {
            flip_lr_prob: 0.0,
            flip_ud_prob: 0.0,
            flip_polarity_prob: 0.0,
            sensor_width,
            sensor_height,
            seed: None,
        }
    }

    /// Set horizontal flip probability
    pub fn with_flip_lr_probability(mut self, prob: f64) -> Self {
        self.flip_lr_prob = prob;
        self
    }

    /// Set vertical flip probability
    pub fn with_flip_ud_probability(mut self, prob: f64) -> Self {
        self.flip_ud_prob = prob;
        self
    }

    /// Set polarity flip probability
    pub fn with_flip_polarity_probability(mut self, prob: f64) -> Self {
        self.flip_polarity_prob = prob;
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
            "flip_lr={:.2}, flip_ud={:.2}, flip_pol={:.2}, size={}x{}",
            self.flip_lr_prob,
            self.flip_ud_prob,
            self.flip_polarity_prob,
            self.sensor_width,
            self.sensor_height
        )
    }

    /// Check if any transformations are enabled
    pub fn is_enabled(&self) -> bool {
        self.flip_lr_prob > 0.0 || self.flip_ud_prob > 0.0 || self.flip_polarity_prob > 0.0
    }
}

impl Validatable for GeometricTransformAugmentation {
    fn validate(&self) -> AugmentationResult<()> {
        if !(0.0..=1.0).contains(&self.flip_lr_prob) {
            return Err(AugmentationError::InvalidProbability(self.flip_lr_prob));
        }
        if !(0.0..=1.0).contains(&self.flip_ud_prob) {
            return Err(AugmentationError::InvalidProbability(self.flip_ud_prob));
        }
        if !(0.0..=1.0).contains(&self.flip_polarity_prob) {
            return Err(AugmentationError::InvalidProbability(
                self.flip_polarity_prob,
            ));
        }
        if self.sensor_width == 0 || self.sensor_height == 0 {
            return Err(AugmentationError::InvalidSensorSize(
                self.sensor_width,
                self.sensor_height,
            ));
        }
        Ok(())
    }
}

impl SingleAugmentation for GeometricTransformAugmentation {
    #[instrument(skip(events), level = "debug")]
    fn apply(&self, events: &Events) -> AugmentationResult<Events> {
        if events.is_empty() {
            return Ok(events.clone());
        }

        self.validate()?;

        let mut rng = if let Some(seed) = self.seed {
            rand::rngs::StdRng::seed_from_u64(seed)
        } else {
            rand::rngs::StdRng::from_entropy()
        };

        // Determine which transforms to apply
        let apply_flip_lr = rng.gen::<f64>() < self.flip_lr_prob;
        let apply_flip_ud = rng.gen::<f64>() < self.flip_ud_prob;
        let apply_flip_polarity = rng.gen::<f64>() < self.flip_polarity_prob;

        debug!(
            "Applying geometric transforms: flip_lr={}, flip_ud={}, flip_polarity={}",
            apply_flip_lr, apply_flip_ud, apply_flip_polarity
        );

        let mut transformed_events = Vec::with_capacity(events.len());

        for event in events {
            let mut transformed_event = *event;

            // Apply horizontal flip
            if apply_flip_lr {
                transformed_event.x = self.sensor_width - 1 - event.x;
            }

            // Apply vertical flip
            if apply_flip_ud {
                transformed_event.y = self.sensor_height - 1 - event.y;
            }

            // Apply polarity flip
            if apply_flip_polarity {
                transformed_event.polarity = !event.polarity;
            }

            transformed_events.push(transformed_event);
        }

        info!(
            "Applied geometric transforms to {} events",
            transformed_events.len()
        );

        Ok(transformed_events)
    }

    fn description(&self) -> String {
        self.description()
    }
}

/// Apply geometric transforms to events
///
/// # Arguments
///
/// * `events` - Input events
/// * `config` - Geometric transform configuration
///
/// # Returns
///
/// * `AugmentationResult<Events>` - Transformed events
#[instrument(skip(events), level = "debug")]
pub fn geometric_transforms(
    events: &Events,
    config: &GeometricTransformAugmentation,
) -> AugmentationResult<Events> {
    let start_time = std::time::Instant::now();

    if events.is_empty() {
        debug!("No events to transform");
        return Ok(Vec::new());
    }

    // Validate configuration
    config.validate()?;

    // If no transformations are enabled, return original events
    if !config.is_enabled() {
        debug!("No geometric transformations enabled");
        return Ok(events.clone());
    }

    // Initialize RNG
    let mut rng = if let Some(seed) = config.seed {
        rand::rngs::StdRng::seed_from_u64(seed)
    } else {
        rand::rngs::StdRng::from_entropy()
    };

    // Determine which transformations to apply (once for entire sequence)
    let apply_flip_lr = rng.gen::<f64>() < config.flip_lr_prob;
    let apply_flip_ud = rng.gen::<f64>() < config.flip_ud_prob;
    let apply_flip_polarity = rng.gen::<f64>() < config.flip_polarity_prob;

    debug!(
        "Applying transformations: flip_lr={}, flip_ud={}, flip_polarity={}",
        apply_flip_lr, apply_flip_ud, apply_flip_polarity
    );

    // Apply transformations to all events
    let mut transformed_events = Vec::with_capacity(events.len());

    for event in events {
        let mut new_x = event.x;
        let mut new_y = event.y;
        let mut new_polarity = event.polarity;

        // Apply horizontal flip: x_new = sensor_width - 1 - x_old
        if apply_flip_lr {
            new_x = config
                .sensor_width
                .saturating_sub(1)
                .saturating_sub(event.x);
        }

        // Apply vertical flip: y_new = sensor_height - 1 - y_old
        if apply_flip_ud {
            new_y = config
                .sensor_height
                .saturating_sub(1)
                .saturating_sub(event.y);
        }

        // Apply polarity reversal: polarity_new = !polarity_old
        if apply_flip_polarity {
            new_polarity = !event.polarity;
        }

        // Create transformed event (timestamp remains unchanged)
        let transformed_event = Event {
            t: event.t,
            x: new_x,
            y: new_y,
            polarity: new_polarity,
        };

        transformed_events.push(transformed_event);
    }

    let processing_time = start_time.elapsed().as_secs_f64();

    info!(
        "Geometric transforms applied: {} events transformed in {:.3}s (LR={}, UD={}, Pol={})",
        events.len(),
        processing_time,
        apply_flip_lr,
        apply_flip_ud,
        apply_flip_polarity
    );

    Ok(transformed_events)
}

/// Apply geometric transforms using Polars operations
#[cfg(feature = "polars")]
#[instrument(skip(df), level = "debug")]
pub fn apply_geometric_transforms_polars(
    df: LazyFrame,
    config: &GeometricTransformAugmentation,
) -> PolarsResult<LazyFrame> {
    debug!("Applying geometric transforms with Polars: {:?}", config);

    // If no transformations are enabled, return original DataFrame
    if !config.is_enabled() {
        debug!("No geometric transformations enabled");
        return Ok(df);
    }

    // Initialize RNG
    let mut rng = if let Some(seed) = config.seed {
        rand::rngs::StdRng::seed_from_u64(seed)
    } else {
        rand::rngs::StdRng::from_entropy()
    };

    // Determine which transformations to apply (once for entire sequence)
    let apply_flip_lr = rng.gen::<f64>() < config.flip_lr_prob;
    let apply_flip_ud = rng.gen::<f64>() < config.flip_ud_prob;
    let apply_flip_polarity = rng.gen::<f64>() < config.flip_polarity_prob;

    debug!(
        "Polars transformations: flip_lr={}, flip_ud={}, flip_polarity={}",
        apply_flip_lr, apply_flip_ud, apply_flip_polarity
    );

    let mut result_df = df;

    // Apply horizontal flip transformation
    if apply_flip_lr {
        result_df = result_df
            .with_columns([(lit(config.sensor_width as i64 - 1) - col(COL_X)).alias(COL_X)]);
    }

    // Apply vertical flip transformation
    if apply_flip_ud {
        result_df = result_df
            .with_columns([(lit(config.sensor_height as i64 - 1) - col(COL_Y)).alias(COL_Y)]);
    }

    // Apply polarity reversal transformation
    if apply_flip_polarity {
        result_df = result_df.with_columns([(lit(1) - col(COL_POLARITY)).alias(COL_POLARITY)]);
    }

    Ok(result_df)
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
        ]
    }

    #[test]
    fn test_geometric_transform_creation() {
        let transform = GeometricTransformAugmentation::new(640, 480)
            .with_flip_lr_probability(0.5)
            .with_flip_ud_probability(0.5)
            .with_flip_polarity_probability(0.3);
        assert_eq!(transform.flip_lr_prob, 0.5);
        assert_eq!(transform.flip_ud_prob, 0.5);
        assert_eq!(transform.flip_polarity_prob, 0.3);
        assert_eq!(transform.sensor_width, 640);
        assert_eq!(transform.sensor_height, 480);
    }

    #[test]
    fn test_validation() {
        // Valid configuration
        let valid_config = GeometricTransformAugmentation::new(640, 480)
            .with_flip_lr_probability(0.5)
            .with_flip_ud_probability(0.5)
            .with_flip_polarity_probability(0.3);
        assert!(valid_config.validate().is_ok());

        // Invalid probability
        let invalid_prob = GeometricTransformAugmentation::new(640, 480)
            .with_flip_lr_probability(1.5)
            .with_flip_ud_probability(0.5)
            .with_flip_polarity_probability(0.3);
        assert!(invalid_prob.validate().is_err());

        // Invalid sensor size
        let invalid_size = GeometricTransformAugmentation::new(0, 480)
            .with_flip_lr_probability(0.5)
            .with_flip_ud_probability(0.5)
            .with_flip_polarity_probability(0.3);
        assert!(invalid_size.validate().is_err());
    }

    #[test]
    fn test_no_transform() {
        let events = create_test_events();
        let transform = GeometricTransformAugmentation::new(640, 480); // All probabilities default to 0.0
        let result = transform.apply(&events).unwrap();

        // Events should be unchanged when all probabilities are 0
        assert_eq!(result.len(), events.len());
        for (original, transformed) in events.iter().zip(result.iter()) {
            assert_eq!(original.t, transformed.t);
            assert_eq!(original.x, transformed.x);
            assert_eq!(original.y, transformed.y);
            assert_eq!(original.polarity, transformed.polarity);
        }
    }

    #[test]
    fn test_deterministic_with_seed() {
        let events = create_test_events();
        let transform1 = GeometricTransformAugmentation::new(640, 480)
            .with_flip_lr_probability(1.0)
            .with_flip_ud_probability(1.0)
            .with_flip_polarity_probability(1.0)
            .with_seed(42);
        let transform2 = GeometricTransformAugmentation::new(640, 480)
            .with_flip_lr_probability(1.0)
            .with_flip_ud_probability(1.0)
            .with_flip_polarity_probability(1.0)
            .with_seed(42);

        let result1 = transform1.apply(&events).unwrap();
        let result2 = transform2.apply(&events).unwrap();

        // Results should be identical with same seed
        assert_eq!(result1.len(), result2.len());
        for (r1, r2) in result1.iter().zip(result2.iter()) {
            assert_eq!(r1.t, r2.t);
            assert_eq!(r1.x, r2.x);
            assert_eq!(r1.y, r2.y);
            assert_eq!(r1.polarity, r2.polarity);
        }
    }
}
