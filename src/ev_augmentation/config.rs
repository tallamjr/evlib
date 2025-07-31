//! Configuration module for event augmentations
//!
//! This module provides configuration structures for all augmentation operations,
//! allowing flexible composition of different augmentation strategies.

use crate::ev_augmentation::{
    CenterCropAugmentation, DecimateAugmentation, DropAreaAugmentation, DropEventAugmentation,
    DropTimeAugmentation, GeometricTransformAugmentation, RandomCropAugmentation,
    SpatialJitterAugmentation, TimeJitterAugmentation, TimeReversalAugmentation,
    TimeSkewAugmentation, UniformNoiseAugmentation,
};
use std::fmt;
use thiserror::Error;

/// Result type for augmentation operations
pub type AugmentationResult<T> = Result<T, AugmentationError>;

/// Error type for augmentation operations
#[derive(Error, Debug)]
pub enum AugmentationError {
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Processing error: {0}")]
    ProcessingError(String),

    #[error("Invalid sensor size: width={0}, height={1}")]
    InvalidSensorSize(u16, u16),

    #[error("Invalid probability: {0} (must be between 0 and 1)")]
    InvalidProbability(f64),

    #[error("Invalid time range: start={0}, end={1}")]
    InvalidTimeRange(f64, f64),

    #[error("Invalid spatial bounds: x={0}-{1}, y={2}-{3}")]
    InvalidSpatialBounds(u16, u16, u16, u16),

    #[error("Insufficient events for operation: required={0}, available={1}")]
    InsufficientEvents(usize, usize),
}

/// Trait for validatable configurations
pub trait Validatable {
    /// Validate this configuration
    fn validate(&self) -> AugmentationResult<()>;
}

/// Main augmentation configuration
///
/// This struct allows composing multiple augmentation operations into a single pipeline.
/// Augmentations are applied in a specific order to ensure consistent results.
///
/// # Example
///
/// ```rust
/// use evlib::ev_augmentation::AugmentationConfig;
///
/// let config = AugmentationConfig::new()
///     .with_spatial_jitter(1.0, 1.0)
///     .with_time_jitter(1000.0)
///     .with_drop_event(0.1); // Drop 10% of events
/// ```
#[derive(Debug, Clone)]
pub struct AugmentationConfig {
    /// Spatial jitter augmentation
    pub spatial_jitter: Option<SpatialJitterAugmentation>,
    /// Geometric transformations (flips and polarity reversal)
    pub geometric_transforms: Option<GeometricTransformAugmentation>,
    /// Time jitter augmentation
    pub time_jitter: Option<TimeJitterAugmentation>,
    /// Time reversal augmentation
    pub time_reversal: Option<TimeReversalAugmentation>,
    /// Time skew augmentation
    pub time_skew: Option<TimeSkewAugmentation>,
    /// Drop event augmentation
    pub drop_event: Option<DropEventAugmentation>,
    /// Drop by time augmentation
    pub drop_time: Option<DropTimeAugmentation>,
    /// Drop by area augmentation
    pub drop_area: Option<DropAreaAugmentation>,
    /// Uniform noise augmentation
    pub uniform_noise: Option<UniformNoiseAugmentation>,
    /// Decimate augmentation
    pub decimate: Option<DecimateAugmentation>,
    /// Center crop augmentation
    pub center_crop: Option<CenterCropAugmentation>,
    /// Random crop augmentation
    pub random_crop: Option<RandomCropAugmentation>,
    /// Whether to ensure temporal order after augmentation
    pub ensure_temporal_order: bool,
    /// Whether to validate configuration before applying
    pub validate_config: bool,
    /// Random seed for reproducibility (None for random)
    pub random_seed: Option<u64>,
}

impl Default for AugmentationConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl AugmentationConfig {
    /// Create a new empty augmentation configuration
    pub fn new() -> Self {
        Self {
            spatial_jitter: None,
            time_jitter: None,
            time_reversal: None,
            time_skew: None,
            drop_event: None,
            drop_time: None,
            drop_area: None,
            uniform_noise: None,
            decimate: None,
            geometric_transforms: None,
            center_crop: None,
            random_crop: None,
            ensure_temporal_order: true,
            validate_config: true,
            random_seed: None,
        }
    }

    /// Add spatial jitter augmentation
    ///
    /// # Arguments
    ///
    /// * `var_x` - Variance for x-coordinate jitter
    /// * `var_y` - Variance for y-coordinate jitter
    pub fn with_spatial_jitter(mut self, var_x: f64, var_y: f64) -> Self {
        self.spatial_jitter = Some(SpatialJitterAugmentation::new(var_x, var_y));
        self
    }

    /// Add spatial jitter with correlation
    ///
    /// # Arguments
    ///
    /// * `var_x` - Variance for x-coordinate jitter
    /// * `var_y` - Variance for y-coordinate jitter
    /// * `sigma_xy` - Covariance between x and y
    pub fn with_spatial_jitter_correlated(mut self, var_x: f64, var_y: f64, sigma_xy: f64) -> Self {
        self.spatial_jitter =
            Some(SpatialJitterAugmentation::new(var_x, var_y).with_correlation(sigma_xy));
        self
    }

    /// Add geometric transformations
    ///
    /// # Arguments
    ///
    /// * `sensor_width` - Sensor width in pixels
    /// * `sensor_height` - Sensor height in pixels
    pub fn with_geometric_transforms(mut self, sensor_width: u16, sensor_height: u16) -> Self {
        self.geometric_transforms = Some(GeometricTransformAugmentation::new(
            sensor_width,
            sensor_height,
        ));
        self
    }

    /// Add geometric transformations with horizontal flip probability
    ///
    /// # Arguments
    ///
    /// * `sensor_width` - Sensor width in pixels
    /// * `sensor_height` - Sensor height in pixels
    /// * `flip_lr_probability` - Probability of horizontal flip (0.0 to 1.0)
    pub fn with_horizontal_flip(
        mut self,
        sensor_width: u16,
        sensor_height: u16,
        flip_lr_probability: f64,
    ) -> Self {
        self.geometric_transforms = Some(
            GeometricTransformAugmentation::new(sensor_width, sensor_height)
                .with_flip_lr_probability(flip_lr_probability),
        );
        self
    }

    /// Add geometric transformations with vertical flip probability
    ///
    /// # Arguments
    ///
    /// * `sensor_width` - Sensor width in pixels
    /// * `sensor_height` - Sensor height in pixels
    /// * `flip_ud_probability` - Probability of vertical flip (0.0 to 1.0)
    pub fn with_vertical_flip(
        mut self,
        sensor_width: u16,
        sensor_height: u16,
        flip_ud_probability: f64,
    ) -> Self {
        self.geometric_transforms = Some(
            GeometricTransformAugmentation::new(sensor_width, sensor_height)
                .with_flip_ud_probability(flip_ud_probability),
        );
        self
    }

    /// Add geometric transformations with polarity flip probability
    ///
    /// # Arguments
    ///
    /// * `sensor_width` - Sensor width in pixels
    /// * `sensor_height` - Sensor height in pixels
    /// * `flip_polarity_probability` - Probability of polarity reversal (0.0 to 1.0)
    pub fn with_polarity_flip(
        mut self,
        sensor_width: u16,
        sensor_height: u16,
        flip_polarity_probability: f64,
    ) -> Self {
        self.geometric_transforms = Some(
            GeometricTransformAugmentation::new(sensor_width, sensor_height)
                .with_flip_polarity_probability(flip_polarity_probability),
        );
        self
    }

    /// Add complete geometric transformations with all flip probabilities
    ///
    /// # Arguments
    ///
    /// * `sensor_width` - Sensor width in pixels
    /// * `sensor_height` - Sensor height in pixels
    /// * `flip_lr_probability` - Probability of horizontal flip (0.0 to 1.0)
    /// * `flip_ud_probability` - Probability of vertical flip (0.0 to 1.0)
    /// * `flip_polarity_probability` - Probability of polarity reversal (0.0 to 1.0)
    pub fn with_all_geometric_flips(
        mut self,
        sensor_width: u16,
        sensor_height: u16,
        flip_lr_probability: f64,
        flip_ud_probability: f64,
        flip_polarity_probability: f64,
    ) -> Self {
        self.geometric_transforms = Some(
            GeometricTransformAugmentation::new(sensor_width, sensor_height)
                .with_flip_lr_probability(flip_lr_probability)
                .with_flip_ud_probability(flip_ud_probability)
                .with_flip_polarity_probability(flip_polarity_probability),
        );
        self
    }

    /// Add time jitter augmentation
    ///
    /// # Arguments
    ///
    /// * `std_us` - Standard deviation in microseconds
    pub fn with_time_jitter(mut self, std_us: f64) -> Self {
        self.time_jitter = Some(TimeJitterAugmentation::new(std_us));
        self
    }

    /// Add time reversal augmentation
    ///
    /// # Arguments
    ///
    /// * `probability` - Probability of applying time reversal (0.0 to 1.0)
    pub fn with_time_reversal(mut self, probability: f64) -> Self {
        self.time_reversal = Some(TimeReversalAugmentation::new(probability));
        self
    }

    /// Add time skew augmentation
    ///
    /// # Arguments
    ///
    /// * `coefficient` - Multiplicative coefficient for timestamps
    pub fn with_time_skew(mut self, coefficient: f64) -> Self {
        self.time_skew = Some(TimeSkewAugmentation::new(coefficient));
        self
    }

    /// Add time skew with offset
    ///
    /// # Arguments
    ///
    /// * `coefficient` - Multiplicative coefficient for timestamps
    /// * `offset` - Additive offset in seconds
    pub fn with_time_skew_offset(mut self, coefficient: f64, offset: f64) -> Self {
        self.time_skew = Some(TimeSkewAugmentation::new(coefficient).with_offset(offset));
        self
    }

    /// Add drop event augmentation
    ///
    /// # Arguments
    ///
    /// * `probability` - Probability of dropping each event (0-1)
    pub fn with_drop_event(mut self, probability: f64) -> Self {
        self.drop_event = Some(DropEventAugmentation::new(probability));
        self
    }

    /// Add drop by time augmentation
    ///
    /// # Arguments
    ///
    /// * `duration_ratio` - Ratio of sequence duration to drop (0-1)
    pub fn with_drop_time(mut self, duration_ratio: f64) -> Self {
        self.drop_time = Some(DropTimeAugmentation::new(duration_ratio));
        self
    }

    /// Add drop by area augmentation
    ///
    /// # Arguments
    ///
    /// * `area_ratio` - Ratio of sensor area to drop (0-1)
    /// * `sensor_width` - Sensor width in pixels
    /// * `sensor_height` - Sensor height in pixels
    pub fn with_drop_area(
        mut self,
        area_ratio: f64,
        sensor_width: u16,
        sensor_height: u16,
    ) -> Self {
        self.drop_area = Some(DropAreaAugmentation::new(
            area_ratio,
            sensor_width,
            sensor_height,
        ));
        self
    }

    /// Add uniform noise augmentation
    ///
    /// # Arguments
    ///
    /// * `n_events` - Number of noise events to add
    /// * `sensor_width` - Sensor width in pixels
    /// * `sensor_height` - Sensor height in pixels
    pub fn with_uniform_noise(
        mut self,
        n_events: usize,
        sensor_width: u16,
        sensor_height: u16,
    ) -> Self {
        self.uniform_noise = Some(UniformNoiseAugmentation::new(
            n_events,
            sensor_width,
            sensor_height,
        ));
        self
    }

    /// Add decimate augmentation
    ///
    /// # Arguments
    ///
    /// * `n` - Keep every nth event per pixel
    pub fn with_decimate(mut self, n: usize) -> Self {
        self.decimate = Some(DecimateAugmentation::new(n));
        self
    }

    /// Add center crop augmentation
    ///
    /// # Arguments
    ///
    /// * `crop_width` - Width of cropped region
    /// * `crop_height` - Height of cropped region
    /// * `sensor_width` - Original sensor width
    /// * `sensor_height` - Original sensor height
    pub fn with_center_crop(
        mut self,
        crop_width: u16,
        crop_height: u16,
        sensor_width: u16,
        sensor_height: u16,
    ) -> Self {
        self.center_crop = Some(CenterCropAugmentation::new(
            crop_width,
            crop_height,
            sensor_width,
            sensor_height,
        ));
        self
    }

    /// Add random crop augmentation
    ///
    /// # Arguments
    ///
    /// * `crop_width` - Width of cropped region
    /// * `crop_height` - Height of cropped region
    /// * `sensor_width` - Original sensor width
    /// * `sensor_height` - Original sensor height
    pub fn with_random_crop(
        mut self,
        crop_width: u16,
        crop_height: u16,
        sensor_width: u16,
        sensor_height: u16,
    ) -> Self {
        self.random_crop = Some(RandomCropAugmentation::new(
            crop_width,
            crop_height,
            sensor_width,
            sensor_height,
        ));
        self
    }

    /// Set whether to ensure temporal order after augmentation
    pub fn with_temporal_order(mut self, ensure: bool) -> Self {
        self.ensure_temporal_order = ensure;
        self
    }

    /// Set whether to validate configuration before applying
    pub fn with_validation(mut self, validate: bool) -> Self {
        self.validate_config = validate;
        self
    }

    /// Set random seed for reproducibility
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Check if any augmentation is configured
    pub fn is_empty(&self) -> bool {
        self.spatial_jitter.is_none()
            && self.time_jitter.is_none()
            && self.time_reversal.is_none()
            && self.time_skew.is_none()
            && self.drop_event.is_none()
            && self.drop_time.is_none()
            && self.drop_area.is_none()
            && self.uniform_noise.is_none()
            && self.decimate.is_none()
            && self.geometric_transforms.is_none()
            && self.center_crop.is_none()
            && self.random_crop.is_none()
    }

    /// Get description of configured augmentations
    pub fn description(&self) -> String {
        let mut parts = Vec::new();

        if let Some(ref aug) = self.spatial_jitter {
            parts.push(format!("spatial_jitter({})", aug.description()));
        }
        if let Some(ref aug) = self.geometric_transforms {
            parts.push(format!("geometric_transforms({})", aug.description()));
        }
        if let Some(ref aug) = self.time_jitter {
            parts.push(format!("time_jitter({})", aug.description()));
        }
        if let Some(ref aug) = self.time_reversal {
            parts.push(format!("time_reversal({})", aug.description()));
        }
        if let Some(ref aug) = self.time_skew {
            parts.push(format!("time_skew({})", aug.description()));
        }
        if let Some(ref aug) = self.drop_event {
            parts.push(format!("drop_event({})", aug.description()));
        }
        if let Some(ref aug) = self.drop_time {
            parts.push(format!("drop_time({})", aug.description()));
        }
        if let Some(ref aug) = self.drop_area {
            parts.push(format!("drop_area({})", aug.description()));
        }
        if let Some(ref aug) = self.uniform_noise {
            parts.push(format!("uniform_noise({})", aug.description()));
        }
        if let Some(ref aug) = self.decimate {
            parts.push(format!("decimate({})", aug.description()));
        }
        if let Some(ref aug) = self.center_crop {
            parts.push(format!("center_crop({})", aug.description()));
        }
        if let Some(ref aug) = self.random_crop {
            parts.push(format!("random_crop({})", aug.description()));
        }

        if parts.is_empty() {
            "No augmentations configured".to_string()
        } else {
            parts.join(", ")
        }
    }
}

impl Validatable for AugmentationConfig {
    fn validate(&self) -> AugmentationResult<()> {
        // Validate individual augmentations
        if let Some(ref aug) = self.spatial_jitter {
            aug.validate()?;
        }
        if let Some(ref aug) = self.geometric_transforms {
            aug.validate()?;
        }
        if let Some(ref aug) = self.time_jitter {
            aug.validate()?;
        }
        if let Some(ref aug) = self.time_reversal {
            aug.validate()?;
        }
        if let Some(ref aug) = self.time_skew {
            aug.validate()?;
        }
        if let Some(ref aug) = self.drop_event {
            aug.validate()?;
        }
        if let Some(ref aug) = self.drop_time {
            aug.validate()?;
        }
        if let Some(ref aug) = self.drop_area {
            aug.validate()?;
        }
        if let Some(ref aug) = self.uniform_noise {
            aug.validate()?;
        }
        if let Some(ref aug) = self.decimate {
            aug.validate()?;
        }
        if let Some(ref aug) = self.center_crop {
            aug.validate()?;
        }
        if let Some(ref aug) = self.random_crop {
            aug.validate()?;
        }

        // Check for conflicting configurations
        if self.drop_event.is_some() && self.drop_time.is_some() && self.drop_area.is_some() {
            // Warning: multiple drop operations may remove more events than expected
            // This is allowed but worth noting
        }

        Ok(())
    }
}

impl fmt::Display for AugmentationConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AugmentationConfig: {}", self.description())
    }
}

/// Common augmentation presets
pub mod presets {
    use super::*;

    /// Light augmentation for validation sets
    pub fn light_augmentation() -> AugmentationConfig {
        AugmentationConfig::new()
            .with_spatial_jitter(0.5, 0.5)
            .with_time_jitter(500.0) // 0.5ms
    }

    /// Standard augmentation for training
    pub fn standard_augmentation() -> AugmentationConfig {
        AugmentationConfig::new()
            .with_spatial_jitter(1.0, 1.0)
            .with_time_jitter(1000.0) // 1ms
            .with_drop_event(0.1) // Drop 10%
    }

    /// Heavy augmentation for robust training
    pub fn heavy_augmentation() -> AugmentationConfig {
        AugmentationConfig::new()
            .with_spatial_jitter(2.0, 2.0)
            .with_time_jitter(2000.0) // 2ms
            .with_drop_event(0.2) // Drop 20%
            .with_uniform_noise(1000, 640, 480) // Add 1000 noise events
    }

    /// Temporal-only augmentation
    pub fn temporal_only() -> AugmentationConfig {
        AugmentationConfig::new()
            .with_time_jitter(1000.0)
            .with_time_reversal(0.5) // 50% chance of reversal
            .with_time_skew(1.1) // 10% time stretch
    }

    /// Spatial-only augmentation
    pub fn spatial_only() -> AugmentationConfig {
        AugmentationConfig::new()
            .with_spatial_jitter(1.0, 1.0)
            .with_drop_area(0.1, 640, 480) // Drop 10% area
    }

    /// Noise injection augmentation
    pub fn noise_injection() -> AugmentationConfig {
        AugmentationConfig::new()
            .with_uniform_noise(2000, 640, 480)
            .with_drop_event(0.05) // Small dropout to balance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_creation() {
        let config = AugmentationConfig::new();
        assert!(config.is_empty());
        assert_eq!(config.description(), "No augmentations configured");
    }

    #[test]
    fn test_config_builder() {
        let config = AugmentationConfig::new()
            .with_spatial_jitter(1.0, 1.0)
            .with_time_jitter(1000.0)
            .with_drop_event(0.1);

        assert!(!config.is_empty());
        assert!(config.spatial_jitter.is_some());
        assert!(config.time_jitter.is_some());
        assert!(config.drop_event.is_some());
    }

    #[test]
    fn test_config_validation() {
        // Valid config
        let valid_config = AugmentationConfig::new()
            .with_spatial_jitter(1.0, 1.0)
            .with_drop_event(0.5);
        assert!(valid_config.validate().is_ok());

        // Invalid config (negative variance)
        let invalid_config = AugmentationConfig::new().with_spatial_jitter(-1.0, 1.0);
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_presets() {
        let light = presets::light_augmentation();
        assert!(light.spatial_jitter.is_some());
        assert!(light.time_jitter.is_some());
        assert!(light.drop_event.is_none());

        let heavy = presets::heavy_augmentation();
        assert!(heavy.spatial_jitter.is_some());
        assert!(heavy.time_jitter.is_some());
        assert!(heavy.drop_event.is_some());
        assert!(heavy.uniform_noise.is_some());
    }
}
