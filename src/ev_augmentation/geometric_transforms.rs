//! Polars-first geometric transformation augmentations for event camera data
//!
//! This module provides geometric transformation functionality using Polars DataFrames
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
//! - Vectorized operations: All transformations use SIMD-optimized Polars operations
//! - Memory efficiency: No intermediate Vec<Event> allocations
//! - Query optimization: Polars optimizes the entire augmentation pipeline
//!
//! # Transformations
//!
//! - Horizontal flip (left-right)
//! - Vertical flip (up-down)
//! - Polarity flip
//!
//! # Example
//!
//! ```rust
//! use polars::prelude::*;
//! use evlib::ev_augmentation::geometric_transforms::*;
//!
//! // Convert events to LazyFrame once
//! let events_df = events_to_dataframe(&events)?.lazy();
//!
//! // Apply geometric transformations with Polars expressions
//! let config = GeometricTransformAugmentation::new(640, 480)
//!     .with_flip_lr_probability(0.5)
//!     .with_flip_polarity_probability(0.3);
//! let transformed = apply_geometric_transforms(events_df, &config)?;
//! ```

use crate::ev_augmentation::{AugmentationError, AugmentationResult, Validatable};
// Removed: use crate::{Event, Events}; - legacy types no longer exist
use rand::{Rng, SeedableRng};
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

// Polars column names for event data consistency
#[cfg(feature = "polars")]
pub const COL_X: &str = "x";
#[cfg(feature = "polars")]
pub const COL_Y: &str = "y";
#[cfg(feature = "polars")]
pub const COL_T: &str = "t";
#[cfg(feature = "polars")]
pub const COL_POLARITY: &str = "polarity";

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

    /// Apply geometric transforms directly to DataFrame (recommended approach)
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
    /// Transformed LazyFrame with geometric transforms applied
    #[cfg(feature = "polars")]
    pub fn apply_to_dataframe(&self, df: LazyFrame) -> PolarsResult<LazyFrame> {
        apply_geometric_transforms(df, self)
    }

    /// Apply geometric transforms directly to DataFrame and return DataFrame
    ///
    /// Convenience method that applies transforms and collects the result.
    ///
    /// # Arguments
    ///
    /// * `df` - Input DataFrame containing event data
    ///
    /// # Returns
    ///
    /// Transformed DataFrame with geometric transforms applied
    #[cfg(feature = "polars")]
    pub fn apply_to_dataframe_eager(&self, df: DataFrame) -> PolarsResult<DataFrame> {
        apply_geometric_transforms(df.lazy(), self)?.collect()
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

// impl SingleAugmentation for GeometricTransformAugmentation {
//     #[cfg_attr(feature = "tracing", instrument(skip(events), level = "debug"))]
//     fn apply(&self, events: &Events) -> AugmentationResult<Events> {
//         // Legacy Vec<Event> interface - convert to DataFrame and back
//         // This is for backward compatibility only
//         #[cfg(feature = "tracing")]
//         tracing::warn!("Using legacy Vec<Event> interface - consider using LazyFrame directly for better performance");
//         #[cfg(not(feature = "tracing"))]
//         eprintln!("[WARN] Using legacy Vec<Event> interface - consider using LazyFrame directly for better performance");
//
//         #[cfg(feature = "polars")]
//         {
//             let df = crate::events_to_dataframe(events)
//                 .map_err(|e| {
//                     AugmentationError::ProcessingError(format!(
//                         "DataFrame conversion failed: {}",
//                         e
//                     ))
//                 })?
//                 .lazy();
//
//             let transformed_df = self.apply_to_dataframe(df).map_err(|e| {
//                 AugmentationError::ProcessingError(format!("Polars transformation failed: {}", e))
//             })?;
//
//             // Convert back to Vec<Event> - this is inefficient but maintains compatibility
//             let result_df = transformed_df.collect().map_err(|e| {
//                 AugmentationError::ProcessingError(format!("LazyFrame collection failed: {}", e))
//             })?;
//
//             // Convert DataFrame back to Events
//             dataframe_to_events(&result_df)
//         }
//
//         #[cfg(not(feature = "polars"))]
//         {
//             geometric_transforms(events, self)
//         }
//     }
//
//     fn description(&self) -> String {
//         self.description()
//     }
// }

// /// Apply geometric transforms to events
// ///
// /// # Arguments
// ///
// /// * `events` - Input events
// /// * `config` - Geometric transform configuration
// ///
// /// # Returns
// ///
// /// * `AugmentationResult<Events>` - Transformed events
// #[cfg_attr(feature = "tracing", instrument(skip(events), level = "debug"))]
// pub fn geometric_transforms(
//     events: &Events,
//     config: &GeometricTransformAugmentation,
// ) -> AugmentationResult<Events> {
//     let start_time = std::time::Instant::now();
//
//     if events.is_empty() {
//         debug!("No events to transform");
//         return Ok(Vec::new());
//     }
//
//     // Validate configuration
//     config.validate()?;
//
//     // If no transformations are enabled, return original events
//     if !config.is_enabled() {
//         debug!("No geometric transformations enabled");
//         return Ok(events.clone());
//     }
//
//     // Initialize RNG
//     let mut rng = if let Some(seed) = config.seed {
//         rand::rngs::StdRng::seed_from_u64(seed)
//     } else {
//         rand::rngs::StdRng::from_entropy()
//     };
//
//     // Determine which transformations to apply (once for entire sequence)
//     let apply_flip_lr = rng.gen::<f64>() < config.flip_lr_prob;
//     let apply_flip_ud = rng.gen::<f64>() < config.flip_ud_prob;
//     let apply_flip_polarity = rng.gen::<f64>() < config.flip_polarity_prob;
//
//     debug!(
//         "Applying transformations: flip_lr={}, flip_ud={}, flip_polarity={}",
//         apply_flip_lr, apply_flip_ud, apply_flip_polarity
//     );
//
//     // Apply transformations to all events
//     let mut transformed_events = Vec::with_capacity(events.len());
//
//     for event in events {
//         let mut new_x = event.x;
//         let mut new_y = event.y;
//         let mut new_polarity = event.polarity;
//
//         // Apply horizontal flip: x_new = sensor_width - 1 - x_old
//         if apply_flip_lr {
//             new_x = config
//                 .sensor_width
//                 .saturating_sub(1)
//                 .saturating_sub(event.x);
//         }
//
//         // Apply vertical flip: y_new = sensor_height - 1 - y_old
//         if apply_flip_ud {
//             new_y = config
//                 .sensor_height
//                 .saturating_sub(1)
//                 .saturating_sub(event.y);
//         }
//
//         // Apply polarity reversal: polarity_new = !polarity_old
//         if apply_flip_polarity {
//             new_polarity = !event.polarity;
//         }
//
//         // Create transformed event (timestamp remains unchanged)
//         let transformed_event = Event {
//             t: event.t,
//             x: new_x,
//             y: new_y,
//             polarity: new_polarity,
//         };
//
//         transformed_events.push(transformed_event);
//     }
//
//     let processing_time = start_time.elapsed().as_secs_f64();
//
//     info!(
//         "Geometric transforms applied: {} events transformed in {:.3}s (LR={}, UD={}, Pol={})",
//         events.len(),
//         processing_time,
//         apply_flip_lr,
//         apply_flip_ud,
//         apply_flip_polarity
//     );
//
//     Ok(transformed_events)
// }

/// Apply geometric transforms using Polars expressions
///
/// This is the main geometric transformation function that works entirely with Polars
/// operations for maximum performance. It applies probabilistic transforms using
/// vectorized operations.
///
/// # Arguments
///
/// * `df` - Input LazyFrame containing event data
/// * `config` - Geometric transformation configuration
///
/// # Returns
///
/// Transformed LazyFrame with geometric transforms applied
///
/// # Example
///
/// ```rust
/// use polars::prelude::*;
/// use evlib::ev_augmentation::geometric_transforms::*;
///
/// let events_df = events_to_dataframe(&events)?.lazy();
/// let config = GeometricTransformAugmentation::new(640, 480)
///     .with_flip_lr_probability(0.5);
/// let transformed = apply_geometric_transforms(events_df, &config)?;
/// ```
#[cfg(feature = "polars")]
#[cfg_attr(feature = "tracing", instrument(skip(df), level = "debug"))]
pub fn apply_geometric_transforms(
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

    info!(
        "Applying geometric transforms: LR={}, UD={}, Pol={}",
        apply_flip_lr, apply_flip_ud, apply_flip_polarity
    );

    let mut result_df = df;

    // Apply horizontal flip transformation: x_new = sensor_width - 1 - x_old
    if apply_flip_lr {
        result_df = result_df
            .with_columns([(lit(config.sensor_width as i64 - 1) - col(COL_X)).alias(COL_X)]);
    }

    // Apply vertical flip transformation: y_new = sensor_height - 1 - y_old
    if apply_flip_ud {
        result_df = result_df
            .with_columns([(lit(config.sensor_height as i64 - 1) - col(COL_Y)).alias(COL_Y)]);
    }

    // Apply polarity reversal transformation: polarity_new = !polarity_old (1 - polarity)
    if apply_flip_polarity {
        result_df = result_df.with_columns([(lit(1) - col(COL_POLARITY)).alias(COL_POLARITY)]);
    }

    Ok(result_df)
}

/// Legacy Polars function for backward compatibility
#[cfg(feature = "polars")]
pub fn apply_geometric_transforms_polars(
    df: LazyFrame,
    config: &GeometricTransformAugmentation,
) -> PolarsResult<LazyFrame> {
    apply_geometric_transforms(df, config)
}

/// Apply geometric transforms directly to LazyFrame - DataFrame-native convenience functions
///
/// These functions apply geometric transformations directly to a LazyFrame for optimal performance.
/// Use these instead of the legacy Vec<Event> versions when possible.

#[cfg(feature = "polars")]
pub fn apply_flip_lr_df(df: LazyFrame, sensor_width: u16) -> PolarsResult<LazyFrame> {
    let config =
        GeometricTransformAugmentation::new(sensor_width, 480).with_flip_lr_probability(1.0);
    apply_geometric_transforms(df, &config)
}

#[cfg(feature = "polars")]
pub fn apply_flip_ud_df(df: LazyFrame, sensor_height: u16) -> PolarsResult<LazyFrame> {
    let config =
        GeometricTransformAugmentation::new(640, sensor_height).with_flip_ud_probability(1.0);
    apply_geometric_transforms(df, &config)
}

#[cfg(feature = "polars")]
pub fn apply_flip_polarity_df(df: LazyFrame) -> PolarsResult<LazyFrame> {
    let config = GeometricTransformAugmentation::new(640, 480).with_flip_polarity_probability(1.0);
    apply_geometric_transforms(df, &config)
}

// /// Helper function to convert DataFrame back to Events (for legacy compatibility)
// #[cfg(feature = "polars")]
// fn dataframe_to_events(df: &DataFrame) -> AugmentationResult<Events> {
//     let height = df.height();
//     let mut events = Vec::with_capacity(height);
//
//     let x_series = df
//         .column(COL_X)
//         .map_err(|e| AugmentationError::ProcessingError(format!("Missing x column: {}", e)))?;
//     let y_series = df
//         .column(COL_Y)
//         .map_err(|e| AugmentationError::ProcessingError(format!("Missing y column: {}", e)))?;
//     let t_series = df
//         .column(COL_T)
//         .map_err(|e| AugmentationError::ProcessingError(format!("Missing t column: {}", e)))?;
//     let p_series = df.column(COL_POLARITY).map_err(|e| {
//         AugmentationError::ProcessingError(format!("Missing polarity column: {}", e))
//     })?;
//
//     let x_values = x_series
//         .i64()
//         .map_err(|e| AugmentationError::ProcessingError(format!("X column type error: {}", e)))?;
//     let y_values = y_series
//         .i64()
//         .map_err(|e| AugmentationError::ProcessingError(format!("Y column type error: {}", e)))?;
//     let t_values = t_series
//         .f64()
//         .map_err(|e| AugmentationError::ProcessingError(format!("T column type error: {}", e)))?;
//     let p_values = p_series.i64().map_err(|e| {
//         AugmentationError::ProcessingError(format!("Polarity column type error: {}", e))
//     })?;
//
//     for i in 0..height {
//         let x = x_values
//             .get(i)
//             .ok_or_else(|| AugmentationError::ProcessingError("Missing x value".to_string()))?
//             as u16;
//         let y = y_values
//             .get(i)
//             .ok_or_else(|| AugmentationError::ProcessingError("Missing y value".to_string()))?
//             as u16;
//         let t = t_values
//             .get(i)
//             .ok_or_else(|| AugmentationError::ProcessingError("Missing t value".to_string()))?;
//         let p = p_values.get(i).ok_or_else(|| {
//             AugmentationError::ProcessingError("Missing polarity value".to_string())
//         })? > 0;
//
//         events.push(Event {
//             x,
//             y,
//             t,
//             polarity: p,
//         });
//     }
//
//     Ok(events)
// }

// #[cfg(test)]
// mod tests {
//     use super::*;
//
//     fn create_test_events() -> Events {
//         vec![
//             Event {
//                 t: 1.0,
//                 x: 100,
//                 y: 200,
//                 polarity: true,
//             },
//             Event {
//                 t: 2.0,
//                 x: 150,
//                 y: 250,
//                 polarity: false,
//             },
//         ]
//     }

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

// #[test]
// fn test_no_transform() {
//     let events = create_test_events();
//     let transform = GeometricTransformAugmentation::new(640, 480); // All probabilities default to 0.0
//     let result = transform.apply(&events).unwrap();
//
//     // Events should be unchanged when all probabilities are 0
//     assert_eq!(result.len(), events.len());
//     for (original, transformed) in events.iter().zip(result.iter()) {
//         assert_eq!(original.t, transformed.t);
//         assert_eq!(original.x, transformed.x);
//         assert_eq!(original.y, transformed.y);
//         assert_eq!(original.polarity, transformed.polarity);
//     }
// }

// #[test]
// fn test_deterministic_with_seed() {
//     let events = create_test_events();
//     let transform1 = GeometricTransformAugmentation::new(640, 480)
//         .with_flip_lr_probability(1.0)
//         .with_flip_ud_probability(1.0)
//         .with_flip_polarity_probability(1.0)
//         .with_seed(42);
//     let transform2 = GeometricTransformAugmentation::new(640, 480)
//         .with_flip_lr_probability(1.0)
//         .with_flip_ud_probability(1.0)
//         .with_flip_polarity_probability(1.0)
//         .with_seed(42);
//
//     let result1 = transform1.apply(&events).unwrap();
//     let result2 = transform2.apply(&events).unwrap();
//
//     // Results should be identical with same seed
//     assert_eq!(result1.len(), result2.len());
//     for (r1, r2) in result1.iter().zip(result2.iter()) {
//         assert_eq!(r1.t, r2.t);
//         assert_eq!(r1.x, r2.x);
//         assert_eq!(r1.y, r2.y);
//         assert_eq!(r1.polarity, r2.polarity);
//     }
// }

// #[cfg(feature = "polars")]
// #[test]
// fn test_geometric_transforms_dataframe_native() -> PolarsResult<()> {
//     use crate::events_to_dataframe;
//
//     let events = create_test_events();
//     let df = events_to_dataframe(&events)?.lazy();
//     let config = GeometricTransformAugmentation::new(640, 480).with_flip_lr_probability(1.0);
//
//     let transformed_df = config.apply_to_dataframe(df)?;
//     let result = transformed_df.collect()?;
//
//     assert_eq!(result.height(), events.len());
//
//     // Check that horizontal flip was applied: x_new = 639 - x_old
//     let original_df = events_to_dataframe(&events)?;
//     let original_x: Vec<i64> = original_df
//         .column(COL_X)?
//         .i64()?
//         .into_no_null_iter()
//         .collect();
//     let transformed_x: Vec<i64> = result.column(COL_X)?.i64()?.into_no_null_iter().collect();
//
//     for (orig_x, trans_x) in original_x.iter().zip(transformed_x.iter()) {
//         assert_eq!(*trans_x, 639 - orig_x);
//     }
//
//     Ok(())
// }

// #[cfg(feature = "polars")]
// #[test]
// fn test_apply_geometric_transforms() -> PolarsResult<()> {
//     use crate::events_to_dataframe;
//
//     let events = create_test_events();
//     let df = events_to_dataframe(&events)?.lazy();
//     let config =
//         GeometricTransformAugmentation::new(640, 480).with_flip_polarity_probability(1.0);
//
//     let transformed = apply_geometric_transforms(df, &config)?;
//     let result = transformed.collect()?;
//     assert_eq!(result.height(), events.len());
//
//     // Check polarity flip was applied
//     let original_df = events_to_dataframe(&events)?;
//     let original_pol: Vec<i64> = original_df
//         .column(COL_POLARITY)?
//         .i64()?
//         .into_no_null_iter()
//         .collect();
//     let transformed_pol: Vec<i64> = result
//         .column(COL_POLARITY)?
//         .i64()?
//         .into_no_null_iter()
//         .collect();
//
//     for (orig_p, trans_p) in original_pol.iter().zip(transformed_pol.iter()) {
//         assert_eq!(*trans_p, 1 - orig_p);
//     }
//
//     Ok(())
// }

// #[cfg(feature = "polars")]
// #[test]
// fn test_geometric_transforms_convenience_functions() -> PolarsResult<()> {
//     use crate::events_to_dataframe;
//
//     let events = create_test_events();
//     let df = events_to_dataframe(&events)?.lazy();
//
//     // Test horizontal flip
//     let flipped_lr = apply_flip_lr_df(df.clone(), 640)?;
//     let result_lr = flipped_lr.collect()?;
//     assert_eq!(result_lr.height(), events.len());
//
//     // Test vertical flip
//     let flipped_ud = apply_flip_ud_df(df.clone(), 480)?;
//     let result_ud = flipped_ud.collect()?;
//     assert_eq!(result_ud.height(), events.len());
//
//     // Test polarity flip
//     let flipped_pol = apply_flip_polarity_df(df)?;
//     let result_pol = flipped_pol.collect()?;
//     assert_eq!(result_pol.height(), events.len());
//
//     Ok(())
// }

// #[cfg(feature = "polars")]
// #[test]
// fn test_geometric_transforms_legacy_compatibility() {
//     let events = create_test_events();
//     let config = GeometricTransformAugmentation::new(640, 480).with_flip_lr_probability(1.0);
//
//     let transformed = config.apply(&events).unwrap();
//     assert_eq!(transformed.len(), events.len());
//
//     // Check that transformation was applied
//     assert_eq!(transformed[0].x, 639 - events[0].x);
// }
// }
