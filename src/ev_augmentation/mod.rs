//! Event augmentation module for evlib
//!
//! This module provides data augmentation operations for event camera data,
//! inspired by the tonic library but implemented in Rust for maximum performance.
//! These augmentations are essential for training neural networks and improving
//! model generalization.
//!
//! # Features
//!
//! - **Spatial augmentations**: spatial jitter, random cropping
//! - **Temporal augmentations**: time jitter, time skew, temporal reversal
//! - **Noise injection**: uniform noise, Gaussian noise
//! - **Event dropping**: drop by probability, time, or area
//! - **Geometric transformations**: flip, rotate, scale
//!
//! # Performance
//!
//! All augmentation operations are optimized for:
//! - Vectorized processing using SIMD instructions
//! - Memory-efficient streaming for large datasets
//! - Parallel processing using Rayon
//! - Zero-copy operations where possible
//!
//! # Usage
//!
//! ```rust
//! use evlib::ev_augmentation::{AugmentationConfig, augment_events};
//! use evlib::{Event, Events};
//!
//! // Create sample events
//! let events = vec![
//!     Event { t: 1.0, x: 100, y: 200, polarity: true },
//!     Event { t: 2.0, x: 150, y: 250, polarity: false },
//! ];
//!
//! // Configure augmentations
//! let config = AugmentationConfig::new()
//!     .with_spatial_jitter(1.0, 1.0)
//!     .with_time_jitter(1000.0);
//!
//! // Apply augmentations
//! let augmented_events = augment_events(&events, &config)?;
//! ```

// Removed: use crate::{Event, Events}; - legacy types no longer exist
use std::fmt;

#[cfg(feature = "tracing")]
use tracing::{debug, info};

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

// Column names for DataFrame consistency
#[cfg(feature = "polars")]
pub const COL_X: &str = "x";
#[cfg(feature = "polars")]
pub const COL_Y: &str = "y";
#[cfg(feature = "polars")]
pub const COL_T: &str = "t";
#[cfg(feature = "polars")]
pub const COL_POLARITY: &str = "polarity";

/// DataFrame-first augmentation function that applies augmentations entirely using LazyFrame operations
///
/// This function processes events through a pipeline of augmentations based on the provided
/// configuration using Polars LazyFrame operations throughout. It provides significant
/// performance improvements over the Vec<Event> approach.
///
/// # Arguments
///
/// * `df` - Input LazyFrame containing event data
/// * `config` - Augmentation configuration specifying which augmentations to apply
///
/// # Returns
///
/// Augmented LazyFrame with all transformations applied
///
/// # Example
///
/// ```rust
/// use polars::prelude::*;
/// use evlib::ev_augmentation::{AugmentationConfig, augment_events_dataframe};
///
/// let events_df = events_to_dataframe(&events)?.lazy();
/// let config = AugmentationConfig::new()
///     .with_spatial_jitter(1.0, 1.0)
///     .with_time_jitter(1000.0);
/// let augmented_df = augment_events_dataframe(events_df, &config)?;
/// ```
#[cfg(feature = "polars")]
pub fn augment_events_dataframe(
    df: LazyFrame,
    config: &AugmentationConfig,
) -> PolarsResult<LazyFrame> {
    debug!(
        "Applying DataFrame-first augmentation pipeline: {:?}",
        config
    );

    // Apply augmentations in order using DataFrame-native implementations
    let mut augmented_df = df;

    // 1. Spatial jitter - modifies spatial coordinates
    if let Some(spatial_jitter) = &config.spatial_jitter {
        augmented_df = spatial_jitter
            .apply_to_dataframe(augmented_df)
            .map_err(|e| {
                PolarsError::ComputeError(format!("Spatial jitter error: {}", e).into())
            })?;
    }

    // 2. Geometric transforms - modifies spatial coordinates and polarity
    if let Some(geometric_transforms) = &config.geometric_transforms {
        augmented_df = geometric_transforms
            .apply_to_dataframe(augmented_df)
            .map_err(|e| {
                PolarsError::ComputeError(format!("Geometric transforms error: {}", e).into())
            })?;
    }

    // 3. Time jitter - modifies timestamps
    if let Some(time_jitter) = &config.time_jitter {
        augmented_df = time_jitter
            .apply_to_dataframe(augmented_df)
            .map_err(|e| PolarsError::ComputeError(format!("Time jitter error: {}", e).into()))?;
    }

    // 4. Event dropping - removes events
    if let Some(drop_event) = &config.drop_event {
        augmented_df = drop_event
            .apply_to_dataframe(augmented_df)
            .map_err(|e| PolarsError::ComputeError(format!("Drop event error: {}", e).into()))?;
    }

    // 5. Drop by time - removes events in time interval
    if let Some(drop_time) = &config.drop_time {
        augmented_df = drop_time
            .apply_to_dataframe(augmented_df)
            .map_err(|e| PolarsError::ComputeError(format!("Drop time error: {}", e).into()))?;
    }

    // 6. Drop by area - removes events in spatial area
    if let Some(drop_area) = &config.drop_area {
        augmented_df = drop_area
            .apply_to_dataframe(augmented_df)
            .map_err(|e| PolarsError::ComputeError(format!("Drop area error: {}", e).into()))?;
    }

    // 7. Uniform noise - adds noise events (should be last)
    if let Some(uniform_noise) = &config.uniform_noise {
        augmented_df = uniform_noise
            .apply_to_dataframe(augmented_df)
            .map_err(|e| PolarsError::ComputeError(format!("Uniform noise error: {}", e).into()))?;
    }

    info!("DataFrame-first augmentation pipeline completed");
    Ok(augmented_df)
}

// Sub-modules
pub mod config;
pub mod crop;
pub mod decimate;
pub mod drop_event;
pub mod geometric_transforms;
pub mod python;
pub mod spatial_jitter;
pub mod time_jitter;
pub mod time_reversal;
pub mod time_skew;
pub mod uniform_noise;

// Re-export core types and functions for convenience
pub use config::{AugmentationConfig, AugmentationError, AugmentationResult, Validatable};
pub use crop::{CenterCropAugmentation, RandomCropAugmentation};
pub use decimate::DecimateAugmentation;
pub use drop_event::{DropAreaAugmentation, DropEventAugmentation, DropTimeAugmentation};
pub use geometric_transforms::GeometricTransformAugmentation;
pub use spatial_jitter::SpatialJitterAugmentation;
pub use time_jitter::TimeJitterAugmentation;
pub use time_reversal::TimeReversalAugmentation;
pub use time_skew::TimeSkewAugmentation;
pub use uniform_noise::UniformNoiseAugmentation;

/* Commented out - legacy Event/Events types no longer exist
/// Trait for individual augmentation implementations
pub trait SingleAugmentation {
    /// Apply this augmentation to a set of events
    fn apply(&self, events: &Events) -> AugmentationResult<Events>;

    /// Get a description of this augmentation
    fn description(&self) -> String;

    /// Check if this augmentation is enabled/configured
    fn is_enabled(&self) -> bool {
        true
    }
}
*/

/* Commented out - legacy Events type no longer exists
/// Main augmentation function that applies a comprehensive augmentation configuration
///
/// This function processes events through a pipeline of augmentations based on the provided
/// configuration. It supports both streaming and batch processing modes.
///
/// # Arguments
///
/// * `events` - Input events to augment
/// * `config` - Augmentation configuration specifying which augmentations to apply
///
/// # Returns
///
/// * `AugmentationResult<Events>` - Augmented events or error
///
/// # Example
///
/// ```rust
/// use evlib::ev_augmentation::{AugmentationConfig, augment_events};
/// use evlib::{Event, Events};
///
/// let events = vec![
///     Event { t: 1.0, x: 100, y: 200, polarity: true },
///     Event { t: 2.0, x: 150, y: 250, polarity: false },
/// ];
///
/// let config = AugmentationConfig::new()
///     .with_spatial_jitter(1.0, 1.0)
///     .with_time_jitter(1000.0);
///
/// let augmented = augment_events(&events, &config)?;
/// ```
pub fn augment_events(events: &Events, config: &AugmentationConfig) -> AugmentationResult<Events> {
    // Use Polars-first implementation when available for better performance
    #[cfg(feature = "polars")]
    {
        augment_events_polars(events, config)
    }

    // Fallback to Vec<Event> implementation when Polars is not available
    #[cfg(not(feature = "polars"))]
    {
        let mut augmented_events = events.clone();

        // Apply augmentations in order
        // 1. Spatial jitter - modifies spatial coordinates
        if let Some(spatial_jitter) = &config.spatial_jitter {
            augmented_events = spatial_jitter.apply(&augmented_events)?;
        }

        // 2. Geometric transforms - modifies spatial coordinates and polarity
        if let Some(geometric_transforms) = &config.geometric_transforms {
            augmented_events = geometric_transforms.apply(&augmented_events)?;
        }

        // 3. Center crop - crops events to centered region
        if let Some(center_crop) = &config.center_crop {
            augmented_events = center_crop.apply(&augmented_events)?;
        }

        // 4. Random crop - crops events to random region
        if let Some(random_crop) = &config.random_crop {
            augmented_events = random_crop.apply(&augmented_events)?;
        }

        // 5. Time jitter - modifies timestamps
        if let Some(time_jitter) = &config.time_jitter {
            augmented_events = time_jitter.apply(&augmented_events)?;
        }

        // 6. Time skew - linear transformation of timestamps
        if let Some(time_skew) = &config.time_skew {
            augmented_events = time_skew.apply(&augmented_events)?;
        }

        // 7. Time reversal - reverses temporal order
        if let Some(time_reversal) = &config.time_reversal {
            augmented_events = time_reversal.apply(&augmented_events)?;
        }

        // 8. Event dropping - removes events
        if let Some(drop_event) = &config.drop_event {
            augmented_events = drop_event.apply(&augmented_events)?;
        }

        // 9. Drop by time - removes events in time interval
        if let Some(drop_time) = &config.drop_time {
            augmented_events = drop_time.apply(&augmented_events)?;
        }

        // 10. Drop by area - removes events in spatial area
        if let Some(drop_area) = &config.drop_area {
            augmented_events = drop_area.apply(&augmented_events)?;
        }

        // 11. Decimate - filters events per pixel
        if let Some(decimate) = &config.decimate {
            augmented_events = decimate.apply(&augmented_events)?;
        }

        // 12. Uniform noise - adds noise events (should be last)
        if let Some(uniform_noise) = &config.uniform_noise {
            augmented_events = uniform_noise.apply(&augmented_events)?;
        }

        Ok(augmented_events)
    }
}
*/

/* Commented out - legacy Events type no longer exists
/// Polars-first augmentation function that applies augmentations entirely using LazyFrame operations
///
/// This function processes events through a pipeline of augmentations based on the provided
/// configuration using Polars LazyFrame operations throughout. It provides significant
/// performance improvements over the Vec<Event> approach.
#[cfg(feature = "polars")]
pub fn augment_events_polars(
    events: &Events,
    config: &AugmentationConfig,
) -> AugmentationResult<Events> {
    // Convert input Vec<Event> to LazyFrame once at the beginning
    let df = crate::events_to_dataframe(events)
        .map_err(|e| {
            AugmentationError::ProcessingError(format!("DataFrame conversion error: {}", e))
        })?
        .lazy();

    // Apply augmentations in order using Polars-first implementations
    let mut augmented_df = df;

    // 1. Spatial jitter - modifies spatial coordinates
    if let Some(spatial_jitter) = &config.spatial_jitter {
        augmented_df = spatial_jitter::apply_spatial_jitter_polars(augmented_df, spatial_jitter)
            .map_err(|e| {
                AugmentationError::ProcessingError(format!("Spatial jitter error: {}", e))
            })?;
    }

    // 2. Geometric transforms - modifies spatial coordinates and polarity
    if let Some(geometric_transforms) = &config.geometric_transforms {
        augmented_df = geometric_transforms::apply_geometric_transforms_polars(
            augmented_df,
            geometric_transforms,
        )
        .map_err(|e| {
            AugmentationError::ProcessingError(format!("Geometric transforms error: {}", e))
        })?;
    }

    // 3. Center crop - crops to center region and remaps coordinates
    if let Some(center_crop) = &config.center_crop {
        augmented_df = crop::apply_center_crop_polars(augmented_df, center_crop)
            .map_err(|e| AugmentationError::ProcessingError(format!("Center crop error: {}", e)))?;
    }

    // 4. Random crop - crops to random region and remaps coordinates
    if let Some(random_crop) = &config.random_crop {
        augmented_df = crop::apply_random_crop_polars(augmented_df, random_crop)
            .map_err(|e| AugmentationError::ProcessingError(format!("Random crop error: {}", e)))?;
    }

    // 5. Time jitter - modifies timestamps
    if let Some(time_jitter) = &config.time_jitter {
        augmented_df = time_jitter::apply_time_jitter_polars(augmented_df, time_jitter)
            .map_err(|e| AugmentationError::ProcessingError(format!("Time jitter error: {}", e)))?;
    }

    // 6. Time skew - linear transformation of timestamps
    if let Some(time_skew) = &config.time_skew {
        augmented_df = time_skew::apply_time_skew_polars(augmented_df, time_skew)
            .map_err(|e| AugmentationError::ProcessingError(format!("Time skew error: {}", e)))?;
    }

    // 7. Time reversal - reverses temporal order
    if let Some(time_reversal) = &config.time_reversal {
        augmented_df = time_reversal::apply_time_reversal_polars(augmented_df, time_reversal)
            .map_err(|e| {
                AugmentationError::ProcessingError(format!("Time reversal error: {}", e))
            })?;
    }

    // 8. Event dropping - removes events
    if let Some(drop_event) = &config.drop_event {
        augmented_df = drop_event::apply_drop_event_polars(augmented_df, drop_event)
            .map_err(|e| AugmentationError::ProcessingError(format!("Drop event error: {}", e)))?;
    }

    // 9. Drop by time - removes events in time interval
    if let Some(drop_time) = &config.drop_time {
        augmented_df = drop_event::apply_drop_time_polars(augmented_df, drop_time)
            .map_err(|e| AugmentationError::ProcessingError(format!("Drop time error: {}", e)))?;
    }

    // 10. Drop by area - removes events in spatial area
    if let Some(drop_area) = &config.drop_area {
        augmented_df = drop_event::apply_drop_area_polars(augmented_df, drop_area)
            .map_err(|e| AugmentationError::ProcessingError(format!("Drop area error: {}", e)))?;
    }

    // 11. Decimate - filters events per pixel
    if let Some(decimate) = &config.decimate {
        augmented_df = decimate::apply_decimate_polars(augmented_df, decimate)
            .map_err(|e| AugmentationError::ProcessingError(format!("Decimate error: {}", e)))?;
    }

    // 12. Uniform noise - adds noise events (should be last)
    if let Some(uniform_noise) = &config.uniform_noise {
        augmented_df = uniform_noise::apply_uniform_noise_polars(augmented_df, uniform_noise)
            .map_err(|e| {
                AugmentationError::ProcessingError(format!("Uniform noise error: {}", e))
            })?;
    }

    // Only convert back to Vec<Event> at the very end for legacy compatibility
    let result_df = augmented_df.collect().map_err(|e| {
        AugmentationError::ProcessingError(format!("LazyFrame collection error: {}", e))
    })?;

    dataframe_to_events(&result_df)
}

/// Convert DataFrame back to Events vector for legacy compatibility
#[cfg(feature = "polars")]
fn dataframe_to_events(df: &DataFrame) -> AugmentationResult<Events> {
    let height = df.height();
    let mut events = Vec::with_capacity(height);

    let x_series = df
        .column(COL_X)
        .map_err(|e| AugmentationError::ProcessingError(format!("Missing x column: {}", e)))?;
    let y_series = df
        .column(COL_Y)
        .map_err(|e| AugmentationError::ProcessingError(format!("Missing y column: {}", e)))?;
    let t_series = df
        .column(COL_T)
        .map_err(|e| AugmentationError::ProcessingError(format!("Missing t column: {}", e)))?;
    let p_series = df.column(COL_POLARITY).map_err(|e| {
        AugmentationError::ProcessingError(format!("Missing polarity column: {}", e))
    })?;

    let x_values = x_series
        .i64()
        .map_err(|e| AugmentationError::ProcessingError(format!("X column type error: {}", e)))?;
    let y_values = y_series
        .i64()
        .map_err(|e| AugmentationError::ProcessingError(format!("Y column type error: {}", e)))?;
    let t_values = t_series
        .f64()
        .map_err(|e| AugmentationError::ProcessingError(format!("T column type error: {}", e)))?;
    let p_values = p_series.i64().map_err(|e| {
        AugmentationError::ProcessingError(format!("Polarity column type error: {}", e))
    })?;

    for i in 0..height {
        let x = x_values
            .get(i)
            .ok_or_else(|| AugmentationError::ProcessingError("X value missing".to_string()))?
            as u16;
        let y = y_values
            .get(i)
            .ok_or_else(|| AugmentationError::ProcessingError("Y value missing".to_string()))?
            as u16;
        let t = t_values
            .get(i)
            .ok_or_else(|| AugmentationError::ProcessingError("T value missing".to_string()))?;
        let polarity = p_values.get(i).ok_or_else(|| {
            AugmentationError::ProcessingError("Polarity value missing".to_string())
        })? > 0;

        events.push(Event { t, x, y, polarity });
    }

    Ok(events)
}
*/

/// Statistics about augmentation operations
#[derive(Debug, Clone)]
pub struct AugmentationStats {
    /// Original number of events
    pub input_count: usize,
    /// Final number of events after augmentation
    pub output_count: usize,
    /// Number of events added
    pub added_count: usize,
    /// Number of events removed
    pub removed_count: usize,
    /// Processing time in seconds
    pub processing_time: f64,
    /// Events processed per second
    pub throughput: f64,
}

impl AugmentationStats {
    /// Create new augmentation statistics
    pub fn new(input_count: usize, output_count: usize, processing_time: f64) -> Self {
        let added_count = output_count.saturating_sub(input_count);
        let removed_count = input_count.saturating_sub(output_count);
        let throughput = if processing_time > 0.0 {
            input_count as f64 / processing_time
        } else {
            0.0
        };

        Self {
            input_count,
            output_count,
            added_count,
            removed_count,
            processing_time,
            throughput,
        }
    }
}

impl fmt::Display for AugmentationStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Augmentation Stats: {} -> {} events ({} added, {} removed) in {:.3}s ({:.0} events/s)",
            self.input_count,
            self.output_count,
            self.added_count,
            self.removed_count,
            self.processing_time,
            self.throughput
        )
    }
}

/* Commented out - legacy Events type no longer exists
/// Apply augmentation with statistics collection
pub fn augment_events_with_stats(
    events: &Events,
    config: &AugmentationConfig,
) -> AugmentationResult<(Events, AugmentationStats)> {
    let start_time = std::time::Instant::now();
    let input_count = events.len();

    let augmented_events = augment_events(events, config)?;

    let processing_time = start_time.elapsed().as_secs_f64();
    let output_count = augmented_events.len();
    let stats = AugmentationStats::new(input_count, output_count, processing_time);

    Ok((augmented_events, stats))
}
*/

/* Commented out - legacy Event/Events types no longer exist
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
        let config = AugmentationConfig::new();
        let augmented = augment_events(&events, &config).unwrap();
        assert_eq!(augmented.len(), events.len());
    }

    #[test]
    fn test_augmentation_stats() {
        let events = create_test_events();
        let config = AugmentationConfig::new().with_drop_event(0.2); // Drop 20% of events
        let (augmented, stats) = augment_events_with_stats(&events, &config).unwrap();

        assert_eq!(stats.input_count, 5);
        assert!(stats.output_count <= 5);
        assert!(stats.removed_count >= 0);
    }
}
*/
