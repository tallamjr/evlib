//! Event cropping augmentations
//!
//! This module implements spatial cropping operations for event camera data:
//! - CenterCrop: Crop events to center region of specified size
//! - RandomCrop: Crop events to random region of specified size
//!
//! Key features:
//! - Coordinate remapping: events are transformed to crop coordinate system
//! - Event filtering: events outside crop region are removed
//! - Polars-first implementations for high performance
//! - Reproducible seeding support for RandomCrop

use crate::ev_augmentation::{
    AugmentationError, AugmentationResult, SingleAugmentation, Validatable,
};
use crate::ev_core::{Event, Events};
use rand::{Rng, SeedableRng};
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
use crate::ev_augmentation::{COL_X, COL_Y};

#[cfg(feature = "polars")]
use polars::prelude::*;

/// Center crop augmentation
///
/// Crops events to the center region of the specified size and remaps coordinates
/// to the crop coordinate system. Events outside the crop region are removed.
///
/// Mathematical transformation:
/// - offset_x = (sensor_width - crop_width) / 2
/// - offset_y = (sensor_height - crop_height) / 2
/// - x_new = x_old - offset_x
/// - y_new = y_old - offset_y
/// - Events with x_old ∉ [offset_x, offset_x + crop_width) are removed
/// - Events with y_old ∉ [offset_y, offset_y + crop_height) are removed
#[derive(Debug, Clone)]
pub struct CenterCropAugmentation {
    /// Width of the crop region
    pub crop_width: u16,
    /// Height of the crop region
    pub crop_height: u16,
    /// Original sensor width
    pub sensor_width: u16,
    /// Original sensor height
    pub sensor_height: u16,
}

impl CenterCropAugmentation {
    /// Create a new center crop augmentation
    ///
    /// # Arguments
    ///
    /// * `crop_width` - Width of the crop region
    /// * `crop_height` - Height of the crop region
    /// * `sensor_width` - Original sensor width
    /// * `sensor_height` - Original sensor height
    pub fn new(crop_width: u16, crop_height: u16, sensor_width: u16, sensor_height: u16) -> Self {
        Self {
            crop_width,
            crop_height,
            sensor_width,
            sensor_height,
        }
    }

    /// Get the crop offset coordinates
    pub fn get_offsets(&self) -> (u16, u16) {
        let offset_x = (self.sensor_width - self.crop_width) / 2;
        let offset_y = (self.sensor_height - self.crop_height) / 2;
        (offset_x, offset_y)
    }

    /// Get the crop bounds (inclusive start, exclusive end)
    pub fn get_bounds(&self) -> (u16, u16, u16, u16) {
        let (offset_x, offset_y) = self.get_offsets();
        (
            offset_x,
            offset_x + self.crop_width,
            offset_y,
            offset_y + self.crop_height,
        )
    }

    /// Get description of this augmentation
    pub fn description(&self) -> String {
        format!(
            "{}x{} from {}x{}",
            self.crop_width, self.crop_height, self.sensor_width, self.sensor_height
        )
    }
}

impl Validatable for CenterCropAugmentation {
    fn validate(&self) -> AugmentationResult<()> {
        if self.crop_width == 0 || self.crop_height == 0 {
            return Err(AugmentationError::InvalidConfig(
                "Crop dimensions must be positive".to_string(),
            ));
        }
        if self.sensor_width == 0 || self.sensor_height == 0 {
            return Err(AugmentationError::InvalidSensorSize(
                self.sensor_width,
                self.sensor_height,
            ));
        }
        if self.crop_width > self.sensor_width || self.crop_height > self.sensor_height {
            return Err(AugmentationError::InvalidConfig(
                "Crop size cannot be larger than sensor size".to_string(),
            ));
        }
        Ok(())
    }
}

impl SingleAugmentation for CenterCropAugmentation {
    fn apply(&self, events: &Events) -> AugmentationResult<Events> {
        center_crop(events, self)
    }

    fn description(&self) -> String {
        format!("Center crop: {}", self.description())
    }
}

/// Random crop augmentation
///
/// Crops events to a randomly positioned region of the specified size and remaps coordinates
/// to the crop coordinate system. Events outside the crop region are removed.
///
/// Mathematical transformation:
/// - offset_x = random uniform from [0, sensor_width - crop_width]
/// - offset_y = random uniform from [0, sensor_height - crop_height]
/// - x_new = x_old - offset_x
/// - y_new = y_old - offset_y
/// - Events with x_old ∉ [offset_x, offset_x + crop_width) are removed
/// - Events with y_old ∉ [offset_y, offset_y + crop_height) are removed
#[derive(Debug, Clone)]
pub struct RandomCropAugmentation {
    /// Width of the crop region
    pub crop_width: u16,
    /// Height of the crop region
    pub crop_height: u16,
    /// Original sensor width
    pub sensor_width: u16,
    /// Original sensor height
    pub sensor_height: u16,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl RandomCropAugmentation {
    /// Create a new random crop augmentation
    ///
    /// # Arguments
    ///
    /// * `crop_width` - Width of the crop region
    /// * `crop_height` - Height of the crop region
    /// * `sensor_width` - Original sensor width
    /// * `sensor_height` - Original sensor height
    pub fn new(crop_width: u16, crop_height: u16, sensor_width: u16, sensor_height: u16) -> Self {
        Self {
            crop_width,
            crop_height,
            sensor_width,
            sensor_height,
            seed: None,
        }
    }

    /// Set random seed for reproducibility
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Get random crop offset coordinates
    pub fn get_random_offsets(&self, rng: &mut impl Rng) -> (u16, u16) {
        let max_offset_x = self.sensor_width - self.crop_width;
        let max_offset_y = self.sensor_height - self.crop_height;

        let offset_x = if max_offset_x > 0 {
            Uniform::new(0, max_offset_x).sample(rng)
        } else {
            0
        };

        let offset_y = if max_offset_y > 0 {
            Uniform::new(0, max_offset_y).sample(rng)
        } else {
            0
        };

        (offset_x, offset_y)
    }

    /// Get description of this augmentation
    pub fn description(&self) -> String {
        format!(
            "{}x{} from {}x{}",
            self.crop_width, self.crop_height, self.sensor_width, self.sensor_height
        )
    }
}

impl Validatable for RandomCropAugmentation {
    fn validate(&self) -> AugmentationResult<()> {
        if self.crop_width == 0 || self.crop_height == 0 {
            return Err(AugmentationError::InvalidConfig(
                "Crop dimensions must be positive".to_string(),
            ));
        }
        if self.sensor_width == 0 || self.sensor_height == 0 {
            return Err(AugmentationError::InvalidSensorSize(
                self.sensor_width,
                self.sensor_height,
            ));
        }
        if self.crop_width > self.sensor_width || self.crop_height > self.sensor_height {
            return Err(AugmentationError::InvalidConfig(
                "Crop size cannot be larger than sensor size".to_string(),
            ));
        }
        Ok(())
    }
}

impl SingleAugmentation for RandomCropAugmentation {
    fn apply(&self, events: &Events) -> AugmentationResult<Events> {
        random_crop(events, self)
    }

    fn description(&self) -> String {
        format!("Random crop: {}", self.description())
    }
}

/// Apply center crop to events
#[cfg_attr(feature = "tracing", instrument(skip(events), fields(n_events = events.len())))]
pub fn center_crop(events: &Events, config: &CenterCropAugmentation) -> AugmentationResult<Events> {
    let start_time = std::time::Instant::now();

    if events.is_empty() {
        debug!("No events to crop");
        return Ok(Vec::new());
    }

    // Validate configuration
    config.validate()?;

    let (offset_x, offset_y) = config.get_offsets();
    let (x_start, x_end, y_start, y_end) = config.get_bounds();

    // Filter and remap events
    let mut cropped_events = Vec::new();
    let mut kept_count = 0;

    for event in events {
        // Check if event is within crop bounds
        if event.x >= x_start && event.x < x_end && event.y >= y_start && event.y < y_end {
            // Remap coordinates to crop coordinate system
            let new_event = Event {
                t: event.t,
                x: event.x - offset_x,
                y: event.y - offset_y,
                polarity: event.polarity,
            };
            cropped_events.push(new_event);
            kept_count += 1;
        }
    }

    let processing_time = start_time.elapsed().as_secs_f64();

    info!(
        "Center crop applied ({}x{} at offset {},{}) : {} -> {} events in {:.3}s",
        config.crop_width,
        config.crop_height,
        offset_x,
        offset_y,
        events.len(),
        kept_count,
        processing_time
    );

    Ok(cropped_events)
}

/// Apply random crop to events
#[cfg_attr(feature = "tracing", instrument(skip(events), fields(n_events = events.len())))]
pub fn random_crop(events: &Events, config: &RandomCropAugmentation) -> AugmentationResult<Events> {
    let start_time = std::time::Instant::now();

    if events.is_empty() {
        debug!("No events to crop");
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

    let (offset_x, offset_y) = config.get_random_offsets(&mut rng);
    let x_start = offset_x;
    let x_end = offset_x + config.crop_width;
    let y_start = offset_y;
    let y_end = offset_y + config.crop_height;

    // Filter and remap events
    let mut cropped_events = Vec::new();
    let mut kept_count = 0;

    for event in events {
        // Check if event is within crop bounds
        if event.x >= x_start && event.x < x_end && event.y >= y_start && event.y < y_end {
            // Remap coordinates to crop coordinate system
            let new_event = Event {
                t: event.t,
                x: event.x - offset_x,
                y: event.y - offset_y,
                polarity: event.polarity,
            };
            cropped_events.push(new_event);
            kept_count += 1;
        }
    }

    let processing_time = start_time.elapsed().as_secs_f64();

    info!(
        "Random crop applied ({}x{} at offset {},{}) : {} -> {} events in {:.3}s",
        config.crop_width,
        config.crop_height,
        offset_x,
        offset_y,
        events.len(),
        kept_count,
        processing_time
    );

    Ok(cropped_events)
}

/// Apply center crop using Polars operations
#[cfg(feature = "polars")]
pub fn apply_center_crop_polars(
    df: LazyFrame,
    config: &CenterCropAugmentation,
) -> PolarsResult<LazyFrame> {
    let (offset_x, offset_y) = config.get_offsets();
    let (x_start, x_end, y_start, y_end) = config.get_bounds();

    // Apply vectorized bounds filtering and coordinate remapping
    let filtered_df = df
        .filter(
            col(COL_X)
                .gt_eq(lit(x_start as i64))
                .and(col(COL_X).lt(lit(x_end as i64)))
                .and(col(COL_Y).gt_eq(lit(y_start as i64)))
                .and(col(COL_Y).lt(lit(y_end as i64))),
        )
        .with_columns([
            (col(COL_X) - lit(offset_x as i64)).alias(COL_X),
            (col(COL_Y) - lit(offset_y as i64)).alias(COL_Y),
        ]);

    Ok(filtered_df)
}

/// Apply random crop using Polars operations
#[cfg(feature = "polars")]
pub fn apply_random_crop_polars(
    df: LazyFrame,
    config: &RandomCropAugmentation,
) -> PolarsResult<LazyFrame> {
    // For random operations, we need to generate the offset once
    // We'll collect the DataFrame, apply the crop, and return as LazyFrame
    let collected_df = df.collect()?;

    // Convert to events, apply random crop, and convert back
    let events = crate::ev_augmentation::dataframe_to_events(&collected_df)
        .map_err(|e| PolarsError::ComputeError(format!("Conversion error: {}", e).into()))?;

    let cropped_events = random_crop(&events, config)
        .map_err(|e| PolarsError::ComputeError(format!("Random crop error: {}", e).into()))?;

    let result_df = crate::ev_core::events_to_dataframe(&cropped_events)?;
    Ok(result_df.lazy())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_events() -> Events {
        let mut events = Vec::new();

        // Create a 5x5 grid of events at coordinates (10*i, 10*j) for i,j in 0..5
        // This gives us events at (0,0), (0,10), ..., (40,40)
        // Total: 25 events in a 50x50 region
        for i in 0..5 {
            for j in 0..5 {
                events.push(Event {
                    t: (i * 5 + j) as f64,
                    x: (i * 10) as u16,
                    y: (j * 10) as u16,
                    polarity: (i + j) % 2 == 0,
                });
            }
        }
        events
    }

    #[test]
    fn test_center_crop_augmentation() {
        let events = create_test_events();
        let config = CenterCropAugmentation::new(30, 30, 50, 50);

        let cropped = config.apply(&events).unwrap();

        // With a 30x30 crop from 50x50 sensor, offset is (10, 10)
        // Events at (10,10), (10,20), (10,30), (20,10), (20,20), (20,30), (30,10), (30,20), (30,30) should be kept
        // After remapping: (0,0), (0,10), (0,20), (10,0), (10,10), (10,20), (20,0), (20,10), (20,20)
        assert!(cropped.len() > 0);
        assert!(cropped.len() <= events.len());

        // Check that all coordinates are within crop bounds
        for event in &cropped {
            assert!(event.x < 30);
            assert!(event.y < 30);
        }
    }

    #[test]
    fn test_random_crop_augmentation() {
        let events = create_test_events();
        let config = RandomCropAugmentation::new(30, 30, 50, 50).with_seed(42);

        let cropped = config.apply(&events).unwrap();

        // Should have some events
        assert!(cropped.len() <= events.len());

        // Check that all coordinates are within crop bounds
        for event in &cropped {
            assert!(event.x < 30);
            assert!(event.y < 30);
        }

        // Test reproducibility
        let cropped2 = config.apply(&events).unwrap();
        assert_eq!(cropped.len(), cropped2.len());

        // Should have same events (due to seed)
        for (e1, e2) in cropped.iter().zip(cropped2.iter()) {
            assert_eq!(e1.x, e2.x);
            assert_eq!(e1.y, e2.y);
            assert_eq!(e1.t, e2.t);
            assert_eq!(e1.polarity, e2.polarity);
        }
    }

    #[test]
    fn test_center_crop_coordinate_remapping() {
        let events = vec![
            Event {
                t: 1.0,
                x: 25,
                y: 25,
                polarity: true,
            }, // Should be at center after crop
        ];

        let config = CenterCropAugmentation::new(30, 30, 50, 50);
        let cropped = center_crop(&events, &config).unwrap();

        assert_eq!(cropped.len(), 1);
        // Original (25, 25) with offset (10, 10) should become (15, 15)
        assert_eq!(cropped[0].x, 15);
        assert_eq!(cropped[0].y, 15);
        assert_eq!(cropped[0].t, 1.0);
        assert_eq!(cropped[0].polarity, true);
    }

    #[test]
    fn test_random_crop_coordinate_remapping() {
        let events = vec![Event {
            t: 1.0,
            x: 25,
            y: 25,
            polarity: true,
        }];

        let config = RandomCropAugmentation::new(20, 20, 40, 40).with_seed(42);
        let cropped = random_crop(&events, &config).unwrap();

        if !cropped.is_empty() {
            // Check coordinate bounds
            assert!(cropped[0].x < 20);
            assert!(cropped[0].y < 20);
            assert_eq!(cropped[0].t, 1.0);
            assert_eq!(cropped[0].polarity, true);
        }
    }

    #[test]
    fn test_crop_edge_cases() {
        let events = create_test_events();

        // Crop size equals sensor size
        let config = CenterCropAugmentation::new(50, 50, 50, 50);
        let cropped = config.apply(&events).unwrap();
        assert_eq!(cropped.len(), events.len());

        // All coordinates should remain the same (offset is 0)
        for (original, cropped) in events.iter().zip(cropped.iter()) {
            assert_eq!(original.x, cropped.x);
            assert_eq!(original.y, cropped.y);
        }

        // Very small crop
        let config = CenterCropAugmentation::new(1, 1, 50, 50);
        let cropped = config.apply(&events).unwrap();
        // Only events exactly at the center pixel should remain
        for event in &cropped {
            assert_eq!(event.x, 0); // After remapping to (0,0)
            assert_eq!(event.y, 0);
        }
    }

    #[test]
    fn test_crop_validation() {
        // Valid configurations
        assert!(CenterCropAugmentation::new(30, 30, 50, 50)
            .validate()
            .is_ok());
        assert!(RandomCropAugmentation::new(30, 30, 50, 50)
            .validate()
            .is_ok());

        // Invalid configurations
        assert!(CenterCropAugmentation::new(0, 30, 50, 50)
            .validate()
            .is_err()); // Zero width
        assert!(CenterCropAugmentation::new(30, 0, 50, 50)
            .validate()
            .is_err()); // Zero height
        assert!(CenterCropAugmentation::new(30, 30, 0, 50)
            .validate()
            .is_err()); // Zero sensor width
        assert!(CenterCropAugmentation::new(30, 30, 50, 0)
            .validate()
            .is_err()); // Zero sensor height
        assert!(CenterCropAugmentation::new(60, 30, 50, 50)
            .validate()
            .is_err()); // Crop larger than sensor
        assert!(CenterCropAugmentation::new(30, 60, 50, 50)
            .validate()
            .is_err()); // Crop larger than sensor

        // Same for random crop
        assert!(RandomCropAugmentation::new(60, 30, 50, 50)
            .validate()
            .is_err());
    }

    #[test]
    fn test_empty_events() {
        let events = Vec::new();

        let center_config = CenterCropAugmentation::new(30, 30, 50, 50);
        let cropped = center_config.apply(&events).unwrap();
        assert!(cropped.is_empty());

        let random_config = RandomCropAugmentation::new(30, 30, 50, 50);
        let cropped = random_config.apply(&events).unwrap();
        assert!(cropped.is_empty());
    }

    #[test]
    fn test_crop_offset_calculations() {
        let config = CenterCropAugmentation::new(30, 40, 100, 80);
        let (offset_x, offset_y) = config.get_offsets();
        assert_eq!(offset_x, 35); // (100 - 30) / 2
        assert_eq!(offset_y, 20); // (80 - 40) / 2

        let (x_start, x_end, y_start, y_end) = config.get_bounds();
        assert_eq!(x_start, 35);
        assert_eq!(x_end, 65); // 35 + 30
        assert_eq!(y_start, 20);
        assert_eq!(y_end, 60); // 20 + 40
    }

    #[cfg(feature = "polars")]
    #[test]
    fn test_center_crop_polars() {
        use crate::ev_core::events_to_dataframe;

        let events = create_test_events();
        let df = events_to_dataframe(&events).unwrap().lazy();

        let config = CenterCropAugmentation::new(30, 30, 50, 50);
        let cropped_df = apply_center_crop_polars(df, &config).unwrap();
        let result = cropped_df.collect().unwrap();

        // Should have some events
        assert!(result.height() > 0);
        assert!(result.height() <= events.len());

        // Check coordinate bounds
        let x_series = result.column(COL_X).unwrap().i64().unwrap();
        let y_series = result.column(COL_Y).unwrap().i64().unwrap();

        for i in 0..result.height() {
            let x = x_series.get(i).unwrap();
            let y = y_series.get(i).unwrap();
            assert!(x >= 0 && x < 30);
            assert!(y >= 0 && y < 30);
        }
    }

    #[cfg(feature = "polars")]
    #[test]
    fn test_random_crop_polars() {
        use crate::ev_core::events_to_dataframe;

        let events = create_test_events();
        let df = events_to_dataframe(&events).unwrap().lazy();

        let config = RandomCropAugmentation::new(30, 30, 50, 50).with_seed(42);
        let cropped_df = apply_random_crop_polars(df, &config).unwrap();
        let result = cropped_df.collect().unwrap();

        // Check coordinate bounds
        if result.height() > 0 {
            let x_series = result.column(COL_X).unwrap().i64().unwrap();
            let y_series = result.column(COL_Y).unwrap().i64().unwrap();

            for i in 0..result.height() {
                let x = x_series.get(i).unwrap();
                let y = y_series.get(i).unwrap();
                assert!(x >= 0 && x < 30);
                assert!(y >= 0 && y < 30);
            }
        }
    }
}
