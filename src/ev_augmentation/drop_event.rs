//! Event dropping augmentations
//!
//! This module implements various strategies for dropping events:
//! - drop_by_probability: Use ev_filtering::downsampling with uniform strategy
//! - drop_by_time: Drop events within a time interval
//! - drop_by_area: Drop events within a spatial region

use crate::ev_augmentation::{
    AugmentationError, AugmentationResult, SingleAugmentation, Validatable,
};
use crate::ev_core::Events;
use crate::ev_filtering::downsampling::DownsamplingFilter;
use rand::SeedableRng;

use rand_distr::{Distribution, Uniform};
use tracing::{debug, info, instrument};

#[cfg(feature = "polars")]
use polars::prelude::*;

/// Drop event by probability augmentation
///
/// This is a wrapper around the existing downsampling functionality
/// for API consistency with tonic.
#[derive(Debug, Clone)]
pub struct DropEventAugmentation {
    /// Probability of dropping each event (0.0 to 1.0)
    pub drop_probability: f64,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl DropEventAugmentation {
    /// Create a new drop event augmentation
    ///
    /// # Arguments
    ///
    /// * `drop_probability` - Probability of dropping each event (0.0 to 1.0)
    pub fn new(drop_probability: f64) -> Self {
        Self {
            drop_probability,
            seed: None,
        }
    }

    /// Set random seed for reproducibility
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Get description of this augmentation
    pub fn description(&self) -> String {
        format!("p={:.2}", self.drop_probability)
    }
}

impl Validatable for DropEventAugmentation {
    fn validate(&self) -> AugmentationResult<()> {
        if self.drop_probability < 0.0 || self.drop_probability > 1.0 {
            return Err(AugmentationError::InvalidProbability(self.drop_probability));
        }
        Ok(())
    }
}

impl SingleAugmentation for DropEventAugmentation {
    fn apply(&self, events: &Events) -> AugmentationResult<Events> {
        // Use existing downsampling functionality
        let keep_rate = 1.0 - self.drop_probability;
        let mut filter = DownsamplingFilter::uniform(keep_rate);
        if let Some(seed) = self.seed {
            filter.random_seed = Some(seed);
        }

        crate::ev_filtering::downsampling::apply_downsampling_filter(events, &filter)
            .map_err(|e| AugmentationError::ProcessingError(e.to_string()))
    }

    fn description(&self) -> String {
        format!("Drop event: {}", self.description())
    }
}

/// Drop events by time interval augmentation
///
/// Drops events within a randomly selected time interval.
#[derive(Debug, Clone)]
pub struct DropTimeAugmentation {
    /// Duration ratio of the interval to drop (0.0 to 1.0)
    pub duration_ratio: f64,
    /// Optional range for random ratio selection
    pub ratio_range: Option<(f64, f64)>,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl DropTimeAugmentation {
    /// Create a new drop time augmentation
    ///
    /// # Arguments
    ///
    /// * `duration_ratio` - Ratio of total duration to drop (0.0 to 1.0)
    pub fn new(duration_ratio: f64) -> Self {
        Self {
            duration_ratio,
            ratio_range: None,
            seed: None,
        }
    }

    /// Create with random duration ratio in range
    pub fn random(min_ratio: f64, max_ratio: f64) -> Self {
        Self {
            duration_ratio: (min_ratio + max_ratio) / 2.0,
            ratio_range: Some((min_ratio, max_ratio)),
            seed: None,
        }
    }

    /// Set random seed for reproducibility
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Get description of this augmentation
    pub fn description(&self) -> String {
        if let Some((min, max)) = self.ratio_range {
            format!("ratio∈[{:.2},{:.2}]", min, max)
        } else {
            format!("ratio={:.2}", self.duration_ratio)
        }
    }
}

impl Validatable for DropTimeAugmentation {
    fn validate(&self) -> AugmentationResult<()> {
        if self.duration_ratio < 0.0 || self.duration_ratio >= 1.0 {
            return Err(AugmentationError::InvalidConfig(
                "Duration ratio must be in [0, 1)".to_string(),
            ));
        }
        if let Some((min, max)) = self.ratio_range {
            if min < 0.0 || max >= 1.0 || min >= max {
                return Err(AugmentationError::InvalidConfig(
                    "Invalid ratio range".to_string(),
                ));
            }
        }
        Ok(())
    }
}

impl SingleAugmentation for DropTimeAugmentation {
    fn apply(&self, events: &Events) -> AugmentationResult<Events> {
        drop_by_time(events, self)
    }

    fn description(&self) -> String {
        format!("Drop time: {}", self.description())
    }
}

/// Drop events by area augmentation
///
/// Drops events within a randomly selected rectangular area.
#[derive(Debug, Clone)]
pub struct DropAreaAugmentation {
    /// Area ratio to drop (0.0 to 1.0)
    pub area_ratio: f64,
    /// Optional range for random ratio selection
    pub ratio_range: Option<(f64, f64)>,
    /// Sensor width
    pub sensor_width: u16,
    /// Sensor height
    pub sensor_height: u16,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl DropAreaAugmentation {
    /// Create a new drop area augmentation
    ///
    /// # Arguments
    ///
    /// * `area_ratio` - Ratio of sensor area to drop (0.0 to 1.0)
    /// * `sensor_width` - Sensor width in pixels
    /// * `sensor_height` - Sensor height in pixels
    pub fn new(area_ratio: f64, sensor_width: u16, sensor_height: u16) -> Self {
        Self {
            area_ratio,
            ratio_range: None,
            sensor_width,
            sensor_height,
            seed: None,
        }
    }

    /// Create with random area ratio in range
    pub fn random(min_ratio: f64, max_ratio: f64, sensor_width: u16, sensor_height: u16) -> Self {
        Self {
            area_ratio: (min_ratio + max_ratio) / 2.0,
            ratio_range: Some((min_ratio, max_ratio)),
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

    /// Get description of this augmentation
    pub fn description(&self) -> String {
        if let Some((min, max)) = self.ratio_range {
            format!(
                "ratio∈[{:.2},{:.2}], {}x{}",
                min, max, self.sensor_width, self.sensor_height
            )
        } else {
            format!(
                "ratio={:.2}, {}x{}",
                self.area_ratio, self.sensor_width, self.sensor_height
            )
        }
    }
}

impl Validatable for DropAreaAugmentation {
    fn validate(&self) -> AugmentationResult<()> {
        if self.sensor_width == 0 || self.sensor_height == 0 {
            return Err(AugmentationError::InvalidSensorSize(
                self.sensor_width,
                self.sensor_height,
            ));
        }
        if self.area_ratio < 0.0 || self.area_ratio >= 1.0 {
            return Err(AugmentationError::InvalidConfig(
                "Area ratio must be in [0, 1)".to_string(),
            ));
        }
        if let Some((min, max)) = self.ratio_range {
            if min < 0.0 || max >= 1.0 || min >= max {
                return Err(AugmentationError::InvalidConfig(
                    "Invalid ratio range".to_string(),
                ));
            }
        }
        Ok(())
    }
}

impl SingleAugmentation for DropAreaAugmentation {
    fn apply(&self, events: &Events) -> AugmentationResult<Events> {
        drop_by_area(events, self)
    }

    fn description(&self) -> String {
        format!("Drop area: {}", self.description())
    }
}

/// Drop events by probability
///
/// Convenience function that uses existing downsampling
pub fn drop_by_probability(events: &Events, probability: f64) -> AugmentationResult<Events> {
    let aug = DropEventAugmentation::new(probability);
    aug.apply(events)
}

/// Drop events within a time interval
#[instrument(skip(events), fields(n_events = events.len()))]
pub fn drop_by_time(events: &Events, config: &DropTimeAugmentation) -> AugmentationResult<Events> {
    let start_time = std::time::Instant::now();

    if events.is_empty() {
        debug!("No events to drop");
        return Ok(Vec::new());
    }

    // Validate configuration
    config.validate()?;

    // Find time range
    let t_start = events
        .iter()
        .map(|e| e.t)
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0.0);
    let t_end = events
        .iter()
        .map(|e| e.t)
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(1.0);

    let total_duration = t_end - t_start;
    if total_duration <= 0.0 {
        return Ok(events.clone());
    }

    // Initialize RNG
    let mut rng = if let Some(seed) = config.seed {
        rand::rngs::StdRng::seed_from_u64(seed)
    } else {
        rand::rngs::StdRng::from_entropy()
    };

    // Get actual ratio to use
    let ratio = if let Some((min, max)) = config.ratio_range {
        Uniform::new(min, max).sample(&mut rng)
    } else {
        config.duration_ratio
    };

    // Calculate drop interval
    let drop_duration = total_duration * ratio;
    let max_start = total_duration - drop_duration;
    let drop_start = if max_start > 0.0 {
        t_start + Uniform::new(0.0, max_start).sample(&mut rng)
    } else {
        t_start
    };
    let drop_end = drop_start + drop_duration;

    // Filter events
    let mut filtered_events = Vec::with_capacity(events.len());
    let mut dropped_count = 0;

    for event in events {
        if event.t >= drop_start && event.t <= drop_end {
            dropped_count += 1;
        } else {
            filtered_events.push(*event);
        }
    }

    let processing_time = start_time.elapsed().as_secs_f64();

    info!(
        "Drop by time applied (interval {:.3}s-{:.3}s): {} -> {} events ({} dropped) in {:.3}s",
        drop_start,
        drop_end,
        events.len(),
        filtered_events.len(),
        dropped_count,
        processing_time
    );

    Ok(filtered_events)
}

/// Drop events within a spatial area
#[instrument(skip(events), fields(n_events = events.len()))]
pub fn drop_by_area(events: &Events, config: &DropAreaAugmentation) -> AugmentationResult<Events> {
    let start_time = std::time::Instant::now();

    if events.is_empty() {
        debug!("No events to drop");
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

    // Get actual ratio to use
    let ratio = if let Some((min, max)) = config.ratio_range {
        Uniform::new(min, max).sample(&mut rng)
    } else {
        config.area_ratio
    };

    // Calculate box dimensions
    let box_width = ((config.sensor_width as f64) * ratio.sqrt()) as u16;
    let box_height = ((config.sensor_height as f64) * ratio.sqrt()) as u16;

    // Random box position
    let max_x = config.sensor_width.saturating_sub(box_width);
    let max_y = config.sensor_height.saturating_sub(box_height);

    let box_x = if max_x > 0 {
        Uniform::new(0, max_x).sample(&mut rng)
    } else {
        0
    };
    let box_y = if max_y > 0 {
        Uniform::new(0, max_y).sample(&mut rng)
    } else {
        0
    };

    let box_x_end = box_x + box_width;
    let box_y_end = box_y + box_height;

    // Filter events
    let mut filtered_events = Vec::with_capacity(events.len());
    let mut dropped_count = 0;

    for event in events {
        if event.x >= box_x && event.x < box_x_end && event.y >= box_y && event.y < box_y_end {
            dropped_count += 1;
        } else {
            filtered_events.push(*event);
        }
    }

    let processing_time = start_time.elapsed().as_secs_f64();

    info!(
        "Drop by area applied (box [{},{})x[{},{})): {} -> {} events ({} dropped) in {:.3}s",
        box_x,
        box_x_end,
        box_y,
        box_y_end,
        events.len(),
        filtered_events.len(),
        dropped_count,
        processing_time
    );

    Ok(filtered_events)
}

/// Apply drop by time using Polars operations
#[cfg(feature = "polars")]
pub fn apply_drop_time_polars(
    df: LazyFrame,
    config: &DropTimeAugmentation,
) -> PolarsResult<LazyFrame> {
    // For random parameters, we need to collect and process
    let collected_df = df.collect()?;
    let events = crate::ev_augmentation::dataframe_to_events(&collected_df)
        .map_err(|e| PolarsError::ComputeError(format!("Conversion error: {}", e).into()))?;
    let filtered = drop_by_time(&events, config)
        .map_err(|e| PolarsError::ComputeError(format!("Drop time error: {}", e).into()))?;
    let filtered_df = crate::ev_core::events_to_dataframe(&filtered)?;
    Ok(filtered_df.lazy())
}

/// Apply drop by area using Polars operations
#[cfg(feature = "polars")]
pub fn apply_drop_area_polars(
    df: LazyFrame,
    config: &DropAreaAugmentation,
) -> PolarsResult<LazyFrame> {
    // For random parameters, we need to collect and process
    let collected_df = df.collect()?;
    let events = crate::ev_augmentation::dataframe_to_events(&collected_df)
        .map_err(|e| PolarsError::ComputeError(format!("Conversion error: {}", e).into()))?;
    let filtered = drop_by_area(&events, config)
        .map_err(|e| PolarsError::ComputeError(format!("Drop area error: {}", e).into()))?;
    let filtered_df = crate::ev_core::events_to_dataframe(&filtered)?;
    Ok(filtered_df.lazy())
}

/// Apply drop event using Polars operations
#[cfg(feature = "polars")]
pub fn apply_drop_event_polars(
    df: LazyFrame,
    config: &DropEventAugmentation,
) -> PolarsResult<LazyFrame> {
    // Use downsampling functionality
    let keep_rate = 1.0 - config.drop_probability;
    let mut filter = DownsamplingFilter::uniform(keep_rate);
    if let Some(seed) = config.seed {
        filter.random_seed = Some(seed);
    }

    crate::ev_filtering::downsampling::apply_downsampling_filter_polars(df, &filter)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_events() -> Events {
        let mut events = Vec::new();
        for i in 0..100 {
            events.push(Event {
                t: i as f64 * 0.01,      // 0 to 0.99 seconds
                x: (i % 40) as u16 * 10, // 0 to 390
                y: (i / 40) as u16 * 10, // 0 to 20
                polarity: i % 2 == 0,
            });
        }
        events
    }

    #[test]
    fn test_drop_event_augmentation() {
        let events = create_test_events();
        let aug = DropEventAugmentation::new(0.3).with_seed(42);

        let filtered = aug.apply(&events).unwrap();

        // Should drop approximately 30% of events
        assert!(filtered.len() < events.len());
        assert!(filtered.len() > events.len() / 2); // But not too many
    }

    #[test]
    fn test_drop_by_time() {
        let events = create_test_events();
        let config = DropTimeAugmentation::new(0.2).with_seed(42);

        let filtered = drop_by_time(&events, &config).unwrap();

        // Should drop approximately 20% of events
        assert!(filtered.len() < events.len());
        assert!(filtered.len() > events.len() * 0.7);

        // Remaining events should not be in a contiguous interval
        let times: Vec<f64> = filtered.iter().map(|e| e.t).collect();
        let mut has_gap = false;
        for i in 1..times.len() {
            if times[i] - times[i - 1] > 0.011 {
                // Larger than normal spacing
                has_gap = true;
                break;
            }
        }
        assert!(has_gap);
    }

    #[test]
    fn test_drop_by_area() {
        let events = create_test_events();
        let config = DropAreaAugmentation::new(0.25, 400, 400).with_seed(42);

        let filtered = drop_by_area(&events, &config).unwrap();

        // Should drop approximately 25% of events
        assert!(filtered.len() < events.len());
        assert!(filtered.len() > events.len() / 2);
    }

    #[test]
    fn test_drop_by_time_random_range() {
        let events = create_test_events();
        let config = DropTimeAugmentation::random(0.1, 0.3).with_seed(42);

        let filtered = drop_by_time(&events, &config).unwrap();

        // Should drop between 10% and 30% of events
        assert!(filtered.len() <= events.len() * 0.9);
        assert!(filtered.len() >= events.len() * 0.7);
    }

    #[test]
    fn test_drop_by_area_random_range() {
        let events = create_test_events();
        let config = DropAreaAugmentation::random(0.1, 0.3, 400, 400).with_seed(42);

        let filtered = drop_by_area(&events, &config).unwrap();

        // Should drop between 10% and 30% of events (approximately)
        assert!(filtered.len() < events.len());
    }

    #[test]
    fn test_validation() {
        // Valid configs
        assert!(DropEventAugmentation::new(0.5).validate().is_ok());
        assert!(DropTimeAugmentation::new(0.5).validate().is_ok());
        assert!(DropAreaAugmentation::new(0.5, 640, 480).validate().is_ok());

        // Invalid configs
        assert!(DropEventAugmentation::new(-0.1).validate().is_err());
        assert!(DropEventAugmentation::new(1.1).validate().is_err());
        assert!(DropTimeAugmentation::new(1.0).validate().is_err());
        assert!(DropAreaAugmentation::new(0.5, 0, 480).validate().is_err());
    }

    #[test]
    fn test_empty_events() {
        let events = Vec::new();

        let filtered = drop_by_time(&events, &DropTimeAugmentation::new(0.5)).unwrap();
        assert!(filtered.is_empty());

        let filtered = drop_by_area(&events, &DropAreaAugmentation::new(0.5, 640, 480)).unwrap();
        assert!(filtered.is_empty());
    }
}
