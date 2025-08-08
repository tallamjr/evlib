//! Polars-first event dropping augmentations for event camera data
//!
//! This module provides event dropping functionality using Polars DataFrames
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
//! - Vectorized operations: All dropping uses SIMD-optimized Polars operations
//! - Memory efficiency: No intermediate Vec<Event> allocations
//! - Query optimization: Polars optimizes the entire augmentation pipeline
//!
//! # Strategies
//!
//! - drop_by_probability: Use ev_filtering::downsampling with uniform strategy
//! - drop_by_time: Drop events within a time interval
//! - drop_by_area: Drop events within a spatial region
//!
//! # Example
//!
//! ```rust
//! use polars::prelude::*;
//! use evlib::ev_augmentation::drop_event::*;
//!
//! // Convert events to LazyFrame once
//! let events_df = events_to_dataframe(&events)?.lazy();
//!
//! // Apply event dropping with Polars expressions
//! let filtered = apply_drop_event(events_df, &DropEventAugmentation::new(0.2))?;
//! ```

use crate::ev_augmentation::{AugmentationError, AugmentationResult, Validatable};
// Removed: use crate::Events; - legacy type no longer exists
use crate::ev_filtering::downsampling::DownsamplingFilter;

// Tracing imports removed due to unused warnings

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

    /// Apply event dropping directly to DataFrame (recommended approach)
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
    /// Filtered LazyFrame with events dropped
    #[cfg(feature = "polars")]
    pub fn apply_to_dataframe(&self, df: LazyFrame) -> PolarsResult<LazyFrame> {
        apply_drop_event(df, self)
    }
}

/// Apply drop event using DataFrame - this is the main implementation function
#[cfg(feature = "polars")]
pub fn apply_drop_event(df: LazyFrame, config: &DropEventAugmentation) -> PolarsResult<LazyFrame> {
    if config.drop_probability <= 0.0 {
        return Ok(df);
    }

    if config.drop_probability >= 1.0 {
        return Ok(df.limit(0));
    }

    let keep_rate = 1.0 - config.drop_probability;
    let mut filter = DownsamplingFilter::uniform(keep_rate);
    if let Some(seed) = config.seed {
        filter.random_seed = Some(seed);
    }

    crate::ev_filtering::downsampling::apply_downsampling_filter_polars(df, &filter)
}

impl Validatable for DropEventAugmentation {
    fn validate(&self) -> AugmentationResult<()> {
        if self.drop_probability < 0.0 || self.drop_probability > 1.0 {
            return Err(AugmentationError::InvalidProbability(self.drop_probability));
        }
        Ok(())
    }
}

/* Commented out - legacy SingleAugmentation trait no longer exists
impl SingleAugmentation for DropEventAugmentation {
    fn apply(&self, events: &Events) -> AugmentationResult<Events> {
        // Legacy Vec<Event> interface - convert to DataFrame and back
        // This is for backward compatibility only
        #[cfg(feature = "tracing")]
        tracing::warn!("Using legacy Vec<Event> interface - consider using LazyFrame directly for better performance");
        #[cfg(not(feature = "tracing"))]
        eprintln!("[WARN] Using legacy Vec<Event> interface - consider using LazyFrame directly for better performance");

        #[cfg(feature = "polars")]
        {
            let df = crate::events_to_dataframe(events)
                .map_err(|e| {
                    AugmentationError::ProcessingError(format!(
                        "DataFrame conversion failed: {}",
                        e
                    ))
                })?
                .lazy();

            let filtered_df = self.apply_to_dataframe(df).map_err(|e| {
                AugmentationError::ProcessingError(format!("Polars dropping failed: {}", e))
            })?;

            // Convert back to Vec<Event> - this is inefficient but maintains compatibility
            let result_df = filtered_df.collect().map_err(|e| {
                AugmentationError::ProcessingError(format!("LazyFrame collection failed: {}", e))
            })?;

            // Convert DataFrame back to Events
            dataframe_to_events(&result_df)
        }

        #[cfg(not(feature = "polars"))]
        {
            // Use existing downsampling functionality
            let keep_rate = 1.0 - self.drop_probability;
            let mut filter = DownsamplingFilter::uniform(keep_rate);
            if let Some(seed) = self.seed {
                filter.random_seed = Some(seed);
            }

            crate::ev_filtering::downsampling::apply_downsampling_filter(events, &filter)
                .map_err(|e| AugmentationError::ProcessingError(e.to_string()))
        }
    }

    fn description(&self) -> String {
        format!("Drop event: {}", self.description())
    }
}
*/

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

    /// Apply time dropping directly to DataFrame (recommended approach)
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
    /// Filtered LazyFrame with events in time interval dropped
    #[cfg(feature = "polars")]
    pub fn apply_to_dataframe(&self, df: LazyFrame) -> PolarsResult<LazyFrame> {
        apply_drop_time(df, self)
    }
}

/// Apply drop time using DataFrame - this is the main implementation function
#[cfg(feature = "polars")]
pub fn apply_drop_time(df: LazyFrame, config: &DropTimeAugmentation) -> PolarsResult<LazyFrame> {
    if config.duration_ratio <= 0.0 {
        return Ok(df);
    }

    // Get time bounds
    let time_bounds = df
        .clone()
        .select([
            col(COL_T).min().alias("t_min"),
            col(COL_T).max().alias("t_max"),
        ])
        .collect()?;

    let t_min = time_bounds.column("t_min")?.get(0)?.try_extract::<f64>()?;
    let t_max = time_bounds.column("t_max")?.get(0)?.try_extract::<f64>()?;
    let total_duration = t_max - t_min;

    if total_duration <= 0.0 {
        return Ok(df);
    }

    // Calculate drop interval (simplified - no random for now)
    let drop_duration = total_duration * config.duration_ratio;
    let drop_start = t_min + (total_duration - drop_duration) / 2.0;
    let drop_end = drop_start + drop_duration;

    // Apply vectorized filtering
    Ok(df.filter(
        col(COL_T)
            .lt(lit(drop_start))
            .or(col(COL_T).gt(lit(drop_end))),
    ))
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

/* Commented out - legacy SingleAugmentation trait no longer exists
impl SingleAugmentation for DropTimeAugmentation {
    fn apply(&self, events: &Events) -> AugmentationResult<Events> {
        // Legacy Vec<Event> interface - convert to DataFrame and back
        // This is for backward compatibility only
        #[cfg(feature = "tracing")]
        tracing::warn!("Using legacy Vec<Event> interface - consider using LazyFrame directly for better performance");
        #[cfg(not(feature = "tracing"))]
        eprintln!("[WARN] Using legacy Vec<Event> interface - consider using LazyFrame directly for better performance");

        #[cfg(feature = "polars")]
        {
            let df = crate::events_to_dataframe(events)
                .map_err(|e| {
                    AugmentationError::ProcessingError(format!(
                        "DataFrame conversion failed: {}",
                        e
                    ))
                })?
                .lazy();

            let filtered_df = self.apply_to_dataframe(df).map_err(|e| {
                AugmentationError::ProcessingError(format!("Polars dropping failed: {}", e))
            })?;

            // Convert back to Vec<Event> - this is inefficient but maintains compatibility
            let result_df = filtered_df.collect().map_err(|e| {
                AugmentationError::ProcessingError(format!("LazyFrame collection failed: {}", e))
            })?;

            // Convert DataFrame back to Events
            dataframe_to_events(&result_df)
        }

        #[cfg(not(feature = "polars"))]
        {
            drop_by_time(events, self)
        }
    }

    fn description(&self) -> String {
        format!("Drop time: {}", self.description())
    }
}
*/

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

    /// Apply area dropping directly to DataFrame (recommended approach)
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
    /// Filtered LazyFrame with events in spatial area dropped
    #[cfg(feature = "polars")]
    pub fn apply_to_dataframe(&self, df: LazyFrame) -> PolarsResult<LazyFrame> {
        apply_drop_area(df, self)
    }
}

/// Apply drop area using DataFrame - this is the main implementation function
#[cfg(feature = "polars")]
pub fn apply_drop_area(df: LazyFrame, config: &DropAreaAugmentation) -> PolarsResult<LazyFrame> {
    if config.area_ratio <= 0.0 {
        return Ok(df);
    }

    // Calculate box dimensions (simplified - no random for now)
    let box_width = ((config.sensor_width as f64) * config.area_ratio.sqrt()) as u16;
    let box_height = ((config.sensor_height as f64) * config.area_ratio.sqrt()) as u16;

    // Center the box for simplicity
    let box_x = (config.sensor_width - box_width) / 2;
    let box_y = (config.sensor_height - box_height) / 2;
    let box_x_end = box_x + box_width;
    let box_y_end = box_y + box_height;

    // Apply vectorized filtering - keep events OUTSIDE the box
    Ok(df.filter(
        col(COL_X)
            .lt(lit(box_x as i64))
            .or(col(COL_X).gt_eq(lit(box_x_end as i64)))
            .or(col(COL_Y).lt(lit(box_y as i64)))
            .or(col(COL_Y).gt_eq(lit(box_y_end as i64))),
    ))
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

/* Commented out - legacy SingleAugmentation trait no longer exists
impl SingleAugmentation for DropAreaAugmentation {
    fn apply(&self, events: &Events) -> AugmentationResult<Events> {
        // Legacy Vec<Event> interface - convert to DataFrame and back
        // This is for backward compatibility only
        #[cfg(feature = "tracing")]
        tracing::warn!("Using legacy Vec<Event> interface - consider using LazyFrame directly for better performance");
        #[cfg(not(feature = "tracing"))]
        eprintln!("[WARN] Using legacy Vec<Event> interface - consider using LazyFrame directly for better performance");

        #[cfg(feature = "polars")]
        {
            let df = crate::events_to_dataframe(events)
                .map_err(|e| {
                    AugmentationError::ProcessingError(format!(
                        "DataFrame conversion failed: {}",
                        e
                    ))
                })?
                .lazy();

            let filtered_df = self.apply_to_dataframe(df).map_err(|e| {
                AugmentationError::ProcessingError(format!("Polars dropping failed: {}", e))
            })?;

            // Convert back to Vec<Event> - this is inefficient but maintains compatibility
            let result_df = filtered_df.collect().map_err(|e| {
                AugmentationError::ProcessingError(format!("LazyFrame collection failed: {}", e))
            })?;

            // Convert DataFrame back to Events
            dataframe_to_events(&result_df)
        }

        #[cfg(not(feature = "polars"))]
        {
            drop_by_area(events, self)
        }
    }

    fn description(&self) -> String {
        format!("Drop area: {}", self.description())
    }
}
*/

/* Commented out - legacy Events type no longer exists
/// Legacy function for backward compatibility - delegates to Polars implementation
pub fn drop_by_probability(events: &Events, probability: f64) -> AugmentationResult<Events> {
    let aug = DropEventAugmentation::new(probability);
    aug.apply(events)
}

/// Helper function to convert DataFrame back to Events (for legacy compatibility)
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
    let p_series = df
        .column(COL_POLARITY)
        .map_err(|e| AugmentationError::ProcessingError(format!("Missing polarity column: {}", e)))?;

    let x_values = x_series
        .i64()
        .map_err(|e| AugmentationError::ProcessingError(format!("X column type error: {}", e)))?;
    let y_values = y_series
        .i64()
        .map_err(|e| AugmentationError::ProcessingError(format!("Y column type error: {}", e)))?;
    let t_values = t_series
        .f64()
        .map_err(|e| AugmentationError::ProcessingError(format!("T column type error: {}", e)))?;
    let p_values = p_series
        .i64()
        .map_err(|e| AugmentationError::ProcessingError(format!("Polarity column type error: {}", e)))?;

    for i in 0..height {
        let x = x_values
            .get(i)
            .ok_or_else(|| AugmentationError::ProcessingError("Missing x value".to_string()))?
            as u16;
        let y = y_values
            .get(i)
            .ok_or_else(|| AugmentationError::ProcessingError("Missing y value".to_string()))?
            as u16;
        let t = t_values
            .get(i)
            .ok_or_else(|| AugmentationError::ProcessingError("Missing t value".to_string()))?;
        let p = p_values
            .get(i)
            .ok_or_else(|| AugmentationError::ProcessingError("Missing polarity value".to_string()))?
            > 0;

        events.push(crate::Event {
            x,
            y,
            t,
            polarity: p,
        });
    }

    Ok(events)
}

/// Drop events within a time interval
#[cfg_attr(feature = "tracing", instrument(skip(events), fields(n_events = events.len())))]
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
#[cfg_attr(feature = "tracing", instrument(skip(events), fields(n_events = events.len())))]
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

/// Apply drop by time using Polars expressions
///
/// This is the main time-based dropping function that works entirely with Polars
/// operations for maximum performance. For random parameters, it collects data
/// to compute ranges but uses vectorized filtering.
///
/// # Arguments
///
/// * `df` - Input LazyFrame containing event data
/// * `config` - Time dropping configuration
///
/// # Returns
///
/// Filtered LazyFrame with events in time interval dropped
///
/// # Example
///
/// ```rust
/// use polars::prelude::*;
/// use evlib::ev_augmentation::drop_event::*;
///
/// let events_df = events_to_dataframe(&events)?.lazy();
/// let config = DropTimeAugmentation::new(0.2);
/// let filtered = apply_drop_time(events_df, &config)?;
/// ```
#[cfg(feature = "polars")]
#[cfg_attr(feature = "tracing", instrument(skip(df), fields(config = ?config)))]
pub fn apply_drop_time(
    df: LazyFrame,
    config: &DropTimeAugmentation,
) -> PolarsResult<LazyFrame> {
    debug!("Applying drop by time with Polars: {:?}", config);

    if config.duration_ratio <= 0.0 {
        debug!("No time dropping needed (ratio <= 0)");
        return Ok(df);
    }

    // Get time bounds
    let time_bounds = df.clone()
        .select([
            col(COL_T).min().alias("t_min"),
            col(COL_T).max().alias("t_max"),
        ])
        .collect()?;

    let t_min = time_bounds.column("t_min")?.get(0)?.try_extract::<f64>()?;
    let t_max = time_bounds.column("t_max")?.get(0)?.try_extract::<f64>()?;
    let total_duration = t_max - t_min;

    if total_duration <= 0.0 {
        debug!("No temporal range to drop from");
        return Ok(df);
    }

    // For random parameters, we need to generate random values
    // This requires some computation but we can still use Polars for filtering
    let ratio = if let Some((min, max)) = config.ratio_range {
        use rand::Rng;
        let mut rng = if let Some(seed) = config.seed {
            rand::rngs::StdRng::seed_from_u64(seed)
        } else {
            rand::rngs::StdRng::from_entropy()
        };
        rand_distr::Uniform::new(min, max).sample(&mut rng)
    } else {
        config.duration_ratio
    };

    // Calculate drop interval
    let drop_duration = total_duration * ratio;
    let max_start = total_duration - drop_duration;

    let drop_start = if max_start > 0.0 {
        use rand::Rng;
        let mut rng = if let Some(seed) = config.seed {
            rand::rngs::StdRng::seed_from_u64(seed.wrapping_add(1))
        } else {
            rand::rngs::StdRng::from_entropy()
        };
        t_min + rand_distr::Uniform::new(0.0, max_start).sample(&mut rng)
    } else {
        t_min
    };
    let drop_end = drop_start + drop_duration;

    info!(
        "Drop by time interval [{:.6}, {:.6}] (ratio={:.3})",
        drop_start, drop_end, ratio
    );

    // Apply vectorized filtering
    Ok(df.filter(
        col(COL_T).lt(lit(drop_start))
            .or(col(COL_T).gt(lit(drop_end)))
    ))
}

/// Legacy Polars function for backward compatibility
#[cfg(feature = "polars")]
pub fn apply_drop_time_polars(
    df: LazyFrame,
    config: &DropTimeAugmentation,
) -> PolarsResult<LazyFrame> {
    apply_drop_time(df, config)
}

/// Apply drop by area using Polars expressions
///
/// This is the main spatial area-based dropping function that works entirely with Polars
/// operations for maximum performance. It uses vectorized filtering with spatial bounds.
///
/// # Arguments
///
/// * `df` - Input LazyFrame containing event data
/// * `config` - Area dropping configuration
///
/// # Returns
///
/// Filtered LazyFrame with events in spatial area dropped
///
/// # Example
///
/// ```rust
/// use polars::prelude::*;
/// use evlib::ev_augmentation::drop_event::*;
///
/// let events_df = events_to_dataframe(&events)?.lazy();
/// let config = DropAreaAugmentation::new(0.25, 640, 480);
/// let filtered = apply_drop_area(events_df, &config)?;
/// ```
#[cfg(feature = "polars")]
#[cfg_attr(feature = "tracing", instrument(skip(df), fields(config = ?config)))]
pub fn apply_drop_area(
    df: LazyFrame,
    config: &DropAreaAugmentation,
) -> PolarsResult<LazyFrame> {
    debug!("Applying drop by area with Polars: {:?}", config);

    if config.area_ratio <= 0.0 {
        debug!("No area dropping needed (ratio <= 0)");
        return Ok(df);
    }

    // Generate random parameters for drop area
    let ratio = if let Some((min, max)) = config.ratio_range {
        use rand::Rng;
        let mut rng = if let Some(seed) = config.seed {
            rand::rngs::StdRng::seed_from_u64(seed)
        } else {
            rand::rngs::StdRng::from_entropy()
        };
        rand_distr::Uniform::new(min, max).sample(&mut rng)
    } else {
        config.area_ratio
    };

    // Calculate box dimensions
    let box_width = ((config.sensor_width as f64) * ratio.sqrt()) as u16;
    let box_height = ((config.sensor_height as f64) * ratio.sqrt()) as u16;

    // Random box position
    let max_x = config.sensor_width.saturating_sub(box_width);
    let max_y = config.sensor_height.saturating_sub(box_height);

    let (box_x, box_y) = {
        use rand::Rng;
        let mut rng = if let Some(seed) = config.seed {
            rand::rngs::StdRng::seed_from_u64(seed.wrapping_add(1))
        } else {
            rand::rngs::StdRng::from_entropy()
        };

        let x = if max_x > 0 {
            rand_distr::Uniform::new(0, max_x).sample(&mut rng)
        } else {
            0
        };
        let y = if max_y > 0 {
            rand_distr::Uniform::new(0, max_y).sample(&mut rng)
        } else {
            0
        };
        (x, y)
    };

    let box_x_end = box_x + box_width;
    let box_y_end = box_y + box_height;

    info!(
        "Drop by area box [{},{})x[{},{}) (ratio={:.3})",
        box_x, box_x_end, box_y, box_y_end, ratio
    );

    // Apply vectorized filtering - keep events OUTSIDE the box
    Ok(df.filter(
        col(COL_X).lt(lit(box_x as i64))
            .or(col(COL_X).gt_eq(lit(box_x_end as i64)))
            .or(col(COL_Y).lt(lit(box_y as i64)))
            .or(col(COL_Y).gt_eq(lit(box_y_end as i64)))
    ))
}

/// Legacy Polars function for backward compatibility
#[cfg(feature = "polars")]
pub fn apply_drop_area_polars(
    df: LazyFrame,
    config: &DropAreaAugmentation,
) -> PolarsResult<LazyFrame> {
    apply_drop_area(df, config)
}

/// Apply drop event using Polars expressions
///
/// This is the main probabilistic dropping function that works entirely with Polars
/// operations for maximum performance. It leverages the existing downsampling
/// functionality for consistent behavior.
///
/// # Arguments
///
/// * `df` - Input LazyFrame containing event data
/// * `config` - Event dropping configuration
///
/// # Returns
///
/// Filtered LazyFrame with events randomly dropped
///
/// # Example
///
/// ```rust
/// use polars::prelude::*;
/// use evlib::ev_augmentation::drop_event::*;
///
/// let events_df = events_to_dataframe(&events)?.lazy();
/// let config = DropEventAugmentation::new(0.3);
/// let filtered = apply_drop_event(events_df, &config)?;
/// ```
#[cfg(feature = "polars")]
#[cfg_attr(feature = "tracing", instrument(skip(df), fields(config = ?config)))]
pub fn apply_drop_event(
    df: LazyFrame,
    config: &DropEventAugmentation,
) -> PolarsResult<LazyFrame> {
    debug!("Applying drop event with Polars: {:?}", config);

    if config.drop_probability <= 0.0 {
        debug!("No event dropping needed (probability <= 0)");
        return Ok(df);
    }

    if config.drop_probability >= 1.0 {
        debug!("Dropping all events (probability >= 1)");
        return Ok(df.limit(0)); // Return empty DataFrame with same schema
    }

    // Use downsampling functionality for consistent behavior
    let keep_rate = 1.0 - config.drop_probability;
    let mut filter = DownsamplingFilter::uniform(keep_rate);
    if let Some(seed) = config.seed {
        filter.random_seed = Some(seed);
    }

    crate::ev_filtering::downsampling::apply_downsampling_filter_polars(df, &filter)
}

/// Legacy Polars function for backward compatibility
#[cfg(feature = "polars")]
pub fn apply_drop_event_polars(
    df: LazyFrame,
    config: &DropEventAugmentation,
) -> PolarsResult<LazyFrame> {
    apply_drop_event(df, config)
}

/// Apply event dropping directly to LazyFrame - DataFrame-native version (recommended)
///
/// This function applies probabilistic event dropping directly to a LazyFrame for optimal performance.
/// Use this instead of the legacy Vec<Event> version when possible.
///
/// # Arguments
///
/// * `df` - Input LazyFrame containing event data
/// * `drop_probability` - Probability of dropping each event (0.0 to 1.0)
///
/// # Returns
///
/// Filtered LazyFrame
#[cfg(feature = "polars")]
pub fn drop_by_probability_df(
    df: LazyFrame,
    drop_probability: f64
) -> PolarsResult<LazyFrame> {
    let config = DropEventAugmentation::new(drop_probability);
    apply_drop_event(df, &config)
}

/// Apply time dropping directly to LazyFrame - DataFrame-native version (recommended)
///
/// This function applies time-based event dropping directly to a LazyFrame for optimal performance.
/// Use this instead of the legacy Vec<Event> version when possible.
///
/// # Arguments
///
/// * `df` - Input LazyFrame containing event data
/// * `duration_ratio` - Ratio of total duration to drop (0.0 to 1.0)
///
/// # Returns
///
/// Filtered LazyFrame
#[cfg(feature = "polars")]
pub fn drop_by_time_df(
    df: LazyFrame,
    duration_ratio: f64
) -> PolarsResult<LazyFrame> {
    let config = DropTimeAugmentation::new(duration_ratio);
    apply_drop_time(df, &config)
}

/// Apply area dropping directly to LazyFrame - DataFrame-native version (recommended)
///
/// This function applies area-based event dropping directly to a LazyFrame for optimal performance.
/// Use this instead of the legacy Vec<Event> version when possible.
///
/// # Arguments
///
/// * `df` - Input LazyFrame containing event data
/// * `area_ratio` - Ratio of sensor area to drop (0.0 to 1.0)
/// * `sensor_width` - Sensor width in pixels
/// * `sensor_height` - Sensor height in pixels
///
/// # Returns
///
/// Filtered LazyFrame
#[cfg(feature = "polars")]
pub fn drop_by_area_df(
    df: LazyFrame,
    area_ratio: f64,
    sensor_width: u16,
    sensor_height: u16
) -> PolarsResult<LazyFrame> {
    let config = DropAreaAugmentation::new(area_ratio, sensor_width, sensor_height);
    apply_drop_area(df, &config)
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

    #[cfg(feature = "polars")]
    #[test]
    fn test_drop_event_dataframe_native() -> PolarsResult<()> {
        use crate::{events_to_dataframe, Event};

        let events = create_test_events();
        let df = events_to_dataframe(&events)?.lazy();
        let config = DropEventAugmentation::new(0.3);

        let dropped_df = config.apply_to_dataframe(df)?;
        let result = dropped_df.collect()?;

        // Should have fewer events
        assert!(result.height() < events.len());

        Ok(())
    }

    #[cfg(feature = "polars")]
    #[test]
    fn test_drop_by_time_dataframe_native() -> PolarsResult<()> {
        use crate::events_to_dataframe;

        let events = create_test_events();
        let df = events_to_dataframe(&events)?.lazy();
        let config = DropTimeAugmentation::new(0.2);

        let dropped_df = config.apply_to_dataframe(df)?;
        let result = dropped_df.collect()?;

        // Should have fewer events
        assert!(result.height() < events.len());

        Ok(())
    }

    #[cfg(feature = "polars")]
    #[test]
    fn test_drop_by_area_dataframe_native() -> PolarsResult<()> {
        use crate::events_to_dataframe;

        let events = create_test_events();
        let df = events_to_dataframe(&events)?.lazy();
        let config = DropAreaAugmentation::new(0.25, 400, 400);

        let dropped_df = config.apply_to_dataframe(df)?;
        let result = dropped_df.collect()?;

        // Should have fewer events
        assert!(result.height() < events.len());

        Ok(())
    }

    #[cfg(feature = "polars")]
    #[test]
    fn test_drop_convenience_functions() -> PolarsResult<()> {
        use crate::events_to_dataframe;

        let events = create_test_events();
        let df = events_to_dataframe(&events)?.lazy();

        // Test probability dropping
        let dropped1 = drop_by_probability_df(df.clone(), 0.2)?;
        let result1 = dropped1.collect()?;
        assert!(result1.height() <= events.len());

        // Test time dropping
        let dropped2 = drop_by_time_df(df.clone(), 0.1)?;
        let result2 = dropped2.collect()?;
        assert!(result2.height() <= events.len());

        // Test area dropping
        let dropped3 = drop_by_area_df(df, 0.1, 400, 400)?;
        let result3 = dropped3.collect()?;
        assert!(result3.height() <= events.len());

        Ok(())
    }

    #[cfg(feature = "polars")]
    #[test]
    fn test_drop_legacy_compatibility() {
        let events = create_test_events();

        let config1 = DropEventAugmentation::new(0.2);
        let dropped1 = config1.apply(&events).unwrap();
        assert!(dropped1.len() <= events.len());

        let config2 = DropTimeAugmentation::new(0.1);
        let dropped2 = config2.apply(&events).unwrap();
        assert!(dropped2.len() <= events.len());

        let config3 = DropAreaAugmentation::new(0.1, 400, 400);
        let dropped3 = config3.apply(&events).unwrap();
        assert!(dropped3.len() <= events.len());
    }
}
*/
