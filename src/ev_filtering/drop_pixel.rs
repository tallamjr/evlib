//! Polars-first pixel dropping and masking operations for event camera data
//!
//! This module provides functionality for selectively dropping or masking
//! specific pixels using Polars DataFrames and LazyFrames for maximum performance
//! and memory efficiency. All operations work directly with Polars expressions
//! and avoid manual iteration over events.
//!
//! # Philosophy
//!
//! This implementation follows a strict Polars-first approach:
//! - Input: LazyFrame (from events_to_dataframe)
//! - Processing: Polars expressions, anti-joins, and filtering operations
//! - Output: LazyFrame (convertible to Vec<Event> only when needed)
//!
//! # Performance Benefits
//!
//! - Vectorized operations: All pixel filtering uses SIMD-optimized Polars operations
//! - Anti-join operations: Efficient pixel exclusion without manual checking
//! - Memory efficiency: No intermediate Vec<Event> allocations
//! - Query optimization: Polars optimizes the entire filtering pipeline
//!
//! # Example
//!
//! ```rust
//! use polars::prelude::*;
//! use evlib::ev_filtering::drop_pixel::*;
//!
//! // Convert events to LazyFrame once
//! let events_df = events_to_dataframe(&events)?.lazy();
//!
//! // Apply pixel filtering with Polars expressions
//! let mask = PixelMask::exclude(bad_pixels);
//! let filtered = apply_drop_pixel_filter(events_df, &DropPixelFilter::new(mask))?;
//! ```

use crate::ev_core::{Event, Events};
use crate::ev_filtering::config::Validatable;
use crate::ev_filtering::{FilterError, FilterResult, SingleFilter};
use polars::prelude::*;
use std::collections::HashSet;
#[cfg(feature = "tracing")]
use tracing::{debug, info, instrument, warn};

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

/// Polars column names for event data
pub const COL_X: &str = "x";
pub const COL_Y: &str = "y";
pub const COL_T: &str = "timestamp";
pub const COL_POLARITY: &str = "polarity";

/// Pixel mask representation optimized for Polars operations
#[derive(Debug, Clone)]
pub struct PixelMask {
    /// Set of pixel coordinates to exclude
    pub excluded_pixels: HashSet<(u16, u16)>,
    /// Set of pixel coordinates to include (if specified, only these are kept)
    pub included_pixels: Option<HashSet<(u16, u16)>>,
    /// Optional mask name for identification
    pub name: Option<String>,
}

impl PixelMask {
    /// Create a new empty pixel mask
    pub fn new() -> Self {
        Self {
            excluded_pixels: HashSet::new(),
            included_pixels: None,
            name: None,
        }
    }

    /// Create a mask with excluded pixels
    pub fn exclude(pixels: HashSet<(u16, u16)>) -> Self {
        Self {
            excluded_pixels: pixels,
            included_pixels: None,
            name: None,
        }
    }

    /// Create a mask with only included pixels
    pub fn include_only(pixels: HashSet<(u16, u16)>) -> Self {
        Self {
            excluded_pixels: HashSet::new(),
            included_pixels: Some(pixels),
            name: None,
        }
    }

    /// Create a rectangular mask
    pub fn rectangle(
        min_x: u16,
        max_x: u16,
        min_y: u16,
        max_y: u16,
        exclude: bool,
    ) -> Result<Self, FilterError> {
        if min_x >= max_x || min_y >= max_y {
            return Err(FilterError::InvalidConfig(
                "Invalid rectangle bounds".to_string(),
            ));
        }

        let mut pixels = HashSet::new();
        for x in min_x..=max_x {
            for y in min_y..=max_y {
                pixels.insert((x, y));
            }
        }

        Ok(if exclude {
            Self::exclude(pixels)
        } else {
            Self::include_only(pixels)
        })
    }

    /// Create a circular mask
    pub fn circle(center_x: u16, center_y: u16, radius: u16, exclude: bool) -> Self {
        let mut pixels = HashSet::new();
        let radius_squared = (radius as f64).powi(2);

        // Calculate bounding box for efficiency
        let min_x = center_x.saturating_sub(radius);
        let max_x = center_x.saturating_add(radius);
        let min_y = center_y.saturating_sub(radius);
        let max_y = center_y.saturating_add(radius);

        for x in min_x..=max_x {
            for y in min_y..=max_y {
                let dx = x as f64 - center_x as f64;
                let dy = y as f64 - center_y as f64;
                let distance_squared = dx * dx + dy * dy;

                if distance_squared <= radius_squared {
                    pixels.insert((x, y));
                }
            }
        }

        if exclude {
            Self::exclude(pixels)
        } else {
            Self::include_only(pixels)
        }
    }

    /// Set mask name
    pub fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    /// Add excluded pixels
    pub fn add_excluded(&mut self, pixels: HashSet<(u16, u16)>) {
        self.excluded_pixels.extend(pixels);
    }

    /// Add a single excluded pixel
    pub fn exclude_pixel(&mut self, x: u16, y: u16) {
        self.excluded_pixels.insert((x, y));
    }

    /// Remove excluded pixels
    pub fn remove_excluded(&mut self, pixels: &HashSet<(u16, u16)>) {
        for pixel in pixels {
            self.excluded_pixels.remove(pixel);
        }
    }

    /// Convert this mask to Polars expressions for efficient filtering
    ///
    /// This is the core of the Polars-first approach - we build Polars expressions
    /// that can be optimized and executed efficiently by the Polars query engine.
    pub fn to_polars_expr(&self) -> PolarsResult<Option<Expr>> {
        let mut conditions = Vec::new();

        // Handle excluded pixels using coordinate-based filtering
        if !self.excluded_pixels.is_empty() {
            // Create exclusion conditions for each excluded pixel
            let mut pixel_conditions = Vec::new();
            for (x, y) in &self.excluded_pixels {
                let pixel_condition = col(COL_X)
                    .eq(lit(*x as i64))
                    .and(col(COL_Y).eq(lit(*y as i64)));
                pixel_conditions.push(pixel_condition);
            }

            // Combine all excluded pixel conditions with OR, then negate
            let exclude_condition = if pixel_conditions.len() == 1 {
                pixel_conditions.into_iter().next().unwrap().not()
            } else {
                pixel_conditions
                    .into_iter()
                    .reduce(|acc, cond| acc.or(cond))
                    .unwrap()
                    .not()
            };

            conditions.push(exclude_condition);
        }

        // Handle included pixels (if specified, only these are allowed)
        if let Some(ref included) = self.included_pixels {
            if !included.is_empty() {
                // Create inclusion conditions for each included pixel
                let mut pixel_conditions = Vec::new();
                for (x, y) in included {
                    let pixel_condition = col(COL_X)
                        .eq(lit(*x as i64))
                        .and(col(COL_Y).eq(lit(*y as i64)));
                    pixel_conditions.push(pixel_condition);
                }

                // Combine all included pixel conditions with OR
                let include_condition = if pixel_conditions.len() == 1 {
                    pixel_conditions.into_iter().next().unwrap()
                } else {
                    pixel_conditions
                        .into_iter()
                        .reduce(|acc, cond| acc.or(cond))
                        .unwrap()
                };

                conditions.push(include_condition);
            } else {
                // Empty include set means no events should pass
                conditions.push(lit(false));
            }
        }

        // Combine all conditions with AND
        match conditions.len() {
            0 => Ok(None), // No filtering needed
            1 => Ok(Some(conditions.into_iter().next().unwrap())),
            _ => {
                let combined = conditions
                    .into_iter()
                    .reduce(|acc, cond| acc.and(cond))
                    .unwrap();
                Ok(Some(combined))
            }
        }
    }

    /// Check if a pixel should be dropped (legacy compatibility method)
    pub fn should_drop(&self, x: u16, y: u16) -> bool {
        // Check excluded pixels
        if self.excluded_pixels.contains(&(x, y)) {
            return true;
        }

        // Check included pixels (if specified, only these are allowed)
        if let Some(ref included) = self.included_pixels {
            return !included.contains(&(x, y));
        }

        false
    }

    /// Get the number of excluded pixels
    pub fn excluded_count(&self) -> usize {
        self.excluded_pixels.len()
    }

    /// Get the number of included pixels (if specified)
    pub fn included_count(&self) -> Option<usize> {
        self.included_pixels.as_ref().map(|set| set.len())
    }

    /// Merge with another mask
    pub fn merge(&mut self, other: &PixelMask) {
        // Merge excluded pixels
        self.excluded_pixels.extend(&other.excluded_pixels);

        // Handle included pixels
        if let (Some(self_included), Some(other_included)) =
            (&self.included_pixels, &other.included_pixels)
        {
            // Take intersection of included pixels
            let intersection: HashSet<_> = self_included
                .intersection(other_included)
                .cloned()
                .collect();
            self.included_pixels = Some(intersection);
        } else if other.included_pixels.is_some() {
            // If other has included pixels but self doesn't, use other's
            self.included_pixels = other.included_pixels.clone();
        }
        // If self has included pixels but other doesn't, keep self's
    }

    /// Get description of this mask
    pub fn description(&self) -> String {
        let mut parts = Vec::new();

        if !self.excluded_pixels.is_empty() {
            parts.push(format!("{} excluded pixels", self.excluded_pixels.len()));
        }

        if let Some(ref included) = self.included_pixels {
            parts.push(format!("{} included pixels", included.len()));
        }

        if let Some(ref name) = self.name {
            parts.push(format!("name: {}", name));
        }

        if parts.is_empty() {
            "empty mask".to_string()
        } else {
            parts.join(", ")
        }
    }
}

impl Default for PixelMask {
    fn default() -> Self {
        Self::new()
    }
}

/// Drop pixel filter configuration optimized for Polars operations
#[derive(Debug, Clone)]
pub struct DropPixelFilter {
    /// Pixel mask to apply
    pub mask: PixelMask,
    /// Whether to validate mask consistency
    pub validate_mask: bool,
    /// Whether to log dropped pixel statistics
    pub log_statistics: bool,
}

impl DropPixelFilter {
    /// Create a new drop pixel filter
    pub fn new(mask: PixelMask) -> Self {
        Self {
            mask,
            validate_mask: true,
            log_statistics: true,
        }
    }

    /// Create filter with excluded pixels
    pub fn exclude(pixels: HashSet<(u16, u16)>) -> Self {
        Self::new(PixelMask::exclude(pixels))
    }

    /// Create filter with only included pixels
    pub fn include_only(pixels: HashSet<(u16, u16)>) -> Self {
        Self::new(PixelMask::include_only(pixels))
    }

    /// Create rectangular mask filter
    pub fn rectangle(
        min_x: u16,
        max_x: u16,
        min_y: u16,
        max_y: u16,
        exclude: bool,
    ) -> Result<Self, FilterError> {
        let mask = PixelMask::rectangle(min_x, max_x, min_y, max_y, exclude)?;
        Ok(Self::new(mask))
    }

    /// Create circular mask filter
    pub fn circle(center_x: u16, center_y: u16, radius: u16, exclude: bool) -> Self {
        let mask = PixelMask::circle(center_x, center_y, radius, exclude);
        Self::new(mask)
    }

    /// Set mask validation
    pub fn with_validation(mut self, validate: bool) -> Self {
        self.validate_mask = validate;
        self
    }

    /// Set statistics logging
    pub fn with_statistics_logging(mut self, log_stats: bool) -> Self {
        self.log_statistics = log_stats;
        self
    }

    /// Get description of this filter
    pub fn description(&self) -> String {
        self.mask.description()
    }
}

impl Validatable for DropPixelFilter {
    fn validate(&self) -> FilterResult<()> {
        if self.validate_mask {
            // Check for conflicts between included and excluded pixels
            if let Some(ref included) = self.mask.included_pixels {
                let conflicts: Vec<_> = included.intersection(&self.mask.excluded_pixels).collect();
                if !conflicts.is_empty() {
                    return Err(FilterError::InvalidConfig(format!(
                        "Mask has {} pixels in both included and excluded sets",
                        conflicts.len()
                    )));
                }
            }

            // Check if mask is too restrictive
            if let Some(ref included) = self.mask.included_pixels {
                if included.is_empty() {
                    return Err(FilterError::InvalidConfig(
                        "Mask includes no pixels - all events would be dropped".to_string(),
                    ));
                }
            }
        }

        Ok(())
    }
}

impl SingleFilter for DropPixelFilter {
    fn apply(&self, events: &Events) -> FilterResult<Events> {
        // Legacy Vec<Event> interface - convert to DataFrame and back
        // This is for backward compatibility only
        warn!("Using legacy Vec<Event> interface - consider using LazyFrame directly for better performance");

        let df = crate::ev_core::events_to_dataframe(events)
            .map_err(|e| {
                FilterError::ProcessingError(format!("DataFrame conversion failed: {}", e))
            })?
            .lazy();

        let filtered_df = apply_drop_pixel_filter(df, self)
            .map_err(|e| FilterError::ProcessingError(format!("Polars filtering failed: {}", e)))?;

        // Convert back to Vec<Event> - this is inefficient but maintains compatibility
        let result_df = filtered_df.collect().map_err(|e| {
            FilterError::ProcessingError(format!("LazyFrame collection failed: {}", e))
        })?;

        // Convert DataFrame back to Events
        dataframe_to_events(&result_df)
    }

    fn description(&self) -> String {
        format!("Drop pixel filter: {}", self.description())
    }

    fn is_enabled(&self) -> bool {
        !self.mask.excluded_pixels.is_empty() || self.mask.included_pixels.is_some()
    }
}

/// Apply drop pixel filtering using Polars expressions
///
/// This is the main pixel filtering function that works entirely with Polars
/// operations for maximum performance. It uses anti-join operations and
/// efficient pixel masking instead of manual iteration.
///
/// # Arguments
///
/// * `df` - Input LazyFrame containing event data
/// * `filter` - Drop pixel filter configuration
///
/// # Returns
///
/// Filtered LazyFrame with specified pixels dropped
///
/// # Example
///
/// ```rust
/// use polars::prelude::*;
/// use evlib::ev_filtering::drop_pixel::*;
///
/// let events_df = events_to_dataframe(&events)?.lazy();
/// let filter = DropPixelFilter::exclude(bad_pixels);
/// let filtered = apply_drop_pixel_filter(events_df, &filter)?;
/// ```
#[cfg_attr(feature = "tracing", instrument(skip(df), fields(filter = ?filter.description())))]
pub fn apply_drop_pixel_filter(df: LazyFrame, filter: &DropPixelFilter) -> PolarsResult<LazyFrame> {
    debug!("Applying drop pixel filter: {:?}", filter.description());

    // Validate filter configuration
    if let Err(e) = filter.validate() {
        warn!("Filter validation failed: {}", e);
        return Err(PolarsError::ComputeError(
            format!("Invalid filter: {}", e).into(),
        ));
    }

    match filter.mask.to_polars_expr()? {
        Some(expr) => {
            debug!("Drop pixel filter expression: {:?}", expr);

            if filter.log_statistics {
                // For statistics, we need to collect both original and filtered counts
                // This is done lazily and only when logging is enabled
                let original_count = df
                    .clone()
                    .select([len().alias("count")])
                    .collect()?
                    .get_row(0)?
                    .0[0]
                    .try_extract::<u32>()?;
                let filtered_df = df.filter(expr);
                let filtered_count = filtered_df
                    .clone()
                    .select([len().alias("count")])
                    .collect()?
                    .get_row(0)?
                    .0[0]
                    .try_extract::<u32>()?;
                let dropped_count = original_count - filtered_count;

                info!(
                    "Drop pixel filtering: {} -> {} events ({} dropped, {:.1}% reduction) - {}",
                    original_count,
                    filtered_count,
                    dropped_count,
                    (dropped_count as f64 / original_count as f64) * 100.0,
                    filter.description()
                );

                Ok(filtered_df)
            } else {
                Ok(df.filter(expr))
            }
        }
        None => {
            debug!("No pixel filtering needed");
            Ok(df)
        }
    }
}

/// Filter events by pixel coordinates using anti-join operations
///
/// This function uses Polars anti-join operations for efficient pixel exclusion
/// instead of manual checking.
///
/// # Arguments
///
/// * `df` - Input LazyFrame
/// * `excluded_pixels` - Set of (x, y) coordinates to exclude
///
/// # Returns
///
/// Filtered LazyFrame
#[cfg_attr(feature = "tracing", instrument(skip(df)))]
pub fn filter_excluded_pixels(
    df: LazyFrame,
    excluded_pixels: &HashSet<(u16, u16)>,
) -> PolarsResult<LazyFrame> {
    if excluded_pixels.is_empty() {
        return Ok(df);
    }

    // Convert excluded pixels to Polars DataFrame for anti-join
    let excluded_coords: Vec<(i64, i64)> = excluded_pixels
        .iter()
        .map(|(x, y)| (*x as i64, *y as i64))
        .collect();

    let excluded_x: Vec<i64> = excluded_coords.iter().map(|(x, _)| *x).collect();
    let excluded_y: Vec<i64> = excluded_coords.iter().map(|(_, y)| *y).collect();

    let excluded_df = df!(
        COL_X => excluded_x,
        COL_Y => excluded_y,
    )?;

    // Use anti-join to exclude pixels efficiently
    // Use left join with suffix and filter null matches (anti-join behavior for Polars 0.49.1)
    let joined = df.join(
        excluded_df.lazy(),
        [col(COL_X), col(COL_Y)],
        [col(COL_X), col(COL_Y)],
        JoinArgs::new(JoinType::Left).with_suffix(Some("_right".into())),
    );

    // Filter out rows where the right-side x column is not null (these are matches to exclude)
    Ok(joined.filter(col(format!("{}_right", COL_X)).is_null()))
}

/// Filter events to include only specific pixels using inner join
///
/// # Arguments
///
/// * `df` - Input LazyFrame
/// * `included_pixels` - Set of (x, y) coordinates to include
///
/// # Returns
///
/// Filtered LazyFrame
#[cfg_attr(feature = "tracing", instrument(skip(df)))]
pub fn filter_included_pixels(
    df: LazyFrame,
    included_pixels: &HashSet<(u16, u16)>,
) -> PolarsResult<LazyFrame> {
    if included_pixels.is_empty() {
        // Empty include set means no events should pass
        return Ok(df.filter(lit(false)));
    }

    // Convert included pixels to Polars DataFrame for inner join
    let included_coords: Vec<(i64, i64)> = included_pixels
        .iter()
        .map(|(x, y)| (*x as i64, *y as i64))
        .collect();

    let included_x: Vec<i64> = included_coords.iter().map(|(x, _)| *x).collect();
    let included_y: Vec<i64> = included_coords.iter().map(|(_, y)| *y).collect();

    let included_df = df!(
        COL_X => included_x,
        COL_Y => included_y,
    )?;

    // Use inner join to include only specified pixels efficiently
    Ok(df
        .join(
            included_df.lazy(),
            [col(COL_X), col(COL_Y)],
            [col(COL_X), col(COL_Y)],
            JoinArgs::new(JoinType::Inner),
        )
        .select([col("*").exclude(["x_right", "y_right"])]))
}

/// Convert DataFrame back to Events vector (for compatibility)
fn dataframe_to_events(df: &DataFrame) -> FilterResult<Events> {
    let height = df.height();
    let mut events = Vec::with_capacity(height);

    let x_series = df
        .column(COL_X)
        .map_err(|e| FilterError::ProcessingError(format!("Missing x column: {}", e)))?;
    let y_series = df
        .column(COL_Y)
        .map_err(|e| FilterError::ProcessingError(format!("Missing y column: {}", e)))?;
    let t_series = df
        .column(COL_T)
        .map_err(|e| FilterError::ProcessingError(format!("Missing t column: {}", e)))?;
    let p_series = df
        .column(COL_POLARITY)
        .map_err(|e| FilterError::ProcessingError(format!("Missing polarity column: {}", e)))?;

    let x_values = x_series
        .i64()
        .map_err(|e| FilterError::ProcessingError(format!("X column type error: {}", e)))?;
    let y_values = y_series
        .i64()
        .map_err(|e| FilterError::ProcessingError(format!("Y column type error: {}", e)))?;
    let t_values = t_series
        .f64()
        .map_err(|e| FilterError::ProcessingError(format!("T column type error: {}", e)))?;
    let p_values = p_series
        .i64()
        .map_err(|e| FilterError::ProcessingError(format!("Polarity column type error: {}", e)))?;

    for i in 0..height {
        let x = x_values.get(i).ok_or_else(|| {
            FilterError::ProcessingError(format!("Missing x value at index {}", i))
        })? as u16;
        let y = y_values.get(i).ok_or_else(|| {
            FilterError::ProcessingError(format!("Missing y value at index {}", i))
        })? as u16;
        let t = t_values.get(i).ok_or_else(|| {
            FilterError::ProcessingError(format!("Missing t value at index {}", i))
        })?;
        let polarity = p_values.get(i).ok_or_else(|| {
            FilterError::ProcessingError(format!("Missing polarity value at index {}", i))
        })? != 0;

        events.push(Event { t, x, y, polarity });
    }

    Ok(events)
}

/// Legacy function - delegates to Polars implementation
pub fn drop_pixels(events: &Events, pixels: HashSet<(u16, u16)>) -> FilterResult<Events> {
    let filter = DropPixelFilter::exclude(pixels);
    filter.apply(events)
}

/// Create a pixel mask from events based on activity using Polars aggregations
pub fn create_pixel_mask(
    events: &Events,
    min_activity: usize,
    max_activity: Option<usize>,
) -> FilterResult<PixelMask> {
    // Convert to DataFrame for Polars operations
    let df = crate::ev_core::events_to_dataframe(events)
        .map_err(|e| FilterError::ProcessingError(format!("DataFrame conversion failed: {}", e)))?;

    // Calculate pixel activity using Polars groupby aggregation
    let pixel_stats = df
        .lazy()
        .group_by([col(COL_X), col(COL_Y)])
        .agg([len().alias("activity")])
        .collect()
        .map_err(|e| {
            FilterError::ProcessingError(format!("Pixel stats calculation failed: {}", e))
        })?;

    let mut excluded_pixels = HashSet::new();

    let x_values = pixel_stats
        .column(COL_X)
        .map_err(|e| FilterError::ProcessingError(format!("Missing x column: {}", e)))?
        .i64()
        .map_err(|e| FilterError::ProcessingError(format!("X column type error: {}", e)))?;
    let y_values = pixel_stats
        .column(COL_Y)
        .map_err(|e| FilterError::ProcessingError(format!("Missing y column: {}", e)))?
        .i64()
        .map_err(|e| FilterError::ProcessingError(format!("Y column type error: {}", e)))?;
    let activity_values = pixel_stats
        .column("activity")
        .map_err(|e| FilterError::ProcessingError(format!("Missing activity column: {}", e)))?
        .u32()
        .map_err(|e| FilterError::ProcessingError(format!("Activity column type error: {}", e)))?;

    for i in 0..pixel_stats.height() {
        let x = x_values.get(i).unwrap() as u16;
        let y = y_values.get(i).unwrap() as u16;
        let activity = activity_values.get(i).unwrap() as usize;

        // Check minimum activity
        if activity < min_activity {
            excluded_pixels.insert((x, y));
            continue;
        }

        // Check maximum activity
        if let Some(max_act) = max_activity {
            if activity > max_act {
                excluded_pixels.insert((x, y));
            }
        }
    }

    Ok(PixelMask::exclude(excluded_pixels))
}

/// Load pixel mask from coordinates list
pub fn load_pixel_mask_from_coords(coords: Vec<(u16, u16)>, exclude: bool) -> PixelMask {
    let pixels: HashSet<_> = coords.into_iter().collect();

    if exclude {
        PixelMask::exclude(pixels)
    } else {
        PixelMask::include_only(pixels)
    }
}

/// Save pixel mask coordinates to vector
pub fn save_pixel_mask_coords(mask: &PixelMask) -> Vec<(u16, u16)> {
    let mut coords = Vec::new();

    // Add excluded pixels
    coords.extend(mask.excluded_pixels.iter().cloned());

    // If there are included pixels, add them with a marker (could be extended)
    if let Some(ref included) = mask.included_pixels {
        coords.extend(included.iter().cloned());
    }

    coords.sort_unstable();
    coords
}

/// Create mask from bad pixel list (common format)
pub fn create_bad_pixel_mask(bad_pixels: &[(u16, u16)]) -> PixelMask {
    let excluded_pixels: HashSet<_> = bad_pixels.iter().cloned().collect();
    PixelMask::exclude(excluded_pixels)
}

/// Create mask from sensor region
pub fn create_sensor_mask(
    sensor_width: u16,
    sensor_height: u16,
    border_size: u16,
) -> Result<PixelMask, FilterError> {
    if border_size * 2 >= sensor_width || border_size * 2 >= sensor_height {
        return Err(FilterError::InvalidConfig(
            "Border size too large for sensor dimensions".to_string(),
        ));
    }

    let mut excluded_pixels = HashSet::new();

    // Add border pixels
    for x in 0..sensor_width {
        for y in 0..sensor_height {
            if x < border_size
                || x >= sensor_width - border_size
                || y < border_size
                || y >= sensor_height - border_size
            {
                excluded_pixels.insert((x, y));
            }
        }
    }

    Ok(PixelMask::exclude(excluded_pixels).with_name("sensor_border".to_string()))
}

/// Get pixel statistics using Polars aggregations
///
/// This function efficiently computes pixel-level statistics using Polars
/// groupby and aggregation operations.
///
/// # Arguments
///
/// * `df` - Input LazyFrame
///
/// # Returns
///
/// DataFrame containing per-pixel statistics
#[cfg_attr(feature = "tracing", instrument(skip(df)))]
pub fn get_pixel_statistics(df: LazyFrame) -> PolarsResult<DataFrame> {
    df.group_by([col(COL_X), col(COL_Y)])
        .agg([
            len().alias("total_events"),
            col(COL_POLARITY).sum().alias("positive_events"),
            (len() - col(COL_POLARITY).sum()).alias("negative_events"),
            col(COL_T).min().alias("first_event_time"),
            col(COL_T).max().alias("last_event_time"),
            (col(COL_T).max() - col(COL_T).min()).alias("activity_duration"),
        ])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ev_core::Event;

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
                x: 100,
                y: 200,
                polarity: false,
            }, // Same pixel as first
            Event {
                t: 5.0,
                x: 300,
                y: 400,
                polarity: true,
            },
        ]
    }

    #[test]
    fn test_pixel_mask_creation() {
        let mask = PixelMask::new();
        assert!(mask.excluded_pixels.is_empty());
        assert!(mask.included_pixels.is_none());

        let excluded = [(100u16, 200u16), (150u16, 250u16)]
            .iter()
            .cloned()
            .collect();
        let mask = PixelMask::exclude(excluded);
        assert_eq!(mask.excluded_count(), 2);

        let included = [(100u16, 200u16), (300u16, 400u16)]
            .iter()
            .cloned()
            .collect();
        let mask = PixelMask::include_only(included);
        assert_eq!(mask.included_count(), Some(2));
    }

    #[test]
    fn test_rectangular_mask() {
        let mask = PixelMask::rectangle(100, 200, 150, 250, true).unwrap();

        assert!(mask.should_drop(150, 200)); // Inside rectangle
        assert!(!mask.should_drop(50, 100)); // Outside rectangle

        // Test invalid rectangle
        assert!(PixelMask::rectangle(200, 100, 150, 250, true).is_err());
    }

    #[test]
    fn test_circular_mask() {
        let mask = PixelMask::circle(100, 100, 10, true);

        assert!(mask.should_drop(100, 100)); // Center
        assert!(mask.should_drop(105, 105)); // Inside circle
        assert!(!mask.should_drop(200, 200)); // Outside circle
    }

    #[test]
    fn test_drop_pixel_filtering() {
        let events = create_test_events();
        let original_count = events.len();

        // Drop pixel (100, 200)
        let mut excluded = HashSet::new();
        excluded.insert((100, 200));

        let filtered = drop_pixels(&events, excluded).unwrap();

        // Should remove 2 events from pixel (100, 200)
        assert_eq!(filtered.len(), original_count - 2);
        assert!(!filtered.iter().any(|e| e.x == 100 && e.y == 200));
    }

    #[test]
    fn test_include_only_filtering() {
        let events = create_test_events();

        // Only include pixels (100, 200) and (150, 250)
        let mut included = HashSet::new();
        included.insert((100, 200));
        included.insert((150, 250));

        let filter = DropPixelFilter::include_only(included);
        let filtered = filter.apply(&events).unwrap();

        // Should only keep events from included pixels
        assert_eq!(filtered.len(), 3); // 2 from (100,200) + 1 from (150,250)
        assert!(filtered
            .iter()
            .all(|e| (e.x == 100 && e.y == 200) || (e.x == 150 && e.y == 250)));
    }

    #[test]
    fn test_mask_merging() {
        let mut mask1 = PixelMask::exclude([(100u16, 200u16)].iter().cloned().collect());
        let mask2 = PixelMask::exclude([(150u16, 250u16)].iter().cloned().collect());

        mask1.merge(&mask2);

        assert_eq!(mask1.excluded_count(), 2);
        assert!(mask1.should_drop(100, 200));
        assert!(mask1.should_drop(150, 250));
    }

    #[test]
    fn test_pixel_mask_validation() {
        // Valid mask
        let mask = PixelMask::exclude([(100u16, 200u16)].iter().cloned().collect());
        let filter = DropPixelFilter::new(mask);
        assert!(filter.validate().is_ok());

        // Invalid mask (empty included pixels)
        let mask = PixelMask::include_only(HashSet::new());
        let filter = DropPixelFilter::new(mask);
        assert!(filter.validate().is_err());
    }

    #[test]
    fn test_activity_based_mask() {
        let events = create_test_events();

        // Create mask for pixels with less than 2 events
        let mask = create_pixel_mask(&events, 2, None).unwrap();

        // Should exclude pixels with only 1 event
        assert!(mask.should_drop(150, 250)); // Has 1 event
        assert!(mask.should_drop(200, 300)); // Has 1 event
        assert!(mask.should_drop(300, 400)); // Has 1 event
        assert!(!mask.should_drop(100, 200)); // Has 2 events
    }

    #[test]
    fn test_bad_pixel_mask() {
        let bad_pixels = [(100u16, 200u16), (300u16, 400u16)];
        let mask = create_bad_pixel_mask(&bad_pixels);

        assert!(mask.should_drop(100, 200));
        assert!(mask.should_drop(300, 400));
        assert!(!mask.should_drop(150, 250));
    }

    #[test]
    fn test_sensor_border_mask() {
        let mask = create_sensor_mask(640, 480, 10).unwrap();

        // Border pixels should be dropped
        assert!(mask.should_drop(5, 5)); // Top-left border
        assert!(mask.should_drop(635, 475)); // Bottom-right border

        // Interior pixels should not be dropped
        assert!(!mask.should_drop(320, 240)); // Center
        assert!(!mask.should_drop(50, 50)); // Interior but near border
    }

    #[test]
    fn test_mask_coordinates_save_load() {
        let original_coords = vec![(100u16, 200u16), (150u16, 250u16), (200u16, 300u16)];
        let mask = load_pixel_mask_from_coords(original_coords.clone(), true);
        let saved_coords = save_pixel_mask_coords(&mask);

        // Should contain all original coordinates
        for coord in &original_coords {
            assert!(saved_coords.contains(coord));
        }
    }

    #[test]
    fn test_empty_events() {
        let events = Vec::new();
        let mask = PixelMask::exclude([(100u16, 200u16)].iter().cloned().collect());
        let filter = DropPixelFilter::new(mask);
        let filtered = filter.apply(&events).unwrap();
        assert!(filtered.is_empty());
    }

    #[test]
    fn test_mask_description() {
        let mask = PixelMask::exclude([(100u16, 200u16)].iter().cloned().collect())
            .with_name("test_mask".to_string());
        let description = mask.description();

        assert!(description.contains("1 excluded pixels"));
        assert!(description.contains("test_mask"));
    }

    #[test]
    fn test_filter_statistics_logging() {
        let events = create_test_events();
        let mask = PixelMask::exclude([(100u16, 200u16)].iter().cloned().collect());
        let filter = DropPixelFilter::new(mask).with_statistics_logging(true);

        let _filtered = filter.apply(&events).unwrap();
        // Statistics should be logged (check via log capture in real tests)
    }

    #[test]
    fn test_sensor_mask_invalid_border() {
        // Border too large for sensor
        let result = create_sensor_mask(100, 100, 60);
        assert!(result.is_err());
    }

    #[cfg(feature = "polars")]
    #[test]
    fn test_polars_pixel_filtering() {
        let events = create_test_events();

        // Convert to DataFrame for Polars operations
        let df = crate::ev_core::events_to_dataframe(&events).unwrap().lazy();

        // Test excluded pixels filtering
        let mut excluded = HashSet::new();
        excluded.insert((100, 200));

        let filtered_df = filter_excluded_pixels(df.clone(), &excluded).unwrap();
        let result_df = filtered_df.collect().unwrap();

        // Should have 3 events remaining (original 5 minus 2 from pixel (100,200))
        assert_eq!(result_df.height(), 3);

        // Test included pixels filtering
        let mut included = HashSet::new();
        included.insert((100, 200));
        included.insert((150, 250));

        let filtered_df = filter_included_pixels(df, &included).unwrap();
        let result_df = filtered_df.collect().unwrap();

        // Should have 3 events (2 from (100,200) + 1 from (150,250))
        assert_eq!(result_df.height(), 3);
    }
}
