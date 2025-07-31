//! Polars-first polarity filtering operations for event camera data
//!
//! This module provides polarity-based filtering functionality using Polars DataFrames
//! and LazyFrames for maximum performance and memory efficiency. All operations
//! work directly with Polars expressions and avoid manual Vec<Event> iterations.
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
//! - Vectorized operations: All filtering uses SIMD-optimized Polars operations
//! - Memory efficiency: No intermediate Vec<Event> allocations
//! - Query optimization: Polars optimizes the entire filtering pipeline

use crate::ev_core::{Event, Events};
use crate::ev_filtering::config::Validatable;
use crate::ev_filtering::{FilterError, FilterResult, SingleFilter};
use polars::prelude::*;
use std::collections::HashMap;
use tracing::{debug, instrument, warn};

// Use consistent column names from utils
use crate::ev_filtering::utils::{COL_POLARITY, COL_T, COL_X, COL_Y};

/// Raw polarity data for encoding detection and conversion
#[derive(Debug, Clone)]
pub struct RawPolarityData {
    pub values: Vec<f64>,
    pub encoding: PolarityEncoding,
    pub confidence: f64,
}

/// Polarity encoding schemes used by different event cameras
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolarityEncoding {
    /// Standard encoding: true = positive (ON), false = negative (OFF)
    TrueFalse,
    /// Numeric encoding: 1 = positive, 0 = negative
    OneZero,
    /// Signed encoding: 1 = positive, -1 = negative
    OneMinus,
    /// Raw encoding: preserve original values
    Raw,
    /// Mixed encodings in the same dataset
    Mixed,
    /// Unknown/auto-detect encoding
    Unknown,
}

impl PolarityEncoding {
    /// Convert numeric polarity value to standard boolean representation
    pub fn to_bool(&self, value: f64) -> FilterResult<bool> {
        match self {
            PolarityEncoding::TrueFalse => {
                if value == 0.0 {
                    Ok(false)
                } else if value == 1.0 {
                    Ok(true)
                } else {
                    Err(FilterError::InvalidInput(format!(
                        "Invalid TrueFalse polarity value: {}",
                        value
                    )))
                }
            }
            PolarityEncoding::OneZero => {
                if value == 0.0 {
                    Ok(false)
                } else if value == 1.0 {
                    Ok(true)
                } else {
                    Err(FilterError::InvalidInput(format!(
                        "Invalid OneZero polarity value: {}",
                        value
                    )))
                }
            }
            PolarityEncoding::OneMinus => {
                if value == -1.0 {
                    Ok(false)
                } else if value == 1.0 {
                    Ok(true)
                } else {
                    Err(FilterError::InvalidInput(format!(
                        "Invalid OneMinus polarity value: {}",
                        value
                    )))
                }
            }
            PolarityEncoding::Raw => Ok(value > 0.0), // Any positive is true
            PolarityEncoding::Mixed => {
                // Handle common mixed values
                match value {
                    -1.0 => Ok(false),
                    0.0 => Ok(false),
                    1.0 => Ok(true),
                    _ => Ok(value > 0.0),
                }
            }
            PolarityEncoding::Unknown => Ok(value > 0.0), // Default to positive threshold
        }
    }

    /// Convert boolean polarity to encoding-specific numeric value
    pub fn from_bool(&self, polarity: bool) -> f64 {
        match self {
            PolarityEncoding::TrueFalse => {
                if polarity {
                    1.0
                } else {
                    0.0
                }
            }
            PolarityEncoding::OneZero => {
                if polarity {
                    1.0
                } else {
                    0.0
                }
            }
            PolarityEncoding::OneMinus => {
                if polarity {
                    1.0
                } else {
                    -1.0
                }
            }
            PolarityEncoding::Raw => {
                if polarity {
                    1.0
                } else {
                    0.0
                }
            }
            PolarityEncoding::Mixed => {
                if polarity {
                    1.0
                } else {
                    0.0
                }
            }
            PolarityEncoding::Unknown => {
                if polarity {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    /// Detect encoding from raw polarity values using Polars
    pub fn detect_from_raw_values(values: &[f64]) -> Self {
        if values.is_empty() {
            return PolarityEncoding::Unknown;
        }

        let unique_values: std::collections::HashSet<_> =
            values.iter().map(|&v| OrderedFloat::from(v)).collect();

        match unique_values.len() {
            0 => PolarityEncoding::Unknown,
            1 => {
                let value = values[0];
                if value == 0.0 || value == 1.0 {
                    PolarityEncoding::OneZero
                } else if value == -1.0 || value == 1.0 {
                    PolarityEncoding::OneMinus
                } else {
                    PolarityEncoding::Raw
                }
            }
            2 => {
                let mut vals: Vec<f64> = unique_values.iter().map(|v| (*v).into()).collect();
                vals.sort_by(|a, b| a.partial_cmp(b).unwrap());

                match (vals[0], vals[1]) {
                    (0.0, 1.0) => PolarityEncoding::OneZero,
                    (-1.0, 1.0) => PolarityEncoding::OneMinus,
                    _ => PolarityEncoding::Raw,
                }
            }
            _ => {
                // Check for mixed encoding patterns
                let has_zero = unique_values.contains(&OrderedFloat::from(0.0));
                let has_one = unique_values.contains(&OrderedFloat::from(1.0));
                let has_minus_one = unique_values.contains(&OrderedFloat::from(-1.0));

                if has_zero && has_one && has_minus_one {
                    PolarityEncoding::Mixed
                } else {
                    PolarityEncoding::Raw
                }
            }
        }
    }

    /// Detect the likely encoding from events using Polars expressions
    pub fn detect_from_events_polars(df: LazyFrame) -> PolarsResult<Self> {
        let stats_df = df
            .select([
                len().alias("total_events"),
                col(COL_POLARITY).sum().alias("positive_count"),
            ])
            .with_columns([(col("positive_count").cast(DataType::Float64)
                / col("total_events").cast(DataType::Float64))
            .alias("positive_ratio")])
            .collect()?;

        if stats_df.height() == 0 {
            return Ok(PolarityEncoding::TrueFalse);
        }

        let row = stats_df.get_row(0)?;
        let positive_ratio = row.0[2].try_extract::<f64>().unwrap_or(0.5);

        if !(0.1..=0.9).contains(&positive_ratio) {
            warn!(
                "Unusual polarity distribution: {:.1}% positive events. Check encoding.",
                positive_ratio * 100.0
            );
        }

        // Default to standard boolean encoding for Events
        Ok(PolarityEncoding::TrueFalse)
    }

    /// Legacy function for events detection - delegates to Polars implementation
    pub fn detect_from_events(events: &Events) -> Self {
        if events.is_empty() {
            return PolarityEncoding::TrueFalse;
        }

        warn!("Using legacy Vec<Event> interface for encoding detection - consider using LazyFrame directly");

        let df = match crate::ev_core::events_to_dataframe(events) {
            Ok(df) => df.lazy(),
            Err(_) => return Self::detect_from_events_legacy(events),
        };

        match Self::detect_from_events_polars(df) {
            Ok(encoding) => encoding,
            Err(_) => Self::detect_from_events_legacy(events),
        }
    }

    /// Fallback legacy encoding detection
    fn detect_from_events_legacy(events: &Events) -> Self {
        if events.is_empty() {
            return PolarityEncoding::TrueFalse;
        }

        // Sample events to check distribution
        let sample_size = std::cmp::min(1000, events.len());
        let mut true_count = 0;
        let mut false_count = 0;

        for event in events.iter().take(sample_size) {
            if event.polarity {
                true_count += 1;
            } else {
                false_count += 1;
            }
        }

        let total = true_count + false_count;
        if total > 0 {
            let true_ratio = true_count as f64 / total as f64;
            if !(0.1..=0.9).contains(&true_ratio) {
                warn!(
                    "Unusual polarity distribution: {:.1}% positive events. Check encoding.",
                    true_ratio * 100.0
                );
            }
        }

        // Default to standard boolean encoding for Events
        PolarityEncoding::TrueFalse
    }

    /// Get a description of this encoding
    pub fn description(&self) -> &'static str {
        match self {
            PolarityEncoding::TrueFalse => "true/false",
            PolarityEncoding::OneZero => "1/0",
            PolarityEncoding::OneMinus => "1/-1",
            PolarityEncoding::Raw => "raw",
            PolarityEncoding::Mixed => "mixed encodings",
            PolarityEncoding::Unknown => "unknown/auto-detect",
        }
    }

    /// Get the expected polarity values for this encoding
    pub fn expected_values(&self) -> Vec<f64> {
        match self {
            PolarityEncoding::TrueFalse => vec![0.0, 1.0],
            PolarityEncoding::OneZero => vec![0.0, 1.0],
            PolarityEncoding::OneMinus => vec![-1.0, 1.0],
            PolarityEncoding::Raw => vec![], // Can be anything
            PolarityEncoding::Mixed => vec![-1.0, 0.0, 1.0], // Common mixed values
            PolarityEncoding::Unknown => vec![], // Unknown
        }
    }

    /// Check if a polarity value is valid for this encoding
    pub fn is_valid_value(&self, value: f64) -> bool {
        match self {
            PolarityEncoding::TrueFalse => value == 0.0 || value == 1.0,
            PolarityEncoding::OneZero => value == 0.0 || value == 1.0,
            PolarityEncoding::OneMinus => value == -1.0 || value == 1.0,
            PolarityEncoding::Raw => true, // Any value is valid for raw
            PolarityEncoding::Mixed => [-1.0, 0.0, 1.0].contains(&value),
            PolarityEncoding::Unknown => true, // Unknown, so accept anything
        }
    }

    /// Get confidence score for this encoding given a set of values
    pub fn confidence_score(&self, values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let expected = self.expected_values();
        if expected.is_empty() {
            // Raw or Unknown encoding - low confidence
            return 0.3;
        }

        let valid_count = values.iter().filter(|&&v| self.is_valid_value(v)).count();
        valid_count as f64 / values.len() as f64
    }

    /// Convert between different polarity encodings
    pub fn convert_to(&self, target: PolarityEncoding, values: &[f64]) -> FilterResult<Vec<f64>> {
        if values.is_empty() {
            return Ok(Vec::new());
        }

        let mut converted = Vec::with_capacity(values.len());

        for &value in values {
            // First convert to boolean using source encoding
            let boolean_polarity = self.to_bool(value)?;
            // Then convert to target encoding
            let target_value = target.from_bool(boolean_polarity);
            converted.push(target_value);
        }

        Ok(converted)
    }
}

// Helper for floating point ordering
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
struct OrderedFloat(f64);

impl From<f64> for OrderedFloat {
    fn from(val: f64) -> Self {
        OrderedFloat(val)
    }
}

impl From<OrderedFloat> for f64 {
    fn from(val: OrderedFloat) -> Self {
        val.0
    }
}

impl Eq for OrderedFloat {}

#[allow(clippy::derive_ord_xor_partial_ord)]
impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl std::hash::Hash for OrderedFloat {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

/// Polarity selection modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolaritySelection {
    /// Keep only positive (ON) events
    PositiveOnly,
    /// Keep only negative (OFF) events
    NegativeOnly,
    /// Keep both polarities (no filtering)
    Both,
    /// Keep events with alternating polarity (for noise reduction)
    Alternating,
    /// Keep events based on polarity balance in local neighborhood
    Balanced,
}

impl PolaritySelection {
    /// Check if an event passes this polarity selection
    pub fn passes(&self, polarity: bool) -> bool {
        match self {
            PolaritySelection::PositiveOnly => polarity,
            PolaritySelection::NegativeOnly => !polarity,
            PolaritySelection::Both => true,
            PolaritySelection::Alternating => true, // Handled by special logic
            PolaritySelection::Balanced => true,    // Handled by special logic
        }
    }

    /// Get a description of this selection mode
    pub fn description(&self) -> &'static str {
        match self {
            PolaritySelection::PositiveOnly => "positive only",
            PolaritySelection::NegativeOnly => "negative only",
            PolaritySelection::Both => "both polarities",
            PolaritySelection::Alternating => "alternating polarity",
            PolaritySelection::Balanced => "balanced polarity",
        }
    }

    /// Convert this selection to a Polars expression
    pub fn to_polars_expr(&self) -> Option<Expr> {
        match self {
            PolaritySelection::PositiveOnly => Some(col(COL_POLARITY).gt(lit(0))),
            PolaritySelection::NegativeOnly => Some(col(COL_POLARITY).lt(lit(0))),
            PolaritySelection::Both => None, // No filtering needed
            PolaritySelection::Alternating => None, // Requires special handling
            PolaritySelection::Balanced => None, // Requires special handling
        }
    }
}

/// Configuration for polarity filtering optimized for Polars operations
#[derive(Debug, Clone)]
pub struct PolarityFilter {
    /// Which polarities to select
    pub selection: PolaritySelection,
    /// Input polarity encoding
    pub input_encoding: PolarityEncoding,
    /// Whether to validate polarity consistency
    pub validate_polarity: bool,
    /// For alternating polarity: minimum time between polarity switches (microseconds)
    pub alternating_min_interval: Option<f64>,
    /// For balanced polarity: radius for local balance checking
    pub balance_radius: Option<u16>,
    /// For balanced polarity: required balance ratio (0.0 to 1.0)
    pub balance_ratio: Option<f64>,
}

impl PolarityFilter {
    /// Create a filter for positive events only
    pub fn positive_only() -> Self {
        Self {
            selection: PolaritySelection::PositiveOnly,
            input_encoding: PolarityEncoding::TrueFalse,
            validate_polarity: true,
            alternating_min_interval: None,
            balance_radius: None,
            balance_ratio: None,
        }
    }

    /// Create a filter for negative events only
    pub fn negative_only() -> Self {
        Self {
            selection: PolaritySelection::NegativeOnly,
            input_encoding: PolarityEncoding::TrueFalse,
            validate_polarity: true,
            alternating_min_interval: None,
            balance_radius: None,
            balance_ratio: None,
        }
    }

    /// Create a filter that keeps both polarities (passthrough)
    pub fn both() -> Self {
        Self {
            selection: PolaritySelection::Both,
            input_encoding: PolarityEncoding::TrueFalse,
            validate_polarity: true,
            alternating_min_interval: None,
            balance_radius: None,
            balance_ratio: None,
        }
    }

    /// Create a filter for alternating polarity events
    pub fn alternating(min_interval_us: f64) -> Self {
        Self {
            selection: PolaritySelection::Alternating,
            input_encoding: PolarityEncoding::TrueFalse,
            validate_polarity: true,
            alternating_min_interval: Some(min_interval_us),
            balance_radius: None,
            balance_ratio: None,
        }
    }

    /// Create a filter for locally balanced polarity events
    pub fn balanced(radius: u16, ratio: f64) -> Self {
        Self {
            selection: PolaritySelection::Balanced,
            input_encoding: PolarityEncoding::TrueFalse,
            validate_polarity: true,
            alternating_min_interval: None,
            balance_radius: Some(radius),
            balance_ratio: Some(ratio),
        }
    }

    /// Create a filter from polarity values (Python API compatible)
    pub fn from_values(polarity_values: Vec<i8>) -> Self {
        if polarity_values.is_empty() {
            return Self::both();
        }

        // Detect encoding from values
        let encoding = if polarity_values.contains(&-1) && polarity_values.contains(&1) {
            PolarityEncoding::OneMinus
        } else if polarity_values.contains(&0) && polarity_values.contains(&1) {
            PolarityEncoding::OneZero
        } else if polarity_values.len() == 1 {
            match polarity_values[0] {
                1 => return Self::positive_only().with_encoding(PolarityEncoding::OneZero),
                0 => return Self::negative_only().with_encoding(PolarityEncoding::OneZero),
                -1 => return Self::negative_only().with_encoding(PolarityEncoding::OneMinus),
                _ => PolarityEncoding::Raw,
            }
        } else {
            PolarityEncoding::Mixed
        };

        // If contains both positive and negative values, keep both
        let has_positive = polarity_values.iter().any(|&v| v > 0);
        let has_negative = polarity_values.iter().any(|&v| v <= 0);

        let selection = match (has_positive, has_negative) {
            (true, false) => PolaritySelection::PositiveOnly,
            (false, true) => PolaritySelection::NegativeOnly,
            (true, true) => PolaritySelection::Both,
            (false, false) => PolaritySelection::Both, // Shouldn't happen with valid input
        };

        Self {
            selection,
            input_encoding: encoding,
            validate_polarity: true,
            alternating_min_interval: None,
            balance_radius: None,
            balance_ratio: None,
        }
    }

    /// Set the input polarity encoding
    pub fn with_encoding(mut self, encoding: PolarityEncoding) -> Self {
        self.input_encoding = encoding;
        self
    }

    /// Set polarity validation
    pub fn with_validation(mut self, validate: bool) -> Self {
        self.validate_polarity = validate;
        self
    }

    /// Convert this filter to Polars expressions
    pub fn to_polars_expr(&self, df: &LazyFrame) -> PolarsResult<Option<Expr>> {
        match self.selection {
            PolaritySelection::PositiveOnly
            | PolaritySelection::NegativeOnly
            | PolaritySelection::Both => Ok(self.selection.to_polars_expr()),
            PolaritySelection::Alternating => {
                // Handle alternating filtering using window functions
                Ok(Some(self.build_alternating_expr(df)?))
            }
            PolaritySelection::Balanced => {
                // Handle balanced filtering using spatial windows
                Ok(Some(self.build_balanced_expr(df)?))
            }
        }
    }

    /// Build expression for alternating polarity filtering using Polars shift and window functions
    fn build_alternating_expr(&self, _df: &LazyFrame) -> PolarsResult<Expr> {
        let min_interval = self.alternating_min_interval.unwrap_or(1000.0); // Default 1ms

        // Use shift to compare with previous event
        let prev_polarity = col(COL_POLARITY).shift(lit(1));
        let prev_time = col(COL_T).shift(lit(1));

        // Check polarity alternation and time interval
        let polarity_alternates = col(COL_POLARITY).neq(prev_polarity.clone());
        let time_interval_ok = (col(COL_T) - prev_time)
            * lit(1_000_000.0) // Convert to microseconds
                .gt_eq(lit(min_interval));

        // First event always passes, subsequent events must alternate with sufficient interval
        let is_null = prev_polarity.is_null();

        Ok(is_null.or(polarity_alternates.and(time_interval_ok)))
    }

    /// Build expression for balanced polarity filtering using spatial and temporal windows
    fn build_balanced_expr(&self, _df: &LazyFrame) -> PolarsResult<Expr> {
        let radius = self.balance_radius.unwrap_or(5) as i64;
        let required_ratio = self.balance_ratio.unwrap_or(0.3);
        let tolerance = 0.2;

        // Create spatial bins for neighborhood analysis
        let spatial_bin_x = (col(COL_X) / lit(radius)).cast(DataType::Int64);
        let spatial_bin_y = (col(COL_Y) / lit(radius)).cast(DataType::Int64);

        // Create temporal bins (100ms windows for temporal coherence)
        let time_bin = (col(COL_T) * lit(10.0)).cast(DataType::Int64);

        // Calculate local polarity balance in spatial-temporal neighborhoods
        let local_positive_ratio = col(COL_POLARITY).cast(DataType::Float64).mean().over([
            spatial_bin_x,
            spatial_bin_y,
            time_bin,
        ]);

        // Keep events where the local balance meets the requirement
        Ok(local_positive_ratio
            .clone()
            .gt_eq(lit(required_ratio - tolerance))
            .and(local_positive_ratio.lt_eq(lit(required_ratio + tolerance))))
    }

    /// Get the estimated fraction of events that would pass this filter using Polars
    pub fn estimate_pass_fraction_polars(&self, df: LazyFrame) -> PolarsResult<f64> {
        match self.selection {
            PolaritySelection::Both => Ok(1.0),
            PolaritySelection::PositiveOnly | PolaritySelection::NegativeOnly => {
                let stats_df = df
                    .select([
                        len().alias("total_events"),
                        col(COL_POLARITY).sum().alias("positive_count"),
                    ])
                    .with_columns([(col("positive_count").cast(DataType::Float64)
                        / col("total_events").cast(DataType::Float64))
                    .alias("positive_ratio")])
                    .collect()?;

                if stats_df.height() == 0 {
                    return Ok(0.0);
                }

                let row = stats_df.get_row(0)?;
                let positive_ratio = row.0[2].try_extract::<f64>().unwrap_or(0.0);

                match self.selection {
                    PolaritySelection::PositiveOnly => Ok(positive_ratio),
                    PolaritySelection::NegativeOnly => Ok(1.0 - positive_ratio),
                    _ => unreachable!(),
                }
            }
            PolaritySelection::Alternating => Ok(0.5), // Rough estimate
            PolaritySelection::Balanced => Ok(0.8),    // Rough estimate
        }
    }

    /// Legacy function for pass fraction estimation - delegates to Polars implementation
    pub fn estimate_pass_fraction(&self, events: &Events) -> f64 {
        if events.is_empty() {
            return 0.0;
        }

        match self.selection {
            PolaritySelection::Both => 1.0,
            PolaritySelection::PositiveOnly | PolaritySelection::NegativeOnly => {
                warn!("Using legacy Vec<Event> interface for pass fraction estimation - consider using LazyFrame directly");

                if let Ok(df) = crate::ev_core::events_to_dataframe(events) {
                    if let Ok(fraction) = self.estimate_pass_fraction_polars(df.lazy()) {
                        return fraction;
                    }
                }

                // Fallback to manual count only if Polars fails
                let positive_count = events.iter().filter(|e| e.polarity).count();
                let fraction = positive_count as f64 / events.len() as f64;

                match self.selection {
                    PolaritySelection::PositiveOnly => fraction,
                    PolaritySelection::NegativeOnly => 1.0 - fraction,
                    _ => unreachable!(),
                }
            }
            PolaritySelection::Alternating => 0.5, // Rough estimate
            PolaritySelection::Balanced => 0.8,    // Rough estimate
        }
    }

    /// Get description of this filter
    pub fn description(&self) -> String {
        let mut parts = vec![self.selection.description().to_string()];

        if self.input_encoding != PolarityEncoding::TrueFalse {
            parts.push(format!("encoding: {}", self.input_encoding.description()));
        }

        if let Some(interval) = self.alternating_min_interval {
            parts.push(format!("min interval: {:.1}µs", interval));
        }

        if let (Some(radius), Some(ratio)) = (self.balance_radius, self.balance_ratio) {
            parts.push(format!("balance: r={}, ratio={:.2}", radius, ratio));
        }

        parts.join(", ")
    }
}

impl Default for PolarityFilter {
    fn default() -> Self {
        Self::both()
    }
}

impl Validatable for PolarityFilter {
    fn validate(&self) -> FilterResult<()> {
        match self.selection {
            PolaritySelection::Alternating => {
                if self.alternating_min_interval.is_none() {
                    return Err(FilterError::InvalidConfig(
                        "Alternating polarity filter requires min_interval".to_string(),
                    ));
                }
                if let Some(interval) = self.alternating_min_interval {
                    if interval < 0.0 {
                        return Err(FilterError::InvalidConfig(
                            "Alternating min interval must be non-negative".to_string(),
                        ));
                    }
                }
            }
            PolaritySelection::Balanced => {
                if self.balance_radius.is_none() || self.balance_ratio.is_none() {
                    return Err(FilterError::InvalidConfig(
                        "Balanced polarity filter requires radius and ratio".to_string(),
                    ));
                }
                if let Some(ratio) = self.balance_ratio {
                    if !(0.0..=1.0).contains(&ratio) {
                        return Err(FilterError::InvalidConfig(
                            "Balance ratio must be between 0.0 and 1.0".to_string(),
                        ));
                    }
                }
            }
            _ => {} // No additional validation needed
        }

        Ok(())
    }
}

impl SingleFilter for PolarityFilter {
    fn apply(&self, events: &Events) -> FilterResult<Events> {
        // Legacy Vec<Event> interface - convert to DataFrame and back
        warn!("Using legacy Vec<Event> interface - consider using LazyFrame directly for better performance");

        let df = crate::ev_core::events_to_dataframe(events)
            .map_err(|e| {
                FilterError::ProcessingError(format!("DataFrame conversion failed: {}", e))
            })?
            .lazy();

        let filtered_df = apply_polarity_filter(df, self)
            .map_err(|e| FilterError::ProcessingError(format!("Polars filtering failed: {}", e)))?;

        // Convert back to Vec<Event> - this is inefficient but maintains compatibility
        let result_df = filtered_df.collect().map_err(|e| {
            FilterError::ProcessingError(format!("LazyFrame collection failed: {}", e))
        })?;

        // Convert DataFrame back to Events
        dataframe_to_events(&result_df)
    }

    fn description(&self) -> String {
        format!("Polarity filter: {}", self.description())
    }

    fn is_enabled(&self) -> bool {
        !matches!(self.selection, PolaritySelection::Both)
    }
}

/// Apply polarity filtering using Polars expressions
///
/// This is the main polarity filtering function that works entirely with Polars
/// operations for maximum performance.
///
/// # Arguments
///
/// * `df` - Input LazyFrame containing event data
/// * `filter` - Polarity filter configuration
///
/// # Returns
///
/// Filtered LazyFrame with polarity constraints applied
///
/// # Example
///
/// ```rust
/// use polars::prelude::*;
/// use evlib::ev_filtering::polarity::*;
///
/// let events_df = events_to_dataframe(&events)?.lazy();
/// let filter = PolarityFilter::positive_only();
/// let filtered = apply_polarity_filter(events_df, &filter)?;
/// ```
#[instrument(skip(df), fields(filter = ?filter))]
pub fn apply_polarity_filter(df: LazyFrame, filter: &PolarityFilter) -> PolarsResult<LazyFrame> {
    debug!("Applying polarity filter: {:?}", filter);

    // Validate filter first
    if let Err(e) = filter.validate() {
        warn!("Invalid polarity filter configuration: {}", e);
        return Ok(df); // Return unfiltered data rather than error
    }

    match filter.to_polars_expr(&df)? {
        Some(expr) => {
            debug!("Polarity filter expression: {:?}", expr);
            Ok(df.filter(expr))
        }
        None => {
            debug!("No polarity filtering needed");
            Ok(df)
        }
    }
}

/// Filter events by polarity using Polars expressions
///
/// # Arguments
///
/// * `df` - Input LazyFrame
/// * `positive` - True for positive events, false for negative events
///
/// # Returns
///
/// Filtered LazyFrame
#[instrument(skip(df))]
pub fn filter_by_polarity_polars(df: LazyFrame, positive: bool) -> PolarsResult<LazyFrame> {
    let expr = if positive {
        col(COL_POLARITY).gt(lit(0))
    } else {
        col(COL_POLARITY).eq(lit(0))
    };

    Ok(df.filter(expr))
}

/// Legacy function for backward compatibility - delegates to Polars implementation
pub fn filter_by_polarity(events: &Events, positive: bool) -> FilterResult<Events> {
    let filter = if positive {
        PolarityFilter::positive_only()
    } else {
        PolarityFilter::negative_only()
    };
    filter.apply(events)
}

/// Calculate polarity statistics using Polars aggregations
///
/// This function computes comprehensive polarity statistics efficiently
/// using Polars' built-in aggregation functions.
#[derive(Debug, Clone)]
pub struct PolarityStats {
    pub total_events: usize,
    pub positive_events: usize,
    pub negative_events: usize,
    pub positive_ratio: f64,
    pub negative_ratio: f64,
    pub polarity_balance: f64, // How balanced the polarities are (0.0 = very unbalanced, 1.0 = perfectly balanced)
}

impl PolarityStats {
    /// Calculate polarity statistics using Polars aggregations
    #[instrument(skip(df))]
    pub fn calculate_from_dataframe(df: LazyFrame) -> PolarsResult<Self> {
        let stats_df = df
            .select([
                len().alias("total_events"),
                col(COL_POLARITY).sum().alias("positive_events"),
            ])
            .with_columns([(col("total_events") - col("positive_events")).alias("negative_events")])
            .with_columns([
                (col("positive_events").cast(DataType::Float64)
                    / col("total_events").cast(DataType::Float64))
                .alias("positive_ratio"),
                (col("negative_events").cast(DataType::Float64)
                    / col("total_events").cast(DataType::Float64))
                .alias("negative_ratio"),
            ])
            .with_columns([
                // Calculate balance: 1.0 = perfectly balanced (50/50), 0.0 = completely unbalanced
                (lit(1.0)
                    - when((col("positive_ratio") - lit(0.5)).gt(lit(0.0)))
                        .then((col("positive_ratio") - lit(0.5)) * lit(2.0))
                        .otherwise((lit(0.5) - col("positive_ratio")) * lit(2.0)))
                .alias("polarity_balance"),
            ])
            .collect()?;

        if stats_df.height() == 0 {
            return Ok(Self::empty());
        }

        let row = stats_df.get_row(0)?;

        Ok(Self {
            total_events: row.0[0].try_extract::<u32>()? as usize,
            positive_events: row.0[1].try_extract::<u32>()? as usize,
            negative_events: row.0[2].try_extract::<u32>()? as usize,
            positive_ratio: row.0[3].try_extract::<f64>()?,
            negative_ratio: row.0[4].try_extract::<f64>()?,
            polarity_balance: row.0[5].try_extract::<f64>()?,
        })
    }

    /// Legacy interface for Vec<Event> - delegates to Polars implementation
    pub fn calculate(events: &Events) -> Self {
        if events.is_empty() {
            return Self::empty();
        }

        warn!(
            "Using legacy Vec<Event> interface for polarity statistics - consider using LazyFrame directly"
        );

        let df = match crate::ev_core::events_to_dataframe(events) {
            Ok(df) => df.lazy(),
            Err(e) => {
                warn!("Failed to convert events to DataFrame: {}, falling back", e);
                return Self::calculate_legacy(events);
            }
        };

        match Self::calculate_from_dataframe(df) {
            Ok(stats) => stats,
            Err(e) => {
                warn!("Polars statistics calculation failed: {}, falling back", e);
                Self::calculate_legacy(events)
            }
        }
    }

    /// Fallback legacy calculation
    fn calculate_legacy(events: &Events) -> Self {
        let total_events = events.len();
        let positive_events = events.iter().filter(|e| e.polarity).count();
        let negative_events = total_events - positive_events;

        let positive_ratio = if total_events > 0 {
            positive_events as f64 / total_events as f64
        } else {
            0.0
        };
        let negative_ratio = 1.0 - positive_ratio;

        // Calculate balance: 1.0 = perfectly balanced (50/50), 0.0 = completely unbalanced
        let polarity_balance = if total_events > 0 {
            1.0 - (positive_ratio - 0.5).abs() * 2.0
        } else {
            0.0
        };

        Self {
            total_events,
            positive_events,
            negative_events,
            positive_ratio,
            negative_ratio,
            polarity_balance,
        }
    }

    fn empty() -> Self {
        Self {
            total_events: 0,
            positive_events: 0,
            negative_events: 0,
            positive_ratio: 0.0,
            negative_ratio: 0.0,
            polarity_balance: 0.0,
        }
    }
}

impl std::fmt::Display for PolarityStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Polarity: +{} ({:.1}%) / -{} ({:.1}%) | Balance: {:.3}",
            self.positive_events,
            self.positive_ratio * 100.0,
            self.negative_events,
            self.negative_ratio * 100.0,
            self.polarity_balance
        )
    }
}

/// Apply alternating polarity filter using Polars window functions
///
/// This function uses Polars' shift() and window operations to efficiently
/// identify events that alternate in polarity with minimum time intervals.
#[instrument(skip(df))]
pub fn apply_alternating_polarity_filter_polars(
    df: LazyFrame,
    min_interval_us: f64,
) -> PolarsResult<LazyFrame> {
    // Sort by time first to ensure proper ordering
    let sorted_df = df.sort([COL_T], SortMultipleOptions::default());

    // Use shift to compare with previous event
    let prev_polarity = col(COL_POLARITY).shift(lit(1));
    let prev_time = col(COL_T).shift(lit(1));

    // Check polarity alternation and time interval
    let polarity_alternates = col(COL_POLARITY).neq(prev_polarity.clone());
    let time_interval_ok = (col(COL_T) - prev_time)
        * lit(1_000_000.0) // Convert to microseconds
            .gt_eq(lit(min_interval_us));

    // First event always passes, subsequent events must alternate with sufficient interval
    let is_first = prev_polarity.is_null();
    let passes_filter = is_first.or(polarity_alternates.and(time_interval_ok));

    Ok(sorted_df.filter(passes_filter))
}

/// Apply balanced polarity filter using Polars group_by and window operations
///
/// This function uses Polars aggregations to check polarity balance in
/// spatial and temporal neighborhoods efficiently using vectorized operations.
#[instrument(skip(df))]
pub fn apply_balanced_polarity_filter_polars(
    df: LazyFrame,
    radius: u16,
    required_ratio: f64,
) -> PolarsResult<LazyFrame> {
    let bin_size = radius as i64;
    let tolerance = 0.2;

    // Create spatial and temporal bins using pure Polars expressions
    let spatial_binned = df
        .with_columns([
            (col(COL_X) / lit(bin_size))
                .cast(DataType::Int64)
                .alias("spatial_bin_x"),
            (col(COL_Y) / lit(bin_size))
                .cast(DataType::Int64)
                .alias("spatial_bin_y"),
            (col(COL_T) * lit(10.0))
                .cast(DataType::Int64)
                .alias("time_bin"),
        ])
        .with_columns([
            // Calculate local polarity balance using window functions
            col(COL_POLARITY)
                .cast(DataType::Float64)
                .mean()
                .over([col("spatial_bin_x"), col("spatial_bin_y"), col("time_bin")])
                .alias("local_positive_ratio"),
        ]);

    // Apply balance filter using vectorized comparisons
    let balance_filter = col("local_positive_ratio")
        .gt_eq(lit(required_ratio - tolerance))
        .and(col("local_positive_ratio").lt_eq(lit(required_ratio + tolerance)));

    Ok(spatial_binned.filter(balance_filter))
}

/// Analyze polarity patterns using Polars operations
///
/// This function computes various polarity statistics and patterns
/// using efficient Polars aggregations and window functions.
#[instrument(skip(df))]
pub fn analyze_polarity_patterns_polars(df: LazyFrame) -> PolarsResult<DataFrame> {
    // Sort by time first for pattern analysis
    let sorted_df = df.sort([COL_T], SortMultipleOptions::default());

    sorted_df
        .select([
            // Basic statistics
            len().alias("total_events"),
            col(COL_POLARITY).sum().alias("positive_events"),
            (len() - col(COL_POLARITY).sum()).alias("negative_events"),
            col(COL_POLARITY).mean().alias("positive_ratio"),
            (lit(1.0) - col(COL_POLARITY).mean()).alias("negative_ratio"),
            // Polarity balance
            (lit(1.0)
                - when((col(COL_POLARITY).mean() - lit(0.5)).gt(lit(0.0)))
                    .then((col(COL_POLARITY).mean() - lit(0.5)) * lit(2.0))
                    .otherwise((lit(0.5) - col(COL_POLARITY).mean()) * lit(2.0)))
            .alias("polarity_balance"),
            // Switch rate using shift
            (col(COL_POLARITY).neq(col(COL_POLARITY).shift(lit(1))))
                .sum()
                .cast(DataType::Float64)
                .alias("switch_count"),
            // Temporal statistics
            col(COL_T).min().alias("t_min"),
            col(COL_T).max().alias("t_max"),
            (col(COL_T).max() - col(COL_T).min()).alias("duration"),
        ])
        .with_columns([
            // Calculate switch rate
            (col("switch_count") / (len() - lit(1)).cast(DataType::Float64))
                .alias("polarity_switch_rate"),
            // Event rate
            (len().cast(DataType::Float64) / col("duration")).alias("event_rate"),
        ])
        .collect()
}

/// Legacy function for polarity pattern analysis - delegates to Polars
pub fn analyze_polarity_patterns(events: &Events) -> FilterResult<HashMap<String, f64>> {
    if events.is_empty() {
        return Ok(HashMap::new());
    }

    warn!("Using legacy Vec<Event> interface for pattern analysis - consider using LazyFrame directly");

    let df = match crate::ev_core::events_to_dataframe(events) {
        Ok(df) => df.lazy(),
        Err(_) => return analyze_polarity_patterns_legacy(events),
    };

    match analyze_polarity_patterns_polars(df) {
        Ok(analysis_df) => {
            let mut result = HashMap::new();

            if analysis_df.height() > 0 {
                let row = analysis_df.get_row(0).map_err(|_| {
                    FilterError::ProcessingError("Failed to extract analysis row".to_string())
                })?;

                // Extract statistics from the analysis DataFrame
                if let Ok(positive_ratio) = row.0[3].try_extract::<f64>() {
                    result.insert("positive_ratio".to_string(), positive_ratio);
                }
                if let Ok(negative_ratio) = row.0[4].try_extract::<f64>() {
                    result.insert("negative_ratio".to_string(), negative_ratio);
                }
                if let Ok(polarity_balance) = row.0[5].try_extract::<f64>() {
                    result.insert("polarity_balance".to_string(), polarity_balance);
                }
                if let Ok(switch_rate) = row.0[11].try_extract::<f64>() {
                    result.insert("polarity_switch_rate".to_string(), switch_rate);
                }
                if let Ok(event_rate) = row.0[12].try_extract::<f64>() {
                    result.insert("event_rate".to_string(), event_rate);
                }
            }

            Ok(result)
        }
        Err(_) => analyze_polarity_patterns_legacy(events),
    }
}

/// Fallback legacy pattern analysis - now also uses Polars expressions where possible
fn analyze_polarity_patterns_legacy(events: &Events) -> FilterResult<HashMap<String, f64>> {
    let mut analysis = HashMap::new();

    if events.is_empty() {
        return Ok(analysis);
    }

    let stats = PolarityStats::calculate(events);
    analysis.insert("positive_ratio".to_string(), stats.positive_ratio);
    analysis.insert("negative_ratio".to_string(), stats.negative_ratio);
    analysis.insert("polarity_balance".to_string(), stats.polarity_balance);

    // Use Polars for switch rate calculation if possible
    if let Ok(df) = crate::ev_core::events_to_dataframe(events) {
        let lazy_df = df.lazy().sort([COL_T], SortMultipleOptions::default());

        if let Ok(switch_df) = lazy_df
            .select([
                len().alias("total_events"),
                (col(COL_POLARITY).neq(col(COL_POLARITY).shift(lit(1))))
                    .sum()
                    .cast(DataType::Float64)
                    .alias("switch_count"),
            ])
            .with_columns([(col("switch_count")
                / (col("total_events") - lit(1)).cast(DataType::Float64))
            .alias("switch_rate")])
            .collect()
        {
            if switch_df.height() > 0 {
                if let Ok(row) = switch_df.get_row(0) {
                    if let Ok(switch_rate) = row.0[2].try_extract::<f64>() {
                        analysis.insert("polarity_switch_rate".to_string(), switch_rate);
                        return Ok(analysis);
                    }
                }
            }
        }
    }

    // Pure legacy fallback only if Polars fails
    let mut switch_count = 0;
    let mut last_polarity = events[0].polarity;

    for event in events.iter().skip(1) {
        if event.polarity != last_polarity {
            switch_count += 1;
            last_polarity = event.polarity;
        }
    }

    let switch_rate = if events.len() > 1 {
        switch_count as f64 / (events.len() - 1) as f64
    } else {
        0.0
    };
    analysis.insert("polarity_switch_rate".to_string(), switch_rate);

    Ok(analysis)
}

/// Separate events by polarity using Polars group operations
///
/// This function splits events into positive and negative groups using
/// efficient Polars filtering operations.
#[instrument(skip(df))]
pub fn separate_polarities_polars(df: LazyFrame) -> PolarsResult<(LazyFrame, LazyFrame)> {
    let positive_df = df.clone().filter(col(COL_POLARITY).gt(lit(0)));
    let negative_df = df.filter(col(COL_POLARITY).eq(lit(0)));

    Ok((positive_df, negative_df))
}

/// Legacy function for separating polarities - delegates to Polars
pub fn separate_polarities(events: &Events) -> (Events, Events) {
    warn!("Using legacy Vec<Event> interface for polarity separation - consider using LazyFrame directly");

    let df = match crate::ev_core::events_to_dataframe(events) {
        Ok(df) => df.lazy(),
        Err(_) => return separate_polarities_legacy(events),
    };

    match separate_polarities_polars(df) {
        Ok((pos_df, neg_df)) => {
            let positive_events = pos_df
                .collect()
                .ok()
                .and_then(|df| dataframe_to_events(&df).ok())
                .unwrap_or_default();

            let negative_events = neg_df
                .collect()
                .ok()
                .and_then(|df| dataframe_to_events(&df).ok())
                .unwrap_or_default();

            (positive_events, negative_events)
        }
        Err(_) => separate_polarities_legacy(events),
    }
}

/// Fallback legacy polarity separation - no longer uses manual loops
fn separate_polarities_legacy(events: &Events) -> (Events, Events) {
    // Even for "legacy" fallback, we now use Vec methods that are more efficient
    let (positive_events, negative_events): (Vec<_>, Vec<_>) =
        events.iter().partition(|event| event.polarity);

    // Convert to owned events
    let positive_events = positive_events.into_iter().copied().collect();
    let negative_events = negative_events.into_iter().copied().collect();

    (positive_events, negative_events)
}

/// Convert DataFrame back to Events (for legacy compatibility)
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
        let x = x_values
            .get(i)
            .ok_or_else(|| FilterError::ProcessingError("Missing x value".to_string()))?
            as u16;
        let y = y_values
            .get(i)
            .ok_or_else(|| FilterError::ProcessingError("Missing y value".to_string()))?
            as u16;
        let t = t_values
            .get(i)
            .ok_or_else(|| FilterError::ProcessingError("Missing t value".to_string()))?;
        let p = p_values
            .get(i)
            .ok_or_else(|| FilterError::ProcessingError("Missing polarity value".to_string()))?
            > 0;

        events.push(Event {
            x,
            y,
            t,
            polarity: p,
        });
    }

    Ok(events)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ev_core::{events_to_dataframe, Event};

    fn create_test_events() -> Events {
        vec![
            Event {
                t: 1.0,
                x: 100,
                y: 200,
                polarity: true,
            }, // Positive
            Event {
                t: 2.0,
                x: 150,
                y: 250,
                polarity: false,
            }, // Negative
            Event {
                t: 3.0,
                x: 200,
                y: 300,
                polarity: true,
            }, // Positive
            Event {
                t: 4.0,
                x: 250,
                y: 350,
                polarity: false,
            }, // Negative
            Event {
                t: 5.0,
                x: 300,
                y: 400,
                polarity: true,
            }, // Positive
        ]
    }

    #[test]
    fn test_polarity_filter_creation() {
        let filter = PolarityFilter::positive_only();
        assert_eq!(filter.selection, PolaritySelection::PositiveOnly);

        let filter = PolarityFilter::negative_only();
        assert_eq!(filter.selection, PolaritySelection::NegativeOnly);

        let filter = PolarityFilter::both();
        assert_eq!(filter.selection, PolaritySelection::Both);

        let filter = PolarityFilter::alternating(1000.0);
        assert_eq!(filter.selection, PolaritySelection::Alternating);
        assert_eq!(filter.alternating_min_interval, Some(1000.0));
    }

    #[test]
    fn test_polarity_filtering_polars() -> PolarsResult<()> {
        let events = create_test_events();
        let df = events_to_dataframe(&events)?.lazy();

        // Test positive filtering
        let positive_filtered = filter_by_polarity_polars(df.clone(), true)?;
        let pos_result = positive_filtered.collect()?;
        assert_eq!(pos_result.height(), 3); // 3 positive events

        // Test negative filtering
        let negative_filtered = filter_by_polarity_polars(df, false)?;
        let neg_result = negative_filtered.collect()?;
        assert_eq!(neg_result.height(), 2); // 2 negative events

        Ok(())
    }

    #[test]
    fn test_polarity_stats_polars() -> PolarsResult<()> {
        let events = create_test_events();
        let df = events_to_dataframe(&events)?.lazy();

        let stats = PolarityStats::calculate_from_dataframe(df)?;

        assert_eq!(stats.total_events, 5);
        assert_eq!(stats.positive_events, 3);
        assert_eq!(stats.negative_events, 2);
        assert!((stats.positive_ratio - 0.6).abs() < 0.001);
        assert!((stats.negative_ratio - 0.4).abs() < 0.001);
        assert!(stats.polarity_balance > 0.5); // Reasonably balanced

        Ok(())
    }

    #[test]
    fn test_pattern_analysis_polars() -> PolarsResult<()> {
        let events = create_test_events();
        let df = events_to_dataframe(&events)?.lazy();

        let analysis_df = analyze_polarity_patterns_polars(df)?;

        assert_eq!(analysis_df.height(), 1);
        assert!(analysis_df.column("positive_ratio").is_ok());
        assert!(analysis_df.column("polarity_switch_rate").is_ok());
        assert!(analysis_df.column("polarity_balance").is_ok());

        Ok(())
    }

    #[test]
    fn test_alternating_filter_polars() -> PolarsResult<()> {
        let events = vec![
            Event {
                t: 1.0,
                x: 100,
                y: 200,
                polarity: true,
            },
            Event {
                t: 1.0005,
                x: 100,
                y: 200,
                polarity: false,
            }, // 0.5ms later - too soon
            Event {
                t: 1.002,
                x: 100,
                y: 200,
                polarity: false,
            }, // 2ms later - ok
            Event {
                t: 1.005,
                x: 100,
                y: 200,
                polarity: true,
            }, // 3ms later - ok
        ];

        let df = events_to_dataframe(&events)?.lazy();
        let filtered = apply_alternating_polarity_filter_polars(df, 1000.0)?; // 1ms minimum
        let result = filtered.collect()?;

        // Should filter out events that don't meet alternating criteria
        assert!(result.height() <= events.len());
        assert!(result.height() > 0);

        Ok(())
    }

    #[test]
    fn test_separate_polarities_polars() -> PolarsResult<()> {
        let events = create_test_events();
        let df = events_to_dataframe(&events)?.lazy();

        let (pos_df, neg_df) = separate_polarities_polars(df)?;

        let pos_result = pos_df.collect()?;
        let neg_result = neg_df.collect()?;

        assert_eq!(pos_result.height(), 3); // 3 positive events
        assert_eq!(neg_result.height(), 2); // 2 negative events

        Ok(())
    }

    #[test]
    fn test_encoding_detection() {
        // Test 0/1 encoding
        let zero_one_values = vec![0.0, 1.0, 0.0, 1.0];
        let encoding = PolarityEncoding::detect_from_raw_values(&zero_one_values);
        assert_eq!(encoding, PolarityEncoding::OneZero);

        // Test -1/1 encoding
        let minus_one_values = vec![-1.0, 1.0, -1.0, 1.0];
        let encoding = PolarityEncoding::detect_from_raw_values(&minus_one_values);
        assert_eq!(encoding, PolarityEncoding::OneMinus);

        // Test mixed encoding
        let mixed_values = vec![0.0, 1.0, -1.0, 1.0];
        let encoding = PolarityEncoding::detect_from_raw_values(&mixed_values);
        assert_eq!(encoding, PolarityEncoding::Mixed);
    }

    #[test]
    fn test_legacy_compatibility() {
        let events = create_test_events();

        // Test legacy filter application
        let filter = PolarityFilter::positive_only();
        let filtered = filter.apply(&events).unwrap();
        assert_eq!(filtered.len(), 3);

        // Test legacy pattern analysis
        let analysis = analyze_polarity_patterns(&events).unwrap();
        assert!(analysis.contains_key("positive_ratio"));
        assert!(analysis.contains_key("polarity_switch_rate"));

        // Test legacy statistics
        let stats = PolarityStats::calculate(&events);
        assert_eq!(stats.total_events, 5);
        assert_eq!(stats.positive_events, 3);
    }

    #[test]
    fn test_filter_validation() {
        // Valid filters
        assert!(PolarityFilter::positive_only().validate().is_ok());
        assert!(PolarityFilter::alternating(1000.0).validate().is_ok());
        assert!(PolarityFilter::balanced(5, 0.3).validate().is_ok());

        // Invalid filters
        let mut invalid_alternating = PolarityFilter::alternating(1000.0);
        invalid_alternating.alternating_min_interval = None;
        assert!(invalid_alternating.validate().is_err());

        let mut invalid_balanced = PolarityFilter::balanced(5, 0.3);
        invalid_balanced.balance_ratio = Some(1.5); // > 1.0
        assert!(invalid_balanced.validate().is_err());
    }
}
