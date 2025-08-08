//! Time reversal augmentation for event data
//!
//! This module implements temporal reversal augmentation, which reverses
//! the temporal order of events and flips polarities to maintain causality.

use crate::ev_augmentation::{AugmentationError, AugmentationResult, Validatable};
// Removed: use crate::{Event, Events}; - legacy types no longer exist
use rand::{Rng, SeedableRng};
#[cfg(feature = "tracing")]
use tracing::{debug, instrument};

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
use crate::ev_augmentation::{COL_POLARITY, COL_T};

#[cfg(feature = "polars")]
use polars::prelude::*;

/// Time reversal augmentation configuration
///
/// Reverses the temporal order of events with a specified probability.
/// When applied, events are sorted in reverse chronological order and
/// polarities are flipped to maintain physical consistency.
///
/// # Example
///
/// ```rust
/// use evlib::ev_augmentation::TimeReversalAugmentation;
///
/// // Apply time reversal with 50% probability
/// let reversal = TimeReversalAugmentation::new(0.5);
/// ```
#[derive(Debug, Clone)]
pub struct TimeReversalAugmentation {
    /// Probability of applying time reversal (0-1)
    pub probability: f64,
    /// Whether to flip polarities when reversing time
    pub flip_polarity: bool,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl TimeReversalAugmentation {
    /// Create new time reversal augmentation
    ///
    /// # Arguments
    ///
    /// * `probability` - Probability of applying time reversal (0-1)
    pub fn new(probability: f64) -> Self {
        Self {
            probability,
            flip_polarity: true, // Default to flipping polarity for physical consistency
            seed: None,
        }
    }

    /// Set whether to flip polarities during time reversal
    pub fn with_polarity_flip(mut self, flip: bool) -> Self {
        self.flip_polarity = flip;
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
            "time_reversal(prob={:.2}, flip_polarity={})",
            self.probability, self.flip_polarity
        )
    }
}

impl Validatable for TimeReversalAugmentation {
    fn validate(&self) -> AugmentationResult<()> {
        if !(0.0..=1.0).contains(&self.probability) {
            return Err(AugmentationError::InvalidProbability(self.probability));
        }
        Ok(())
    }
}

/* Commented out - legacy SingleAugmentation trait no longer exists
impl SingleAugmentation for TimeReversalAugmentation {
    #[cfg_attr(feature = "tracing", instrument(skip(events), level = "debug"))]
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

        // Determine whether to apply time reversal
        let apply_reversal = rng.gen::<f64>() < self.probability;

        if !apply_reversal {
            debug!("Time reversal not applied (probability check failed)");
            return Ok(events.clone());
        }

        debug!(
            "Applying time reversal with polarity flip: {}",
            self.flip_polarity
        );

        // Find min and max timestamps for reversal
        let min_time = events.iter().map(|e| e.t).fold(f64::INFINITY, f64::min);
        let max_time = events.iter().map(|e| e.t).fold(f64::NEG_INFINITY, f64::max);

        let time_span = max_time - min_time;

        // Create reversed events
        let mut reversed_events: Vec<Event> = events
            .iter()
            .map(|event| {
                let reversed_time = max_time - (event.t - min_time);
                let reversed_polarity = if self.flip_polarity {
                    !event.polarity
                } else {
                    event.polarity
                };

                Event {
                    t: reversed_time,
                    x: event.x,
                    y: event.y,
                    polarity: reversed_polarity,
                }
            })
            .collect();

        // Sort by timestamp to maintain temporal order
        reversed_events.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap());

        info!(
            "Applied time reversal to {} events (time_span={:.3}s, polarity_flip={})",
            reversed_events.len(),
            time_span,
            self.flip_polarity
        );

        Ok(reversed_events)
    }

    fn description(&self) -> String {
        self.description()
    }
}
*/

/* Commented out - legacy Events type no longer exists
/// Apply time reversal to events
///
/// # Arguments
///
/// * `events` - Input events
/// * `config` - Time reversal configuration
///
/// # Returns
///
/// * `AugmentationResult<Events>` - Time-reversed events
#[cfg_attr(feature = "tracing", instrument(skip(events), level = "debug"))]
pub fn time_reversal(
    events: &Events,
    config: &TimeReversalAugmentation,
) -> AugmentationResult<Events> {
    config.apply(events)
}
*/

/// Apply time reversal using Polars operations
#[cfg(feature = "polars")]
#[cfg_attr(feature = "tracing", instrument(skip(df), level = "debug"))]
pub fn apply_time_reversal_polars(
    df: LazyFrame,
    config: &TimeReversalAugmentation,
) -> AugmentationResult<LazyFrame> {
    config.validate()?;

    let mut rng = if let Some(seed) = config.seed {
        rand::rngs::StdRng::seed_from_u64(seed)
    } else {
        rand::rngs::StdRng::from_entropy()
    };

    // Determine whether to apply time reversal
    let apply_reversal = rng.gen::<f64>() < config.probability;

    if !apply_reversal {
        debug!("Time reversal not applied (probability check failed) - Polars");
        return Ok(df);
    }

    debug!(
        "Applying time reversal (Polars) with polarity flip: {}",
        config.flip_polarity
    );

    // Apply time reversal using Polars operations
    let mut result = df.with_columns([
        // Calculate reversed timestamps: max_time - (t - min_time)
        (col(COL_T).max().over([lit(1)]) - (col(COL_T) - col(COL_T).min().over([lit(1)])))
            .alias(COL_T),
    ]);

    // Flip polarities if requested
    if config.flip_polarity {
        result = result.with_columns([
            // Flip polarity: 1 - polarity (assuming 0/1 encoding)
            (lit(1) - col(COL_POLARITY)).alias(COL_POLARITY),
        ]);
    }

    // Sort by timestamp to maintain temporal order
    result = result.sort([COL_T], Default::default());

    Ok(result)
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
            Event {
                t: 4.0,
                x: 250,
                y: 350,
                polarity: false,
            },
        ]
    }

    #[test]
    fn test_time_reversal_creation() {
        let reversal = TimeReversalAugmentation::new(0.5);
        assert_eq!(reversal.probability, 0.5);
        assert!(reversal.flip_polarity);
    }

    #[test]
    fn test_validation() {
        // Valid configuration
        let valid_config = TimeReversalAugmentation::new(0.5);
        assert!(valid_config.validate().is_ok());

        // Invalid probability
        let invalid_prob = TimeReversalAugmentation::new(1.5);
        assert!(invalid_prob.validate().is_err());

        let invalid_negative = TimeReversalAugmentation::new(-0.1);
        assert!(invalid_negative.validate().is_err());
    }

    #[test]
    fn test_no_reversal() {
        let events = create_test_events();
        let reversal = TimeReversalAugmentation::new(0.0); // Never apply
        let result = reversal.apply(&events).unwrap();

        // Events should be unchanged
        assert_eq!(result.len(), events.len());
        for (original, result) in events.iter().zip(result.iter()) {
            assert_eq!(original.t, result.t);
            assert_eq!(original.x, result.x);
            assert_eq!(original.y, result.y);
            assert_eq!(original.polarity, result.polarity);
        }
    }

    #[test]
    fn test_deterministic_reversal() {
        let events = create_test_events();
        let reversal1 = TimeReversalAugmentation::new(1.0).with_seed(42); // Always apply
        let reversal2 = TimeReversalAugmentation::new(1.0).with_seed(42); // Always apply

        let result1 = reversal1.apply(&events).unwrap();
        let result2 = reversal2.apply(&events).unwrap();

        // Results should be identical with same seed
        assert_eq!(result1.len(), result2.len());
        for (r1, r2) in result1.iter().zip(result2.iter()) {
            assert_eq!(r1.t, r2.t);
            assert_eq!(r1.x, r2.x);
            assert_eq!(r1.y, r2.y);
            assert_eq!(r1.polarity, r2.polarity);
        }
    }

    #[test]
    fn test_time_reversal_properties() {
        let events = create_test_events();
        let reversal = TimeReversalAugmentation::new(1.0).with_seed(42); // Always apply
        let result = reversal.apply(&events).unwrap();

        assert_eq!(result.len(), events.len());

        // Check that timestamps are properly reversed
        // First event should become last (chronologically)
        assert!(result[0].t < result[1].t); // Still sorted
        assert!(result[1].t < result[2].t);
        assert!(result[2].t < result[3].t);

        // Check that polarities are flipped
        for (original, reversed) in events.iter().zip(result.iter()) {
            assert_eq!(original.polarity, !reversed.polarity);
            assert_eq!(original.x, reversed.x); // Spatial coordinates unchanged
            assert_eq!(original.y, reversed.y);
        }
    }

    #[test]
    fn test_time_reversal_without_polarity_flip() {
        let events = create_test_events();
        let reversal = TimeReversalAugmentation::new(1.0)
            .with_polarity_flip(false)
            .with_seed(42);
        let result = reversal.apply(&events).unwrap();

        assert_eq!(result.len(), events.len());

        // Check that polarities are NOT flipped
        for (original, reversed) in events.iter().zip(result.iter()) {
            assert_eq!(original.polarity, reversed.polarity);
        }
    }
}
