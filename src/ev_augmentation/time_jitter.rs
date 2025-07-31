//! Time jitter augmentation for event data
//!
//! This module implements temporal jittering by adding samples from a Gaussian
//! distribution to event timestamps, simulating timing uncertainty and sensor noise.

use crate::ev_augmentation::{
    AugmentationError, AugmentationResult, SingleAugmentation, Validatable,
};
use crate::ev_core::{Event, Events};
use rand::SeedableRng;

#[cfg(feature = "polars")]
use crate::ev_augmentation::COL_T;
use rand_distr::{Distribution, Normal};
use tracing::{debug, info, instrument};

#[cfg(feature = "polars")]
use polars::prelude::*;

/// Time jitter augmentation configuration
///
/// Changes timestamp for each event by drawing samples from a Gaussian distribution
/// and adding them to each timestamp. This simulates timing uncertainty in event cameras.
///
/// # Example
///
/// ```rust
/// use evlib::ev_augmentation::TimeJitterAugmentation;
///
/// // Create jitter with 1ms standard deviation
/// let jitter = TimeJitterAugmentation::new(1000.0);
///
/// // Create jitter that clips negative timestamps
/// let jitter = TimeJitterAugmentation::new(500.0)
///     .with_clipping(true);
/// ```
#[derive(Debug, Clone)]
pub struct TimeJitterAugmentation {
    /// Standard deviation in microseconds
    pub std_us: f64,
    /// Whether to clip events with negative timestamps
    pub clip_negative: bool,
    /// Whether to sort events after jittering
    pub sort_timestamps: bool,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl TimeJitterAugmentation {
    /// Create a new time jitter augmentation
    ///
    /// # Arguments
    ///
    /// * `std_us` - Standard deviation of the time jitter in microseconds
    pub fn new(std_us: f64) -> Self {
        Self {
            std_us,
            clip_negative: false,
            sort_timestamps: false,
            seed: None,
        }
    }

    /// Enable clipping of events with negative timestamps
    pub fn with_clipping(mut self, clip: bool) -> Self {
        self.clip_negative = clip;
        self
    }

    /// Enable sorting of events after jittering
    pub fn with_sorting(mut self, sort: bool) -> Self {
        self.sort_timestamps = sort;
        self
    }

    /// Set random seed for reproducibility
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Get description of this augmentation
    pub fn description(&self) -> String {
        format!("std={:.1}µs", self.std_us)
    }
}

impl Validatable for TimeJitterAugmentation {
    fn validate(&self) -> AugmentationResult<()> {
        if self.std_us < 0.0 {
            return Err(AugmentationError::InvalidConfig(
                "Time jitter standard deviation must be non-negative".to_string(),
            ));
        }
        Ok(())
    }
}

impl SingleAugmentation for TimeJitterAugmentation {
    fn apply(&self, events: &Events) -> AugmentationResult<Events> {
        time_jitter(events, self)
    }

    fn description(&self) -> String {
        format!("Time jitter: {}", self.description())
    }
}

/// Apply time jitter to events
///
/// This function adds Gaussian noise to event timestamps, simulating timing uncertainty.
/// Events can optionally be clipped if they get negative timestamps, and sorted after jittering.
///
/// # Arguments
///
/// * `events` - Input events to augment
/// * `config` - Time jitter configuration
///
/// # Returns
///
/// * `AugmentationResult<Events>` - Jittered events
#[instrument(skip(events), fields(n_events = events.len()))]
pub fn time_jitter(events: &Events, config: &TimeJitterAugmentation) -> AugmentationResult<Events> {
    let start_time = std::time::Instant::now();

    if events.is_empty() {
        debug!("No events to jitter");
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

    // Create normal distribution for jitter (in seconds)
    let std_seconds = config.std_us / 1_000_000.0;
    let distribution = Normal::new(0.0, std_seconds)
        .map_err(|e| AugmentationError::InvalidConfig(format!("Invalid distribution: {}", e)))?;

    // Apply jitter to each event
    let mut jittered_events = Vec::with_capacity(events.len());
    let mut clipped_count = 0;

    for event in events {
        let jitter = distribution.sample(&mut rng);
        let new_t = event.t + jitter;

        // Check for negative timestamps if clipping is enabled
        if config.clip_negative && new_t < 0.0 {
            clipped_count += 1;
            continue;
        }

        // Create jittered event
        let jittered_event = Event {
            t: new_t.max(0.0), // Ensure non-negative even without clipping
            x: event.x,
            y: event.y,
            polarity: event.polarity,
        };

        jittered_events.push(jittered_event);
    }

    // Sort if requested
    if config.sort_timestamps {
        jittered_events.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));
    }

    let processing_time = start_time.elapsed().as_secs_f64();
    let output_count = jittered_events.len();

    info!(
        "Time jitter applied: {} -> {} events ({} clipped) in {:.3}s",
        events.len(),
        output_count,
        clipped_count,
        processing_time
    );

    Ok(jittered_events)
}

/// Apply time jitter using Polars operations (Polars-first implementation)
///
/// This implementation uses vectorized operations for better performance on large datasets.
///
/// # Arguments
///
/// * `df` - Input LazyFrame containing event data
/// * `config` - Time jitter configuration
///
/// # Returns
///
/// * `PolarsResult<LazyFrame>` - Jittered events as LazyFrame
#[cfg(feature = "polars")]
#[instrument(skip(df), fields(config = ?config))]
pub fn apply_time_jitter_polars(
    df: LazyFrame,
    config: &TimeJitterAugmentation,
) -> PolarsResult<LazyFrame> {
    debug!("Applying time jitter with Polars: {:?}", config);

    // Convert standard deviation to seconds
    let _std_seconds = config.std_us / 1_000_000.0;

    // Generate random jitter values
    let jittered_df = df.with_columns([
        // Generate random normal values for jitter
        // Note: Polars doesn't have built-in normal distribution, so we use uniform and Box-Muller transform
        (col(COL_T).count().over([lit(1)]).alias("n_events")),
    ]);

    // For now, collect and use the Vec implementation for true normal distribution
    // A fully vectorized Polars implementation would require custom expressions
    let collected_df = jittered_df.collect()?;

    // Convert to Events
    let events = crate::ev_augmentation::dataframe_to_events(&collected_df)
        .map_err(|e| PolarsError::ComputeError(format!("Conversion error: {}", e).into()))?;

    // Apply jitter
    let jittered = time_jitter(&events, config)
        .map_err(|e| PolarsError::ComputeError(format!("Jitter error: {}", e).into()))?;

    // Convert back to DataFrame
    let jittered_df = crate::ev_core::events_to_dataframe(&jittered)?;

    Ok(jittered_df.lazy())
}

/// Convenience function for simple time jitter
///
/// # Arguments
///
/// * `events` - Input events
/// * `std_us` - Standard deviation in microseconds
pub fn apply_time_jitter_simple(events: &Events, std_us: f64) -> AugmentationResult<Events> {
    let config = TimeJitterAugmentation::new(std_us);
    time_jitter(events, &config)
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
        ]
    }

    #[test]
    fn test_time_jitter_creation() {
        let jitter = TimeJitterAugmentation::new(1000.0);
        assert_eq!(jitter.std_us, 1000.0);
        assert!(!jitter.clip_negative);
        assert!(!jitter.sort_timestamps);
    }

    #[test]
    fn test_time_jitter_validation() {
        // Valid configuration
        let valid = TimeJitterAugmentation::new(100.0);
        assert!(valid.validate().is_ok());

        // Invalid: negative standard deviation
        let invalid = TimeJitterAugmentation::new(-100.0);
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_time_jitter_application() {
        let events = create_test_events();
        let config = TimeJitterAugmentation::new(100.0).with_seed(42);

        let jittered = time_jitter(&events, &config).unwrap();

        // Should have same number of events
        assert_eq!(jittered.len(), events.len());

        // Coordinates and polarities should be unchanged
        for (orig, jit) in events.iter().zip(jittered.iter()) {
            assert_eq!(orig.x, jit.x);
            assert_eq!(orig.y, jit.y);
            assert_eq!(orig.polarity, jit.polarity);
        }

        // Timestamps should be different (with very high probability)
        let times_changed = events
            .iter()
            .zip(jittered.iter())
            .any(|(orig, jit)| (orig.t - jit.t).abs() > 1e-10);
        assert!(times_changed);
    }

    #[test]
    fn test_time_jitter_with_clipping() {
        let events = vec![
            Event {
                t: 0.001, // Very close to zero
                x: 100,
                y: 200,
                polarity: true,
            },
            Event {
                t: 1.0,
                x: 150,
                y: 250,
                polarity: false,
            },
        ];

        // Large jitter with clipping
        let config = TimeJitterAugmentation::new(10000.0) // 10ms std
            .with_clipping(true)
            .with_seed(42);

        let jittered = time_jitter(&events, &config).unwrap();

        // Some events may be clipped
        assert!(jittered.len() <= events.len());

        // All remaining events should have non-negative timestamps
        for event in &jittered {
            assert!(event.t >= 0.0);
        }
    }

    #[test]
    fn test_time_jitter_with_sorting() {
        let mut events = create_test_events();
        // Add an event that will likely be out of order after jittering
        events.push(Event {
            t: 2.5,
            x: 300,
            y: 400,
            polarity: true,
        });

        let config = TimeJitterAugmentation::new(500000.0) // 0.5s std - large jitter
            .with_sorting(true)
            .with_seed(42);

        let jittered = time_jitter(&events, &config).unwrap();

        // Check that events are sorted
        for i in 1..jittered.len() {
            assert!(jittered[i].t >= jittered[i - 1].t);
        }
    }

    #[test]
    fn test_time_jitter_reproducibility() {
        let events = create_test_events();
        let config = TimeJitterAugmentation::new(1000.0).with_seed(12345);

        let jittered1 = time_jitter(&events, &config).unwrap();
        let jittered2 = time_jitter(&events, &config).unwrap();

        // With same seed, results should be identical
        assert_eq!(jittered1.len(), jittered2.len());
        for (e1, e2) in jittered1.iter().zip(jittered2.iter()) {
            assert!((e1.t - e2.t).abs() < 1e-10);
        }
    }

    #[test]
    fn test_time_jitter_empty_events() {
        let events = Vec::new();
        let config = TimeJitterAugmentation::new(1000.0);
        let jittered = time_jitter(&events, &config).unwrap();
        assert!(jittered.is_empty());
    }
}
