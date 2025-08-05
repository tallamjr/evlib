//! Uniform noise augmentation for event data
//!
//! This module implements noise injection by adding uniformly distributed
//! synthetic events across the sensor dimensions and time range.

use crate::ev_augmentation::{
    AugmentationError, AugmentationResult, SingleAugmentation, Validatable,
};

#[cfg(feature = "polars")]
use crate::ev_augmentation::COL_T;
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
use polars::prelude::*;

/// Uniform noise augmentation configuration
///
/// Adds a fixed number of noise events that are uniformly distributed
/// across sensor dimensions and the time range of existing events.
/// This simulates background activity and sensor noise.
///
/// # Example
///
/// ```rust
/// use evlib::ev_augmentation::UniformNoiseAugmentation;
///
/// // Add 1000 noise events to a 640x480 sensor
/// let noise = UniformNoiseAugmentation::new(1000, 640, 480);
///
/// // Add noise with balanced polarities
/// let noise = UniformNoiseAugmentation::new(500, 640, 480)
///     .with_polarity_balance(true);
/// ```
#[derive(Debug, Clone)]
pub struct UniformNoiseAugmentation {
    /// Number of noise events to add
    pub n_events: usize,
    /// Sensor width in pixels
    pub sensor_width: u16,
    /// Sensor height in pixels
    pub sensor_height: u16,
    /// Whether to balance positive and negative polarities
    pub balance_polarities: bool,
    /// Whether to sort events after adding noise
    pub sort_timestamps: bool,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl UniformNoiseAugmentation {
    /// Create a new uniform noise augmentation
    ///
    /// # Arguments
    ///
    /// * `n_events` - Number of noise events to add
    /// * `sensor_width` - Sensor width in pixels
    /// * `sensor_height` - Sensor height in pixels
    pub fn new(n_events: usize, sensor_width: u16, sensor_height: u16) -> Self {
        Self {
            n_events,
            sensor_width,
            sensor_height,
            balance_polarities: false,
            sort_timestamps: true,
            seed: None,
        }
    }

    /// Set whether to balance positive and negative polarities
    pub fn with_polarity_balance(mut self, balance: bool) -> Self {
        self.balance_polarities = balance;
        self
    }

    /// Set whether to sort events after adding noise
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
        format!(
            "n={}, sensor={}x{}",
            self.n_events, self.sensor_width, self.sensor_height
        )
    }
}

impl Validatable for UniformNoiseAugmentation {
    fn validate(&self) -> AugmentationResult<()> {
        if self.sensor_width == 0 || self.sensor_height == 0 {
            return Err(AugmentationError::InvalidSensorSize(
                self.sensor_width,
                self.sensor_height,
            ));
        }
        Ok(())
    }
}

impl SingleAugmentation for UniformNoiseAugmentation {
    fn apply(&self, events: &Events) -> AugmentationResult<Events> {
        uniform_noise(events, self)
    }

    fn description(&self) -> String {
        format!("Uniform noise: {}", self.description())
    }
}

/// Apply uniform noise to events
///
/// This function adds uniformly distributed noise events across the sensor
/// dimensions and time range of the input events.
///
/// # Arguments
///
/// * `events` - Input events to augment
/// * `config` - Uniform noise configuration
///
/// # Returns
///
/// * `AugmentationResult<Events>` - Original events plus noise events
#[cfg_attr(feature = "tracing", instrument(skip(events), fields(n_events = events.len())))]
pub fn uniform_noise(
    events: &Events,
    config: &UniformNoiseAugmentation,
) -> AugmentationResult<Events> {
    let start_time = std::time::Instant::now();

    // Validate configuration
    config.validate()?;

    // Handle empty events case
    if events.is_empty() {
        debug!("No events provided, generating noise in default time range");
        // Generate noise in a default time range if no events exist
        return generate_noise_only(config, 0.0, 1.0);
    }

    // Find time range
    let min_time = events
        .iter()
        .map(|e| e.t)
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0.0);
    let max_time = events
        .iter()
        .map(|e| e.t)
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(1.0);

    // Initialize RNG
    let mut rng = if let Some(seed) = config.seed {
        rand::rngs::StdRng::seed_from_u64(seed)
    } else {
        rand::rngs::StdRng::from_entropy()
    };

    // Create distributions
    let x_dist = Uniform::new(0, config.sensor_width);
    let y_dist = Uniform::new(0, config.sensor_height);
    let t_dist = Uniform::new(min_time, max_time);

    // Generate noise events
    let mut noise_events = Vec::with_capacity(config.n_events);

    for i in 0..config.n_events {
        let polarity = if config.balance_polarities {
            i % 2 == 0
        } else {
            rng.gen_bool(0.5)
        };

        let noise_event = Event {
            t: t_dist.sample(&mut rng),
            x: x_dist.sample(&mut rng),
            y: y_dist.sample(&mut rng),
            polarity,
        };

        noise_events.push(noise_event);
    }

    // Combine with original events
    let mut combined_events = events.clone();
    combined_events.extend(noise_events);

    // Sort if requested
    if config.sort_timestamps {
        combined_events.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));
    }

    let processing_time = start_time.elapsed().as_secs_f64();
    let noise_count = config.n_events;
    let output_count = combined_events.len();

    info!(
        "Uniform noise applied: {} + {} noise = {} events in {:.3}s",
        events.len(),
        noise_count,
        output_count,
        processing_time
    );

    Ok(combined_events)
}

/// Generate noise events only (when no input events provided)
fn generate_noise_only(
    config: &UniformNoiseAugmentation,
    min_time: f64,
    max_time: f64,
) -> AugmentationResult<Events> {
    let mut rng = if let Some(seed) = config.seed {
        rand::rngs::StdRng::seed_from_u64(seed)
    } else {
        rand::rngs::StdRng::from_entropy()
    };

    let x_dist = Uniform::new(0, config.sensor_width);
    let y_dist = Uniform::new(0, config.sensor_height);
    let t_dist = Uniform::new(min_time, max_time);

    let mut noise_events = Vec::with_capacity(config.n_events);

    for i in 0..config.n_events {
        let polarity = if config.balance_polarities {
            i % 2 == 0
        } else {
            rng.gen_bool(0.5)
        };

        let noise_event = Event {
            t: t_dist.sample(&mut rng),
            x: x_dist.sample(&mut rng),
            y: y_dist.sample(&mut rng),
            polarity,
        };

        noise_events.push(noise_event);
    }

    if config.sort_timestamps {
        noise_events.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));
    }

    Ok(noise_events)
}

/// Apply uniform noise using Polars operations (Polars-first implementation)
///
/// This implementation uses vectorized operations for better performance on large datasets.
///
/// # Arguments
///
/// * `df` - Input LazyFrame containing event data
/// * `config` - Uniform noise configuration
///
/// # Returns
///
/// * `PolarsResult<LazyFrame>` - Combined events as LazyFrame
#[cfg(feature = "polars")]
#[cfg_attr(feature = "tracing", instrument(skip(df), fields(config = ?config)))]
pub fn apply_uniform_noise_polars(
    df: LazyFrame,
    config: &UniformNoiseAugmentation,
) -> PolarsResult<LazyFrame> {
    debug!("Applying uniform noise with Polars: {:?}", config);

    // Get time range from existing events
    let time_stats = df
        .clone()
        .select([
            col(COL_T).min().alias("min_time"),
            col(COL_T).max().alias("max_time"),
        ])
        .collect()?;

    let _min_time = time_stats.column("min_time")?.f64()?.get(0).unwrap_or(0.0);
    let _max_time = time_stats.column("max_time")?.f64()?.get(0).unwrap_or(1.0);

    // For now, use the Vec implementation and convert
    // A fully vectorized Polars implementation would require creating a new DataFrame
    let collected_df = df.collect()?;
    let events = crate::ev_augmentation::dataframe_to_events(&collected_df)
        .map_err(|e| PolarsError::ComputeError(format!("Conversion error: {}", e).into()))?;

    let noisy = uniform_noise(&events, config)
        .map_err(|e| PolarsError::ComputeError(format!("Noise error: {}", e).into()))?;

    let noisy_df = crate::ev_core::events_to_dataframe(&noisy)?;

    Ok(noisy_df.lazy())
}

/// Convenience function for simple uniform noise
///
/// # Arguments
///
/// * `events` - Input events
/// * `n_events` - Number of noise events to add
/// * `sensor_width` - Sensor width
/// * `sensor_height` - Sensor height
pub fn add_uniform_noise_simple(
    events: &Events,
    n_events: usize,
    sensor_width: u16,
    sensor_height: u16,
) -> AugmentationResult<Events> {
    let config = UniformNoiseAugmentation::new(n_events, sensor_width, sensor_height);
    uniform_noise(events, &config)
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
    fn test_uniform_noise_creation() {
        let noise = UniformNoiseAugmentation::new(100, 640, 480);
        assert_eq!(noise.n_events, 100);
        assert_eq!(noise.sensor_width, 640);
        assert_eq!(noise.sensor_height, 480);
        assert!(!noise.balance_polarities);
        assert!(noise.sort_timestamps);
    }

    #[test]
    fn test_uniform_noise_validation() {
        // Valid configuration
        let valid = UniformNoiseAugmentation::new(100, 640, 480);
        assert!(valid.validate().is_ok());

        // Invalid: zero sensor size
        let invalid = UniformNoiseAugmentation::new(100, 0, 480);
        assert!(invalid.validate().is_err());

        let invalid = UniformNoiseAugmentation::new(100, 640, 0);
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_uniform_noise_application() {
        let events = create_test_events();
        let config = UniformNoiseAugmentation::new(10, 640, 480).with_seed(42);

        let noisy = uniform_noise(&events, &config).unwrap();

        // Should have original events plus noise
        assert_eq!(noisy.len(), events.len() + 10);

        // First events should be original (if sorted)
        // This depends on the time distribution, so we just check count
    }

    #[test]
    fn test_uniform_noise_bounds() {
        let events = create_test_events();
        let config = UniformNoiseAugmentation::new(100, 640, 480).with_seed(42);

        let noisy = uniform_noise(&events, &config).unwrap();

        // All noise events should be within bounds
        for event in noisy.iter().skip(events.len()) {
            assert!(event.x < 640);
            assert!(event.y < 480);
            assert!(event.t >= 1.0 && event.t <= 3.0); // Time range of original events
        }
    }

    #[test]
    fn test_uniform_noise_polarity_balance() {
        let events = create_test_events();
        let config = UniformNoiseAugmentation::new(100, 640, 480)
            .with_polarity_balance(true)
            .with_seed(42);

        let noisy = uniform_noise(&events, &config).unwrap();

        // Count polarities in noise events
        let noise_events: Vec<_> = noisy.iter().skip(events.len()).collect();
        let positive_count = noise_events.iter().filter(|e| e.polarity).count();
        let negative_count = noise_events.iter().filter(|e| !e.polarity).count();

        // Should be exactly balanced (or differ by at most 1 for odd numbers)
        assert!((positive_count as i32 - negative_count as i32).abs() <= 1);
    }

    #[test]
    fn test_uniform_noise_sorting() {
        let events = create_test_events();
        let config = UniformNoiseAugmentation::new(50, 640, 480)
            .with_sorting(true)
            .with_seed(42);

        let noisy = uniform_noise(&events, &config).unwrap();

        // Check that events are sorted
        for i in 1..noisy.len() {
            assert!(noisy[i].t >= noisy[i - 1].t);
        }
    }

    #[test]
    fn test_uniform_noise_reproducibility() {
        let events = create_test_events();
        let config = UniformNoiseAugmentation::new(20, 640, 480).with_seed(12345);

        let noisy1 = uniform_noise(&events, &config).unwrap();
        let noisy2 = uniform_noise(&events, &config).unwrap();

        // With same seed, results should be identical
        assert_eq!(noisy1.len(), noisy2.len());
        for (e1, e2) in noisy1.iter().zip(noisy2.iter()) {
            assert_eq!(e1.x, e2.x);
            assert_eq!(e1.y, e2.y);
            assert!((e1.t - e2.t).abs() < 1e-10);
            assert_eq!(e1.polarity, e2.polarity);
        }
    }

    #[test]
    fn test_uniform_noise_empty_events() {
        let events = Vec::new();
        let config = UniformNoiseAugmentation::new(10, 640, 480);
        let noisy = uniform_noise(&events, &config).unwrap();

        // Should generate noise in default time range
        assert_eq!(noisy.len(), 10);

        // All should be within sensor bounds
        for event in &noisy {
            assert!(event.x < 640);
            assert!(event.y < 480);
            assert!(event.t >= 0.0 && event.t <= 1.0); // Default time range
        }
    }
}
