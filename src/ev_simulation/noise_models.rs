//! Noise models for realistic event camera simulation
//!
//! Implements various noise sources found in real event cameras:
//! - Shot noise (Poisson statistics)
//! - Dark current noise
//! - Pixel mismatch
//! - Thermal noise
//! - Temporal correlation effects

use crate::ev_core::Event;

/// Comprehensive noise model configuration
#[derive(Debug, Clone, Default)]
pub struct NoiseModelConfig {
    /// Shot noise parameters
    pub shot_noise: ShotNoiseConfig,
    /// Dark current noise parameters
    pub dark_current: DarkCurrentConfig,
    /// Pixel mismatch parameters
    pub pixel_mismatch: PixelMismatchConfig,
    /// Thermal noise parameters
    pub thermal_noise: ThermalNoiseConfig,
    /// Temporal correlation parameters
    pub temporal_correlation: TemporalCorrelationConfig,
}

/// Shot noise configuration (Poisson statistics)
#[derive(Debug, Clone)]
pub struct ShotNoiseConfig {
    /// Enable shot noise
    pub enabled: bool,
    /// Shot noise scaling factor
    pub scale_factor: f64,
    /// Minimum photon count for noise calculation
    pub min_photon_count: f64,
}

impl Default for ShotNoiseConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            scale_factor: 1.0,
            min_photon_count: 1.0,
        }
    }
}

/// Dark current noise configuration
#[derive(Debug, Clone)]
pub struct DarkCurrentConfig {
    /// Enable dark current noise
    pub enabled: bool,
    /// Dark current rate (events/pixel/second)
    pub rate_hz: f64,
    /// Temperature dependency (events/pixel/second/°C)
    pub temperature_coefficient: f64,
    /// Reference temperature (°C)
    pub reference_temperature: f64,
}

impl Default for DarkCurrentConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            rate_hz: 0.01,
            temperature_coefficient: 0.001,
            reference_temperature: 25.0,
        }
    }
}

/// Pixel mismatch configuration
#[derive(Debug, Clone)]
pub struct PixelMismatchConfig {
    /// Enable pixel mismatch
    pub enabled: bool,
    /// Standard deviation of threshold mismatch
    pub threshold_std: f64,
    /// Standard deviation of gain mismatch
    pub gain_std: f64,
    /// Spatial correlation length (pixels)
    pub correlation_length: f64,
}

impl Default for PixelMismatchConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold_std: 0.05,
            gain_std: 0.02,
            correlation_length: 2.0,
        }
    }
}

/// Thermal noise configuration
#[derive(Debug, Clone)]
pub struct ThermalNoiseConfig {
    /// Enable thermal noise
    pub enabled: bool,
    /// Thermal noise variance
    pub variance: f64,
    /// Temperature (°C)
    pub temperature: f64,
}

impl Default for ThermalNoiseConfig {
    fn default() -> Self {
        Self {
            enabled: false, // Usually negligible for event cameras
            variance: 1e-6,
            temperature: 25.0,
        }
    }
}

/// Temporal correlation configuration
#[derive(Debug, Clone)]
pub struct TemporalCorrelationConfig {
    /// Enable temporal correlation
    pub enabled: bool,
    /// Correlation time constant (seconds)
    pub time_constant: f64,
    /// Correlation strength [0, 1]
    pub correlation_strength: f64,
}

impl Default for TemporalCorrelationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            time_constant: 0.001, // 1ms
            correlation_strength: 0.1,
        }
    }
}

/// Noise model implementation
pub struct NoiseModel {
    config: NoiseModelConfig,

    // Internal state for correlated noise
    pixel_gains: Vec<f64>,
    pixel_thresholds: Vec<f64>,
    temporal_state: Vec<f64>,

    // Statistics
    total_noise_events: u64,
    shot_noise_events: u64,
    dark_current_events: u64,
}

impl NoiseModel {
    /// Create new noise model
    pub fn new(config: NoiseModelConfig, width: usize, height: usize) -> Self {
        let total_pixels = width * height;

        // Initialize pixel mismatch parameters
        let pixel_gains = Self::generate_pixel_gains(total_pixels, &config.pixel_mismatch);
        let pixel_thresholds =
            Self::generate_pixel_thresholds(total_pixels, &config.pixel_mismatch);
        let temporal_state = vec![0.0; total_pixels];

        println!("Noise model initialized:");
        println!("  Shot noise: {}", config.shot_noise.enabled);
        println!("  Dark current: {} Hz", config.dark_current.rate_hz);
        println!(
            "  Pixel mismatch: σ_thresh={:.3}, σ_gain={:.3}",
            config.pixel_mismatch.threshold_std, config.pixel_mismatch.gain_std
        );

        Self {
            config,
            pixel_gains,
            pixel_thresholds,
            temporal_state,
            total_noise_events: 0,
            shot_noise_events: 0,
            dark_current_events: 0,
        }
    }

    /// Generate pixel gain variations
    fn generate_pixel_gains(num_pixels: usize, config: &PixelMismatchConfig) -> Vec<f64> {
        if !config.enabled || config.gain_std <= 0.0 {
            return vec![1.0; num_pixels];
        }

        let mut gains = Vec::with_capacity(num_pixels);
        for _ in 0..num_pixels {
            // Gaussian distribution around 1.0
            let gain = 1.0 + Self::random_normal() * config.gain_std;
            gains.push(gain.max(0.1)); // Clamp to reasonable range
        }

        gains
    }

    /// Generate pixel threshold variations
    fn generate_pixel_thresholds(num_pixels: usize, config: &PixelMismatchConfig) -> Vec<f64> {
        if !config.enabled || config.threshold_std <= 0.0 {
            return vec![0.0; num_pixels];
        }

        let mut thresholds = Vec::with_capacity(num_pixels);
        for _ in 0..num_pixels {
            // Gaussian distribution around 0.0
            let threshold_offset = Self::random_normal() * config.threshold_std;
            thresholds.push(threshold_offset);
        }

        thresholds
    }

    /// Generate normally distributed random number (Box-Muller transform)
    fn random_normal() -> f64 {
        use std::f64::consts::PI;

        static mut SPARE: Option<f64> = None;
        static mut HAS_SPARE: bool = false;

        unsafe {
            if HAS_SPARE {
                HAS_SPARE = false;
                return SPARE.unwrap();
            }

            HAS_SPARE = true;
            let u = fastrand::f64();
            let v = fastrand::f64();
            let mag = (-2.0 * u.ln()).sqrt();
            SPARE = Some(mag * (2.0 * PI * v).cos());
            mag * (2.0 * PI * v).sin()
        }
    }

    /// Apply noise model to events
    pub fn apply_noise(&mut self, events: Vec<Event>, timestamp_us: f64, dt_us: f64) -> Vec<Event> {
        let mut noisy_events = events;

        // Apply pixel mismatch to existing events
        if self.config.pixel_mismatch.enabled {
            self.apply_pixel_mismatch(&mut noisy_events);
        }

        // Add shot noise events
        if self.config.shot_noise.enabled {
            let shot_events = self.generate_shot_noise(&noisy_events, timestamp_us);
            self.shot_noise_events += shot_events.len() as u64;
            noisy_events.extend(shot_events);
        }

        // Add dark current events
        if self.config.dark_current.enabled {
            let dark_events = self.generate_dark_current_events(timestamp_us, dt_us);
            self.dark_current_events += dark_events.len() as u64;
            noisy_events.extend(dark_events);
        }

        // Apply temporal correlation
        if self.config.temporal_correlation.enabled {
            self.apply_temporal_correlation(&mut noisy_events, timestamp_us);
        }

        // Update statistics
        self.total_noise_events = self.shot_noise_events + self.dark_current_events;

        // Sort by timestamp
        noisy_events.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap());

        noisy_events
    }

    /// Apply pixel mismatch to existing events
    fn apply_pixel_mismatch(&self, events: &mut [Event]) {
        let width = (self.pixel_gains.len() as f64).sqrt() as usize; // Assume square for simplicity

        for event in events {
            let pixel_idx = event.y as usize * width + event.x as usize;
            if pixel_idx < self.pixel_gains.len() {
                // Apply gain mismatch (affects event timing/probability)
                let gain = self.pixel_gains[pixel_idx];

                // Apply threshold mismatch (affects event position slightly)
                let threshold_offset = self.pixel_thresholds[pixel_idx];

                // Modify event position slightly based on threshold mismatch
                let pos_noise = threshold_offset * 0.1; // Small position shift
                event.x = ((event.x as f64 + pos_noise).clamp(0.0, width as f64 - 1.0)) as u16;

                // Modify event timing slightly based on gain mismatch
                let time_noise = (1.0 - gain) * 1e-6; // Small time shift (microseconds)
                event.t += time_noise;
            }
        }
    }

    /// Generate shot noise events
    fn generate_shot_noise(&self, events: &[Event], _timestamp_us: f64) -> Vec<Event> {
        if events.is_empty() {
            return Vec::new();
        }

        let mut shot_events = Vec::new();

        // For each real event, potentially generate shot noise events nearby
        for event in events {
            let photon_count = 10.0; // Assume ~10 photons per event
            let noise_events =
                Self::poisson_sample(photon_count * self.config.shot_noise.scale_factor);

            for _ in 0..noise_events {
                // Generate noise event near original event
                let noise_x_offset = (Self::random_normal() * 0.5).round() as i32;
                let noise_y_offset = (Self::random_normal() * 0.5).round() as i32;

                let new_x = ((event.x as i32 + noise_x_offset).max(0) as u16).min(639); // Clamp to sensor
                let new_y = ((event.y as i32 + noise_y_offset).max(0) as u16).min(479);

                let time_jitter = Self::random_normal() * 1e-6; // 1μs jitter

                shot_events.push(Event {
                    t: event.t + time_jitter,
                    x: new_x,
                    y: new_y,
                    polarity: event.polarity,
                });
            }
        }

        shot_events
    }

    /// Generate dark current noise events
    fn generate_dark_current_events(&self, timestamp_us: f64, dt_us: f64) -> Vec<Event> {
        let total_pixels = self.pixel_gains.len();
        let dt_seconds = dt_us / 1_000_000.0;

        // Calculate temperature-dependent rate
        let temp_factor = 1.0
            + self.config.dark_current.temperature_coefficient
                * (self.config.thermal_noise.temperature
                    - self.config.dark_current.reference_temperature);
        let effective_rate = self.config.dark_current.rate_hz * temp_factor;

        // Expected number of dark current events
        let expected_events = total_pixels as f64 * effective_rate * dt_seconds;
        let num_events = Self::poisson_sample(expected_events);

        let mut dark_events = Vec::with_capacity(num_events);
        let width = (total_pixels as f64).sqrt() as u16;
        let height = total_pixels as u16 / width;

        for _ in 0..num_events {
            let x = fastrand::u16(0..width);
            let y = fastrand::u16(0..height);
            let polarity = if fastrand::bool() { 1 } else { -1 };
            let time_offset = fastrand::f64() * dt_us;

            dark_events.push(Event {
                t: (timestamp_us + time_offset) / 1_000_000.0,
                x,
                y,
                polarity: polarity > 0,
            });
        }

        dark_events
    }

    /// Apply temporal correlation effects
    fn apply_temporal_correlation(&mut self, events: &mut [Event], _timestamp_us: f64) {
        if !self.config.temporal_correlation.enabled {
            return;
        }

        // Apply exponential decay to temporal state
        let decay_factor = (-1.0 / self.config.temporal_correlation.time_constant).exp();
        for state in &mut self.temporal_state {
            *state *= decay_factor;
        }

        // Update temporal state based on current events
        let width = (self.temporal_state.len() as f64).sqrt() as usize;

        for event in events.iter_mut() {
            let pixel_idx = event.y as usize * width + event.x as usize;
            if pixel_idx < self.temporal_state.len() {
                // Add to temporal state
                self.temporal_state[pixel_idx] += 1.0;

                // Apply correlation effect to event timing
                let correlation_effect = self.temporal_state[pixel_idx]
                    * self.config.temporal_correlation.correlation_strength
                    * 1e-6;
                event.t += correlation_effect;
            }
        }
    }

    /// Poisson sampling
    fn poisson_sample(lambda: f64) -> usize {
        if lambda <= 0.0 {
            return 0;
        }

        if lambda < 30.0 {
            // Use Knuth's algorithm for small lambda
            let l = (-lambda).exp();
            let mut k = 0;
            let mut p = 1.0;

            loop {
                k += 1;
                p *= fastrand::f64();
                if p <= l {
                    break;
                }
            }

            k - 1
        } else {
            // Use normal approximation for large lambda
            let normal_sample = Self::random_normal();
            (lambda + lambda.sqrt() * normal_sample).round().max(0.0) as usize
        }
    }

    /// Get noise statistics
    pub fn get_noise_stats(&self) -> NoiseStats {
        NoiseStats {
            total_noise_events: self.total_noise_events,
            shot_noise_events: self.shot_noise_events,
            dark_current_events: self.dark_current_events,
            noise_rate_hz: self.calculate_noise_rate(),
        }
    }

    /// Calculate current noise rate
    fn calculate_noise_rate(&self) -> f64 {
        // This would track noise rate over time in a real implementation
        self.config.dark_current.rate_hz
    }

    /// Reset noise model state
    pub fn reset(&mut self) {
        self.temporal_state.fill(0.0);
        self.total_noise_events = 0;
        self.shot_noise_events = 0;
        self.dark_current_events = 0;
    }
}

/// Noise statistics
#[derive(Debug, Clone)]
pub struct NoiseStats {
    pub total_noise_events: u64,
    pub shot_noise_events: u64,
    pub dark_current_events: u64,
    pub noise_rate_hz: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_model_config_default() {
        let config = NoiseModelConfig::default();
        assert!(config.shot_noise.enabled);
        assert!(config.dark_current.enabled);
        assert!(config.pixel_mismatch.enabled);
        assert!(!config.thermal_noise.enabled);
        assert!(!config.temporal_correlation.enabled);
    }

    #[test]
    fn test_shot_noise_config() {
        let config = ShotNoiseConfig::default();
        assert!(config.enabled);
        assert_eq!(config.scale_factor, 1.0);
        assert_eq!(config.min_photon_count, 1.0);
    }

    #[test]
    fn test_dark_current_config() {
        let config = DarkCurrentConfig::default();
        assert!(config.enabled);
        assert_eq!(config.rate_hz, 0.01);
        assert_eq!(config.reference_temperature, 25.0);
    }

    #[test]
    fn test_noise_model_creation() {
        let config = NoiseModelConfig::default();
        let noise_model = NoiseModel::new(config, 640, 480);

        assert_eq!(noise_model.pixel_gains.len(), 640 * 480);
        assert_eq!(noise_model.pixel_thresholds.len(), 640 * 480);
        assert_eq!(noise_model.temporal_state.len(), 640 * 480);
    }

    #[test]
    fn test_poisson_sampling() {
        // Test edge cases
        assert_eq!(NoiseModel::poisson_sample(0.0), 0);

        // Test small lambda
        let small_sample = NoiseModel::poisson_sample(2.0);
        assert!(small_sample < 20); // Very unlikely to get >20 with lambda=2

        // Test large lambda
        let large_sample = NoiseModel::poisson_sample(100.0);
        assert!(large_sample > 50 && large_sample < 150); // Should be around 100
    }

    #[test]
    fn test_pixel_gain_generation() {
        let config = PixelMismatchConfig::default();
        let gains = NoiseModel::generate_pixel_gains(100, &config);

        assert_eq!(gains.len(), 100);

        // Check that gains are reasonable (around 1.0)
        let avg_gain: f64 = gains.iter().sum::<f64>() / gains.len() as f64;
        assert!((avg_gain - 1.0).abs() < 0.1); // Should be close to 1.0
    }

    #[test]
    fn test_random_normal_generation() {
        let mut samples = Vec::new();
        for _ in 0..1000 {
            samples.push(NoiseModel::random_normal());
        }

        // Check mean is close to 0
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!(mean.abs() < 0.1);

        // Check standard deviation is close to 1
        let variance: f64 =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        let std_dev = variance.sqrt();
        assert!((std_dev - 1.0).abs() < 0.1);
    }
}
