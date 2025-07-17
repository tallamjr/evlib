//! Event camera simulation - Video to Events conversion
//!
//! This module implements event camera simulation following the ESIM approach:
//! - Intensity-based event generation with configurable thresholds
//! - Noise models and camera parameter simulation
//! - Support for various video formats and real-time processing

use candle_core::{DType, Device, Result as CandleResult, Tensor};
use std::path::Path;

pub mod esim;
pub mod noise_models;
pub mod video_processing;

#[cfg(feature = "gstreamer")]
pub mod gstreamer_video;

#[cfg(feature = "gstreamer")]
pub mod realtime_stream;

#[cfg(feature = "python")]
pub mod python;

use crate::ev_core::Event;

/// Event camera simulation configuration
#[derive(Debug, Clone)]
pub struct SimulationConfig {
    /// Image resolution (width, height)
    pub resolution: (u32, u32),
    /// Positive contrast threshold
    pub contrast_threshold_pos: f64,
    /// Negative contrast threshold
    pub contrast_threshold_neg: f64,
    /// Refractory period in microseconds
    pub refractory_period_us: f64,
    /// Simulation timestep in microseconds
    pub timestep_us: f64,
    /// Enable noise simulation
    pub enable_noise: bool,
    /// Noise parameters
    pub noise_config: NoiseConfig,
    /// Camera model parameters
    pub camera_config: CameraConfig,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            resolution: (640, 480),
            contrast_threshold_pos: 0.2,
            contrast_threshold_neg: 0.2,
            refractory_period_us: 100.0,
            timestep_us: 1000.0, // 1ms default timestep
            enable_noise: true,
            noise_config: NoiseConfig::default(),
            camera_config: CameraConfig::default(),
        }
    }
}

/// Noise model configuration
#[derive(Debug, Clone)]
pub struct NoiseConfig {
    /// Shot noise variance
    pub shot_noise_variance: f64,
    /// Dark current noise rate (events/pixel/second)
    pub dark_current_rate: f64,
    /// Pixel mismatch standard deviation
    pub pixel_mismatch_std: f64,
    /// Enable temporal noise correlation
    pub temporal_correlation: bool,
}

impl Default for NoiseConfig {
    fn default() -> Self {
        Self {
            shot_noise_variance: 0.1,
            dark_current_rate: 0.01,
            pixel_mismatch_std: 0.05,
            temporal_correlation: false,
        }
    }
}

/// Camera model configuration
#[derive(Debug, Clone)]
pub struct CameraConfig {
    /// Lens distortion parameters [k1, k2, k3, p1, p2]
    pub distortion: [f64; 5],
    /// Camera matrix [fx, fy, cx, cy]
    pub intrinsics: [f64; 4],
    /// Exposure time in microseconds
    pub exposure_time_us: f64,
    /// Frame rate for reference timing
    pub fps: f64,
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            distortion: [0.0, 0.0, 0.0, 0.0, 0.0],
            intrinsics: [320.0, 320.0, 320.0, 240.0], // fx, fy, cx, cy
            exposure_time_us: 33333.0,                // ~30fps
            fps: 30.0,
        }
    }
}

/// Event simulator state
pub struct EventSimulator {
    config: SimulationConfig,
    device: Device,

    // Internal state
    last_intensity: Option<Tensor>,
    last_timestamp: f64,
    pixel_states: Vec<PixelState>,
    frame_counter: u64,
}

/// Per-pixel simulation state
#[derive(Debug, Clone)]
struct PixelState {
    last_event_time: f64,
    intensity_buffer: f64,
    #[allow(dead_code)]
    reference_intensity: f64,
}

impl PixelState {
    fn new() -> Self {
        Self {
            last_event_time: 0.0,
            intensity_buffer: 0.0,
            reference_intensity: 0.0,
        }
    }
}

impl EventSimulator {
    /// Create a new event simulator
    pub fn new(config: SimulationConfig, device: Device) -> CandleResult<Self> {
        let total_pixels = (config.resolution.0 * config.resolution.1) as usize;
        let pixel_states = vec![PixelState::new(); total_pixels];

        println!("Event simulator initialized:");
        println!(
            "  Resolution: {}x{}",
            config.resolution.0, config.resolution.1
        );
        println!(
            "  Contrast thresholds: +{:.2}, -{:.2}",
            config.contrast_threshold_pos, config.contrast_threshold_neg
        );
        println!("  Timestep: {:.1}μs", config.timestep_us);

        Ok(Self {
            config,
            device,
            last_intensity: None,
            last_timestamp: 0.0,
            pixel_states,
            frame_counter: 0,
        })
    }

    /// Simulate events from a single frame
    pub fn simulate_frame(
        &mut self,
        intensity_frame: &Tensor,
        timestamp_us: f64,
    ) -> CandleResult<Vec<Event>> {
        let mut events = Vec::new();

        // Validate frame dimensions
        let frame_shape = intensity_frame.shape();
        if frame_shape.dims().len() != 2 {
            return Err(candle_core::Error::Msg(
                "Frame must be 2D (height, width)".to_string(),
            ));
        }

        let (height, width) = (frame_shape.dims()[0], frame_shape.dims()[1]);
        if width != self.config.resolution.0 as usize || height != self.config.resolution.1 as usize
        {
            return Err(candle_core::Error::Msg(format!(
                "Frame size {}x{} doesn't match config {}x{}",
                width, height, self.config.resolution.0, self.config.resolution.1
            )));
        }

        // Convert frame to device and normalize
        let current_intensity = intensity_frame.to_device(&self.device)?;
        let current_intensity = self.normalize_intensity(&current_intensity)?;

        // Generate events if we have a previous frame
        if let Some(last_intensity) = self.last_intensity.take() {
            events = self.generate_events_from_diff(
                &last_intensity,
                &current_intensity,
                self.last_timestamp,
                timestamp_us,
            )?;
        }

        // Update state
        self.last_intensity = Some(current_intensity);
        self.last_timestamp = timestamp_us;
        self.frame_counter += 1;

        // Apply noise if enabled
        if self.config.enable_noise {
            events = self.apply_noise_model(events, timestamp_us)?;
        }

        Ok(events)
    }

    /// Normalize intensity frame to [0, 1] range
    fn normalize_intensity(&self, intensity: &Tensor) -> CandleResult<Tensor> {
        // Convert to f32 if needed
        let intensity_f32 = intensity.to_dtype(DType::F32)?;

        // Normalize to [0, 1] range
        let min_val = intensity_f32.min(1)?.min(0)?;
        let max_val = intensity_f32.max(1)?.max(0)?;
        let range = (&max_val - &min_val)?;

        // Avoid division by zero
        let range = range.clamp(1e-8, f32::INFINITY)?;
        let normalized = (intensity_f32.broadcast_sub(&min_val))?.broadcast_div(&range)?;

        Ok(normalized)
    }

    /// Generate events from intensity difference
    fn generate_events_from_diff(
        &mut self,
        last_intensity: &Tensor,
        current_intensity: &Tensor,
        last_time: f64,
        current_time: f64,
    ) -> CandleResult<Vec<Event>> {
        let mut events = Vec::new();

        // Calculate log intensity change
        let last_log = last_intensity.log()?;
        let current_log = current_intensity.log()?;
        let log_diff = (current_log - last_log)?;

        // Convert to CPU for pixel-wise processing
        let log_diff_cpu = log_diff.to_device(&Device::Cpu)?;
        let log_diff_data = log_diff_cpu.flatten_all()?.to_vec1::<f32>()?;

        let width = self.config.resolution.0 as usize;
        let height = self.config.resolution.1 as usize;
        let _dt = current_time - last_time;

        // Process each pixel
        for (idx, &log_change) in log_diff_data.iter().enumerate() {
            let y = idx / width;
            let x = idx % width;

            if x >= width || y >= height {
                continue;
            }

            let pixel_state = &mut self.pixel_states[idx];

            // Check refractory period
            if current_time - pixel_state.last_event_time < self.config.refractory_period_us {
                continue;
            }

            // Accumulate intensity change
            pixel_state.intensity_buffer += log_change as f64;

            // Check for positive events
            if pixel_state.intensity_buffer >= self.config.contrast_threshold_pos {
                let event_time = Self::interpolate_event_time(
                    last_time,
                    current_time,
                    pixel_state.intensity_buffer,
                    self.config.contrast_threshold_pos,
                );

                events.push(Event {
                    t: event_time / 1_000_000.0, // Convert μs to seconds
                    x: x as u16,
                    y: y as u16,
                    polarity: true, // Positive event
                });

                pixel_state.intensity_buffer -= self.config.contrast_threshold_pos;
                pixel_state.last_event_time = event_time;
            }

            // Check for negative events
            if pixel_state.intensity_buffer <= -self.config.contrast_threshold_neg {
                let event_time = Self::interpolate_event_time(
                    last_time,
                    current_time,
                    pixel_state.intensity_buffer.abs(),
                    self.config.contrast_threshold_neg,
                );

                events.push(Event {
                    t: event_time / 1_000_000.0, // Convert μs to seconds
                    x: x as u16,
                    y: y as u16,
                    polarity: false, // Negative event
                });

                pixel_state.intensity_buffer += self.config.contrast_threshold_neg;
                pixel_state.last_event_time = event_time;
            }
        }

        // Sort events by timestamp
        events.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap());

        Ok(events)
    }

    /// Interpolate event timing within frame interval
    fn interpolate_event_time(t1: f64, t2: f64, accumulated: f64, threshold: f64) -> f64 {
        // Linear interpolation based on when threshold was crossed
        let ratio = (threshold / accumulated).clamp(0.0, 1.0);
        t1 + ratio * (t2 - t1)
    }

    /// Apply noise model to events
    fn apply_noise_model(
        &self,
        mut events: Vec<Event>,
        timestamp_us: f64,
    ) -> CandleResult<Vec<Event>> {
        if !self.config.enable_noise {
            return Ok(events);
        }

        // Add dark current noise events
        let dark_events = self.generate_dark_current_events(timestamp_us)?;
        events.extend(dark_events);

        // Apply pixel mismatch (modify existing event positions slightly)
        for event in &mut events {
            if self.config.noise_config.pixel_mismatch_std > 0.0 {
                // Add small random offset to position (simulate pixel mismatch)
                let noise_x = fastrand::f64() * self.config.noise_config.pixel_mismatch_std;
                let noise_y = fastrand::f64() * self.config.noise_config.pixel_mismatch_std;

                event.x = ((event.x as f64 + noise_x)
                    .clamp(0.0, self.config.resolution.0 as f64 - 1.0))
                    as u16;
                event.y = ((event.y as f64 + noise_y)
                    .clamp(0.0, self.config.resolution.1 as f64 - 1.0))
                    as u16;
            }
        }

        // Sort events by timestamp again after adding noise
        events.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap());

        Ok(events)
    }

    /// Generate dark current noise events
    fn generate_dark_current_events(&self, timestamp_us: f64) -> CandleResult<Vec<Event>> {
        let mut noise_events = Vec::new();

        if self.config.noise_config.dark_current_rate <= 0.0 {
            return Ok(noise_events);
        }

        let total_pixels = (self.config.resolution.0 * self.config.resolution.1) as f64;
        let dt_seconds = self.config.timestep_us / 1_000_000.0;
        let expected_events =
            total_pixels * self.config.noise_config.dark_current_rate * dt_seconds;

        // Poisson sampling for number of noise events
        let num_noise_events = Self::poisson_sample(expected_events);

        for _ in 0..num_noise_events {
            let x = fastrand::u32(0..self.config.resolution.0) as u16;
            let y = fastrand::u32(0..self.config.resolution.1) as u16;
            let polarity = if fastrand::bool() { 1 } else { -1 };
            let time_offset = fastrand::f64() * self.config.timestep_us;

            noise_events.push(Event {
                t: (timestamp_us + time_offset) / 1_000_000.0,
                x,
                y,
                polarity: polarity > 0,
            });
        }

        Ok(noise_events)
    }

    /// Simple Poisson sampling
    fn poisson_sample(lambda: f64) -> u32 {
        if lambda <= 0.0 {
            return 0;
        }

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

        (k - 1) as u32
    }

    /// Get simulation statistics
    pub fn get_stats(&self) -> SimulationStats {
        SimulationStats {
            frames_processed: self.frame_counter,
            total_pixels: (self.config.resolution.0 * self.config.resolution.1) as u64,
            current_time_us: self.last_timestamp,
            avg_pixel_activity: self.calculate_avg_pixel_activity(),
        }
    }

    /// Calculate average pixel activity
    fn calculate_avg_pixel_activity(&self) -> f64 {
        if self.pixel_states.is_empty() {
            return 0.0;
        }

        let total_activity: f64 = self
            .pixel_states
            .iter()
            .map(|state| state.intensity_buffer.abs())
            .sum();

        total_activity / self.pixel_states.len() as f64
    }

    /// Reset simulation state
    pub fn reset(&mut self) {
        self.last_intensity = None;
        self.last_timestamp = 0.0;
        self.frame_counter = 0;

        for state in &mut self.pixel_states {
            *state = PixelState::new();
        }
    }
}

/// Simulation statistics
#[derive(Debug, Clone)]
pub struct SimulationStats {
    pub frames_processed: u64,
    pub total_pixels: u64,
    pub current_time_us: f64,
    pub avg_pixel_activity: f64,
}

/// High-level simulation interface
pub struct VideoToEventsConverter {
    simulator: EventSimulator,
    video_processor: Option<video_processing::VideoProcessor>,
}

impl VideoToEventsConverter {
    /// Create a new video-to-events converter
    pub fn new(config: SimulationConfig, device: Device) -> CandleResult<Self> {
        let simulator = EventSimulator::new(config, device)?;

        Ok(Self {
            simulator,
            video_processor: None,
        })
    }

    /// Process video file to generate events
    pub fn convert_video_file<P: AsRef<Path>>(
        &mut self,
        video_path: P,
    ) -> CandleResult<Vec<Event>> {
        // Initialize video processor if needed
        if self.video_processor.is_none() {
            self.video_processor = Some(video_processing::VideoProcessor::new()?);
        }

        let processor = self.video_processor.as_mut().unwrap();
        let frames = processor.load_video_frames(video_path)?;

        let mut all_events = Vec::new();

        for (frame_idx, frame) in frames.into_iter().enumerate() {
            let timestamp_us = frame_idx as f64 * self.simulator.config.timestep_us;
            let events = self.simulator.simulate_frame(&frame, timestamp_us)?;
            all_events.extend(events);
        }

        Ok(all_events)
    }

    /// Process single frame to generate events
    pub fn convert_frame(&mut self, frame: &Tensor, timestamp_us: f64) -> CandleResult<Vec<Event>> {
        self.simulator.simulate_frame(frame, timestamp_us)
    }

    /// Get simulation statistics
    pub fn get_stats(&self) -> SimulationStats {
        self.simulator.get_stats()
    }

    /// Reset converter state
    pub fn reset(&mut self) {
        self.simulator.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulation_config_default() {
        let config = SimulationConfig::default();
        assert_eq!(config.resolution, (640, 480));
        assert_eq!(config.contrast_threshold_pos, 0.2);
        assert_eq!(config.contrast_threshold_neg, 0.2);
        assert!(config.enable_noise);
    }

    #[test]
    fn test_noise_config_default() {
        let noise_config = NoiseConfig::default();
        assert_eq!(noise_config.shot_noise_variance, 0.1);
        assert_eq!(noise_config.dark_current_rate, 0.01);
        assert!(!noise_config.temporal_correlation);
    }

    #[test]
    fn test_camera_config_default() {
        let camera_config = CameraConfig::default();
        assert_eq!(camera_config.distortion, [0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(camera_config.intrinsics, [320.0, 320.0, 320.0, 240.0]);
        assert_eq!(camera_config.fps, 30.0);
    }

    #[test]
    fn test_pixel_state_creation() {
        let state = PixelState::new();
        assert_eq!(state.last_event_time, 0.0);
        assert_eq!(state.intensity_buffer, 0.0);
        assert_eq!(state.reference_intensity, 0.0);
    }

    #[test]
    fn test_event_simulator_creation() {
        let config = SimulationConfig::default();
        let device = Device::Cpu;
        let simulator = EventSimulator::new(config.clone(), device).unwrap();

        assert_eq!(simulator.config.resolution, config.resolution);
        assert_eq!(simulator.frame_counter, 0);
        assert_eq!(simulator.pixel_states.len(), (640 * 480) as usize);
    }

    #[test]
    fn test_poisson_sampling() {
        // Test edge cases
        assert_eq!(EventSimulator::poisson_sample(0.0), 0);

        // Test positive lambda
        let sample = EventSimulator::poisson_sample(2.0);
        assert!(sample < 20); // Very unlikely to get >20 with lambda=2
    }
}
