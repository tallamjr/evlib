//! Real-time event visualization with minimal Python overhead
//!
//! This module provides high-performance event visualization by processing
//! and rendering events directly in Rust, only passing the final image to Python.

// Removed: use crate::{Event, Events}; - legacy types no longer exist
use image::{ImageBuffer, Rgb};
use std::collections::VecDeque;
use std::time::Instant;

#[cfg(feature = "polars")]
use polars::prelude::*;

/// Configuration for real-time visualization
#[derive(Debug, Clone)]
pub struct RealtimeVisualizationConfig {
    /// Target display resolution
    pub display_width: u32,
    pub display_height: u32,
    /// Event decay time in milliseconds
    pub event_decay_ms: f32,
    /// Maximum events to visualize
    pub max_events: usize,
    /// Show FPS counter
    pub show_fps: bool,
    /// Background color (RGB)
    pub background_color: [u8; 3],
    /// Positive event color (RGB)
    pub positive_color: [u8; 3],
    /// Negative event color (RGB)
    pub negative_color: [u8; 3],
}

impl Default for RealtimeVisualizationConfig {
    fn default() -> Self {
        Self {
            display_width: 640,
            display_height: 480,
            event_decay_ms: 50.0,
            max_events: 5000,
            show_fps: true,
            background_color: [255, 255, 255],
            positive_color: [255, 0, 0],
            negative_color: [0, 0, 255],
        }
    }
}

/// Real-time event visualizer
pub struct RealtimeEventVisualizer {
    pub config: RealtimeVisualizationConfig,
    /// Event buffer with timestamps
    event_buffer: VecDeque<(Event, Instant)>,
    /// Current frame buffer
    frame_buffer: ImageBuffer<Rgb<u8>, Vec<u8>>,
    /// FPS tracking
    last_frame_time: Instant,
    fps_history: VecDeque<f32>,
    current_fps: f32,
}

impl RealtimeEventVisualizer {
    /// Create a new real-time visualizer
    pub fn new(config: RealtimeVisualizationConfig) -> Self {
        let frame_buffer = ImageBuffer::from_pixel(
            config.display_width,
            config.display_height,
            Rgb(config.background_color),
        );

        Self {
            config,
            event_buffer: VecDeque::with_capacity(10000),
            frame_buffer,
            last_frame_time: Instant::now(),
            fps_history: VecDeque::with_capacity(30),
            current_fps: 0.0,
        }
    }

    /// Add events to the visualization buffer
    pub fn add_events(&mut self, events: Events) {
        self.add_events_impl(events);
    }

    /// Add events from DataFrame to the visualization buffer
    #[cfg(feature = "polars")]
    pub fn add_events_from_dataframe(&mut self, df: LazyFrame) -> Result<(), PolarsError> {
        let events = dataframe_to_events_for_visualization(df)?;
        self.add_events_impl(events);
        Ok(())
    }

    /// Internal implementation for adding events
    fn add_events_impl(&mut self, events: Events) {
        let now = Instant::now();

        // Add new events
        for event in events {
            if self.event_buffer.len() >= self.config.max_events {
                self.event_buffer.pop_front();
            }
            self.event_buffer.push_back((event, now));
        }

        // Remove old events
        let decay_duration = std::time::Duration::from_millis(self.config.event_decay_ms as u64);
        while let Some((_, timestamp)) = self.event_buffer.front() {
            if now.duration_since(*timestamp) > decay_duration {
                self.event_buffer.pop_front();
            } else {
                break;
            }
        }
    }

    /// Render current frame
    pub fn render_frame(&mut self) -> &[u8] {
        let now = Instant::now();

        // Clear frame
        for pixel in self.frame_buffer.pixels_mut() {
            *pixel = Rgb(self.config.background_color);
        }

        // Draw events with decay
        let decay_ms = self.config.event_decay_ms;
        for (event, timestamp) in &self.event_buffer {
            let age_ms = now.duration_since(*timestamp).as_millis() as f32;
            if age_ms < decay_ms {
                let alpha = 1.0 - (age_ms / decay_ms);
                let color = if event.polarity {
                    self.config.positive_color
                } else {
                    self.config.negative_color
                };

                // Draw event with alpha blending
                let x = event.x as u32;
                let y = event.y as u32;
                if x < self.config.display_width && y < self.config.display_height {
                    let pixel = self.frame_buffer.get_pixel_mut(x, y);
                    for i in 0..3 {
                        let bg = self.config.background_color[i] as f32;
                        let fg = color[i] as f32;
                        pixel[i] = (bg * (1.0 - alpha) + fg * alpha) as u8;
                    }
                }
            }
        }

        // Update FPS
        let frame_time = now.duration_since(self.last_frame_time).as_secs_f32();
        self.last_frame_time = now;
        let current_fps = 1.0 / frame_time;
        self.fps_history.push_back(current_fps);
        if self.fps_history.len() > 30 {
            self.fps_history.pop_front();
        }
        self.current_fps = self.fps_history.iter().sum::<f32>() / self.fps_history.len() as f32;

        // Draw FPS counter if enabled
        if self.config.show_fps {
            // Simple FPS text rendering (placeholder - would need proper text rendering)
            // For now, we'll just return the FPS value separately
        }

        self.frame_buffer.as_raw()
    }

    /// Get current FPS
    pub fn get_fps(&self) -> f32 {
        self.current_fps
    }

    /// Clear all events
    pub fn clear(&mut self) {
        self.event_buffer.clear();
    }

    /// Get frame dimensions
    pub fn get_frame_dimensions(&self) -> (u32, u32) {
        (self.config.display_width, self.config.display_height)
    }
}

/// Create an optimized event visualization pipeline
pub struct EventVisualizationPipeline {
    pub visualizer: RealtimeEventVisualizer,
    frame_count: u64,
    total_events: u64,
}

impl EventVisualizationPipeline {
    pub fn new(config: RealtimeVisualizationConfig) -> Self {
        Self {
            visualizer: RealtimeEventVisualizer::new(config),
            frame_count: 0,
            total_events: 0,
        }
    }

    /// Process and visualize events, returning raw RGB frame data
    pub fn process_events(&mut self, events: Events) -> (Vec<u8>, f32, (u32, u32)) {
        self.process_events_impl(events)
    }

    /// Process and visualize events from DataFrame, returning raw RGB frame data
    #[cfg(feature = "polars")]
    pub fn process_events_from_dataframe(
        &mut self,
        df: LazyFrame,
    ) -> Result<(Vec<u8>, f32, (u32, u32)), PolarsError> {
        let events = dataframe_to_events_for_visualization(df)?;
        Ok(self.process_events_impl(events))
    }

    /// Internal implementation for processing events
    fn process_events_impl(&mut self, events: Events) -> (Vec<u8>, f32, (u32, u32)) {
        self.total_events += events.len() as u64;
        self.visualizer.add_events(events);
        self.frame_count += 1;

        let frame_data = self.visualizer.render_frame().to_vec();
        let fps = self.visualizer.get_fps();
        let dimensions = self.visualizer.get_frame_dimensions();

        (frame_data, fps, dimensions)
    }

    /// Get statistics
    pub fn get_stats(&self) -> (u64, u64, f32) {
        (
            self.frame_count,
            self.total_events,
            self.visualizer.get_fps(),
        )
    }

    /// Reset the pipeline
    pub fn reset(&mut self) {
        self.visualizer.clear();
        self.frame_count = 0;
        self.total_events = 0;
    }
}

/// Helper function to convert DataFrame back to Events for visualization
#[cfg(feature = "polars")]
fn dataframe_to_events_for_visualization(df: LazyFrame) -> Result<Events, PolarsError> {
    let df = df.collect()?;

    let x_series = df.column("x")?;
    let y_series = df.column("y")?;
    let t_series = df.column("t")?;
    let polarity_series = df.column("polarity")?;

    let x_values = x_series.i64()?.into_no_null_iter().collect::<Vec<_>>();
    let y_values = y_series.i64()?.into_no_null_iter().collect::<Vec<_>>();
    let t_values = t_series.f64()?.into_no_null_iter().collect::<Vec<_>>();
    let polarity_values = polarity_series
        .i64()?
        .into_no_null_iter()
        .collect::<Vec<_>>();

    let events = x_values
        .into_iter()
        .zip(y_values)
        .zip(t_values)
        .zip(polarity_values)
        .map(|(((x, y), t), p)| Event {
            x: x as u16,
            y: y as u16,
            t,
            polarity: p > 0,
        })
        .collect();

    Ok(events)
}
