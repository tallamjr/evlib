// Video Writer module for event camera visualization
// Provides efficient video output for rendered event frames

use crate::ev_formats::streaming::Event;
use image::{Rgb, RgbImage};
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum VideoWriterError {
    #[error("Invalid output path: {0}")]
    InvalidPath(String),
    #[error("Unsupported video format: {0}")]
    UnsupportedFormat(String),
    #[error("Failed to create video writer: {0}")]
    CreationFailed(String),
    #[error("Failed to write frame: {0}")]
    WriteFrameFailed(String),
    #[error("Video writer not initialized")]
    NotInitialized,
}

pub type Result<T> = std::result::Result<T, VideoWriterError>;

/// Configuration for video output
#[derive(Debug, Clone)]
pub struct VideoConfig {
    /// Output video width in pixels
    pub width: u32,
    /// Output video height in pixels
    pub height: u32,
    /// Frames per second
    pub fps: f64,
    /// Video codec (e.g., "mp4v", "XVID")
    pub codec: String,
    /// Video quality (0-100, higher is better)
    pub quality: u32,
    /// Whether to output color video
    pub is_color: bool,
}

impl Default for VideoConfig {
    fn default() -> Self {
        Self {
            width: 640,
            height: 480,
            fps: 30.0,
            codec: "mp4v".to_string(),
            quality: 90,
            is_color: true,
        }
    }
}

/// Frame data structure for video writing
#[derive(Debug, Clone)]
pub struct VideoFrame {
    /// Frame data as RGB or grayscale bytes
    pub data: Vec<u8>,
    /// Frame width
    pub width: u32,
    /// Frame height
    pub height: u32,
    /// Number of channels (1 for grayscale, 3 for RGB)
    pub channels: u32,
    /// Frame timestamp in seconds
    pub timestamp: f64,
}

impl VideoFrame {
    /// Create a new RGB video frame
    pub fn new_rgb(width: u32, height: u32, data: Vec<u8>, timestamp: f64) -> Self {
        Self {
            data,
            width,
            height,
            channels: 3,
            timestamp,
        }
    }

    /// Create a new grayscale video frame
    pub fn new_grayscale(width: u32, height: u32, data: Vec<u8>, timestamp: f64) -> Self {
        Self {
            data,
            width,
            height,
            channels: 1,
            timestamp,
        }
    }

    /// Convert from RgbImage
    pub fn from_rgb_image(img: &RgbImage, timestamp: f64) -> Self {
        let width = img.width();
        let height = img.height();
        let mut data = Vec::with_capacity((width * height * 3) as usize);

        for y in 0..height {
            for x in 0..width {
                let pixel = img.get_pixel(x, y);
                data.push(pixel[0]); // R
                data.push(pixel[1]); // G
                data.push(pixel[2]); // B
            }
        }

        Self::new_rgb(width, height, data, timestamp)
    }

    /// Validate frame data consistency
    pub fn is_valid(&self) -> bool {
        let expected_size = (self.width * self.height * self.channels) as usize;
        self.data.len() == expected_size
    }
}

/// High-performance video writer for event visualization
///
/// Note: This is a trait-based approach that can be implemented with different backends
/// For now, we provide a specification that can be used by Python bindings with OpenCV
pub trait VideoWriter {
    /// Initialize the video writer with given configuration
    fn initialize(&mut self, output_path: &Path, config: &VideoConfig) -> Result<()>;

    /// Write a single frame to the video
    fn write_frame(&mut self, frame: &VideoFrame) -> Result<()>;

    /// Finalize and close the video file
    fn finalize(&mut self) -> Result<()>;

    /// Check if the writer is initialized and ready
    fn is_initialized(&self) -> bool;

    /// Get current frame count
    fn frame_count(&self) -> u64;

    /// Get video configuration
    fn config(&self) -> &VideoConfig;
}

/// Statistics for video writing performance
#[derive(Debug, Clone, Default)]
pub struct VideoWriterStats {
    /// Total frames written
    pub frames_written: u64,
    /// Total bytes written
    pub bytes_written: u64,
    /// Duration of video in seconds
    pub duration_seconds: f64,
    /// Average write time per frame in milliseconds
    pub avg_frame_write_time_ms: f64,
    /// Peak memory usage during writing
    pub peak_memory_mb: f64,
}

/// Mock implementation for testing and as a reference
/// Real implementation would use ffmpeg or similar
pub struct MockVideoWriter {
    config: Option<VideoConfig>,
    frame_count: u64,
    stats: VideoWriterStats,
    initialized: bool,
}

impl MockVideoWriter {
    pub fn new() -> Self {
        Self {
            config: None,
            frame_count: 0,
            stats: VideoWriterStats::default(),
            initialized: false,
        }
    }

    pub fn get_stats(&self) -> &VideoWriterStats {
        &self.stats
    }
}

impl VideoWriter for MockVideoWriter {
    fn initialize(&mut self, output_path: &Path, config: &VideoConfig) -> Result<()> {
        // Validate output path
        if let Some(parent) = output_path.parent() {
            if !parent.exists() {
                return Err(VideoWriterError::InvalidPath(format!(
                    "Parent directory does not exist: {}",
                    parent.display()
                )));
            }
        }

        // Validate extension
        let extension = output_path
            .extension()
            .and_then(|ext| ext.to_str())
            .ok_or_else(|| {
                VideoWriterError::UnsupportedFormat("No file extension provided".to_string())
            })?;

        match extension.to_lowercase().as_str() {
            "mp4" | "avi" | "mov" | "mkv" => {}
            _ => {
                return Err(VideoWriterError::UnsupportedFormat(format!(
                    "Unsupported video format: {}",
                    extension
                )))
            }
        }

        self.config = Some(config.clone());
        self.initialized = true;
        self.frame_count = 0;
        self.stats = VideoWriterStats::default();

        Ok(())
    }

    fn write_frame(&mut self, frame: &VideoFrame) -> Result<()> {
        if !self.initialized {
            return Err(VideoWriterError::NotInitialized);
        }

        // Validate frame
        if !frame.is_valid() {
            return Err(VideoWriterError::WriteFrameFailed(
                "Frame data size does not match dimensions".to_string(),
            ));
        }

        let config = self.config.as_ref().unwrap();
        if frame.width != config.width || frame.height != config.height {
            return Err(VideoWriterError::WriteFrameFailed(format!(
                "Frame dimensions {}x{} do not match configured {}x{}",
                frame.width, frame.height, config.width, config.height
            )));
        }

        // Update statistics
        self.frame_count += 1;
        self.stats.frames_written = self.frame_count;
        self.stats.bytes_written += frame.data.len() as u64;
        self.stats.duration_seconds = self.frame_count as f64 / config.fps;

        // Mock: In a real implementation, this would write to the video file
        // For now, just simulate the operation

        Ok(())
    }

    fn finalize(&mut self) -> Result<()> {
        if !self.initialized {
            return Err(VideoWriterError::NotInitialized);
        }

        // Mock finalization
        self.initialized = false;
        Ok(())
    }

    fn is_initialized(&self) -> bool {
        self.initialized
    }

    fn frame_count(&self) -> u64 {
        self.frame_count
    }

    fn config(&self) -> &VideoConfig {
        self.config.as_ref().unwrap()
    }
}

/// Utility functions for video frame processing
pub mod frame_utils {
    use super::*;
    use crate::ev_formats::streaming::Event;

    /// Convert events to a video frame using polarity-based coloring
    pub fn events_to_frame(
        events: &[Event],
        width: u32,
        height: u32,
        timestamp: f64,
        positive_color: [u8; 3],
        negative_color: [u8; 3],
        background_color: [u8; 3],
    ) -> VideoFrame {
        let mut frame_data = vec![0u8; (width * height * 3) as usize];

        // Fill background
        for i in (0..frame_data.len()).step_by(3) {
            frame_data[i] = background_color[0]; // R
            frame_data[i + 1] = background_color[1]; // G
            frame_data[i + 2] = background_color[2]; // B
        }

        // Draw events
        for event in events {
            if event.x as u32 >= width || event.y as u32 >= height {
                continue; // Skip out-of-bounds events
            }

            let pixel_idx = ((event.y as u32 * width + event.x as u32) * 3) as usize;

            if pixel_idx + 2 < frame_data.len() {
                let color = if event.polarity {
                    positive_color
                } else {
                    negative_color
                };

                frame_data[pixel_idx] = color[0]; // R
                frame_data[pixel_idx + 1] = color[1]; // G
                frame_data[pixel_idx + 2] = color[2]; // B
            }
        }

        VideoFrame::new_rgb(width, height, frame_data, timestamp)
    }

    /// Apply temporal decay to a frame
    pub fn apply_decay(frame: &mut VideoFrame, decay_factor: f32) -> Result<()> {
        if decay_factor < 0.0 || decay_factor > 1.0 {
            return Err(VideoWriterError::WriteFrameFailed(
                "Decay factor must be between 0.0 and 1.0".to_string(),
            ));
        }

        for pixel in frame.data.iter_mut() {
            *pixel = (*pixel as f32 * decay_factor) as u8;
        }

        Ok(())
    }

    /// Overlay statistics text on frame (simplified version)
    pub fn overlay_stats(
        frame: &mut VideoFrame,
        stats_text: &[String],
        _color: [u8; 3],
    ) -> Result<()> {
        // This is a simplified implementation
        // A real implementation would need a text rendering library

        // For now, just mark a small region to indicate stats area
        let stats_height = 20 * stats_text.len() as u32;
        let stats_width = 200u32;

        if frame.width < stats_width || frame.height < stats_height {
            return Ok(()); // Skip if frame is too small
        }

        // Draw a simple border for stats area
        for y in 0..stats_height.min(frame.height) {
            for x in 0..stats_width.min(frame.width) {
                if x == 0 || y == 0 || x == stats_width - 1 || y == stats_height - 1 {
                    let pixel_idx = ((y * frame.width + x) * frame.channels) as usize;
                    if pixel_idx + 2 < frame.data.len() {
                        frame.data[pixel_idx] = 128; // Gray border
                        frame.data[pixel_idx + 1] = 128;
                        frame.data[pixel_idx + 2] = 128;
                    }
                }
            }
        }

        Ok(())
    }
}

/// Builder for video writer configuration
pub struct VideoConfigBuilder {
    config: VideoConfig,
}

impl VideoConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: VideoConfig::default(),
        }
    }

    pub fn resolution(mut self, width: u32, height: u32) -> Self {
        self.config.width = width;
        self.config.height = height;
        self
    }

    pub fn fps(mut self, fps: f64) -> Self {
        self.config.fps = fps;
        self
    }

    pub fn codec(mut self, codec: impl Into<String>) -> Self {
        self.config.codec = codec.into();
        self
    }

    pub fn quality(mut self, quality: u32) -> Self {
        self.config.quality = quality.clamp(0, 100);
        self
    }

    pub fn color(mut self, is_color: bool) -> Self {
        self.config.is_color = is_color;
        self
    }

    pub fn build(self) -> VideoConfig {
        self.config
    }
}

impl Default for VideoConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_video_config_builder() {
        let config = VideoConfigBuilder::new()
            .resolution(1920, 1080)
            .fps(60.0)
            .codec("H264")
            .quality(95)
            .color(true)
            .build();

        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.fps, 60.0);
        assert_eq!(config.codec, "H264");
        assert_eq!(config.quality, 95);
        assert!(config.is_color);
    }

    #[test]
    fn test_video_frame_validation() {
        let frame = VideoFrame::new_rgb(640, 480, vec![0u8; 640 * 480 * 3], 0.0);
        assert!(frame.is_valid());

        let invalid_frame = VideoFrame::new_rgb(640, 480, vec![0u8; 100], 0.0);
        assert!(!invalid_frame.is_valid());
    }

    #[test]
    fn test_mock_video_writer() {
        let mut writer = MockVideoWriter::new();
        let config = VideoConfig::default();
        let output_path = PathBuf::from("test.mp4");

        // Test initialization
        assert!(!writer.is_initialized());
        writer.initialize(&output_path, &config).unwrap();
        assert!(writer.is_initialized());

        // Test frame writing
        let frame = VideoFrame::new_rgb(640, 480, vec![0u8; 640 * 480 * 3], 0.0);
        writer.write_frame(&frame).unwrap();
        assert_eq!(writer.frame_count(), 1);

        // Test finalization
        writer.finalize().unwrap();
        assert!(!writer.is_initialized());
    }

    #[test]
    fn test_events_to_frame() {
        use crate::ev_formats::streaming::Event;

        let events = vec![
            Event {
                x: 100,
                y: 200,
                t: 1.0,
                polarity: true,
            },
            Event {
                x: 150,
                y: 250,
                t: 1.0,
                polarity: false,
            },
        ];

        let frame = frame_utils::events_to_frame(
            &events,
            640,
            480,
            1.0,
            [255, 0, 0], // Red for positive
            [0, 0, 255], // Blue for negative
            [0, 0, 0],   // Black background
        );

        assert!(frame.is_valid());
        assert_eq!(frame.width, 640);
        assert_eq!(frame.height, 480);
        assert_eq!(frame.channels, 3);
    }
}
