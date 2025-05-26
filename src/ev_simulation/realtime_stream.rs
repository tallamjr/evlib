//! Real-time event stream processing for webcam input
//!
//! Combines GStreamer video capture with ESIM event simulation
//! to create real-time event streams from live video input.

use super::esim::{EsimConfig, EsimConverter};
use super::gstreamer_video::VideoCapture;
use crate::ev_core::Event;
use candle_core::{Device, Result as CandleResult, Tensor};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Configuration for real-time event streaming
#[derive(Debug, Clone)]
pub struct RealtimeStreamConfig {
    /// ESIM simulation configuration
    pub esim_config: EsimConfig,
    /// Target frames per second for processing
    pub target_fps: f64,
    /// Maximum event buffer size
    pub max_buffer_size: usize,
    /// Enable automatic frame rate adjustment
    pub auto_adjust_fps: bool,
    /// Video device ID (0 = default camera)
    pub device_id: u32,
    /// Processing timeout in milliseconds
    pub processing_timeout_ms: u64,
}

impl Default for RealtimeStreamConfig {
    fn default() -> Self {
        let mut esim_config = EsimConfig::default();
        // Optimise for real-time performance
        esim_config.base_config.contrast_threshold_pos = 0.15;
        esim_config.base_config.contrast_threshold_neg = 0.15;
        esim_config.use_bilinear_interpolation = false; // Disable for speed
        esim_config.adaptive_thresholding = false;

        Self {
            esim_config,
            target_fps: 30.0,
            max_buffer_size: 10000,
            auto_adjust_fps: true,
            device_id: 0,
            processing_timeout_ms: 100,
        }
    }
}

/// Statistics for real-time streaming performance
#[derive(Debug, Clone)]
pub struct StreamingStats {
    /// Total frames processed
    pub frames_processed: u64,
    /// Total events generated
    pub events_generated: u64,
    /// Current processing FPS
    pub current_fps: f64,
    /// Average events per frame
    pub avg_events_per_frame: f64,
    /// Current event buffer size
    pub buffer_size: usize,
    /// Total dropped frames
    pub dropped_frames: u64,
    /// Processing latency in milliseconds
    pub avg_latency_ms: f64,
}

impl Default for StreamingStats {
    fn default() -> Self {
        Self {
            frames_processed: 0,
            events_generated: 0,
            current_fps: 0.0,
            avg_events_per_frame: 0.0,
            buffer_size: 0,
            dropped_frames: 0,
            avg_latency_ms: 0.0,
        }
    }
}

/// Real-time event stream processor
pub struct RealtimeEventStream {
    config: RealtimeStreamConfig,
    device: Device,
    video_capture: VideoCapture,
    esim_converter: EsimConverter,

    // Streaming state
    is_streaming: bool,
    event_buffer: Arc<Mutex<VecDeque<Event>>>,
    stats: Arc<Mutex<StreamingStats>>,

    // Performance tracking
    last_frame_time: Instant,
    frame_times: VecDeque<Duration>,
    target_frame_duration: Duration,
}

impl RealtimeEventStream {
    /// Create new real-time event stream
    pub fn new(config: RealtimeStreamConfig, device: Device) -> CandleResult<Self> {
        println!("Initialising real-time event stream...");

        // Create video capture
        let video_capture = VideoCapture::new(device.clone())?;

        // Create ESIM converter
        let esim_converter = EsimConverter::new(config.esim_config.clone(), device.clone())?;

        let target_frame_duration = Duration::from_secs_f64(1.0 / config.target_fps);

        println!("Real-time event stream initialised:");
        println!("  Target FPS: {:.1}", config.target_fps);
        println!(
            "  Device: {}",
            match device {
                Device::Cpu => "CPU",
                Device::Cuda(_) => "CUDA",
                Device::Metal(_) => "Metal",
            }
        );
        println!("  Max buffer size: {}", config.max_buffer_size);

        Ok(Self {
            config,
            device,
            video_capture,
            esim_converter,
            is_streaming: false,
            event_buffer: Arc::new(Mutex::new(VecDeque::new())),
            stats: Arc::new(Mutex::new(StreamingStats::default())),
            last_frame_time: Instant::now(),
            frame_times: VecDeque::new(),
            target_frame_duration,
        })
    }

    /// Start real-time streaming from webcam
    pub fn start_streaming(&mut self) -> CandleResult<()> {
        if self.is_streaming {
            return Err(candle_core::Error::Msg("Already streaming".to_string()));
        }

        println!("Starting webcam capture...");
        self.video_capture.start_webcam(self.config.device_id)?;

        self.is_streaming = true;
        self.last_frame_time = Instant::now();

        // Reset statistics
        {
            let mut stats = self.stats.lock().unwrap();
            *stats = StreamingStats::default();
        }

        println!("Real-time streaming started");
        Ok(())
    }

    /// Process next frame and generate events
    pub fn process_next_frame(&mut self) -> CandleResult<bool> {
        if !self.is_streaming {
            return Err(candle_core::Error::Msg(
                "Not currently streaming".to_string(),
            ));
        }

        let frame_start = Instant::now();

        // Try to capture frame with timeout
        match self.video_capture.next_frame()? {
            Some(frame) => {
                // Generate timestamp
                let timestamp_us =
                    frame_start.duration_since(self.last_frame_time).as_micros() as f64;

                // Convert frame to events using ESIM
                let events = self.esim_converter.convert_frame(&frame, timestamp_us)?;

                // Add events to buffer
                self.add_events_to_buffer(events)?;

                // Update performance stats
                self.update_stats(frame_start);

                // Check if we need to adjust frame rate
                if self.config.auto_adjust_fps {
                    self.adjust_frame_rate();
                }

                Ok(true)
            }
            None => {
                // No frame available, increment dropped frames
                {
                    let mut stats = self.stats.lock().unwrap();
                    stats.dropped_frames += 1;
                }
                Ok(false)
            }
        }
    }

    /// Get events from buffer (non-blocking)
    pub fn get_events(&self, max_count: Option<usize>) -> Vec<Event> {
        let mut buffer = self.event_buffer.lock().unwrap();
        let count = max_count.unwrap_or(buffer.len()).min(buffer.len());

        let mut events = Vec::with_capacity(count);
        for _ in 0..count {
            if let Some(event) = buffer.pop_front() {
                events.push(event);
            }
        }

        // Update buffer size in stats
        {
            let mut stats = self.stats.lock().unwrap();
            stats.buffer_size = buffer.len();
        }

        events
    }

    /// Get all events from buffer (blocking until events available)
    pub fn get_events_blocking(&self, timeout_ms: Option<u64>) -> Vec<Event> {
        let timeout =
            Duration::from_millis(timeout_ms.unwrap_or(self.config.processing_timeout_ms));
        let start_time = Instant::now();

        loop {
            let events = self.get_events(None);
            if !events.is_empty() {
                return events;
            }

            if start_time.elapsed() > timeout {
                break;
            }

            thread::sleep(Duration::from_millis(1));
        }

        Vec::new()
    }

    /// Stop streaming
    pub fn stop_streaming(&mut self) -> CandleResult<()> {
        if !self.is_streaming {
            return Ok(());
        }

        println!("Stopping real-time streaming...");
        self.video_capture.stop()?;
        self.is_streaming = false;

        // Clear event buffer
        self.event_buffer.lock().unwrap().clear();

        let final_stats = self.get_stats();
        println!("Streaming session completed:");
        println!("  Frames processed: {}", final_stats.frames_processed);
        println!("  Events generated: {}", final_stats.events_generated);
        println!("  Average FPS: {:.1}", final_stats.current_fps);
        println!(
            "  Average events/frame: {:.1}",
            final_stats.avg_events_per_frame
        );
        println!("  Dropped frames: {}", final_stats.dropped_frames);

        Ok(())
    }

    /// Check if currently streaming
    pub fn is_streaming(&self) -> bool {
        self.is_streaming
    }

    /// Get current streaming statistics
    pub fn get_stats(&self) -> StreamingStats {
        self.stats.lock().unwrap().clone()
    }

    /// Add events to buffer with overflow protection
    fn add_events_to_buffer(&self, events: Vec<Event>) -> CandleResult<()> {
        let mut buffer = self.event_buffer.lock().unwrap();

        // Check for buffer overflow
        if buffer.len() + events.len() > self.config.max_buffer_size {
            let overflow = (buffer.len() + events.len()) - self.config.max_buffer_size;
            // Remove oldest events to make space
            for _ in 0..overflow {
                buffer.pop_front();
            }
        }

        // Add new events
        for event in events {
            buffer.push_back(event);
        }

        Ok(())
    }

    /// Update performance statistics
    fn update_stats(&mut self, frame_start: Instant) {
        let processing_time = frame_start.elapsed();
        self.frame_times.push_back(processing_time);

        // Keep only recent frame times (last 30 frames)
        if self.frame_times.len() > 30 {
            self.frame_times.pop_front();
        }

        let mut stats = self.stats.lock().unwrap();
        stats.frames_processed += 1;

        // Calculate current FPS
        if self.frame_times.len() > 1 {
            let total_time: Duration = self.frame_times.iter().sum();
            stats.current_fps = self.frame_times.len() as f64 / total_time.as_secs_f64();
        }

        // Update average latency
        stats.avg_latency_ms = processing_time.as_millis() as f64;

        // Update events per frame
        if stats.frames_processed > 0 {
            stats.avg_events_per_frame =
                stats.events_generated as f64 / stats.frames_processed as f64;
        }
    }

    /// Automatically adjust frame rate based on performance
    fn adjust_frame_rate(&mut self) {
        if self.frame_times.len() < 10 {
            return; // Need more samples
        }

        let avg_processing_time: Duration =
            self.frame_times.iter().sum::<Duration>() / self.frame_times.len() as u32;

        // If processing is taking longer than target, reduce FPS
        if avg_processing_time > self.target_frame_duration {
            let new_fps = (self.config.target_fps * 0.9).max(5.0);
            if (new_fps - self.config.target_fps).abs() > 0.1 {
                println!(
                    "Adjusting target FPS: {:.1} -> {:.1}",
                    self.config.target_fps, new_fps
                );
                self.config.target_fps = new_fps;
                self.target_frame_duration = Duration::from_secs_f64(1.0 / new_fps);
            }
        }
        // If processing is fast enough, try to increase FPS
        else if avg_processing_time < self.target_frame_duration / 2 {
            let new_fps = (self.config.target_fps * 1.1).min(60.0);
            if (new_fps - self.config.target_fps).abs() > 0.1 {
                println!(
                    "Adjusting target FPS: {:.1} -> {:.1}",
                    self.config.target_fps, new_fps
                );
                self.config.target_fps = new_fps;
                self.target_frame_duration = Duration::from_secs_f64(1.0 / new_fps);
            }
        }
    }

    /// Reset stream state
    pub fn reset(&mut self) {
        self.esim_converter.reset();
        self.event_buffer.lock().unwrap().clear();
        self.frame_times.clear();

        let mut stats = self.stats.lock().unwrap();
        *stats = StreamingStats::default();
    }

    /// Get current configuration
    pub fn get_config(&self) -> &RealtimeStreamConfig {
        &self.config
    }

    /// Update ESIM parameters during streaming
    pub fn update_esim_params(
        &mut self,
        contrast_threshold: Option<f64>,
        target_fps: Option<f64>,
    ) -> CandleResult<()> {
        if let Some(threshold) = contrast_threshold {
            println!("Updating contrast threshold: {:.3}", threshold);
            // Note: In a full implementation, we'd update the ESIM converter parameters
            // For now, we store the new values for future frames
            self.config.esim_config.base_config.contrast_threshold_pos = threshold;
            self.config.esim_config.base_config.contrast_threshold_neg = threshold;
        }

        if let Some(fps) = target_fps {
            println!("Updating target FPS: {:.1}", fps);
            self.config.target_fps = fps;
            self.target_frame_duration = Duration::from_secs_f64(1.0 / fps);
        }

        Ok(())
    }
}

/// High-level interface for real-time event streaming
pub struct EventStreamManager {
    stream: RealtimeEventStream,
    processing_thread: Option<thread::JoinHandle<()>>,
    should_stop: Arc<Mutex<bool>>,
}

impl EventStreamManager {
    /// Create new event stream manager
    pub fn new(config: RealtimeStreamConfig, device: Device) -> CandleResult<Self> {
        let stream = RealtimeEventStream::new(config, device)?;

        Ok(Self {
            stream,
            processing_thread: None,
            should_stop: Arc::new(Mutex::new(false)),
        })
    }

    /// Start background processing thread
    pub fn start_background_processing(&mut self) -> CandleResult<()> {
        if self.processing_thread.is_some() {
            return Err(candle_core::Error::Msg(
                "Background processing already running".to_string(),
            ));
        }

        self.stream.start_streaming()?;

        let should_stop = Arc::clone(&self.should_stop);
        *should_stop.lock().unwrap() = false;

        println!("Starting background processing thread...");

        // Note: In a complete implementation, we'd pass the stream to the thread
        // For now, we'll just create a placeholder thread
        let handle = thread::spawn(move || {
            let mut frame_count = 0;
            while !*should_stop.lock().unwrap() {
                // Simulate processing
                thread::sleep(Duration::from_millis(33)); // ~30 FPS
                frame_count += 1;

                if frame_count % 100 == 0 {
                    println!("Background thread processed {} frames", frame_count);
                }
            }
            println!("Background processing thread stopped");
        });

        self.processing_thread = Some(handle);
        Ok(())
    }

    /// Stop background processing
    pub fn stop_background_processing(&mut self) -> CandleResult<()> {
        if let Some(handle) = self.processing_thread.take() {
            *self.should_stop.lock().unwrap() = true;
            handle
                .join()
                .map_err(|_| candle_core::Error::Msg("Failed to join thread".to_string()))?;
        }

        self.stream.stop_streaming()?;
        Ok(())
    }

    /// Get events from stream
    pub fn get_events(&self, max_count: Option<usize>) -> Vec<Event> {
        self.stream.get_events(max_count)
    }

    /// Get streaming statistics
    pub fn get_stats(&self) -> StreamingStats {
        self.stream.get_stats()
    }

    /// Update parameters
    pub fn update_params(
        &mut self,
        contrast_threshold: Option<f64>,
        target_fps: Option<f64>,
    ) -> CandleResult<()> {
        self.stream
            .update_esim_params(contrast_threshold, target_fps)
    }
}

impl Drop for EventStreamManager {
    fn drop(&mut self) {
        let _ = self.stop_background_processing();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_realtime_config_default() {
        let config = RealtimeStreamConfig::default();
        assert_eq!(config.target_fps, 30.0);
        assert_eq!(config.max_buffer_size, 10000);
        assert!(config.auto_adjust_fps);
        assert_eq!(config.device_id, 0);
        assert!(!config.esim_config.use_bilinear_interpolation); // Optimised for performance
    }

    #[test]
    fn test_realtime_stream_creation() {
        let config = RealtimeStreamConfig::default();
        let device = Device::Cpu;
        let stream = RealtimeEventStream::new(config, device);

        assert!(stream.is_ok());
        let stream = stream.unwrap();
        assert!(!stream.is_streaming());
    }

    #[test]
    fn test_streaming_stats_default() {
        let stats = StreamingStats::default();
        assert_eq!(stats.frames_processed, 0);
        assert_eq!(stats.events_generated, 0);
        assert_eq!(stats.current_fps, 0.0);
        assert_eq!(stats.buffer_size, 0);
    }

    #[test]
    fn test_event_stream_manager_creation() {
        let config = RealtimeStreamConfig::default();
        let device = Device::Cpu;
        let manager = EventStreamManager::new(config, device);

        assert!(manager.is_ok());
    }

    #[test]
    fn test_get_empty_events() {
        let config = RealtimeStreamConfig::default();
        let device = Device::Cpu;
        let stream = RealtimeEventStream::new(config, device).unwrap();

        let events = stream.get_events(Some(10));
        assert!(events.is_empty());
    }
}
