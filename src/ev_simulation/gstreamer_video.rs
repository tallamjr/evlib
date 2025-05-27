//! GStreamer video processing integration for evlib
//!
//! Provides real video file processing and webcam capture capabilities
//! using the GStreamer multimedia framework.

#[cfg(feature = "gstreamer")]
use gstreamer as gst;
#[cfg(feature = "gstreamer")]
use gstreamer::prelude::*;
#[cfg(feature = "gstreamer")]
use gstreamer_app as gst_app;
#[cfg(feature = "gstreamer")]
use gstreamer_video as gst_video;

use candle_core::{Device, Result as CandleResult, Tensor};
use std::path::Path;
use std::sync::{Arc, Mutex};

/// GStreamer-based video processor
#[cfg(feature = "gstreamer")]
pub struct GstVideoProcessor {
    device: Device,
    pipeline: Option<gst::Pipeline>,
    app_sink: Option<gst_app::AppSink>,
    frame_buffer: Arc<Mutex<Vec<Tensor>>>,
}

#[cfg(feature = "gstreamer")]
impl GstVideoProcessor {
    /// Create new GStreamer video processor
    pub fn new(device: Device) -> CandleResult<Self> {
        // Initialize GStreamer
        gst::init()
            .map_err(|e| candle_core::Error::Msg(format!("GStreamer init failed: {}", e)))?;

        println!("GStreamer video processor initialized");
        println!("  GStreamer version: {}", gst::version_string());

        Ok(Self {
            device,
            pipeline: None,
            app_sink: None,
            frame_buffer: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Load video from file using GStreamer
    pub fn load_video_file<P: AsRef<Path>>(&mut self, video_path: P) -> CandleResult<Vec<Tensor>> {
        let path = video_path.as_ref();
        println!("Loading video with GStreamer: {}", path.display());

        // Create GStreamer pipeline for video file
        let pipeline_desc = format!(
            "filesrc location=\"{}\" ! decodebin ! videoconvert ! video/x-raw,format=RGB ! appsink name=sink",
            path.display()
        );

        self.create_pipeline(&pipeline_desc)?;
        self.run_pipeline()?;

        // Return collected frames
        let frames = self.frame_buffer.lock().unwrap().clone();
        self.cleanup_pipeline();

        println!("Loaded {} frames from video", frames.len());
        Ok(frames)
    }

    /// Start webcam capture
    pub fn start_webcam_capture(&mut self, device_id: u32) -> CandleResult<()> {
        println!("Starting webcam capture (device: {})", device_id);

        // Create GStreamer pipeline for webcam
        let pipeline_desc = if cfg!(target_os = "linux") {
            format!(
                "v4l2src device=/dev/video{} ! videoconvert ! video/x-raw,format=RGB,width=640,height=480 ! appsink name=sink",
                device_id
            )
        } else if cfg!(target_os = "macos") {
            format!(
                "avfvideosrc device-index={} ! videoconvert ! video/x-raw,format=RGB,width=640,height=480 ! appsink name=sink",
                device_id
            )
        } else if cfg!(target_os = "windows") {
            format!(
                "ksvideosrc device-index={} ! videoconvert ! video/x-raw,format=RGB,width=640,height=480 ! appsink name=sink",
                device_id
            )
        } else {
            return Err(candle_core::Error::Msg(
                "Unsupported platform for webcam".to_string(),
            ));
        };

        self.create_pipeline(&pipeline_desc)?;
        self.start_pipeline()?;

        Ok(())
    }

    /// Capture single frame from webcam
    pub fn capture_frame(&mut self) -> CandleResult<Option<Tensor>> {
        if let Some(ref app_sink) = self.app_sink {
            // Try to pull a sample from the sink
            if let Some(sample) = app_sink.try_pull_sample(gst::ClockTime::from_mseconds(100)) {
                let buffer = sample
                    .buffer()
                    .ok_or_else(|| candle_core::Error::Msg("No buffer in sample".to_string()))?;

                let caps = sample
                    .caps()
                    .ok_or_else(|| candle_core::Error::Msg("No caps in sample".to_string()))?;

                let frame_tensor = self.buffer_to_tensor(&buffer.to_owned(), &caps.to_owned())?;
                return Ok(Some(frame_tensor));
            }
        }

        Ok(None)
    }

    /// Stop webcam capture
    pub fn stop_capture(&mut self) -> CandleResult<()> {
        self.stop_pipeline()?;
        self.cleanup_pipeline();
        println!("Webcam capture stopped");
        Ok(())
    }

    /// Create GStreamer pipeline from description
    fn create_pipeline(&mut self, pipeline_desc: &str) -> CandleResult<()> {
        let pipeline = gst::parse::launch(pipeline_desc)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to create pipeline: {}", e)))?
            .downcast::<gst::Pipeline>()
            .map_err(|_| candle_core::Error::Msg("Failed to downcast to pipeline".to_string()))?;

        // Get the appsink element
        let app_sink = pipeline
            .by_name("sink")
            .ok_or_else(|| candle_core::Error::Msg("Could not find appsink".to_string()))?
            .downcast::<gst_app::AppSink>()
            .map_err(|_| candle_core::Error::Msg("Failed to downcast to appsink".to_string()))?;

        // Configure the appsink
        app_sink.set_property("emit-signals", true);
        app_sink.set_property("sync", false);

        self.pipeline = Some(pipeline);
        self.app_sink = Some(app_sink);

        Ok(())
    }

    /// Run pipeline to completion (for video files)
    fn run_pipeline(&mut self) -> CandleResult<()> {
        let pipeline = self
            .pipeline
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("No pipeline created".to_string()))?;

        let app_sink = self
            .app_sink
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("No appsink created".to_string()))?;

        // Set pipeline to playing state
        pipeline
            .set_state(gst::State::Playing)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to start pipeline: {}", e)))?;

        // Collect all frames
        let mut frame_count = 0;
        loop {
            if let Some(sample) = app_sink.try_pull_sample(gst::ClockTime::from_seconds(1)) {
                let buffer = sample
                    .buffer()
                    .ok_or_else(|| candle_core::Error::Msg("No buffer in sample".to_string()))?;

                let caps = sample
                    .caps()
                    .ok_or_else(|| candle_core::Error::Msg("No caps in sample".to_string()))?;

                let frame_tensor = self.buffer_to_tensor(&buffer.to_owned(), &caps.to_owned())?;
                self.frame_buffer.lock().unwrap().push(frame_tensor);

                frame_count += 1;
                if frame_count % 30 == 0 {
                    println!("Processed {} frames", frame_count);
                }
            } else {
                // Check if we've reached end of stream
                let bus = pipeline.bus().unwrap();
                if let Some(msg) =
                    bus.pop_filtered(&[gst::MessageType::Eos, gst::MessageType::Error])
                {
                    match msg.view() {
                        gst::MessageView::Eos(_) => {
                            println!("End of stream reached");
                            break;
                        }
                        gst::MessageView::Error(err) => {
                            return Err(candle_core::Error::Msg(format!(
                                "Pipeline error: {}",
                                err.error()
                            )));
                        }
                        _ => {}
                    }
                }
            }
        }

        Ok(())
    }

    /// Start pipeline for streaming (webcam)
    fn start_pipeline(&mut self) -> CandleResult<()> {
        let pipeline = self
            .pipeline
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("No pipeline created".to_string()))?;

        pipeline
            .set_state(gst::State::Playing)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to start pipeline: {}", e)))?;

        Ok(())
    }

    /// Stop pipeline
    fn stop_pipeline(&mut self) -> CandleResult<()> {
        if let Some(ref pipeline) = self.pipeline {
            pipeline
                .set_state(gst::State::Null)
                .map_err(|e| candle_core::Error::Msg(format!("Failed to stop pipeline: {}", e)))?;
        }
        Ok(())
    }

    /// Clean up pipeline resources
    fn cleanup_pipeline(&mut self) {
        self.pipeline = None;
        self.app_sink = None;
        self.frame_buffer.lock().unwrap().clear();
    }

    /// Convert GStreamer buffer to Candle tensor
    fn buffer_to_tensor(&self, buffer: &gst::Buffer, caps: &gst::Caps) -> CandleResult<Tensor> {
        // Get video info from caps
        let video_info = gst_video::VideoInfo::from_caps(caps)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to get video info: {}", e)))?;

        let width = video_info.width() as usize;
        let height = video_info.height() as usize;

        // Map buffer for reading
        let map = buffer
            .map_readable()
            .map_err(|_| candle_core::Error::Msg("Failed to map buffer".to_string()))?;

        let data = map.as_slice();

        // Convert RGB data to f32 and normalize to [0, 1]
        let mut frame_data = Vec::with_capacity(width * height * 3);
        for &byte in data {
            frame_data.push(byte as f32 / 255.0);
        }

        // Create tensor (H, W, C) format
        let tensor = Tensor::from_vec(frame_data, (height, width, 3), &self.device)?;

        // Convert to grayscale for event simulation (H, W)
        // Use weighted sum: 0.299*R + 0.587*G + 0.114*B
        let r_channel = tensor.narrow(2, 0, 1)?.squeeze(2)?;
        let g_channel = tensor.narrow(2, 1, 1)?.squeeze(2)?;
        let b_channel = tensor.narrow(2, 2, 1)?.squeeze(2)?;

        let grayscale = ((&r_channel * 0.299)? + (&g_channel * 0.587)? + (&b_channel * 0.114)?)?;

        Ok(grayscale)
    }

    /// Get available video devices
    pub fn list_video_devices() -> Vec<String> {
        // This would enumerate available video devices
        // Implementation depends on platform
        #[cfg(target_os = "linux")]
        {
            // Check /dev/video* devices
            let mut devices = Vec::new();
            for i in 0..10 {
                let device_path = format!("/dev/video{}", i);
                if std::path::Path::new(&device_path).exists() {
                    devices.push(format!("Video Device {} ({})", i, device_path));
                }
            }
            devices
        }
        #[cfg(not(target_os = "linux"))]
        {
            vec!["Default Camera (0)".to_string()]
        }
    }
}

/// Fallback video processor when GStreamer is not available
#[cfg(not(feature = "gstreamer"))]
pub struct GstVideoProcessor {
    device: Device,
}

#[cfg(not(feature = "gstreamer"))]
impl GstVideoProcessor {
    pub fn new(device: Device) -> CandleResult<Self> {
        println!("GStreamer support not enabled - using fallback processor");
        Ok(Self { device })
    }

    pub fn load_video_file<P: AsRef<Path>>(&mut self, _video_path: P) -> CandleResult<Vec<Tensor>> {
        Err(candle_core::Error::Msg(
            "GStreamer support not enabled. Please compile with --features gstreamer".to_string(),
        ))
    }

    pub fn start_webcam_capture(&mut self, _device_id: u32) -> CandleResult<()> {
        Err(candle_core::Error::Msg(
            "GStreamer support not enabled. Please compile with --features gstreamer".to_string(),
        ))
    }

    pub fn capture_frame(&mut self) -> CandleResult<Option<Tensor>> {
        Err(candle_core::Error::Msg(
            "GStreamer support not enabled. Please compile with --features gstreamer".to_string(),
        ))
    }

    pub fn stop_capture(&mut self) -> CandleResult<()> {
        Ok(())
    }

    pub fn list_video_devices() -> Vec<String> {
        vec!["GStreamer support not enabled".to_string()]
    }
}

/// High-level video capture interface
pub struct VideoCapture {
    processor: GstVideoProcessor,
    is_capturing: bool,
}

impl VideoCapture {
    /// Create new video capture instance
    pub fn new(device: Device) -> CandleResult<Self> {
        let processor = GstVideoProcessor::new(device)?;

        Ok(Self {
            processor,
            is_capturing: false,
        })
    }

    /// Load video from file
    pub fn load_video<P: AsRef<Path>>(&mut self, video_path: P) -> CandleResult<Vec<Tensor>> {
        self.processor.load_video_file(video_path)
    }

    /// Start capturing from webcam
    pub fn start_webcam(&mut self, device_id: u32) -> CandleResult<()> {
        if self.is_capturing {
            return Err(candle_core::Error::Msg("Already capturing".to_string()));
        }

        self.processor.start_webcam_capture(device_id)?;
        self.is_capturing = true;
        Ok(())
    }

    /// Capture next frame from webcam
    pub fn next_frame(&mut self) -> CandleResult<Option<Tensor>> {
        if !self.is_capturing {
            return Err(candle_core::Error::Msg(
                "Not currently capturing".to_string(),
            ));
        }

        self.processor.capture_frame()
    }

    /// Stop webcam capture
    pub fn stop(&mut self) -> CandleResult<()> {
        if self.is_capturing {
            self.processor.stop_capture()?;
            self.is_capturing = false;
        }
        Ok(())
    }

    /// Check if currently capturing
    pub fn is_capturing(&self) -> bool {
        self.is_capturing
    }

    /// List available video devices
    pub fn list_devices() -> Vec<String> {
        GstVideoProcessor::list_video_devices()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_video_capture_creation() {
        let capture = VideoCapture::new(Device::Cpu);
        assert!(capture.is_ok());

        let capture = capture.unwrap();
        assert!(!capture.is_capturing());
    }

    #[test]
    fn test_list_devices() {
        let devices = VideoCapture::list_devices();
        assert!(!devices.is_empty());

        for device in devices {
            println!("Available device: {}", device);
        }
    }

    #[cfg(feature = "gstreamer")]
    #[test]
    fn test_gstreamer_initialization() {
        let processor = GstVideoProcessor::new(Device::Cpu);
        assert!(processor.is_ok());
    }

    #[cfg(not(feature = "gstreamer"))]
    #[test]
    fn test_fallback_processor() {
        let mut processor = GstVideoProcessor::new(Device::Cpu).unwrap();

        // Should return error for video loading without GStreamer
        let result = processor.load_video_file("test.mp4");
        assert!(result.is_err());

        // Should return error for webcam without GStreamer
        let result = processor.start_webcam_capture(0);
        assert!(result.is_err());
    }
}
