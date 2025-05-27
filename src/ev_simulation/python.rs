//! Python bindings for event simulation (Video-to-Events)

#[cfg(feature = "gstreamer")]
use super::esim::EsimConfig;
#[cfg(feature = "gstreamer")]
use super::realtime_stream::{RealtimeEventStream, RealtimeStreamConfig, StreamingStats};
use super::{SimulationConfig, SimulationStats, VideoToEventsConverter};
use crate::ev_core::{Event, DEVICE};
use candle_core::Tensor;
use numpy::{IntoPyArray, PyReadonlyArray2};
use pyo3::prelude::*;
use std::path::PathBuf;

/// Python wrapper for SimulationConfig
#[pyclass]
#[derive(Clone)]
pub struct PySimulationConfig {
    pub inner: SimulationConfig,
}

#[pymethods]
impl PySimulationConfig {
    #[new]
    #[pyo3(signature = (
        resolution = None,
        contrast_threshold_pos = None,
        contrast_threshold_neg = None,
        refractory_period_us = None,
        timestep_us = None,
        enable_noise = None
    ))]
    pub fn new(
        resolution: Option<(u32, u32)>,
        contrast_threshold_pos: Option<f64>,
        contrast_threshold_neg: Option<f64>,
        refractory_period_us: Option<f64>,
        timestep_us: Option<f64>,
        enable_noise: Option<bool>,
    ) -> Self {
        let mut config = SimulationConfig::default();

        if let Some(res) = resolution {
            config.resolution = res;
        }
        if let Some(pos_thresh) = contrast_threshold_pos {
            config.contrast_threshold_pos = pos_thresh;
        }
        if let Some(neg_thresh) = contrast_threshold_neg {
            config.contrast_threshold_neg = neg_thresh;
        }
        if let Some(refrac) = refractory_period_us {
            config.refractory_period_us = refrac;
        }
        if let Some(timestep) = timestep_us {
            config.timestep_us = timestep;
        }
        if let Some(noise) = enable_noise {
            config.enable_noise = noise;
        }

        Self { inner: config }
    }

    #[getter]
    pub fn resolution(&self) -> (u32, u32) {
        self.inner.resolution
    }

    #[setter]
    pub fn set_resolution(&mut self, resolution: (u32, u32)) {
        self.inner.resolution = resolution;
    }

    #[getter]
    pub fn contrast_threshold_pos(&self) -> f64 {
        self.inner.contrast_threshold_pos
    }

    #[setter]
    pub fn set_contrast_threshold_pos(&mut self, threshold: f64) {
        self.inner.contrast_threshold_pos = threshold;
    }

    #[getter]
    pub fn contrast_threshold_neg(&self) -> f64 {
        self.inner.contrast_threshold_neg
    }

    #[setter]
    pub fn set_contrast_threshold_neg(&mut self, threshold: f64) {
        self.inner.contrast_threshold_neg = threshold;
    }

    #[getter]
    pub fn enable_noise(&self) -> bool {
        self.inner.enable_noise
    }

    #[setter]
    pub fn set_enable_noise(&mut self, enable: bool) {
        self.inner.enable_noise = enable;
    }

    pub fn __repr__(&self) -> String {
        format!(
            "SimulationConfig(resolution={:?}, thresholds=({:.2}, {:.2}), noise={})",
            self.inner.resolution,
            self.inner.contrast_threshold_pos,
            self.inner.contrast_threshold_neg,
            self.inner.enable_noise
        )
    }
}

/// Python wrapper for VideoToEventsConverter
#[pyclass]
pub struct PyVideoToEventsConverter {
    converter: VideoToEventsConverter,
}

#[pymethods]
impl PyVideoToEventsConverter {
    #[new]
    pub fn new(config: &PySimulationConfig) -> PyResult<Self> {
        let converter = VideoToEventsConverter::new(config.inner.clone(), DEVICE.clone())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;

        Ok(Self { converter })
    }

    /// Convert a single frame to events
    #[pyo3(signature = (frame, timestamp_us = 0.0))]
    pub fn convert_frame(
        &mut self,
        py: Python<'_>,
        frame: PyReadonlyArray2<f32>,
        timestamp_us: f64,
    ) -> PyResult<(PyObject, PyObject, PyObject, PyObject)> {
        // Convert numpy array to Tensor
        let frame_array = frame.as_array();
        let frame_data: Vec<f32> = frame_array.iter().cloned().collect();
        let shape = frame_array.shape();
        let height = shape[0];
        let width = shape[1];

        let tensor = Tensor::from_vec(frame_data, (height, width), &DEVICE.clone())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;

        // Convert frame to events
        let events = self
            .converter
            .convert_frame(&tensor, timestamp_us)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;

        // Convert events to numpy arrays
        let (xs, ys, ts, ps) = events_to_arrays(events);

        Ok((
            xs.into_pyarray(py).to_object(py),
            ys.into_pyarray(py).to_object(py),
            ts.into_pyarray(py).to_object(py),
            ps.into_pyarray(py).to_object(py),
        ))
    }

    /// Convert video file to events
    pub fn convert_video_file(
        &mut self,
        py: Python<'_>,
        video_path: &str,
    ) -> PyResult<(PyObject, PyObject, PyObject, PyObject)> {
        let events = self
            .converter
            .convert_video_file(PathBuf::from(video_path))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;

        // Convert events to numpy arrays
        let (xs, ys, ts, ps) = events_to_arrays(events);

        Ok((
            xs.into_pyarray(py).to_object(py),
            ys.into_pyarray(py).to_object(py),
            ts.into_pyarray(py).to_object(py),
            ps.into_pyarray(py).to_object(py),
        ))
    }

    /// Get simulation statistics
    pub fn get_stats(&self) -> PySimulationStats {
        PySimulationStats {
            inner: self.converter.get_stats(),
        }
    }

    /// Reset simulation state
    pub fn reset(&mut self) {
        self.converter.reset();
    }
}

/// Python wrapper for SimulationStats
#[pyclass]
pub struct PySimulationStats {
    inner: SimulationStats,
}

#[pymethods]
impl PySimulationStats {
    #[getter]
    pub fn frames_processed(&self) -> u64 {
        self.inner.frames_processed
    }

    #[getter]
    pub fn total_pixels(&self) -> u64 {
        self.inner.total_pixels
    }

    #[getter]
    pub fn current_time_us(&self) -> f64 {
        self.inner.current_time_us
    }

    #[getter]
    pub fn avg_pixel_activity(&self) -> f64 {
        self.inner.avg_pixel_activity
    }

    pub fn __repr__(&self) -> String {
        format!(
            "SimulationStats(frames={}, pixels={}, time={:.1}μs, activity={:.3})",
            self.inner.frames_processed,
            self.inner.total_pixels,
            self.inner.current_time_us,
            self.inner.avg_pixel_activity
        )
    }
}

/// High-level function to convert video frames to events
#[pyfunction]
#[pyo3(signature = (
    frames,
    contrast_threshold_pos = 0.2,
    contrast_threshold_neg = 0.2,
    refractory_period_us = 100.0,
    timestep_us = 1000.0,
    enable_noise = true
))]
pub fn video_to_events_py(
    py: Python<'_>,
    frames: PyReadonlyArray2<f32>,
    contrast_threshold_pos: f64,
    contrast_threshold_neg: f64,
    refractory_period_us: f64,
    timestep_us: f64,
    enable_noise: bool,
) -> PyResult<(PyObject, PyObject, PyObject, PyObject)> {
    // Create simulation config
    let frame_array = frames.as_array();
    let shape = frame_array.shape();
    let height = shape[0] as u32;
    let width = shape[1] as u32;

    let config = SimulationConfig {
        resolution: (width, height),
        contrast_threshold_pos,
        contrast_threshold_neg,
        refractory_period_us,
        timestep_us,
        enable_noise,
        ..Default::default()
    };

    // Create converter
    let mut converter = VideoToEventsConverter::new(config, DEVICE.clone())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;

    // Convert frame data to tensor
    let frame_data: Vec<f32> = frame_array.iter().cloned().collect();
    let tensor = Tensor::from_vec(
        frame_data,
        (height as usize, width as usize),
        &DEVICE.clone(),
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;

    // Generate events
    let events = converter
        .convert_frame(&tensor, 0.0)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;

    // Convert to numpy arrays
    let (xs, ys, ts, ps) = events_to_arrays(events);

    Ok((
        xs.into_pyarray(py).to_object(py),
        ys.into_pyarray(py).to_object(py),
        ts.into_pyarray(py).to_object(py),
        ps.into_pyarray(py).to_object(py),
    ))
}

/// Convert Vec<Event> to separate arrays
fn events_to_arrays(events: Vec<Event>) -> (Vec<i64>, Vec<i64>, Vec<f64>, Vec<i64>) {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut ts = Vec::new();
    let mut ps = Vec::new();

    for event in events {
        xs.push(event.x as i64);
        ys.push(event.y as i64);
        ts.push(event.t);
        ps.push(event.polarity as i64);
    }

    (xs, ys, ts, ps)
}

/// ESIM-style simulation from intensity frames
#[pyfunction]
#[pyo3(signature = (
    intensity_old,
    intensity_new,
    threshold = 0.2,
    _refractory_period_us = 100.0
))]
pub fn esim_simulate_py(
    py: Python<'_>,
    intensity_old: PyReadonlyArray2<f32>,
    intensity_new: PyReadonlyArray2<f32>,
    threshold: f64,
    _refractory_period_us: f64,
) -> PyResult<(PyObject, PyObject, PyObject, PyObject)> {
    let old_array = intensity_old.as_array();
    let new_array = intensity_new.as_array();

    if old_array.shape() != new_array.shape() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Frame shapes must match",
        ));
    }

    let shape = old_array.shape();
    let height = shape[0];
    let width = shape[1];

    let mut events = Vec::new();

    // Simple ESIM-style simulation
    for y in 0..height {
        for x in 0..width {
            let old_intensity = old_array[[y, x]] as f64;
            let new_intensity = new_array[[y, x]] as f64;

            // Avoid log(0)
            let old_log = (old_intensity + 1e-6).ln();
            let new_log = (new_intensity + 1e-6).ln();
            let log_diff = new_log - old_log;

            // Generate positive events
            if log_diff > threshold {
                events.push(Event {
                    x: x as u16,
                    y: y as u16,
                    t: 0.0,
                    polarity: 1,
                });
            }
            // Generate negative events
            else if log_diff < -threshold {
                events.push(Event {
                    x: x as u16,
                    y: y as u16,
                    t: 0.0,
                    polarity: -1,
                });
            }
        }
    }

    // Convert to arrays
    let (xs, ys, ts, ps) = events_to_arrays(events);

    Ok((
        xs.into_pyarray(py).to_object(py),
        ys.into_pyarray(py).to_object(py),
        ts.into_pyarray(py).to_object(py),
        ps.into_pyarray(py).to_object(py),
    ))
}

// Real-time streaming bindings (only available with GStreamer feature)
#[cfg(feature = "gstreamer")]
mod realtime_bindings {
    use super::*;

    /// Python wrapper for RealtimeStreamConfig
    #[pyclass]
    #[derive(Clone)]
    pub struct PyRealtimeStreamConfig {
        pub inner: RealtimeStreamConfig,
    }

    #[pymethods]
    impl PyRealtimeStreamConfig {
        #[new]
        #[pyo3(signature = (
            target_fps = None,
            max_buffer_size = None,
            device_id = None,
            contrast_threshold = None,
            resolution = None,
            auto_adjust_fps = None
        ))]
        pub fn new(
            target_fps: Option<f64>,
            max_buffer_size: Option<usize>,
            device_id: Option<u32>,
            contrast_threshold: Option<f64>,
            resolution: Option<(u32, u32)>,
            auto_adjust_fps: Option<bool>,
        ) -> Self {
            let mut config = RealtimeStreamConfig::default();

            if let Some(fps) = target_fps {
                config.target_fps = fps;
            }
            if let Some(buffer_size) = max_buffer_size {
                config.max_buffer_size = buffer_size;
            }
            if let Some(device) = device_id {
                config.device_id = device;
            }
            if let Some(threshold) = contrast_threshold {
                config.esim_config.base_config.contrast_threshold_pos = threshold;
                config.esim_config.base_config.contrast_threshold_neg = threshold;
            }
            if let Some(res) = resolution {
                config.esim_config.base_config.resolution = res;
            }
            if let Some(auto_adjust) = auto_adjust_fps {
                config.auto_adjust_fps = auto_adjust;
            }

            Self { inner: config }
        }

        #[getter]
        pub fn target_fps(&self) -> f64 {
            self.inner.target_fps
        }

        #[setter]
        pub fn set_target_fps(&mut self, fps: f64) {
            self.inner.target_fps = fps;
        }

        #[getter]
        pub fn max_buffer_size(&self) -> usize {
            self.inner.max_buffer_size
        }

        #[setter]
        pub fn set_max_buffer_size(&mut self, size: usize) {
            self.inner.max_buffer_size = size;
        }

        #[getter]
        pub fn device_id(&self) -> u32 {
            self.inner.device_id
        }

        #[setter]
        pub fn set_device_id(&mut self, device_id: u32) {
            self.inner.device_id = device_id;
        }

        #[getter]
        pub fn contrast_threshold(&self) -> f64 {
            self.inner.esim_config.base_config.contrast_threshold_pos
        }

        #[setter]
        pub fn set_contrast_threshold(&mut self, threshold: f64) {
            self.inner.esim_config.base_config.contrast_threshold_pos = threshold;
            self.inner.esim_config.base_config.contrast_threshold_neg = threshold;
        }

        #[getter]
        pub fn resolution(&self) -> (u32, u32) {
            self.inner.esim_config.base_config.resolution
        }

        #[setter]
        pub fn set_resolution(&mut self, resolution: (u32, u32)) {
            self.inner.esim_config.base_config.resolution = resolution;
        }

        #[getter]
        pub fn auto_adjust_fps(&self) -> bool {
            self.inner.auto_adjust_fps
        }

        #[setter]
        pub fn set_auto_adjust_fps(&mut self, auto_adjust: bool) {
            self.inner.auto_adjust_fps = auto_adjust;
        }

        pub fn __repr__(&self) -> String {
            format!(
                "RealtimeStreamConfig(fps={:.1}, buffer_size={}, device={}, threshold={:.3}, resolution={:?})",
                self.inner.target_fps,
                self.inner.max_buffer_size,
                self.inner.device_id,
                self.inner.esim_config.base_config.contrast_threshold_pos,
                self.inner.esim_config.base_config.resolution
            )
        }
    }

    /// Python wrapper for StreamingStats
    #[pyclass]
    pub struct PyStreamingStats {
        inner: StreamingStats,
    }

    #[pymethods]
    impl PyStreamingStats {
        #[getter]
        pub fn frames_processed(&self) -> u64 {
            self.inner.frames_processed
        }

        #[getter]
        pub fn events_generated(&self) -> u64 {
            self.inner.events_generated
        }

        #[getter]
        pub fn current_fps(&self) -> f64 {
            self.inner.current_fps
        }

        #[getter]
        pub fn avg_events_per_frame(&self) -> f64 {
            self.inner.avg_events_per_frame
        }

        #[getter]
        pub fn buffer_size(&self) -> usize {
            self.inner.buffer_size
        }

        #[getter]
        pub fn dropped_frames(&self) -> u64 {
            self.inner.dropped_frames
        }

        #[getter]
        pub fn avg_latency_ms(&self) -> f64 {
            self.inner.avg_latency_ms
        }

        pub fn __repr__(&self) -> String {
            format!(
                "StreamingStats(frames={}, events={}, fps={:.1}, buffer={}, dropped={})",
                self.inner.frames_processed,
                self.inner.events_generated,
                self.inner.current_fps,
                self.inner.buffer_size,
                self.inner.dropped_frames
            )
        }
    }

    /// Python wrapper for RealtimeEventStream
    #[pyclass]
    pub struct PyRealtimeEventStream {
        stream: RealtimeEventStream,
    }

    #[pymethods]
    impl PyRealtimeEventStream {
        #[new]
        pub fn new(config: &PyRealtimeStreamConfig) -> PyResult<Self> {
            let stream = RealtimeEventStream::new(config.inner.clone(), DEVICE.clone())
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;

            Ok(Self { stream })
        }

        /// Start webcam streaming
        pub fn start_streaming(&mut self) -> PyResult<()> {
            self.stream
                .start_streaming()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))
        }

        /// Stop streaming
        pub fn stop_streaming(&mut self) -> PyResult<()> {
            self.stream
                .stop_streaming()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))
        }

        /// Check if currently streaming
        pub fn is_streaming(&self) -> bool {
            self.stream.is_streaming()
        }

        /// Process next frame (non-blocking)
        pub fn process_next_frame(&mut self) -> PyResult<bool> {
            self.stream
                .process_next_frame()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))
        }

        /// Get events from buffer
        #[pyo3(signature = (max_count = None))]
        pub fn get_events(
            &self,
            py: Python<'_>,
            max_count: Option<usize>,
        ) -> PyResult<(PyObject, PyObject, PyObject, PyObject)> {
            let events = self.stream.get_events(max_count);
            let (xs, ys, ts, ps) = events_to_arrays(events);

            Ok((
                xs.into_pyarray(py).to_object(py),
                ys.into_pyarray(py).to_object(py),
                ts.into_pyarray(py).to_object(py),
                ps.into_pyarray(py).to_object(py),
            ))
        }

        /// Get events with timeout (blocking)
        #[pyo3(signature = (timeout_ms = None))]
        pub fn get_events_blocking(
            &self,
            py: Python<'_>,
            timeout_ms: Option<u64>,
        ) -> PyResult<(PyObject, PyObject, PyObject, PyObject)> {
            let events = self.stream.get_events_blocking(timeout_ms);
            let (xs, ys, ts, ps) = events_to_arrays(events);

            Ok((
                xs.into_pyarray(py).to_object(py),
                ys.into_pyarray(py).to_object(py),
                ts.into_pyarray(py).to_object(py),
                ps.into_pyarray(py).to_object(py),
            ))
        }

        /// Update streaming parameters
        #[pyo3(signature = (contrast_threshold = None, target_fps = None))]
        pub fn update_params(
            &mut self,
            contrast_threshold: Option<f64>,
            target_fps: Option<f64>,
        ) -> PyResult<()> {
            self.stream
                .update_esim_params(contrast_threshold, target_fps)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))
        }

        /// Get streaming statistics
        pub fn get_stats(&self) -> PyStreamingStats {
            PyStreamingStats {
                inner: self.stream.get_stats(),
            }
        }

        /// Reset stream state
        pub fn reset(&mut self) {
            self.stream.reset();
        }

        pub fn __repr__(&self) -> String {
            let stats = self.stream.get_stats();
            format!(
                "RealtimeEventStream(streaming={}, fps={:.1}, frames={}, events={})",
                self.stream.is_streaming(),
                stats.current_fps,
                stats.frames_processed,
                stats.events_generated
            )
        }
    }

    /// High-level function to create a real-time event stream
    #[pyfunction]
    #[pyo3(signature = (
        target_fps = 30.0,
        contrast_threshold = 0.15,
        device_id = 0,
        max_buffer_size = 10000,
        resolution = None
    ))]
    pub fn create_realtime_stream_py(
        target_fps: f64,
        contrast_threshold: f64,
        device_id: u32,
        max_buffer_size: usize,
        resolution: Option<(u32, u32)>,
    ) -> PyResult<PyRealtimeEventStream> {
        let mut config = RealtimeStreamConfig::default();
        config.target_fps = target_fps;
        config.max_buffer_size = max_buffer_size;
        config.device_id = device_id;
        config.esim_config.base_config.contrast_threshold_pos = contrast_threshold;
        config.esim_config.base_config.contrast_threshold_neg = contrast_threshold;

        if let Some(res) = resolution {
            config.esim_config.base_config.resolution = res;
        }

        let stream = RealtimeEventStream::new(config, DEVICE.clone())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;

        Ok(PyRealtimeEventStream { stream })
    }

    /// Check if GStreamer is available for real-time streaming
    #[pyfunction]
    pub fn is_realtime_available() -> bool {
        true // Available when compiled with gstreamer feature
    }

    // Re-export the realtime bindings are already public in this module
}

// Export realtime bindings conditionally
#[cfg(feature = "gstreamer")]
pub use realtime_bindings::{
    create_realtime_stream_py, is_realtime_available, PyRealtimeEventStream,
    PyRealtimeStreamConfig, PyStreamingStats,
};

// Fallback functions when GStreamer is not available
#[cfg(not(feature = "gstreamer"))]
#[pyfunction]
pub fn is_realtime_available() -> bool {
    false
}

#[cfg(not(feature = "gstreamer"))]
#[pyfunction]
pub fn create_realtime_stream_py(
    _target_fps: f64,
    _contrast_threshold: f64,
    _device_id: u32,
    _max_buffer_size: usize,
    _resolution: Option<(u32, u32)>,
) -> PyResult<()> {
    Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
        "Real-time streaming requires GStreamer support. Please compile with --features gstreamer",
    ))
}
