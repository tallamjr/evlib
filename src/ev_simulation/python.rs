//! Python bindings for event simulation (Video-to-Events)

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
