//! Python bindings for streaming event processing

use super::*;
use crate::ev_core::Event;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::PathBuf;

/// Python wrapper for StreamingConfig
#[pyclass]
#[derive(Clone)]
pub struct PyStreamingConfig {
    pub inner: StreamingConfig,
}

#[pymethods]
impl PyStreamingConfig {
    #[new]
    #[pyo3(signature = (
        window_size_us = 50_000,
        max_events_per_batch = 100_000,
        buffer_size = 1_000_000,
        timeout_ms = 100,
        voxel_method = "count".to_string(),
        num_bins = 5,
        resolution = (240, 180)
    ))]
    pub fn new(
        window_size_us: u64,
        max_events_per_batch: usize,
        buffer_size: usize,
        timeout_ms: u64,
        voxel_method: String,
        num_bins: u32,
        resolution: (u16, u16),
    ) -> Self {
        let config = StreamingConfig {
            window_size_us,
            max_events_per_batch,
            device: Device::Cpu,
            model_config: ModelLoadConfig::default(),
            buffer_size,
            timeout_ms,
            voxel_method,
            num_bins,
            resolution,
        };

        Self { inner: config }
    }

    /// Get configuration as dictionary
    pub fn to_dict(&self, py: Python) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("window_size_us", self.inner.window_size_us)?;
        dict.set_item("max_events_per_batch", self.inner.max_events_per_batch)?;
        dict.set_item("buffer_size", self.inner.buffer_size)?;
        dict.set_item("timeout_ms", self.inner.timeout_ms)?;
        dict.set_item("voxel_method", &self.inner.voxel_method)?;
        dict.set_item("num_bins", self.inner.num_bins)?;
        dict.set_item("resolution", self.inner.resolution)?;
        Ok(dict.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "StreamingConfig(window_size_us={}, max_events_per_batch={}, buffer_size={}, timeout_ms={}, num_bins={}, resolution={:?})",
            self.inner.window_size_us,
            self.inner.max_events_per_batch,
            self.inner.buffer_size,
            self.inner.timeout_ms,
            self.inner.num_bins,
            self.inner.resolution
        )
    }
}

/// Python wrapper for StreamingStats
#[pyclass]
#[derive(Clone)]
pub struct PyStreamingStats {
    pub inner: StreamingStats,
}

#[pymethods]
impl PyStreamingStats {
    /// Get stats as dictionary
    pub fn to_dict(&self, py: Python) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("total_events_processed", self.inner.total_events_processed)?;
        dict.set_item("total_frames_generated", self.inner.total_frames_generated)?;
        dict.set_item("average_latency_ms", self.inner.average_latency_ms)?;
        dict.set_item("events_per_second", self.inner.events_per_second)?;
        dict.set_item("frames_per_second", self.inner.frames_per_second)?;
        dict.set_item("buffer_utilization", self.inner.buffer_utilization)?;
        dict.set_item("processing_errors", self.inner.processing_errors)?;
        Ok(dict.into())
    }

    #[getter]
    pub fn total_events_processed(&self) -> u64 {
        self.inner.total_events_processed
    }

    #[getter]
    pub fn total_frames_generated(&self) -> u64 {
        self.inner.total_frames_generated
    }

    #[getter]
    pub fn average_latency_ms(&self) -> f64 {
        self.inner.average_latency_ms
    }

    #[getter]
    pub fn events_per_second(&self) -> f64 {
        self.inner.events_per_second
    }

    #[getter]
    pub fn frames_per_second(&self) -> f64 {
        self.inner.frames_per_second
    }

    #[getter]
    pub fn buffer_utilization(&self) -> f64 {
        self.inner.buffer_utilization
    }

    #[getter]
    pub fn processing_errors(&self) -> u64 {
        self.inner.processing_errors
    }

    fn __repr__(&self) -> String {
        format!(
            "StreamingStats(events_processed={}, frames_generated={}, latency={:.2}ms, eps={:.0}, fps={:.1}, buffer_util={:.1}%)",
            self.inner.total_events_processed,
            self.inner.total_frames_generated,
            self.inner.average_latency_ms,
            self.inner.events_per_second,
            self.inner.frames_per_second,
            self.inner.buffer_utilization * 100.0
        )
    }
}

/// Python wrapper for StreamingProcessor
#[pyclass]
pub struct PyStreamingProcessor {
    processor: StreamingProcessor,
}

#[pymethods]
impl PyStreamingProcessor {
    #[new]
    pub fn new(config: &PyStreamingConfig) -> Self {
        Self {
            processor: StreamingProcessor::new(config.inner.clone()),
        }
    }

    /// Load model for reconstruction
    pub fn load_model(&mut self, model_path: String) -> PyResult<()> {
        let path = PathBuf::from(model_path);
        self.processor.load_model(&path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load model: {}",
                e
            ))
        })?;
        Ok(())
    }

    /// Process events and return reconstructed frame
    pub fn process_events(
        &mut self,
        xs: PyReadonlyArray1<i64>,
        ys: PyReadonlyArray1<i64>,
        ts: PyReadonlyArray1<f64>,
        ps: PyReadonlyArray1<i64>,
    ) -> PyResult<Option<Py<PyArray2<f32>>>> {
        let py = xs.py();

        // Convert to events
        let events = self.convert_to_events(&xs, &ys, &ts, &ps)?;

        // Process events
        let result = self.processor.process_events(events).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Processing failed: {}", e))
        })?;

        // Convert result to numpy array if available
        if let Some(tensor) = result {
            let shape = tensor.dims();

            // Handle different tensor shapes: [H, W], [1, H, W], [B, H, W], [B, C, H, W]
            let (height, width) = match shape.len() {
                2 => (shape[0], shape[1]), // [H, W]
                3 => (shape[1], shape[2]), // [C, H, W] or [1, H, W]
                4 => (shape[2], shape[3]), // [B, C, H, W]
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Unsupported tensor shape: {:?}",
                        shape
                    )))
                }
            };

            // Convert tensor to 2D array
            let data = if shape.len() == 2 {
                tensor.to_vec2::<f32>().map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Failed to convert tensor: {}",
                        e
                    ))
                })?
            } else {
                // For 3D/4D tensors, take the first 2D slice or average across channels
                let tensor_2d = if shape.len() == 3 {
                    if shape[0] == 1 {
                        // [1, H, W] -> squeeze first dimension
                        tensor.squeeze(0).map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                                "Failed to squeeze tensor: {}",
                                e
                            ))
                        })?
                    } else {
                        // [C, H, W] -> take mean across channels
                        tensor.mean(0).map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                                "Failed to mean tensor: {}",
                                e
                            ))
                        })?
                    }
                } else {
                    // [B, C, H, W] -> take first batch and mean across channels
                    let first_batch = tensor.get(0).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "Failed to get first batch: {}",
                            e
                        ))
                    })?;
                    if shape[1] == 1 {
                        first_batch.squeeze(0).map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                                "Failed to squeeze tensor: {}",
                                e
                            ))
                        })?
                    } else {
                        first_batch.mean(0).map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                                "Failed to mean tensor: {}",
                                e
                            ))
                        })?
                    }
                };

                tensor_2d.to_vec2::<f32>().map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Failed to convert tensor: {}",
                        e
                    ))
                })?
            };

            let flat_data: Vec<f32> = data.into_iter().flatten().collect();
            let output_array = flat_data
                .into_pyarray(py)
                .reshape([height, width])
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Failed to reshape: {}",
                        e
                    ))
                })?;

            Ok(Some(output_array.to_owned()))
        } else {
            Ok(None)
        }
    }

    /// Get current statistics
    pub fn get_stats(&self) -> PyStreamingStats {
        PyStreamingStats {
            inner: self.processor.get_stats(),
        }
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.processor.reset_stats();
    }

    /// Check if processor is ready
    pub fn is_ready(&self) -> bool {
        self.processor.is_ready()
    }

    /// Get buffer status (length, utilization)
    pub fn get_buffer_status(&self) -> (usize, f64) {
        self.processor.get_buffer_status()
    }
}

impl PyStreamingProcessor {
    fn convert_to_events(
        &self,
        xs: &PyReadonlyArray1<i64>,
        ys: &PyReadonlyArray1<i64>,
        ts: &PyReadonlyArray1<f64>,
        ps: &PyReadonlyArray1<i64>,
    ) -> PyResult<Vec<Event>> {
        let xs_array = xs.as_array();
        let ys_array = ys.as_array();
        let ts_array = ts.as_array();
        let ps_array = ps.as_array();

        let mut events = Vec::with_capacity(xs_array.len());
        for i in 0..xs_array.len() {
            events.push(Event {
                x: xs_array[i] as u16,
                y: ys_array[i] as u16,
                t: ts_array[i],
                polarity: ps_array[i] > 0,
            });
        }

        // Sort by timestamp
        events.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));

        Ok(events)
    }
}

/// Python wrapper for EventStream
#[pyclass]
pub struct PyEventStream {
    stream: EventStream,
}

#[pymethods]
impl PyEventStream {
    #[new]
    pub fn new(config: &PyStreamingConfig) -> Self {
        Self {
            stream: EventStream::new(config.inner.clone()),
        }
    }

    /// Load model for the stream
    pub fn load_model(&mut self, model_path: String) -> PyResult<()> {
        let path = PathBuf::from(model_path);
        self.stream.load_model(&path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load model: {}",
                e
            ))
        })?;
        Ok(())
    }

    /// Start the stream
    pub fn start(&mut self) -> PyResult<()> {
        self.stream.start().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to start stream: {}",
                e
            ))
        })?;
        Ok(())
    }

    /// Stop the stream
    pub fn stop(&mut self) {
        self.stream.stop();
    }

    /// Process a batch of events
    pub fn process_batch(
        &mut self,
        xs: PyReadonlyArray1<i64>,
        ys: PyReadonlyArray1<i64>,
        ts: PyReadonlyArray1<f64>,
        ps: PyReadonlyArray1<i64>,
    ) -> PyResult<Option<Py<PyArray2<f32>>>> {
        let py = xs.py();

        // Convert to events
        let events = self.convert_to_events(&xs, &ys, &ts, &ps)?;

        // Process batch
        let result = self.stream.process_batch(events).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Batch processing failed: {}",
                e
            ))
        })?;

        // Convert result if available
        if let Some(tensor) = result {
            let shape = tensor.dims();

            // Handle different tensor shapes: [H, W], [1, H, W], [B, H, W], [B, C, H, W]
            let (height, width) = match shape.len() {
                2 => (shape[0], shape[1]), // [H, W]
                3 => (shape[1], shape[2]), // [C, H, W] or [1, H, W]
                4 => (shape[2], shape[3]), // [B, C, H, W]
                _ => {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Unsupported tensor shape: {:?}",
                        shape
                    )))
                }
            };

            // Convert tensor to 2D array
            let data = if shape.len() == 2 {
                tensor.to_vec2::<f32>().map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Failed to convert tensor: {}",
                        e
                    ))
                })?
            } else {
                // For 3D/4D tensors, take the first 2D slice or average across channels
                let tensor_2d = if shape.len() == 3 {
                    if shape[0] == 1 {
                        // [1, H, W] -> squeeze first dimension
                        tensor.squeeze(0).map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                                "Failed to squeeze tensor: {}",
                                e
                            ))
                        })?
                    } else {
                        // [C, H, W] -> take mean across channels
                        tensor.mean(0).map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                                "Failed to mean tensor: {}",
                                e
                            ))
                        })?
                    }
                } else {
                    // [B, C, H, W] -> take first batch and mean across channels
                    let first_batch = tensor.get(0).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "Failed to get first batch: {}",
                            e
                        ))
                    })?;
                    if shape[1] == 1 {
                        first_batch.squeeze(0).map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                                "Failed to squeeze tensor: {}",
                                e
                            ))
                        })?
                    } else {
                        first_batch.mean(0).map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                                "Failed to mean tensor: {}",
                                e
                            ))
                        })?
                    }
                };

                tensor_2d.to_vec2::<f32>().map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Failed to convert tensor: {}",
                        e
                    ))
                })?
            };

            let flat_data: Vec<f32> = data.into_iter().flatten().collect();
            let output_array = flat_data
                .into_pyarray(py)
                .reshape([height, width])
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Failed to reshape: {}",
                        e
                    ))
                })?;

            Ok(Some(output_array.to_owned()))
        } else {
            Ok(None)
        }
    }

    /// Get performance statistics
    pub fn get_performance_stats(&self) -> PyStreamingStats {
        PyStreamingStats {
            inner: self.stream.get_performance_stats(),
        }
    }

    /// Check if stream is running
    pub fn is_running(&self) -> bool {
        self.stream.is_running()
    }
}

impl PyEventStream {
    fn convert_to_events(
        &self,
        xs: &PyReadonlyArray1<i64>,
        ys: &PyReadonlyArray1<i64>,
        ts: &PyReadonlyArray1<f64>,
        ps: &PyReadonlyArray1<i64>,
    ) -> PyResult<Vec<Event>> {
        let xs_array = xs.as_array();
        let ys_array = ys.as_array();
        let ts_array = ts.as_array();
        let ps_array = ps.as_array();

        let mut events = Vec::with_capacity(xs_array.len());
        for i in 0..xs_array.len() {
            events.push(Event {
                x: xs_array[i] as u16,
                y: ys_array[i] as u16,
                t: ts_array[i],
                polarity: ps_array[i] > 0,
            });
        }

        // Sort by timestamp
        events.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));

        Ok(events)
    }
}

/// Create streaming configuration with default values
#[pyfunction]
pub fn create_streaming_config() -> PyStreamingConfig {
    PyStreamingConfig {
        inner: StreamingConfig::default(),
    }
}

/// Process events in streaming mode (functional interface)
#[pyfunction]
#[pyo3(signature = (
    xs,
    ys,
    ts,
    ps,
    config = None,
    model_path = None
))]
pub fn process_events_streaming(
    xs: PyReadonlyArray1<i64>,
    ys: PyReadonlyArray1<i64>,
    ts: PyReadonlyArray1<f64>,
    ps: PyReadonlyArray1<i64>,
    config: Option<&PyStreamingConfig>,
    model_path: Option<String>,
) -> PyResult<Option<Py<PyArray2<f32>>>> {
    let py = xs.py();

    // Use provided config or default
    let config = config.map(|c| c.inner.clone()).unwrap_or_default();

    // Create processor
    let mut processor = StreamingProcessor::new(config);

    // Load model if provided
    if let Some(path) = model_path {
        let model_path = PathBuf::from(path);
        processor.load_model(&model_path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Failed to load model: {}",
                e
            ))
        })?;
    }

    // Convert to events
    let xs_array = xs.as_array();
    let ys_array = ys.as_array();
    let ts_array = ts.as_array();
    let ps_array = ps.as_array();

    let mut events = Vec::with_capacity(xs_array.len());
    for i in 0..xs_array.len() {
        events.push(Event {
            x: xs_array[i] as u16,
            y: ys_array[i] as u16,
            t: ts_array[i],
            polarity: ps_array[i] > 0,
        });
    }

    // Process events
    let result = processor.process_events(events).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Processing failed: {}", e))
    })?;

    // Convert result if available
    if let Some(tensor) = result {
        let shape = tensor.dims();

        // Handle different tensor shapes: [H, W], [1, H, W], [B, H, W], [B, C, H, W]
        let (height, width) = match shape.len() {
            2 => (shape[0], shape[1]), // [H, W]
            3 => (shape[1], shape[2]), // [C, H, W] or [1, H, W]
            4 => (shape[2], shape[3]), // [B, C, H, W]
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unsupported tensor shape: {:?}",
                    shape
                )))
            }
        };

        // Convert tensor to 2D array
        let data = if shape.len() == 2 {
            tensor.to_vec2::<f32>().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to convert tensor: {}",
                    e
                ))
            })?
        } else {
            // For 3D/4D tensors, take the first 2D slice or average across channels
            let tensor_2d = if shape.len() == 3 {
                if shape[0] == 1 {
                    // [1, H, W] -> squeeze first dimension
                    tensor.squeeze(0).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "Failed to squeeze tensor: {}",
                            e
                        ))
                    })?
                } else {
                    // [C, H, W] -> take mean across channels
                    tensor.mean(0).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "Failed to mean tensor: {}",
                            e
                        ))
                    })?
                }
            } else {
                // [B, C, H, W] -> take first batch and mean across channels
                let first_batch = tensor.get(0).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Failed to get first batch: {}",
                        e
                    ))
                })?;
                if shape[1] == 1 {
                    first_batch.squeeze(0).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "Failed to squeeze tensor: {}",
                            e
                        ))
                    })?
                } else {
                    first_batch.mean(0).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "Failed to mean tensor: {}",
                            e
                        ))
                    })?
                }
            };

            tensor_2d.to_vec2::<f32>().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to convert tensor: {}",
                    e
                ))
            })?
        };

        let flat_data: Vec<f32> = data.into_iter().flatten().collect();
        let output_array = flat_data
            .into_pyarray(py)
            .reshape([height, width])
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to reshape: {}",
                    e
                ))
            })?;

        Ok(Some(output_array.to_owned()))
    } else {
        Ok(None)
    }
}
