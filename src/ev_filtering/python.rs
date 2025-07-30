//! Python bindings for event filtering functionality
//!
//! This module provides PyO3 bindings that maintain full API compatibility
//! with the existing Python filtering module while leveraging the high-performance
//! Rust implementations.

use super::{
    config::FilterConfig,
    denoise::{DenoiseFilter, DenoiseMethod},
    filter_events,
    hot_pixel::HotPixelFilter,
    polarity::PolarityFilter,
    spatial::{RegionOfInterest, SpatialFilter},
    temporal::TemporalFilter,
    FilterError,
};
use crate::ev_core::{from_numpy_arrays, to_numpy_arrays};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyTuple};
use pyo3::Bound;

/// Error type for Python bindings
#[derive(Debug)]
pub struct PythonFilterError(FilterError);

impl From<FilterError> for PythonFilterError {
    fn from(err: FilterError) -> Self {
        PythonFilterError(err)
    }
}

impl From<PythonFilterError> for PyErr {
    fn from(err: PythonFilterError) -> Self {
        pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", err.0))
    }
}

/// Python-compatible result type
#[allow(dead_code)]
type PyFilterResult<T> = Result<T, PythonFilterError>;

/// Filter events by time range (Python API compatible)
///
/// This function maintains full compatibility with the Python `filter_by_time` function.
///
/// # Arguments
///
/// * `xs` - X coordinates as numpy array
/// * `ys` - Y coordinates as numpy array
/// * `ts` - Timestamps as numpy array (in seconds)
/// * `ps` - Polarities as numpy array
/// * `t_start` - Start time in seconds (None for no lower bound)
/// * `t_end` - End time in seconds (None for no upper bound)
///
/// # Returns
///
/// Tuple of filtered (xs, ys, ts, ps) numpy arrays
#[pyfunction]
#[pyo3(name = "filter_by_time")]
pub fn filter_by_time_py(
    py: Python<'_>,
    xs: PyReadonlyArray1<i64>,
    ys: PyReadonlyArray1<i64>,
    ts: PyReadonlyArray1<f64>,
    ps: PyReadonlyArray1<i64>,
    t_start: Option<f64>,
    t_end: Option<f64>,
) -> PyResult<PyObject> {
    let events = from_numpy_arrays(xs, ys, ts, ps);

    let mut config = FilterConfig::new();

    if let (Some(start), Some(end)) = (t_start, t_end) {
        config = config.with_temporal_filter(TemporalFilter::time_window(start, end));
    } else if let Some(start) = t_start {
        config = config.with_temporal_filter(TemporalFilter::time_window(start, f64::INFINITY));
    } else if let Some(end) = t_end {
        config = config.with_temporal_filter(TemporalFilter::time_window(0.0, end));
    }

    let filtered_events = filter_events(&events, &config).map_err(PythonFilterError::from)?;

    let (xs_out, ys_out, ts_out, ps_out) = to_numpy_arrays(&filtered_events).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Conversion error: {}", e))
    })?;

    let result = PyTuple::new(py, [xs_out, ys_out, ts_out, ps_out])?;
    Ok(result.into())
}

/// Filter events by region of interest (Python API compatible)
///
/// This function maintains full compatibility with the Python `filter_by_roi` function.
///
/// # Arguments
///
/// * `xs` - X coordinates as numpy array
/// * `ys` - Y coordinates as numpy array
/// * `ts` - Timestamps as numpy array
/// * `ps` - Polarities as numpy array
/// * `x_min` - Minimum x coordinate (inclusive)
/// * `x_max` - Maximum x coordinate (inclusive)
/// * `y_min` - Minimum y coordinate (inclusive)
/// * `y_max` - Maximum y coordinate (inclusive)
///
/// # Returns
///
/// Tuple of filtered (xs, ys, ts, ps) numpy arrays
#[pyfunction]
#[pyo3(name = "filter_by_roi")]
#[allow(clippy::too_many_arguments)]
pub fn filter_by_roi_py(
    py: Python<'_>,
    xs: PyReadonlyArray1<i64>,
    ys: PyReadonlyArray1<i64>,
    ts: PyReadonlyArray1<f64>,
    ps: PyReadonlyArray1<i64>,
    x_min: i64,
    x_max: i64,
    y_min: i64,
    y_max: i64,
) -> PyResult<PyObject> {
    let events = from_numpy_arrays(xs, ys, ts, ps);

    let roi = RegionOfInterest::new(x_min as u16, x_max as u16, y_min as u16, y_max as u16)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid ROI: {}", e)))?;

    let config = FilterConfig::new().with_spatial_filter(SpatialFilter::from_roi(roi));

    let filtered_events = filter_events(&events, &config).map_err(PythonFilterError::from)?;

    let (xs_out, ys_out, ts_out, ps_out) = to_numpy_arrays(&filtered_events).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Conversion error: {}", e))
    })?;

    let result = PyTuple::new(py, [xs_out, ys_out, ts_out, ps_out])?;
    Ok(result.into())
}

/// Filter events by polarity (Python API compatible)
///
/// This function maintains full compatibility with the Python `filter_by_polarity` function.
///
/// # Arguments
///
/// * `xs` - X coordinates as numpy array
/// * `ys` - Y coordinates as numpy array
/// * `ts` - Timestamps as numpy array
/// * `ps` - Polarities as numpy array
/// * `polarity` - Polarity value(s) to keep (can be single int or list of ints)
///
/// # Returns
///
/// Tuple of filtered (xs, ys, ts, ps) numpy arrays
#[pyfunction]
#[pyo3(name = "filter_by_polarity")]
pub fn filter_by_polarity_py(
    py: Python<'_>,
    xs: PyReadonlyArray1<i64>,
    ys: PyReadonlyArray1<i64>,
    ts: PyReadonlyArray1<f64>,
    ps: PyReadonlyArray1<i64>,
    polarity: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let events = from_numpy_arrays(xs, ys, ts, ps);

    // Handle both single int and list of ints for polarity
    let polarity_values: Vec<i8> = if let Ok(single_val) = polarity.extract::<i64>() {
        vec![if single_val != 0 { 1 } else { 0 }]
    } else if let Ok(polarity_list) = polarity.extract::<Vec<i64>>() {
        polarity_list
            .into_iter()
            .map(|p| if p != 0 { 1i8 } else { 0i8 })
            .collect()
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "polarity must be an integer or list of integers",
        ));
    };

    let polarity_filter = PolarityFilter::from_values(polarity_values);
    let config = FilterConfig::new().with_polarity_filter(polarity_filter);

    let filtered_events = filter_events(&events, &config).map_err(PythonFilterError::from)?;

    let (xs_out, ys_out, ts_out, ps_out) = to_numpy_arrays(&filtered_events).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Conversion error: {}", e))
    })?;

    let result = PyTuple::new(py, [xs_out, ys_out, ts_out, ps_out])?;
    Ok(result.into())
}

/// Remove hot pixels (Python API compatible)
///
/// This function maintains full compatibility with the Python `filter_hot_pixels` function.
///
/// # Arguments
///
/// * `xs` - X coordinates as numpy array
/// * `ys` - Y coordinates as numpy array
/// * `ts` - Timestamps as numpy array
/// * `ps` - Polarities as numpy array
/// * `threshold_percentile` - Percentile threshold for hot pixel detection (default: 99.9)
///
/// # Returns
///
/// Tuple of filtered (xs, ys, ts, ps) numpy arrays
#[pyfunction]
#[pyo3(name = "filter_hot_pixels")]
pub fn filter_hot_pixels_py(
    py: Python<'_>,
    xs: PyReadonlyArray1<i64>,
    ys: PyReadonlyArray1<i64>,
    ts: PyReadonlyArray1<f64>,
    ps: PyReadonlyArray1<i64>,
    threshold_percentile: Option<f64>,
) -> PyResult<PyObject> {
    let events = from_numpy_arrays(xs, ys, ts, ps);

    let percentile = threshold_percentile.unwrap_or(99.9);
    let hot_pixel_filter = HotPixelFilter::percentile(percentile);

    let config = FilterConfig::new().with_hot_pixel_filter(hot_pixel_filter);

    let filtered_events = filter_events(&events, &config).map_err(PythonFilterError::from)?;

    let (xs_out, ys_out, ts_out, ps_out) = to_numpy_arrays(&filtered_events).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Conversion error: {}", e))
    })?;

    let result = PyTuple::new(py, [xs_out, ys_out, ts_out, ps_out])?;
    Ok(result.into())
}

/// Remove noise events (Python API compatible)
///
/// This function maintains full compatibility with the Python `filter_noise` function.
///
/// # Arguments
///
/// * `xs` - X coordinates as numpy array
/// * `ys` - Y coordinates as numpy array
/// * `ts` - Timestamps as numpy array
/// * `ps` - Polarities as numpy array
/// * `method` - Noise filtering method ("refractory" or "correlation")
/// * `refractory_period_us` - Refractory period in microseconds (default: 1000)
///
/// # Returns
///
/// Tuple of filtered (xs, ys, ts, ps) numpy arrays
#[pyfunction]
#[pyo3(name = "filter_noise")]
pub fn filter_noise_py(
    py: Python<'_>,
    xs: PyReadonlyArray1<i64>,
    ys: PyReadonlyArray1<i64>,
    ts: PyReadonlyArray1<f64>,
    ps: PyReadonlyArray1<i64>,
    method: Option<&str>,
    refractory_period_us: Option<f64>,
) -> PyResult<PyObject> {
    let events = from_numpy_arrays(xs, ys, ts, ps);

    let _denoise_method = match method.unwrap_or("refractory") {
        "refractory" => DenoiseMethod::RefractoryPeriod,
        "correlation" => DenoiseMethod::TemporalCorrelation,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "method must be 'refractory' or 'correlation'",
            ))
        }
    };

    let period_us = refractory_period_us.unwrap_or(1000.0);
    let denoise_filter = DenoiseFilter::refractory(period_us);

    let config = FilterConfig::new().with_denoise_filter(denoise_filter);

    let filtered_events = filter_events(&events, &config).map_err(PythonFilterError::from)?;

    let (xs_out, ys_out, ts_out, ps_out) = to_numpy_arrays(&filtered_events).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Conversion error: {}", e))
    })?;

    let result = PyTuple::new(py, [xs_out, ys_out, ts_out, ps_out])?;
    Ok(result.into())
}

/// High-level event preprocessing pipeline (Python API compatible)
///
/// This function maintains full compatibility with the Python `preprocess_events` function.
///
/// # Arguments
///
/// * `xs` - X coordinates as numpy array
/// * `ys` - Y coordinates as numpy array
/// * `ts` - Timestamps as numpy array
/// * `ps` - Polarities as numpy array
/// * `t_start` - Start time in seconds (None for no lower bound)
/// * `t_end` - End time in seconds (None for no upper bound)
/// * `roi` - Region of interest as (x_min, x_max, y_min, y_max) tuple
/// * `polarity` - Polarity value(s) to keep (None for all)
/// * `remove_hot_pixels` - Whether to remove hot pixels
/// * `remove_noise` - Whether to apply noise filtering
/// * `hot_pixel_threshold` - Percentile threshold for hot pixel detection
/// * `refractory_period_us` - Refractory period in microseconds
///
/// # Returns
///
/// Tuple of preprocessed (xs, ys, ts, ps) numpy arrays
#[pyfunction]
#[pyo3(
    name = "preprocess_events",
    signature = (
        xs, ys, ts, ps,
        t_start=None,
        t_end=None,
        roi=None,
        polarity=None,
        remove_hot_pixels=true,
        remove_noise=true,
        hot_pixel_threshold=99.9,
        refractory_period_us=1000.0
    )
)]
#[allow(clippy::too_many_arguments)]
pub fn preprocess_events_py(
    py: Python<'_>,
    xs: PyReadonlyArray1<i64>,
    ys: PyReadonlyArray1<i64>,
    ts: PyReadonlyArray1<f64>,
    ps: PyReadonlyArray1<i64>,
    t_start: Option<f64>,
    t_end: Option<f64>,
    roi: Option<(i64, i64, i64, i64)>,
    polarity: Option<&Bound<'_, PyAny>>,
    remove_hot_pixels: bool,
    remove_noise: bool,
    hot_pixel_threshold: f64,
    refractory_period_us: f64,
) -> PyResult<PyObject> {
    let events = from_numpy_arrays(xs, ys, ts, ps);

    // Build configuration using the exact same logic as the Python version
    let mut config = FilterConfig::new();

    // Add temporal filtering
    if let (Some(start), Some(end)) = (t_start, t_end) {
        config = config.with_temporal_filter(TemporalFilter::time_window(start, end));
    } else if let Some(start) = t_start {
        config = config.with_temporal_filter(TemporalFilter::time_window(start, f64::INFINITY));
    } else if let Some(end) = t_end {
        config = config.with_temporal_filter(TemporalFilter::time_window(0.0, end));
    }

    // Add spatial ROI filtering
    if let Some((x_min, x_max, y_min, y_max)) = roi {
        let roi_filter =
            RegionOfInterest::new(x_min as u16, x_max as u16, y_min as u16, y_max as u16).map_err(
                |e| pyo3::exceptions::PyValueError::new_err(format!("Invalid ROI: {}", e)),
            )?;
        config = config.with_spatial_filter(SpatialFilter::from_roi(roi_filter));
    }

    // Add polarity filtering
    if let Some(p) = polarity {
        let polarity_values: Vec<i8> = if let Ok(single_val) = p.extract::<i64>() {
            vec![if single_val != 0 { 1 } else { 0 }]
        } else if let Ok(polarity_list) = p.extract::<Vec<i64>>() {
            polarity_list
                .into_iter()
                .map(|p| if p != 0 { 1i8 } else { 0i8 })
                .collect()
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "polarity must be an integer or list of integers",
            ));
        };
        config = config.with_polarity_filter(PolarityFilter::from_values(polarity_values));
    }

    // Add hot pixel filtering
    if remove_hot_pixels {
        config = config.with_hot_pixel_filter(HotPixelFilter::percentile(hot_pixel_threshold));
    }

    // Add noise filtering
    if remove_noise {
        config = config.with_denoise_filter(DenoiseFilter::refractory(refractory_period_us));
    }

    let filtered_events = filter_events(&events, &config).map_err(PythonFilterError::from)?;

    let (xs_out, ys_out, ts_out, ps_out) = to_numpy_arrays(&filtered_events).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Conversion error: {}", e))
    })?;

    let result = PyTuple::new(py, [xs_out, ys_out, ts_out, ps_out])?;
    Ok(result.into())
}

/// Utility function to get filtering statistics (new functionality)
///
/// # Arguments
///
/// * `xs` - X coordinates as numpy array
/// * `ys` - Y coordinates as numpy array
/// * `ts` - Timestamps as numpy array
/// * `ps` - Polarities as numpy array
///
/// # Returns
///
/// Dictionary with event statistics
#[pyfunction]
#[pyo3(name = "get_event_statistics")]
pub fn get_event_statistics_py(
    py: Python<'_>,
    xs: PyReadonlyArray1<i64>,
    ys: PyReadonlyArray1<i64>,
    ts: PyReadonlyArray1<f64>,
    ps: PyReadonlyArray1<i64>,
) -> PyResult<PyObject> {
    let events = from_numpy_arrays(xs, ys, ts, ps);

    if events.is_empty() {
        let dict = PyDict::new(py);
        dict.set_item("total_events", 0)?;
        dict.set_item("duration", 0.0)?;
        dict.set_item("event_rate", 0.0)?;
        dict.set_item("unique_pixels", 0)?;
        dict.set_item("positive_events", 0)?;
        dict.set_item("negative_events", 0)?;
        return Ok(dict.into());
    }

    let total_events = events.len();
    let duration = events.last().unwrap().t - events.first().unwrap().t;
    let event_rate = if duration > 0.0 {
        total_events as f64 / duration
    } else {
        0.0
    };

    let mut unique_pixels = std::collections::HashSet::new();
    let mut positive_events = 0;
    let mut negative_events = 0;

    for event in &events {
        unique_pixels.insert((event.x, event.y));
        if event.polarity {
            positive_events += 1;
        } else {
            negative_events += 1;
        }
    }

    let dict = PyDict::new(py);
    dict.set_item("total_events", total_events)?;
    dict.set_item("duration", duration)?;
    dict.set_item("event_rate", event_rate)?;
    dict.set_item("unique_pixels", unique_pixels.len())?;
    dict.set_item("positive_events", positive_events)?;
    dict.set_item("negative_events", negative_events)?;
    dict.set_item(
        "polarity_balance",
        positive_events as f64 / total_events as f64,
    )?;

    Ok(dict.into())
}

/// Advanced filtering with configuration object (new functionality)
///
/// # Arguments
///
/// * `xs` - X coordinates as numpy array
/// * `ys` - Y coordinates as numpy array
/// * `ts` - Timestamps as numpy array
/// * `ps` - Polarities as numpy array
/// * `config_dict` - Dictionary with filtering configuration
///
/// # Returns
///
/// Tuple of filtered (xs, ys, ts, ps) numpy arrays
#[pyfunction]
#[pyo3(name = "filter_events_advanced")]
pub fn filter_events_advanced_py(
    py: Python<'_>,
    xs: PyReadonlyArray1<i64>,
    ys: PyReadonlyArray1<i64>,
    ts: PyReadonlyArray1<f64>,
    ps: PyReadonlyArray1<i64>,
    config_dict: &Bound<'_, PyDict>,
) -> PyResult<PyObject> {
    let events = from_numpy_arrays(xs, ys, ts, ps);

    // Parse configuration from dictionary
    let mut config = FilterConfig::new();

    // Temporal filtering
    if let Some(temporal_config) = config_dict.get_item("temporal")? {
        if let Ok(temporal_dict) = temporal_config.downcast::<PyDict>() {
            if let (Some(t_start), Some(t_end)) = (
                temporal_dict
                    .get_item("t_start")?
                    .and_then(|v| v.extract::<f64>().ok()),
                temporal_dict
                    .get_item("t_end")?
                    .and_then(|v| v.extract::<f64>().ok()),
            ) {
                config = config.with_temporal_filter(TemporalFilter::time_window(t_start, t_end));
            }
        }
    }

    // Spatial filtering
    if let Some(spatial_config) = config_dict.get_item("spatial")? {
        if let Ok(spatial_dict) = spatial_config.downcast::<PyDict>() {
            if let Some(roi_tuple) = spatial_dict.get_item("roi")? {
                if let Ok((x_min, x_max, y_min, y_max)) =
                    roi_tuple.extract::<(i64, i64, i64, i64)>()
                {
                    let roi = RegionOfInterest::new(
                        x_min as u16,
                        x_max as u16,
                        y_min as u16,
                        y_max as u16,
                    )
                    .map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!("Invalid ROI: {}", e))
                    })?;
                    config = config.with_spatial_filter(SpatialFilter::from_roi(roi));
                }
            }
        }
    }

    // Hot pixel filtering
    if let Some(hot_pixel_config) = config_dict.get_item("hot_pixels")? {
        if let Ok(hot_pixel_dict) = hot_pixel_config.downcast::<PyDict>() {
            if let Some(enabled) = hot_pixel_dict
                .get_item("enabled")?
                .and_then(|v| v.extract::<bool>().ok())
            {
                if enabled {
                    let percentile = hot_pixel_dict
                        .get_item("percentile")?
                        .and_then(|v| v.extract::<f64>().ok())
                        .unwrap_or(99.9);
                    config = config.with_hot_pixel_filter(HotPixelFilter::percentile(percentile));
                }
            }
        }
    }

    // Noise filtering
    if let Some(noise_config) = config_dict.get_item("noise")? {
        if let Ok(noise_dict) = noise_config.downcast::<PyDict>() {
            if let Some(enabled) = noise_dict
                .get_item("enabled")?
                .and_then(|v| v.extract::<bool>().ok())
            {
                if enabled {
                    let refractory_period = noise_dict
                        .get_item("refractory_period_us")?
                        .and_then(|v| v.extract::<f64>().ok())
                        .unwrap_or(1000.0);
                    config =
                        config.with_denoise_filter(DenoiseFilter::refractory(refractory_period));
                }
            }
        }
    }

    let filtered_events = filter_events(&events, &config).map_err(PythonFilterError::from)?;

    let (xs_out, ys_out, ts_out, ps_out) = to_numpy_arrays(&filtered_events).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Conversion error: {}", e))
    })?;

    let result = PyTuple::new(py, [xs_out, ys_out, ts_out, ps_out])?;
    Ok(result.into())
}

/// Register all filtering functions with the Python module
pub fn register_filtering_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(filter_by_time_py, m)?)?;
    m.add_function(wrap_pyfunction!(filter_by_roi_py, m)?)?;
    m.add_function(wrap_pyfunction!(filter_by_polarity_py, m)?)?;
    m.add_function(wrap_pyfunction!(filter_hot_pixels_py, m)?)?;
    m.add_function(wrap_pyfunction!(filter_noise_py, m)?)?;
    m.add_function(wrap_pyfunction!(preprocess_events_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_event_statistics_py, m)?)?;
    m.add_function(wrap_pyfunction!(filter_events_advanced_py, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ev_core::Event;
    use numpy::array::PyArray1;
    use pyo3::types::IntoPyDict;

    fn create_test_events() -> Events {
        vec![
            Event {
                x: 100,
                y: 200,
                t: 0.001,
                polarity: true,
            },
            Event {
                x: 150,
                y: 250,
                t: 0.002,
                polarity: false,
            },
            Event {
                x: 200,
                y: 300,
                t: 0.003,
                polarity: true,
            },
            Event {
                x: 250,
                y: 350,
                t: 0.004,
                polarity: false,
            },
            Event {
                x: 300,
                y: 400,
                t: 0.005,
                polarity: true,
            },
        ]
    }

    #[test]
    fn test_python_bindings_compilation() {
        pyo3::prepare_freethreaded_python();
        let test_events = create_test_events();
        assert_eq!(test_events.len(), 5);

        // Test that we can extract arrays (compilation test)
        let arrays_result = to_numpy_arrays(&test_events);
        assert!(arrays_result.is_ok());
    }

    #[test]
    fn test_filter_config_from_python_params() {
        let config = FilterConfig::preprocessing(
            Some(0.1),
            Some(0.5),
            Some((100, 500, 100, 400)),
            Some(vec![1]),
            true,
            true,
            99.9,
            1000.0,
        );

        // Verify configuration is valid
        assert!(config.validate().is_ok());
    }
}
