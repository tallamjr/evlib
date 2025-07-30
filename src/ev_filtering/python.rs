//! Python bindings for event filtering functionality
//!
//! This module provides PyO3 bindings for polars-first event filtering.
//! All functions work with Polars LazyFrames for high-performance processing.

use super::{
    config::FilterConfig,
    denoise::{DenoiseFilter, DenoiseMethod},
    hot_pixel::HotPixelFilter,
    polarity::PolarityFilter,
    spatial::{RegionOfInterest, SpatialFilter},
    temporal::TemporalFilter,
};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::Bound;

/// Filter events by time range (LazyFrame compatible)
///
/// This function accepts a LazyFrame and returns a filtered LazyFrame.
/// It leverages the high-performance Rust filtering implementation via polars.
///
/// # Arguments
///
/// * `events_lf` - Input events as LazyFrame
/// * `t_start` - Start time in seconds (None for no lower bound)
/// * `t_end` - End time in seconds (None for no upper bound)
///
/// # Returns
///
/// Filtered LazyFrame
#[pyfunction]
#[pyo3(name = "filter_by_time")]
pub fn filter_by_time_lf_py(
    events_lf: &Bound<'_, PyAny>,
    t_start: Option<f64>,
    t_end: Option<f64>,
) -> PyResult<PyObject> {
    #[cfg(feature = "polars")]
    {
        // Extract LazyFrame from Python
        let lazy_frame = crate::ev_core::python::extract_lazy_frame(events_lf)?;

        // Build temporal filter configuration
        let mut config = FilterConfig::new();

        if let (Some(start), Some(end)) = (t_start, t_end) {
            config = config.with_temporal_filter(TemporalFilter::time_window(start, end));
        } else if let Some(start) = t_start {
            config = config.with_temporal_filter(TemporalFilter::time_window(start, f64::INFINITY));
        } else if let Some(end) = t_end {
            config = config.with_temporal_filter(TemporalFilter::time_window(0.0, end));
        }

        // Apply temporal filter using Polars LazyFrame operations
        let filtered_lf = if let Some(temporal_filter) = &config.temporal_filter {
            super::temporal::apply_temporal_filter(lazy_frame, temporal_filter).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Temporal filter error: {}", e))
            })?
        } else {
            lazy_frame
        };

        // Convert back to Python LazyFrame
        crate::ev_core::python::lazy_frame_to_python(filtered_lf, events_lf.py())
    }

    #[cfg(not(feature = "polars"))]
    {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Polars feature is required for LazyFrame filtering",
        ))
    }
}

/// Filter events by region of interest (LazyFrame compatible)
///
/// # Arguments
///
/// * `events_lf` - Input events as LazyFrame
/// * `x_min` - Minimum x coordinate (inclusive)
/// * `x_max` - Maximum x coordinate (inclusive)
/// * `y_min` - Minimum y coordinate (inclusive)
/// * `y_max` - Maximum y coordinate (inclusive)
///
/// # Returns
///
/// Filtered LazyFrame
#[pyfunction]
#[pyo3(name = "filter_by_roi")]
pub fn filter_by_roi_lf_py(
    events_lf: &Bound<'_, PyAny>,
    x_min: i64,
    x_max: i64,
    y_min: i64,
    y_max: i64,
) -> PyResult<PyObject> {
    #[cfg(feature = "polars")]
    {
        // Extract LazyFrame from Python
        let lazy_frame = crate::ev_core::python::extract_lazy_frame(events_lf)?;

        // Build spatial filter configuration
        let roi = RegionOfInterest::new(x_min as u16, x_max as u16, y_min as u16, y_max as u16)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid ROI: {}", e)))?;

        let config = FilterConfig::new().with_spatial_filter(SpatialFilter::from_roi(roi));

        // Apply spatial filter using Polars LazyFrame operations
        let filtered_lf = if let Some(spatial_filter) = &config.spatial_filter {
            super::spatial::apply_spatial_filter(lazy_frame, spatial_filter).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Spatial filter error: {}", e))
            })?
        } else {
            lazy_frame
        };

        // Convert back to Python LazyFrame
        crate::ev_core::python::lazy_frame_to_python(filtered_lf, events_lf.py())
    }

    #[cfg(not(feature = "polars"))]
    {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Polars feature is required for LazyFrame filtering",
        ))
    }
}

/// Filter events by polarity (LazyFrame compatible)
///
/// # Arguments
///
/// * `events_lf` - Input events as LazyFrame
/// * `polarity` - Polarity value(s) to keep
///
/// # Returns
///
/// Filtered LazyFrame
#[pyfunction]
#[pyo3(name = "filter_by_polarity")]
pub fn filter_by_polarity_lf_py(
    events_lf: &Bound<'_, PyAny>,
    polarity: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    #[cfg(feature = "polars")]
    {
        // Extract LazyFrame from Python
        let lazy_frame = crate::ev_core::python::extract_lazy_frame(events_lf)?;

        // Handle both single int and list of ints for polarity
        let polarity_values: Vec<i8> = if let Ok(single_val) = polarity.extract::<i64>() {
            vec![if single_val > 0 { 1 } else { -1 }]
        } else if let Ok(polarity_list) = polarity.extract::<Vec<i64>>() {
            polarity_list
                .into_iter()
                .map(|p| if p > 0 { 1i8 } else { -1i8 })
                .collect()
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "polarity must be an integer or list of integers",
            ));
        };

        let polarity_filter = PolarityFilter::from_values(polarity_values);
        let config = FilterConfig::new().with_polarity_filter(polarity_filter);

        // Apply polarity filter using Polars LazyFrame operations
        let filtered_lf = if let Some(polarity_filter) = &config.polarity_filter {
            super::polarity::apply_polarity_filter(lazy_frame, polarity_filter).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Polarity filter error: {}", e))
            })?
        } else {
            lazy_frame
        };

        // Convert back to Python LazyFrame
        crate::ev_core::python::lazy_frame_to_python(filtered_lf, events_lf.py())
    }

    #[cfg(not(feature = "polars"))]
    {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Polars feature is required for LazyFrame filtering",
        ))
    }
}

/// Remove hot pixels (LazyFrame compatible)
///
/// # Arguments
///
/// * `events_lf` - Input events as LazyFrame
/// * `threshold_percentile` - Percentile threshold for hot pixel detection
///
/// # Returns
///
/// Filtered LazyFrame
#[pyfunction]
#[pyo3(name = "filter_hot_pixels")]
pub fn filter_hot_pixels_lf_py(
    events_lf: &Bound<'_, PyAny>,
    threshold_percentile: Option<f64>,
) -> PyResult<PyObject> {
    #[cfg(feature = "polars")]
    {
        // Extract LazyFrame from Python
        let lazy_frame = crate::ev_core::python::extract_lazy_frame(events_lf)?;

        let percentile = threshold_percentile.unwrap_or(99.9);
        let hot_pixel_filter = HotPixelFilter::percentile(percentile);
        let config = FilterConfig::new().with_hot_pixel_filter(hot_pixel_filter);

        // Apply hot pixel filter using Polars LazyFrame operations
        let filtered_lf = if let Some(hot_pixel_filter) = &config.hot_pixel_filter {
            super::hot_pixel::apply_hot_pixel_filter(lazy_frame, hot_pixel_filter).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Hot pixel filter error: {}", e))
            })?
        } else {
            lazy_frame
        };

        // Convert back to Python LazyFrame
        crate::ev_core::python::lazy_frame_to_python(filtered_lf, events_lf.py())
    }

    #[cfg(not(feature = "polars"))]
    {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Polars feature is required for LazyFrame filtering",
        ))
    }
}

/// Remove noise events (LazyFrame compatible)
///
/// # Arguments
///
/// * `events_lf` - Input events as LazyFrame
/// * `method` - Noise filtering method
/// * `refractory_period_us` - Refractory period in microseconds
///
/// # Returns
///
/// Filtered LazyFrame
#[pyfunction]
#[pyo3(name = "filter_noise")]
pub fn filter_noise_lf_py(
    events_lf: &Bound<'_, PyAny>,
    method: Option<&str>,
    refractory_period_us: Option<f64>,
) -> PyResult<PyObject> {
    #[cfg(feature = "polars")]
    {
        // Extract LazyFrame from Python
        let lazy_frame = crate::ev_core::python::extract_lazy_frame(events_lf)?;

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

        // Apply denoise filter using Polars LazyFrame operations
        let filtered_lf = if let Some(denoise_filter) = &config.denoise_filter {
            super::denoise::apply_denoise_filter_polars(lazy_frame, denoise_filter).map_err(
                |e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Denoise filter error: {}",
                        e
                    ))
                },
            )?
        } else {
            lazy_frame
        };

        // Convert back to Python LazyFrame
        crate::ev_core::python::lazy_frame_to_python(filtered_lf, events_lf.py())
    }

    #[cfg(not(feature = "polars"))]
    {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Polars feature is required for LazyFrame filtering",
        ))
    }
}

/// Register all filtering functions with the Python module
pub fn register_filtering_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // LazyFrame-compatible functions (polars-first API)
    m.add_function(wrap_pyfunction!(filter_by_time_lf_py, m)?)?;
    m.add_function(wrap_pyfunction!(filter_by_roi_lf_py, m)?)?;
    m.add_function(wrap_pyfunction!(filter_by_polarity_lf_py, m)?)?;
    m.add_function(wrap_pyfunction!(filter_hot_pixels_lf_py, m)?)?;
    m.add_function(wrap_pyfunction!(filter_noise_lf_py, m)?)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_config_compilation() {
        // Test that we can create filter configurations
        let config = FilterConfig::new()
            .with_temporal_filter(TemporalFilter::time_window(0.1, 0.5))
            .with_spatial_filter(SpatialFilter::from_roi(
                RegionOfInterest::new(100, 500, 100, 400).unwrap(),
            ))
            .with_polarity_filter(PolarityFilter::from_values(vec![1]))
            .with_hot_pixel_filter(HotPixelFilter::percentile(99.9))
            .with_denoise_filter(DenoiseFilter::refractory(1000.0));

        // Verify configuration is valid
        assert!(config.validate().is_ok());
    }
}
