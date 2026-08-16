// Core modules (only working functionality)
pub mod ev_formats;
pub mod ev_representations;
pub mod ev_simulation;

// Tracing configuration for structured logging
pub mod tracing_config;

// Deep learning models are handled via Python interface in python/evlib/models/

// numpy use removed due to unused warnings

// Python utility functions (previously in ev_core::python)
pub mod python {
    use pyo3::prelude::*;
    use pyo3::types::PyAny;

    pub fn extract_lazy_frame(py_obj: &Bound<'_, PyAny>) -> PyResult<polars::prelude::LazyFrame> {
        use polars::prelude::IntoLazy;
        use pyo3_polars::PyDataFrame;

        // Try to extract a DataFrame first and convert to LazyFrame
        if let Ok(pydf) = py_obj.extract::<PyDataFrame>() {
            return Ok(pydf.0.lazy());
        }

        // Try to call .lazy() method on the Python object if it's a DataFrame
        if let Ok(lazy_method) = py_obj.getattr("lazy") {
            if let Ok(lazy_result) = lazy_method.call0() {
                // Try to extract the resulting object as a DataFrame (might be a LazyFrame wrapper)
                if let Ok(pydf) = lazy_result.extract::<PyDataFrame>() {
                    return Ok(pydf.0.lazy());
                }
            }
        }

        Err(pyo3::exceptions::PyTypeError::new_err(
            "Expected a Polars DataFrame - ensure you're using pl.DataFrame(...)",
        ))
    }

    pub fn lazy_frame_to_python(
        lf: polars::prelude::LazyFrame,
        py: Python<'_>,
    ) -> PyResult<PyObject> {
        use pyo3::IntoPyObject;
        use pyo3_polars::PyDataFrame;

        // Convert LazyFrame to DataFrame and wrap in PyDataFrame
        let df = lf.collect().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to collect LazyFrame: {}", e))
        })?;
        let py_dataframe = PyDataFrame(df);
        Ok(py_dataframe.into_pyobject(py)?.into())
    }
}

// Test modules
// #[cfg(test)]
// mod test_evt2_detection;
// #[cfg(test)]
// mod test_polarity_conversion;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

/// Minimal Python module with only working functionality
///
/// This library provides basic event camera data processing with focus on
/// working file loading and core functionality only.
#[pymodule]
fn _evlib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register helper functions
    m.add_function(wrap_pyfunction!(version, m)?)?;

    // Add top-level load_events function (wrapper around formats.load_events)
    // PyO3 0.25 API compatible
    m.add_function(wrap_pyfunction!(ev_formats::python::load_events_py, m)?)?;

    // Add top-level detect_format function (wrapper around formats.detect_format)
    m.add_function(wrap_pyfunction!(ev_formats::python::detect_format_py, m)?)?;

    // Add top-level save functions (wrappers around formats functions)
    #[cfg(all(unix, feature = "hdf5"))]
    m.add_function(wrap_pyfunction!(
        ev_formats::python::save_events_to_hdf5_py,
        m
    )?)?;

    // Register legacy "core" module using migrated functions from ev_formats
    // These functions maintain backward compatibility for existing Python code
    let core_submodule = PyModule::new(m.py(), "core")?;
    core_submodule.add_function(wrap_pyfunction!(
        ev_formats::python::events_to_block_py,
        &core_submodule
    )?)?;
    core_submodule.add_function(wrap_pyfunction!(
        ev_formats::python::merge_events_py,
        &core_submodule
    )?)?;
    core_submodule.add_function(wrap_pyfunction!(
        ev_formats::python::add_random_events_py,
        &core_submodule
    )?)?;
    core_submodule.add_function(wrap_pyfunction!(
        ev_formats::python::remove_events_py,
        &core_submodule
    )?)?;
    m.add_submodule(&core_submodule)?;

    // Note: Representations module now implemented in Python (issue #34)
    // Rust representations module and PyO3 functions removed in favor of pure Python implementation

    // Register ev_formats module as "formats" in Python - CORE WORKING FUNCTIONALITY
    let formats_submodule = PyModule::new(m.py(), "formats")?;
    // PyO3 0.25 API compatible bindings
    formats_submodule.add_function(wrap_pyfunction!(
        ev_formats::python::load_events_py,
        &formats_submodule
    )?)?;

    #[cfg(all(unix, feature = "hdf5"))]
    formats_submodule.add_function(wrap_pyfunction!(
        ev_formats::python::save_events_to_hdf5_py,
        &formats_submodule
    )?)?;
    formats_submodule.add_function(wrap_pyfunction!(
        ev_formats::python::save_events_to_text_py,
        &formats_submodule
    )?)?;

    // Add format detection functions
    formats_submodule.add_function(wrap_pyfunction!(
        ev_formats::python::detect_format_py,
        &formats_submodule
    )?)?;
    formats_submodule.add_function(wrap_pyfunction!(
        ev_formats::python::get_format_description_py,
        &formats_submodule
    )?)?;

    // Add ECF testing function
    formats_submodule.add_function(wrap_pyfunction!(
        ev_formats::python::test_prophesee_ecf_decode_py,
        &formats_submodule
    )?)?;

    m.add_submodule(&formats_submodule)?;

    // Register tracing_config module for Python logging control
    let tracing_submodule = PyModule::new(m.py(), "tracing_config")?;
    tracing_submodule.add_function(wrap_pyfunction!(
        tracing_config::python::init_py,
        &tracing_submodule
    )?)?;
    tracing_submodule.add_function(wrap_pyfunction!(
        tracing_config::python::init_debug_py,
        &tracing_submodule
    )?)?;
    tracing_submodule.add_function(wrap_pyfunction!(
        tracing_config::python::init_with_filter_py,
        &tracing_submodule
    )?)?;
    tracing_submodule.add_function(wrap_pyfunction!(
        tracing_config::python::init_production_py,
        &tracing_submodule
    )?)?;
    tracing_submodule.add_function(wrap_pyfunction!(
        tracing_config::python::init_development_py,
        &tracing_submodule
    )?)?;
    m.add_submodule(&tracing_submodule)?;

    // Register ev_representations module as "representations_rs" in Python - dense scatter-add engine.
    // Named "_rs" to avoid colliding with the pure-Python evlib.representations module.
    let representations_submodule = PyModule::new(m.py(), "representations_rs")?;
    ev_representations::python::register_representations_functions(&representations_submodule)?;
    m.add_submodule(&representations_submodule)?;

    // Build info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}

/// Returns the version of the library
#[pyfunction]
fn version() -> PyResult<String> {
    Ok(env!("CARGO_PKG_VERSION").to_string())
}
