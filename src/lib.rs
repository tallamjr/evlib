// Core modules (only working functionality)
pub mod ev_core;
pub mod ev_formats;
pub mod ev_representations;

// Removed modules with non-working implementations
// pub mod ev_processing;    // Removed - broken neural network implementations
// pub mod ev_visualization; // Removed - limited/broken functionality

// Re-export core types for easier usage
pub use ev_core::{Event, Events, DEVICE};

// Test modules
// #[cfg(test)]
// mod test_evt2_detection;
// #[cfg(test)]
// mod test_polarity_conversion;

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::wrap_pyfunction;

/// Minimal Python module with only working functionality
///
/// This library provides basic event camera data processing with focus on
/// working file loading and core functionality only.
#[cfg(feature = "python")]
#[pymodule]
fn evlib(py: Python, m: &PyModule) -> PyResult<()> {
    // Register helper functions
    m.add_function(wrap_pyfunction!(version, py)?)?;

    // Register ev_core module as "core" in Python
    let core_submodule = PyModule::new(py, "core")?;
    #[cfg(feature = "python")]
    {
        core_submodule.add_function(wrap_pyfunction!(ev_core::python::events_to_block_py, py)?)?;
        core_submodule.add_function(wrap_pyfunction!(ev_core::python::merge_events, py)?)?;
    }
    m.add_submodule(core_submodule)?;

    // Register ev_representations module as "representations" in Python
    let representations_submodule = PyModule::new(py, "representations")?;
    m.add_submodule(representations_submodule)?;

    // Register ev_formats module as "formats" in Python - CORE WORKING FUNCTIONALITY
    let formats_submodule = PyModule::new(py, "formats")?;
    formats_submodule.add_function(wrap_pyfunction!(ev_formats::python::load_events_py, py)?)?;

    formats_submodule.add_function(wrap_pyfunction!(
        ev_formats::python::save_events_to_hdf5_py,
        py
    )?)?;
    formats_submodule.add_function(wrap_pyfunction!(
        ev_formats::python::save_events_to_text_py,
        py
    )?)?;

    // Add format detection functions
    formats_submodule.add_function(wrap_pyfunction!(ev_formats::python::detect_format_py, py)?)?;
    formats_submodule.add_function(wrap_pyfunction!(
        ev_formats::python::get_format_description_py,
        py
    )?)?;

    m.add_submodule(formats_submodule)?;

    // Build info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}

/// Returns the version of the library
#[cfg(feature = "python")]
#[pyfunction]
fn version() -> PyResult<String> {
    Ok(env!("CARGO_PKG_VERSION").to_string())
}

/// Returns the version of the library (non-Python version)
#[cfg(not(feature = "python"))]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
