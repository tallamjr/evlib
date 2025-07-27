// Core modules (only working functionality)
pub mod ev_core;
pub mod ev_formats;
pub mod ev_representations;

// Removed modules with non-working implementations
// pub mod ev_processing;    // Removed - broken neural network implementations
// pub mod ev_visualization; // Removed - limited/broken functionality

// Re-export core types for easier usage
pub use ev_core::{Event, Events};

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
fn evlib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register helper functions
    m.add_function(wrap_pyfunction!(version, m)?)?;

    // Add top-level load_events function (wrapper around formats.load_events)
    // PyO3 0.25 API compatible
    m.add_function(wrap_pyfunction!(ev_formats::python::load_events_py, m)?)?;

    // Add top-level detect_format function (wrapper around formats.detect_format)
    m.add_function(wrap_pyfunction!(ev_formats::python::detect_format_py, m)?)?;

    // Add top-level save functions (wrappers around formats functions)
    m.add_function(wrap_pyfunction!(
        ev_formats::python::save_events_to_hdf5_py,
        m
    )?)?;

    // Register ev_core module as "core" in Python
    let core_submodule = PyModule::new(m.py(), "core")?;
    // PyO3 0.25 API compatible
    #[cfg(feature = "python")]
    {
        core_submodule.add_function(wrap_pyfunction!(
            ev_core::python::events_to_block_py,
            &core_submodule
        )?)?;
        core_submodule.add_function(wrap_pyfunction!(
            ev_core::python::merge_events,
            &core_submodule
        )?)?;
    }
    m.add_submodule(&core_submodule)?;

    // Register ev_representations module as "representations" in Python
    let representations_submodule = PyModule::new(m.py(), "representations")?;

    // Add Rust-based representation functions
    #[cfg(feature = "python")]
    {
        representations_submodule.add_function(wrap_pyfunction!(
            ev_representations::python::create_stacked_histogram_py,
            &representations_submodule
        )?)?;
        representations_submodule.add_function(wrap_pyfunction!(
            ev_representations::python::create_mixed_density_stack_py,
            &representations_submodule
        )?)?;
        representations_submodule.add_function(wrap_pyfunction!(
            ev_representations::python::create_voxel_grid_py,
            &representations_submodule
        )?)?;

        // Add clean aliases without _py suffix for better API
        representations_submodule.add(
            "create_stacked_histogram",
            representations_submodule.getattr("create_stacked_histogram_py")?,
        )?;
        representations_submodule.add(
            "create_mixed_density_stack",
            representations_submodule.getattr("create_mixed_density_stack_py")?,
        )?;
        representations_submodule.add(
            "create_voxel_grid",
            representations_submodule.getattr("create_voxel_grid_py")?,
        )?;
    }

    m.add_submodule(&representations_submodule)?;

    // Register ev_formats module as "formats" in Python - CORE WORKING FUNCTIONALITY
    let formats_submodule = PyModule::new(m.py(), "formats")?;
    // PyO3 0.25 API compatible bindings
    formats_submodule.add_function(wrap_pyfunction!(
        ev_formats::python::load_events_py,
        &formats_submodule
    )?)?;

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

    // Add Apache Arrow integration functions (requires both python and arrow features)
    #[cfg(all(feature = "python", feature = "arrow"))]
    {
        formats_submodule.add_function(wrap_pyfunction!(
            ev_formats::python::load_events_to_pyarrow,
            &formats_submodule
        )?)?;
        formats_submodule.add_function(wrap_pyfunction!(
            ev_formats::python::pyarrow_to_events_py,
            &formats_submodule
        )?)?;
    }

    m.add_submodule(&formats_submodule)?;

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
