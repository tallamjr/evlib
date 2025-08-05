// Core modules (only working functionality)
pub mod ev_augmentation;
pub mod ev_core;
pub mod ev_filtering;
pub mod ev_formats;
pub mod ev_representations;

// Tracing configuration for structured logging
pub mod tracing_config;

// Deep learning models are handled via Python interface in python/evlib/models/

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

    // Add top-level arrow functions (requires both python and arrow features)
    #[cfg(all(feature = "python", feature = "arrow"))]
    {
        m.add_function(wrap_pyfunction!(
            ev_formats::python::load_events_to_pyarrow,
            m
        )?)?;
    }

    // Add top-level save functions (wrappers around formats functions)
    #[cfg(not(windows))]
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
        representations_submodule.add_function(wrap_pyfunction!(
            ev_representations::python::create_enhanced_voxel_grid_py,
            &representations_submodule
        )?)?;
        representations_submodule.add_function(wrap_pyfunction!(
            ev_representations::python::create_enhanced_frame_py,
            &representations_submodule
        )?)?;
        representations_submodule.add_function(wrap_pyfunction!(
            ev_representations::python::create_timesurface_py,
            &representations_submodule
        )?)?;
        representations_submodule.add_function(wrap_pyfunction!(
            ev_representations::python::create_averaged_timesurface_py,
            &representations_submodule
        )?)?;
        representations_submodule.add_function(wrap_pyfunction!(
            ev_representations::python::create_bina_rep_py,
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
        representations_submodule.add(
            "create_enhanced_voxel_grid",
            representations_submodule.getattr("create_enhanced_voxel_grid_py")?,
        )?;
        representations_submodule.add(
            "create_enhanced_frame",
            representations_submodule.getattr("create_enhanced_frame_py")?,
        )?;
        representations_submodule.add(
            "create_timesurface",
            representations_submodule.getattr("create_timesurface_py")?,
        )?;
        representations_submodule.add(
            "create_averaged_timesurface",
            representations_submodule.getattr("create_averaged_timesurface_py")?,
        )?;
        representations_submodule.add(
            "create_bina_rep",
            representations_submodule.getattr("create_bina_rep_py")?,
        )?;
    }

    m.add_submodule(&representations_submodule)?;

    // Also add key representation functions to top-level module for convenience
    #[cfg(feature = "python")]
    {
        m.add_function(wrap_pyfunction!(
            ev_representations::python::create_stacked_histogram_py,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(
            ev_representations::python::create_mixed_density_stack_py,
            m
        )?)?;
        m.add_function(wrap_pyfunction!(
            ev_representations::python::create_voxel_grid_py,
            m
        )?)?;

        // Add clean aliases without _py suffix for top-level access
        m.add(
            "create_stacked_histogram",
            m.getattr("create_stacked_histogram_py")?,
        )?;
        m.add(
            "create_mixed_density_stack",
            m.getattr("create_mixed_density_stack_py")?,
        )?;
        m.add("create_voxel_grid", m.getattr("create_voxel_grid_py")?)?;
    }

    // Register ev_formats module as "formats" in Python - CORE WORKING FUNCTIONALITY
    let formats_submodule = PyModule::new(m.py(), "formats")?;
    // PyO3 0.25 API compatible bindings
    formats_submodule.add_function(wrap_pyfunction!(
        ev_formats::python::load_events_py,
        &formats_submodule
    )?)?;

    #[cfg(not(windows))]
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

    // Register tracing_config module for Python logging control
    let tracing_submodule = PyModule::new(m.py(), "tracing_config")?;
    #[cfg(feature = "python")]
    {
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

        // Functions are already exported with clean names via #[pyo3(name = "...")] attributes
        // No need for additional aliases
    }
    m.add_submodule(&tracing_submodule)?;

    // Register ev_filtering module as "filtering" in Python - NEW HIGH-PERFORMANCE FILTERING
    let filtering_submodule = PyModule::new(m.py(), "filtering")?;
    #[cfg(feature = "python")]
    {
        ev_filtering::python::register_filtering_functions(&filtering_submodule)?;
    }
    m.add_submodule(&filtering_submodule)?;

    // Also add filtering functions to top-level module for convenience
    #[cfg(feature = "python")]
    {
        ev_filtering::python::register_filtering_functions(m)?;
    }

    // Register ev_augmentation module as "ev_augmentation" in Python - NEW AUGMENTATION FUNCTIONALITY
    let augmentation_submodule = PyModule::new(m.py(), "ev_augmentation")?;
    #[cfg(feature = "python")]
    {
        ev_augmentation::python::register_augmentation_functions(&augmentation_submodule)?;
    }
    m.add_submodule(&augmentation_submodule)?;

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
