use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

// Core modules
pub mod augmentation;
// Contrast maximization module is disabled pending API updates in Rust code
// pub mod contrast_maximization;
pub mod data_formats;
pub mod events_core;
pub mod representations;
pub mod visualization;

// Legacy module for backward compatibility
mod transforms;

// Re-export core types for easier usage
pub use events_core::{Event, Events, DEVICE};

/// A Python module implemented in Rust for event camera processing
///
/// This library provides tools for working with event-based vision data,
/// including data loading, augmentation, representations, and visualization.
#[pymodule]
fn evlib(py: Python, m: &PyModule) -> PyResult<()> {
    // Register helper functions
    m.add_function(wrap_pyfunction!(version, py)?)?;

    // Register events_core module and functions
    let events_submodule = PyModule::new(py, "events")?;
    events_submodule.add_function(wrap_pyfunction!(
        events_core::python::events_to_block_py,
        py
    )?)?;
    events_submodule.add_function(wrap_pyfunction!(events_core::python::merge_events, py)?)?;
    m.add_submodule(events_submodule)?;

    // Register augmentation module and functions
    let augmentation_submodule = PyModule::new(py, "augmentation")?;

    // Add random_events function
    augmentation_submodule.add_function(wrap_pyfunction!(
        augmentation::python::add_random_events_py,
        py
    )?)?;

    // Add other augmentation functions from transforms
    augmentation_submodule.add_function(wrap_pyfunction!(add_correlated_events, py)?)?;
    augmentation_submodule.add_function(wrap_pyfunction!(remove_events_legacy, py)?)?;
    augmentation_submodule.add_function(wrap_pyfunction!(flip_events_x, py)?)?;
    augmentation_submodule.add_function(wrap_pyfunction!(flip_events_y, py)?)?;
    augmentation_submodule.add_function(wrap_pyfunction!(clip_events_to_bounds, py)?)?;
    augmentation_submodule.add_function(wrap_pyfunction!(rotate_events, py)?)?;

    m.add_submodule(augmentation_submodule)?;

    // Register representations module and functions
    let representations_submodule = PyModule::new(py, "representations")?;
    representations_submodule.add_function(wrap_pyfunction!(
        representations::python::events_to_voxel_grid_py,
        py
    )?)?;
    m.add_submodule(representations_submodule)?;

    // Register data_formats module and functions
    let formats_submodule = PyModule::new(py, "formats")?;
    formats_submodule.add_function(wrap_pyfunction!(data_formats::python::load_events_py, py)?)?;
    m.add_submodule(formats_submodule)?;

    // Register visualization module and functions
    let viz_submodule = PyModule::new(py, "visualization")?;
    viz_submodule.add_function(wrap_pyfunction!(
        visualization::python::draw_events_to_image_py,
        py
    )?)?;
    m.add_submodule(viz_submodule)?;

    // Register old functions directly at the root level for backwards compatibility
    // Import from the original modules
    use events_core::events_to_block;
    use events_core::python::add_random_events as add_random_events_legacy;
    use events_core::python::merge_events;
    use events_core::python::remove_events as remove_events_legacy;
    use transforms::{
        add_correlated_events, clip_events_to_bounds, flip_events_x, flip_events_y, rotate_events,
    };

    // Register legacy functions for backward compatibility
    m.add_function(wrap_pyfunction!(events_to_block, py)?)?;
    m.add_function(wrap_pyfunction!(merge_events, py)?)?;
    m.add_function(wrap_pyfunction!(add_random_events_legacy, py)?)?;
    m.add_function(wrap_pyfunction!(remove_events_legacy, py)?)?;
    m.add_function(wrap_pyfunction!(add_correlated_events, py)?)?;
    m.add_function(wrap_pyfunction!(flip_events_x, py)?)?;
    m.add_function(wrap_pyfunction!(flip_events_y, py)?)?;
    m.add_function(wrap_pyfunction!(clip_events_to_bounds, py)?)?;
    m.add_function(wrap_pyfunction!(rotate_events, py)?)?;

    // Build info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}

/// Returns the version of the library
#[pyfunction]
fn version() -> PyResult<String> {
    Ok(env!("CARGO_PKG_VERSION").to_string())
}
