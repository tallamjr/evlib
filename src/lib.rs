use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

// Core modules
pub mod ev_augmentation;
pub mod ev_core;
pub mod ev_formats;
pub mod ev_processing;
pub mod ev_representations;
pub mod ev_transforms;
pub mod ev_visualization;

// Re-export core types for easier usage
pub use ev_core::{Event, Events, DEVICE};

/// A Python module implemented in Rust for event camera processing
///
/// This library provides tools for working with event-based vision data,
/// including data loading, augmentation, representations, and visualization.
#[pymodule]
fn evlib(py: Python, m: &PyModule) -> PyResult<()> {
    // Register helper functions
    m.add_function(wrap_pyfunction!(version, py)?)?;

    // Register ev_core module as "core" in Python
    let core_submodule = PyModule::new(py, "core")?;
    core_submodule.add_function(wrap_pyfunction!(ev_core::python::events_to_block_py, py)?)?;
    core_submodule.add_function(wrap_pyfunction!(ev_core::python::merge_events, py)?)?;
    m.add_submodule(core_submodule)?;

    // Register ev_augmentation module as "augmentation" in Python
    let augmentation_submodule = PyModule::new(py, "augmentation")?;

    // Add random_events function
    augmentation_submodule.add_function(wrap_pyfunction!(
        ev_augmentation::python::add_random_events_py,
        py
    )?)?;

    // Add transform functions
    augmentation_submodule
        .add_function(wrap_pyfunction!(ev_transforms::add_correlated_events, py)?)?;
    augmentation_submodule.add_function(wrap_pyfunction!(ev_core::python::remove_events, py)?)?;
    augmentation_submodule.add_function(wrap_pyfunction!(ev_transforms::flip_events_x, py)?)?;
    augmentation_submodule.add_function(wrap_pyfunction!(ev_transforms::flip_events_y, py)?)?;
    augmentation_submodule
        .add_function(wrap_pyfunction!(ev_transforms::clip_events_to_bounds, py)?)?;
    augmentation_submodule.add_function(wrap_pyfunction!(ev_transforms::rotate_events, py)?)?;

    m.add_submodule(augmentation_submodule)?;

    // Register ev_representations module as "representations" in Python
    let representations_submodule = PyModule::new(py, "representations")?;
    representations_submodule.add_function(wrap_pyfunction!(
        ev_representations::python::events_to_voxel_grid_py,
        py
    )?)?;
    m.add_submodule(representations_submodule)?;

    // Register ev_formats module as "formats" in Python
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

    // Add the iterator classes
    formats_submodule.add_class::<ev_formats::python::PyEventFileIterator>()?;
    formats_submodule.add_class::<ev_formats::python::PyTimeWindowIter>()?;

    m.add_submodule(formats_submodule)?;

    // Register ev_visualization module as "visualization" in Python
    let viz_submodule = PyModule::new(py, "visualization")?;
    viz_submodule.add_function(wrap_pyfunction!(
        ev_visualization::python::draw_events_to_image_py,
        py
    )?)?;
    m.add_submodule(viz_submodule)?;

    // Register ev_processing module as "processing" in Python
    let processing_submodule = PyModule::new(py, "processing")?;
    processing_submodule.add_function(wrap_pyfunction!(
        ev_processing::reconstruction::python::events_to_video_py,
        py
    )?)?;
    processing_submodule.add_function(wrap_pyfunction!(
        ev_processing::reconstruction::python::reconstruct_events_to_frames_py,
        py
    )?)?;
    m.add_submodule(processing_submodule)?;

    // No legacy functionality - all functions are registered in their respective modules

    // Build info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}

/// Returns the version of the library
#[pyfunction]
fn version() -> PyResult<String> {
    Ok(env!("CARGO_PKG_VERSION").to_string())
}
