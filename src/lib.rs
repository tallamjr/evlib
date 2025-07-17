// Core modules
pub mod ev_augmentation;
pub mod ev_core;
pub mod ev_formats;
pub mod ev_processing;
pub mod ev_representations;
pub mod ev_simulation;
pub mod ev_transforms;
pub mod ev_visualization;

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

/// A Python module implemented in Rust for event camera processing
///
/// This library provides tools for working with event-based vision data,
/// including data loading, augmentation, representations, and visualization.
#[cfg(feature = "python")]
#[pymodule]
fn evlib(py: Python, m: &PyModule) -> PyResult<()> {
    // Register helper functions
    m.add_function(wrap_pyfunction!(version, py)?)?;

    // Voxel grid functions have been removed

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
    // Voxel grid functions have been removed
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

    // Add format detection functions
    formats_submodule.add_function(wrap_pyfunction!(ev_formats::python::detect_format_py, py)?)?;
    formats_submodule.add_function(wrap_pyfunction!(
        ev_formats::python::get_format_description_py,
        py
    )?)?;

    m.add_submodule(formats_submodule)?;

    // Register ev_visualization module as "visualization" in Python
    let viz_submodule = PyModule::new(py, "visualization")?;
    viz_submodule.add_function(wrap_pyfunction!(
        ev_visualization::python::draw_events_to_image_py,
        py
    )?)?;

    // Add web server functionality
    viz_submodule.add_function(wrap_pyfunction!(
        ev_visualization::web_server::python::create_web_server,
        py
    )?)?;
    viz_submodule.add_function(wrap_pyfunction!(
        ev_visualization::web_server::python::create_web_server_config,
        py
    )?)?;
    viz_submodule.add_class::<ev_visualization::web_server::python::PyWebServerConfig>()?;
    viz_submodule.add_class::<ev_visualization::web_server::python::PyEventWebServer>()?;

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
    processing_submodule.add_function(wrap_pyfunction!(
        ev_processing::reconstruction::python::events_to_video_advanced_py,
        py
    )?)?;
    processing_submodule.add_function(wrap_pyfunction!(
        ev_processing::reconstruction::python_temporal::events_to_video_temporal_py,
        py
    )?)?;

    // Add E2Vid class for direct access to Rust implementation
    processing_submodule.add_class::<ev_processing::reconstruction::e2vid::E2Vid>()?;

    // SPADE and SSL models are now accessible through the unified Python API
    // in the evlib.models module (evlib.models.SPADE and evlib.models.SSL)
    // The separate Python bindings have been deprecated in favor of the unified interface

    // Register model zoo functions
    #[cfg(feature = "python")]
    {
        processing_submodule.add_function(wrap_pyfunction!(
            ev_processing::model_zoo::python::list_available_models,
            py
        )?)?;
        processing_submodule.add_function(wrap_pyfunction!(
            ev_processing::model_zoo::python::download_model,
            py
        )?)?;
        processing_submodule.add_function(wrap_pyfunction!(
            ev_processing::model_zoo::python::get_model_info_py,
            py
        )?)?;
    }

    m.add_submodule(processing_submodule)?;

    // Register ev_processing::streaming module as "streaming" in Python
    let streaming_submodule = PyModule::new(py, "streaming")?;
    streaming_submodule.add_function(wrap_pyfunction!(
        ev_processing::streaming::python::create_streaming_config,
        py
    )?)?;
    streaming_submodule.add_function(wrap_pyfunction!(
        ev_processing::streaming::python::process_events_streaming,
        py
    )?)?;
    streaming_submodule.add_class::<ev_processing::streaming::python::PyStreamingConfig>()?;
    streaming_submodule.add_class::<ev_processing::streaming::python::PyStreamingStats>()?;
    streaming_submodule.add_class::<ev_processing::streaming::python::PyStreamingProcessor>()?;
    streaming_submodule.add_class::<ev_processing::streaming::python::PyEventStream>()?;

    // Enhanced streaming functions will be available through Python utilities
    // TODO: Add Rust streaming functions once compilation issues are resolved

    m.add_submodule(streaming_submodule)?;

    // Register ev_simulation module as "simulation" in Python
    let simulation_submodule = PyModule::new(py, "simulation")?;
    simulation_submodule.add_function(wrap_pyfunction!(
        ev_simulation::python::video_to_events_py,
        simulation_submodule
    )?)?;
    simulation_submodule.add_function(wrap_pyfunction!(
        ev_simulation::python::esim_simulate_py,
        simulation_submodule
    )?)?;

    // Add real-time streaming functions (conditionally available with GStreamer)
    simulation_submodule.add_function(wrap_pyfunction!(
        ev_simulation::python::is_realtime_available,
        simulation_submodule
    )?)?;

    #[cfg(feature = "gstreamer")]
    {
        simulation_submodule.add_function(wrap_pyfunction!(
            ev_simulation::python::create_realtime_stream_py,
            simulation_submodule
        )?)?;
        simulation_submodule.add_class::<ev_simulation::python::PyRealtimeStreamConfig>()?;
        simulation_submodule.add_class::<ev_simulation::python::PyRealtimeEventStream>()?;
        simulation_submodule.add_class::<ev_simulation::python::PyStreamingStats>()?;
    }

    #[cfg(not(feature = "gstreamer"))]
    {
        simulation_submodule.add_function(wrap_pyfunction!(
            ev_simulation::python::create_realtime_stream_py,
            simulation_submodule
        )?)?;
    }

    simulation_submodule.add_class::<ev_simulation::python::PySimulationConfig>()?;
    simulation_submodule.add_class::<ev_simulation::python::PyVideoToEventsConverter>()?;
    simulation_submodule.add_class::<ev_simulation::python::PySimulationStats>()?;
    m.add_submodule(simulation_submodule)?;

    // Register ev_visualization module as "visualization" in Python
    let visualization_submodule = PyModule::new(py, "visualization")?;
    visualization_submodule.add_function(wrap_pyfunction!(
        ev_visualization::python::draw_events_to_image_py,
        py
    )?)?;
    visualization_submodule
        .add_class::<ev_visualization::python::PyRealtimeVisualizationConfig>()?;
    visualization_submodule
        .add_class::<ev_visualization::python::PyEventVisualizationPipeline>()?;

    // Terminal visualization (optional)
    #[cfg(feature = "terminal")]
    {
        visualization_submodule.add_function(wrap_pyfunction!(
            ev_visualization::python::create_terminal_event_viewer,
            py
        )?)?;
        visualization_submodule
            .add_class::<ev_visualization::python::PyTerminalVisualizationConfig>()?;
        visualization_submodule
            .add_class::<ev_visualization::python::PyTerminalEventVisualizer>()?;
    }

    m.add_submodule(visualization_submodule)?;

    // No legacy functionality - all functions are registered in their respective modules

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
