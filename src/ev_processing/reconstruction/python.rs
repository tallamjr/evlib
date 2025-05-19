// Python bindings for event-based reconstruction

use super::e2vid::{E2Vid, E2VidConfig};
use crate::ev_core::Event;
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyList;

/// Python wrapper to create video reconstructions from events using E2VID
#[pyfunction]
#[pyo3(name = "events_to_video")]
pub fn events_to_video_py(
    xs: PyReadonlyArray1<i64>,
    ys: PyReadonlyArray1<i64>,
    ts: PyReadonlyArray1<f64>,
    ps: PyReadonlyArray1<i64>,
    height: usize,
    width: usize,
    num_bins: Option<usize>,
) -> PyResult<Py<PyArray3<f32>>> {
    let py = xs.py();

    // Convert arrays to native Rust types for easier processing
    let xs_array = xs.as_array();
    let ys_array = ys.as_array();
    let ts_array = ts.as_array();
    let ps_array = ps.as_array();

    // Create events vector from the arrays
    let mut events = Vec::with_capacity(xs_array.len());
    for i in 0..xs_array.len() {
        events.push(Event {
            x: xs_array[i] as u16,
            y: ys_array[i] as u16,
            t: ts_array[i],
            polarity: ps_array[i] as i8,
        });
    }

    // Sort events by timestamp
    events.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));

    // Create E2VID config with specified parameters
    let mut config = E2VidConfig::default();
    if let Some(bins) = num_bins {
        config.num_bins = bins;
    }

    // Create E2VID reconstructor
    let mut e2vid = E2Vid::with_config(height, width, config);

    // Process events
    let output = e2vid.process_events(&events).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Error processing events: {}", e))
    })?;

    // Convert tensor to numpy array
    let output_vec = output.to_vec2().map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Error converting output to vector: {}",
            e
        ))
    })?;

    // Reshape output to match image dimensions
    let mut frame = vec![0.0; height * width];
    for y in 0..height {
        for x in 0..width {
            frame[y * width + x] = output_vec[y][x];
        }
    }

    // Convert to numpy array (reshaping to 3D for RGB)
    let output_shape = [height, width, 1];
    let output_array = frame.into_pyarray(py).reshape(output_shape).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Error reshaping output array: {}",
            e
        ))
    })?;

    Ok(output_array.to_owned())
}

/// Python wrapper to reconstruct multiple frames from events
#[pyfunction]
#[pyo3(name = "reconstruct_events_to_frames")]
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_events_to_frames_py(
    xs: PyReadonlyArray1<i64>,
    ys: PyReadonlyArray1<i64>,
    ts: PyReadonlyArray1<f64>,
    ps: PyReadonlyArray1<i64>,
    height: usize,
    width: usize,
    num_frames: usize,
    num_bins: Option<usize>,
) -> PyResult<Py<PyList>> {
    let py = xs.py();

    // Convert arrays to native Rust types for easier processing
    let xs_array = xs.as_array();
    let ys_array = ys.as_array();
    let ts_array = ts.as_array();
    let ps_array = ps.as_array();

    // Create events vector from the arrays
    let mut events = Vec::with_capacity(xs_array.len());
    for i in 0..xs_array.len() {
        events.push(Event {
            x: xs_array[i] as u16,
            y: ys_array[i] as u16,
            t: ts_array[i],
            polarity: ps_array[i] as i8,
        });
    }

    // Sort events by timestamp
    events.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));

    // Create E2VID config with specified parameters
    let mut config = E2VidConfig::default();
    if let Some(bins) = num_bins {
        config.num_bins = bins;
    }

    // Create E2VID reconstructor
    let mut e2vid = E2Vid::with_config(height, width, config);

    // Get time range
    let t_min = events.first().map(|e| e.t).unwrap_or(0.0);
    let t_max = events.last().map(|e| e.t).unwrap_or(1.0);
    let time_step = (t_max - t_min) / (num_frames as f64);

    // Create empty list for frames
    let frames = PyList::empty(py);

    for i in 0..num_frames {
        // Define time window for this frame
        let t_end = t_min + time_step * ((i + 1) as f64);

        // Get events up to this time
        let frame_events = events
            .iter()
            .filter(|&e| e.t <= t_end)
            .cloned()
            .collect::<Vec<_>>();

        // Skip if no events
        if frame_events.is_empty() {
            let empty_shape = [height, width, 1];
            let empty_frame = vec![0.0; height * width]
                .into_pyarray(py)
                .reshape(empty_shape)?;
            frames.append(empty_frame)?;
            continue;
        }

        // Process events
        let output = e2vid.process_events(&frame_events).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Error processing events: {}",
                e
            ))
        })?;

        // Convert tensor to numpy array
        let output_vec = output.to_vec2().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Error converting output to vector: {}",
                e
            ))
        })?;

        // Reshape output to match image dimensions
        let mut frame = vec![0.0; height * width];
        for y in 0..height {
            for x in 0..width {
                frame[y * width + x] = output_vec[y][x];
            }
        }

        // Convert to numpy array (reshaping to 3D for RGB compatibility)
        let output_shape = [height, width, 1];
        let output_array = frame.into_pyarray(py).reshape(output_shape)?;

        // Add to list
        frames.append(output_array)?;
    }

    Ok(frames.into())
}
