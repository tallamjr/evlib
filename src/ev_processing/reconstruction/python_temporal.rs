// Python bindings for temporal event reconstruction (E2VID+ and FireNet+)

use super::e2vid_plus::{E2VidPlus, FireNetPlus};
use crate::ev_core::Event;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use numpy::{IntoPyArray, PyArray4, PyReadonlyArray1};
use pyo3::prelude::*;

/// Python wrapper for E2VID+ temporal reconstruction
#[pyfunction]
#[pyo3(name = "events_to_video_temporal")]
#[allow(clippy::too_many_arguments)]
pub fn events_to_video_temporal_py(
    xs: PyReadonlyArray1<i64>,
    ys: PyReadonlyArray1<i64>,
    ts: PyReadonlyArray1<f64>,
    ps: PyReadonlyArray1<i64>,
    height: usize,
    width: usize,
    num_frames: usize,
    num_bins: Option<usize>,
    model_type: Option<&str>,
) -> PyResult<Py<PyArray4<f32>>> {
    let py = xs.py();
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);

    // Convert arrays to native Rust types
    let xs_array = xs.as_array();
    let ys_array = ys.as_array();
    let ts_array = ts.as_array();
    let ps_array = ps.as_array();

    // Create events vector
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

    // Get time range
    let t_min = events.first().map(|e| e.t).unwrap_or(0.0);
    let t_max = events.last().map(|e| e.t).unwrap_or(1.0);
    let time_step = (t_max - t_min) / (num_frames as f64);

    // Create voxel grids for each frame
    let mut voxel_tensors = Vec::new();
    let bins = num_bins.unwrap_or(5);

    for i in 0..num_frames {
        let t_start = t_min + time_step * (i as f64);
        let t_end = t_min + time_step * ((i + 1) as f64);

        // Get events in this time window
        let frame_events: Vec<&Event> = events
            .iter()
            .filter(|e| e.t >= t_start && e.t < t_end)
            .collect();

        // Create voxel grid for this frame
        let mut voxel_grid = vec![0.0f32; bins * height * width];

        for event in frame_events {
            let bin_idx = ((event.t - t_start) / (t_end - t_start) * (bins as f64)) as usize;
            let bin_idx = bin_idx.min(bins - 1);
            let idx = bin_idx * height * width + (event.y as usize) * width + (event.x as usize);

            if idx < voxel_grid.len() {
                voxel_grid[idx] += if event.polarity > 0 { 1.0 } else { -1.0 };
            }
        }

        // Convert to tensor
        let voxel_tensor =
            Tensor::from_vec(voxel_grid, &[bins, height, width], &device).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Error creating tensor: {}",
                    e
                ))
            })?;
        voxel_tensors.push(voxel_tensor);
    }

    // Stack all voxel grids
    let input_tensor = Tensor::stack(&voxel_tensors, 0).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Error stacking tensors: {}", e))
    })?;

    // Add batch dimension
    let input_tensor = input_tensor.unsqueeze(0).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Error adding batch dimension: {}",
            e
        ))
    })?;

    // Create model based on type
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let output = match model_type {
        Some("firenet_plus") | Some("firenet+") => {
            // Use FireNet+
            let model = FireNetPlus::new(vb, bins).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Error creating FireNet+: {}",
                    e
                ))
            })?;

            model.forward(&input_tensor).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Error in FireNet+ forward pass: {}",
                    e
                ))
            })?
        }
        _ => {
            // Default to E2VID+
            let base_channels = if model_type == Some("e2vid_plus_small") {
                16
            } else {
                32
            };
            let model = E2VidPlus::new(vb, bins, base_channels, num_frames).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Error creating E2VID+: {}",
                    e
                ))
            })?;

            model.forward(&input_tensor).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Error in E2VID+ forward pass: {}",
                    e
                ))
            })?
        }
    };

    // Convert output to numpy array
    // Output shape is (batch=1, seq_len, channels=1, height, width)
    let output = output.squeeze(0).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Error squeezing batch dimension: {}",
            e
        ))
    })?;

    let output_vec = output
        .flatten_all()
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Error flattening output: {}",
                e
            ))
        })?
        .to_vec1::<f32>()
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Error converting to vector: {}",
                e
            ))
        })?;

    // Reshape to (num_frames, height, width, 1)
    let output_shape = [num_frames, height, width, 1];
    let output_array = output_vec
        .into_pyarray(py)
        .reshape(output_shape)
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Error reshaping output array: {}",
                e
            ))
        })?;

    Ok(output_array.to_owned())
}
