// Python bindings for SPADE-E2VID models
use super::spade_e2vid::{HybridSpadeE2Vid, SpadeE2Vid, SpadeE2VidLite};
use crate::ev_representations::voxel_grid::VoxelGrid;
use candle_core::{DType, Device};
use candle_nn::VarMap;
use numpy::{IntoPyArray, PyArray4, PyReadonlyArray2};
use pyo3::prelude::*;

/// Reconstruct video from events using SPADE-E2VID
///
/// Args:
///     events (numpy.ndarray): Events array with shape (N, 4) containing [x, y, t, p]
///     height (int): Height of the output frames
///     width (int): Width of the output frames
///     num_frames (int): Number of frames to reconstruct
///     model_type (str): Model variant: "spade", "hybrid", or "lite"
///     voxel_channels (int): Number of voxel channels (default: 5)
///     base_channels (int): Base channel count for the model (default: 64)
///     use_skip_connections (bool): Whether to use skip connections (default: True)
///
/// Returns:
///     numpy.ndarray: Reconstructed frames with shape (num_frames, 1, height, width)
#[pyfunction]
#[pyo3(signature = (events, height, width, num_frames, model_type="spade", voxel_channels=5, base_channels=64, use_skip_connections=true))]
pub fn events_to_video_spade(
    py: Python,
    events: PyReadonlyArray2<f32>,
    height: usize,
    width: usize,
    num_frames: usize,
    model_type: &str,
    voxel_channels: usize,
    base_channels: usize,
    use_skip_connections: bool,
) -> PyResult<Py<PyArray4<f32>>> {
    // Convert events to tensor
    let events_array = events.as_array();
    let shape = events_array.shape();
    if shape[1] != 4 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Events array must have shape (N, 4) with columns [x, y, t, p]",
        ));
    }

    // Create voxel grid from events
    let mut voxel_grid = VoxelGrid::new(voxel_channels, height, width);

    // Process events into voxel grid
    for i in 0..shape[0] {
        let x = events_array[[i, 0]] as usize;
        let y = events_array[[i, 1]] as usize;
        let t = events_array[[i, 2]];
        let p = events_array[[i, 3]];

        if x < width && y < height {
            voxel_grid.add_event(x, y, t, p > 0.0);
        }
    }

    // Create model
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // Helper function to handle model forward pass
    let process_model = |model: &dyn candle_core::Module| -> PyResult<candle_core::Tensor> {
        let voxel_tensor = voxel_grid.to_tensor(&device).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Voxel tensor error: {}", e))
        })?;
        let voxel_batch = voxel_tensor.unsqueeze(0).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Unsqueeze error: {}", e))
        })?;

        model.forward(&voxel_batch).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Forward pass error: {}", e))
        })
    };

    // Select model variant and run
    let reconstructed = match model_type {
        "hybrid" => {
            let model = HybridSpadeE2Vid::new(vb, voxel_channels, base_channels).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Model creation error: {}",
                    e
                ))
            })?;
            process_model(&model)?
        }
        "lite" => {
            let model = SpadeE2VidLite::new(
                vb,
                voxel_channels,
                base_channels / 2, // Lite uses fewer channels
            )
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Model creation error: {}",
                    e
                ))
            })?;
            process_model(&model)?
        }
        _ => {
            // Default to standard SPADE
            let model = SpadeE2Vid::new(vb, voxel_channels, base_channels, use_skip_connections)
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Model creation error: {}",
                        e
                    ))
                })?;
            process_model(&model)?
        }
    };

    // Convert to numpy array
    let output_shape = reconstructed.dims();
    let flattened: Vec<f32> = reconstructed
        .flatten_all()
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Flatten error: {}", e))
        })?
        .to_vec1()
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("To vec error: {}", e))
        })?;

    // Create numpy array
    let output_array = flattened.into_pyarray(py).reshape([
        output_shape[0],
        output_shape[1],
        output_shape[2],
        output_shape[3],
    ])?;

    Ok(output_array.to_owned())
}

/// Get available SPADE model variants
#[pyfunction]
pub fn get_spade_models() -> Vec<&'static str> {
    vec!["spade", "hybrid", "lite"]
}

/// Get SPADE model information
#[pyfunction]
pub fn get_spade_model_info(model_type: &str) -> PyResult<String> {
    let info = match model_type {
        "spade" => {
            "SPADE-E2VID: Full model with spatial adaptive normalization for high-quality reconstruction. \
             Best quality but higher computational cost."
        },
        "hybrid" => {
            "Hybrid SPADE-E2VID: Combines SPADE with standard normalization for balanced performance. \
             Good quality with moderate computational cost."
        },
        "lite" => {
            "SPADE-E2VID Lite: Lightweight variant with fewer channels and simplified architecture. \
             Fastest inference with acceptable quality."
        },
        _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Unknown model type: {}. Available: spade, hybrid, lite", model_type)
        )),
    };

    Ok(info.to_string())
}

/// Initialize SPADE model weights from file (placeholder for future implementation)
#[pyfunction]
pub fn load_spade_weights(_model_type: &str, _weights_path: &str) -> PyResult<()> {
    // Placeholder for loading pre-trained weights
    Ok(())
}

pub fn register_spade_module(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "spade")?;
    m.add_function(wrap_pyfunction!(events_to_video_spade, m)?)?;
    m.add_function(wrap_pyfunction!(get_spade_models, m)?)?;
    m.add_function(wrap_pyfunction!(get_spade_model_info, m)?)?;
    m.add_function(wrap_pyfunction!(load_spade_weights, m)?)?;
    parent_module.add_submodule(m)?;
    Ok(())
}
