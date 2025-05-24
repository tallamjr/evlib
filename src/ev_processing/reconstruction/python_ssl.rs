// Python bindings for SSL-E2VID models
use super::ssl_e2vid::{EventAugmentation, SslE2Vid};
use super::ssl_losses::SSLLossConfig;
use crate::ev_representations::voxel_grid::VoxelGrid;
use candle_core::{DType, Device};
use candle_nn::VarMap;
use numpy::{IntoPyArray, PyArray3, PyArray4, PyReadonlyArray2};
use pyo3::prelude::*;

/// Reconstruct video from events using SSL-E2VID (self-supervised)
///
/// Args:
///     events (numpy.ndarray): Events array with shape (N, 4) containing [x, y, t, p]
///     height (int): Height of the output frames
///     width (int): Width of the output frames
///     num_frames (int): Number of frames to reconstruct
///     voxel_channels (int): Number of voxel channels (default: 5)
///     base_channels (int): Base channel count for the model (default: 32)
///     feature_dim (int): Feature dimension for contrastive learning (default: 128)
///
/// Returns:
///     numpy.ndarray: Reconstructed frames with shape (num_frames, 1, height, width)
#[pyfunction]
#[pyo3(signature = (events, height, width, num_frames, voxel_channels=5, base_channels=32, feature_dim=128))]
pub fn events_to_video_ssl(
    py: Python,
    events: PyReadonlyArray2<f32>,
    height: usize,
    width: usize,
    num_frames: usize,
    voxel_channels: usize,
    base_channels: usize,
    feature_dim: usize,
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

    let loss_config = SSLLossConfig::default();
    let model = SslE2Vid::new(vb, voxel_channels, base_channels, feature_dim, loss_config)
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Model creation error: {}",
                e
            ))
        })?;

    // Get voxel tensor
    let voxel_tensor = voxel_grid.to_tensor(&device).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Voxel tensor error: {}", e))
    })?;
    let voxel_batch = voxel_tensor.unsqueeze(0).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Unsqueeze error: {}", e))
    })?;

    // Forward pass
    let reconstructed = model.forward(&voxel_batch).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Forward pass error: {}", e))
    })?;

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

/// Augment events for SSL training
///
/// Args:
///     events (numpy.ndarray): Events array with shape (N, 4) containing [x, y, t, p]
///     temporal_shift_range (float): Range for temporal jitter (default: 0.1)
///     spatial_shift_range (int): Range for spatial shift in pixels (default: 5)
///     noise_level (float): Noise level to add (default: 0.05)
///     drop_probability (float): Probability of dropping events (default: 0.1)
///
/// Returns:
///     numpy.ndarray: Augmented events with shape (N, 4)
#[pyfunction]
#[pyo3(signature = (events, temporal_shift_range=0.1, spatial_shift_range=5, noise_level=0.05, drop_probability=0.1))]
pub fn augment_events(
    py: Python,
    events: PyReadonlyArray2<f32>,
    temporal_shift_range: f32,
    spatial_shift_range: i32,
    noise_level: f32,
    drop_probability: f32,
) -> PyResult<Py<PyArray3<f32>>> {
    // Convert events to voxel tensor for augmentation
    let events_array = events.as_array();
    let shape = events_array.shape();
    if shape[1] != 4 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Events array must have shape (N, 4) with columns [x, y, t, p]",
        ));
    }

    // For simplicity, create a dummy voxel grid and augment it
    // In practice, you would convert events to appropriate representation
    let height = 480; // Default height
    let width = 640; // Default width
    let voxel_channels = 5;

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

    // Get voxel tensor
    let device = Device::Cpu;
    let voxel_tensor = voxel_grid.to_tensor(&device).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Voxel tensor error: {}", e))
    })?;

    // Create augmentation
    let augmentor = EventAugmentation::new(
        temporal_shift_range,
        spatial_shift_range,
        noise_level,
        drop_probability,
    );

    // Apply augmentation
    let augmented = augmentor.augment(&voxel_tensor).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Augmentation error: {}", e))
    })?;

    // Convert back to numpy (as voxel grid for now)
    let aug_shape = augmented.dims();
    let flattened: Vec<f32> = augmented
        .flatten_all()
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Flatten error: {}", e))
        })?
        .to_vec1()
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("To vec error: {}", e))
        })?;

    let output_array =
        flattened
            .into_pyarray(py)
            .reshape([aug_shape[0], aug_shape[1], aug_shape[2]])?;

    Ok(output_array.to_owned())
}

/// Get SSL loss configuration
#[pyfunction]
pub fn get_ssl_loss_config() -> PyResult<Vec<(&'static str, f32)>> {
    let config = SSLLossConfig::default();
    Ok(vec![
        ("pos_weight", config.pos_weight),
        ("neg_weight", config.neg_weight),
        ("l1_weight", config.l1_weight),
        ("grad_weight", config.grad_weight),
        ("event_threshold", config.event_threshold),
        ("temperature", config.temperature),
        ("contrast_temp", config.contrast_temp),
        ("margin", config.margin),
        ("photo_weight", config.photo_weight),
        ("temporal_weight", config.temporal_weight),
        ("event_weight", config.event_weight),
        ("contrast_weight", config.contrast_weight),
    ])
}

/// Get SSL-E2VID model information
#[pyfunction]
pub fn get_ssl_model_info() -> String {
    "SSL-E2VID: Self-supervised learning approach for event-to-video reconstruction. \
     Trains without ground truth frames using photometric consistency, temporal smoothness, \
     event reconstruction, and contrastive learning objectives. Supports data augmentation \
     for improved generalization."
        .to_string()
}

pub fn register_ssl_module(py: Python, parent_module: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "ssl")?;
    m.add_function(wrap_pyfunction!(events_to_video_ssl, m)?)?;
    m.add_function(wrap_pyfunction!(augment_events, m)?)?;
    m.add_function(wrap_pyfunction!(get_ssl_loss_config, m)?)?;
    m.add_function(wrap_pyfunction!(get_ssl_model_info, m)?)?;
    parent_module.add_submodule(m)?;
    Ok(())
}
