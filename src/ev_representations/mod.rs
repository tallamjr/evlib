// Representations module
// Converting event streams into different tensor representations for ML and visualization

pub mod voxel_grid;

use crate::ev_core::{Events, DEVICE};
use candle_core::{DType, Result, Tensor};

/// Create a voxel grid representation of events
///
/// Voxel grids divide events into time bins, creating a 3D tensor (bins, height, width)
/// where each bin represents events occurring during a specific time slice.
///
/// # Arguments
/// * `events` - Event stream to convert
/// * `resolution` - Sensor resolution (width, height)
/// * `num_bins` - Number of time bins to divide events into
/// * `voxel_method` - Method to accumulate events:
///   - "count" - Count events in each voxel
///   - "binary" - 1 if any event in voxel, 0 otherwise
///   - "polarity" - Sum polarities in each voxel (positive/negative events can cancel)
///   - "polaritySeparate" - Create 2*num_bins with positive and negative events separated
pub fn events_to_voxel_grid(
    events: &Events,
    resolution: (u16, u16),
    num_bins: u32,
    voxel_method: &str,
) -> Result<Tensor> {
    let (width, height) = (resolution.0 as usize, resolution.1 as usize);

    // Handle empty events case
    if events.is_empty() {
        return if voxel_method == "polaritySeparate" {
            Tensor::zeros((2 * num_bins as usize, height, width), DType::F32, &DEVICE)
        } else {
            Tensor::zeros((num_bins as usize, height, width), DType::F32, &DEVICE)
        };
    }

    // Determine time range
    let t_min = events.first().unwrap().t;
    let t_max = events.last().unwrap().t;
    let dt = t_max - t_min + f64::EPSILON; // Total time span

    let mut grid = if voxel_method == "polaritySeparate" {
        vec![0f32; 2 * num_bins as usize * height * width]
    } else {
        vec![0f32; num_bins as usize * height * width]
    };

    // Process each event
    for ev in events {
        // Calculate which bin this event belongs to
        let bin = (((ev.t - t_min) / dt) * (num_bins as f64)) as usize;
        let bin_index = bin.min(num_bins as usize - 1); // Ensure last event goes into last bin

        // For polaritySeparate, separate positive and negative events
        let bin_offset = if voxel_method == "polaritySeparate" {
            if ev.polarity > 0 {
                bin_index
            } else {
                bin_index + num_bins as usize
            }
        } else {
            bin_index
        };

        // Calculate index in the flat grid
        let idx = bin_offset * (width * height) + (ev.y as usize * width + ev.x as usize);

        // Update the grid based on the method
        match voxel_method {
            "binary" => {
                grid[idx] = 1.0; // Mark presence
            }
            "polarity" => {
                grid[idx] += ev.polarity as f32; // Sum polarities
            }
            "polaritySeparate" => {
                grid[idx] += 1.0; // Count events separately by polarity
            }
            _ => {
                // Default "count"
                grid[idx] += 1.0; // Count events
            }
        }
    }

    // Convert to Candle Tensor
    let dims = if voxel_method == "polaritySeparate" {
        (2 * num_bins as usize, height, width)
    } else {
        (num_bins as usize, height, width)
    };
    Tensor::from_vec(grid, dims, &DEVICE)
}

/// Create a timestamp image (time surface) representation of events
///
/// A timestamp image is a 2D grid where each pixel's value represents
/// the timestamp of the most recent event at that location. This can be used
/// to visualize the temporal dynamics and for creating time-based features.
///
/// # Arguments
/// * `events` - Event stream to convert
/// * `resolution` - Sensor resolution (width, height)
/// * `normalize` - If true, normalize timestamps to [0,1] range
/// * `polarity_separate` - If true, create separate time surfaces for positive and negative events
pub fn events_to_timestamp_image(
    events: &Events,
    resolution: (u16, u16),
    normalize: bool,
    polarity_separate: bool,
) -> Result<Tensor> {
    let (width, height) = (resolution.0 as usize, resolution.1 as usize);

    // Handle empty events
    if events.is_empty() {
        return if polarity_separate {
            Tensor::zeros((2, height, width), DType::F32, &DEVICE)
        } else {
            Tensor::zeros((1, height, width), DType::F32, &DEVICE)
        };
    }

    // Initialize timestamp images
    let mut timestamps_pos = vec![0f32; width * height];
    let mut timestamps_neg = vec![0f32; width * height];

    // Find time range for normalization
    let t_min = if normalize {
        events.first().unwrap().t as f32
    } else {
        0.0
    };

    let t_max = if normalize {
        events.last().unwrap().t as f32
    } else {
        1.0
    };

    let t_range = t_max - t_min;

    // Process each event
    for ev in events {
        let idx = ev.y as usize * width + ev.x as usize;
        let t_value = if normalize {
            ((ev.t as f32 - t_min) / t_range).clamp(0.0, 1.0)
        } else {
            ev.t as f32
        };

        // Update the appropriate timestamp image
        if ev.polarity > 0 {
            timestamps_pos[idx] = t_value;
        } else {
            timestamps_neg[idx] = t_value;
        }
    }

    // Convert to tensor
    if polarity_separate {
        let all_timestamps = [&timestamps_pos[..], &timestamps_neg[..]].concat();
        Tensor::from_vec(all_timestamps, (2, height, width), &DEVICE)
    } else {
        // Combine both polarities, taking the most recent timestamp
        let mut timestamps = vec![0f32; width * height];
        for i in 0..timestamps.len() {
            timestamps[i] = if timestamps_pos[i] >= timestamps_neg[i] {
                timestamps_pos[i]
            } else {
                timestamps_neg[i]
            };
        }
        Tensor::from_vec(timestamps, (1, height, width), &DEVICE)
    }
}

/// Create an event count image (spatial histogram of events)
///
/// # Arguments
/// * `events` - Event stream to convert
/// * `resolution` - Sensor resolution (width, height)
/// * `polarity_as_channel` - If true, create a 2-channel image with positive and negative events separated
pub fn events_to_count_image(
    events: &Events,
    resolution: (u16, u16),
    polarity_as_channel: bool,
) -> Result<Tensor> {
    let (width, height) = (resolution.0 as usize, resolution.1 as usize);

    // Initialize count images
    let mut counts_pos = vec![0i32; width * height];
    let mut counts_neg = vec![0i32; width * height];

    // Count events at each pixel
    for ev in events {
        let idx = ev.y as usize * width + ev.x as usize;
        if ev.polarity > 0 {
            counts_pos[idx] += 1;
        } else {
            counts_neg[idx] += 1;
        }
    }

    // Convert to tensor
    if polarity_as_channel {
        // Create a 2-channel image [pos_counts, neg_counts]
        let counts_pos_f32: Vec<f32> = counts_pos.iter().map(|&x| x as f32).collect();
        let counts_neg_f32: Vec<f32> = counts_neg.iter().map(|&x| x as f32).collect();

        let all_counts = [&counts_pos_f32[..], &counts_neg_f32[..]].concat();
        Tensor::from_vec(all_counts, (2, height, width), &DEVICE)
    } else {
        // Create a single-channel image with combined counts
        let mut counts = vec![0f32; width * height];
        for i in 0..counts.len() {
            counts[i] = (counts_pos[i] + counts_neg[i]) as f32;
        }
        Tensor::from_vec(counts, (1, height, width), &DEVICE)
    }
}

/// Create an event frame by accumulating events into an image
///
/// Similar to a count image but optionally applies normalization and
/// can be configured to use different accumulation methods.
///
/// # Arguments
/// * `events` - Event stream to convert
/// * `resolution` - Sensor resolution (width, height)
/// * `method` - Accumulation method: "count", "polarity", or "times"
/// * `normalize` - If true, normalize the output to [0,1] range
pub fn events_to_frame(
    events: &Events,
    resolution: (u16, u16),
    method: &str,
    normalize: bool,
) -> Result<Tensor> {
    let (width, height) = (resolution.0 as usize, resolution.1 as usize);
    let mut frame = vec![0f32; width * height];

    if events.is_empty() {
        return Tensor::from_vec(frame, (1, height, width), &DEVICE);
    }

    // Accumulate based on method
    match method {
        "polarity" => {
            // Sum polarities
            for ev in events {
                let idx = ev.y as usize * width + ev.x as usize;
                frame[idx] += ev.polarity as f32;
            }
        }
        "times" => {
            // Normalize timestamps to [0,1] and use as pixel intensities
            let t_min = events.first().unwrap().t;
            let t_max = events.last().unwrap().t;
            let t_range = t_max - t_min;

            if t_range > 0.0 {
                for ev in events {
                    let idx = ev.y as usize * width + ev.x as usize;
                    let t_norm = ((ev.t - t_min) / t_range) as f32;
                    // Use the most recent event's normalized time
                    frame[idx] = t_norm;
                }
            }
        }
        _ => {
            // "count" (default)
            // Count events
            for ev in events {
                let idx = ev.y as usize * width + ev.x as usize;
                frame[idx] += 1.0;
            }
        }
    };

    // Normalize if requested
    if normalize && !events.is_empty() {
        let max_val = frame.iter().fold(0.0f32, |a, &b| a.max(b));
        if max_val > 0.0 {
            for val in &mut frame {
                *val /= max_val;
            }
        }
    }

    Tensor::from_vec(frame, (1, height, width), &DEVICE)
}

/// Create a time window representation of events
///
/// This splits the event stream into time windows and creates a representation
/// for each window, allowing time-based processing of events.
///
/// # Arguments
/// * `events` - Event stream to convert
/// * `resolution` - Sensor resolution (width, height)
/// * `window_duration` - Duration of each time window in seconds
/// * `representation` - Type of representation to use for each window ("voxel", "count", "polarity")
pub fn events_to_time_windows(
    events: &Events,
    resolution: (u16, u16),
    window_duration: f64,
    representation: &str,
) -> Result<Vec<Tensor>> {
    if events.is_empty() {
        return Ok(Vec::new());
    }

    // Determine time range
    let t_min = events.first().unwrap().t;
    let t_max = events.last().unwrap().t;
    let total_duration = t_max - t_min;

    // Calculate number of windows
    let num_windows = (total_duration / window_duration).ceil() as usize;
    let mut result = Vec::with_capacity(num_windows);

    // Split events into time windows
    let mut current_window = Vec::new();
    let mut current_end_time = t_min + window_duration;

    let mut event_index = 0;
    while event_index < events.len() {
        let event = &events[event_index];

        if event.t <= current_end_time {
            // Event belongs to current window
            current_window.push(*event);
            event_index += 1;
        } else {
            // Process current window
            if !current_window.is_empty() {
                let tensor = match representation {
                    "voxel" => events_to_voxel_grid(&current_window, resolution, 5, "count")?,
                    "count" => events_to_count_image(&current_window, resolution, false)?,
                    _ => events_to_frame(&current_window, resolution, "polarity", false)?,
                };
                result.push(tensor);
            } else {
                // Add empty tensor if no events in window
                let tensor = match representation {
                    "voxel" => Tensor::zeros(
                        (5, resolution.1 as usize, resolution.0 as usize),
                        DType::F32,
                        &DEVICE,
                    )?,
                    _ => Tensor::zeros(
                        (1, resolution.1 as usize, resolution.0 as usize),
                        DType::F32,
                        &DEVICE,
                    )?,
                };
                result.push(tensor);
            }

            // Start new window
            current_window.clear();
            current_end_time += window_duration;
        }
    }

    // Process final window if not empty
    if !current_window.is_empty() {
        let tensor = match representation {
            "voxel" => events_to_voxel_grid(&current_window, resolution, 5, "count")?,
            "count" => events_to_count_image(&current_window, resolution, false)?,
            _ => events_to_frame(&current_window, resolution, "polarity", false)?,
        };
        result.push(tensor);
    }

    Ok(result)
}

/// Python bindings for the representations module
#[cfg(feature = "python")]
pub mod python {
    use super::*;
    use crate::ev_core::from_numpy_arrays;
    use numpy::{IntoPyArray, PyReadonlyArray1};
    use pyo3::prelude::*;

    /// Python binding for voxel grid conversion
    #[pyfunction]
    #[pyo3(name = "events_to_voxel_grid")]
    #[allow(clippy::too_many_arguments)]
    pub fn events_to_voxel_grid_py(
        py: Python<'_>,
        xs: PyReadonlyArray1<i64>,
        ys: PyReadonlyArray1<i64>,
        ts: PyReadonlyArray1<f64>,
        ps: PyReadonlyArray1<i64>,
        num_bins: usize,
        resolution: Option<(i64, i64)>,
        voxel_method: Option<&str>,
    ) -> PyResult<PyObject> {
        // Convert to our internal types
        let events = from_numpy_arrays(xs, ys, ts, ps);

        // Determine resolution
        let res = match resolution {
            Some((w, h)) => (w as u16, h as u16),
            None => {
                let max_x = events.iter().map(|e| e.x).max().unwrap_or(0) + 1;
                let max_y = events.iter().map(|e| e.y).max().unwrap_or(0) + 1;
                (max_x, max_y)
            }
        };

        // Use default method if not specified
        let method = voxel_method.unwrap_or("count");

        // Create voxel grid
        let voxel_tensor = events_to_voxel_grid(&events, res, num_bins as u32, method)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;

        // Convert to numpy array
        let shape = voxel_tensor.shape();
        let data: Vec<f32> = voxel_tensor
            .to_vec1()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;

        // Create numpy array with correct shape
        let dims = shape.dims();
        let array = numpy::ndarray::Array::from_shape_vec(
            (dims[0] as usize, dims[1] as usize, dims[2] as usize),
            data,
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", e)))?;

        Ok(array.into_pyarray(py).to_object(py))
    }
}
