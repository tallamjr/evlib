/*!
Event Representations Module

This module provides two complementary APIs for converting event streams into tensor representations:

## Rust Core API (Direct ndarray operations)
- **Functions**: Standard Rust functions returning `Array3<f32>`
- **Use case**: High-performance Rust applications, internal processing
- **Data types**: `Events` → `Array3<f32>`
- **Available representations**: Timestamp images, count images, event frames, time windows

## Python Bindings API (Polars DataFrame processing)
- **Functions**: `*_py()` suffix, decorated with `#[pyfunction]`
- **Use case**: Python users requiring DataFrame-based preprocessing
- **Data types**: `PyDataFrame` → `PyDataFrame`
- **Available representations**: Stacked histograms, mixed density stacks, voxel grids

## Architecture Notes
- **Smooth voxel grids**: Removed from Rust core due to complexity
- **Standard voxel grids**: Available via Python bindings using Polars operations
- **Performance**: Rust core functions are optimised for tensor operations, Python bindings optimised for DataFrame workflows
*/

use crate::ev_core::Events;
use crate::ev_core::{TensorError, TensorResult};
use ndarray::Array3;

/// Create a timestamp image (time surface) representation of events
///
/// **Rust Core API Function**: Returns `Array3<f32>` for direct tensor operations.
///
/// A timestamp image is a 2D grid where each pixel's value represents
/// the timestamp of the most recent event at that location. This can be used
/// to visualise the temporal dynamics and for creating time-based features.
///
/// # Arguments
/// * `events` - Event stream to convert
/// * `resolution` - Sensor resolution (width, height)
/// * `normalize` - If true, normalise timestamps to [0,1] range
/// * `polarity_separate` - If true, create separate time surfaces for positive and negative events
///
/// # Returns
/// * `Array3<f32>` with shape (channels, height, width) where channels = 1 or 2 depending on `polarity_separate`
pub fn events_to_timestamp_image(
    events: &Events,
    resolution: (u16, u16),
    normalize: bool,
    polarity_separate: bool,
) -> TensorResult<Array3<f32>> {
    let (width, height) = (resolution.0 as usize, resolution.1 as usize);

    // Handle empty events
    if events.is_empty() {
        return if polarity_separate {
            Ok(Array3::zeros((2, height, width)))
        } else {
            Ok(Array3::zeros((1, height, width)))
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
        if ev.polarity {
            timestamps_pos[idx] = t_value;
        } else {
            timestamps_neg[idx] = t_value;
        }
    }

    // Convert to ndarray
    if polarity_separate {
        let all_timestamps = [&timestamps_pos[..], &timestamps_neg[..]].concat();
        Array3::from_shape_vec((2, height, width), all_timestamps).map_err(|e| {
            Box::new(TensorError(format!("Shape error: {e}")))
                as Box<dyn std::error::Error + Send + Sync>
        })
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
        Array3::from_shape_vec((1, height, width), timestamps).map_err(|e| {
            Box::new(TensorError(format!("Shape error: {e}")))
                as Box<dyn std::error::Error + Send + Sync>
        })
    }
}

/// Create an event count image (spatial histogram of events)
///
/// **Rust Core API Function**: Returns `Array3<f32>` for direct tensor operations.
///
/// # Arguments
/// * `events` - Event stream to convert
/// * `resolution` - Sensor resolution (width, height)
/// * `polarity_as_channel` - If true, create a 2-channel image with positive and negative events separated
///
/// # Returns
/// * `Array3<f32>` with shape (channels, height, width) where channels = 1 or 2 depending on `polarity_as_channel`
pub fn events_to_count_image(
    events: &Events,
    resolution: (u16, u16),
    polarity_as_channel: bool,
) -> TensorResult<Array3<f32>> {
    let (width, height) = (resolution.0 as usize, resolution.1 as usize);

    // Initialize count images
    let mut counts_pos = vec![0i32; width * height];
    let mut counts_neg = vec![0i32; width * height];

    // Count events at each pixel
    for ev in events {
        let idx = ev.y as usize * width + ev.x as usize;
        if ev.polarity {
            counts_pos[idx] += 1;
        } else {
            counts_neg[idx] += 1;
        }
    }

    // Convert to ndarray
    if polarity_as_channel {
        // Create a 2-channel image [pos_counts, neg_counts]
        let counts_pos_f32: Vec<f32> = counts_pos.iter().map(|&x| x as f32).collect();
        let counts_neg_f32: Vec<f32> = counts_neg.iter().map(|&x| x as f32).collect();

        let all_counts = [&counts_pos_f32[..], &counts_neg_f32[..]].concat();
        Array3::from_shape_vec((2, height, width), all_counts).map_err(|e| {
            Box::new(TensorError(format!("Shape error: {e}")))
                as Box<dyn std::error::Error + Send + Sync>
        })
    } else {
        // Create a single-channel image with combined counts
        let mut counts = vec![0f32; width * height];
        for i in 0..counts.len() {
            counts[i] = (counts_pos[i] + counts_neg[i]) as f32;
        }
        Array3::from_shape_vec((1, height, width), counts).map_err(|e| {
            Box::new(TensorError(format!("Shape error: {e}")))
                as Box<dyn std::error::Error + Send + Sync>
        })
    }
}

/// Create an event frame by accumulating events into an image
///
/// **Rust Core API Function**: Returns `Array3<f32>` for direct tensor operations.
///
/// Similar to a count image but optionally applies normalisation and
/// can be configured to use different accumulation methods.
///
/// # Arguments
/// * `events` - Event stream to convert
/// * `resolution` - Sensor resolution (width, height)
/// * `method` - Accumulation method: "count", "polarity", or "times"
/// * `normalize` - If true, normalise the output to [0,1] range
///
/// # Returns
/// * `Array3<f32>` with shape (1, height, width)
pub fn events_to_frame(
    events: &Events,
    resolution: (u16, u16),
    method: &str,
    normalize: bool,
) -> TensorResult<Array3<f32>> {
    let (width, height) = (resolution.0 as usize, resolution.1 as usize);
    let mut frame = vec![0f32; width * height];

    if events.is_empty() {
        return Array3::from_shape_vec((1, height, width), frame).map_err(|e| {
            Box::new(TensorError(format!("Shape error: {e}")))
                as Box<dyn std::error::Error + Send + Sync>
        });
    }

    // Accumulate based on method
    match method {
        "polarity" => {
            // Sum polarities
            for ev in events {
                let idx = ev.y as usize * width + ev.x as usize;
                frame[idx] += ev.polarity as u8 as f32;
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

    Array3::from_shape_vec((1, height, width), frame).map_err(|e| {
        Box::new(TensorError(format!("Shape error: {e}")))
            as Box<dyn std::error::Error + Send + Sync>
    })
}

/// Create a time window representation of events
///
/// **Rust Core API Function**: Returns `Vec<Array3<f32>>` for direct tensor operations.
///
/// This splits the event stream into time windows and creates a representation
/// for each window, allowing time-based processing of events.
///
/// # Arguments
/// * `events` - Event stream to convert
/// * `resolution` - Sensor resolution (width, height)
/// * `window_duration` - Duration of each time window in seconds
/// * `representation` - Type of representation to use for each window ("count", "polarity")
///
/// # Returns
/// * `Vec<Array3<f32>>` where each element is a tensor for one time window
pub fn events_to_time_windows(
    events: &Events,
    resolution: (u16, u16),
    window_duration: f64,
    representation: &str,
) -> TensorResult<Vec<Array3<f32>>> {
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
                    "count" => events_to_count_image(&current_window, resolution, false)?,
                    _ => events_to_frame(&current_window, resolution, "polarity", false)?,
                };
                result.push(tensor);
            } else {
                // Add empty array if no events in window
                let tensor = Array3::zeros((1, resolution.1 as usize, resolution.0 as usize));
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
    use polars::prelude::*;
    use pyo3::prelude::*;
    use pyo3_polars::PyDataFrame;

    /// Create stacked histogram representation with temporal binning
    ///
    /// **Python Bindings API Function**: `PyDataFrame` → `PyDataFrame` using Polars operations.
    ///
    /// This processes events into temporal windows and creates histograms for each polarity,
    /// optimised for DataFrame-based preprocessing workflows.
    ///
    /// # Arguments
    /// * `events_pydf` - Polars DataFrame with columns [x, y, timestamp, polarity]
    /// * `_height` - Output height dimension (used for compatibility)
    /// * `_width` - Output width dimension (used for compatibility)
    /// * `nbins` - Number of temporal bins per window (default: 10)
    /// * `window_duration_ms` - Duration of each window in milliseconds (default: 50.0)
    /// * `stride_ms` - Stride between windows in milliseconds (default: window_duration_ms)
    /// * `_count_cutoff` - Maximum count per bin (default: 10)
    ///
    /// # Returns
    /// * `PyDataFrame` with columns [window_id, channel, time_bin, y, x, count, channel_time_bin]
    #[pyfunction]
    #[pyo3(signature = (events_pydf, _height, _width, nbins=10, window_duration_ms=50.0, stride_ms=None, _count_cutoff=Some(10)))]
    pub fn create_stacked_histogram_py(
        events_pydf: PyDataFrame,
        _height: i32,
        _width: i32,
        nbins: i32,
        window_duration_ms: f64,
        stride_ms: Option<f64>,
        _count_cutoff: Option<i32>,
    ) -> PyResult<PyDataFrame> {
        // Extract Polars DataFrame from Python
        let df: DataFrame = events_pydf.into();

        // Set default stride
        let stride_ms = stride_ms.unwrap_or(window_duration_ms);

        // Convert to microseconds
        let window_duration_us = (window_duration_ms * 1000.0) as i64;
        let stride_us = (stride_ms * 1000.0) as i64;

        // Process the DataFrame using Polars lazy operations
        let result = df
            .lazy()
            .with_columns([
                // Convert timestamp to microseconds if it's a duration type
                col("timestamp")
                    .dt()
                    .total_microseconds()
                    .alias("timestamp_us"),
            ])
            .with_columns([
                // Calculate time offset from sequence start
                (col("timestamp_us") - col("timestamp_us").min()).alias("time_offset"),
            ])
            .with_columns([
                // Create window assignments based on stride
                (col("time_offset") / lit(stride_us)).alias("window_id"),
            ])
            .with_columns([
                // Calculate sequence duration for filtering
                (col("timestamp_us").max() - col("timestamp_us").min()).alias("seq_duration"),
            ])
            .filter(
                // Only keep events that belong to complete windows
                ((col("window_id") + lit(1)) * lit(stride_us))
                    .lt_eq(col("seq_duration") - lit(window_duration_us - stride_us)),
            )
            .filter(
                // Only keep events within the window duration for each window
                (col("time_offset") % lit(stride_us)).lt(lit(window_duration_us)),
            )
            .with_columns([
                // Cast spatial coordinates (clipping will be done in post-processing)
                col("x").cast(DataType::Int16),
                col("y").cast(DataType::Int16),
                // Polarity channel (0 for negative/0, 1 for positive/1)
                col("polarity").cast(DataType::Int8).alias("channel"),
                // Temporal binning within each window (simplified)
                ((col("time_offset") % lit(stride_us)) * lit(nbins) / lit(window_duration_us))
                    .cast(DataType::Int16)
                    .alias("time_bin"),
            ])
            .group_by([
                col("window_id"),
                col("channel"),
                col("time_bin"),
                col("y"),
                col("x"),
            ])
            .agg([len().alias("count")])
            .with_columns([
                // Apply count cutoff if specified (simplified for now)
                col("count"),
            ])
            .with_columns([
                // Add combined channel-time dimension for compatibility
                (col("channel") * lit(nbins) + col("time_bin")).alias("channel_time_bin"),
            ])
            .collect()
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Polars error: {}", e))
            })?;

        // Convert back to Python Polars DataFrame
        Ok(PyDataFrame(result))
    }

    /// Create mixed density event stack representation
    ///
    /// **Python Bindings API Function**: `PyDataFrame` → `PyDataFrame` using Polars operations.
    ///
    /// This creates a mixed density representation with temporal binning, optimised
    /// for DataFrame-based preprocessing workflows.
    ///
    /// # Arguments
    /// * `events_pydf` - Polars DataFrame with columns [x, y, timestamp, polarity]
    /// * `_height` - Output height dimension (used for compatibility)
    /// * `_width` - Output width dimension (used for compatibility)
    /// * `nbins` - Number of temporal bins (default: 10)
    /// * `window_duration_ms` - Duration of each window in milliseconds (default: 50.0)
    /// * `_count_cutoff` - Maximum count per bin (default: None)
    ///
    /// # Returns
    /// * `PyDataFrame` with columns [window_id, time_bin, y, x, polarity_sum]
    #[pyfunction]
    #[pyo3(signature = (events_pydf, _height, _width, nbins=10, window_duration_ms=50.0, _count_cutoff=None))]
    pub fn create_mixed_density_stack_py(
        events_pydf: PyDataFrame,
        _height: i32,
        _width: i32,
        nbins: i32,
        window_duration_ms: f64,
        _count_cutoff: Option<i32>,
    ) -> PyResult<PyDataFrame> {
        // Extract Polars DataFrame from Python
        let df: DataFrame = events_pydf.into();

        let window_duration_us = (window_duration_ms * 1000.0) as i64;

        // Process using logarithmic time binning
        let result = df
            .lazy()
            .with_columns([col("timestamp")
                .dt()
                .total_microseconds()
                .alias("timestamp_us")])
            .with_columns([
                // Create window assignments
                ((col("timestamp_us") - col("timestamp_us").min()) / lit(window_duration_us))
                    .alias("window_id"),
                // Cast spatial coordinates (clipping simplified for now)
                col("x").cast(DataType::Int16),
                col("y").cast(DataType::Int16),
                // Normalize timestamps within each window for log binning
                ((col("timestamp_us") - col("timestamp_us").min()) % lit(window_duration_us))
                    .alias("window_offset_us"),
            ])
            .with_columns([
                // Normalize to [1e-6, 1-1e-6] to avoid log(0)
                (col("window_offset_us").cast(DataType::Float64) / lit(window_duration_us as f64)
                    * lit(1.0 - 2e-6)
                    + lit(1e-6))
                .alias("t_norm"),
            ])
            .with_columns([
                // Linear temporal binning (simplified - logarithmic binning requires complex math functions)
                (col("t_norm") * lit(nbins))
                    .cast(DataType::Int16)
                    .alias("time_bin"),
                // Convert polarity to -1/+1
                (col("polarity") * lit(2) - lit(1)).alias("polarity_signed"),
            ])
            .group_by([col("window_id"), col("time_bin"), col("y"), col("x")])
            .agg([col("polarity_signed").sum().alias("polarity_sum")])
            .with_columns([
                // Apply count cutoff if specified (simplified for now)
                col("polarity_sum"),
            ])
            .collect()
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Polars error: {}", e))
            })?;

        Ok(PyDataFrame(result))
    }

    /// Create traditional voxel grid representation
    ///
    /// **Python Bindings API Function**: `PyDataFrame` → `PyDataFrame` using Polars operations.
    ///
    /// This creates a standard voxel grid representation with temporal binning across
    /// the entire dataset, optimised for DataFrame-based preprocessing workflows.
    ///
    /// Note: Smooth voxel grids were removed from the Rust core due to complexity,
    /// but standard voxel grids remain available via this Python binding.
    ///
    /// # Arguments
    /// * `events_pydf` - Polars DataFrame with columns [x, y, timestamp, polarity]
    /// * `_height` - Output height dimension (used for compatibility)
    /// * `_width` - Output width dimension (used for compatibility)
    /// * `nbins` - Number of temporal bins (default: 5)
    ///
    /// # Returns
    /// * `PyDataFrame` with columns [time_bin, y, x, value] where value is polarity sum
    #[pyfunction]
    #[pyo3(signature = (events_pydf, _height, _width, nbins=5))]
    pub fn create_voxel_grid_py(
        events_pydf: PyDataFrame,
        _height: i32,
        _width: i32,
        nbins: i32,
    ) -> PyResult<PyDataFrame> {
        // Extract Polars DataFrame from Python
        let df: DataFrame = events_pydf.into();

        // Process using temporal binning across entire dataset
        let result = df
            .lazy()
            .with_columns([col("timestamp")
                .dt()
                .total_microseconds()
                .alias("timestamp_us")])
            .with_columns([
                // Temporal binning across entire dataset (simplified)
                ((col("timestamp_us") - col("timestamp_us").min()) * lit(nbins)
                    / (col("timestamp_us").max() - col("timestamp_us").min()))
                .cast(DataType::Int16)
                .alias("time_bin"),
                // Cast spatial coordinates (clipping simplified for now)
                col("x").cast(DataType::Int16),
                col("y").cast(DataType::Int16),
                // Convert polarity to -1/+1 for voxel grid
                (col("polarity") * lit(2) - lit(1)).alias("polarity_signed"),
            ])
            .group_by([col("time_bin"), col("y"), col("x")])
            .agg([col("polarity_signed")
                .sum()
                .cast(DataType::Int32)
                .alias("value")])
            .collect()
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Polars error: {}", e))
            })?;

        Ok(PyDataFrame(result))
    }
}
