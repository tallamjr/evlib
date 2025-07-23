// Representations module
// Converting event streams into different tensor representations for ML and visualization

// Smooth voxel grid module has been removed

use crate::ev_core::Events;
use crate::ev_core::{TensorError, TensorResult};
use ndarray::Array3;

// Voxel grid functionality has been removed

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
/// # Arguments
/// * `events` - Event stream to convert
/// * `resolution` - Sensor resolution (width, height)
/// * `polarity_as_channel` - If true, create a 2-channel image with positive and negative events separated
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
        Box::new(TensorError(format!("Shape error: {}", e)))
            as Box<dyn std::error::Error + Send + Sync>
    })
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
/// * `representation` - Type of representation to use for each window ("count", "polarity")
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
    // All Python bindings have been removed due to cleanup

    // Voxel grid Python binding has been removed

    // Smooth voxel grid Python binding has been removed
}
