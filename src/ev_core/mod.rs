// Core event data structures and types
// This module defines the fundamental data structures for event-based vision

use ndarray::Array2;
// Remove candle dependencies - replaced with ndarray

// Python bindings module (optional)
#[cfg(feature = "python")]
pub mod python;

// Tensor result type (replacing candle's Result)
pub type TensorResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

// Simple error type for tensor operations
#[derive(Debug)]
pub struct TensorError(pub String);

impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TensorError: {}", self.0)
    }
}

impl std::error::Error for TensorError {}

/// Core event data structure.
/// Represents a single event from an event camera.
#[derive(Clone, Copy, Debug)]
pub struct Event {
    pub t: f64,         // timestamp (in seconds or microseconds, depending on dataset)
    pub x: u16,         // x coordinate (pixel column)
    pub y: u16,         // y coordinate (pixel row)
    pub polarity: bool, // true for positive (ON event), false for negative (OFF event)
}

/// A collection of events
pub type Events = Vec<Event>;

/// Create an empty list of events with a pre-allocated capacity
pub fn events_with_capacity(capacity: usize) -> Events {
    Events::with_capacity(capacity)
}

/// Converts a set of events into a block/array representation
pub fn events_to_tensor(events: &Events) -> TensorResult<Array2<f32>> {
    let n = events.len();

    if n == 0 {
        // Return an empty array with shape (0, 4)
        return Ok(Array2::zeros((0, 4)));
    }

    let mut data = Vec::with_capacity(n * 4);

    for ev in events {
        data.extend_from_slice(&[
            ev.x as f32,
            ev.y as f32,
            ev.t as f32,
            if ev.polarity { 1.0 } else { 0.0 },
        ]);
    }

    // Create Nx4 array from the data
    Array2::from_shape_vec((n, 4), data).map_err(|e| {
        Box::new(TensorError(format!("Shape error: {}", e)))
            as Box<dyn std::error::Error + Send + Sync>
    })
}

/// Convert Python event arrays into our internal Events type
#[cfg(feature = "python")]
pub fn from_numpy_arrays(
    xs: numpy::PyReadonlyArray1<i64>,
    ys: numpy::PyReadonlyArray1<i64>,
    ts: numpy::PyReadonlyArray1<f64>,
    ps: numpy::PyReadonlyArray1<i64>,
) -> Events {
    let n = xs.len();
    let mut events = Events::with_capacity(n);

    let xs_array = xs.as_array();
    let ys_array = ys.as_array();
    let ts_array = ts.as_array();
    let ps_array = ps.as_array();

    for i in 0..n {
        events.push(Event {
            x: xs_array[i] as u16,
            y: ys_array[i] as u16,
            t: ts_array[i],
            polarity: ps_array[i] > 0,
        });
    }

    events
}

/// Split events by polarity into positive and negative sets
pub fn split_by_polarity(events: &Events) -> (Events, Events) {
    let mut pos_events = Vec::new();
    let mut neg_events = Vec::new();

    for &ev in events {
        if ev.polarity {
            pos_events.push(ev);
        } else {
            neg_events.push(ev);
        }
    }

    (pos_events, neg_events)
}

/// Merge multiple sets of events into a single chronologically sorted list
pub fn merge_events(event_sets: &[Events]) -> Events {
    // Calculate total capacity needed
    let total_capacity = event_sets.iter().map(|events| events.len()).sum();

    // Merge all events into one vector
    let mut merged = Events::with_capacity(total_capacity);
    for events in event_sets {
        merged.extend_from_slice(events);
    }

    // Sort by timestamp
    merged.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));

    merged
}

/// Filter events by time range [t_min, t_max]
pub fn filter_by_time(events: &Events, t_min: f64, t_max: f64) -> Events {
    events
        .iter()
        .filter(|&e| e.t >= t_min && e.t <= t_max)
        .copied()
        .collect()
}

/// Compute the bounding box of the events: (min_x, min_y, max_x, max_y)
pub fn bounding_box(events: &Events) -> Option<(u16, u16, u16, u16)> {
    if events.is_empty() {
        return None;
    }

    let mut min_x = u16::MAX;
    let mut min_y = u16::MAX;
    let mut max_x = 0;
    let mut max_y = 0;

    for ev in events {
        min_x = min_x.min(ev.x);
        min_y = min_y.min(ev.y);
        max_x = max_x.max(ev.x);
        max_y = max_y.max(ev.y);
    }

    Some((min_x, min_y, max_x, max_y))
}

/// Infer sensor resolution (width, height) from events
pub fn infer_resolution(events: &Events) -> (u16, u16) {
    match bounding_box(events) {
        Some((_, _, max_x, max_y)) => (max_x + 1, max_y + 1),
        None => (0, 0),
    }
}
