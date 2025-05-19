// Core event data structures and types
// This module defines the fundamental data structures for event-based vision

use candle_core::{DType, Device, Result, Tensor};
use lazy_static::lazy_static;
// Remove unused import

// Python bindings module
pub mod python;

lazy_static! {
    // Default device for tensor operations (CPU by default)
    pub static ref DEVICE: Device = Device::Cpu;
    // The user can override this with GPU if needed using features
}

/// Core event data structure.
/// Represents a single event from an event camera.
#[derive(Clone, Copy, Debug)]
pub struct Event {
    pub t: f64,       // timestamp (in seconds or microseconds, depending on dataset)
    pub x: u16,       // x coordinate (pixel column)
    pub y: u16,       // y coordinate (pixel row)
    pub polarity: i8, // +1 for positive (ON event), -1 for negative (OFF event)
}

/// A collection of events
pub type Events = Vec<Event>;

/// Create an empty list of events with a pre-allocated capacity
pub fn events_with_capacity(capacity: usize) -> Events {
    Events::with_capacity(capacity)
}

/// Converts a set of events into a block/tensor representation
pub fn events_to_tensor(events: &Events) -> Result<Tensor> {
    let n = events.len();

    if n == 0 {
        // Return an empty tensor with shape (0, 4)
        return Tensor::zeros((0, 4), DType::F32, &DEVICE);
    }

    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    let mut ts = Vec::with_capacity(n);
    let mut ps = Vec::with_capacity(n);

    for ev in events {
        xs.push(ev.x as f32);
        ys.push(ev.y as f32);
        ts.push(ev.t as f32);
        ps.push(ev.polarity as f32);
    }

    // Stack the arrays into a Nx4 tensor
    let xs = Tensor::from_vec(xs, (n, 1), &DEVICE)?;
    let ys = Tensor::from_vec(ys, (n, 1), &DEVICE)?;
    let ts = Tensor::from_vec(ts, (n, 1), &DEVICE)?;
    let ps = Tensor::from_vec(ps, (n, 1), &DEVICE)?;

    Tensor::cat(&[xs, ys, ts, ps], 1)
}

/// Convert Python event arrays into our internal Events type
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
            polarity: ps_array[i] as i8,
        });
    }

    events
}

/// Split events by polarity into positive and negative sets
pub fn split_by_polarity(events: &Events) -> (Events, Events) {
    let mut pos_events = Vec::new();
    let mut neg_events = Vec::new();

    for &ev in events {
        if ev.polarity >= 0 {
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
