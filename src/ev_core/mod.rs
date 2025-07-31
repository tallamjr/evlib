// Core event data structures and types
// This module defines the fundamental data structures for event-based vision

use ndarray::Array2;
// Remove candle dependencies - replaced with ndarray

#[cfg(feature = "polars")]
use polars::prelude::*;

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
        let error_msg = &self.0;
        write!(f, "TensorError: {error_msg}")
    }
}

impl std::error::Error for TensorError {}

/// Core event data structure.
/// Represents a single event from an event camera.
#[derive(Clone, Copy, Debug, PartialEq)]
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
        Box::new(TensorError(format!("Shape error: {e}")))
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
    let n = xs.as_array().len();
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

/// Convert Events to Polars DataFrame for high-performance operations
#[cfg(feature = "polars")]
pub fn events_to_dataframe(events: &Events) -> PolarsResult<DataFrame> {
    use polars::prelude::*;

    let n = events.len();

    // Pre-allocate vectors
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    let mut ts = Vec::with_capacity(n);
    let mut ps = Vec::with_capacity(n);

    // Convert events to vectors
    for event in events {
        xs.push(event.x as i64);
        ys.push(event.y as i64);
        ts.push(event.t);
        ps.push(if event.polarity { 1i64 } else { 0i64 });
    }

    // Create Polars DataFrame
    DataFrame::new(vec![
        Series::new("x".into(), xs).into(),
        Series::new("y".into(), ys).into(),
        Series::new("t".into(), ts).into(),
        Series::new("polarity".into(), ps).into(),
    ])
}

/// Convert Events to Polars DataFrame and then to Python numpy arrays using Polars' efficient conversion
#[cfg(all(feature = "python", feature = "polars"))]
pub fn events_to_numpy_via_polars(
    events: &Events,
) -> pyo3::PyResult<(
    pyo3::PyObject,
    pyo3::PyObject,
    pyo3::PyObject,
    pyo3::PyObject,
)> {
    use pyo3::prelude::*;

    // Convert to Polars DataFrame
    let df = events_to_dataframe(events)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Polars error: {}", e)))?;

    Python::with_gil(|py| {
        // Use Polars' efficient to_numpy conversion
        let x_series = df.column("x").unwrap();
        let y_series = df.column("y").unwrap();
        let t_series = df.column("t").unwrap();
        let p_series = df.column("polarity").unwrap();

        // Convert each series to numpy array using fallback method
        // TODO: Update when to_numpy is available for newer Polars versions
        use numpy::IntoPyArray;
        let x_values: Vec<i64> = x_series
            .as_materialized_series()
            .i64()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Polars error: {}", e)))?
            .into_no_null_iter()
            .collect();
        let y_values: Vec<i64> = y_series
            .as_materialized_series()
            .i64()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Polars error: {}", e)))?
            .into_no_null_iter()
            .collect();
        let t_values: Vec<f64> = t_series
            .as_materialized_series()
            .f64()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Polars error: {}", e)))?
            .into_no_null_iter()
            .collect();
        let p_values: Vec<i64> = p_series
            .as_materialized_series()
            .i64()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Polars error: {}", e)))?
            .into_no_null_iter()
            .collect();

        let xs_numpy = x_values.into_pyarray(py).into();
        let ys_numpy = y_values.into_pyarray(py).into();
        let ts_numpy = t_values.into_pyarray(py).into();
        let ps_numpy = p_values.into_pyarray(py).into();

        Ok((xs_numpy, ys_numpy, ts_numpy, ps_numpy))
    })
}

/// Fallback manual conversion for when Polars feature is not available
#[cfg(all(feature = "python", not(feature = "polars")))]
pub fn events_to_numpy_fallback(
    events: &Events,
) -> (
    ndarray::Array1<i64>,
    ndarray::Array1<i64>,
    ndarray::Array1<f64>,
    ndarray::Array1<i64>,
) {
    let n = events.len();

    let mut xs = ndarray::Array1::<i64>::zeros(n);
    let mut ys = ndarray::Array1::<i64>::zeros(n);
    let mut ts = ndarray::Array1::<f64>::zeros(n);
    let mut ps = ndarray::Array1::<i64>::zeros(n);

    for (i, event) in events.iter().enumerate() {
        xs[i] = event.x as i64;
        ys[i] = event.y as i64;
        ts[i] = event.t;
        ps[i] = if event.polarity { 1 } else { 0 };
    }

    (xs, ys, ts, ps)
}

/// Primary function to convert Events to numpy arrays - uses Polars if available, fallback otherwise
#[cfg(feature = "python")]
pub fn to_numpy_arrays(
    events: &Events,
) -> Result<
    (
        pyo3::PyObject,
        pyo3::PyObject,
        pyo3::PyObject,
        pyo3::PyObject,
    ),
    Box<dyn std::error::Error>,
> {
    #[cfg(feature = "polars")]
    {
        events_to_numpy_via_polars(events).map_err(|e| e.into())
    }

    #[cfg(not(feature = "polars"))]
    {
        use numpy::IntoPyArray;
        let (xs, ys, ts, ps) = events_to_numpy_fallback(events);
        Python::with_gil(|py| {
            Ok((
                xs.into_pyarray(py).into(),
                ys.into_pyarray(py).into(),
                ts.into_pyarray(py).into(),
                ps.into_pyarray(py).into(),
            ))
        })
    }
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
