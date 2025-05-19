// Event augmentation module
// Provides tools to transform event streams for data augmentation

use crate::ev_core::{infer_resolution, Event, Events};
use rand::prelude::*;
use rand_distr::{Distribution, Normal};

/// Add N random noise events uniformly over the space (0..width, 0..height) and time range.
pub fn add_random_events(
    events: &mut Events,
    num_new: usize,
    width: u16,
    height: u16,
    t_range: (f64, f64),
) {
    let mut rng = rand::thread_rng();
    let (t_min, t_max) = t_range;
    for _ in 0..num_new {
        let x = rng.gen_range(0..width);
        let y = rng.gen_range(0..height);
        let t = rng.gen_range(t_min..t_max);
        let polarity = if rng.gen_bool(0.5) { 1 } else { -1 };
        events.push(Event { t, x, y, polarity });
    }
}

/// Remove N events at random from the event stream (in-place).
pub fn remove_events(events: &mut Events, num_remove: usize) {
    let mut rng = rand::thread_rng();
    let n = events.len();

    if num_remove >= n {
        events.clear();
        return;
    }

    // Use Fisher-Yates shuffle to select indices to remove
    let mut indices: Vec<usize> = (0..n).collect();
    for i in 0..num_remove {
        let j = rng.gen_range(i..n);
        indices.swap(i, j);
    }

    // Sort indices in descending order to avoid shifting problems when removing
    indices[0..num_remove].sort_by(|a, b| b.cmp(a));

    // Remove the selected indices
    for &idx in &indices[0..num_remove] {
        events.swap_remove(idx);
    }
}

/// Add N new events around existing events, using Gaussian offsets in space and time.
pub fn add_correlated_events(events: &mut Events, num_new: usize, sigma_xy: f32, sigma_t: f32) {
    let mut rng = rand::thread_rng();
    // Gaussian distributions for offsets (mean 0, given sigma)
    let gauss_xy = Normal::new(0.0, sigma_xy).unwrap();
    let gauss_t = Normal::new(0.0, sigma_t).unwrap();
    let orig_len = events.len();

    if orig_len == 0 {
        return; // No events to correlate with
    }

    // Infer sensor resolution as max bounds
    let resolution = infer_resolution(events);

    for _ in 0..num_new {
        // Pick a random original event as the base
        let &base_event = &events[rng.gen_range(0..orig_len)];

        // Sample offsets
        let dx = gauss_xy.sample(&mut rng);
        let dy = gauss_xy.sample(&mut rng);
        let dt = gauss_t.sample(&mut rng);

        // Compute new event attributes with bounds checking
        let new_x = (base_event.x as f32 + dx)
            .round()
            .clamp(0.0, (resolution.0 - 1) as f32);
        let new_y = (base_event.y as f32 + dy)
            .round()
            .clamp(0.0, (resolution.1 - 1) as f32);
        let new_t = base_event.t + dt as f64;

        // Add the new event
        events.push(Event {
            t: new_t,
            x: new_x as u16,
            y: new_y as u16,
            polarity: base_event.polarity,
        });
    }
}

/// Flip events horizontally (along the X axis).
pub fn flip_events_x(events: &mut Events, sensor_width: u16) {
    for e in events.iter_mut() {
        e.x = sensor_width - 1 - e.x;
    }
}

/// Flip events vertically (along the Y axis).
pub fn flip_events_y(events: &mut Events, sensor_height: u16) {
    for e in events.iter_mut() {
        e.y = sensor_height - 1 - e.y;
    }
}

/// Crop events to a rectangular region [x0,x1) x [y0,y1).
pub fn crop_events(events: &mut Events, x0: u16, y0: u16, x1: u16, y1: u16) {
    events.retain(|e| e.x >= x0 && e.x < x1 && e.y >= y0 && e.y < y1);

    // Optionally, shift coords to start at (0,0) if needed
    if x0 > 0 || y0 > 0 {
        for e in events.iter_mut() {
            e.x -= x0;
            e.y -= y0;
        }
    }
}

/// Rotate events by angle theta (in degrees) around the center (cx, cy).
pub fn rotate_events(
    events: &mut Events,
    theta_deg: f32,
    cx: f32,
    cy: f32,
    clip_to_bounds: bool,
    sensor_size: Option<(u16, u16)>,
) {
    let theta = theta_deg.to_radians();
    let (sin_t, cos_t) = theta.sin_cos();

    let sensor_bounds = match sensor_size {
        Some(size) => size,
        None => infer_resolution(events),
    };

    let (width, height) = (sensor_bounds.0, sensor_bounds.1);

    // Create temporary vector to hold rotated events
    let mut rotated_events = Vec::with_capacity(events.len());

    for &e in events.iter() {
        // Translate to origin (center at 0,0), apply rotation, translate back
        let xf = e.x as f32 - cx;
        let yf = e.y as f32 - cy;
        let x_rot = cos_t * xf - sin_t * yf + cx;
        let y_rot = sin_t * xf + cos_t * yf + cy;

        // Round to nearest integer pixel
        let x_new = x_rot.round() as i32;
        let y_new = y_rot.round() as i32;

        // Check if the rotated point is within bounds
        if !clip_to_bounds
            || (x_new >= 0 && x_new < width as i32 && y_new >= 0 && y_new < height as i32)
        {
            rotated_events.push(Event {
                t: e.t,
                x: x_new.clamp(0, (width - 1) as i32) as u16,
                y: y_new.clamp(0, (height - 1) as i32) as u16,
                polarity: e.polarity,
            });
        }
    }

    // Replace original events with rotated ones
    events.clear();
    events.extend(rotated_events);
}

/// Transform events using different warp models and parameters
pub mod warp {
    use super::*;

    /// Apply a linear velocity warp to events, shifting each event's
    /// position according to v * (t - t_ref)
    pub fn linear_velocity_warp(events: &mut Events, vx: f64, vy: f64, t_ref: f64) {
        for e in events.iter_mut() {
            let dt = e.t - t_ref;
            let dx = vx * dt;
            let dy = vy * dt;

            // Calculate new position
            let new_x = (e.x as f64 + dx).round();
            let new_y = (e.y as f64 + dy).round();

            // Clamp to valid u16 range
            e.x = new_x.clamp(0.0, u16::MAX as f64) as u16;
            e.y = new_y.clamp(0.0, u16::MAX as f64) as u16;
        }
    }
}

/// Python bindings for event augmentation
#[cfg(feature = "python")]
pub mod python {
    use super::*;
    use crate::ev_core::from_numpy_arrays;
    use numpy::{PyReadonlyArray1, ToPyArray};
    use pyo3::prelude::*;

    /// Add random events
    #[pyfunction]
    #[pyo3(name = "add_random_events")]
    #[pyo3(signature = (
        xs,
        ys,
        ts,
        ps,
        to_add,
        sensor_resolution = None,
        sort = true
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn add_random_events_py(
        py: Python<'_>,
        xs: PyReadonlyArray1<i64>,
        ys: PyReadonlyArray1<i64>,
        ts: PyReadonlyArray1<f64>,
        ps: PyReadonlyArray1<i64>,
        to_add: usize,
        sensor_resolution: Option<(i64, i64)>,
        sort: bool,
    ) -> PyResult<PyObject> {
        // Convert to our internal types
        let mut events = from_numpy_arrays(xs, ys, ts, ps);

        // Determine resolution
        let resolution = match sensor_resolution {
            Some((w, h)) => (w as u16, h as u16),
            None => {
                let res = infer_resolution(&events);
                (res.0, res.1)
            }
        };

        // Get time range
        let t_min = events
            .iter()
            .map(|e| e.t)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);
        let t_max = events
            .iter()
            .map(|e| e.t)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(1.0);

        // Add random events
        add_random_events(
            &mut events,
            to_add,
            resolution.0,
            resolution.1,
            (t_min, t_max),
        );

        // Sort if requested
        if sort {
            events.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));
        }

        // Convert back to numpy arrays
        let mut xs_vec = Vec::with_capacity(events.len());
        let mut ys_vec = Vec::with_capacity(events.len());
        let mut ts_vec = Vec::with_capacity(events.len());
        let mut ps_vec = Vec::with_capacity(events.len());

        for ev in &events {
            xs_vec.push(ev.x as i64);
            ys_vec.push(ev.y as i64);
            ts_vec.push(ev.t);
            ps_vec.push(ev.polarity as i64);
        }

        // Create numpy arrays
        let xs_array = numpy::ndarray::Array1::from(xs_vec);
        let ys_array = numpy::ndarray::Array1::from(ys_vec);
        let ts_array = numpy::ndarray::Array1::from(ts_vec);
        let ps_array = numpy::ndarray::Array1::from(ps_vec);

        // Convert to Python objects
        let xs_py = xs_array.to_pyarray(py).to_object(py);
        let ys_py = ys_array.to_pyarray(py).to_object(py);
        let ts_py = ts_array.to_pyarray(py).to_object(py);
        let ps_py = ps_array.to_pyarray(py).to_object(py);

        // Create result tuple
        let result = pyo3::types::PyTuple::new(py, &[xs_py, ys_py, ts_py, ps_py]);

        Ok(result.into())
    }
}
