// Python bindings for ev_core

use super::{from_numpy_arrays, Events};
use ndarray::Array1;
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use rand::prelude::*;

/// Convert events to a block representation
#[pyfunction]
#[pyo3(name = "events_to_block")]
pub fn events_to_block_py(
    py: Python<'_>,
    xs: PyReadonlyArray1<i64>,
    ys: PyReadonlyArray1<i64>,
    ts: PyReadonlyArray1<f64>,
    ps: PyReadonlyArray1<i64>,
) -> PyResult<PyObject> {
    let events = from_numpy_arrays(xs, ys, ts, ps);

    // Create a 2D array with shape (n, 4)
    let n = events.len();
    let mut block = ndarray::Array2::<f64>::zeros((n, 4));

    // Fill in the values
    for (i, ev) in events.iter().enumerate() {
        block[[i, 0]] = ev.x as f64;
        block[[i, 1]] = ev.y as f64;
        block[[i, 2]] = ev.t;
        block[[i, 3]] = ev.polarity as f64;
    }

    Ok(block.into_pyarray(py).to_object(py))
}

/// Parameters for adding random events to an event stream
pub struct AddRandomEventsParams {
    pub to_add: usize,
    pub sensor_resolution: Option<(i64, i64)>,
    pub sort: bool,
    pub return_merged: bool,
}

impl Default for AddRandomEventsParams {
    fn default() -> Self {
        Self {
            to_add: 0,
            sensor_resolution: None,
            sort: true,
            return_merged: true,
        }
    }
}

/// Add random events drawn from a uniform distribution
#[pyfunction]
#[pyo3(name = "add_random_events")]
#[pyo3(signature = (xs, ys, ts, ps, to_add, sensor_resolution=None, sort=true, return_merged=true))]
#[allow(clippy::too_many_arguments)]
pub fn add_random_events(
    py: Python<'_>,
    xs: PyReadonlyArray1<i64>,
    ys: PyReadonlyArray1<i64>,
    ts: PyReadonlyArray1<f64>,
    ps: PyReadonlyArray1<i64>,
    to_add: usize,
    sensor_resolution: Option<(i64, i64)>,
    sort: bool,
    return_merged: bool,
) -> PyResult<PyObject> {
    let params = AddRandomEventsParams {
        to_add,
        sensor_resolution,
        sort,
        return_merged,
    };
    add_random_events_impl(py, xs, ys, ts, ps, params)
}

fn add_random_events_impl(
    py: Python<'_>,
    xs: PyReadonlyArray1<i64>,
    ys: PyReadonlyArray1<i64>,
    ts: PyReadonlyArray1<f64>,
    ps: PyReadonlyArray1<i64>,
    params: AddRandomEventsParams,
) -> PyResult<PyObject> {
    let xs_array = xs.as_array();
    let ys_array = ys.as_array();
    let ts_array = ts.as_array();
    let ps_array = ps.as_array();

    // Generate random events
    let max_x = match params.sensor_resolution {
        Some((w, _)) => w - 1,
        None => xs_array.fold(0, |acc, &x| acc.max(x)),
    };

    let max_y = match params.sensor_resolution {
        Some((_, h)) => h - 1,
        None => ys_array.fold(0, |acc, &y| acc.max(y)),
    };

    let mut rng = thread_rng();

    let mut xs_new = Vec::with_capacity(params.to_add);
    let mut ys_new = Vec::with_capacity(params.to_add);
    let mut ts_new = Vec::with_capacity(params.to_add);
    let mut ps_new = Vec::with_capacity(params.to_add);

    let min_ts = ts_array.fold(f64::INFINITY, |acc, &t| acc.min(t));
    let max_ts = ts_array.fold(f64::NEG_INFINITY, |acc, &t| acc.max(t));

    for _ in 0..params.to_add {
        xs_new.push(rng.gen_range(0..=max_x));
        ys_new.push(rng.gen_range(0..=max_y));
        ts_new.push(rng.gen_range(min_ts..=max_ts));
        ps_new.push(if rng.gen_bool(0.5) { 1 } else { -1 });
    }

    if params.return_merged {
        // Merge both arrays
        let mut all_xs = Vec::with_capacity(xs_array.len() + xs_new.len());
        let mut all_ys = Vec::with_capacity(ys_array.len() + ys_new.len());
        let mut all_ts: Vec<f64> = Vec::with_capacity(ts_array.len() + ts_new.len());
        let mut all_ps = Vec::with_capacity(ps_array.len() + ps_new.len());

        all_xs.extend(xs_array.iter());
        all_xs.extend(xs_new.iter());

        all_ys.extend(ys_array.iter());
        all_ys.extend(ys_new.iter());

        all_ts.extend(ts_array.iter());
        all_ts.extend(ts_new.iter());

        all_ps.extend(ps_array.iter());
        all_ps.extend(ps_new.iter());

        let merged_xs = Array1::from(all_xs);
        let merged_ys = Array1::from(all_ys);
        let merged_ts = Array1::from(all_ts);
        let merged_ps = Array1::from(all_ps);

        if params.sort {
            // Sort by timestamp
            let mut indices: Vec<usize> = (0..merged_ts.len()).collect();
            indices.sort_by(|&i, &j| merged_ts[i].partial_cmp(&merged_ts[j]).unwrap());

            let sorted_xs = indices
                .iter()
                .map(|&i| merged_xs[i])
                .collect::<Array1<i64>>();
            let sorted_ys = indices
                .iter()
                .map(|&i| merged_ys[i])
                .collect::<Array1<i64>>();
            let sorted_ts = indices
                .iter()
                .map(|&i| merged_ts[i])
                .collect::<Array1<f64>>();
            let sorted_ps = indices
                .iter()
                .map(|&i| merged_ps[i])
                .collect::<Array1<i64>>();

            // Convert arrays to Python objects
            let xs_py = sorted_xs.into_pyarray(py).to_object(py);
            let ys_py = sorted_ys.into_pyarray(py).to_object(py);
            let ts_py = sorted_ts.into_pyarray(py).to_object(py);
            let ps_py = sorted_ps.into_pyarray(py).to_object(py);

            // Create result tuple
            let result = PyTuple::new(py, &[xs_py, ys_py, ts_py, ps_py]);

            Ok(result.into())
        } else {
            // Convert arrays to Python objects without sorting
            let xs_py = merged_xs.into_pyarray(py).to_object(py);
            let ys_py = merged_ys.into_pyarray(py).to_object(py);
            let ts_py = merged_ts.into_pyarray(py).to_object(py);
            let ps_py = merged_ps.into_pyarray(py).to_object(py);

            // Create result tuple
            let result = PyTuple::new(py, &[xs_py, ys_py, ts_py, ps_py]);

            Ok(result.into())
        }
    } else {
        // Return only the new events
        let xs_new_array = Array1::from(xs_new);
        let ys_new_array = Array1::from(ys_new);
        let ts_new_array = Array1::from(ts_new);
        let ps_new_array = Array1::from(ps_new);

        if params.sort {
            // Sort by timestamp
            let mut indices: Vec<usize> = (0..ts_new_array.len()).collect();
            indices.sort_by(|&i, &j| ts_new_array[i].partial_cmp(&ts_new_array[j]).unwrap());

            let sorted_xs = indices
                .iter()
                .map(|&i| xs_new_array[i])
                .collect::<Array1<i64>>();
            let sorted_ys = indices
                .iter()
                .map(|&i| ys_new_array[i])
                .collect::<Array1<i64>>();
            let sorted_ts = indices
                .iter()
                .map(|&i| ts_new_array[i])
                .collect::<Array1<f64>>();
            let sorted_ps = indices
                .iter()
                .map(|&i| ps_new_array[i])
                .collect::<Array1<i64>>();

            // Convert arrays to Python objects
            let xs_py = sorted_xs.into_pyarray(py).to_object(py);
            let ys_py = sorted_ys.into_pyarray(py).to_object(py);
            let ts_py = sorted_ts.into_pyarray(py).to_object(py);
            let ps_py = sorted_ps.into_pyarray(py).to_object(py);

            // Create result tuple
            let result = PyTuple::new(py, &[xs_py, ys_py, ts_py, ps_py]);

            Ok(result.into())
        } else {
            // Convert arrays to Python objects
            let xs_py = xs_new_array.into_pyarray(py).to_object(py);
            let ys_py = ys_new_array.into_pyarray(py).to_object(py);
            let ts_py = ts_new_array.into_pyarray(py).to_object(py);
            let ps_py = ps_new_array.into_pyarray(py).to_object(py);

            // Create result tuple
            let result = PyTuple::new(py, &[xs_py, ys_py, ts_py, ps_py]);

            Ok(result.into())
        }
    }
}

/// Remove events by random selection
#[pyfunction]
#[pyo3(name = "remove_events")]
#[pyo3(signature = (xs, ys, ts, ps, to_remove, add_noise=0))]
pub fn remove_events(
    py: Python<'_>,
    xs: PyReadonlyArray1<i64>,
    ys: PyReadonlyArray1<i64>,
    ts: PyReadonlyArray1<f64>,
    ps: PyReadonlyArray1<i64>,
    to_remove: usize,
    add_noise: usize,
) -> PyResult<PyObject> {
    let xs_array = xs.as_array();
    let ys_array = ys.as_array();
    let ts_array = ts.as_array();
    let ps_array = ps.as_array();

    let n = xs_array.len();

    if to_remove >= n {
        // Return empty arrays
        let empty_xs = Array1::<i64>::zeros(0);
        let empty_ys = Array1::<i64>::zeros(0);
        let empty_ts = Array1::<f64>::zeros(0);
        let empty_ps = Array1::<i64>::zeros(0);

        // Convert arrays to Python objects
        let xs_py = empty_xs.into_pyarray(py).to_object(py);
        let ys_py = empty_ys.into_pyarray(py).to_object(py);
        let ts_py = empty_ts.into_pyarray(py).to_object(py);
        let ps_py = empty_ps.into_pyarray(py).to_object(py);

        // Create result tuple
        let result = PyTuple::new(py, &[xs_py, ys_py, ts_py, ps_py]);

        return Ok(result.into());
    }

    let to_select = n - to_remove;

    // Generate random indices without replacement
    let mut rng = thread_rng();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rng);
    indices.truncate(to_select);
    indices.sort();

    // Extract selected events
    let selected_xs = indices
        .iter()
        .map(|&i| xs_array[i])
        .collect::<Array1<i64>>();
    let selected_ys = indices
        .iter()
        .map(|&i| ys_array[i])
        .collect::<Array1<i64>>();
    let selected_ts = indices
        .iter()
        .map(|&i| ts_array[i])
        .collect::<Array1<f64>>();
    let selected_ps = indices
        .iter()
        .map(|&i| ps_array[i])
        .collect::<Array1<i64>>();

    if add_noise == 0 {
        // Convert arrays to Python objects
        let xs_py = selected_xs.into_pyarray(py).to_object(py);
        let ys_py = selected_ys.into_pyarray(py).to_object(py);
        let ts_py = selected_ts.into_pyarray(py).to_object(py);
        let ps_py = selected_ps.into_pyarray(py).to_object(py);

        // Create result tuple
        let result = PyTuple::new(py, &[xs_py, ys_py, ts_py, ps_py]);

        Ok(result.into())
    } else {
        // Generate random events for noise
        let max_x = xs_array.fold(0, |acc, &x| acc.max(x));
        let max_y = ys_array.fold(0, |acc, &y| acc.max(y));

        let mut xs_noise = Vec::with_capacity(add_noise);
        let mut ys_noise = Vec::with_capacity(add_noise);
        let mut ts_noise = Vec::with_capacity(add_noise);
        let mut ps_noise = Vec::with_capacity(add_noise);

        let min_ts = ts_array.fold(f64::INFINITY, |acc, &t| acc.min(t));
        let max_ts = ts_array.fold(f64::NEG_INFINITY, |acc, &t| acc.max(t));

        for _ in 0..add_noise {
            xs_noise.push(rng.gen_range(0..=max_x));
            ys_noise.push(rng.gen_range(0..=max_y));
            ts_noise.push(rng.gen_range(min_ts..=max_ts));
            ps_noise.push(if rng.gen_bool(0.5) { 1 } else { -1 });
        }

        // Merge selected events and noise
        let mut all_xs = Vec::with_capacity(selected_xs.len() + add_noise);
        let mut all_ys = Vec::with_capacity(selected_ys.len() + add_noise);
        let mut all_ts: Vec<f64> = Vec::with_capacity(selected_ts.len() + add_noise);
        let mut all_ps = Vec::with_capacity(selected_ps.len() + add_noise);

        all_xs.extend(selected_xs.iter());
        all_xs.extend(xs_noise.iter());

        all_ys.extend(selected_ys.iter());
        all_ys.extend(ys_noise.iter());

        all_ts.extend(selected_ts.iter());
        all_ts.extend(ts_noise.iter());

        all_ps.extend(selected_ps.iter());
        all_ps.extend(ps_noise.iter());

        let merged_xs = Array1::from(all_xs);
        let merged_ys = Array1::from(all_ys);
        let merged_ts = Array1::from(all_ts);
        let merged_ps = Array1::from(all_ps);

        // Sort by timestamp
        let mut indices: Vec<usize> = (0..merged_ts.len()).collect();
        indices.sort_by(|&i, &j| merged_ts[i].partial_cmp(&merged_ts[j]).unwrap());

        let sorted_xs = indices
            .iter()
            .map(|&i| merged_xs[i])
            .collect::<Array1<i64>>();
        let sorted_ys = indices
            .iter()
            .map(|&i| merged_ys[i])
            .collect::<Array1<i64>>();
        let sorted_ts = indices
            .iter()
            .map(|&i| merged_ts[i])
            .collect::<Array1<f64>>();
        let sorted_ps = indices
            .iter()
            .map(|&i| merged_ps[i])
            .collect::<Array1<i64>>();

        // Convert arrays to Python objects
        let xs_py = sorted_xs.into_pyarray(py).to_object(py);
        let ys_py = sorted_ys.into_pyarray(py).to_object(py);
        let ts_py = sorted_ts.into_pyarray(py).to_object(py);
        let ps_py = sorted_ps.into_pyarray(py).to_object(py);

        // Create result tuple
        let result = PyTuple::new(py, &[xs_py, ys_py, ts_py, ps_py]);

        Ok(result.into())
    }
}
/// Merge multiple sets of events into a single chronologically sorted list
#[pyfunction]
#[pyo3(name = "merge_events")]
pub fn merge_events(py: Python<'_>, event_sets: &PyTuple) -> PyResult<PyObject> {
    // Collect all event sets
    let mut all_sets: Vec<Events> = Vec::new();
    for event_set in event_sets.iter() {
        let tuple = event_set.extract::<&PyTuple>()?;
        if tuple.len() != 4 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Each event set must be a tuple of (xs, ys, ts, ps)",
            ));
        }
        let xs = tuple.get_item(0)?.extract::<PyReadonlyArray1<i64>>()?;
        let ys = tuple.get_item(1)?.extract::<PyReadonlyArray1<i64>>()?;
        let ts = tuple.get_item(2)?.extract::<PyReadonlyArray1<f64>>()?;
        let ps = tuple.get_item(3)?.extract::<PyReadonlyArray1<i64>>()?;
        let events = from_numpy_arrays(xs, ys, ts, ps);
        all_sets.push(events);
    }
    // Merge and sort events using core implementation
    let merged = super::merge_events(&all_sets);
    // Convert merged events to numpy arrays
    let xs: Vec<i64> = merged.iter().map(|e| e.x as i64).collect();
    let ys: Vec<i64> = merged.iter().map(|e| e.y as i64).collect();
    let ts: Vec<f64> = merged.iter().map(|e| e.t).collect();
    let ps: Vec<i64> = merged.iter().map(|e| e.polarity as i64).collect();
    let xs_py = Array1::from(xs).into_pyarray(py).to_object(py);
    let ys_py = Array1::from(ys).into_pyarray(py).to_object(py);
    let ts_py = Array1::from(ts).into_pyarray(py).to_object(py);
    let ps_py = Array1::from(ps).into_pyarray(py).to_object(py);
    let result = PyTuple::new(py, &[xs_py, ys_py, ts_py, ps_py]);
    Ok(result.into())
}
