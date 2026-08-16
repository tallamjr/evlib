//! PyO3 bindings for the event simulator (`evlib._evlib.simulation_rs`).

use numpy::{
    IntoPyArray, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use super::config::{SimError, SimulatorConfig};
use super::simulator::{EventBatch, EventSimulator};

fn to_py_err(e: SimError) -> PyErr {
    match e {
        SimError::InvalidConfig(_)
        | SimError::ShapeMismatch { .. }
        | SimError::NonMonotonicTime { .. }
        | SimError::EmptyBatch => PyValueError::new_err(e.to_string()),
        SimError::NotInitialised | SimError::Backend(_) => PyRuntimeError::new_err(e.to_string()),
    }
}

fn batch_to_py<'py>(py: Python<'py>, b: EventBatch) -> PyResult<Bound<'py, PyTuple>> {
    let x: Vec<i16> = b.x.into_iter().map(|v| v as i16).collect();
    let y: Vec<i16> = b.y.into_iter().map(|v| v as i16).collect();
    PyTuple::new(
        py,
        [
            x.into_pyarray(py).into_any(),
            y.into_pyarray(py).into_any(),
            b.t_ns.into_pyarray(py).into_any(),
            b.p.into_pyarray(py).into_any(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
fn build_config(
    width: u32,
    height: u32,
    c_pos: f32,
    c_neg: f32,
    threshold_sigma: f32,
    refractory_ns: i64,
    log_eps: f32,
    seed: u64,
) -> SimulatorConfig {
    SimulatorConfig {
        width,
        height,
        c_pos,
        c_neg,
        threshold_sigma,
        refractory_ns,
        log_eps,
        seed,
    }
}

/// Simulate events from a (T, H, W) uint8 (intensity) or float32 (log intensity) stack.
#[pyfunction]
#[pyo3(name = "simulate_frames", signature = (frames, timestamps_ns, *, c_pos, c_neg, threshold_sigma, refractory_ns, log_eps, seed, sort))]
#[allow(clippy::too_many_arguments)]
pub fn simulate_frames_py<'py>(
    py: Python<'py>,
    frames: &Bound<'py, PyAny>,
    timestamps_ns: PyReadonlyArray1<'py, i64>,
    c_pos: f32,
    c_neg: f32,
    threshold_sigma: f32,
    refractory_ns: i64,
    log_eps: f32,
    seed: u64,
    sort: bool,
) -> PyResult<Bound<'py, PyTuple>> {
    let t = timestamps_ns.as_slice()?.to_vec();
    if let Ok(u8_frames) = frames.extract::<PyReadonlyArray3<'py, u8>>() {
        let shape = u8_frames.shape().to_vec();
        let cfg = build_config(
            shape[2] as u32,
            shape[1] as u32,
            c_pos,
            c_neg,
            threshold_sigma,
            refractory_ns,
            log_eps,
            seed,
        );
        let data = u8_frames.as_slice()?.to_vec();
        let out = py
            .allow_threads(move || {
                EventSimulator::new(cfg).and_then(|mut s| s.run_u8(&data, &t, sort))
            })
            .map_err(to_py_err)?;
        return batch_to_py(py, out);
    }
    if let Ok(f32_frames) = frames.extract::<PyReadonlyArray3<'py, f32>>() {
        let shape = f32_frames.shape().to_vec();
        let cfg = build_config(
            shape[2] as u32,
            shape[1] as u32,
            c_pos,
            c_neg,
            threshold_sigma,
            refractory_ns,
            log_eps,
            seed,
        );
        let data = f32_frames.as_slice()?.to_vec();
        let out = py
            .allow_threads(move || {
                EventSimulator::new(cfg).and_then(|mut s| s.run_log(&data, &t, sort))
            })
            .map_err(to_py_err)?;
        return batch_to_py(py, out);
    }
    Err(PyTypeError::new_err(
        "frames must be a C-contiguous (T, H, W) uint8 or float32 array",
    ))
}

/// Stateful simulator: feed frames one at a time.
#[pyclass(name = "EventSimulator")]
pub struct PyEventSimulator {
    inner: EventSimulator,
}

#[pymethods]
impl PyEventSimulator {
    #[new]
    #[pyo3(signature = (*, width, height, c_pos=0.2, c_neg=0.2, threshold_sigma=0.0, refractory_ns=0, log_eps=1e-3, seed=0))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        width: u32,
        height: u32,
        c_pos: f32,
        c_neg: f32,
        threshold_sigma: f32,
        refractory_ns: i64,
        log_eps: f32,
        seed: u64,
    ) -> PyResult<Self> {
        let cfg = build_config(
            width,
            height,
            c_pos,
            c_neg,
            threshold_sigma,
            refractory_ns,
            log_eps,
            seed,
        );
        Ok(Self {
            inner: EventSimulator::new(cfg).map_err(to_py_err)?,
        })
    }

    #[getter]
    fn is_initialised(&self) -> bool {
        self.inner.is_initialised()
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    /// (c_pos, c_neg) per-pixel maps as (H, W) float32 arrays.
    #[allow(clippy::type_complexity)]
    fn thresholds<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>)> {
        let (h, w) = (
            self.inner.config().height as usize,
            self.inner.config().width as usize,
        );
        let (cp, cn) = self.inner.thresholds();
        let cp = cp.to_vec().into_pyarray(py).reshape([h, w])?;
        let cn = cn.to_vec().into_pyarray(py).reshape([h, w])?;
        Ok((cp, cn))
    }

    /// One (H, W) uint8 or float32 frame at `t_ns`; returns (x, y, t_ns, p) arrays.
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        frame: &Bound<'py, PyAny>,
        t_ns: i64,
    ) -> PyResult<Bound<'py, PyTuple>> {
        // Copy to an owned, Send buffer before allow_threads: the PyReadonlyArray
        // borrow is GIL-bound and cannot cross the closure, same as simulate_frames_py.
        if let Ok(f) = frame.extract::<PyReadonlyArray2<'py, u8>>() {
            let data = f.as_slice()?.to_vec();
            let inner = &mut self.inner;
            let out = py
                .allow_threads(move || {
                    let mut out = EventBatch::default();
                    inner.step_u8(&data, t_ns, &mut out).map(|_| out)
                })
                .map_err(to_py_err)?;
            batch_to_py(py, out)
        } else if let Ok(f) = frame.extract::<PyReadonlyArray2<'py, f32>>() {
            let data = f.as_slice()?.to_vec();
            let inner = &mut self.inner;
            let out = py
                .allow_threads(move || {
                    let mut out = EventBatch::default();
                    inner.step_log(&data, t_ns, &mut out).map(|_| out)
                })
                .map_err(to_py_err)?;
            batch_to_py(py, out)
        } else {
            Err(PyTypeError::new_err(
                "frame must be a C-contiguous (H, W) uint8 or float32 array",
            ))
        }
    }
}

pub fn register_simulation_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate_frames_py, m)?)?;
    m.add_class::<PyEventSimulator>()?;
    Ok(())
}
