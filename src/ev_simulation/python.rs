//! PyO3 bindings for the event simulator (`evlib._evlib.simulation_rs`).

use numpy::{
    IntoPyArray, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyReadwriteArray1, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use rayon::prelude::*;

use super::config::{SimError, SimulatorConfig};
#[cfg(feature = "cuda")]
use super::cuda::EventSimulatorCuda;
use super::simulator::{EventBatch, EventSimulator};

fn to_py_err(e: SimError) -> PyErr {
    match e {
        SimError::InvalidConfig(_)
        | SimError::ShapeMismatch { .. }
        | SimError::NonMonotonicTime { .. }
        | SimError::InvalidInput { .. }
        | SimError::EmptyBatch => PyValueError::new_err(e.to_string()),
        SimError::NotInitialised | SimError::Backend(_) => PyRuntimeError::new_err(e.to_string()),
    }
}

/// Reinterpret a `Vec<u16>` as `Vec<i16>` without a copy (same size and alignment).
fn reinterpret_i16(v: Vec<u16>) -> Vec<i16> {
    let mut v = std::mem::ManuallyDrop::new(v);
    let (ptr, len, cap) = (v.as_mut_ptr(), v.len(), v.capacity());
    // SAFETY: u16 and i16 have identical size and alignment, so the allocation
    // layout is the same; the source Vec is not dropped.
    unsafe { Vec::from_raw_parts(ptr as *mut i16, len, cap) }
}

fn batch_to_py<'py>(py: Python<'py>, b: EventBatch) -> PyResult<Bound<'py, PyTuple>> {
    let x = reinterpret_i16(b.x);
    let y = reinterpret_i16(b.y);
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

/// Shape check for a stateful `run`: (T, H, W) must match the simulator size.
fn check_run_shape(shape: &[usize], t_len: usize, height: u32, width: u32) -> PyResult<()> {
    if shape[0] != t_len || shape[1] != height as usize || shape[2] != width as usize {
        return Err(PyValueError::new_err(format!(
            "frames shape {:?} does not match (len(timestamps_ns), height, width) = {:?}",
            shape,
            (t_len, height, width)
        )));
    }
    Ok(())
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
        let data = u8_frames.as_slice()?;
        let out = py
            .allow_threads(move || {
                EventSimulator::new(cfg).and_then(|mut s| s.run_u8(data, &t, sort))
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
        let data = f32_frames.as_slice()?;
        let out = py
            .allow_threads(move || {
                EventSimulator::new(cfg).and_then(|mut s| s.run_log(data, &t, sort))
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
    /// Events are grouped by row chunk; the order is not stable across thread counts.
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        frame: &Bound<'py, PyAny>,
        t_ns: i64,
    ) -> PyResult<Bound<'py, PyTuple>> {
        // The readonly borrow pins the array; its slice is Send and crosses allow_threads.
        if let Ok(f) = frame.extract::<PyReadonlyArray2<'py, u8>>() {
            let data = f.as_slice()?;
            let inner = &mut self.inner;
            let out = py
                .allow_threads(move || {
                    let mut out = EventBatch::default();
                    inner.step_u8(data, t_ns, &mut out).map(|_| out)
                })
                .map_err(to_py_err)?;
            batch_to_py(py, out)
        } else if let Ok(f) = frame.extract::<PyReadonlyArray2<'py, f32>>() {
            let data = f.as_slice()?;
            let inner = &mut self.inner;
            let out = py
                .allow_threads(move || {
                    let mut out = EventBatch::default();
                    inner.step_log(data, t_ns, &mut out).map(|_| out)
                })
                .map_err(to_py_err)?;
            batch_to_py(py, out)
        } else {
            Err(PyTypeError::new_err(
                "frame must be a C-contiguous (H, W) uint8 or float32 array",
            ))
        }
    }

    /// A (T, H, W) uint8 or float32 stack at `timestamps_ns`; state carries over to
    /// the next call, so batches give the same events as one whole-stack run.
    /// With `sort=False` events are grouped by row chunk and the order is not
    /// stable across thread counts.
    #[pyo3(signature = (frames, timestamps_ns, *, sort=false))]
    fn run<'py>(
        &mut self,
        py: Python<'py>,
        frames: &Bound<'py, PyAny>,
        timestamps_ns: PyReadonlyArray1<'py, i64>,
        sort: bool,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let t = timestamps_ns.as_slice()?.to_vec();
        let (h, w) = (self.inner.config().height, self.inner.config().width);
        if let Ok(f) = frames.extract::<PyReadonlyArray3<'py, u8>>() {
            check_run_shape(f.shape(), t.len(), h, w)?;
            let data = f.as_slice()?;
            let inner = &mut self.inner;
            let out = py
                .allow_threads(move || inner.run_u8(data, &t, sort))
                .map_err(to_py_err)?;
            batch_to_py(py, out)
        } else if let Ok(f) = frames.extract::<PyReadonlyArray3<'py, f32>>() {
            check_run_shape(f.shape(), t.len(), h, w)?;
            let data = f.as_slice()?;
            let inner = &mut self.inner;
            let out = py
                .allow_threads(move || inner.run_log(data, &t, sort))
                .map_err(to_py_err)?;
            batch_to_py(py, out)
        } else {
            Err(PyTypeError::new_err(
                "frames must be a C-contiguous (T, H, W) uint8 or float32 array",
            ))
        }
    }
}

/// Floor-divide an int64 nanosecond array by 1000 in place (parallel), the same
/// result as numpy `t_ns // 1000`. Used by the DataFrame builder on the fresh
/// timestamp column, where numpy's serial division was the largest cost.
#[pyfunction]
#[pyo3(name = "ns_to_us_inplace")]
pub fn ns_to_us_inplace_py(py: Python<'_>, mut t_ns: PyReadwriteArray1<'_, i64>) -> PyResult<()> {
    let data = t_ns.as_slice_mut()?;
    py.allow_threads(move || {
        data.par_chunks_mut(1 << 16).for_each(|chunk| {
            for v in chunk {
                *v = v.div_euclid(1000);
            }
        });
    });
    Ok(())
}

/// True when the crate was built with the `cuda` feature and `libevsim.so` loads on a device.
#[pyfunction]
#[pyo3(name = "cuda_available")]
pub fn cuda_available_py() -> bool {
    #[cfg(feature = "cuda")]
    {
        super::cuda::cuda_available()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// CUDA twin of `simulate_frames`; same arguments and output.
#[cfg(feature = "cuda")]
#[pyfunction]
#[pyo3(name = "simulate_frames_cuda", signature = (frames, timestamps_ns, *, c_pos, c_neg, threshold_sigma, refractory_ns, log_eps, seed, sort))]
#[allow(clippy::too_many_arguments)]
pub fn simulate_frames_cuda_py<'py>(
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
        let data = u8_frames.as_slice()?;
        let out = py
            .allow_threads(move || {
                EventSimulatorCuda::new(cfg).and_then(|mut s| s.run_u8(data, &t, sort))
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
        let data = f32_frames.as_slice()?;
        let out = py
            .allow_threads(move || {
                EventSimulatorCuda::new(cfg).and_then(|mut s| s.run_log(data, &t, sort))
            })
            .map_err(to_py_err)?;
        return batch_to_py(py, out);
    }
    Err(PyTypeError::new_err(
        "frames must be a C-contiguous (T, H, W) uint8 or float32 array",
    ))
}

/// Stateful CUDA simulator with the same interface as `EventSimulator`.
#[cfg(feature = "cuda")]
#[pyclass(name = "EventSimulatorCuda")]
pub struct PyEventSimulatorCuda {
    inner: EventSimulatorCuda,
}

#[cfg(feature = "cuda")]
#[pymethods]
impl PyEventSimulatorCuda {
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
            inner: EventSimulatorCuda::new(cfg).map_err(to_py_err)?,
        })
    }

    #[getter]
    fn is_initialised(&self) -> bool {
        self.inner.is_initialised()
    }

    fn reset(&mut self) -> PyResult<()> {
        self.inner.reset().map_err(to_py_err)
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
    /// Events are grouped by row chunk; the order is not stable across thread counts.
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        frame: &Bound<'py, PyAny>,
        t_ns: i64,
    ) -> PyResult<Bound<'py, PyTuple>> {
        if let Ok(f) = frame.extract::<PyReadonlyArray2<'py, u8>>() {
            let data = f.as_slice()?;
            let inner = &mut self.inner;
            let out = py
                .allow_threads(move || inner.step_u8(data, t_ns))
                .map_err(to_py_err)?;
            batch_to_py(py, out)
        } else if let Ok(f) = frame.extract::<PyReadonlyArray2<'py, f32>>() {
            let data = f.as_slice()?;
            let inner = &mut self.inner;
            let out = py
                .allow_threads(move || inner.step_log(data, t_ns))
                .map_err(to_py_err)?;
            batch_to_py(py, out)
        } else {
            Err(PyTypeError::new_err(
                "frame must be a C-contiguous (H, W) uint8 or float32 array",
            ))
        }
    }

    /// A (T, H, W) uint8 or float32 stack at `timestamps_ns`; state carries over to
    /// the next call, so batches give the same events as one whole-stack run.
    /// With `sort=False` events are grouped by row chunk and the order is not
    /// stable across thread counts.
    #[pyo3(signature = (frames, timestamps_ns, *, sort=false))]
    fn run<'py>(
        &mut self,
        py: Python<'py>,
        frames: &Bound<'py, PyAny>,
        timestamps_ns: PyReadonlyArray1<'py, i64>,
        sort: bool,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let t = timestamps_ns.as_slice()?.to_vec();
        let (h, w) = (self.inner.config().height, self.inner.config().width);
        if let Ok(f) = frames.extract::<PyReadonlyArray3<'py, u8>>() {
            check_run_shape(f.shape(), t.len(), h, w)?;
            let data = f.as_slice()?;
            let inner = &mut self.inner;
            let out = py
                .allow_threads(move || inner.run_u8(data, &t, sort))
                .map_err(to_py_err)?;
            batch_to_py(py, out)
        } else if let Ok(f) = frames.extract::<PyReadonlyArray3<'py, f32>>() {
            check_run_shape(f.shape(), t.len(), h, w)?;
            let data = f.as_slice()?;
            let inner = &mut self.inner;
            let out = py
                .allow_threads(move || inner.run_log(data, &t, sort))
                .map_err(to_py_err)?;
            batch_to_py(py, out)
        } else {
            Err(PyTypeError::new_err(
                "frames must be a C-contiguous (T, H, W) uint8 or float32 array",
            ))
        }
    }
}

pub fn register_simulation_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate_frames_py, m)?)?;
    m.add_class::<PyEventSimulator>()?;
    m.add_function(wrap_pyfunction!(cuda_available_py, m)?)?;
    m.add_function(wrap_pyfunction!(ns_to_us_inplace_py, m)?)?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(simulate_frames_cuda_py, m)?)?;
        m.add_class::<PyEventSimulatorCuda>()?;
    }
    Ok(())
}
