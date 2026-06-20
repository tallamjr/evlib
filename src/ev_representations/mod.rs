//! Event representations: dense scatter-add stacked histogram (RVT-identical).

pub mod stacked_histogram_dense;

#[cfg(feature = "cuda")]
pub mod stacked_histogram_cuda;

#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod stacked_histogram_metal;

pub use stacked_histogram_dense::stacked_histogram_dense;

#[cfg(feature = "python")]
pub mod python {
    use super::stacked_histogram_dense::stacked_histogram_dense;
    use numpy::{IntoPyArray, PyArray4, PyReadonlyArray1};
    use pyo3::prelude::*;

    /// Build the dense stacked-histogram for one batch of windows.
    ///
    /// Args (all sorted by `t`, non-decreasing, covering the batch):
    /// - `t`, `x`, `y`, `p`: int64 event arrays.
    /// - `grid`: int64 window END timestamps for this batch.
    /// - `delta_t_us`: window length in microseconds.
    /// - `nbins`: time bins per polarity (output has 2 * nbins channels).
    /// - `count_cutoff`: per-pixel saturation count.
    /// - `row_map` (len full H), `col_map` (len full W): source -> output index, -1 if dropped.
    /// - `out_h`, `out_w`: downsampled output dimensions.
    ///
    /// Returns a uint8 array of shape `(n_windows, 2 * nbins, out_h, out_w)`.
    #[pyfunction]
    #[pyo3(name = "stacked_histogram_dense")]
    #[allow(clippy::too_many_arguments)]
    pub fn stacked_histogram_dense_py<'py>(
        py: Python<'py>,
        t: PyReadonlyArray1<i64>,
        x: PyReadonlyArray1<i64>,
        y: PyReadonlyArray1<i64>,
        p: PyReadonlyArray1<i64>,
        grid: PyReadonlyArray1<i64>,
        delta_t_us: i64,
        nbins: usize,
        count_cutoff: u32,
        row_map: PyReadonlyArray1<i64>,
        col_map: PyReadonlyArray1<i64>,
        out_h: usize,
        out_w: usize,
    ) -> PyResult<Bound<'py, PyArray4<u8>>> {
        let t = t.as_slice()?;
        let x = x.as_slice()?;
        let y = y.as_slice()?;
        let p = p.as_slice()?;
        let grid = grid.as_slice()?;
        let row_map = row_map.as_slice()?;
        let col_map = col_map.as_slice()?;

        if x.len() != t.len() || y.len() != t.len() || p.len() != t.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "t, x, y, p must have equal length",
            ));
        }

        let out = py.allow_threads(|| {
            stacked_histogram_dense(
                t,
                x,
                y,
                p,
                grid,
                delta_t_us,
                nbins,
                count_cutoff,
                row_map,
                col_map,
                out_h,
                out_w,
            )
        });

        Ok(out.into_pyarray(py))
    }

    /// CUDA scatter-add variant: identical args/output, computed on the GPU via the loaded
    /// librvt_scatter.so. Only present when evlib is built with `--features cuda`.
    #[cfg(feature = "cuda")]
    #[pyfunction]
    #[pyo3(name = "stacked_histogram_dense_cuda")]
    #[allow(clippy::too_many_arguments)]
    pub fn stacked_histogram_dense_cuda_py<'py>(
        py: Python<'py>,
        t: PyReadonlyArray1<i64>,
        x: PyReadonlyArray1<i32>,
        y: PyReadonlyArray1<i32>,
        p: PyReadonlyArray1<i32>,
        grid: PyReadonlyArray1<i64>,
        delta_t_us: i64,
        nbins: usize,
        count_cutoff: u32,
        row_map: PyReadonlyArray1<i64>,
        col_map: PyReadonlyArray1<i64>,
        out_h: usize,
        out_w: usize,
    ) -> PyResult<Bound<'py, PyArray4<u8>>> {
        use super::stacked_histogram_cuda::stacked_histogram_dense_cuda;
        let t = t.as_slice()?;
        let x = x.as_slice()?;
        let y = y.as_slice()?;
        let p = p.as_slice()?;
        let grid = grid.as_slice()?;
        let row_map = row_map.as_slice()?;
        let col_map = col_map.as_slice()?;
        if x.len() != t.len() || y.len() != t.len() || p.len() != t.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "t, x, y, p must have equal length",
            ));
        }
        let out = py
            .allow_threads(|| {
                stacked_histogram_dense_cuda(
                    t,
                    x,
                    y,
                    p,
                    grid,
                    delta_t_us,
                    nbins,
                    count_cutoff,
                    row_map,
                    col_map,
                    out_h,
                    out_w,
                )
            })
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        Ok(out.into_pyarray(py))
    }

    /// Metal scatter-add variant (Apple Silicon): identical args/output, computed on the GPU via
    /// metal-rs. Only present when evlib is built with `--features metal` on macOS.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[pyfunction]
    #[pyo3(name = "stacked_histogram_dense_metal")]
    #[allow(clippy::too_many_arguments)]
    pub fn stacked_histogram_dense_metal_py<'py>(
        py: Python<'py>,
        t: PyReadonlyArray1<i64>,
        x: PyReadonlyArray1<i32>,
        y: PyReadonlyArray1<i32>,
        p: PyReadonlyArray1<i32>,
        grid: PyReadonlyArray1<i64>,
        delta_t_us: i64,
        nbins: usize,
        count_cutoff: u32,
        row_map: PyReadonlyArray1<i64>,
        col_map: PyReadonlyArray1<i64>,
        out_h: usize,
        out_w: usize,
    ) -> PyResult<Bound<'py, PyArray4<u8>>> {
        use super::stacked_histogram_metal::stacked_histogram_dense_metal;
        let t = t.as_slice()?;
        let x = x.as_slice()?;
        let y = y.as_slice()?;
        let p = p.as_slice()?;
        let grid = grid.as_slice()?;
        let row_map = row_map.as_slice()?;
        let col_map = col_map.as_slice()?;
        if x.len() != t.len() || y.len() != t.len() || p.len() != t.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "t, x, y, p must have equal length",
            ));
        }
        let out = py
            .allow_threads(|| {
                stacked_histogram_dense_metal(
                    t,
                    x,
                    y,
                    p,
                    grid,
                    delta_t_us,
                    nbins,
                    count_cutoff,
                    row_map,
                    col_map,
                    out_h,
                    out_w,
                )
            })
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        Ok(out.into_pyarray(py))
    }

    /// Register the representations functions onto a module.
    pub fn register_representations_functions(m: &Bound<'_, PyModule>) -> PyResult<()> {
        use pyo3::wrap_pyfunction;
        m.add_function(wrap_pyfunction!(stacked_histogram_dense_py, m)?)?;
        #[cfg(feature = "cuda")]
        m.add_function(wrap_pyfunction!(stacked_histogram_dense_cuda_py, m)?)?;
        #[cfg(all(feature = "metal", target_os = "macos"))]
        m.add_function(wrap_pyfunction!(stacked_histogram_dense_metal_py, m)?)?;
        Ok(())
    }
}
