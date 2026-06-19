//! CUDA dense scatter-add backend.
//!
//! Loads the nvcc-built `librvt_scatter.so` at runtime (via `libloading`, so the Rust extension
//! has no link-time CUDA dependency) and calls its `scatter_windows` host function. The CUDA path
//! produces output bit-identical to the CPU dense backend (`stacked_histogram_dense`), computed on
//! the GPU: one atomic scatter-add kernel over all event/window memberships with the downsample
//! folded into the write, plus a clip+cast-to-u8 kernel.
//!
//! The shared library is located via the `EVLIB_CUDA_LIB` environment variable (a full path), or
//! `librvt_scatter.so` resolved through the normal dynamic-loader search path.

use ndarray::Array4;
use std::os::raw::{c_int, c_longlong, c_uint};
use std::sync::OnceLock;

type ScatterFn = unsafe extern "C" fn(
    *const c_longlong, // t
    *const c_int,      // x (int32, native h5 dtype)
    *const c_int,      // y
    *const c_int,      // p
    c_longlong,        // n_events
    *const c_longlong, // grid (window end timestamps)
    c_int,             // n_windows
    c_longlong,        // delta_t
    c_int,             // nbins
    c_uint,            // count_cutoff
    *const c_longlong, // row_map (len = height)
    *const c_longlong, // col_map (len = width)
    c_int,             // width
    c_int,             // height
    c_int,             // out_h
    c_int,             // out_w
    *mut u8,           // out_host
) -> c_int;

fn lib_path() -> String {
    std::env::var("EVLIB_CUDA_LIB").unwrap_or_else(|_| "librvt_scatter.so".to_string())
}

// The CUDA shared library is loaded once per process and cached: dlopen (and the CUDA context it
// initialises) is expensive, so reloading it per batch dominated the runtime.
static LIB: OnceLock<libloading::Library> = OnceLock::new();

fn cuda_lib() -> Result<&'static libloading::Library, String> {
    if let Some(l) = LIB.get() {
        return Ok(l);
    }
    let path = lib_path();
    let loaded =
        unsafe { libloading::Library::new(&path) }.map_err(|e| format!("failed to load CUDA library {path}: {e}"))?;
    // Another thread may win the race; either way LIB ends up set, so just read it back.
    let _ = LIB.set(loaded);
    Ok(LIB.get().expect("CUDA library set"))
}

/// Build the dense stacked-histogram for one batch of windows on the GPU.
///
/// Same arguments and output shape `(n_windows, 2*nbins, out_h, out_w)` as the CPU
/// `stacked_histogram_dense`; `row_map`/`col_map` give the source->output index (or -1 if dropped),
/// with `height = row_map.len()` and `width = col_map.len()`.
#[allow(clippy::too_many_arguments)]
pub fn stacked_histogram_dense_cuda(
    t: &[i64],
    x: &[i32],
    y: &[i32],
    p: &[i32],
    grid: &[i64],
    delta_t_us: i64,
    nbins: usize,
    count_cutoff: u32,
    row_map: &[i64],
    col_map: &[i64],
    out_h: usize,
    out_w: usize,
) -> Result<Array4<u8>, String> {
    let n_windows = grid.len();
    let channels = 2 * nbins;
    let buf_len = n_windows * channels * out_h * out_w;
    let mut out = vec![0u8; buf_len];

    let libh = cuda_lib()?;
    unsafe {
        let func: libloading::Symbol<ScatterFn> = libh
            .get(b"scatter_windows")
            .map_err(|e| format!("missing symbol scatter_windows: {e}"))?;
        let rc = func(
            t.as_ptr(),
            x.as_ptr(),
            y.as_ptr(),
            p.as_ptr(),
            t.len() as c_longlong,
            grid.as_ptr(),
            n_windows as c_int,
            delta_t_us as c_longlong,
            nbins as c_int,
            count_cutoff as c_uint,
            row_map.as_ptr(),
            col_map.as_ptr(),
            col_map.len() as c_int,
            row_map.len() as c_int,
            out_h as c_int,
            out_w as c_int,
            out.as_mut_ptr(),
        );
        if rc != 0 {
            return Err(format!("scatter_windows returned CUDA error code {rc}"));
        }
    }
    Array4::from_shape_vec((n_windows, channels, out_h, out_w), out)
        .map_err(|e| format!("failed to shape CUDA output: {e}"))
}
