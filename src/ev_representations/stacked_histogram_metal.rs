//! Metal GPU dense scatter-add backend (Apple Silicon).
//!
//! Mirrors the CUDA backend but for Apple GPUs: the MSL kernel is compiled at runtime via metal-rs
//! (no offline Metal toolchain needed), and Apple Silicon's unified memory (StorageModeShared) means
//! there is no host<->device copy. The host does the cheap per-window slicing + prefix-sum; one
//! scatter kernel (one thread per event/window membership, window reconstructed via in-kernel binary
//! search) atomic-adds into a dense u32 buffer with the downsample folded into the write; a clip+cast
//! kernel produces u8.
//!
//! Bit-identical to the CPU dense backend: Apple GPUs have no f64, but the time-bin is computed with
//! INTEGER division (`(t - t0) * nbins / denom`), whose floor equals the f64 floor for these
//! microsecond magnitudes, so there is no float-precision caveat.

use metal::{CompileOptions, Device, MTLResourceOptions, MTLSize};
use ndarray::Array4;
use std::ffi::c_void;

const MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void scatter_k(
    device const long*  t       [[buffer(0)]],
    device const int*   x       [[buffer(1)]],
    device const int*   y       [[buffer(2)]],
    device const int*   p       [[buffer(3)]],
    device const long*  starts  [[buffer(4)]],
    device const long*  prefix  [[buffer(5)]],
    device const long*  t0_arr  [[buffer(6)]],
    device const long*  denom   [[buffer(7)]],
    device const long*  row_map [[buffer(8)]],
    device const long*  col_map [[buffer(9)]],
    device atomic_uint* accum   [[buffer(10)]],
    constant int&  n_windows    [[buffer(11)]],
    constant long& n_memb       [[buffer(12)]],
    constant int&  nbins        [[buffer(13)]],
    constant int&  width        [[buffer(14)]],
    constant int&  height       [[buffer(15)]],
    constant int&  out_h        [[buffer(16)]],
    constant int&  out_w        [[buffer(17)]],
    uint gid [[thread_position_in_grid]])
{
    long m = (long)gid;
    if (m >= n_memb) return;
    // largest w with prefix[w] <= m (binary search over prefix[0..n_windows])
    int lo = 0, hi = n_windows + 1;
    while (lo < hi) { int mid = (lo + hi) >> 1; if (prefix[mid] <= m) lo = mid + 1; else hi = mid; }
    int w = lo - 1;
    long i = starts[w] + (m - prefix[w]);
    int yi = y[i], xi = x[i];
    if (yi < 0 || yi >= height || xi < 0 || xi >= width) return;
    long yo = row_map[yi]; if (yo < 0) return;
    long xo = col_map[xi]; if (xo < 0) return;
    long num = (t[i] - t0_arr[w]) * (long)nbins;
    long tidx = num / denom[w];                  // integer floor (operands non-negative)
    if (tidx > nbins - 1) tidx = nbins - 1;
    long pol = p[i] > 0 ? p[i] : 0;
    long chan = pol * (long)nbins + tidx;
    long plane = (long)out_h * out_w;
    long channels = 2 * (long)nbins;
    long flat = w * channels * plane + chan * plane + yo * out_w + xo;
    atomic_fetch_add_explicit(&accum[flat], 1u, memory_order_relaxed);
}

kernel void clip_cast_k(
    device const uint* accum [[buffer(0)]],
    device uchar* out        [[buffer(1)]],
    constant long& n         [[buffer(2)]],
    constant uint& cutoff    [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    long i = (long)gid;
    if (i >= n) return;
    uint v = accum[i];
    out[i] = (uchar)(v < cutoff ? v : cutoff);
}
"#;

#[allow(clippy::too_many_arguments)]
pub fn stacked_histogram_dense_metal(
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
    let plane = out_h * out_w;
    let buf_len = n_windows * channels * plane;
    let n_events = t.len();

    // Host: per-window slices, t0, integer denom, start offsets, prefix-sum of slice sizes.
    let mut t0 = vec![0i64; n_windows];
    let mut denom = vec![1i64; n_windows];
    let mut starts = vec![0i64; n_windows];
    let mut prefix = vec![0i64; n_windows + 1];
    for w in 0..n_windows {
        let s = t.partition_point(|&v| v < grid[w] - delta_t_us); // first >= lo
        let e = t.partition_point(|&v| v <= grid[w]); // first > t_end
        if e <= s {
            prefix[w + 1] = prefix[w];
            continue;
        }
        starts[w] = s as i64;
        t0[w] = t[s];
        let span = t[e - 1] - t[s];
        denom[w] = if span > 1 { span } else { 1 };
        prefix[w + 1] = prefix[w] + (e - s) as i64;
    }
    let n_memb = prefix[n_windows];

    let device = Device::system_default().ok_or("no Metal device available")?;
    let lib = device
        .new_library_with_source(MSL, &CompileOptions::new())
        .map_err(|e| format!("MSL compile failed: {e}"))?;
    let pso_scatter = {
        let f = lib
            .get_function("scatter_k", None)
            .map_err(|e| e.to_string())?;
        device
            .new_compute_pipeline_state_with_function(&f)
            .map_err(|e| e.to_string())?
    };
    let pso_clip = {
        let f = lib
            .get_function("clip_cast_k", None)
            .map_err(|e| e.to_string())?;
        device
            .new_compute_pipeline_state_with_function(&f)
            .map_err(|e| e.to_string())?
    };
    let queue = device.new_command_queue();
    let shared = MTLResourceOptions::StorageModeShared;

    let i64buf = |v: &[i64]| {
        let n = v.len().max(1);
        device.new_buffer_with_data(v.as_ptr() as *const c_void, (n * 8) as u64, shared)
    };
    let i32buf = |v: &[i32]| {
        let n = v.len().max(1);
        device.new_buffer_with_data(v.as_ptr() as *const c_void, (n * 4) as u64, shared)
    };

    let b_t = i64buf(t);
    let b_x = i32buf(x);
    let b_y = i32buf(y);
    let b_p = i32buf(p);
    let b_starts = i64buf(&starts);
    let b_prefix = i64buf(&prefix);
    let b_t0 = i64buf(&t0);
    let b_denom = i64buf(&denom);
    let b_row = i64buf(row_map);
    let b_col = i64buf(col_map);
    let b_accum = device.new_buffer((buf_len * 4) as u64, shared);
    let b_out = device.new_buffer(buf_len as u64, shared);
    // new_buffer is not guaranteed zeroed; zero the accumulator (shared storage = CPU-visible).
    unsafe {
        std::ptr::write_bytes(b_accum.contents() as *mut u8, 0, buf_len * 4);
    }

    let nw = n_windows as i32;
    let nb = nbins as i32;
    let width = col_map.len() as i32;
    let height = row_map.len() as i32;
    let oh = out_h as i32;
    let ow = out_w as i32;
    let buf_len_i64 = buf_len as i64;

    let set_i32 = |enc: &metal::ComputeCommandEncoderRef, idx: u64, v: &i32| {
        enc.set_bytes(idx, 4, v as *const i32 as *const c_void);
    };

    let cmd = queue.new_command_buffer();
    if n_memb > 0 && n_events > 0 {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pso_scatter);
        for (idx, b) in [
            &b_t, &b_x, &b_y, &b_p, &b_starts, &b_prefix, &b_t0, &b_denom, &b_row, &b_col, &b_accum,
        ]
        .iter()
        .enumerate()
        {
            enc.set_buffer(idx as u64, Some(b), 0);
        }
        set_i32(enc, 11, &nw);
        enc.set_bytes(12, 8, &n_memb as *const i64 as *const c_void);
        set_i32(enc, 13, &nb);
        set_i32(enc, 14, &width);
        set_i32(enc, 15, &height);
        set_i32(enc, 16, &oh);
        set_i32(enc, 17, &ow);
        let tg = pso_scatter.max_total_threads_per_threadgroup().min(256);
        enc.dispatch_threads(MTLSize::new(n_memb as u64, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding();
    }
    {
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pso_clip);
        enc.set_buffer(0, Some(&b_accum), 0);
        enc.set_buffer(1, Some(&b_out), 0);
        enc.set_bytes(2, 8, &buf_len_i64 as *const i64 as *const c_void);
        enc.set_bytes(3, 4, &count_cutoff as *const u32 as *const c_void);
        let tg = pso_clip.max_total_threads_per_threadgroup().min(256);
        enc.dispatch_threads(MTLSize::new(buf_len as u64, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding();
    }
    cmd.commit();
    cmd.wait_until_completed();

    let out_slice = unsafe { std::slice::from_raw_parts(b_out.contents() as *const u8, buf_len) };
    Array4::from_shape_vec((n_windows, channels, out_h, out_w), out_slice.to_vec())
        .map_err(|e| format!("failed to shape Metal output: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ev_representations::stacked_histogram_dense::stacked_histogram_dense;

    #[test]
    fn metal_matches_cpu_dense() {
        // Small synthetic batch: identity downsample maps, a few windows, boundary + off-sensor.
        let row_map: Vec<i64> = (0..4).collect();
        let col_map: Vec<i64> = (0..4).collect();
        let t: Vec<i64> = vec![0, 10, 20, 50_000, 50_000, 60_000, 99_999, 100_000];
        let x: Vec<i32> = vec![0, 1, 2, 3, 0, 1, 2, 9]; // last is off-sensor (x=9 >= width 4)
        let y: Vec<i32> = vec![0, 1, 2, 3, 0, 1, 2, 3];
        let p: Vec<i32> = vec![1, 0, 1, 0, 1, 0, 1, 1];
        let grid: Vec<i64> = vec![50_000, 100_000];
        let nbins = 2usize;
        let cutoff = 10u32;
        let cpu = stacked_histogram_dense(
            &t,
            &x.iter().map(|&v| v as i64).collect::<Vec<_>>(),
            &y.iter().map(|&v| v as i64).collect::<Vec<_>>(),
            &p.iter().map(|&v| v as i64).collect::<Vec<_>>(),
            &grid,
            50_000,
            nbins,
            cutoff,
            &row_map,
            &col_map,
            4,
            4,
        );
        let metal = stacked_histogram_dense_metal(
            &t, &x, &y, &p, &grid, 50_000, nbins, cutoff, &row_map, &col_map, 4, 4,
        )
        .expect("metal backend");
        assert_eq!(cpu.shape(), metal.shape());
        assert_eq!(
            cpu, metal,
            "Metal output must be bit-identical to CPU dense"
        );
    }
}
