//! Dense scatter-add engine for the RVT stacked-histogram representation.
//!
//! This is a direct port of the verified numpy prototype that is bit-identical to the
//! RVT torch reference across all windows of the Gen4 validation sequence. For each
//! window it counts events straight into a preallocated dense accumulation buffer
//! (equivalent to torch `tensor.put_(accumulate=True)`), clips each pixel to the count
//! cutoff and casts to `u8`. This avoids the Polars hash group-by build entirely.
//!
//! Correctness invariants that MUST be preserved (each is load-bearing and was the
//! cause of a real mismatch during development):
//!
//! 1. Time-bin index `tidx` is computed in `f64`. torch's int64 true-divide yields a
//!    correctly-rounded float; Polars `f32` division used an approximate SIMD reciprocal
//!    that rounded the wrong way at exact bin boundaries. `f64` matches torch bit-for-bit.
//! 2. Window slices are RVT's `[searchsorted(t, T-dt, "left"), searchsorted(t, T, "right"))`.
//!    On a shared boundary (gap == dt) an event is included in two windows. Because each
//!    window is sliced independently this happens naturally; do not deduplicate.
//! 3. Counts are accumulated in a wide integer (`u32`) and only then clipped to the
//!    cutoff and cast to `u8`. The reference has no pixel that wraps a u8, so wide-int +
//!    clip is bit-identical to torch's u8 accumulation; do NOT emulate u8 wraparound.

use ndarray::Array4;

/// Build the dense stacked-histogram for one batch of windows.
///
/// All event arrays (`t`, `x`, `y`, `p`) cover the batch and are sorted by `t`
/// (non-decreasing). `grid` holds the window END timestamps for this batch.
/// `row_map` (length full H) and `col_map` (length full W) map a source row/column to
/// its downsampled output index, or `-1` if that row/column is dropped by the gather.
///
/// Returns an array of shape `(n_windows, 2 * nbins, out_h, out_w)` of `u8`.
#[allow(clippy::too_many_arguments)]
pub fn stacked_histogram_dense(
    t: &[i64],
    x: &[i64],
    y: &[i64],
    p: &[i64],
    grid: &[i64],
    delta_t_us: i64,
    nbins: usize,
    count_cutoff: u32,
    row_map: &[i64],
    col_map: &[i64],
    out_h: usize,
    out_w: usize,
) -> Array4<u8> {
    let n_windows = grid.len();
    let channels = 2 * nbins;
    let plane = out_h * out_w;
    let buf_len = channels * plane;

    let mut out = Array4::<u8>::zeros((n_windows, channels, out_h, out_w));

    // Reused per-window accumulation buffer (wide int, zeroed between windows).
    let mut accum = vec![0u32; buf_len];
    let nbins_i64 = nbins as i64;
    let nbins_f64 = nbins as f64;
    let max_tidx = (nbins - 1) as i64;

    for (w, &t_end) in grid.iter().enumerate() {
        // RVT window slice: [searchsorted(t, t_end - dt, "left"),
        //                    searchsorted(t, t_end, "right")).
        let lo = t_end - delta_t_us;
        let s = lower_bound(t, lo);
        let e = upper_bound(t, t_end);
        if e <= s {
            continue;
        }

        let tw = &t[s..e];
        let t0 = tw[0];
        let t1 = tw[e - 1 - s];
        let denom = (t1 - t0).max(1) as f64;

        // Scatter-add into the dense accumulation buffer.
        for i in s..e {
            // Drop off-sensor coordinates. A raw Gen4 stream can carry a coordinate beyond the
            // nominal sensor extent (e.g. x = 1284 on a 1280-wide sensor); the Polars path drops
            // these via the downsample inner-join / is_between filter, so the dense path must too
            // to stay bit-identical to RVT (and to avoid an out-of-bounds index into the maps).
            let yi = y[i];
            let xi = x[i];
            if yi < 0 || yi as usize >= row_map.len() || xi < 0 || xi as usize >= col_map.len() {
                continue;
            }
            let yo = row_map[yi as usize];
            if yo < 0 {
                continue;
            }
            let xo = col_map[xi as usize];
            if xo < 0 {
                continue;
            }

            // tidx = floor((t - t0) / denom * nbins), clipped to nbins - 1. f64 throughout.
            let frac = ((t[i] - t0) as f64) / denom * nbins_f64;
            let mut tidx = frac.floor() as i64;
            if tidx > max_tidx {
                tidx = max_tidx;
            }

            // channel = clip(p, 0, ..) * nbins + tidx.
            let pol = if p[i] > 0 { p[i] } else { 0 };
            let chan = pol * nbins_i64 + tidx;

            let flat = (chan as usize) * plane + (yo as usize) * out_w + (xo as usize);
            accum[flat] += 1;
        }

        // Clip to cutoff, cast to u8, write into the output window, and zero the buffer.
        let mut window = out.index_axis_mut(ndarray::Axis(0), w);
        let window_slice = window.as_slice_mut().expect("contiguous window slice");
        for (dst, src) in window_slice.iter_mut().zip(accum.iter_mut()) {
            let v = (*src).min(count_cutoff);
            *dst = v as u8;
            *src = 0;
        }
    }

    out
}

/// Index of the first element `>= value` (numpy `searchsorted(..., "left")`).
fn lower_bound(arr: &[i64], value: i64) -> usize {
    let mut lo = 0usize;
    let mut hi = arr.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if arr[mid] < value {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Index of the first element `> value` (numpy `searchsorted(..., "right")`).
fn upper_bound(arr: &[i64], value: i64) -> usize {
    let mut lo = 0usize;
    let mut hi = arr.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if arr[mid] <= value {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_bounds_match_numpy() {
        let arr = [10i64, 10, 20, 30, 30, 30];
        assert_eq!(lower_bound(&arr, 10), 0);
        assert_eq!(upper_bound(&arr, 10), 2);
        assert_eq!(lower_bound(&arr, 30), 3);
        assert_eq!(upper_bound(&arr, 30), 6);
        assert_eq!(lower_bound(&arr, 25), 3);
        assert_eq!(upper_bound(&arr, 5), 0);
    }

    #[test]
    fn out_of_bounds_source_coords_are_dropped() {
        // Source sensor is 2x2 (row_map / col_map length 2). A real Gen4 stream can carry an
        // off-sensor coordinate (e.g. x = 1284 on a 1280-wide sensor); such events must be
        // DROPPED, matching the Polars path (downsample inner-join / is_between filter) and the
        // committed RVT reference - not panic and not corrupt a neighbouring pixel.
        let row_map = [0i64, 1];
        let col_map = [0i64, 1];
        // Two events at (y=0): the first in-bounds (x=0), the second off-sensor (x=4, y=3).
        let t = [0i64, 0];
        let x = [0i64, 4];
        let y = [0i64, 3];
        let p = [1i64, 1];
        let grid = [0i64];
        let out = stacked_histogram_dense(
            &t, &x, &y, &p, &grid, 50_000, 2, 10, &row_map, &col_map, 2, 2,
        );
        // Only the in-bounds event is counted: channel = 1*2 + 0 = 2 at (y=0, x=0).
        assert_eq!(out[[0, 2, 0, 0]], 1);
        // Total counts across the whole window equal 1 (the off-sensor event was dropped).
        assert_eq!(out.iter().map(|&v| v as u32).sum::<u32>(), 1);
    }

    #[test]
    fn single_window_counts_and_clips() {
        // 2x2 output, identity maps, nbins=2 so 4 channels.
        let row_map = [0i64, 1];
        let col_map = [0i64, 1];
        // Three events at pixel (y=0, x=0), polarity 1, all at t=0 so tidx=0.
        // channel = 1*2 + 0 = 2.
        let t = [0i64, 0, 0];
        let x = [0i64, 0, 0];
        let y = [0i64, 0, 0];
        let p = [1i64, 1, 1];
        let grid = [0i64];
        let out = stacked_histogram_dense(
            &t, &x, &y, &p, &grid, 50_000, 2, 10, &row_map, &col_map, 2, 2,
        );
        assert_eq!(out.shape(), &[1, 4, 2, 2]);
        assert_eq!(out[[0, 2, 0, 0]], 3);
        // cutoff clip
        let p2 = [1i64; 15];
        let t2 = [0i64; 15];
        let x2 = [0i64; 15];
        let y2 = [0i64; 15];
        let out2 = stacked_histogram_dense(
            &t2, &x2, &y2, &p2, &grid, 50_000, 2, 10, &row_map, &col_map, 2, 2,
        );
        assert_eq!(out2[[0, 2, 0, 0]], 10);
    }
}
