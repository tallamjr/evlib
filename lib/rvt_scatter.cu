// CUDA dense scatter-add for the RVT stacked-histogram (evlib backend="cuda").
//
// scatter_windows() takes one batch of events (sorted by t) plus the batch's window-END grid and
// does the cheap host-side work (per-window slices via binary search, t0/denom, and a per-window
// start offset + prefix-sum of slice sizes), then runs ONE scatter kernel: one thread per
// event-window membership, each reconstructing its (window, event) from the tiny prefix array via
// in-kernel binary search (so no big membership-index arrays are transferred). The kernel does an
// atomicAdd into a dense u32 buffer with the downsample folded into the write; a second kernel
// clips to the cutoff and casts to u8. Output (n_windows, 2*nbins, out_h, out_w). The time-bin is
// computed in f64 to match evlib's Rust/Polars binning bit-for-bit. Off-sensor coords are dropped.
// x/y/p are int32 (their native h5 dtype) to halve the host->device transfer.
//
//   nvcc -O3 -arch=sm_89 -cudart static -shared -Xcompiler -fPIC -o librvt_scatter.so rvt_scatter.cu

#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cstdio>

// Largest w in [0, n_windows) with prefix[w] <= m (prefix has n_windows+1 entries, non-decreasing,
// prefix[0]=0, prefix[n_windows]=n_memb). Correct even across empty windows (equal prefixes).
__device__ __forceinline__ int find_window(const long long *prefix, int n_windows, long long m) {
  int lo = 0, hi = n_windows + 1;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (prefix[mid] <= m)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo - 1;
}

__global__ void scatter_k(const long long *t, const int *x, const int *y, const int *p,
                          const long long *starts, const long long *prefix, long long n_memb,
                          int n_windows, const long long *t0_arr, const double *denom_arr,
                          int nbins, const long long *row_map, const long long *col_map,
                          int width, int height, int out_h, int out_w, unsigned int *accum) {
  long long m = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= n_memb) return;
  int w = find_window(prefix, n_windows, m);
  long long i = starts[w] + (m - prefix[w]);
  int yi = y[i], xi = x[i];
  if (yi < 0 || yi >= height || xi < 0 || xi >= width) return;
  long long yo = row_map[yi];
  if (yo < 0) return;
  long long xo = col_map[xi];
  if (xo < 0) return;
  double frac = (double)(t[i] - t0_arr[w]) / denom_arr[w] * (double)nbins;
  long long tidx = (long long)floor(frac);
  if (tidx > nbins - 1) tidx = nbins - 1;
  long long pol = p[i] > 0 ? p[i] : 0;
  long long chan = pol * nbins + tidx;
  long long plane = (long long)out_h * out_w;
  long long channels = 2LL * nbins;
  long long flat = w * channels * plane + chan * plane + yo * out_w + xo;
  atomicAdd(&accum[flat], 1u);
}

__global__ void clip_cast_k(const unsigned int *accum, unsigned char *out, long long n,
                            unsigned int cutoff) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned int v = accum[i];
  out[i] = (unsigned char)(v < cutoff ? v : cutoff);
}

#define CK(call)                                                                   \
  do {                                                                             \
    cudaError_t _e = (call);                                                       \
    if (_e != cudaSuccess) {                                                       \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e),          \
              __FILE__, __LINE__);                                                 \
      return (int)_e;                                                              \
    }                                                                              \
  } while (0)

static long long lower_bound_ll(const long long *a, long long n, long long v) {
  return (long long)(std::lower_bound(a, a + n, v) - a);  // first >= v
}
static long long upper_bound_ll(const long long *a, long long n, long long v) {
  return (long long)(std::upper_bound(a, a + n, v) - a);  // first > v
}

extern "C" int scatter_windows(const long long *t, const int *x, const int *y, const int *p,
                               long long n_events, const long long *grid, int n_windows,
                               long long delta_t, int nbins, unsigned int cutoff,
                               const long long *row_map, const long long *col_map, int width,
                               int height, int out_h, int out_w, unsigned char *out_host) {
  long long channels = 2LL * nbins;
  long long plane = (long long)out_h * out_w;
  long long buf_len = (long long)n_windows * channels * plane;

  // Host: per-window slices, t0/denom, start offsets, and prefix-sum of slice sizes.
  std::vector<long long> t0_arr(n_windows), starts(n_windows), prefix(n_windows + 1);
  std::vector<double> denom_arr(n_windows);
  prefix[0] = 0;
  for (int w = 0; w < n_windows; ++w) {
    long long s = lower_bound_ll(t, n_events, grid[w] - delta_t);
    long long e = upper_bound_ll(t, n_events, grid[w]);
    if (e <= s) {
      starts[w] = 0;
      t0_arr[w] = 0;
      denom_arr[w] = 1.0;
      prefix[w + 1] = prefix[w];
      continue;
    }
    starts[w] = s;
    t0_arr[w] = t[s];
    long long span = t[e - 1] - t[s];
    denom_arr[w] = (double)(span > 1 ? span : 1);
    prefix[w + 1] = prefix[w] + (e - s);
  }
  long long n_memb = prefix[n_windows];

  long long *d_t, *d_starts, *d_prefix, *d_t0, *d_row, *d_col;
  int *d_x, *d_y, *d_p;
  double *d_denom;
  unsigned int *d_accum;
  unsigned char *d_out;

  CK(cudaMalloc(&d_t, n_events * sizeof(long long)));
  CK(cudaMalloc(&d_x, n_events * sizeof(int)));
  CK(cudaMalloc(&d_y, n_events * sizeof(int)));
  CK(cudaMalloc(&d_p, n_events * sizeof(int)));
  CK(cudaMalloc(&d_starts, n_windows * sizeof(long long)));
  CK(cudaMalloc(&d_prefix, (n_windows + 1) * sizeof(long long)));
  CK(cudaMalloc(&d_t0, n_windows * sizeof(long long)));
  CK(cudaMalloc(&d_denom, n_windows * sizeof(double)));
  CK(cudaMalloc(&d_row, height * sizeof(long long)));
  CK(cudaMalloc(&d_col, width * sizeof(long long)));
  CK(cudaMalloc(&d_accum, buf_len * sizeof(unsigned int)));
  CK(cudaMalloc(&d_out, buf_len * sizeof(unsigned char)));

  CK(cudaMemcpy(d_t, t, n_events * sizeof(long long), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_x, x, n_events * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_y, y, n_events * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_p, p, n_events * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_starts, starts.data(), n_windows * sizeof(long long), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_prefix, prefix.data(), (n_windows + 1) * sizeof(long long), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_t0, t0_arr.data(), n_windows * sizeof(long long), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_denom, denom_arr.data(), n_windows * sizeof(double), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_row, row_map, height * sizeof(long long), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_col, col_map, width * sizeof(long long), cudaMemcpyHostToDevice));
  CK(cudaMemset(d_accum, 0, buf_len * sizeof(unsigned int)));

  int threads = 256;
  if (n_memb > 0) {
    long long blocks = (n_memb + threads - 1) / threads;
    scatter_k<<<(unsigned int)blocks, threads>>>(d_t, d_x, d_y, d_p, d_starts, d_prefix, n_memb,
                                                 n_windows, d_t0, d_denom, nbins, d_row, d_col,
                                                 width, height, out_h, out_w, d_accum);
    CK(cudaGetLastError());
  }
  long long cblocks = (buf_len + threads - 1) / threads;
  clip_cast_k<<<(unsigned int)cblocks, threads>>>(d_accum, d_out, buf_len, cutoff);
  CK(cudaGetLastError());

  CK(cudaMemcpy(out_host, d_out, buf_len * sizeof(unsigned char), cudaMemcpyDeviceToHost));

  cudaFree(d_t); cudaFree(d_x); cudaFree(d_y); cudaFree(d_p);
  cudaFree(d_starts); cudaFree(d_prefix); cudaFree(d_t0); cudaFree(d_denom);
  cudaFree(d_row); cudaFree(d_col); cudaFree(d_accum); cudaFree(d_out);
  return 0;
}
