// CUDA backend for the evlib event simulator (evlib.simulation device="cuda").
//
// One thread per pixel walks the batch of T log-intensity frames. Pass 1 counts
// events on a copy of the pixel state; an exclusive scan of the counts gives
// each pixel its write offset; pass 2 walks again, writes (x, y, t, p) at
// offset + k, and commits the advanced state (l_ref, t_last, prev). If the
// total exceeds the caller's capacity the run returns 1 with the required
// count and leaves the state untouched, so the caller can retry.
//
// The per-pixel arithmetic is a transliteration of src/ev_simulation/pixel.rs:
// f32 reference and thresholds, f64 crossing fraction clamped to [0, 1],
// floored to whole nanoseconds, refractory test `t_ev - t_last >= refractory`
// with t_last == LLONG_MIN meaning "no event yet". The reference moves on
// every crossing, including crossings the refractory period drops.
//
// Threshold maps are generated on the host so CPU and CUDA share identical maps.
//
//   nvcc -O3 -arch=sm_89 -cudart static -shared -Xcompiler -fPIC -o libevsim.so esim_kernel.cu

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define EVSIM_CHECK(call)                                                       \
  do {                                                                          \
    cudaError_t evsim_err_ = (call);                                            \
    if (evsim_err_ != cudaSuccess) return -(int)evsim_err_;                     \
  } while (0)

static const long long NO_EVENT = LLONG_MIN;

extern "C" int evsim_destroy(void *handle);

struct EvsimHandle {
  int width;
  int height;
  long long n;
  long long refractory_ns;
  // Persistent per-pixel state.
  float *d_cpos;
  float *d_cneg;
  float *d_lref;
  long long *d_tlast;
  float *d_prev;
  long long prev_t;
  int initialised;
  // Per-call scratch, grown on demand.
  float *d_frames;
  long long frames_cap;  // in floats
  long long *d_tns;
  int tns_cap;
  long long *d_counts;   // n entries
  long long *d_offsets;  // n entries
  unsigned short *d_out_x;
  unsigned short *d_out_y;
  long long *d_out_t;
  signed char *d_out_p;
  long long out_cap;
};

__device__ __forceinline__ long long crossing_time(float l0, long long t0, float l1,
                                                   long long t1, float lc) {
  double dl = (double)l1 - (double)l0;
  double frac;
  if (dl == 0.0) {
    frac = 1.0;
  } else {
    frac = ((double)lc - (double)l0) / dl;
    frac = fmin(fmax(frac, 0.0), 1.0);
  }
  return t0 + (long long)floor((double)(t1 - t0) * frac);
}

// Walk pixel i across the batch. WRITE=false only counts; WRITE=true writes events
// at out_*[offset + k] and stores the advanced state.
template <bool WRITE>
__device__ __forceinline__ long long walk_pixel(
    long long i, long long n, const float *frames, const long long *t_ns, int n_frames,
    float cpos, float cneg, long long refractory_ns, int init, float lref_in,
    long long tlast_in, float prev_in, long long prev_t, int width, long long offset,
    unsigned short *out_x, unsigned short *out_y, long long *out_t, signed char *out_p,
    float *lref_out, long long *tlast_out) {
  float l_ref;
  long long t_last;
  float l0;
  long long t0;
  int k_start;
  if (init) {
    l0 = frames[i];
    t0 = t_ns[0];
    l_ref = l0;
    t_last = NO_EVENT;
    k_start = 1;
  } else {
    l0 = prev_in;
    t0 = prev_t;
    l_ref = lref_in;
    t_last = tlast_in;
    k_start = 0;
  }
  unsigned short px = (unsigned short)(i % width);
  unsigned short py = (unsigned short)(i / width);
  long long count = 0;
  for (int k = k_start; k < n_frames; ++k) {
    float l1 = frames[(long long)k * n + i];
    long long t1 = t_ns[k];
    if (l1 > l0) {
      while (l1 >= l_ref + cpos) {
        float lc = l_ref + cpos;
        long long t_ev = crossing_time(l0, t0, l1, t1, lc);
        l_ref = lc;
        if (t_last == NO_EVENT || t_ev - t_last >= refractory_ns) {
          if (WRITE) {
            long long o = offset + count;
            out_x[o] = px;
            out_y[o] = py;
            out_t[o] = t_ev;
            out_p[o] = 1;
          }
          t_last = t_ev;
          ++count;
        }
      }
    } else if (l1 < l0) {
      while (l1 <= l_ref - cneg) {
        float lc = l_ref - cneg;
        long long t_ev = crossing_time(l0, t0, l1, t1, lc);
        l_ref = lc;
        if (t_last == NO_EVENT || t_ev - t_last >= refractory_ns) {
          if (WRITE) {
            long long o = offset + count;
            out_x[o] = px;
            out_y[o] = py;
            out_t[o] = t_ev;
            out_p[o] = -1;
          }
          t_last = t_ev;
          ++count;
        }
      }
    }
    l0 = l1;
    t0 = t1;
  }
  if (WRITE) {
    *lref_out = l_ref;
    *tlast_out = t_last;
  }
  return count;
}

__global__ void count_k(long long n, const float *frames, const long long *t_ns, int n_frames,
                        const float *cpos, const float *cneg, long long refractory_ns, int init,
                        const float *lref, const long long *tlast, const float *prev,
                        long long prev_t, int width, long long *counts) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  counts[i] = walk_pixel<false>(i, n, frames, t_ns, n_frames, cpos[i], cneg[i], refractory_ns,
                                init, lref[i], tlast[i], prev[i], prev_t, width, 0, nullptr,
                                nullptr, nullptr, nullptr, nullptr, nullptr);
}

__global__ void write_k(long long n, const float *frames, const long long *t_ns, int n_frames,
                        const float *cpos, const float *cneg, long long refractory_ns, int init,
                        float *lref, long long *tlast, float *prev, long long prev_t, int width,
                        const long long *offsets, unsigned short *out_x, unsigned short *out_y,
                        long long *out_t, signed char *out_p) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float lref_new;
  long long tlast_new;
  walk_pixel<true>(i, n, frames, t_ns, n_frames, cpos[i], cneg[i], refractory_ns, init, lref[i],
                   tlast[i], prev[i], prev_t, width, offsets[i], out_x, out_y, out_t, out_p,
                   &lref_new, &tlast_new);
  lref[i] = lref_new;
  tlast[i] = tlast_new;
  prev[i] = frames[(long long)(n_frames - 1) * n + i];
}

template <typename T>
static int grow(T **ptr, long long *cap, long long needed) {
  if (needed <= *cap) return 0;
  if (*ptr != nullptr) EVSIM_CHECK(cudaFree(*ptr));
  *ptr = nullptr;
  *cap = 0;
  EVSIM_CHECK(cudaMalloc((void **)ptr, (size_t)needed * sizeof(T)));
  *cap = needed;
  return 0;
}

extern "C" {

// Returns 0 on success, negative CUDA error code otherwise. All arrays are host pointers.
int evsim_create(int width, int height, const float *c_pos, const float *c_neg,
                 long long refractory_ns, void **handle) {
  if (width <= 0 || height <= 0 || c_pos == nullptr || c_neg == nullptr || handle == nullptr) {
    return -(int)cudaErrorInvalidValue;
  }
  EvsimHandle *h = (EvsimHandle *)calloc(1, sizeof(EvsimHandle));
  if (h == nullptr) return -(int)cudaErrorMemoryAllocation;
  h->width = width;
  h->height = height;
  h->n = (long long)width * (long long)height;
  h->refractory_ns = refractory_ns;
  h->initialised = 0;
  size_t nf = (size_t)h->n * sizeof(float);
  size_t nl = (size_t)h->n * sizeof(long long);
  cudaError_t e = cudaSuccess;
  if (e == cudaSuccess) e = cudaMalloc((void **)&h->d_cpos, nf);
  if (e == cudaSuccess) e = cudaMalloc((void **)&h->d_cneg, nf);
  if (e == cudaSuccess) e = cudaMalloc((void **)&h->d_lref, nf);
  if (e == cudaSuccess) e = cudaMalloc((void **)&h->d_tlast, nl);
  if (e == cudaSuccess) e = cudaMalloc((void **)&h->d_prev, nf);
  if (e == cudaSuccess) e = cudaMalloc((void **)&h->d_counts, nl);
  if (e == cudaSuccess) e = cudaMalloc((void **)&h->d_offsets, nl);
  if (e == cudaSuccess) e = cudaMemcpy(h->d_cpos, c_pos, nf, cudaMemcpyHostToDevice);
  if (e == cudaSuccess) e = cudaMemcpy(h->d_cneg, c_neg, nf, cudaMemcpyHostToDevice);
  if (e != cudaSuccess) {
    evsim_destroy(h);
    return -(int)e;
  }
  *handle = h;
  return 0;
}

int evsim_reset(void *handle) {
  if (handle == nullptr) return -(int)cudaErrorInvalidValue;
  ((EvsimHandle *)handle)->initialised = 0;
  return 0;
}

// frames: T*H*W float32 log intensity (host); t_ns: T host, strictly increasing and after the
// previous frame (the caller checks this). On return *n_events is the count and out_* (host,
// preallocated to capacity) are filled. If the count exceeds capacity, returns 1 and *n_events
// holds the required capacity, with state NOT advanced.
int evsim_run(void *handle, const float *frames, const long long *t_ns, int n_frames,
              unsigned short *out_x, unsigned short *out_y, long long *out_t, signed char *out_p,
              long long capacity, long long *n_events) {
  if (handle == nullptr || frames == nullptr || t_ns == nullptr || n_frames <= 0 ||
      n_events == nullptr || capacity < 0) {
    return -(int)cudaErrorInvalidValue;
  }
  EvsimHandle *h = (EvsimHandle *)handle;
  long long n = h->n;
  int init = h->initialised ? 0 : 1;

  long long frames_len = (long long)n_frames * n;
  int rc = grow(&h->d_frames, &h->frames_cap, frames_len);
  if (rc != 0) return rc;
  {
    long long cap = h->tns_cap;
    rc = grow(&h->d_tns, &cap, (long long)n_frames);
    h->tns_cap = (int)cap;
    if (rc != 0) return rc;
  }
  EVSIM_CHECK(cudaMemcpy(h->d_frames, frames, (size_t)frames_len * sizeof(float),
                         cudaMemcpyHostToDevice));
  EVSIM_CHECK(cudaMemcpy(h->d_tns, t_ns, (size_t)n_frames * sizeof(long long),
                         cudaMemcpyHostToDevice));

  const int threads = 256;
  const int blocks = (int)((n + threads - 1) / threads);
  count_k<<<blocks, threads>>>(n, h->d_frames, h->d_tns, n_frames, h->d_cpos, h->d_cneg,
                               h->refractory_ns, init, h->d_lref, h->d_tlast, h->d_prev,
                               h->prev_t, h->width, h->d_counts);
  EVSIM_CHECK(cudaGetLastError());
  EVSIM_CHECK(cudaDeviceSynchronize());

  thrust::device_ptr<long long> counts_ptr(h->d_counts);
  thrust::device_ptr<long long> offsets_ptr(h->d_offsets);
  thrust::exclusive_scan(counts_ptr, counts_ptr + n, offsets_ptr);
  EVSIM_CHECK(cudaGetLastError());
  EVSIM_CHECK(cudaDeviceSynchronize());

  long long last_offset = 0, last_count = 0;
  EVSIM_CHECK(cudaMemcpy(&last_offset, h->d_offsets + (n - 1), sizeof(long long),
                         cudaMemcpyDeviceToHost));
  EVSIM_CHECK(cudaMemcpy(&last_count, h->d_counts + (n - 1), sizeof(long long),
                         cudaMemcpyDeviceToHost));
  long long total = last_offset + last_count;
  *n_events = total;
  if (total > capacity) return 1;

  if (total > 0) {
    if (out_x == nullptr || out_y == nullptr || out_t == nullptr || out_p == nullptr) {
      return -(int)cudaErrorInvalidValue;
    }
    long long cap_x = h->out_cap, cap_y = h->out_cap, cap_t = h->out_cap, cap_p = h->out_cap;
    rc = grow(&h->d_out_x, &cap_x, total);
    if (rc != 0) return rc;
    rc = grow(&h->d_out_y, &cap_y, total);
    if (rc != 0) return rc;
    rc = grow(&h->d_out_t, &cap_t, total);
    if (rc != 0) return rc;
    rc = grow(&h->d_out_p, &cap_p, total);
    if (rc != 0) return rc;
    h->out_cap = total > h->out_cap ? total : h->out_cap;
  }

  write_k<<<blocks, threads>>>(n, h->d_frames, h->d_tns, n_frames, h->d_cpos, h->d_cneg,
                               h->refractory_ns, init, h->d_lref, h->d_tlast, h->d_prev,
                               h->prev_t, h->width, h->d_offsets, h->d_out_x, h->d_out_y,
                               h->d_out_t, h->d_out_p);
  EVSIM_CHECK(cudaGetLastError());
  EVSIM_CHECK(cudaDeviceSynchronize());
  h->prev_t = t_ns[n_frames - 1];
  h->initialised = 1;

  if (total > 0) {
    EVSIM_CHECK(cudaMemcpy(out_x, h->d_out_x, (size_t)total * sizeof(unsigned short),
                           cudaMemcpyDeviceToHost));
    EVSIM_CHECK(cudaMemcpy(out_y, h->d_out_y, (size_t)total * sizeof(unsigned short),
                           cudaMemcpyDeviceToHost));
    EVSIM_CHECK(cudaMemcpy(out_t, h->d_out_t, (size_t)total * sizeof(long long),
                           cudaMemcpyDeviceToHost));
    EVSIM_CHECK(cudaMemcpy(out_p, h->d_out_p, (size_t)total * sizeof(signed char),
                           cudaMemcpyDeviceToHost));
  }
  return 0;
}

int evsim_destroy(void *handle) {
  if (handle == nullptr) return 0;
  EvsimHandle *h = (EvsimHandle *)handle;
  cudaError_t first = cudaSuccess;
#define EVSIM_FREE(p)                                       \
  do {                                                      \
    if ((p) != nullptr) {                                   \
      cudaError_t e_ = cudaFree(p);                         \
      if (e_ != cudaSuccess && first == cudaSuccess) first = e_; \
    }                                                       \
  } while (0)
  EVSIM_FREE(h->d_cpos);
  EVSIM_FREE(h->d_cneg);
  EVSIM_FREE(h->d_lref);
  EVSIM_FREE(h->d_tlast);
  EVSIM_FREE(h->d_prev);
  EVSIM_FREE(h->d_frames);
  EVSIM_FREE(h->d_tns);
  EVSIM_FREE(h->d_counts);
  EVSIM_FREE(h->d_offsets);
  EVSIM_FREE(h->d_out_x);
  EVSIM_FREE(h->d_out_y);
  EVSIM_FREE(h->d_out_t);
  EVSIM_FREE(h->d_out_p);
#undef EVSIM_FREE
  free(h);
  return first == cudaSuccess ? 0 : -(int)first;
}

}  // extern "C"
