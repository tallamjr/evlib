// CUDA backend for the evlib event simulator (evlib.simulation device="cuda").
//
// One thread per pixel walks a batch of T frames, float32 log intensity or
// uint8 intensity mapped through a 256-entry log LUT on the device. Pass 1
// counts events on a copy of the pixel state; an exclusive scan of the counts
// gives each pixel its write offset; pass 2 walks again, writes (t, packed
// pixel and polarity) at offset + k and commits the advanced state (l_ref,
// t_last, prev). With the sort flag the (t, packed) pairs are sorted by t on
// the device (cub radix sort, 64-bit signed key, 32-bit payload; only the key
// bits in which the batch's time range differs are sorted, ties keep the
// pixel-grouped order). A decode kernel then unpacks x, y, p.
//
// Two call styles share the pipeline:
//   evsim_run / evsim_run_u8: legacy, caller-owned host buffers of `capacity`
//     events; returns 1 with the required count and untouched state on overflow.
//   evsim_run2: one stream, pinned staging (evsim_stage) with async copies,
//     device and pinned host result buffers owned by the handle and grown
//     geometrically, so there is no capacity and no retry; returns pointers
//     into the pinned result buffers, valid until the next call.
//
// The per-pixel arithmetic is a transliteration of src/ev_simulation/pixel.rs:
// f32 reference and thresholds, f64 crossing fraction clamped to [0, 1],
// floored to whole nanoseconds, refractory test `t_ev - t_last >= refractory`
// with t_last == LLONG_MIN meaning "no event yet". The reference moves on
// every crossing, including crossings the refractory period drops.
//
// Threshold maps are generated on the host so CPU and CUDA share identical maps.
//
//   nvcc -O3 -std=c++17 -arch=sm_89 -cudart static -shared -Xcompiler -fPIC \
//        -o libevsim.so esim_kernel.cu

#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define EVSIM_CHECK(call)                                                       \
  do {                                                                          \
    cudaError_t evsim_err_ = (call);                                            \
    if (evsim_err_ != cudaSuccess) return -(int)evsim_err_;                     \
  } while (0)

static const long long NO_EVENT = LLONG_MIN;

// evsim_run2 flags.
#define EVSIM_SORT 1         // sort events by t on the device
#define EVSIM_STAGED 2       // frames are already in the pinned staging buffer (evsim_stage)
#define EVSIM_RESIDENT 4     // reuse the frames and t_ns of the previous call (benchmarks)
#define EVSIM_NO_DOWNLOAD 8  // leave the results on the device (benchmarks)

// Timing slots (ms): 0 whole call (host clock), 1 host copy into the staging
// buffer, 2 upload, 3 count, 4 scan, 5 write, 6 sort, 7 decode, 8 download.
#define EVSIM_N_TIMINGS 9
#define EVSIM_N_EVENTS 9

extern "C" int evsim_destroy(void *handle);

struct EvsimHandle {
  int width;
  int height;
  long long n;
  long long refractory_ns;
  cudaStream_t stream;
  // Persistent per-pixel state.
  float *d_cpos;
  float *d_cneg;
  float *d_lref;
  long long *d_tlast;
  float *d_prev;
  float *d_lut;  // 256 log values for uint8 frames (evsim_set_lut)
  int lut_set;
  long long prev_t;
  int initialised;
  // Frames of the current call, retained for EVSIM_RESIDENT.
  unsigned char *d_frames;
  long long frames_cap;  // bytes
  long long *d_tns;
  long long tns_cap;
  int resident_kind;
  int resident_frames;
  // Pinned host staging: frames (evsim_stage) and t_ns.
  unsigned char *h_stage;
  long long stage_cap;  // bytes
  long long *h_tns;
  long long h_tns_cap;
  long long *h_total;  // 2 pinned entries: last offset, last count
  // Per-pixel counts and offsets plus the cub scan scratch.
  long long *d_counts;
  long long *d_offsets;
  void *d_scan_tmp;
  size_t scan_tmp_cap;
  // Device event buffers: write targets, sort outputs, decoded columns.
  long long *d_t;
  unsigned int *d_pk;
  long long *d_t2;
  unsigned int *d_pk2;
  unsigned short *d_x;
  unsigned short *d_y;
  signed char *d_p;
  long long ev_cap;
  void *d_sort_tmp;
  size_t sort_tmp_cap;
  // Pinned host result buffers returned by evsim_run2.
  unsigned short *h_x;
  unsigned short *h_y;
  long long *h_t;
  signed char *h_p;
  long long res_cap;
  long long last_total;
  // Optional per-stage timing (evsim_set_timing).
  int timing;
  cudaEvent_t ev[EVSIM_N_EVENTS];
  double timings[EVSIM_N_TIMINGS];
};

// Record timing event k on the handle's stream when timing is on.
static inline cudaError_t mark(EvsimHandle *h, int k) {
  if (!h->timing) return cudaSuccess;
  return cudaEventRecord(h->ev[k], h->stream);
}

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

// Frame sample as float32 log intensity: KIND 0 reads float32, KIND 1 reads uint8
// through the LUT (in shared memory).
template <int KIND>
__device__ __forceinline__ float sample(const void *frames, long long idx, const float *lut) {
  if (KIND == 0) return ((const float *)frames)[idx];
  return lut[((const unsigned char *)frames)[idx]];
}

// Walk pixel i across the batch. WRITE=false only counts; WRITE=true writes
// (t, packed pixel index and polarity) at out_*[offset + k] and stores the
// advanced state. Packed = (i << 1) | (polarity > 0).
template <bool WRITE, int KIND>
__device__ __forceinline__ long long walk_pixel(
    long long i, long long n, const void *frames, const float *lut, const long long *t_ns,
    int n_frames, float cpos, float cneg, long long refractory_ns, int init, float lref_in,
    long long tlast_in, float prev_in, long long prev_t, long long offset, long long *out_t,
    unsigned int *out_pk, float *lref_out, long long *tlast_out) {
  float l_ref;
  long long t_last;
  float l0;
  long long t0;
  int k_start;
  if (init) {
    l0 = sample<KIND>(frames, i, lut);
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
  const unsigned int pk_pos = ((unsigned int)i << 1) | 1u;
  const unsigned int pk_neg = (unsigned int)i << 1;
  long long count = 0;
  for (int k = k_start; k < n_frames; ++k) {
    float l1 = sample<KIND>(frames, (long long)k * n + i, lut);
    long long t1 = t_ns[k];
    if (l1 > l0) {
      while (l1 >= l_ref + cpos) {
        float lc = l_ref + cpos;
        long long t_ev = crossing_time(l0, t0, l1, t1, lc);
        l_ref = lc;
        if (t_last == NO_EVENT || t_ev - t_last >= refractory_ns) {
          if (WRITE) {
            long long o = offset + count;
            out_t[o] = t_ev;
            out_pk[o] = pk_pos;
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
            out_t[o] = t_ev;
            out_pk[o] = pk_neg;
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

// Load the 256-entry LUT into shared memory (KIND 1 only); every thread must reach it.
template <int KIND>
__device__ __forceinline__ void load_lut(const float *d_lut, float *s_lut) {
  if (KIND == 1) {
    for (int j = threadIdx.x; j < 256; j += blockDim.x) s_lut[j] = d_lut[j];
    __syncthreads();
  }
}

template <int KIND>
__global__ void count_k(long long n, const void *frames, const float *d_lut,
                        const long long *t_ns, int n_frames, const float *cpos, const float *cneg,
                        long long refractory_ns, int init, const float *lref,
                        const long long *tlast, const float *prev, long long prev_t,
                        long long *counts) {
  __shared__ float s_lut[256];
  load_lut<KIND>(d_lut, s_lut);
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  counts[i] = walk_pixel<false, KIND>(i, n, frames, s_lut, t_ns, n_frames, cpos[i], cneg[i],
                                      refractory_ns, init, lref[i], tlast[i], prev[i], prev_t, 0,
                                      nullptr, nullptr, nullptr, nullptr);
}

template <int KIND>
__global__ void write_k(long long n, const void *frames, const float *d_lut,
                        const long long *t_ns, int n_frames, const float *cpos, const float *cneg,
                        long long refractory_ns, int init, float *lref, long long *tlast,
                        float *prev, long long prev_t, const long long *offsets, long long *out_t,
                        unsigned int *out_pk) {
  __shared__ float s_lut[256];
  load_lut<KIND>(d_lut, s_lut);
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float lref_new;
  long long tlast_new;
  walk_pixel<true, KIND>(i, n, frames, s_lut, t_ns, n_frames, cpos[i], cneg[i], refractory_ns,
                         init, lref[i], tlast[i], prev[i], prev_t, offsets[i], out_t, out_pk,
                         &lref_new, &tlast_new);
  lref[i] = lref_new;
  tlast[i] = tlast_new;
  prev[i] = sample<KIND>(frames, (long long)(n_frames - 1) * n + i, s_lut);
}

// Unpack (pixel << 1 | polarity) into x, y and -1/1 polarity.
__global__ void decode_k(long long total, const unsigned int *pk, int width, unsigned short *x,
                         unsigned short *y, signed char *p) {
  long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= total) return;
  unsigned int v = pk[e];
  unsigned int i = v >> 1;
  x[e] = (unsigned short)(i % (unsigned int)width);
  y[e] = (unsigned short)(i / (unsigned int)width);
  p[e] = (v & 1u) ? 1 : -1;
}

// Device buffer growth: geometric, so repeated batches settle at a fixed size.
static long long grown_cap(long long cap, long long needed) {
  long long next = cap + cap / 2;
  return next > needed ? next : needed;
}

template <typename T>
static int grow_device(T **ptr, long long *cap, long long needed) {
  if (needed <= *cap) return 0;
  long long next = grown_cap(*cap, needed);
  if (*ptr != nullptr) EVSIM_CHECK(cudaFree(*ptr));
  *ptr = nullptr;
  *cap = 0;
  EVSIM_CHECK(cudaMalloc((void **)ptr, (size_t)next * sizeof(T)));
  *cap = next;
  return 0;
}

template <typename T>
static int grow_pinned(T **ptr, long long *cap, long long needed) {
  if (needed <= *cap) return 0;
  long long next = grown_cap(*cap, needed);
  if (*ptr != nullptr) EVSIM_CHECK(cudaFreeHost(*ptr));
  *ptr = nullptr;
  *cap = 0;
  EVSIM_CHECK(cudaHostAlloc((void **)ptr, (size_t)next * sizeof(T), cudaHostAllocDefault));
  *cap = next;
  return 0;
}

// Scratch for cub calls: exact size, kept between calls.
static int grow_scratch(void **ptr, size_t *cap, size_t needed) {
  if (needed <= *cap) return 0;
  if (*ptr != nullptr) EVSIM_CHECK(cudaFree(*ptr));
  *ptr = nullptr;
  *cap = 0;
  EVSIM_CHECK(cudaMalloc(ptr, needed));
  *cap = needed;
  return 0;
}

static int grow_events(EvsimHandle *h, long long total) {
  if (total <= h->ev_cap) return 0;
  const long long cap = h->ev_cap;
  long long c = cap;
  int rc = grow_device(&h->d_t, &c, total);
  if (rc != 0) return rc;
  c = cap;
  rc = grow_device(&h->d_pk, &c, total);
  if (rc != 0) return rc;
  c = cap;
  rc = grow_device(&h->d_t2, &c, total);
  if (rc != 0) return rc;
  c = cap;
  rc = grow_device(&h->d_pk2, &c, total);
  if (rc != 0) return rc;
  c = cap;
  rc = grow_device(&h->d_x, &c, total);
  if (rc != 0) return rc;
  c = cap;
  rc = grow_device(&h->d_y, &c, total);
  if (rc != 0) return rc;
  c = cap;
  rc = grow_device(&h->d_p, &c, total);
  if (rc != 0) return rc;
  h->ev_cap = c;
  return 0;
}

static int grow_results(EvsimHandle *h, long long total) {
  if (total <= h->res_cap) return 0;
  const long long cap = h->res_cap;
  long long c = cap;
  int rc = grow_pinned(&h->h_x, &c, total);
  if (rc != 0) return rc;
  c = cap;
  rc = grow_pinned(&h->h_y, &c, total);
  if (rc != 0) return rc;
  c = cap;
  rc = grow_pinned(&h->h_t, &c, total);
  if (rc != 0) return rc;
  c = cap;
  rc = grow_pinned(&h->h_p, &c, total);
  if (rc != 0) return rc;
  h->res_cap = c;
  return 0;
}

static inline int blocks_for(long long items, int threads) {
  return (int)((items + threads - 1) / threads);
}

// Pass 1 and the scan on the stream; the total is read back through pinned memory
// (this synchronises the stream).
template <int KIND>
static int count_and_scan(EvsimHandle *h, int n_frames, int init, long long *total) {
  const int threads = 256;
  const long long n = h->n;
  const int blocks = blocks_for(n, threads);
  count_k<KIND><<<blocks, threads, 0, h->stream>>>(
      n, h->d_frames, h->d_lut, h->d_tns, n_frames, h->d_cpos, h->d_cneg, h->refractory_ns,
      init, h->d_lref, h->d_tlast, h->d_prev, h->prev_t, h->d_counts);
  EVSIM_CHECK(cudaGetLastError());
  EVSIM_CHECK(mark(h, 3));
  size_t need = 0;
  EVSIM_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, need, h->d_counts, h->d_offsets, n,
                                            h->stream));
  int rc = grow_scratch(&h->d_scan_tmp, &h->scan_tmp_cap, need);
  if (rc != 0) return rc;
  EVSIM_CHECK(cub::DeviceScan::ExclusiveSum(h->d_scan_tmp, h->scan_tmp_cap, h->d_counts,
                                            h->d_offsets, n, h->stream));
  EVSIM_CHECK(cudaMemcpyAsync(&h->h_total[0], h->d_offsets + (n - 1), sizeof(long long),
                              cudaMemcpyDeviceToHost, h->stream));
  EVSIM_CHECK(cudaMemcpyAsync(&h->h_total[1], h->d_counts + (n - 1), sizeof(long long),
                              cudaMemcpyDeviceToHost, h->stream));
  EVSIM_CHECK(mark(h, 4));
  EVSIM_CHECK(cudaStreamSynchronize(h->stream));
  *total = h->h_total[0] + h->h_total[1];
  return 0;
}

// Pass 2 into d_t / d_pk (grown to total), then commit the host-side state.
template <int KIND>
static int write_pass(EvsimHandle *h, const long long *t_ns, int n_frames, int init,
                      long long total) {
  int rc = grow_events(h, total);
  if (rc != 0) return rc;
  const int threads = 256;
  const int blocks = blocks_for(h->n, threads);
  write_k<KIND><<<blocks, threads, 0, h->stream>>>(
      h->n, h->d_frames, h->d_lut, h->d_tns, n_frames, h->d_cpos, h->d_cneg, h->refractory_ns,
      init, h->d_lref, h->d_tlast, h->d_prev, h->prev_t, h->d_offsets, h->d_t, h->d_pk);
  EVSIM_CHECK(cudaGetLastError());
  EVSIM_CHECK(mark(h, 5));
  h->prev_t = t_ns[n_frames - 1];
  h->initialised = 1;
  return 0;
}

// Sort (t, packed) pairs by t into d_t2 / d_pk2. Only the bits in which the batch's
// time range [t_lo, t_hi] differs are sorted; cub handles the signed key.
static int sort_pass(EvsimHandle *h, long long total, long long t_lo, long long t_hi) {
  unsigned long long diff = (unsigned long long)(t_lo ^ t_hi);
  int end_bit = 1;
  if (diff != 0) {
    int lead = 0;
    while (((diff >> (63 - lead)) & 1ull) == 0) ++lead;
    end_bit = 64 - lead;
  }
  size_t need = 0;
  EVSIM_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, need, h->d_t, h->d_t2, h->d_pk, h->d_pk2,
                                              total, 0, end_bit, h->stream));
  int rc = grow_scratch(&h->d_sort_tmp, &h->sort_tmp_cap, need);
  if (rc != 0) return rc;
  EVSIM_CHECK(cub::DeviceRadixSort::SortPairs(h->d_sort_tmp, h->sort_tmp_cap, h->d_t, h->d_t2,
                                              h->d_pk, h->d_pk2, total, 0, end_bit, h->stream));
  return 0;
}

static int decode_pass(EvsimHandle *h, const unsigned int *pk, long long total) {
  if (total == 0) return 0;
  const int threads = 256;
  decode_k<<<blocks_for(total, threads), threads, 0, h->stream>>>(total, pk, h->width, h->d_x,
                                                                   h->d_y, h->d_p);
  EVSIM_CHECK(cudaGetLastError());
  return 0;
}

static int download_async(EvsimHandle *h, const long long *d_t, long long total,
                          unsigned short *x, unsigned short *y, long long *t, signed char *p) {
  if (total == 0) return 0;
  EVSIM_CHECK(cudaMemcpyAsync(x, h->d_x, (size_t)total * sizeof(unsigned short),
                              cudaMemcpyDeviceToHost, h->stream));
  EVSIM_CHECK(cudaMemcpyAsync(y, h->d_y, (size_t)total * sizeof(unsigned short),
                              cudaMemcpyDeviceToHost, h->stream));
  EVSIM_CHECK(cudaMemcpyAsync(t, d_t, (size_t)total * sizeof(long long), cudaMemcpyDeviceToHost,
                              h->stream));
  EVSIM_CHECK(cudaMemcpyAsync(p, h->d_p, (size_t)total * sizeof(signed char),
                              cudaMemcpyDeviceToHost, h->stream));
  return 0;
}

static int finish_timings(EvsimHandle *h, std::chrono::steady_clock::time_point host_start,
                          double stage_ms) {
  if (!h->timing) return 0;
  EVSIM_CHECK(cudaEventSynchronize(h->ev[EVSIM_N_EVENTS - 1]));
  // Slot k + 1 is the span from event k to event k + 1 (event 0 to 1 is the host copy).
  for (int k = 1; k + 1 < EVSIM_N_EVENTS; ++k) {
    float ms = 0.f;
    EVSIM_CHECK(cudaEventElapsedTime(&ms, h->ev[k], h->ev[k + 1]));
    h->timings[k + 1] = ms;
  }
  h->timings[1] = stage_ms;
  h->timings[0] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                            host_start)
                      .count();
  return 0;
}

// Make room for a batch of n_frames frames of `elem` bytes per pixel and T timestamps.
static int reserve_batch(EvsimHandle *h, int n_frames, size_t elem, long long *frames_bytes) {
  long long bytes = (long long)n_frames * h->n * (long long)elem;
  int rc = grow_device(&h->d_frames, &h->frames_cap, bytes);
  if (rc != 0) return rc;
  rc = grow_device(&h->d_tns, &h->tns_cap, (long long)n_frames);
  if (rc != 0) return rc;
  rc = grow_pinned(&h->h_tns, &h->h_tns_cap, (long long)n_frames);
  if (rc != 0) return rc;
  *frames_bytes = bytes;
  return 0;
}

// Legacy body of evsim_run (KIND 0) and evsim_run_u8 (KIND 1): synchronous copies
// from and to caller memory, capacity check between pass 1 and pass 2.
template <int KIND>
static int run_legacy(void *handle, const void *frames, const long long *t_ns, int n_frames,
                      unsigned short *out_x, unsigned short *out_y, long long *out_t,
                      signed char *out_p, long long capacity, long long *n_events) {
  if (handle == nullptr || frames == nullptr || t_ns == nullptr || n_frames <= 0 ||
      n_events == nullptr || capacity < 0) {
    return -(int)cudaErrorInvalidValue;
  }
  EvsimHandle *h = (EvsimHandle *)handle;
  if (KIND == 1 && !h->lut_set) return -(int)cudaErrorInvalidValue;
  const int init = h->initialised ? 0 : 1;
  auto host_start = std::chrono::steady_clock::now();
  long long frames_bytes = 0;
  int rc = reserve_batch(h, n_frames, KIND == 0 ? sizeof(float) : 1, &frames_bytes);
  if (rc != 0) return rc;
  EVSIM_CHECK(mark(h, 0));
  EVSIM_CHECK(mark(h, 1));
  EVSIM_CHECK(cudaMemcpyAsync(h->d_frames, frames, (size_t)frames_bytes, cudaMemcpyHostToDevice,
                              h->stream));
  EVSIM_CHECK(cudaMemcpyAsync(h->d_tns, t_ns, (size_t)n_frames * sizeof(long long),
                              cudaMemcpyHostToDevice, h->stream));
  EVSIM_CHECK(mark(h, 2));
  h->resident_kind = KIND;
  h->resident_frames = n_frames;
  long long total = 0;
  rc = count_and_scan<KIND>(h, n_frames, init, &total);
  if (rc != 0) return rc;
  *n_events = total;
  if (total > capacity) return 1;
  if (total > 0 &&
      (out_x == nullptr || out_y == nullptr || out_t == nullptr || out_p == nullptr)) {
    return -(int)cudaErrorInvalidValue;
  }
  rc = write_pass<KIND>(h, t_ns, n_frames, init, total);
  if (rc != 0) return rc;
  EVSIM_CHECK(mark(h, 6));
  rc = decode_pass(h, h->d_pk, total);
  if (rc != 0) return rc;
  EVSIM_CHECK(mark(h, 7));
  rc = download_async(h, h->d_t, total, out_x, out_y, out_t, out_p);
  if (rc != 0) return rc;
  EVSIM_CHECK(mark(h, 8));
  EVSIM_CHECK(cudaStreamSynchronize(h->stream));
  h->last_total = total;
  return finish_timings(h, host_start, 0.0);
}

extern "C" {

// Returns 0 on success, negative CUDA error code otherwise. All arrays are host pointers.
int evsim_create(int width, int height, const float *c_pos, const float *c_neg,
                 long long refractory_ns, void **handle) {
  if (width <= 0 || height <= 0 || c_pos == nullptr || c_neg == nullptr || handle == nullptr) {
    return -(int)cudaErrorInvalidValue;
  }
  // The packed payload keeps the pixel index in 31 bits.
  if ((long long)width * (long long)height > 0x7FFFFFFFLL) return -(int)cudaErrorInvalidValue;
  EvsimHandle *h = (EvsimHandle *)calloc(1, sizeof(EvsimHandle));
  if (h == nullptr) return -(int)cudaErrorMemoryAllocation;
  h->width = width;
  h->height = height;
  h->n = (long long)width * (long long)height;
  h->refractory_ns = refractory_ns;
  h->initialised = 0;
  size_t nf = (size_t)h->n * sizeof(float);
  size_t nl = (size_t)h->n * sizeof(long long);
  cudaError_t e = cudaStreamCreateWithFlags(&h->stream, cudaStreamNonBlocking);
  if (e == cudaSuccess) e = cudaMalloc((void **)&h->d_cpos, nf);
  if (e == cudaSuccess) e = cudaMalloc((void **)&h->d_cneg, nf);
  if (e == cudaSuccess) e = cudaMalloc((void **)&h->d_lref, nf);
  if (e == cudaSuccess) e = cudaMalloc((void **)&h->d_tlast, nl);
  if (e == cudaSuccess) e = cudaMalloc((void **)&h->d_prev, nf);
  if (e == cudaSuccess) e = cudaMalloc((void **)&h->d_counts, nl);
  if (e == cudaSuccess) e = cudaMalloc((void **)&h->d_offsets, nl);
  if (e == cudaSuccess) e = cudaMalloc((void **)&h->d_lut, 256 * sizeof(float));
  if (e == cudaSuccess)
    e = cudaHostAlloc((void **)&h->h_total, 2 * sizeof(long long), cudaHostAllocDefault);
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
  return run_legacy<0>(handle, frames, t_ns, n_frames, out_x, out_y, out_t, out_p, capacity,
                       n_events);
}

// Set the 256-entry float32 log LUT used for uint8 frames (host pointer).
int evsim_set_lut(void *handle, const float *lut) {
  if (handle == nullptr || lut == nullptr) return -(int)cudaErrorInvalidValue;
  EvsimHandle *h = (EvsimHandle *)handle;
  EVSIM_CHECK(cudaMemcpy(h->d_lut, lut, 256 * sizeof(float), cudaMemcpyHostToDevice));
  h->lut_set = 1;
  return 0;
}

// Same as evsim_run for T*H*W uint8 intensity frames; the LUT from evsim_set_lut is
// applied on the device. Fails with cudaErrorInvalidValue if no LUT was set.
int evsim_run_u8(void *handle, const unsigned char *frames, const long long *t_ns, int n_frames,
                 unsigned short *out_x, unsigned short *out_y, long long *out_t,
                 signed char *out_p, long long capacity, long long *n_events) {
  return run_legacy<1>(handle, frames, t_ns, n_frames, out_x, out_y, out_t, out_p, capacity,
                       n_events);
}

// Pinned staging buffer of at least `bytes` bytes for the next evsim_run2 call with
// EVSIM_STAGED. The pointer stays valid until the next evsim_stage or evsim_destroy.
int evsim_stage(void *handle, long long bytes, void **pinned) {
  if (handle == nullptr || pinned == nullptr || bytes < 0) return -(int)cudaErrorInvalidValue;
  EvsimHandle *h = (EvsimHandle *)handle;
  int rc = grow_pinned(&h->h_stage, &h->stage_cap, bytes);
  if (rc != 0) return rc;
  *pinned = h->h_stage;
  return 0;
}

// Run one batch. kind: 0 float32 log frames, 1 uint8 frames. flags: EVSIM_* bits.
// frames: host pointer (ignored with EVSIM_STAGED or EVSIM_RESIDENT); t_ns: T host
// timestamps (always read). Results are returned as pointers into pinned host
// buffers owned by the handle (valid until the next call); *n_events is the count.
// With EVSIM_NO_DOWNLOAD the pointers are null and the events stay on the device.
int evsim_run2(void *handle, int kind, const void *frames, const long long *t_ns, int n_frames,
               int flags, const unsigned short **out_x, const unsigned short **out_y,
               const long long **out_t, const signed char **out_p, long long *n_events) {
  if (handle == nullptr || t_ns == nullptr || n_frames <= 0 || n_events == nullptr ||
      out_x == nullptr || out_y == nullptr || out_t == nullptr || out_p == nullptr ||
      (kind != 0 && kind != 1)) {
    return -(int)cudaErrorInvalidValue;
  }
  EvsimHandle *h = (EvsimHandle *)handle;
  if (kind == 1 && !h->lut_set) return -(int)cudaErrorInvalidValue;
  const int staged = (flags & EVSIM_STAGED) != 0;
  const int resident = (flags & EVSIM_RESIDENT) != 0;
  if (!staged && !resident && frames == nullptr) return -(int)cudaErrorInvalidValue;
  if (resident && (h->resident_kind != kind || h->resident_frames != n_frames)) {
    return -(int)cudaErrorInvalidValue;
  }
  const int init = h->initialised ? 0 : 1;
  const long long t_lo = init ? t_ns[0] : h->prev_t;
  const long long t_hi = t_ns[n_frames - 1];
  auto host_start = std::chrono::steady_clock::now();
  const size_t elem = kind == 0 ? sizeof(float) : 1;
  long long frames_bytes = 0;
  int rc = reserve_batch(h, n_frames, elem, &frames_bytes);
  if (rc != 0) return rc;
  if (staged && h->stage_cap < frames_bytes) return -(int)cudaErrorInvalidValue;
  *out_x = nullptr;
  *out_y = nullptr;
  *out_t = nullptr;
  *out_p = nullptr;

  double stage_ms = 0.0;
  EVSIM_CHECK(mark(h, 0));
  if (!resident) {
    if (!staged) {
      // Pageable source: copy into the pinned staging buffer so the upload is async.
      auto s0 = std::chrono::steady_clock::now();
      rc = grow_pinned(&h->h_stage, &h->stage_cap, frames_bytes);
      if (rc != 0) return rc;
      memcpy(h->h_stage, frames, (size_t)frames_bytes);
      stage_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0)
                     .count();
    }
    memcpy(h->h_tns, t_ns, (size_t)n_frames * sizeof(long long));
    EVSIM_CHECK(mark(h, 1));
    EVSIM_CHECK(cudaMemcpyAsync(h->d_frames, h->h_stage, (size_t)frames_bytes,
                                cudaMemcpyHostToDevice, h->stream));
    EVSIM_CHECK(cudaMemcpyAsync(h->d_tns, h->h_tns, (size_t)n_frames * sizeof(long long),
                                cudaMemcpyHostToDevice, h->stream));
    h->resident_kind = kind;
    h->resident_frames = n_frames;
  } else {
    EVSIM_CHECK(mark(h, 1));
  }
  EVSIM_CHECK(mark(h, 2));

  long long total = 0;
  rc = kind == 0 ? count_and_scan<0>(h, n_frames, init, &total)
                 : count_and_scan<1>(h, n_frames, init, &total);
  if (rc != 0) return rc;
  rc = kind == 0 ? write_pass<0>(h, t_ns, n_frames, init, total)
                 : write_pass<1>(h, t_ns, n_frames, init, total);
  if (rc != 0) return rc;

  const long long *res_t = h->d_t;
  const unsigned int *res_pk = h->d_pk;
  if ((flags & EVSIM_SORT) && total > 1) {
    rc = sort_pass(h, total, t_lo, t_hi);
    if (rc != 0) return rc;
    res_t = h->d_t2;
    res_pk = h->d_pk2;
  }
  EVSIM_CHECK(mark(h, 6));
  rc = decode_pass(h, res_pk, total);
  if (rc != 0) return rc;
  EVSIM_CHECK(mark(h, 7));
  if (!(flags & EVSIM_NO_DOWNLOAD)) {
    rc = grow_results(h, total);
    if (rc != 0) return rc;
    rc = download_async(h, res_t, total, h->h_x, h->h_y, h->h_t, h->h_p);
    if (rc != 0) return rc;
    *out_x = h->h_x;
    *out_y = h->h_y;
    *out_t = h->h_t;
    *out_p = h->h_p;
  }
  EVSIM_CHECK(mark(h, 8));
  EVSIM_CHECK(cudaStreamSynchronize(h->stream));
  h->last_total = total;
  *n_events = total;
  return finish_timings(h, host_start, stage_ms);
}

// Turn per-stage timing on or off for this handle (off by default).
int evsim_set_timing(void *handle, int on) {
  if (handle == nullptr) return -(int)cudaErrorInvalidValue;
  EvsimHandle *h = (EvsimHandle *)handle;
  if (on && !h->timing) {
    for (int k = 0; k < EVSIM_N_EVENTS; ++k) EVSIM_CHECK(cudaEventCreate(&h->ev[k]));
  }
  h->timing = on ? 1 : 0;
  return 0;
}

// Copy up to n stage times in milliseconds from the last timed run into out_ms.
// Returns the number of slots available (EVSIM_N_TIMINGS).
int evsim_last_timings(void *handle, double *out_ms, int n) {
  if (handle == nullptr || (out_ms == nullptr && n > 0)) return -(int)cudaErrorInvalidValue;
  EvsimHandle *h = (EvsimHandle *)handle;
  for (int k = 0; k < n && k < EVSIM_N_TIMINGS; ++k) out_ms[k] = h->timings[k];
  return EVSIM_N_TIMINGS;
}

int evsim_destroy(void *handle) {
  if (handle == nullptr) return 0;
  EvsimHandle *h = (EvsimHandle *)handle;
  cudaError_t first = cudaSuccess;
#define EVSIM_FREE(p)                                                \
  do {                                                               \
    if ((p) != nullptr) {                                            \
      cudaError_t e_ = cudaFree(p);                                  \
      if (e_ != cudaSuccess && first == cudaSuccess) first = e_;     \
    }                                                                \
  } while (0)
#define EVSIM_FREE_HOST(p)                                           \
  do {                                                               \
    if ((p) != nullptr) {                                            \
      cudaError_t e_ = cudaFreeHost(p);                              \
      if (e_ != cudaSuccess && first == cudaSuccess) first = e_;     \
    }                                                                \
  } while (0)
  if (h->stream != nullptr) {
    cudaError_t e_ = cudaStreamSynchronize(h->stream);
    if (e_ != cudaSuccess && first == cudaSuccess) first = e_;
  }
  EVSIM_FREE(h->d_cpos);
  EVSIM_FREE(h->d_cneg);
  EVSIM_FREE(h->d_lref);
  EVSIM_FREE(h->d_tlast);
  EVSIM_FREE(h->d_prev);
  EVSIM_FREE(h->d_lut);
  EVSIM_FREE(h->d_frames);
  EVSIM_FREE(h->d_tns);
  EVSIM_FREE(h->d_counts);
  EVSIM_FREE(h->d_offsets);
  EVSIM_FREE(h->d_scan_tmp);
  EVSIM_FREE(h->d_t);
  EVSIM_FREE(h->d_pk);
  EVSIM_FREE(h->d_t2);
  EVSIM_FREE(h->d_pk2);
  EVSIM_FREE(h->d_x);
  EVSIM_FREE(h->d_y);
  EVSIM_FREE(h->d_p);
  EVSIM_FREE(h->d_sort_tmp);
  EVSIM_FREE_HOST(h->h_stage);
  EVSIM_FREE_HOST(h->h_tns);
  EVSIM_FREE_HOST(h->h_total);
  EVSIM_FREE_HOST(h->h_x);
  EVSIM_FREE_HOST(h->h_y);
  EVSIM_FREE_HOST(h->h_t);
  EVSIM_FREE_HOST(h->h_p);
#undef EVSIM_FREE
#undef EVSIM_FREE_HOST
  if (h->timing) {
    for (int k = 0; k < EVSIM_N_EVENTS; ++k) {
      cudaError_t e_ = cudaEventDestroy(h->ev[k]);
      if (e_ != cudaSuccess && first == cudaSuccess) first = e_;
    }
  }
  if (h->stream != nullptr) {
    cudaError_t e_ = cudaStreamDestroy(h->stream);
    if (e_ != cudaSuccess && first == cudaSuccess) first = e_;
  }
  free(h);
  return first == cudaSuccess ? 0 : -(int)first;
}

}  // extern "C"
