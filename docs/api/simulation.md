# Simulation API Reference

`evlib.simulation` turns intensity frames into events with the ESIM threshold-crossing model. The kernel is Rust (`src/ev_simulation/`, exposed as `evlib.simulation_rs`), runs rows-parallel on the CPU with rayon or on an NVIDIA GPU through a runtime-loaded CUDA library, and returns Polars DataFrames in the evlib schema. It needs no PyTorch.

## The model

1. Each frame is converted to log intensity: `L = ln(I / 255 + log_eps)` (uint8 input) or taken as-is (float32 input).
2. Every pixel keeps a reference level `l_ref` (set from the first frame) and the time of its last event.
3. Between two frames the log intensity is interpolated linearly in time; each crossing of `l_ref + c_pos` (or `l_ref - c_neg`) emits one event of polarity `+1` (or `-1`) at the crossing time and moves `l_ref` to the crossed level.
4. Crossing times are floored to whole nanoseconds; an event inside the refractory period of the pixel is dropped but still moves `l_ref`.
5. `threshold_sigma > 0` draws per-pixel thresholds from a normal distribution around `c_pos` and `c_neg` with a fixed `seed`, so runs are reproducible.

## `simulate_frames`

```text
def simulate_frames(frames, timestamps_ns, config=None, sort=True) -> pl.DataFrame
```

`frames` is a `(T, H, W)` array, `uint8` intensity or `float32` log intensity; `timestamps_ns` is `int64` nanoseconds, strictly increasing. The result has columns `x` Int16, `y` Int16, `t` Duration(us) and `polarity` Int8, sorted by `t` unless `sort=False`. With `sort=False` (and for `ESIMSimulator.step_ns`) events are grouped by row chunk and the order is not stable across thread counts. The first frame only initialises the state, so it emits no events.

```python
import numpy as np
from PIL import Image
from evlib.simulation import ESIMConfig, simulate_frames

root = "data/slider_depth"
frames, t_ns = [], []
for line in open(f"{root}/images.txt"):
    secs, rel = line.split()
    frames.append(np.asarray(Image.open(f"{root}/{rel}").convert("L")))
    t_ns.append(round(float(secs) * 1e9))
frames = np.stack(frames)                       # (87, 180, 240) uint8
t_ns = np.asarray(t_ns, dtype=np.int64)

config = ESIMConfig(positive_threshold=0.2, negative_threshold=0.2, device="cpu")
events = simulate_frames(frames, t_ns, config)  # 6,650,175 events, sorted by t
print(events.schema)
```

`ESIMConfig` fields: `positive_threshold=0.2`, `negative_threshold=0.2`, `refractory_period_ms=0.0`, `log_eps=1e-3`, `threshold_sigma=0.0`, `seed=0`, `device="auto"`. Presets are in `ESIM_CONFIGS` (`default`, `high_sensitivity`, `low_sensitivity`, `low_noise`, `mismatch`) via `get_esim_config(name)`.

## `ESIMSimulator`

A stateful simulator for streaming input. `process_frame(frame, seconds)` takes one `(H, W)` grey or `(H, W, 3)` RGB uint8 frame (BT.601 luma) or a float32 log frame and returns `(x, y, t_s, p)` NumPy arrays; `step_ns(frame, t_ns)` is the raw form; `simulate(frames, timestamps_ns, sort=True)` feeds a whole `(T, H, W)` slice and returns a DataFrame. State carries over between calls, so slices give the same events as one `simulate_frames` call. `reset()` clears the state; `thresholds()` returns the per-pixel `(c_pos, c_neg)` maps; `device` reports the resolved backend.

```python
import numpy as np
import polars as pl
from PIL import Image
from evlib.simulation import ESIMConfig, ESIMSimulator

root = "data/slider_depth"
lines = [line.split() for line in open(f"{root}/images.txt")]

sim = ESIMSimulator(ESIMConfig(), width=240, height=180)
for secs, rel in lines[:20]:
    frame = np.asarray(Image.open(f"{root}/{rel}").convert("L"))
    x, y, t_s, p = sim.process_frame(frame, float(secs))   # first call only initialises

# Batches keep the state too: slices give the same events as one call.
frames = np.stack([np.asarray(Image.open(f"{root}/{rel}").convert("L")) for _, rel in lines])
t_ns = np.asarray([round(float(secs) * 1e9) for secs, _ in lines], dtype=np.int64)
sim.reset()
chunks = [sim.simulate(frames[a:a + 32], t_ns[a:a + 32]) for a in range(0, len(frames), 32)]
events = pl.concat(chunks)                      # 6,650,175 events, same as simulate_frames
```

## `VideoToEvents`

`VideoToEvents(esim_config, video_config)` decodes a video file with OpenCV (`pip install evlib[plot]`) and simulates it: `process_video(path)` returns one DataFrame (chunked internally for long clips), `process_frames_streaming(path, chunk_frames=64)` yields one DataFrame per chunk with the state kept between chunks, and `get_video_info(path)` reports the source properties. `VideoConfig` sets `width`, `height`, `fps`, `start_time`, `end_time`, `frame_skip` and `grayscale`. `evlib.simulation.video_to_events(path, esim_config, video_config)` is the one-call form.

## Front end

evlib takes frames plus timestamps. Frame interpolation and video decode for VID2E-style upsampling (FILM, SuperSloMo, adaptive bisection) live outside evlib; the [lumin](https://github.com/tallamjr/lumin) front end produces the upsampled frame stacks and hands them to `simulate_frames`.

## Devices and the CUDA build

`ESIMConfig(device=...)` selects the backend: `"cpu"` (rayon over rows, all cores), `"cuda"`, or `"auto"` (CUDA when available, else CPU). `evlib.simulation_rs.cuda_available()` reports whether the CUDA backend can run. Both backends use float32 state and float64 crossing times and give bit-identical event sets; `tests/test_simulation_cuda.py` checks that on the slider_depth frames.

The CUDA backend is a runtime-loaded shared library, so the default wheel has no CUDA dependency:

```bash
# Build the kernel (nvcc on PATH or NVCC=/path/to/nvcc; arch defaults to native)
scripts/build_cuda_kernels.sh target/cuda sm_89
export EVLIB_CUDA_SIM_LIB=$PWD/target/cuda/libevsim.so

# Build evlib with the loader
maturin develop --release --features cuda
python -c "import evlib; print(evlib.simulation_rs.cuda_available())"   # True
```

`EVLIB_CUDA_SIM_LIB` defaults to `libevsim.so` on the loader path. Requesting `device="cuda"` without the backend raises `RuntimeError`.

How the CUDA path runs a batch: uint8 frames are copied into a pinned staging buffer and uploaded as bytes (the log LUT is applied on the device; float32 log frames are uploaded as they are), one thread per pixel counts events, a scan gives per-pixel offsets, a second pass writes the events, and with `sort=True` the batch is sorted by time on the device (radix sort on the 64-bit timestamp, stable for equal timestamps) before the columns are copied back through pinned memory. Device and pinned buffers are owned by the simulator and grow geometrically, so there is no per-call allocation on the device. Feeding 32 frames per call is enough to reach the numbers below; larger batches gain a little more.

## Conformance with rpg_vid2e

`tests/test_vid2e_conformance.py` compares the kernel with `esim_torch` from [rpg_vid2e](https://github.com/uzh-rpg/rpg_vid2e) (commit `ecbb11a`) on the 87 slider_depth frames at thresholds 0.2/0.2, against tracked reference digests. Measured on 2026-08-16: total and positive event counts agree within 0.121 % and 0.120 % (bar 0.25 %); per-frame counts within 0.75 % max; 10x10 block counts within 1.64 % max; the first 200 events sorted by `(t, y, x, p)` have identical `x`, `y`, `p` and `t` within 1 ns. The residual comes from exact ties: a uint8 pixel that returns to a grey level seen before makes `L1` equal `l_ref + k*c` exactly, and float32 rounding resolves the tie differently in the two kernels (98.4 % of the divergent pixel cells). esim_torch also computes event times in float32, so timestamps are only compared on the early sample where one float32 ulp is below 1 ns.

## Benchmark

Measured 2026-08-17 on arg1 (AMD Threadripper 7960X, 24 cores / 48 threads; RTX 4090; `lib/bench_simulation.py`) on the gate 1 Big Buck Bunny frame folders at thresholds 0.2/0.2, uint8 frames, median of 3 after 1 warm-up, default glibc allocator. "kernel" is the unsorted raw-array call; "wall" is `ESIMSimulator.simulate` (sorted Polars DataFrame); "device" is the CUDA kernel alone (frames resident on the device, events left on the device, per-stage CUDA event times). The stack is fed in slices through one persistent simulator.

| input | events | backend | batch | kernel s | events/s (kernel) | wall s | events/s (wall) | events/s (device) | video-hours per GPU-day |
|---|---|---|---|---|---|---|---|---|---|
| 900 frames 320x320 (29.97 s) | 21,212,272 | CPU 48 threads | 256 | 0.047 | 451 M | 0.163 | 130 M | | 4,422 |
| 900 frames 320x320 (29.97 s) | 21,212,272 | CUDA | 32 | 0.052 | 406 M | 0.059 | 362 M | 1,445 M | 12,273 |
| 900 frames 320x320 (29.97 s) | 21,212,272 | CUDA | whole | 0.036 | 585 M | 0.045 | 471 M | 1,847 M | 15,982 |
| 900 frames 640x480 (29.97 s) | 67,819,884 | CPU 48 threads | 256 | 0.127 | 534 M | 0.463 | 147 M | | 1,554 |
| 900 frames 640x480 (29.97 s) | 67,819,884 | CUDA | 32 | 0.127 | 533 M | 0.146 | 465 M | 1,892 M | 4,935 |
| 900 frames 640x480 (29.97 s) | 67,819,884 | CUDA | whole | 0.112 | 603 M | 0.133 | 509 M | 2,156 M | 5,393 |
| 8,000 upsampled frames 320x320 (16.73 s) | 26,568,646 | CPU 48 threads | whole | 0.212 | 125 M | 0.338 | 79 M | | 1,189 |
| 8,000 upsampled frames 320x320 (16.73 s) | 26,568,646 | CUDA | 256 | 0.124 | 213 M | 0.145 | 183 M | 521 M | 2,768 |
| 8,000 upsampled frames 640x480 (10.61 s) | 50,453,777 | CPU 48 threads | whole | 0.552 | 91 M | 0.805 | 63 M | | 316 |
| 8,000 upsampled frames 640x480 (10.61 s) | 50,453,777 | CUDA | 256 | 0.277 | 182 M | 0.304 | 166 M | 538 M | 836 |

For reference, rpg_vid2e's esim_torch CUDA kernel with the frames already on the GPU reaches 668.5 M events/s (320x320) and 643.1 M events/s (640x480) at 32 frames per launch on the same host, and its shipped PNG-to-npz script 7.2 M and 11.8 M events/s; the comparable evlib number is the "device" column (1.4 to 2.2 G events/s on the raw frames). One CPU thread (`RAYON_NUM_THREADS=1`) does 20.7 M events/s kernel-only on the 320x320 input, 3.2x the single-thread esim_py C++ path (6.5 M events/s on the same frames). On the upsampled inputs the CUDA path is bound by the frame upload (2.4 GB of uint8 frames for 50 M events at 640x480, about 48 bytes of input per event against 4 on the raw frames), so it runs at 120 to 220 M events/s there. Both backends give the same event sets; the CPU path is the fallback and is limited by the host sort and first-touch page faults on the output.
