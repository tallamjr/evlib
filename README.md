<img align="left" src="./evlogo.png" alt="" width="70">

# `evlib`: Event Camera Utilities in Rust

<div style="text-align: center;" align="center">

[![PyPI Version](https://img.shields.io/pypi/v/evlib.svg)](https://pypi.org/project/evlib/)
[![Python Versions](https://img.shields.io/pypi/pyversions/evlib.svg)](https://pypi.org/project/evlib/)
[![Python](https://github.com/tallamjr/evlib/actions/workflows/pytest.yml/badge.svg)](https://github.com/tallamjr/evlib/actions/workflows/pytest.yml)
[![Build](https://github.com/tallamjr/evlib/actions/workflows/build.yml/badge.svg)](https://github.com/tallamjr/evlib/actions/workflows/build.yml)
[![Codecov](https://codecov.io/gh/tallamjr/evlib/branch/master/graph/badge.svg)](https://codecov.io/gh/tallamjr/evlib)
[![License](https://img.shields.io/github/license/tallamjr/evlib)](https://github.com/tallamjr/evlib/blob/master/LICENSE.md)

</div>

A high-performance (or some might say: _blazingly fast_) implementation of event
camera utilities using Rust with Python bindings via PyO3.

This library is insipred by numerous event camera libraries such
[`event_utils`](https://github.com/TimoStoff/event_utils) Python library but
reimplemented in Rust for significantly better performance.

> [!Warning]
>
> **This is a super experimental project and will have frequent breaking changes.
> It is primary being developed as a learning project for understanding Event
> Camera data processing and Event-Vision algorithms.**

<!-- mtoc-start -->

* [⬇ Installation](#-installation)
  * [Development Setup](#development-setup)
* [🗺️ Roadmap and Current Features](#-roadmap-and-current-features)
* [🚀 Performance](#-performance)
  * [Single-core Performance](#single-core-performance)
  * [Multi-core vs Single-core Performance](#multi-core-vs-single-core-performance)
  * [Why Rust is Faster](#why-rust-is-faster)
* [⮑ Module Structure](#-module-structure)
  * [Basic Usage](#basic-usage)
  * [Loading Event Data](#loading-event-data)
  * [Event Augmentation](#event-augmentation)
  * [Event Transformations](#event-transformations)
  * [Event Representations (Voxel Grid)](#event-representations-voxel-grid)
  * [Event Visualisation](#event-visualisation)
  * [Event-to-Video Reconstruction](#event-to-video-reconstruction)
* [⚖️ License](#-license)

<!-- mtoc-end -->

## ⬇ Installation

```bash
# Using pip
pip install evlib

# Using uv (recommended)
uv pip install evlib
```

For development:

```bash
# Using pip
pip install -e ".[dev]"

# Using uv (recommended)
uv pip install -e ".[dev]"
```

Installing with visualisation tools:

```bash
# Using pip
pip install -e ".[plot]"

# Using uv (recommended)
uv pip install -e ".[plot]"
```

For all dependencies including development, plotting, numpy, and Jupyter support:

```bash
# Using pip
pip install -e ".[all]"

# Using uv (recommended)
uv pip install -e ".[all]"
```

### Development Setup

For detailed development setup instructions, see [BUILD.md](BUILD.md).

Quick setup:

```bash
# Clone repository
git clone https://github.com/yourusername/evlib.git
cd evlib

# Create virtual environment
uv venv --python <python-version> # 3.12 recommended
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install pip

# Install for development using uv (recommended)
uv pip install -e ".[dev]"

# Or using pip
pip install -e ".[dev]"

# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run tests
pytest
```

## 🗺️ Roadmap and Current Features

- Core event data structures and manipulation
- Event data loading and saving
- Event augmentation
  - Random event addition
  - Correlated event addition
  - Event removal
- Event transformations
  - Flipping events along x and y axes
  - Clipping events to bounds
  - Rotating events
- Event representations
  - Voxel grid representation
- Event visualisation and display
- Event-to-video reconstruction

`evlib` aims to become a comprehensive toolkit for event camera data processing,
combining high-performance Rust implementations with Python bindings for ease of
use. A tracking issue can be found [here](https://github.com/tallamjr/evlib/issues/1)

| Algorithm/Feature          | Description                                 | Status         |
| -------------------------- | ------------------------------------------- | -------------- |
| Core Event Data Structures | Basic event representation and manipulation | ✅ Implemented |
| Event Augmentation         | Random/correlated event addition/removal    | ✅ Implemented |
| Event Transformations      | Flipping, rotation, clipping                | ✅ Implemented |
| Voxel Grid                 | Event-to-voxel grid conversion              | ✅ Implemented |
| Visualisation              | Event-to-image conversion tools             | ✅ Implemented |
| E2VID (Basic)              | Simple event-to-video reconstruction        | ✅ Implemented |
| OpenEB Format Support      | Compatibility with OpenEB data formats      | ⏳ Planned     |
| OpenEB HAL Integration     | Hardware abstraction for cameras            | ⏳ Planned     |
| OpenEB Streaming           | Real-time event stream processing           | ⏳ Planned     |
| E2VID (Advanced)           | Neural network reconstruction               | ⏳ Planned     |
| Vid2E Simulation           | Video-to-event conversion                   | ⏳ Planned     |
| ESIM Framework             | Event camera simulation                     | ⏳ Planned     |
| HyperE2VID                 | Advanced reconstruction with hypernetworks  | ⏳ Planned     |
| RVT Object Detection       | Event-based object detection                | ⏳ Planned     |
| Optical Flow               | Event-based optical flow estimation         | ⏳ Planned     |
| Depth Estimation           | Event-based depth estimation                | ⏳ Planned     |

## 🚀 Performance

Evlib is significantly faster than pure Python implementations, thanks to its
Rust backend. The benchmark compares the Rust-backed evlib implementation
against equivalent pure Python implementations of the same functions, in both
single-core and multi-core scenarios.

### Single-core Performance

| Operation         | Python Time (s) | Rust Time (s) | Speedup |
| ----------------- | --------------- | ------------- | ------- |
| events_to_block   | 0.040431        | 0.000860      | 47.03x  |
| add_random_events | 0.018615        | 0.003421      | 5.44x   |
| flip_events_x     | 0.000023        | 0.000283      | 0.08x   |

_Benchmark performed with 100,000 events on a single core_

### Multi-core vs Single-core Performance

| Operation         | Python (1 core) | Python (10 cores) | Rust (1 core) | Rust vs Py (1 core) | Rust vs Py (10 cores) |
| ----------------- | --------------- | ----------------- | ------------- | ------------------- | --------------------- |
| events_to_block   | 0.040431 s      | 0.315156 s        | 0.000860 s    | 47.03x              | 366.58x               |
| add_random_events | 0.018615 s      | 0.360760 s        | 0.003421 s    | 5.44x               | 105.44x               |
| flip_events_x     | 0.000023 s      | 0.303467 s        | 0.000283 s    | 0.08x               | 1072.67x              |

_Benchmark performed with 100,000 events. Note that for these specific
operations and data sizes, the multi-core Python implementation is slower due to
process creation overhead._

### Why Rust is Faster

The significant performance gains come from several factors:

1. **Compiled vs Interpreted**: Rust is compiled to native machine code, while Python is interpreted
2. **Memory Management**: Rust's ownership model allows for efficient memory use without garbage collection
3. **Low-level Optimisations**: Rust can take advantage of SIMD (Single Instruction Multiple Data) vectorisation
4. **Static Typing**: Rust's type system enables compiler optimisations that aren't possible with Python's dynamic typing
5. **Zero-cost Abstractions**: Rust provides high-level abstractions without runtime overhead
6. **Efficient Concurrency**: Rust's thread safety guarantees and lack of GIL allow for better parallelisation

Run `python examples/benchmark.py` to benchmark on your own system.

## ⮑ Module Structure

The library is organized into the following modules:

- `evlib.core`: Core event data structures and functions
- `evlib.augmentation`: Event augmentation utilities
- `evlib.formats`: Data loading and saving
- `evlib.representations`: Event representation algorithms (e.g., voxel grid)
- `evlib.visualization`: Visualisation tools
- `evlib.processing`: Advanced event processing (including event-to-video reconstruction)

### Basic Usage

```python
import numpy as np
import evlib

# Create example event data
xs = np.array([10, 20, 30, 40], dtype=np.int64)
ys = np.array([50, 60, 70, 80], dtype=np.int64)
ts = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
ps = np.array([1, -1, 1, -1], dtype=np.int64)

# Convert to block representation
block = evlib.core.events_to_block_py(xs, ys, ts, ps)
print(f"Block shape: {block.shape}")  # (4, 4)
```

### Loading Event Data

```python
import evlib

# Load events from file (automatically detects format)
xs, ys, ts, ps = evlib.formats.load_events_py("data/slider_depth/events.txt")

# Save events to HDF5 format
evlib.formats.save_events_to_hdf5_py(xs, ys, ts, ps, "output.h5")

# Save events to text format
evlib.formats.save_events_to_text_py(xs, ys, ts, ps, "output.txt")
```

### Event Augmentation

```python
import numpy as np
import evlib
import matplotlib.pyplot as plt

# Create sample event data
xs = np.array([50, 60, 70, 80, 90], dtype=np.int64)
ys = np.array([50, 60, 70, 80, 90], dtype=np.int64)
ts = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
ps = np.array([1, -1, 1, -1, 1], dtype=np.int64)

# Add random events
to_add = 20
new_xs, new_ys, new_ts, new_ps = evlib.augmentation.add_random_events_py(xs, ys, ts, ps, to_add)
print(f"Original events: {len(xs)}, After adding random events: {len(new_xs)}")

# Add correlated events (events near existing ones)
to_add = 15
xy_std = 2.0  # Standard deviation for x,y coordinates
ts_std = 0.005  # Standard deviation for timestamps

new_xs, new_ys, new_ts, new_ps = evlib.augmentation.add_correlated_events(
    xs, ys, ts, ps, to_add,
    xy_std=xy_std,
    ts_std=ts_std
)
```

### Event Transformations

```python
import numpy as np
import evlib

# Create sample event data
xs = np.array([10, 20, 30, 40, 50, 60, 70], dtype=np.int64)
ys = np.array([15, 25, 35, 45, 55, 65, 75], dtype=np.int64)
ts = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float64)
ps = np.array([1, -1, 1, -1, 1, -1, 1], dtype=np.int64)

# Set the sensor resolution
sensor_resolution = (100, 100)  # (height, width)

# Flip events along x-axis
flipped_x_xs, flipped_x_ys, flipped_x_ts, flipped_x_ps = evlib.augmentation.flip_events_x(
    xs, ys, ts, ps, sensor_resolution
)

# Flip events along y-axis
flipped_y_xs, flipped_y_ys, flipped_y_ts, flipped_y_ps = evlib.augmentation.flip_events_y(
    xs, ys, ts, ps, sensor_resolution
)

# Rotate events by 45 degrees
theta_radians = np.pi / 4  # 45 degrees
center_of_rotation = (50, 50)  # Center of rotation
rotated_xs, rotated_ys, theta_returned, center_returned = evlib.augmentation.rotate_events(
    xs, ys, ts, ps,
    sensor_resolution=sensor_resolution,
    theta_radians=theta_radians,
    center_of_rotation=center_of_rotation
)

# Clip events to bounds
bounds = [30, 70, 30, 70]  # [min_y, max_y, min_x, max_x]
clipped_xs, clipped_ys, clipped_ts, clipped_ps = evlib.augmentation.clip_events_to_bounds(
    xs, ys, ts, ps, bounds
)
```

### Event Representations (Voxel Grid)

```python
import numpy as np
import evlib

# Create event data (1D arrays)
xs = np.array([10, 20, 30, 40, 50], dtype=np.int64)
ys = np.array([15, 25, 35, 45, 55], dtype=np.int64)
ts = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
ps = np.array([1, -1, 1, -1, 1], dtype=np.int64)

# Convert events to voxel grid
num_bins = 5
resolution = (100, 100)  # (width, height)
method = "count"  # Options: "count", "polarity", "time"

# Pass parameters in correct order
voxel_grid = evlib.representations.events_to_voxel_grid_py(
    xs, ys, ts, ps, num_bins, resolution, method
)

print(f"Voxel grid shape: {voxel_grid.shape}")  # (5, 100, 100)
```

### Event Visualisation

```python
import numpy as np
import matplotlib.pyplot as plt
import evlib
import os

# Create directory for saved figures
os.makedirs("examples/figures", exist_ok=True)

# Create event data
xs = np.array([10, 20, 30, 40, 50], dtype=np.int64)
ys = np.array([15, 25, 35, 45, 55], dtype=np.int64)
ts = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
ps = np.array([1, -1, 1, -1, 1], dtype=np.int64)

# Draw events to image
resolution = (100, 100)  # (width, height)
color_mode = "red-blue"  # Options: "red-blue", "grayscale"

# Pass parameters in correct order
event_image = evlib.visualization.draw_events_to_image_py(
    xs, ys, ts, ps, resolution, color_mode
)

plt.figure(figsize=(10, 8))
plt.imshow(event_image)
plt.title("Event Visualisation")
plt.axis('off')

# Save figure (optional)
plt.savefig("examples/figures/event_visualization.png", bbox_inches="tight")
plt.show()
```

### Event-to-Video Reconstruction

```python
import numpy as np
import matplotlib.pyplot as plt
import evlib
import os

# Create directory for saved figures
os.makedirs("examples/figures", exist_ok=True)

# Load events
xs, ys, ts, ps = evlib.formats.load_events_py("data/slider_depth/events.txt")

# Use subset of events for faster processing (optional)
max_events = 10000
xs = xs[:max_events]
ys = ys[:max_events]
ts = ts[:max_events]
ps = ps[:max_events]

# Determine sensor resolution from events
height = int(max(ys)) + 1
width = int(max(xs)) + 1

# Reconstruct a single frame from events
num_bins = 5  # Number of time bins for voxel grid
reconstructed_frame = evlib.processing.events_to_video_py(
    xs, ys, ts, ps,
    height=height,
    width=width,
    num_bins=num_bins
)

# Display the reconstructed frame
plt.figure(figsize=(10, 8))
plt.imshow(reconstructed_frame, cmap='gray')
plt.title("Reconstructed Frame from Events")
plt.axis('off')

# Save figure (optional)
plt.savefig("examples/figures/reconstructed_frame.png", bbox_inches="tight")
plt.show()

# For multiple frames (reconstructing a sequence)
# Define time windows and reconstruct frames for each
reconstructed_frames = []
t_min, t_max = ts.min(), ts.max()
num_frames = 5
time_step = (t_max - t_min) / num_frames

for i in range(num_frames):
    t_end = t_min + time_step * (i + 1)
    mask = ts <= t_end
    frame_xs = xs[mask]
    frame_ys = ys[mask]
    frame_ts = ts[mask]
    frame_ps = ps[mask]

    frame = evlib.processing.events_to_video_py(
        frame_xs, frame_ys, frame_ts, frame_ps,
        height=height,
        width=width,
        num_bins=num_bins
    )

    reconstructed_frames.append(frame)

    # Save each frame (optional)
    plt.figure(figsize=(10, 8))
    plt.imshow(frame, cmap="gray")
    plt.title(f"Reconstructed Frame {i+1}")
    plt.axis("off")
    plt.savefig(f"examples/figures/reconstructed_frame_{i+1}.png", bbox_inches="tight")
    plt.close()
```

## ⚖️ License

MIT
