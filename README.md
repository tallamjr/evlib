<table align="center">
  <tr>
    <td>
      <img src="./evlogo.png" width="70" alt="evlib logo" />
    </td>
    <td>
      <h1 style="margin: 0;">
        <code>evlib</code>: Event Camera Utilities in Rust
      </h1>
    </td>
  </tr>
</table>

<div style="text-align: center;" align="center">

[![PyPI Version](https://img.shields.io/pypi/v/evlib.svg)](https://pypi.org/project/evlib/)
[![Python Versions](https://img.shields.io/pypi/pyversions/evlib.svg)](https://pypi.org/project/evlib/)
[![Python](https://github.com/tallamjr/evlib/actions/workflows/pytest.yml/badge.svg)](https://github.com/tallamjr/evlib/actions/workflows/pytest.yml)
[![Rust](https://github.com/tallamjr/evlib/actions/workflows/rust.yml/badge.svg)](https://github.com/tallamjr/evlib/actions/workflows/rust.yml)
![Crates.io Version](https://img.shields.io/crates/v/evlib)
[![Build](https://github.com/tallamjr/evlib/actions/workflows/build.yml/badge.svg)](https://github.com/tallamjr/evlib/actions/workflows/build.yml)
[![Release](https://github.com/tallamjr/evlib/actions/workflows/release.yml/badge.svg)](https://github.com/tallamjr/evlib/actions/workflows/release.yml)
[![Codecov](https://codecov.io/gh/tallamjr/evlib/branch/master/graph/badge.svg)](https://codecov.io/gh/tallamjr/evlib)
[![License](https://img.shields.io/github/license/tallamjr/evlib)](https://github.com/tallamjr/evlib/blob/master/LICENSE.md)

</div>

A high-performance implementation of event camera utilities using Rust with Python bindings via PyO3.

This library is inspired by numerous event camera libraries such as
[`event_utils`](https://github.com/TimoStoff/event_utils) Python library but
reimplemented in Rust for significantly better performance.

> [!Warning]
>
> **This is an experimental project with frequent breaking changes.
> Many advanced features are still under development - see current status below.**

> [!Note]
>
> **January 2025 Update**: This library has undergone a comprehensive audit to ensure
> all claims match actual implementation. See `REPORT.md` for full details.
> We are committed to honest documentation going forward.

<!-- mtoc-start -->

* [⬇ Installation](#-installation)
  * [Development Setup](#development-setup)
* [🧪 Testing](#-testing)
  * [Rust Tests](#rust-tests)
  * [Python Tests](#python-tests)
* [📊 Current Status](#-current-status)
* [⮑ Module Structure](#-module-structure)
  * [Basic Usage](#basic-usage)
  * [Loading Event Data](#loading-event-data)
  * [Event Augmentation](#event-augmentation)
  * [Event Transformations](#event-transformations)
  * [Event Representations (Voxel Grid)](#event-representations-voxel-grid)
  * [Event Visualisation](#event-visualisation)
  * [Event-to-Video Reconstruction](#event-to-video-reconstruction)
  * [Event Simulation](#event-simulation)
* [🗺️ Roadmap and Current Features](#-roadmap-and-current-features)
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

## 🧪 Testing

evlib includes a comprehensive test suite covering Rust core functionality and Python bindings.

### Rust Tests

Run native Rust tests for core functionality:

```bash
# Run all Rust tests
cargo test

# Run specific test module
cargo test smooth_voxel

# Run with output
cargo test -- --nocapture

# Run specific test file
cargo test --test test_smooth_voxel
```

### Python Tests

Run Python tests for bindings and integration:

```bash
# Run all Python tests
pytest

# Run specific test file
pytest tests/test_evlib.py

# Run with verbose output
pytest -v

# Test notebooks (if available)
pytest --nbmake examples/

# Run specific functionality tests
pytest tests/test_representations.py::test_smooth_voxel
```

## 📊 Current Status

> **Status**: ⚠️ **Core functionality working, advanced features in development**

### ✅ **Working Features** (Verified January 2025)

**Core Functionality**:
- ✅ Event data structures and manipulation
- ✅ Event data loading and saving (HDF5, text formats)
- ✅ Event augmentation (random addition, correlated events, removal)
- ✅ Event transformations (flipping, rotation, clipping)
- ✅ Voxel grid representations (standard and smooth with interpolation)
- ✅ Event visualisation and display
- ✅ Event simulation (ESIM) - video-to-events conversion

**Neural Network Models**:
- ✅ **E2VID UNet** - basic event-to-video reconstruction
- ✅ **FireNet** - lightweight reconstruction variant

**Infrastructure**:
- ✅ Python bindings via PyO3
- ✅ Model downloading infrastructure
- ✅ Cross-platform support (macOS, Linux, Windows)

### 🚧 **In Development**

**Model Loading**:
- ⚠️ PyTorch weight loading (downloads models but integration needs work)
- ⚠️ ONNX Runtime integration (infrastructure exists, needs refinement)

**Advanced Models**:
- 🔲 E2VID+ (temporal features)
- 🔲 FireNet+ (enhanced lightweight variant)
- 🔲 SPADE-E2VID (spatially-adaptive normalization)
- 🔲 SSL-E2VID (self-supervised learning)
- 🔲 ET-Net (transformer-based)
- 🔲 HyperE2VID (dynamic convolutions)

### 📋 **Planned Features**

- OpenEB format support and HAL integration
- Real-time streaming capabilities
- Hardware acceleration (CUDA/Metal optimization)
- Production deployment tools
- ROS2 integration
- Performance benchmarking suite

## ⮑ Module Structure

The library is organized into the following modules:

- `evlib.core`: Core event data structures and functions
- `evlib.augmentation`: Event augmentation utilities
- `evlib.formats`: Data loading and saving
- `evlib.representations`: Event representation algorithms (e.g., voxel grid)
- `evlib.visualization`: Visualisation tools
- `evlib.processing`: Event processing and reconstruction
- `evlib.simulation`: Event simulation and video-to-events conversion

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
block = evlib.core.events_to_block(xs, ys, ts, ps)
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

# Create sample event data
xs = np.array([50, 60, 70, 80, 90], dtype=np.int64)
ys = np.array([50, 60, 70, 80, 90], dtype=np.int64)
ts = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
ps = np.array([1, -1, 1, -1, 1], dtype=np.int64)

# Add random events
to_add = 20
new_xs, new_ys, new_ts, new_ps = evlib.augmentation.add_random_events(xs, ys, ts, ps, to_add)
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

voxel_grid = evlib.representations.events_to_voxel_grid(
    xs, ys, ts, ps, num_bins, resolution, method
)

print(f"Voxel grid shape: {voxel_grid.shape}")  # (5, 100, 100)

# Convert events to smooth voxel grid with interpolation
interpolation = "trilinear"  # Options: "trilinear", "bilinear", "temporal"
smooth_voxel_grid = evlib.representations.events_to_smooth_voxel_grid(
    xs, ys, ts, ps, num_bins, resolution, interpolation
)

print(f"Smooth voxel grid shape: {smooth_voxel_grid.shape}")  # (5, 100, 100)
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
reconstructed_frame = evlib.processing.events_to_video_advanced(
    xs, ys, ts, ps,
    height=height,
    width=width,
    num_bins=num_bins,
    model_type="unet"  # Options: "unet", "firenet"
)

# Display the reconstructed frame
plt.figure(figsize=(10, 8))
plt.imshow(reconstructed_frame, cmap='gray')
plt.title("Reconstructed Frame from Events")
plt.axis('off')

# Save figure (optional)
plt.savefig("examples/figures/reconstructed_frame.png", bbox_inches="tight")
plt.show()
```

### Event Simulation

```python
import numpy as np
import evlib

# Create two intensity frames for ESIM simulation
height, width = 100, 100
intensity_old = np.ones((height, width), dtype=np.float32) * 0.3
intensity_new = np.ones((height, width), dtype=np.float32) * 0.7

# Generate events using ESIM simulation
xs, ys, ts, ps = evlib.simulation.esim_simulate_py(
    intensity_old,
    intensity_new,
    threshold=0.2,
    refractory_period_us=100.0
)

print(f"Generated {len(xs)} events from intensity change")

# Create a video-to-events converter for more advanced simulation
config = evlib.simulation.PySimulationConfig(
    resolution=(width, height),
    contrast_threshold_pos=0.2,
    contrast_threshold_neg=0.2,
    enable_noise=True
)

converter = evlib.simulation.PyVideoToEventsConverter(config)

# Convert a single frame to events
test_frame = np.random.rand(height, width).astype(np.float32)
xs2, ys2, ts2, ps2 = converter.convert_frame(test_frame, timestamp_us=0.0)

print(f"Converter generated {len(xs2)} events from random frame")
```

## 🗺️ Roadmap and Current Features

`evlib` aims to become a comprehensive toolkit for event camera data processing,
combining high-performance Rust implementations with Python bindings for ease of
use. A tracking issue can be found [here](https://github.com/tallamjr/evlib/issues/1)

### Status Legend
- ✅ **Implemented**: Feature is working and tested
- ⚠️ **Partial**: Basic functionality works, but needs improvement
- 🔲 **Planned**: Feature is planned but not yet implemented

| Algorithm/Feature          | Description                                 | Status         |
| -------------------------- | ------------------------------------------- | -------------- |
| Core Event Data Structures | Basic event representation and manipulation | ✅ Implemented |
| Event Augmentation         | Random/correlated event addition/removal    | ✅ Implemented |
| Event Transformations      | Flipping, rotation, clipping                | ✅ Implemented |
| Voxel Grid                 | Event-to-voxel grid conversion              | ✅ Implemented |
| Smooth Voxel Grid          | Interpolated voxel grid representation      | ✅ Implemented |
| Visualisation              | Event-to-image conversion tools             | ✅ Implemented |
| Event Simulation (ESIM)    | Video-to-events conversion                  | ✅ Implemented |
| E2VID UNet                 | Basic event-to-video reconstruction         | ✅ Implemented |
| FireNet                    | Lightweight reconstruction variant           | ✅ Implemented |
| PyTorch Weight Loading     | Load pre-trained .pth models               | ⚠️ Partial     |
| ONNX Runtime Integration   | ONNX model inference                        | ⚠️ Partial     |
| E2VID+                     | Enhanced E2VID with temporal features       | 🔲 Planned     |
| FireNet+                   | Enhanced FireNet variant                    | 🔲 Planned     |
| SPADE-E2VID               | Spatially-adaptive normalization            | 🔲 Planned     |
| SSL-E2VID                 | Self-supervised learning approach           | 🔲 Planned     |
| ET-Net                    | Transformer-based reconstruction            | 🔲 Planned     |
| HyperE2VID                | Dynamic convolutions with hypernetworks     | 🔲 Planned     |
| OpenEB Format Support      | Compatibility with OpenEB data formats      | 🔲 Planned     |
| OpenEB HAL Integration     | Hardware abstraction for cameras            | 🔲 Planned     |
| Real-time Streaming        | Real-time event stream processing           | 🔲 Planned     |
| Hardware Acceleration      | CUDA/Metal optimization                     | 🔲 Planned     |
| RVT Object Detection       | Event-based object detection                | 🔲 Planned     |
| Optical Flow               | Event-based optical flow estimation         | 🔲 Planned     |
| Depth Estimation           | Event-based depth estimation                | 🔲 Planned     |

For detailed development progress and upcoming features, see [TODO.md](TODO.md).

## ⚖️ License

MIT
