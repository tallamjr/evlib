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
reimplemented in Rust for improved performance in certain operations.

> [!Warning]
>
> **This is an experimental project with frequent breaking changes.
> Many advanced features are still under development - see current status below.**


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
  * [Real-time Webcam Event Streaming](#real-time-webcam-event-streaming)
  * [Ultra-Fast Terminal Visualization](#ultra-fast-terminal-visualization)
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

#### Prerequisites
- Rust (latest stable version)
- Python 3.10+ (3.12 recommended)
- uv (recommended) or pip with venv

#### Quick Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/evlib.git
cd evlib

# Create virtual environment
uv venv --python 3.12  # or your preferred version
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

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

#### Building the Rust Extension

**Method 1: Using Maturin (Recommended)**

```bash
# Development build (fast iteration)
maturin develop

# Release build for testing performance
maturin develop --release

# Build wheel for distribution
maturin build --release
```

**Method 2: Using Custom Build Script**

```bash
# Make script executable
chmod +x build.sh

# Development build
./build.sh --debug

# Release build
./build.sh
```

#### Advanced Build Options

**Cross-compilation with Zig:**

```bash
# Install Zig and maturin with Zig support
pip install "maturin[zig]"

# Build Linux wheels from macOS
maturin build --release --target x86_64-unknown-linux-gnu --manylinux 2014 --zig

# Build ARM wheels
maturin build --release --target aarch64-unknown-linux-gnu --manylinux 2014 --zig
```

**Platform-specific wheels:**

```bash
# Universal2 wheel on macOS
maturin build --release --universal2

# Manylinux wheels for Linux
maturin build --release --manylinux=2014
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

**Real-time Visualization**:
- ✅ **Terminal-based visualization** - ultra-fast Ratatui rendering (60-120+ FPS)
- ✅ **Rust-optimized OpenCV demo** - high-performance event visualization (30-60 FPS)
- ✅ **Real-time event streaming** - low-latency webcam processing

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

### Real-time Webcam Event Streaming

**Requirements**: GStreamer development libraries must be installed on your system.

```bash
# macOS (using Homebrew)
brew install gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad gst-plugins-ugly

# Ubuntu/Debian
sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
                     libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base \
                     gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
                     gstreamer1.0-plugins-ugly

# CentOS/RHEL/Fedora
sudo dnf install gstreamer1-devel gstreamer1-plugins-base-devel \
                 gstreamer1-plugins-good gstreamer1-plugins-bad-free \
                 gstreamer1-plugins-ugly-free
```

**Build with GStreamer support**:

```bash
# Rebuild evlib with GStreamer feature enabled
maturin develop --features gstreamer

# Or for release build
maturin develop --release --features gstreamer
```

**Interactive Webcam Demo**:

```bash
# Run the interactive webcam event streaming demo
python examples/webcam_event_demo.py --fps 30 --threshold 0.15 --device_id 0
```

**Interactive Controls**:
- **Arrow Keys**: Adjust contrast thresholds (↑/↓ for positive, ←/→ for negative)
- **+/-**: Increase/decrease frame rate
- **r**: Reset to original configuration
- **s**: Save current frame and events
- **q/Escape**: Quit the demo

**Python API Example**:

```python
import evlib
import numpy as np
import time

# Configure real-time streaming
config = evlib.simulation.PyRealtimeStreamConfig(
    width=640,
    height=480,
    fps=30.0,
    contrast_threshold_pos=0.15,
    contrast_threshold_neg=0.15,
    refractory_period_us=100.0,
    device_id=0,
    enable_noise=False
)

# Create and start stream
stream = evlib.simulation.PyRealtimeEventStream(config)
stream.start()

try:
    for _ in range(100):  # Process 100 frames
        # Get latest events
        events = stream.get_events()

        if len(events) > 0:
            print(f"Received {len(events)} events")

            # Access event data
            for event in events:
                x, y, timestamp, polarity = event
                # Process individual events...

        # Get performance statistics
        stats = stream.get_stats()
        if stats.frames_processed % 30 == 0:  # Every second at 30fps
            print(f"FPS: {stats.current_fps:.1f}, "
                  f"Events/sec: {stats.events_per_second:.0f}")

        time.sleep(0.033)  # ~30 FPS

finally:
    stream.stop()
```

**Configuration Options**:

```python
# Advanced configuration
config = evlib.simulation.PyRealtimeStreamConfig(
    width=1280,                    # Camera resolution width
    height=720,                    # Camera resolution height
    fps=60.0,                      # Target frame rate
    contrast_threshold_pos=0.2,     # Positive contrast threshold
    contrast_threshold_neg=0.2,     # Negative contrast threshold
    refractory_period_us=50.0,     # Refractory period in microseconds
    device_id=0,                   # Camera device ID
    enable_noise=True,             # Enable noise simulation
    buffer_size=10000,             # Event buffer size
    enable_smoothing=True          # Enable temporal smoothing
)
```

### Ultra-Fast Terminal Visualization

For maximum performance event visualization, `evlib` provides a pure Rust terminal-based visualizer using [Ratatui](https://ratatui.rs/). This eliminates all GUI overhead and provides the fastest possible visualization experience.

#### Features

- **⚡ Ultra-high performance**: 60-120+ FPS rendering directly in terminal
- **🚀 Zero GUI overhead**: No OpenCV, X11, or graphics libraries required
- **🔧 SSH-friendly**: Works over SSH and in headless environments
- **📊 Real-time statistics**: Live FPS, event counts, and performance metrics
- **🎮 Interactive controls**: Keyboard controls for all parameters
- **🎨 Color-coded events**: Red for positive, blue for negative events with decay

#### Installation & Build

**Prerequisites:**
```bash
# macOS
brew install hdf5 pkg-config cmake gstreamer

# Ubuntu/Debian
sudo apt-get install libhdf5-dev pkg-config cmake libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
```

**Build the terminal binary:**
```bash
# Build ultra-fast terminal visualizer
cargo build --release --features "terminal gstreamer" --bin evlib-terminal --no-default-features
```

#### Usage

**Basic usage:**
```bash
# Default settings (most stable)
./target/release/evlib-terminal

# High-performance settings
./target/release/evlib-terminal --fps 120 --threshold 0.3 --decay 30 --max-events 500

# Lower resolution for maximum FPS
./target/release/evlib-terminal --width 320 --height 240 --fps 120 --threshold 0.4 --decay 20

# Show all available options
./target/release/evlib-terminal --help
```

**Advanced configuration:**
```bash
# Custom camera and processing settings
./target/release/evlib-terminal \
    --device_id 0 \
    --fps 60 \
    --threshold 0.25 \
    --width 640 \
    --height 480 \
    --buffer_size 500 \
    --decay 50 \
    --max-events 1000 \
    --no-stats  # Disable statistics for even better performance
```

#### Terminal Controls

| Key | Action |
|-----|--------|
| `q`, `Esc` | Quit application |
| `p`, `Space` | Pause/Resume streaming |
| `r` | Reset all statistics |
| `s` | Toggle statistics display |
| `+`/`-` | Adjust event decay time |
| `h`, `F1` | Toggle help screen |

#### Performance Comparison

| Visualization Method | Expected FPS | Overhead | Use Case |
|---------------------|-------------|----------|----------|
| **Rust Terminal Binary** | **60-120+ FPS** | Minimal | Maximum performance |
| Python + Rust Visualization | 30-60 FPS | Moderate | Development/debugging |
| Original Python Demo | 7-15 FPS | High | Basic functionality |

#### Performance Tuning

**For maximum FPS:**
- **Increase threshold** (`--threshold 0.3-0.5`): Reduces event count
- **Lower resolution** (`--width 320 --height 240`): Faster processing
- **Reduce decay time** (`--decay 20-30`): Fewer events on screen
- **Limit events** (`--max-events 300-500`): Prevent overload
- **Disable statistics** (`--no-stats`): Eliminate display overhead

**Recommended settings by use case:**
```bash
# Real-time development (good balance)
./target/release/evlib-terminal --fps 60 --threshold 0.3 --width 640 --height 480

# Maximum performance (demos/showcases)
./target/release/evlib-terminal --fps 120 --threshold 0.4 --width 320 --height 240 --decay 20 --max-events 300

# High quality (research/analysis)
./target/release/evlib-terminal --fps 30 --threshold 0.2 --width 1280 --height 720 --decay 100
```

#### Alternative Visualization Options

**1. Optimized Python Demo with Rust Backend:**
```bash
# Build with Python and Rust visualization support
maturin develop --release --features "python gstreamer"

# Run optimized Python demo (30-60 FPS)
python examples/webcam_event_demo_opencv.py --fps 60 --threshold 0.25

# Toggle between Rust and Python visualization with 'u' key
# Use 'f' key for fast mode, 'm' for real-time mode
```

**2. Terminal Visualization from Python:**
```bash
# Build with terminal support from Python
maturin develop --release --features "python terminal gstreamer"

# Run terminal demo from Python
python examples/webcam_event_demo_terminal.py --fps 60 --threshold 0.25 --decay 50
```

#### Troubleshooting

**Build Issues:**
```bash
# Missing dependencies
brew install hdf5 pkg-config cmake gstreamer  # macOS
sudo apt-get install libhdf5-dev pkg-config cmake libgstreamer1.0-dev  # Ubuntu

# Clean build if encountering issues
cargo clean
cargo build --release --features "terminal gstreamer" --bin evlib-terminal --no-default-features
```

**Runtime Issues:**
- **"No camera found"**: Try different `--device_id` values (0, 1, 2...)
- **Low frame rate**: Increase `--threshold`, lower resolution, reduce `--max-events`
- **Permission errors**: Check camera permissions in system settings

**Performance Tips:**
- Terminal size affects rendering speed - smaller terminals = higher FPS
- Good lighting helps generate stable events
- USB 3.0 cameras typically perform better than USB 2.0
- Close other applications using the camera

The **Rust terminal binary provides the absolute fastest event visualization** by eliminating all unnecessary overhead and rendering directly to the terminal using highly optimized Rust code.

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
| Terminal Visualization     | Ultra-fast Ratatui terminal rendering       | ✅ Implemented |
| Rust-optimized Visualization | High-performance OpenCV event display    | ✅ Implemented |
| OpenEB Format Support      | Compatibility with OpenEB data formats      | 🔲 Planned     |
| OpenEB HAL Integration     | Hardware abstraction for cameras            | 🔲 Planned     |
| Real-time Streaming        | Real-time event stream processing           | ✅ Implemented |
| Hardware Acceleration      | CUDA/Metal optimization                     | 🔲 Planned     |
| RVT Object Detection       | Event-based object detection                | 🔲 Planned     |
| Optical Flow               | Event-based optical flow estimation         | 🔲 Planned     |
| Depth Estimation           | Event-based depth estimation                | 🔲 Planned     |

For detailed development progress and upcoming features, see [TODO.md](TODO.md).

## 📊 Performance Characteristics

The Rust implementation shows mixed performance compared to pure Python:

| Operation | Rust vs Python (single-core) | Notes |
|-----------|------------------------------|-------|
| `events_to_block` | **1.3x faster** | Modest improvement for array conversion |
| `add_random_events` | **5x slower** | Complex operations favor NumPy |
| `flip_events_x` | **50x slower** | Simple operations better in NumPy |

**Key Takeaways:**
- Rust excels at complex algorithms and memory-intensive operations
- Python/NumPy is superior for simple array operations
- Performance varies significantly by operation type
- Multi-threading overhead affects some benchmarks

Run `python examples/benchmark.py` to see performance on your system.

## ⚖️ License

MIT
