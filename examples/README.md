# evlib Examples

This directory contains examples demonstrating how to use the `evlib` library for event camera data processing.

## Module Structure

The module structure has been updated:
- `evlib.core` - Core functionality for event handling (formerly `evlib.events`)
- `evlib.augmentation` - Event augmentation functions
- `evlib.formats` - Data loading and saving
- `evlib.representations` - Event representations (voxel grid, etc.)
- `evlib.visualization` - Visualization utilities
- `evlib.processing` - Neural network models and reconstruction algorithms
- `evlib.simulation` - Event simulation from video data

## Examples

### GStreamer Integration (NEW)

#### Webcam Capture Demo
```bash
python gstreamer_webcam_demo.py
```
Demonstrates real-time webcam capture with GStreamer:
- Live video capture from default webcam
- Event simulation from captured frames
- Real-time processing pipeline
- Event data export and analysis

#### Video File Processing Demo
```bash
python gstreamer_video_file_demo.py [video_file.mp4]
```
Demonstrates video file processing with GStreamer:
- Support for multiple video formats (MP4, AVI, MOV, etc.)
- ESIM-style event simulation with configurable parameters
- Comprehensive event analysis and statistics
- Multi-format event data export

#### Complete Event Simulation Pipeline
```bash
# Webcam capture
python gstreamer_event_simulation_complete.py --source webcam

# Video file processing
python gstreamer_event_simulation_complete.py --source video --file path/to/video.mp4

# With reconstruction
python gstreamer_event_simulation_complete.py --source webcam --reconstruct
```
Demonstrates the complete pipeline:
- Real-time video capture OR video file processing
- Advanced ESIM algorithm with noise modelling
- Event-based reconstruction integration
- Performance benchmarking and analysis
- Rich visualizations and statistics

#### Interactive Jupyter Notebook
```bash
jupyter notebook gstreamer_integration_demo.ipynb
```
Interactive notebook demonstrating:
- Step-by-step GStreamer integration
- Event simulation with parameter tuning
- Comprehensive visualizations
- Real-time analysis and statistics

### Basic Usage
```
python basic_usage.py
```
Demonstrates basic functionality:
- Creating event data arrays
- Converting to block representation
- Using basic utility functions

### Event Augmentation
```
python event_augmentation.py
```
Demonstrates event augmentation techniques:
- Adding random events
- Adding correlated events (noise near existing events)
- Removing events
- Visualizing the results

### Event Transformations
```
python event_transformations.py
```
Demonstrates spatial transformations:
- Flipping events along x and y axes
- Rotating events by an angle
- Clipping events to bounds
- Visualizing transformations

### Synthetic DVS Data
```
python synthetic_dvs_data.py
```
Demonstrates creating and visualizing synthetic event data:
- Generating events from moving patterns
- Applying transformations to event streams
- Visualizing events in 3D space (x, y, t)
- Creating animations from event data

### Benchmark
```
python benchmark.py
```
Benchmarks the Rust-backed implementation against pure Python:
- Measures performance for common operations
- Compares execution times
- Verifies correctness of results

## Requirements

### Basic Examples
- NumPy
- Matplotlib
- evlib

### GStreamer Examples (Additional)
- GStreamer system libraries
- evlib built with gstreamer feature
- Optional: Jupyter for notebook examples

## Installation

### Standard Installation
```bash
pip install evlib
```

### Development Installation
```bash
pip install -e ".[dev]"
```

### GStreamer Integration
For GStreamer examples, build evlib with GStreamer support:
```bash
# Install GStreamer system libraries first
# macOS: brew install gstreamer gst-plugins-base gst-plugins-good
# Ubuntu: sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev

# Build evlib with GStreamer support
maturin develop --features gstreamer
```

## Usage Notes

1. Run any basic example directly:
```bash
python examples/basic_usage.py
```

2. For GStreamer examples:
```bash
# Test webcam capture
python examples/gstreamer_webcam_demo.py

# Process a video file
python examples/gstreamer_video_file_demo.py path/to/video.mp4

# Interactive notebook
jupyter notebook examples/gstreamer_integration_demo.ipynb
```

3. For performance-critical applications, build in release mode:
```bash
maturin develop --release --features gstreamer
```

## Troubleshooting

### GStreamer Issues
- **Import Error**: Ensure GStreamer system libraries are installed
- **Build Failure**: Check that pkg-config can find GStreamer libraries
- **Runtime Error**: Verify webcam permissions and device availability

### Performance Tips
- Use release builds for benchmarking: `maturin develop --release`
- Enable hardware acceleration when available
- Adjust event simulation parameters based on your video content
