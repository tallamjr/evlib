# evlib Examples

This directory contains comprehensive examples demonstrating how to use the `evlib` library for event camera data processing, with a focus on high-performance Polars DataFrame integration.

## Module Structure

The module structure has been updated:
- `evlib.core` - Core functionality for event handling (formerly `evlib.events`)
- `evlib.augmentation` - Event augmentation functions
- `evlib.formats` - Data loading and saving with Polars DataFrame support
- `evlib.representations` - Event representations (voxel grid, etc.)
- `evlib.visualization` - Visualization utilities
- `evlib.processing` - Neural network models and reconstruction algorithms
- `evlib.simulation` - Event simulation from video data
- `evlib.polars_utils` - High-performance Polars DataFrame utility functions

## Examples

### FEATURE: Polars DataFrame Integration (HIGH-PERFORMANCE)

#### Complete Polars API Overview
```bash
jupyter notebook polars_integration_example.ipynb
```
Comprehensive demonstration of Polars DataFrame functionality:
- Multiple API levels (low-level to high-level)
- Performance comparisons (up to 97x speedup)
- Advanced query operations and lazy evaluation
- Round-trip conversions and data integrity
- Memory efficiency and optimization techniques

#### Polars Utility Functions Demo
```bash
jupyter notebook polars_utility_functions_demo.ipynb
```
Advanced utility functions for event data analysis:
- Spatial, temporal, and polarity filtering
- Statistical analysis and aggregation operations
- Temporal windowing and event rate analysis
- Activity hotspot detection and clustering
- Region-of-interest analysis

#### Python Script Demo
```bash
python polars_integration_demo.py
```
Standalone Python script demonstrating:
- All Polars DataFrame loading methods
- Performance benchmarking against NumPy
- Advanced query patterns and optimizations
- Real-world analysis workflows

#### Real-world Analysis Examples
```bash
python polars_realworld_analysis.py
```
Practical examples for common use cases:
- Activity detection and tracking
- Temporal pattern analysis
- Spatial activity mapping
- Multi-dataset comparison
- Real-time processing simulation
- Event stream quality assessment

#### Streaming Large Datasets
```bash
jupyter notebook streaming_large_datasets_demo.ipynb
```
Memory-efficient processing of large event datasets:
- Chunked processing techniques
- Lazy evaluation strategies
- Memory management best practices
- Performance optimization for large files

#### Rust Polars Demo
```bash
cargo run --example polars_demo --features polars
```
Rust-level Polars integration demonstration:
- Direct Rust DataFrame operations
- Performance benchmarking
- Memory-efficient conversions
- Advanced DataFrame manipulations

### DATA: Data Analysis and Visualization

#### Event Data Exploration Notebooks
```bash
# HDF5 data analysis
jupyter notebook eda_etram_h5.ipynb
jupyter notebook eda_gen4_h5.ipynb
jupyter notebook eda_original_h5.ipynb

# Event visualization
jupyter notebook events_viz_0.ipynb
jupyter notebook events_viz_1.ipynb
```
Comprehensive data exploration notebooks:
- Dataset characterization and statistics
- Event pattern visualization
- Temporal and spatial analysis
- Data quality assessment

#### Stacked Histogram Demo
```bash
python stacked_histogram_demo.py
```
Demonstrates efficient event representation creation:
- Stacked histogram generation
- Performance optimization techniques
- Integration with Polars DataFrames

### FEATURE: GStreamer Integration (NEW)

#### Webcam Capture Demo
```bash
python gstreamer_webcam_demo.py
```
Demonstrates real-time webcam capture with GStreamer:
- Live video capture from default webcam
- Event simulation from captured frames
- Real-time processing pipeline
- Event data export and analysis

#### Synthetic Event Generation Demo
```bash
python synthetic_event_generation_demo.py [pattern_name]
```
Demonstrates synthetic event generation for testing and development:
- Generates synthetic video patterns (moving sine waves, etc.)
- ESIM-style event simulation with configurable parameters
- Comprehensive event analysis and statistics
- Multi-format event data export
- NOTE: Does not use real video files; use external tools to extract frames

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

### DOCUMENTATION: Format Support and Data Loading

#### Format Reader Examples
```bash
python reader_examples.py
python reader_showcase.py
```
Demonstrates comprehensive format support:
- Automatic format detection
- Loading from H5, AEDAT, EVT2/3, AER, and text formats
- Performance comparison across formats
- Error handling and validation

#### eTram Dataset Examples
```bash
python etram_data_loading.py
python simple_etram_usage.py
```
Specific examples for eTram automotive dataset:
- Large-scale dataset processing
- Sparse event distribution handling
- Memory-efficient loading strategies

#### EVT2.1 Format Example
```bash
python evt21_example.py
```
Demonstrates modern EVT2.1 format support:
- Binary format parsing
- Efficient event extraction
- Format-specific optimizations

### TREND: Performance and Benchmarking

#### Comprehensive Benchmarks
```bash
python benchmark.py
```
Benchmarks the Rust-backed implementation:
- NumPy vs Polars performance comparison
- Memory usage analysis
- Scalability testing
- Operation-specific benchmarks

### TOOL: Basic Usage and Utilities

#### Basic Functionality
```bash
# Basic examples are integrated into Polars demos
# See polars_integration_demo.py for modern approaches
```
Note: Basic usage examples have been enhanced and integrated into the Polars demonstration scripts for better performance and modern best practices.

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

### Core Examples
- NumPy
- Matplotlib
- evlib

### Polars Examples (Recommended)
- NumPy
- Polars (≥0.45.0)
- Jupyter (for notebooks)
- evlib[polars]

### GStreamer Examples (Additional)
- GStreamer system libraries
- evlib built with gstreamer feature
- Optional: Jupyter for notebook examples

## Installation

### Standard Installation
```bash
pip install evlib
```

### With Polars Support (Recommended)
```bash
pip install evlib[polars]
```

### Development Installation
```bash
pip install -e ".[dev,polars]"
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

### Running Examples

#### 1. Polars DataFrame Examples (Recommended for performance)
```bash
# Interactive notebooks (best for learning)
jupyter notebook examples/polars_integration_example.ipynb
jupyter notebook examples/polars_utility_functions_demo.ipynb

# Python scripts (for automation)
python examples/polars_integration_demo.py
python examples/polars_realworld_analysis.py

# Rust examples (for developers)
cargo run --example polars_demo --features polars
```

#### 2. Data Analysis Examples
```bash
# Explore different datasets
jupyter notebook examples/eda_etram_h5.ipynb
jupyter notebook examples/eda_original_h5.ipynb

# Visualization and analysis
python examples/stacked_histogram_demo.py
```

#### 3. Format and Loading Examples
```bash
python examples/reader_showcase.py
python examples/etram_data_loading.py
```

#### 4. For GStreamer examples:
```bash
# Test webcam capture
python examples/gstreamer_webcam_demo.py

# Generate synthetic events
python examples/synthetic_event_generation_demo.py my_pattern

# Interactive notebook
jupyter notebook examples/gstreamer_integration_demo.ipynb
```

3. For performance-critical applications, build in release mode:
```bash
maturin develop --release --features gstreamer
```

## Performance Expectations

### Polars DataFrame Performance
Using Polars DataFrames provides significant performance improvements:

| Operation | NumPy | Polars | Speedup |
|-----------|-------|--------|---------|
| Loading 1M events | 0.45s | 0.28s | 1.6x |
| Filtering by polarity | 0.089s | 0.0009s | 97x |
| Spatial filtering | 0.234s | 0.011s | 21x |
| Group by polarity | 0.156s | 0.019s | 8x |
| Temporal binning | 0.891s | 0.045s | 20x |

*Benchmarks on Apple M1 with 16GB RAM*

### Memory Efficiency
- Polars DataFrames: ~15% more memory than NumPy arrays
- Significantly better performance per memory unit
- Optimized for large dataset processing

## Example Categories

### FEATURE: High-Performance (Polars)
- `polars_integration_example.ipynb` - Complete API overview
- `polars_utility_functions_demo.ipynb` - Advanced utilities
- `polars_integration_demo.py` - Performance demonstrations
- `polars_realworld_analysis.py` - Practical applications
- `streaming_large_datasets_demo.ipynb` - Memory-efficient processing

### DATA: Data Analysis
- `eda_*_h5.ipynb` - Dataset exploration
- `events_viz_*.ipynb` - Visualization techniques
- `stacked_histogram_demo.py` - Representation creation

### DOCUMENTATION: Format Support
- `reader_showcase.py` - Format compatibility
- `etram_data_loading.py` - Large dataset handling
- `evt21_example.py` - Modern format support

### FEATURE: Simulation and Processing
- `gstreamer_*` - Real-time processing
- `synthetic_*` - Event simulation

## Troubleshooting

### Polars Issues
- **Import Error**: Install with `pip install polars` or `pip install evlib[polars]`
- **Performance Issues**: Use lazy evaluation and optimize query order
- **Memory Errors**: Use chunked processing for large datasets
- **API Errors**: Import from `evlib.polars_utils` for utility functions

### GStreamer Issues
- **Import Error**: Ensure GStreamer system libraries are installed
- **Build Failure**: Check that pkg-config can find GStreamer libraries
- **Runtime Error**: Verify webcam permissions and device availability

### Performance Tips
- **Use Polars DataFrames** for datasets >10K events
- **Use lazy evaluation** for complex multi-step operations
- **Apply filters early** to reduce data size
- **Use release builds** for benchmarking: `maturin develop --release`
- **Leverage built-in utility functions** instead of custom implementations

## Getting Started

### For New Users
1. Start with `polars_integration_example.ipynb` to understand the API
2. Try `polars_integration_demo.py` to see performance benefits
3. Explore real-world examples in `polars_realworld_analysis.py`
4. Use utility functions from `polars_utility_functions_demo.ipynb`

### For Existing Users
1. Review the migration guide in `docs/user-guide/polars-migration-guide.md`
2. Compare performance with existing workflows using benchmark examples
3. Gradually migrate critical operations to Polars DataFrames
4. Leverage lazy evaluation for complex pipelines

The examples in this directory demonstrate the full capabilities of evlib, with particular emphasis on the high-performance Polars DataFrame integration that provides up to 97x speedup for common operations.
