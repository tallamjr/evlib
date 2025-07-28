<table align="center">
  <tr>
    <td>
      <img src="./docs/evlogo.png" width="70" alt="evlib logo" />
    </td>
    <td>
      <h1 style="margin: 0;">
        <code>evlib</code>: Event Camera Data Processing Library
      </h1>
    </td>
  </tr>
</table>

<div style="text-align: center;" align="center">

[![PyPI Version](https://img.shields.io/pypi/v/evlib.svg)](https://pypi.org/project/evlib/)
[![Python Versions](https://img.shields.io/pypi/pyversions/evlib.svg)](https://pypi.org/project/evlib/)
[![Documentation](https://readthedocs.org/projects/evlib/badge/?version=latest)](https://evlib.readthedocs.io/en/latest/?badge=latest)
[![Python](https://github.com/tallamjr/evlib/actions/workflows/pytest.yml/badge.svg)](https://github.com/tallamjr/evlib/actions/workflows/pytest.yml)
[![Rust](https://github.com/tallamjr/evlib/actions/workflows/rust.yml/badge.svg)](https://github.com/tallamjr/evlib/actions/workflows/rust.yml)
[![License](https://img.shields.io/github/license/tallamjr/evlib)](https://github.com/tallamjr/evlib/blob/master/LICENSE.md)

</div>

An event camera processing library with Rust backend and Python bindings,
designed for scalable data processing with real-world event camera datasets.

## Core Features

- **Universal Format Support**: Load data from H5, AEDAT, EVT2/3, AER, and text formats
- **Automatic Format Detection**: No need to specify format types manually
- **Polars DataFrame Integration**: High-performance DataFrame operations with up to 360M events/s filtering
- **Event Filtering**: Comprehensive filtering with temporal, spatial, and polarity options
- **Event Representations**: Stacked histograms, voxel grids, and mixed density stacks
- **Neural Network Models**: E2VID model loading and inference
- **Real-time Data Processing**: Handle large datasets (550MB+ files) efficiently
- **Polarity Encoding**: Automatic conversion between 0/1 and -1/1 polarities
- **Rust Performance**: Memory-safe, high-performance backend with Python bindings

**In Development:** Advanced neural network processing (hopefully with Rust
backend, maybe Candle) Real-time visualization (Only simulated working at the
moment — see `wasm-evlib`)

**Note**: The Rust backend currently focuses on data loading and processing,
with Python modules providing advanced features like filtering and
representations.

---

<!-- mtoc-start -->

* [Quick Start](#quick-start)
  * [Basic Usage](#basic-usage)
  * [Advanced Filtering](#advanced-filtering)
  * [Event Representations](#event-representations)
* [Installation](#installation)
  * [Basic Installation](#basic-installation)
  * [Development Installation](#development-installation)
  * [System Dependencies](#system-dependencies)
  * [Performance-Optimized Installation](#performance-optimized-installation)
* [Polars DataFrame Integration](#polars-dataframe-integration)
  * [Key Benefits](#key-benefits)
  * [API Overview](#api-overview)
    * [Loading Data](#loading-data)
    * [Advanced Features](#advanced-features)
    * [Utility Functions](#utility-functions)
  * [Performance Benchmarks](#performance-benchmarks)
  * [Benchmarking and Monitoring](#benchmarking-and-monitoring)
  * [Performance Examples](#performance-examples)
    * [Optimal Loading for Different File Sizes](#optimal-loading-for-different-file-sizes)
    * [Memory Monitoring](#memory-monitoring)
  * [Troubleshooting Large Files](#troubleshooting-large-files)
    * [Memory Constraints](#memory-constraints)
    * [Performance Tuning](#performance-tuning)
    * [Common Issues and Solutions](#common-issues-and-solutions)
* [Available Python Modules](#available-python-modules)
  * [Core Modules](#core-modules)
  * [Module Overview](#module-overview)
* [Examples](#examples)
* [Development](#development)
  * [Testing](#testing)
    * [Core Testing](#core-testing)
    * [Documentation Testing](#documentation-testing)
    * [Code Quality](#code-quality)
  * [Building](#building)
    * [Requirements](#requirements)
* [Community & Support](#community--support)
* [License](#license)

<!-- mtoc-end -->

## Quick Start

### Basic Usage
```python
import evlib

# Load events from any supported format (automatic detection)
df = evlib.load_events("data/slider_depth/events.txt").collect(engine='streaming')

# Or load as LazyFrame for memory-efficient processing
lf = evlib.load_events("data/slider_depth/events.txt")

# Basic event information
print(f"Loaded {len(df)} events")
print(f"Resolution: {df['x'].max()} x {df['y'].max()}")
print(f"Duration: {df['timestamp'].max() - df['timestamp'].min()}")

# Convert to NumPy arrays for compatibility
x_coords = df['x'].to_numpy()
y_coords = df['y'].to_numpy()
timestamps = df['timestamp'].to_numpy()
polarities = df['polarity'].to_numpy()
```

### Advanced Filtering
```python
import evlib
import evlib.filtering as evf

# High-level preprocessing pipeline
processed = evlib.filtering.preprocess_events(
    "data/slider_depth/events.txt",
    t_start=0.1, t_end=0.5,
    roi=(100, 500, 100, 400),
    polarity=1,
    remove_hot_pixels=True,
    remove_noise=True,
    hot_pixel_threshold=99.9,
    refractory_period_us=1000
)

# Individual filters (work with LazyFrames)
events = evlib.load_events("data/slider_depth/events.txt")
time_filtered = evlib.filtering.filter_by_time(events, t_start=0.1, t_end=0.5)
spatial_filtered = evlib.filtering.filter_by_roi(time_filtered, x_min=100, x_max=500, y_min=100, y_max=400)
clean_events = evlib.filtering.filter_hot_pixels(spatial_filtered, threshold_percentile=99.9)
denoised = evlib.filtering.filter_noise(clean_events, method="refractory", refractory_period_us=1000)
```

### Event Representations
```python
import evlib
import evlib.representations as evr

# Create voxel grid representation (reliable alternative to stacked histogram)
events = evlib.load_events("data/slider_depth/events.txt")
events_df = events.collect()
voxel_df = evr.create_voxel_grid_py(
    events_df,
    _height=480, _width=640,
    nbins=10
)

# Create mixed density stack representation
mixed_df = evr.create_mixed_density_stack_py(
    events_df,
    _height=480, _width=640,
    nbins=10, window_duration_ms=50.0
)
print(f"Created voxel grid with {len(voxel_df)} entries and mixed density stack with {len(mixed_df)} entries")

# High-level preprocessing for neural networks
events = evlib.load_events("data/slider_depth/events.txt")
events_df = events.collect()
data_df = evr.create_voxel_grid_py(
    events_df,
    _height=480, _width=640,
    nbins=10
)
print(f"Preprocessed {len(data_df)} entries for detection pipeline")

# Performance benchmarking against RVT (manual comparison available)
import time

start_time = time.time()
events = evlib.load_events("data/slider_depth/events.txt")
events_df = events.collect()
results_df = evr.create_voxel_grid_py(events_df, _height=480, _width=640, nbins=10)
processing_time = time.time() - start_time

print(f"evlib processing: {processing_time:.3f}s for {len(results_df)} voxel grid entries")
print("Implement equivalent RVT PyTorch pipeline for direct comparison")
```

## Installation

### Basic Installation
```bash
pip install evlib

# For Polars DataFrame support (recommended)
pip install evlib[polars]
```

### Development Installation
```bash
# Clone the repository
git clone https://github.com/tallamjr/evlib.git
cd evlib

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode with all features
pip install -e ".[dev,polars]"

# Build the Rust extensions
maturin develop
```

### System Dependencies
```bash
# Ubuntu/Debian
sudo apt install libhdf5-dev pkg-config

# macOS
brew install hdf5 pkg-config
```

### Performance-Optimized Installation

For optimal performance, ensure you have the recommended system configuration:

**System Requirements:**
- **RAM**: 8GB+ recommended for files >100M events
- **Python**: 3.10+ (3.12 recommended for best performance)
- **Polars**: Latest version for advanced DataFrame operations

**Installation for Performance:**
```bash
# Install with Polars support (recommended)
pip install "evlib[polars]"

# For development with all performance features
pip install "evlib[dev,polars]"

# Verify installation with benchmark
python -c "import evlib; print('evlib installed successfully')"
python benchmark_memory.py  # Test memory efficiency
```

**Optional Performance Dependencies:**
```bash
# For advanced memory monitoring
pip install psutil

# For parallel processing (already included in dev)
pip install multiprocessing-logging
```

## Polars DataFrame Integration

evlib provides comprehensive Polars DataFrame support for high-performance event data processing:

### Key Benefits
- **Performance**: 1.9M+ events/s loading speed, 360M+ events/s filtering speed
- **Memory Efficiency**: ~23 bytes/event (5x better than typical 110 bytes/event)
- **Expressive Queries**: SQL-like operations for complex data analysis
- **Lazy Evaluation**: Query optimization for better performance
- **Ecosystem Integration**: Seamless integration with data science tools

### API Overview

#### Loading Data
```python
import evlib

# Load as LazyFrame (recommended)
events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()  # Collect to DataFrame when needed

# Automatic format detection and optimization
events = evlib.load_events("data/slider_depth/events.txt")  # EVT2 format automatically detected
print(f"Format: {evlib.formats.detect_format('data/slider_depth/events.txt')}")
print(f"Description: {evlib.formats.get_format_description('EVT2')}")

```

#### Advanced Features
```python
import evlib
import polars as pl

# Chain operations with LazyFrames for optimal performance
events = evlib.load_events("data/slider_depth/events.txt")
result = events.filter(pl.col("polarity") == 1).with_columns([
    pl.col("timestamp").dt.total_microseconds().alias("time_us"),
    (pl.col("x") + pl.col("y")).alias("diagonal_pos")
]).collect()

# Memory-efficient temporal analysis
time_stats = events.with_columns([
    pl.col("timestamp").dt.total_microseconds().alias("time_us")
]).group_by([
    (pl.col("time_us") // 1_000_000).alias("time_second")  # Group by second
]).agg([
    pl.len().alias("event_count"),
    pl.col("polarity").mean().alias("avg_polarity")
]).collect()

# Combine with filtering module for complex operations
import evlib.filtering as evf
filtered = evlib.filtering.filter_by_time(events, t_start=0.1, t_end=0.5)
analysis = filtered.with_columns([
    pl.col("timestamp").dt.total_microseconds().alias("time_us")
]).collect()
```

#### Utility Functions
```python
import evlib
import polars as pl
import evlib.filtering as evf

# Built-in format detection
format_info = evlib.formats.detect_format("data/slider_depth/events.txt")
print(f"Detected format: {format_info}")

# Spatial filtering using dedicated filtering functions (preferred)
events = evlib.load_events("data/slider_depth/events.txt")
spatial_filtered = evlib.filtering.filter_by_roi(events, x_min=100, x_max=200, y_min=50, y_max=150)

# Or using direct Polars operations
manual_filtered = events.filter(
    (pl.col("x") >= 100) & (pl.col("x") <= 200) &
    (pl.col("y") >= 50) & (pl.col("y") <= 150)
)

# Temporal analysis with Polars operations
rates = events.with_columns([
    pl.col("timestamp").dt.total_microseconds().alias("time_us")
]).group_by([
    (pl.col("time_us") // 10_000).alias("time_10ms")  # Group by 10ms
]).agg([
    pl.len().alias("event_rate"),
    pl.col("polarity").mean().alias("avg_polarity")
]).collect()

# Save processed data
processed = evlib.filtering.preprocess_events("data/slider_depth/events.txt", t_start=0.1, t_end=0.5)
processed_df = processed.collect()
x, y, t_us, p = processed_df.select(["x", "y", "timestamp", "polarity"]).to_numpy().T
# Ensure correct dtypes for save function
x = x.astype(np.int64)
y = y.astype(np.int64)
p = p.astype(np.int64)
# Convert microseconds to seconds for save function
t = t_us.astype(np.float64) / 1_000_000
evlib.formats.save_events_to_hdf5(x, y, t, p, "output.h5")
print(f"Successfully saved {len(x)} processed events to HDF5")
```

### Performance Benchmarks

![Performance Benchmarks](./benches/performance_benchmark.png)

**Benchmark Results:**
- **Loading Speed**: 1.9M+ events/second average across formats
- **Filter Speed**: 360M+ events/second for complex filtering operations
- **Memory Efficiency**: ~23 bytes/event
- **Format Performance**: RAW binary (2.6M events/s) > HDF5 (2.5M events/s) > Text (0.6M events/s)

### Benchmarking and Monitoring

Run performance benchmarks to verify optimizations:

```bash
# Verify README performance claims and generate plots
python benches/benchmark_performance_readme.py

# Memory efficiency benchmark
python benches/benchmark_memory.py

# Test with your own data
python -c "
import evlib
import time
start = time.time()
events = evlib.load_events('data/slider_depth/events.txt')
df = events.collect()
print(f'Loaded {len(df):,} events in {time.time()-start:.2f}s')
print(f'Format: {evlib.detect_format(\"data/slider_depth/events.txt\")}')
print(f'Memory per event: {df.estimated_size() / len(df):.1f} bytes')
"
```

### Performance Examples

#### Optimal Loading for Different File Sizes
```python
import evlib
import evlib.filtering as evf
import polars as pl

# Small files (<5M events) - Direct loading
events_small = evlib.load_events("data/slider_depth/events.txt")
df_small = events_small.collect()

# Large files (>5M events) - Automatic streaming
events_large = evlib.load_events("data/slider_depth/events.txt")
# Same API, automatically uses streaming for memory efficiency

# Memory-efficient filtering on large datasets using filtering module
filtered = evlib.filtering.filter_by_time(events_large, t_start=1.0, t_end=2.0)
positive_events = evlib.filtering.filter_by_polarity(filtered, polarity=1)

# Or using direct Polars operations
manual_filtered = events_large.filter(
    (pl.col("timestamp").dt.total_microseconds() / 1_000_000 > 1.0) &
    (pl.col("polarity") == 1)
).collect()
```

#### Memory Monitoring
```python
import evlib
import psutil
import os

def monitor_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

# Monitor memory usage during loading
initial_mem = monitor_memory()
events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()
peak_mem = monitor_memory()

print(f"Memory used: {peak_mem - initial_mem:.1f} MB")
print(f"Memory per event: {(peak_mem - initial_mem) * 1024 * 1024 / len(df):.1f} bytes")
print(f"Polars DataFrame size: {df.estimated_size() / 1024 / 1024:.1f} MB")
```

### Troubleshooting Large Files

#### Memory Constraints
- **Automatic Streaming**: Files >5M events use streaming by default
- **LazyFrame Operations**: Memory-efficient processing without full materialization
- **Memory Monitoring**: Use `benchmark_memory.py` to track usage
- **System Requirements**: Recommend 8GB+ RAM for files >100M events

#### Performance Tuning
- **Optimal Chunk Size**: System automatically calculates based on available memory
- **LazyFrame Operations**: Use `.lazy()` for complex filtering chains
- **Memory-Efficient Formats**: RAW binary formats provide best performance, followed by HDF5
- **Progress Reporting**: Large files show progress during loading

#### Common Issues and Solutions

**Issue**: Out of memory errors
```python
import evlib
import evlib.filtering as evf

# Solution: Use filtering before collecting (streaming activates automatically)
events = evlib.load_events("data/slider_depth/events.txt")
# Streaming activates automatically for files >5M events

# Apply filtering before collecting to reduce memory usage
filtered = evlib.filtering.filter_by_time(events, t_start=0.1, t_end=0.5)
df = filtered.collect()  # Only collect when needed

# Or stream to disk using Polars
filtered.sink_parquet("filtered_events.parquet")
```

**Issue**: Slow loading performance
```python
import evlib
import evlib.filtering as evf
import polars as pl

# Solution: Use LazyFrame for complex operations and filtering module
events = evlib.load_events("data/slider_depth/events.txt")

# Use filtering module for optimized operations
result = evlib.filtering.filter_by_roi(events, x_min=0, x_max=640, y_min=0, y_max=480)
df = result.collect()

# Or chain Polars operations
result = events.filter(pl.col("polarity") == 1).select(["x", "y", "timestamp"]).collect()
```

**Issue**: Memory usage higher than expected
```python
import evlib

# Solution: Monitor and verify optimization
events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()
print(f"Memory efficiency: {df.estimated_size() / len(df)} bytes/event")
print(f"DataFrame schema: {df.schema}")
print(f"Number of events: {len(df):,}")

# Check format detection
format_info = evlib.formats.detect_format("data/slider_depth/events.txt")
print(f"Format: {format_info}")
```

## Available Python Modules

evlib provides several Python modules for different aspects of event processing:

### Core Modules
- **`evlib.formats`**: Direct Rust access for format loading and detection
- **`evlib.filtering`**: High-performance event filtering with Polars
- **`evlib.representations`**: Event representations (stacked histograms, voxel grids)
- **`evlib.models`**: Neural network model loading and inference (Under construction)

### Module Overview
```python
import evlib
import evlib.filtering as evf
import evlib.representations as evr

# Core event loading (returns Polars LazyFrame)
events = evlib.load_events("data/slider_depth/events.txt")

# Format detection and description
format_info = evlib.formats.detect_format("data/slider_depth/events.txt")
description = evlib.formats.get_format_description("HDF5")

# Advanced filtering
filtered = evlib.filtering.preprocess_events("data/slider_depth/events.txt", t_start=0.1, t_end=0.5)
time_filtered = evlib.filtering.filter_by_time(events, t_start=0.1, t_end=0.5)

# Event representations
events = evlib.load_events("data/slider_depth/events.txt")
events_df = events.collect()
voxel_df = evr.create_voxel_grid_py(events_df, _height=480, _width=640, nbins=10)
mixed_df = evr.create_mixed_density_stack_py(events_df, _height=480, _width=640, nbins=10)
print(f"Created voxel grid with {len(voxel_df)} entries and mixed density stack with {len(mixed_df)} entries")

# Neural network models (limited functionality)
# from evlib.models import ModelConfig  # If available - under development

# Data saving (need to get arrays first)
import numpy as np
df = events.collect()
x, y, t_dur, p = df.select(["x", "y", "timestamp", "polarity"]).to_numpy().T
# Ensure correct dtypes for save functions
x = x.astype(np.int64)
y = y.astype(np.int64)
p = p.astype(np.int64)
# Convert Duration to seconds for save functions
t = t_dur.astype('float64') / 1e6  # Convert microseconds to seconds
evlib.formats.save_events_to_hdf5(x, y, t, p, "output.h5")
evlib.formats.save_events_to_text(x, y, t, p, "output.txt")
print(f"Successfully saved {len(x)} events to both HDF5 and text formats")
```

## Examples

Run examples:
```bash
# Test all notebooks
pytest --nbmake examples/

# Run specific examples
python examples/simple_example.py
python examples/filtering_demo.py
python examples/stacked_histogram_demo.py
```

## Development

### Testing

#### Core Testing
```bash
# Run all tests (Python and Rust)
pytest
cargo test

# Test specific modules
pytest tests/test_filtering.py
pytest tests/test_representations.py
pytest tests/test_evlib_exact_match.py

# Test notebooks (including examples)
pytest --nbmake examples/

# Test with coverage
pytest --cov=evlib
```

#### Documentation Testing
All code examples in the documentation are automatically tested to ensure they work correctly:

```bash
# Test all documentation examples
pytest --markdown-docs docs/

# Test specific documentation file
pytest --markdown-docs docs/getting-started/quickstart.md

# Use the convenient test script
python scripts/test_docs.py --list    # List testable files
python scripts/test_docs.py --report  # Generate report

# Test specific documentation section
pytest --markdown-docs docs/user-guide/
pytest --markdown-docs docs/getting-started/
```

#### Code Quality
```bash
# Format code
black python/ tests/ examples/
cargo fmt

# Run linting
ruff check python/ tests/
cargo clippy

# Check types
mypy python/evlib/
```

### Building

#### Requirements

- **Rust**: Stable toolchain (see `rust-toolchain.toml`)
- **Python**: ≥3.10 (3.12 recommended)
- **Maturin**: For building Python extensions

```bash
# Development build
maturin develop --features python # python required to register python modules

# Build with features
maturin develop --features polars
maturin develop --features pytorch

# Release build
maturin build --release
```

## Community & Support

![xkcd](https://imgs.xkcd.com/comics/the_best_camera.png){ width=100% }

- **GitHub**: [tallamjr/evlib](https://github.com/tallamjr/evlib)
- **Issues**: Report bugs and request features
- **Discussions**: Community Q&A and ideas

## License

MIT License - see [LICENSE.md](LICENSE.md) for details.
