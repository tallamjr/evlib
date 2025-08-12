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
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://tallamjr.github.io/evlib/)
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

*Note**: The Rust backend currently focuses on data loading and processing,
with Python modules providing advanced features like filtering and
representations.

---

<!-- mtoc-start -->

* [Quick Start](#quick-start)
  * [What are Event Cameras?](#what-are-event-cameras)
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
* [High-Performance PyTorch DataLoader](#high-performance-pytorch-dataloader)
  * [Key Features](#key-features)
  * [Quick Start](#quick-start-1)
  * [Architecture Overview](#architecture-overview)
  * [Performance Benefits](#performance-benefits)
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

### What are Event Cameras?

Event cameras (also called neuromorphic or dynamic vision sensors) differ
fundamentally from traditional frame-based cameras. Instead of capturing images
at fixed frame rates, they operate asynchronously, with each pixel independently
reporting changes in brightness as they occur.

Each **event** is represented as a 4-tuple:

$$e = (x, y, t, p)$$

Where:
- $x, y \in \mathbb{N}$: Pixel coordinates in the sensor array (e.g., $0 \leq x < 640$, $0 \leq y < 480$)
- $t \in \mathbb{R}^+$: Timestamp when the brightness change occurred (microsecond precision)
- $p \in \{-1, +1\}$ or $\{0, 1\}$: Polarity indicating brightness change direction

An event is triggered when the logarithmic brightness change exceeds a threshold:

$$\log(L(x,y,t)) - \log(L(x,y,t_{last})) > \pm C$$

where $L(x,y,t)$ is the brightness at pixel $(x,y)$ at time $t$, and $C$ is the contrast threshold.

**Key advantages:**
- **High temporal resolution**: Microsecond precision vs. millisecond frame intervals
- **High dynamic range**: 120dB+ vs. ~60dB for conventional cameras
- **Low power consumption**: Only active pixels generate data
- **No motion blur**: Events capture instantaneous changes
- **Sparse data**: Only reports meaningful changes, reducing bandwidth

Event cameras excel at tracking fast motion, operating in challenging lighting
conditions, and applications requiring precise temporal information like
robotics, autonomous vehicles, and augmented reality.

Below is an overview of what the data "looks" like...

<center>
<img src="./wasm-evlib/wasm-sim.png" style="width:480px;height:320px;">
</center>


### Basic Usage
```python
import evlib

# Load events from any supported format (automatic detection)
df = evlib.load_events("data/prophersee/samples/evt2/80_balls.raw").collect(engine='streaming')

# Or load as LazyFrame for memory-efficient processing
lf = evlib.load_events("data/prophersee/samples/evt2/80_balls.raw")

# Basic event information
print(f"Loaded {len(df)} events")
print(f"Resolution: {df['x'].max()} x {df['y'].max()}")
print(f"Duration: {df['t'].max() - df['t'].min()}")

# Convert to NumPy arrays for compatibility
x_coords = df['x'].to_numpy()
y_coords = df['y'].to_numpy()
timestamps = df['t'].to_numpy()
polarities = df['polarity'].to_numpy()
```

### Advanced Filtering
```python notest
import evlib
import polars as pl

# Load events as LazyFrame for efficient processing
events = evlib.load_events("data/prophersee/samples/evt3/pedestrians.raw")

# Time filtering using Polars operations
time_filtered = events.with_columns([
    (pl.col('t').dt.total_microseconds() / 1_000_000).alias('time_seconds')
]).filter(
    (pl.col('time_seconds') >= 0.1) & (pl.col('time_seconds') <= 0.5)
)

# Spatial filtering (Region of Interest)
spatial_filtered = time_filtered.filter(
    (pl.col('x') >= 100) & (pl.col('x') <= 500) &
    (pl.col('y') >= 100) & (pl.col('y') <= 400)
)

# Polarity filtering
polarity_filtered = spatial_filtered.filter(pl.col('polarity') == 1)

# Collect final results
filtered_df = polarity_filtered.collect()
print(f"Filtered to {len(filtered_df)} events")
```

### Event Representations

evlib provides comprehensive event representation functions for computer vision and neural network applications:

```python notest
import evlib
import evlib.representations as evr
import polars as pl

# Load events and create representations
events = evlib.load_events("data/prophersee/samples/hdf5/pedestrians.hdf5")
events_df = events.collect()

# Create stacked histogram (replaces RVT preprocessing)
hist = evr.create_stacked_histogram(
    events_df,
    height=180, width=240,
    bins=5, window_duration_ms=50.0,
    _count_cutoff=5
)
print(f"Created stacked histogram with {len(hist)} spatial bins")

# Create mixed density stack representation
density = evr.create_mixed_density_stack(
    events_df,
    height=180, width=240,
    window_duration_ms=50.0
)
print(f"Created mixed density stack with {len(density)} entries")

# Create voxel grid representation
voxel = evr.create_voxel_grid(
    events_df,
    height=180, width=240,
    n_time_bins=3
)
print(f"Created voxel grid with {len(voxel)} voxels")

# Advanced representations (require data type conversion)
# Convert timestamp and ensure proper dtypes for advanced functions
small_events = events.limit(10000).collect()
converted_events = small_events.with_columns([
    pl.col('t').dt.total_microseconds().cast(pl.Float64).alias('t_microseconds'),
    pl.col('x').cast(pl.Int64),
    pl.col('y').cast(pl.Int64),
    pl.col('polarity').cast(pl.Int64)
]).select(['x', 'y', 't_microseconds', 'polarity']).rename({'t_microseconds': 't'})

# Create time surface representation
time_surface = evr.create_timesurface(
    converted_events,
    height=180, width=240,
    dt=50000.0,    # time step in microseconds
    tau=10000.0    # decay constant in microseconds
)
print(f"Created time surface with {len(time_surface)} pixels")

# Create averaged time surface
avg_time_surface = evr.create_averaged_timesurface(
    converted_events,
    height=180, width=240,
    cell_size=1, surface_size=1,
    time_window=50000.0, tau=10000.0
)
print(f"Created averaged time surface with {len(avg_time_surface)} pixels")
```

## Installation

### Basic Installation
```bash
pip install evlib

# For Polars DataFrame support (recommended)
pip install evlib[polars]

# For PyTorch integration
pip install evlib[pytorch]
```

### Development Installation

We recommend using [uv](https://docs.astral.sh/uv/getting-started/installation/) for fast, reliable Python package management:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/tallamjr/evlib.git
cd evlib

# Create virtual environment and install dependencies
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev,polars]"

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

# For development with all performance features (using uv)
uv pip install "evlib[dev,polars]"

# Verify installation with benchmark
python -c "import evlib; print('evlib installed successfully')"
python benchmark_memory.py  # Test memory efficiency
```

**Optional Performance Dependencies:**
```bash
# For advanced memory monitoring
uv pip install psutil

# For parallel processing (already included in dev)
uv pip install multiprocessing-logging
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
events = evlib.load_events("data/prophersee/samples/evt2/80_balls.raw")
df = events.collect()  # Collect to DataFrame when needed

# Automatic format detection and optimization
events = evlib.load_events("data/prophersee/samples/evt2/80_balls.raw")  # EVT2 format automatically detected
print(f"Format: {evlib.formats.detect_format('data/prophersee/samples/evt2/80_balls.raw')}")
print(f"Description: {evlib.formats.get_format_description('EVT2')}")

```

#### Advanced Features
```python notest
import evlib
import polars as pl

# Chain operations with LazyFrames for optimal performance
events = evlib.load_events("data/prophersee/samples/hdf5/pedestrians.hdf5")
result = events.filter(pl.col("polarity") == 1).with_columns([
    pl.col("t").dt.total_microseconds().alias("time_us"),
    (pl.col("x") + pl.col("y")).alias("diagonal_pos")
]).collect()

# Memory-efficient temporal analysis
time_stats = events.with_columns([
    pl.col("t").dt.total_microseconds().alias("time_us")
]).group_by([
    (pl.col("time_us") // 1_000_000).alias("time_second")  # Group by second
]).agg([
    pl.len().alias("event_count"),
    pl.col("polarity").mean().alias("avg_polarity")
]).collect()

# Complex filtering operations with Polars
filtered = events.with_columns([
    (pl.col('t').dt.total_microseconds() / 1_000_000).alias('time_seconds')
]).filter(
    (pl.col('time_seconds') >= 0.1) & (pl.col('time_seconds') <= 0.5)
)
analysis = filtered.with_columns([
    pl.col("t").dt.total_microseconds().alias("time_us")
]).collect()
```

#### Utility Functions
```python notest
import evlib
import polars as pl
import evlib.filtering as evf

# Built-in format detection
format_info = evlib.formats.detect_format("data/prophersee/samples/evt3/pedestrians.raw")
print(f"Detected format: {format_info}")

# Spatial filtering using Polars operations
events = evlib.load_events("data/prophersee/samples/evt3/pedestrians.raw")
spatial_filtered = events.filter(
    (pl.col("x") >= 100) & (pl.col("x") <= 200) &
    (pl.col("y") >= 50) & (pl.col("y") <= 150)
)

# Chain multiple filters efficiently
complex_filtered = events.filter(
    (pl.col("x") >= 100) & (pl.col("x") <= 200) &
    (pl.col("y") >= 50) & (pl.col("y") <= 150) &
    (pl.col("polarity") == 1)
)

# Temporal analysis with Polars operations
rates = events.with_columns([
    pl.col("t").dt.total_microseconds().alias("time_us")
]).group_by([
    (pl.col("time_us") // 10_000).alias("time_10ms")  # Group by 10ms
]).agg([
    pl.len().alias("event_rate"),
    pl.col("polarity").mean().alias("avg_polarity")
]).collect()

# Save processed data (working example)
processed = events.with_columns([
    (pl.col('t').dt.total_microseconds() / 1_000_000).alias('time_seconds')
]).filter(
    (pl.col('time_seconds') >= 0.1) & (pl.col('time_seconds') <= 0.5)
)
processed_df = processed.collect()
data_arrays = processed_df.select(["x", "y", "t", "polarity"]).to_numpy()
x, y, t_us, p = data_arrays.T
# Convert Duration microseconds to seconds for save function
t = t_us.astype('float64') / 1_000_000
evlib.formats.save_events_to_hdf5(x.astype('int16'), y.astype('int16'), t, p.astype('int8'), "output.h5")
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
events = evlib.load_events('data/prophersee/samples/evt2/80_balls.raw')
df = events.collect()
print(f'Loaded {len(df):,} events in {time.time()-start:.2f}s')
print(f'Format: {evlib.detect_format(\"data/prophersee/samples/evt2/80_balls.raw\")}')
print(f'Memory per event: {df.estimated_size() / len(df):.1f} bytes')
"
```

### Performance Examples

#### Optimal Loading for Different File Sizes
```python notest
import evlib
import evlib.filtering as evf
import polars as pl

# Small files (<5M events) - Direct loading
events_small = evlib.load_events("data/prophersee/samples/evt2/80_balls.raw")
df_small = events_small.collect()

# Large files (>5M events) - Automatic streaming
events_large = evlib.load_events("data/prophersee/samples/hdf5/pedestrians.hdf5")
# Same API, automatically uses streaming for memory efficiency

# Memory-efficient filtering on large datasets using Polars
filtered = events_large.with_columns([
    (pl.col('t').dt.total_microseconds() / 1_000_000).alias('time_seconds')
]).filter(
    (pl.col('time_seconds') >= 1.0) & (pl.col('time_seconds') <= 2.0)
)
positive_events = filtered.filter(pl.col("polarity") == 1)

# Collect only when needed for memory efficiency
result_df = positive_events.collect()
print(f"Filtered to {len(result_df)} events")
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
events = evlib.load_events("data/prophersee/samples/evt2/80_balls.raw")
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
```python notest
import evlib
import evlib.filtering as evf

# Solution: Use filtering before collecting (streaming activates automatically)
events = evlib.load_events("data/prophersee/samples/hdf5/pedestrians.hdf5")
# Streaming activates automatically for files >5M events

# Apply filtering before collecting to reduce memory usage
filtered = events.with_columns([
    (pl.col('t').dt.total_microseconds() / 1_000_000).alias('time_seconds')
]).filter(
    (pl.col('time_seconds') >= 0.1) & (pl.col('time_seconds') <= 0.5)
)
df = filtered.collect()  # Only collect when needed

# Or stream to disk using Polars
filtered.sink_parquet("filtered_events.parquet")
```

**Issue**: Slow loading performance
```python
import evlib
import evlib.filtering as evf
import polars as pl

# Solution: Use LazyFrame for complex operations
events = evlib.load_events("data/prophersee/samples/evt2/80_balls.raw")

# Use Polars operations for optimized filtering
result = events.filter(
    (pl.col("x") >= 0) & (pl.col("x") <= 640) &
    (pl.col("y") >= 0) & (pl.col("y") <= 480)
)
df = result.collect()

# Or chain Polars operations
result = events.filter(pl.col("polarity") == 1).select(["x", "y", "t"]).collect()
```

**Issue**: Memory usage higher than expected
```python notest
import evlib

# Solution: Monitor and verify optimization
events = evlib.load_events("data/prophersee/samples/evt3/pedestrians.raw")
df = events.collect()
print(f"Memory efficiency: {df.estimated_size() / len(df)} bytes/event")
print(f"DataFrame schema: {df.schema}")
print(f"Number of events: {len(df):,}")

# Check format detection
format_info = evlib.formats.detect_format("data/prophersee/samples/evt3/pedestrians.raw")
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
```python notest
import evlib
import evlib.filtering as evf
import evlib.representations as evr

# Core event loading (returns Polars LazyFrame)
events = evlib.load_events("data/prophersee/samples/hdf5/pedestrians.hdf5")

# Format detection and description
format_info = evlib.formats.detect_format("data/prophersee/samples/hdf5/pedestrians.hdf5")
description = evlib.formats.get_format_description("HDF5")

# Advanced filtering using Polars operations
filtered = events.with_columns([
    (pl.col('t').dt.total_microseconds() / 1_000_000).alias('time_seconds')
]).filter(
    (pl.col('time_seconds') >= 0.1) & (pl.col('time_seconds') <= 0.5)
)
time_filtered = filtered.collect()

# Event representations (working examples)
events_df = events.collect()
hist = evr.create_stacked_histogram(events_df, height=180, width=240, bins=5)
voxel = evr.create_voxel_grid(events_df, height=180, width=240, n_time_bins=3)

# Advanced representations (with proper data conversion)
small_events = events.limit(10000).collect()
converted_events = small_events.with_columns([
    pl.col('t').dt.total_microseconds().cast(pl.Float64).alias('t_microseconds'),
    pl.col('x').cast(pl.Int64),
    pl.col('y').cast(pl.Int64),
    pl.col('polarity').cast(pl.Int64)
]).select(['x', 'y', 't_microseconds', 'polarity']).rename({'t_microseconds': 't'})
time_surface = evr.create_timesurface(converted_events, height=180, width=240, dt=50000.0, tau=10000.0)

# Neural network models (limited functionality)
from evlib.models import ModelConfig  # If available

# Data saving (working examples)
df = events.collect()
data_arrays = df.select(["x", "y", "t", "polarity"]).to_numpy()
x, y, t_us, p = data_arrays.T
# Convert Duration microseconds to seconds for save functions
t = t_us.astype('float64') / 1_000_000
evlib.formats.save_events_to_hdf5(x.astype('int16'), y.astype('int16'), t, p.astype('int8'), "output.h5")
evlib.formats.save_events_to_text(x.astype('int16'), y.astype('int16'), t, p.astype('int8'), "output.txt")
```

## High-Performance PyTorch DataLoader

evlib includes an optimized PyTorch dataloader implementation that showcases best practices for event camera data processing:

### Key Features
- **Polars → PyTorch Integration**: Native `.to_torch()` conversion for zero-copy data transfer
- **RVT Preprocessing**: Loads real RVT (Recurrent Vision Transformer) preprocessed data
- **Statistical Feature Extraction**: Efficiently extracts 91 features from stacked histograms
- **High Throughput**: Achieves 13,000+ samples/sec training throughput
- **Memory Efficient**: Lazy evaluation and batched processing

### Quick Start
```python
# New: Use the built-in PyTorch integration
import evlib
import torch
from torch.utils.data import DataLoader
from evlib.pytorch import create_dataloader, load_rvt_data, PolarsDataset, create_rvt_transform

# Option 1: Raw event data (since RVT data not available in CI)
events = evlib.load_events("data/slider_depth/events.txt")
dataloader = create_dataloader(events, data_type="events", batch_size=256)

# Option 2: Manual setup for custom transforms (for RVT data when available)
# lazy_df = load_rvt_data("data/gen4_1mpx_processed_RVT/val/moorea_2019-02-21_000_td_2257500000_2317500000")

# Option 3: Raw event data from various formats
# events = evlib.load_events("data/eTram/h5/val_2/val_night_007_td.h5")  # eTram dataset
# events = evlib.load_events("data/prophersee/samples/hdf5/pedestrians.hdf5")  # Prophesee format
events = evlib.load_events("data/slider_depth/events.txt")  # Text format
dataloader = create_dataloader(events, data_type="events")

# Option 4: Advanced - Custom transform using provided functions (for RVT data)
# if lazy_df is not None:
#     # Use the built-in RVT transform
#     transform = create_rvt_transform()
#     dataset = PolarsDataset(lazy_df, batch_size=256, shuffle=True,
#                            transform=transform, drop_last=True)
#     dataloader = DataLoader(dataset, batch_size=None, num_workers=0)

# Option 5: Custom transform (if you need to modify the feature extraction)
def custom_split_features_labels(batch):
    """Custom transform to separate RVT features and labels from Polars batch"""
    feature_tensors = []

    # Add all temporal bin features (mean, std, max, nonzero for each bin)
    for bin_idx in range(20):
        for stat in ["mean", "std", "max", "nonzero"]:
            key = f"bin_{bin_idx:02d}_{stat}"
            if key in batch:
                feature_tensors.append(batch[key])

    # Add bounding box features
    for key in ["bbox_x", "bbox_y", "bbox_w", "bbox_h", "bbox_area"]:
        if key in batch:
            feature_tensors.append(batch[key])

    # Add activity features
    for key in ["total_activity", "active_pixels", "temporal_center"]:
        if key in batch:
            feature_tensors.append(batch[key])

    # Add normalized features (note: actual feature name is "t_norm", not "timestamp_norm")
    for key in ["t_norm", "bbox_area_norm", "activity_norm"]:
        if key in batch:
            feature_tensors.append(batch[key])

    # Stack into feature matrix and extract labels
    features = torch.stack(feature_tensors, dim=1)  # Shape: (batch_size, 91)
    labels = batch["label"].long()                  # Shape: (batch_size,)

    return {"features": features, "labels": labels}

# Train with real event camera data
for batch in dataloader:
    features = batch["features"]  # Shape: (256, 91) - 91 statistical features
    labels = batch["labels"]      # Shape: (256,) - object class labels

    # Your PyTorch training loop here
    outputs = model(features)
    loss = criterion(outputs, labels)
    # ... backward pass, optimizer step, etc.
```

### Architecture Overview
```
RVT HDF5 Data → Feature Extraction → Polars LazyFrame → .to_torch() → PyTorch Training
```

The dataloader demonstrates:
- Loading compressed HDF5 event representations (1198 samples, 20 temporal bins, 360×640 resolution)
- Statistical feature extraction (mean, std, max, nonzero) per temporal bin
- Object detection labels with bounding boxes and confidence scores
- Polars LazyFrame operations for memory-efficient processing
- Native PyTorch tensor conversion for optimal performance

### Performance Benefits
- **95%+ accuracy** on real 3-class classification tasks
- **13,262 samples/sec** training throughput
- **Memory efficient** processing of large event datasets
- **Zero-copy conversion** between Polars and PyTorch

See `examples/polars_pytorch_simplified.py` for the complete implementation and adapt it for your own event camera datasets.

## Examples

Run examples:
```bash
# Test all notebooks
pytest --nbmake examples/

# Run specific examples
python examples/simple_example.py
python examples/filtering_demo.py
python examples/stacked_histogram_demo.py

# Run the high-performance PyTorch dataloader example
python examples/polars_pytorch_simplified.py
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

![xkcd](https://imgs.xkcd.com/comics/universal_converter_box.png)

- **GitHub**: [tallamjr/evlib](https://github.com/tallamjr/evlib)
- **Issues**: Report bugs and request features
- **Discussions**: Community Q&A and ideas

## License

MIT License - see [LICENSE.md](LICENSE.md) for details.
