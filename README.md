# evlib: Event Camera Data Processing Library

<table align="center">
  <tr>
    <td>
      <img src="./docs/evlogo.png" width="70" alt="evlib logo" />
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
[![Documentation](https://readthedocs.org/projects/evlib/badge/?version=latest)](https://evlib.readthedocs.io/en/latest/?badge=latest)
[![Python](https://github.com/tallamjr/evlib/actions/workflows/pytest.yml/badge.svg)](https://github.com/tallamjr/evlib/actions/workflows/pytest.yml)
[![Rust](https://github.com/tallamjr/evlib/actions/workflows/rust.yml/badge.svg)](https://github.com/tallamjr/evlib/actions/workflows/rust.yml)
[![License](https://img.shields.io/github/license/tallamjr/evlib)](https://github.com/tallamjr/evlib/blob/master/LICENSE.md)

</div>

A robust event camera processing library with Rust backend and Python bindings, designed for reliable data processing with real-world event camera datasets.

## Current Status (v0.2.3)

**Fully Working Features:**
- Universal format support (H5, AEDAT, EVT2/3, AER, text)
- Automatic format detection
- Polars DataFrame integration (up to 97x speedup)
- High-performance event filtering
- Event representations (stacked histograms, voxel grids)
- Neural network model loading (E2VID)
- Large file support (tested with 200M+ events)

**In Development:**
- Advanced neural network processing (Rust backend temporarily disabled)
- Real-time visualization (Rust backend temporarily disabled)

**Note**: The core functionality is stable and production-ready. The Rust backend currently focuses on data loading and processing, with Python modules providing advanced features like filtering and representations.

## Core Features

- **Universal Format Support**: Load data from H5, AEDAT, EVT2/3, AER, and text formats
- **Automatic Format Detection**: No need to specify format types manually
- **Polars DataFrame Integration**: High-performance DataFrame operations with up to 97x speedup
- **Event Filtering**: Comprehensive filtering with temporal, spatial, and polarity options
- **Event Representations**: Stacked histograms, voxel grids, and mixed density stacks
- **Neural Network Models**: E2VID model loading and inference
- **Real-time Data Processing**: Handle large datasets (550MB+ files) efficiently
- **Polarity Encoding**: Automatic conversion between 0/1 and -1/1 polarities
- **Rust Performance**: Memory-safe, high-performance backend with Python bindings

## Quick Start

### Basic Usage
```python
import evlib

# Load events from any supported format (automatic detection)
df = evlib.load_events("path/to/your/data.h5").collect()

# Or load as LazyFrame for memory-efficient processing
lf = evlib.load_events("path/to/your/data.h5")

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
import evlib.filtering as evf

# High-level preprocessing pipeline
processed = evf.preprocess_events(
    "path/to/your/data.h5",
    t_start=0.1, t_end=0.5,
    roi=(100, 500, 100, 400),
    polarity=1,
    remove_hot_pixels=True,
    remove_noise=True,
    hot_pixel_threshold=99.9,
    refractory_period_us=1000
)

# Individual filters (work with LazyFrames)
events = evlib.load_events("path/to/your/data.h5")
time_filtered = evf.filter_by_time(events, t_start=0.1, t_end=0.5)
spatial_filtered = evf.filter_by_roi(time_filtered, x_min=100, x_max=500, y_min=100, y_max=400)
clean_events = evf.filter_hot_pixels(spatial_filtered, threshold_percentile=99.9)
denoised = evf.filter_noise(clean_events, method="refractory", refractory_period_us=1000)
```

### Event Representations
```python
import evlib.representations as evr

# Create stacked histogram (replaces RVT preprocessing)
hist = evr.create_stacked_histogram(
    "path/to/your/data.h5",
    height=480, width=640,
    nbins=10, window_duration_ms=50.0,
    count_cutoff=10
)

# Create mixed density stack representation
density = evr.create_mixed_density_stack(
    "path/to/your/data.h5",
    height=480, width=640,
    nbins=10, window_duration_ms=50.0
)

# Create voxel grid representation
voxel = evr.create_voxel_grid(
    "path/to/your/data.h5",
    height=480, width=640,
    nbins=5
)

# High-level preprocessing for neural networks
data = evr.preprocess_for_detection(
    "path/to/your/data.h5",
    representation="stacked_histogram",
    height=480, width=640,
    nbins=10, window_duration_ms=50
)

# Performance benchmarking against RVT
results = evr.benchmark_vs_rvt("path/to/your/data.h5", height=480, width=640)
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
- **Performance**: Up to 97x faster filtering compared to NumPy arrays
- **Memory Efficiency**: Optimized data structures reduce memory usage
- **Expressive Queries**: SQL-like operations for complex data analysis
- **Lazy Evaluation**: Query optimization for better performance
- **Ecosystem Integration**: Seamless integration with data science tools

### API Overview

#### Loading Data
```python
# Load as LazyFrame (recommended)
events = evlib.load_events("data.h5")
df = events.collect()  # Collect to DataFrame when needed

# Automatic format detection and optimization
events = evlib.load_events("data.evt2")  # EVT2 format automatically detected
print(f"Format: {evlib.detect_format('data.evt2')}")
print(f"Description: {evlib.get_format_description('EVT2')}")

# Direct Rust access (returns NumPy arrays)
x, y, t, p = evlib.formats.load_events("data.h5")
```

#### Advanced Features
```python
import polars as pl

# Chain operations with LazyFrames for optimal performance
events = evlib.load_events("data.h5")
result = events.filter(pl.col("polarity") == 1).with_columns([
    pl.col("timestamp").dt.total_microseconds().alias("time_us"),
    (pl.col("x") + pl.col("y")).alias("diagonal_pos")
]).collect()

# Memory-efficient temporal analysis
temporal_stats = events.group_by_dynamic(
    "timestamp",
    every="1s"
).agg([
    pl.len().alias("event_count"),
    pl.col("polarity").mean().alias("avg_polarity")
]).collect()

# Combine with filtering module for complex operations
import evlib.filtering as evf
filtered = evf.filter_by_time(events, t_start=0.1, t_end=0.5)
analysis = filtered.with_columns([
    pl.col("timestamp").dt.total_microseconds().alias("time_us")
]).collect()
```

#### Utility Functions
```python
import polars as pl
import evlib.filtering as evf

# Built-in format detection
format_info = evlib.detect_format("data.h5")
print(f"Detected format: {format_info}")

# Spatial filtering using dedicated filtering functions (preferred)
events = evlib.load_events("data.h5")
spatial_filtered = evf.filter_by_roi(events, x_min=100, x_max=200, y_min=50, y_max=150)

# Or using direct Polars operations
manual_filtered = events.filter(
    (pl.col("x") >= 100) & (pl.col("x") <= 200) &
    (pl.col("y") >= 50) & (pl.col("y") <= 150)
)

# Temporal analysis with Polars operations
rates = events.group_by_dynamic("timestamp", every="10ms").agg([
    pl.len().alias("event_rate"),
    pl.col("polarity").mean().alias("avg_polarity")
]).collect()

# Save processed data
processed = evf.preprocess_events("data.h5", t_start=0.1, t_end=0.5)
x, y, t, p = processed.select(["x", "y", "timestamp", "polarity"]).collect().to_numpy().T
evlib.save_events_to_hdf5(x, y, t, p, "output.h5")
```

### Performance Benchmarks

| Operation | NumPy | Polars | Speedup |
|-----------|-------|--------|---------|
| Loading 1M events | 0.08s | 0.05s | 1.6x |
| Filtering by polarity | 0.012s | 0.0001s | 97x |
| Spatial filtering | 0.045s | 0.002s | 23x |
| Group by polarity | 0.025s | 0.003s | 8x |
| Temporal binning | 0.156s | 0.008s | 19x |

*Benchmarks on Apple M1 with 16GB RAM*

## Performance Optimizations

### Memory Efficiency
- **Direct Polars Integration**: Zero-copy architecture with single-pass construction
- **Memory Usage**: ~110 bytes/event including overhead (previously ~200+ bytes/event)
- **Automatic Streaming**: Files >5M events automatically use chunked processing
- **Memory-Efficient Types**: Optimized data types (Int16 for x/y, Int8 for polarity)
- **Scalability**: Support for files up to 1B+ events without memory exhaustion

### Processing Speed
- **Load Speed**: 600k+ events/s for typical files (measured on real datasets)
- **Filter Speed**: 400M+ events/s using LazyFrame operations
- **Streaming Performance**: 1M+ events/s for large files (>100M events)
- **Format Support**: All formats (EVT2, HDF5, Text) optimized with format-specific encoding

### Scalability Features
- **LazyFrame Processing**: Memory-efficient operations without full materialization
- **Direct Polars Integration**: Zero-copy construction for optimal memory usage
- **Large File Support**: Tested with files up to 1.6GB (200M+ events)
- **Error Recovery**: Graceful handling of memory constraints and large files

### Benchmarking and Monitoring

Run performance benchmarks to verify optimizations:

```bash
# Verify README performance claims
python benches/benchmark_performance_readme.py

# Memory efficiency benchmark
python benches/benchmark_memory.py

# Test with your own data
python -c "
import evlib
import time
start = time.time()
events = evlib.load_events('your_file.h5')
df = events.collect()
print(f'Loaded {len(df):,} events in {time.time()-start:.2f}s')
print(f'Format: {evlib.detect_format("your_file.h5")}')
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
events_small = evlib.load_events("small_file.h5")
df_small = events_small.collect()

# Large files (>5M events) - Automatic streaming
events_large = evlib.load_events("large_file.h5")
# Same API, but uses streaming internally for memory efficiency

# Memory-efficient filtering on large datasets using filtering module
filtered = evf.filter_by_time(events_large, t_start=1.0, t_end=2.0)
positive_events = evf.filter_by_polarity(filtered, polarity=1)

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
events = evlib.load_events("data.h5")
df = events.collect()
peak_mem = monitor_memory()

print(f"Memory used: {peak_mem - initial_mem:.1f} MB")
print(f"Memory per event: {(peak_mem - initial_mem) * 1024 * 1024 / len(df):.1f} bytes")
print(f"Polars DataFrame size: {df.estimated_size() / 1024 / 1024:.1f} MB")
```

### Troubleshooting Large Files

#### Memory Constraints
- **Automatic Streaming**: Files >5M events use streaming by default (when implemented)
- **LazyFrame Operations**: Memory-efficient processing without full materialization
- **Memory Monitoring**: Use `benchmark_memory.py` to track usage
- **System Requirements**: Recommend 8GB+ RAM for files >100M events

#### Performance Tuning
- **Optimal Chunk Size**: System automatically calculates based on available memory
- **LazyFrame Operations**: Use `.lazy()` for complex filtering chains
- **Memory-Efficient Formats**: HDF5 generally most efficient, followed by EVT2
- **Progress Reporting**: Large files show progress during loading

#### Common Issues and Solutions

**Issue**: Out of memory errors
```python
# Solution: Use filtering before collecting and verify streaming is enabled
events = evlib.load_events("large_file.h5")
# Streaming activates automatically for files >5M events

# Apply filtering before collecting to reduce memory usage
import evlib.filtering as evf
filtered = evf.filter_by_time(events, t_start=0.1, t_end=0.5)
df = filtered.collect()  # Only collect when needed

# Or stream to disk using Polars
filtered.sink_parquet("filtered_events.parquet")
```

**Issue**: Slow loading performance
```python
# Solution: Use LazyFrame for complex operations and filtering module
events = evlib.load_events("file.h5")

# Use filtering module for optimized operations
import evlib.filtering as evf
result = evf.filter_by_roi(events, x_min=0, x_max=640, y_min=0, y_max=480)
df = result.collect()

# Or chain Polars operations
result = events.filter(pl.col("polarity") == 1).select(["x", "y", "timestamp"]).collect()
```

**Issue**: Memory usage higher than expected
```python
# Solution: Monitor and verify optimization
import evlib
events = evlib.load_events("file.h5")
df = events.collect()
print(f"Memory efficiency: {df.estimated_size() / len(df)} bytes/event")
print(f"DataFrame schema: {df.schema}")
print(f"Number of events: {len(df):,}")

# Check format detection
format_info = evlib.detect_format("file.h5")
print(f"Format: {format_info}")
```

### Performance Metrics Summary

| Metric | Previous | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Memory per event | ~200+ bytes | ~35 bytes | 80%+ reduction |
| Loading speed | ~300k events/s | 2.2M+ events/s | 7x+ improvement |
| Filter speed | ~50M events/s | 467M+ events/s | 9x+ improvement |
| Max file size | ~50M events | 200M+ events tested | 4x+ improvement |
| Memory efficiency | Variable | Consistent ~35 bytes/event | Predictable |

*Performance measured on Apple M1 with 16GB RAM using real-world datasets*

**Verified Performance Claims:**
- Loading speed: 2.2M events/s (exceeds 600k target)
- Filter speed: 467M events/s (exceeds 400M target)
- Memory efficiency: 35 bytes/event (well under 110 target)
- Large file support: Successfully tested with 200M+ events

## Available Python Modules

evlib provides several Python modules for different aspects of event processing:

### Core Modules
- **`evlib.formats`**: Direct Rust access for format loading and detection
- **`evlib.filtering`**: High-performance event filtering with Polars
- **`evlib.representations`**: Event representations (stacked histograms, voxel grids)
- **`evlib.models`**: Neural network model loading and inference

### Module Overview
```python
# Core event loading (returns Polars LazyFrame)
import evlib
events = evlib.load_events("data.h5")

# Format detection and description
format_info = evlib.detect_format("data.h5")
description = evlib.get_format_description("HDF5")

# Advanced filtering
import evlib.filtering as evf
filtered = evf.preprocess_events("data.h5", t_start=0.1, t_end=0.5)
time_filtered = evf.filter_by_time(events, t_start=0.1, t_end=0.5)

# Event representations
import evlib.representations as evr
hist = evr.create_stacked_histogram("data.h5", height=480, width=640, nbins=10)
voxel = evr.create_voxel_grid("data.h5", height=480, width=640, nbins=5)

# Neural network models (limited functionality)
from evlib.models import ModelConfig  # If available

# Direct Rust access (returns NumPy arrays)
x, y, t, p = evlib.formats.load_events("data.h5")

# Data saving
evlib.save_events_to_hdf5(x, y, t, p, "output.h5")
evlib.save_events_to_text(x, y, t, p, "output.txt")
```

## Examples

The `examples/` directory contains comprehensive notebooks demonstrating:

### Core Examples
- **H5 Data Processing**: Loading and processing HDF5 event data
- **eTram Dataset**: Working with sparse event distributions
- **Gen4 Data**: Processing modern event camera formats
- **Event Filtering**: Comprehensive filtering examples
- **Event Representations**: Creating stacked histograms and voxel grids
- **Performance Benchmarks**: Performance comparison with other tools

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

evlib includes comprehensive testing for both code and documentation:

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

**Documentation Testing Features:**
- **266 code examples** across 27 documentation files are automatically tested
- **Mock evlib module** allows testing even when full library isn't built
- **CI/CD integration** runs tests on every push and pull request
- **Comprehensive reporting** shows which examples work and which need attention

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
```bash
# Development build
maturin develop

# Build with features
maturin develop --features polars
maturin develop --features pytorch

# Release build
maturin build --release
```

### Build Requirements
- **Rust**: Stable toolchain (see `rust-toolchain.toml`)
- **Python**: ≥3.10 (3.12 recommended)
- **Maturin**: For building Python extensions

## Community & Support

- **GitHub**: [tallamjr/evlib](https://github.com/tallamjr/evlib)
- **Issues**: Report bugs and request features
- **Discussions**: Community Q&A and ideas

## License

MIT License - see [LICENSE.md](LICENSE.md) for details.
