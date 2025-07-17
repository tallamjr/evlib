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

## Core Features

- **Universal Format Support**: Load data from H5, AEDAT, EVT2/3, AER, and text formats
- **Automatic Format Detection**: No need to specify format types manually
- **Polars DataFrame Integration**: High-performance DataFrame operations with up to 97x speedup
- **Stacked Histogram Representations**: Efficient event-to-representation conversion
- **Real-time Data Processing**: Handle large datasets (550MB+ files) efficiently
- **Polarity Encoding**: Automatic conversion between 0/1 and -1/1 polarities
- **Rust Performance**: Memory-safe, high-performance backend with Python bindings

## Quick Start

### NumPy Arrays (Traditional)
```python
import evlib

# Load events as NumPy arrays
x, y, t, p = evlib.load_events("path/to/your/data.h5")

# Create stacked histogram representation
histogram = evlib.create_event_histogram(x, y, t, p, height=480, width=640)

# Filter events by time range
filtered_x, filtered_y, filtered_t, filtered_p = evlib.filter_events_by_time(
    x, y, t, p, start_time=0.1, end_time=0.2
)
```

### Polars DataFrames (High-Performance)
```python
import evlib
import polars as pl

# Load events as Polars DataFrame
df = evlib.load_events_as_polars_dataframe("path/to/your/data.h5")

# Fast filtering and analysis with Polars
filtered = df.filter(
    (pl.col("timestamp") > 0.1) & 
    (pl.col("timestamp") < 0.2) &
    (pl.col("polarity") == 1)
)

# Advanced analysis with LazyFrames
stats = df.lazy().group_by("polarity").agg([
    pl.len().alias("count"),
    pl.col("x").mean().alias("mean_x"),
    pl.col("y").mean().alias("mean_y")
]).collect()
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
# Method 1: Direct DataFrame loading (recommended)
df = evlib.load_events_as_polars_dataframe("data.h5")

# Method 2: Enhanced load_events with format selection
data_dict = evlib.load_events("data.h5", output_format="polars")
df = pl.DataFrame(data_dict)

# Method 3: Unified interface
df = evlib.enhanced_load_events("data.h5", output_format="polars")
```

#### Advanced Features
```python
# DataFrame with metadata (event indices, time deltas)
df_meta = evlib.events_to_polars_dataframe_with_metadata(
    x, y, t, p, include_metadata=True
)

# Temporal analysis DataFrame
df_temporal = evlib.events_to_polars_temporal_dataframe(
    x, y, t, p, time_window_us=1000.0, spatial_bin_size=10
)

# LazyFrame for query optimization
lazy_df = evlib.create_events_lazyframe(data_dict)
result = lazy_df.filter(pl.col("polarity") == 1).collect()
```

#### Utility Functions
```python
from evlib.polars_utils import *

# Spatial filtering
filtered = filter_events_spatial(df, x_range=(100, 200), y_range=(50, 150))

# Temporal analysis
rates = event_rate_analysis(df, window_size_ms=5.0)

# Activity hotspots
hotspots = activity_hotspot_detection(df, grid_size=20, top_k=5)
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

## Examples

The `examples/` directory contains comprehensive notebooks demonstrating:

### Core Examples
- **H5 Data Processing**: Loading and processing HDF5 event data
- **eTram Dataset**: Working with sparse event distributions
- **Gen4 Data**: Processing modern event camera formats
- **Data Visualization**: Creating event representations and plots

### Polars-Specific Examples
- **`polars_integration_example.ipynb`**: Complete Polars API overview
- **`polars_utility_functions_demo.ipynb`**: Advanced utility functions
- **`polars_integration_demo.py`**: Python script demonstrating core features
- **`streaming_large_datasets_demo.ipynb`**: Memory-efficient processing

Run examples:
```bash
# Test all notebooks
pytest --nbmake examples/

# Run Polars-specific examples
jupyter notebook examples/polars_integration_example.ipynb
python examples/polars_integration_demo.py

# Test streaming with large datasets
jupyter notebook examples/streaming_large_datasets_demo.ipynb
```

## Development

### Testing
```bash
# Run all tests (includes Polars integration tests)
pytest
cargo test

# Test Polars functionality specifically
pytest tests/test_polars_integration_python.py
cargo test --features polars polars

# Test notebooks (including Polars examples)
pytest --nbmake examples/

# Test with coverage
pytest --cov=evlib

# Format code
black python/ tests/ examples/
cargo fmt
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
