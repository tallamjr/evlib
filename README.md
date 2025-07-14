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
- **Stacked Histogram Representations**: Efficient event-to-representation conversion
- **Real-time Data Processing**: Handle large datasets (550MB+ files) efficiently
- **Polarity Encoding**: Automatic conversion between 0/1 and -1/1 polarities
- **Rust Performance**: Memory-safe, high-performance backend with Python bindings

## Quick Start

```python
import evlib

# Load events from any supported format
events = evlib.load_events("path/to/your/data.h5")

# Create stacked histogram representation
histogram = evlib.create_event_histogram(events, height=480, width=640)

# Filter events by time range
filtered = evlib.filter_events_by_time(events, start_time=0.1, end_time=0.2)
```

## Installation

### Basic Installation
```bash
pip install evlib
```

### Development Installation
```bash
# Clone the repository
git clone https://github.com/tallamjr/evlib.git
cd evlib

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

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

## Examples

The `examples/` directory contains comprehensive notebooks demonstrating:

- **H5 Data Processing**: Loading and processing HDF5 event data
- **eTram Dataset**: Working with sparse event distributions
- **Gen4 Data**: Processing modern event camera formats
- **Data Visualization**: Creating event representations and plots

Run examples:
```bash
# Test all notebooks
pytest --nbmake examples/

# Run specific example
jupyter notebook examples/original_h5_data_exploration.ipynb
```

## Development

### Testing
```bash
# Run all tests
pytest
cargo test

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
