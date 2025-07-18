# evlib: Event Camera Utilities

<div align="center">
  <img src="https://raw.githubusercontent.com/tallamjr/evlib/master/evlogo.png" width="100" alt="evlib logo" />
</div>

**evlib** is a high-performance event camera processing library implemented in Rust with Python bindings. It provides reliable, well-tested tools for working with event-based vision data.

## Philosophy: Robust Over Rapid

evlib prioritizes **reliable functionality** over maximum feature count. Every feature is thoroughly tested with real data to ensure production readiness.

## Current Status

**Core Features Verified (January 2025)**

**Fully Working:**
- **Event Data I/O**: Universal format support (HDF5, EVT2/3, AEDAT, AER, Text)
- **Event Filtering**: Comprehensive filtering with Polars integration
- **Event Representations**: Stacked histograms, voxel grids, and mixed density stacks
- **Neural Networks**: E2VID model loading and inference (Python)
- **Performance**: High-performance DataFrame operations with Polars

**In Development:**
- **Advanced Processing**: Rust-based neural network processing (temporarily disabled)
- **Visualization**: Real-time terminal visualization (temporarily disabled)

## Quick Start

```python
import evlib
import evlib.filtering as evf
import evlib.representations as evr

# Load events as Polars LazyFrame
lf = evlib.load_events("events.h5")

# High-performance filtering
filtered = evf.filter_by_time(lf, t_start=0.1, t_end=0.5)
processed = evf.preprocess_events(
    "events.h5",
    t_start=0.1, t_end=0.5,
    roi=(100, 500, 100, 400),
    remove_hot_pixels=True
)

# Create event representations
hist = evr.create_stacked_histogram(
    "events.h5",
    height=480, width=640,
    nbins=10
)

# Direct Rust access (returns NumPy arrays)
x, y, t, p = evlib.formats.load_events("events.h5")
```

## Performance Philosophy

evlib provides **exceptional performance** through zero-copy architecture and Apache Arrow integration:

| Operation | Performance | Technology |
|-----------|-------------|------------|
| **Data Loading** | 2.18M events/s | Direct Polars construction via Arrow |
| **Memory Efficiency** | 35.8 bytes/event | Optimised data types + columnar layout |
| **Filtering** | 463M events/s | LazyFrame vectorised operations |
| **Large Files** | Automatic streaming | Memory-efficient chunk processing |

**Key Technologies:**
- **Apache Arrow**: Columnar memory format for zero-copy operations
- **Polars Integration**: High-performance DataFrame operations
- **Rust Backend**: Memory-safe, optimised implementations
- **Single-Pass Processing**: Eliminate intermediate data copies

See **[Zero-Copy Architecture](development/zero-copy-architecture.md)** for technical details.

## Key Features

### Comprehensive File Format Support
- **Universal formats**: HDF5, EVT2/3, AEDAT, AER, and text files
- **Automatic detection**: No need to specify format types manually
- **Advanced filtering**: Time windows, spatial bounds, polarity selection

### High-Performance Event Processing
- **Polars integration**: Up to 97x speedup for filtering operations
- **Event filtering**: Comprehensive filtering with temporal, spatial, and polarity options
- **Hot pixel removal**: Statistical outlier detection and removal
- **Noise filtering**: Refractory period and temporal noise removal

### Event Representations
- **Stacked histograms**: Efficient temporal binning for neural networks
- **Voxel grids**: Traditional quantized temporal representations
- **Mixed density stacks**: Logarithmic time binning with polarity accumulation
- **RVT replacement**: High-performance alternatives to PyTorch preprocessing

### Neural Network Integration
- **E2VID models**: Model loading and inference capabilities
- **ONNX support**: Runtime inference for deployment
- **Preprocessing pipelines**: Ready-to-use neural network input preparation

## Documentation Structure

- **[Getting Started](getting-started/installation.md)**: Installation and basic usage
- **[User Guide](user-guide/loading-data.md)**: Comprehensive tutorials
- **[API Reference](api/core.md)**: Detailed function documentation
  - **[Filtering API](api/filtering.md)**: High-performance event filtering
  - **[Representations API](api/representations.md)**: Event representations and preprocessing
  - **[Formats API](api/formats.md)**: File format support and detection
- **[Examples](examples/notebooks.md)**: Jupyter notebooks and scripts

## Quality Assurance

- **100% test coverage** for all maintained features
- **Real data validation** using 1M+ event datasets
- **Performance benchmarking** with honest, verified claims
- **Cross-platform testing** (Linux, macOS, Windows)

## Community

- **GitHub**: [tallamjr/evlib](https://github.com/tallamjr/evlib)
- **Documentation**: [evlib.readthedocs.io](https://evlib.readthedocs.io)
- **Issues**: Report bugs and request features on GitHub

---

*Built with care for the event-based vision community*
