# evlib: Event Camera Utilities

<div align="center">
  <img src="https://raw.githubusercontent.com/tallamjr/evlib/master/docs/evlogo.png" width="100" alt="evlib logo" />
</div>

**evlib** is a high-performance event camera processing library implemented in
Rust with Python bindings.

**In Development:**

**Advanced Processing**: Additional data processing capabilities

## Quick Start

```python
import evlib
import evlib.filtering as evf
import evlib.representations as evr

# Load events as Polars LazyFrame - use actual file paths
events = evlib.load_events("data/slider_depth/events.txt")

# High-performance filtering and preprocessing
filtered = evf.preprocess_events(
    "data/slider_depth/events.txt",
    t_start=0.1, t_end=0.5,
    roi=(100, 500, 100, 400),
    remove_hot_pixels=True
)

# Create event representations
hist = evr.create_stacked_histogram(
    "data/slider_depth/events.txt",
    height=480, width=640,
    nbins=10
)

# Direct access to DataFrame columns
df = events.collect()
x, y, p = df['x'].to_numpy(), df['y'].to_numpy(), df['polarity'].to_numpy()
# Convert Duration timestamps to seconds (float64)
t = df['timestamp'].dt.total_seconds().to_numpy()
```

**Key Technologies:**
* **Apache Arrow**: Columnar memory format for zero-copy operations
* **Polars Integration**: High-performance DataFrame operations
* **Rust Backend**: Memory-safe, optimised implementations
* **Single-Pass Processing**: Eliminate intermediate data copies

See **[Zero-Copy Architecture](development/zero-copy-architecture.md)** for technical details.

## Documentation Structure

- **[Getting Started](getting-started/installation.md)**: Installation and basic usage
- **[User Guide](user-guide/loading-data.md)**: Comprehensive tutorials
- **[API Reference](api/core.md)**: Detailed function documentation
  - **[Filtering API](api/filtering.md)**: High-performance event filtering
  - **[Representations API](api/representations.md)**: Event representations and preprocessing
  - **[Formats API](api/formats.md)**: File format support and detection
- **[Examples](examples/notebooks.md)**: Jupyter notebooks and scripts

## Community

- **GitHub**: [tallamjr/evlib](https://github.com/tallamjr/evlib)
- **Documentation**: [evlib.readthedocs.io](https://evlib.readthedocs.io)
- **Issues**: Report bugs and request features on GitHub
