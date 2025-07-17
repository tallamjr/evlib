"""
evlib: Event Camera Data Processing Library

A robust event camera processing library with Rust backend and Python bindings.

## Core Features

- **Universal Format Support**: Load data from H5, AEDAT, EVT2/3, AER, and text formats
- **Automatic Format Detection**: No need to specify format types manually
- **Polars DataFrame Support**: High-performance DataFrame operations
- **Stacked Histogram Representations**: Efficient event-to-representation conversion
- **Rust Performance**: Memory-safe, high-performance backend with Python bindings

## Quick Start

### Polars LazyFrames (High-Performance)
```python
import evlib
import polars as pl

# Load events as Polars LazyFrame
lf = evlib.load_events("path/to/your/data.h5")

# Fast filtering and analysis with Polars (lazy evaluation)
filtered = lf.filter(
    (pl.col("timestamp") > 0.1) &
    (pl.col("timestamp") < 0.2) &
    (pl.col("polarity") == 1)
)

# Collect to DataFrame when needed
df = filtered.collect()

# Direct access to Rust formats module if needed
# x, y, t, p = evlib.formats.load_events("path/to/your/data.h5")
```

### Direct Rust Access (Advanced)
```python
import evlib

# Direct access to Rust formats module (returns NumPy arrays)
x, y, t, p = evlib.formats.load_events("path/to/your/data.h5")

# Create stacked histogram representation
histogram = evlib.create_event_histogram(x, y, t, p, height=480, width=640)
```

## Available Functions

### Data Loading Functions
- `load_events()`: Load events as Polars LazyFrame (main function)
- `formats.load_events()`: Direct Rust access returning NumPy arrays (advanced)
- `detect_format()`: Automatic format detection
- `save_events_to_hdf5()`: Save events in HDF5 format
- `save_events_to_text()`: Save events as text

### High-Performance Representation Functions
- `create_stacked_histogram()`: Create stacked histogram representations (Polars-based)
- `create_mixed_density_stack()`: Create mixed density event stacks (Polars-based)
- `create_voxel_grid()`: Create voxel grid representations (Polars-based)
- `preprocess_for_detection()`: High-level API for neural network preprocessing
- `benchmark_vs_rvt()`: Performance comparison with PyTorch approaches

"""

# Import submodules (with graceful fallback)
try:
    from . import models  # noqa: F401

    _models_available = True
except ImportError:
    _models_available = False

try:
    from . import representations  # noqa: F401

    _representations_available = True

    # Import key representation functions directly
    from .representations import benchmark_vs_rvt  # noqa: F401
    from .representations import (
        create_mixed_density_stack,  # noqa: F401
        create_stacked_histogram,  # noqa: F401
        create_voxel_grid,  # noqa: F401
        preprocess_for_detection,  # noqa: F401
    )
except ImportError:
    _representations_available = False

# Import streaming utilities
try:
    from . import streaming_utils  # noqa: F401

    _streaming_utils_available = True
except ImportError:
    _streaming_utils_available = False

# Import high-performance Polars preprocessing (consolidated into representations module)
_representations_polars_available = False

# Import data reading functions from Rust module
try:
    from .evlib import formats

    _formats_available = True

    # Make data reading functions directly accessible
    try:
        save_events_to_hdf5 = formats.save_events_to_hdf5
        save_events_to_text = formats.save_events_to_text
        detect_format = formats.detect_format
        get_format_description = formats.get_format_description

        _polars_available = True

    except AttributeError:
        # Some functions might not be available in this build
        _formats_available = False
        _polars_available = False
        _polars_utils_available = False

except ImportError:
    _formats_available = False
    _polars_available = False
    _polars_utils_available = False

# Export the available functionality
__all__ = []

if _models_available:
    __all__.append("models")
if _streaming_utils_available:
    __all__.append("streaming_utils")
if _representations_available:
    __all__.extend(
        [
            "representations",
            "create_stacked_histogram",
            "create_mixed_density_stack",
            "create_voxel_grid",
            "preprocess_for_detection",
            "benchmark_vs_rvt",
        ]
    )
if _formats_available:
    format_exports = [
        "formats",
        "load_events",
        "save_events_to_hdf5",
        "save_events_to_text",
        "detect_format",
        "get_format_description",
    ]
    __all__.extend(format_exports)


# Main load_events function that returns a Polars LazyFrame
def load_events(path, **kwargs):
    """
    Load events as Polars LazyFrame.

    Args:
        path: Path to event file
        **kwargs: Additional arguments (t_start, t_end, min_x, max_x, min_y, max_y, polarity, sort, etc.)

    Returns:
        Polars LazyFrame with columns [x, y, timestamp, polarity]
        - timestamp is always converted to Duration type in microseconds
    """
    if not _formats_available:
        raise ImportError("Formats module not available")

    # Use unified load_events function (now returns Polars data directly)
    data_dict = formats.load_events(path, **kwargs)

    # Convert the dictionary to Polars LazyFrame
    import polars as pl

    # Handle the duration column properly
    if "timestamp" in data_dict:
        df = pl.DataFrame(data_dict)
        # The timestamp is already converted to microseconds in Rust
        df = df.with_columns([pl.col("timestamp").cast(pl.Duration(time_unit="us"))])
        return df.lazy()
    else:
        # Empty case
        return pl.DataFrame(data_dict).lazy()
