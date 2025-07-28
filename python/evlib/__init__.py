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

# Or use high-level filtering functions
filtered = evlib.filter_by_time(lf, t_start=0.1, t_end=0.2)
filtered = evlib.filter_by_polarity(filtered, polarity=1)

# Complete preprocessing pipeline
processed = evlib.preprocess_events(
    "path/to/your/data.h5",
    t_start=0.1, t_end=0.5,
    roi=(100, 500, 100, 400),
    polarity=1,
    remove_hot_pixels=True,
    remove_noise=True
)

# Collect to DataFrame when needed
df = processed.collect()

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

### Event Filtering (New)
```python
import evlib

# Apply temporal and spatial filtering
filtered_events = evlib.filter_by_time("path/to/data.h5", t_start=0.1, t_end=1.0)
roi_events = evlib.filter_by_roi(filtered_events, x_min=100, x_max=500, y_min=100, y_max=400)

# Complete preprocessing pipeline
processed_events = evlib.preprocess_events(
    "path/to/data.h5",
    t_start=0.1, t_end=1.0,
    roi=(100, 500, 100, 400),
    polarity=1,
    remove_hot_pixels=True,
    remove_noise=True
)

# Use with representations
histogram = evlib.create_stacked_histogram(processed_events, height=480, width=640)
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

### Event Filtering Functions
- `filter_by_time()`: Filter events by time range (start/end times)
- `filter_by_roi()`: Filter events by spatial region of interest
- `filter_by_polarity()`: Filter events by polarity (positive/negative)
- `filter_hot_pixels()`: Remove hot pixels using statistical detection
- `filter_noise()`: Apply noise filtering (refractory period, etc.)
- `preprocess_events()`: Complete preprocessing pipeline with all filters

"""

import os

# Import the compiled Rust extension module
try:
    import importlib.util
    import glob

    # Find the compiled module file
    current_dir = os.path.dirname(__file__)
    so_files = glob.glob(os.path.join(current_dir, "evlib.cpython-*.so"))

    if so_files:
        spec = importlib.util.spec_from_file_location("evlib", so_files[0])
        rust_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rust_module)

        # CRITICAL FIX: Make this module appear as a package so Python allows submodule imports
        import sys

        current_module = sys.modules[__name__]
        if not hasattr(current_module, "__path__"):
            current_module.__path__ = [current_dir]

        # Access submodules from the compiled module
        core = rust_module.core
        formats = rust_module.formats
        representations = rust_module.representations

        # CRITICAL: Register submodules in sys.modules so they can be imported with dot notation
        sys.modules[__name__ + ".core"] = core
        sys.modules[__name__ + ".formats"] = formats
        sys.modules[__name__ + ".representations"] = representations

        # Make key functions directly accessible
        save_events_to_hdf5 = formats.save_events_to_hdf5
        save_events_to_text = formats.save_events_to_text
        detect_format = formats.detect_format
        get_format_description = formats.get_format_description
    else:
        raise ImportError("Compiled Rust module not found")

except ImportError as e:
    raise ImportError(f"Failed to import evlib Rust module: {e}")

# Configure Polars GPU acceleration if available
try:
    import polars as pl

    def _configure_polars_engine():
        """Configure Polars engine with GPU support and graceful fallback to streaming."""
        # Check if GPU is explicitly requested
        gpu_engine_requested = os.environ.get("POLARS_ENGINE_AFFINITY", "").lower() == "gpu"

        if gpu_engine_requested:
            try:
                # Try to set GPU engine if requested via environment variable
                pl.Config.set_engine_affinity("gpu")
                return "gpu"
            except Exception:
                pass

        # Auto-detect and try GPU engine if available (only if not explicitly requested)
        if not gpu_engine_requested:
            # Only enable GPU mode for NVIDIA CUDA GPUs
            import subprocess

            try:
                # Check if nvidia-smi is available (indicates NVIDIA GPU)
                subprocess.run(["nvidia-smi"], capture_output=True, check=True)

                # Test if GPU operations work with Polars
                test_df = pl.DataFrame({"test": [1, 2, 3]})
                pl.Config.set_engine_affinity("gpu")
                _ = test_df.select(pl.col("test") * 2)
                return "gpu"
            except (subprocess.CalledProcessError, FileNotFoundError, Exception):
                # NVIDIA GPU not available, set streaming engine
                pass

        # NVIDIA GPU not available, set streaming engine for optimal performance
        pl.Config.set_engine_affinity("streaming")
        return "streaming"

    # Configure the engine and store result
    _engine_type = _configure_polars_engine()
    _gpu_available = _engine_type == "gpu"

except ImportError:
    _gpu_available = False
    _engine_type = "streaming"

# Import optional Python-only submodules with graceful fallback
try:
    from . import models
except ImportError:
    models = None

# Import validation module
try:
    from . import validation
    from .validation import validate_events, quick_validate_events

    # Import validation_demo functions
    try:
        from .validation_demo import validate_loaded_events
    except ImportError:
        validate_loaded_events = None
except ImportError:
    validation = None
    validate_events = None
    quick_validate_events = None
    validate_loaded_events = None

# Import Python filtering module (not yet implemented in Rust)
try:
    from . import filtering
    from .filtering import (
        filter_by_time,
        filter_by_roi,
        filter_by_polarity,
        filter_hot_pixels,
        filter_noise,
        preprocess_events,
    )
except ImportError:
    filtering = None

# Use Rust representations submodule (no Python representations.py)
# Import individual functions from Rust submodules for convenience
# Import from Rust representations submodule (functions have _py suffix)
create_mixed_density_stack = representations.create_mixed_density_stack_py
create_stacked_histogram = representations.create_stacked_histogram_py
create_voxel_grid = representations.create_voxel_grid_py

# These were in the Python version but not yet in Rust
# preprocess_for_detection = representations.preprocess_for_detection
# benchmark_vs_rvt = representations.benchmark_vs_rvt

try:
    from . import hdf5_prophesee
except ImportError:
    hdf5_prophesee = None

try:
    from . import ecf_decoder
except ImportError:
    ecf_decoder = None

try:
    from . import hdf5_diagnostic
except ImportError:
    hdf5_diagnostic = None

try:
    from . import streaming_utils
except ImportError:
    streaming_utils = None

# Import version
try:
    __version__ = getattr(formats, "__version__", None)
    if not __version__:
        raise ImportError("Version not found in compiled module")
except ImportError:
    # Fallback to reading from Cargo.toml
    import pathlib

    try:
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                # Manual parsing fallback
                import re

                _cargo_toml_path = pathlib.Path(__file__).parent.parent.parent / "Cargo.toml"
                with open(_cargo_toml_path, "r") as f:
                    content = f.read()
                version_match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
                if version_match:
                    __version__ = version_match.group(1)
                else:
                    __version__ = "unknown"
                raise ImportError  # Skip the tomllib parsing below

        _cargo_toml_path = pathlib.Path(__file__).parent.parent.parent / "Cargo.toml"
        with open(_cargo_toml_path, "rb") as f:
            _cargo_data = tomllib.load(f)
        __version__ = _cargo_data["package"]["version"]
    except (FileNotFoundError, KeyError, AttributeError):
        __version__ = "unknown"


def get_recommended_engine():
    """
    Get the recommended Polars engine for evlib operations.

    Returns:
        str: 'gpu' if GPU is available, otherwise 'streaming' for large datasets
    """
    return _engine_type if _engine_type == "gpu" else "streaming"


def collect_with_optimal_engine(lazy_frame):
    """
    Collect a Polars LazyFrame using the optimal engine for evlib operations.

    Args:
        lazy_frame: Polars LazyFrame to collect

    Returns:
        Polars DataFrame
    """
    engine = get_recommended_engine()
    return lazy_frame.collect(engine=engine)


def setup_hdf5_plugins():
    """
    Set up HDF5 compression plugins for reading Prophesee files.

    Call this function before loading Prophesee HDF5 files if you encounter
    plugin-related errors.

    Returns:
        bool: True if setup was successful, False otherwise
    """
    try:
        import hdf5plugin

        # Set the environment variable
        os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGIN_PATH

        # Register plugins if available
        if hasattr(hdf5plugin, "register"):
            hdf5plugin.register()

        return True

    except ImportError:
        return False
    except Exception:
        return False


def diagnose_hdf5(file_path=None):
    """
    Diagnose HDF5 plugin setup and test a file if provided.

    Args:
        file_path: Optional path to Prophesee HDF5 file to test
    """
    try:
        from .hdf5_diagnostic import (
            diagnose_hdf5_plugins,
            setup_hdf5_plugins as diag_setup,
            test_prophesee_file,
            print_solutions,
        )

        plugins_ok = diagnose_hdf5_plugins()

        if plugins_ok:
            diag_setup()
            if file_path:
                test_prophesee_file(file_path)

        print_solutions()

    except ImportError:
        pass


def load_events(path, validate=None, **kwargs):
    """
    Load events as Polars LazyFrame with optional validation.

    Args:
        path: Path to event file
        validate: Optional validation mode:
            - None (default): No validation
            - 'quick': Fast validation for obvious errors
            - 'strict': Full validation with sensor-specific constraints
            - 'prophesee_gen4': Strict validation for Prophesee Gen4 (1280x720)
            - 'davis346': Strict validation for DAVIS346 (346x240)
        **kwargs: Additional arguments (t_start, t_end, min_x, max_x, min_y, max_y, polarity, sort, etc.)

    Returns:
        Polars LazyFrame with columns [x, y, timestamp, polarity]
        - timestamp is always converted to Duration type in microseconds

    Raises:
        ValueError: If validation fails and validate is specified

    Example:
        # Basic loading (no validation)
        events = evlib.load_events("data.h5")

        # With validation for data quality assurance
        events = evlib.load_events("data.h5", validate='prophesee_gen4')

        # Quick validation for garbage detection
        events = evlib.load_events("data.h5", validate='quick')
    """
    # Load data using Rust formats module
    data_dict = formats.load_events(path, **kwargs)

    # Convert the dictionary to Polars LazyFrame
    import polars as pl

    # Handle the duration column properly
    if "timestamp" in data_dict:
        # Define explicit schema to preserve efficient data types from Rust
        schema = {
            "x": pl.Int16,  # Coordinates: sufficient for event camera resolutions
            "y": pl.Int16,  # Coordinates: sufficient for event camera resolutions
            "timestamp": pl.Int64,  # Timestamp: needs full precision for microseconds
            "polarity": pl.Int8,  # Polarity: -1/1 fits in single byte
        }

        df = pl.DataFrame(data_dict, schema=schema)
        # The timestamp is already converted to microseconds in Rust
        df = df.with_columns([pl.col("timestamp").cast(pl.Duration(time_unit="us"))])
        lazy_df = df.lazy()

        # Apply validation if requested
        if validate and validation:
            # Convert timestamp back to float seconds for validation
            validation_df = lazy_df.with_columns(
                [pl.col("timestamp").dt.total_microseconds().cast(pl.Float64) / 1_000_000.0]
            )

            if validate == "quick":
                if not quick_validate_events(validation_df):
                    raise ValueError(
                        "Quick validation failed - data contains obvious errors (coordinates out of bounds, invalid polarities, etc.)"
                    )

            elif validate in ["strict", "prophesee_gen4", "davis346"]:
                sensor_type = "generic_large" if validate == "strict" else validate
                result = validate_events(validation_df, sensor_type=sensor_type, strict=True)

                if not result["valid"]:
                    error_msg = f"Validation failed for {sensor_type} sensor constraints:"
                    for error in result["errors"]:
                        error_msg += f"\n  - {error['type']}: {error['message'][:100]}..."
                    raise ValueError(error_msg)

        return lazy_df
    else:
        # Empty case - use same schema for consistency
        schema = {
            "x": pl.Int16,
            "y": pl.Int16,
            "timestamp": pl.Duration(time_unit="us"),
            "polarity": pl.Int8,
        }
        return pl.DataFrame(data_dict, schema=schema).lazy()


# Define exports
__all__ = [
    "__version__",
    "core",
    "formats",
    "load_events",
    "save_events_to_hdf5",
    "save_events_to_text",
    "detect_format",
    "get_format_description",
    "get_recommended_engine",
    "collect_with_optimal_engine",
    "setup_hdf5_plugins",
    "diagnose_hdf5",
]

# Add optional modules to exports if available
if models:
    __all__.append("models")
if validation:
    __all__.extend(["validation", "validate_events", "quick_validate_events", "validate_loaded_events"])
if representations:
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
if filtering:
    __all__.extend(
        [
            "filtering",
            "filter_by_time",
            "filter_by_roi",
            "filter_by_polarity",
            "filter_hot_pixels",
            "filter_noise",
            "preprocess_events",
        ]
    )
if streaming_utils:
    __all__.append("streaming_utils")
if hdf5_prophesee:
    __all__.append("hdf5_prophesee")
if ecf_decoder:
    __all__.append("ecf_decoder")
if hdf5_diagnostic:
    __all__.append("hdf5_diagnostic")
