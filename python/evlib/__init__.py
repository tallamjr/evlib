"""
evlib: Event Camera Data Processing Library

A robust event camera processing library with Python-first representations and Rust backend.

## Core Features

- **Universal Format Support**: Load data from H5, AEDAT, EVT2/3, AER, and text formats
- **Automatic Format Detection**: No need to specify format types manually
- **Polars DataFrame Support**: High-performance DataFrame operations
- **Pure Python Representations**: Efficient event-to-representation conversion with direct Polars API
- **Rust Performance**: Memory-safe, high-performance backend with Python bindings

## Quick Start

```python
import evlib
import polars as pl

# Load events as Polars LazyFrame
events = evlib.load_events("path/to/your/data.h5")

# Fast filtering using Polars expressions
filtered = events.filter(
    (pl.col("t").dt.total_microseconds() / 1_000_000 > 0.1) &
    (pl.col("t").dt.total_microseconds() / 1_000_000 < 0.2) &
    (pl.col("polarity") == 1)
)

# Or use Rust filtering functions directly
filtered = evlib.filtering.filter_by_time(events, t_start=0.1, t_end=0.2)
filtered = evlib.filtering.filter_by_polarity(filtered, polarity=1)

# Create representations
histogram = evlib.create_stacked_histogram(filtered, height=480, width=640)

# Direct access to Rust formats module (returns a Polars LazyFrame with
# columns x, y, t, polarity; call .collect() to materialise a DataFrame)
events = evlib.formats.load_events("path/to/your/data.h5")
```

## Available Modules

- `evlib.formats`: Data loading and format detection
- `evlib.filtering`: High-performance event filtering
- `evlib.representations`: Event-to-representation conversion
- `evlib.simulation`: Event camera simulation (ESIM algorithm for video-to-events)
- `evlib.visualization`: Event visualization tools
- `evlib.models`: Deep learning models (E2VID, RVT)
- `evlib.core`: Core data structures and utilities

"""

import os
import sys

# Import the compiled Rust extension module as a private submodule (evlib._evlib).
# Maturin builds it under the package as `evlib._evlib`, so the Python package
# __init__.py (this file) is the import entry point for `import evlib`.
try:
    from . import _evlib as _rust
except ImportError as e:
    raise ImportError(f"Failed to import evlib Rust module: {e}")

# Access Rust submodules from the compiled module
core = _rust.core
formats = _rust.formats

# Register Rust submodules in sys.modules so `import evlib.core` / `import evlib.formats`
# work with dot notation. Filtering is a pure-Python module, registered below.
sys.modules[__name__ + ".core"] = core
sys.modules[__name__ + ".formats"] = formats

# Expose the Rust dense-representation engine as `evlib.representations_rs` if present.
if hasattr(_rust, "representations_rs"):
    globals()["representations_rs"] = _rust.representations_rs
    sys.modules[__name__ + ".representations_rs"] = _rust.representations_rs

# Make key functions directly accessible.
# save_events_to_hdf5 handled below with fallback logic.
version = _rust.version  # Rust-provided build version function (evlib.version())
save_events_to_text = formats.save_events_to_text
detect_format = formats.detect_format
get_format_description = formats.get_format_description

# Configure Polars GPU acceleration if available
try:
    import polars as pl

    def _configure_polars_engine():
        """Configure Polars engine with GPU support and graceful fallback to streaming."""
        # Check if GPU is explicitly requested
        gpu_engine_requested = (
            os.environ.get("POLARS_ENGINE_AFFINITY", "").lower() == "gpu"
        )

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

# Import Python representations module (migration from Rust PyO3 to pure Python)
try:
    from . import representations
except ImportError:
    representations = None

# Import Python filtering module (migration from Rust PyO3 to pure Python)
try:
    from . import filtering as python_filtering
except ImportError:
    python_filtering = None

# Import optional Python-only submodules with graceful fallback
try:
    from . import models
except ImportError:
    models = None

try:
    from . import visualization
except ImportError:
    visualization = None

try:
    from . import simulation
except ImportError:
    simulation = None

# RVT preprocessing pipeline. This is the RVT-identical histogram path; the
# legacy representations.create_stacked_histogram computes a different quantity.
try:
    from . import rvt
except ImportError:
    rvt = None


# Make representation functions directly accessible for backwards compatibility
if representations:
    # Debug: Check if we're entering this block
    import os

    if os.environ.get("DEBUG_EVLIB"):
        print(f"DEBUG: representations is available: {representations}")
        print(
            f"DEBUG: representations has preprocess_for_detection: {hasattr(representations, 'preprocess_for_detection')}"
        )

    # Explicitly assign to module namespace
    globals()["create_stacked_histogram"] = representations.create_stacked_histogram
    globals()["create_mixed_density_stack"] = representations.create_mixed_density_stack
    globals()["create_voxel_grid"] = representations.create_voxel_grid
    globals()["preprocess_for_detection"] = representations.preprocess_for_detection
    globals()["benchmark_vs_rvt"] = representations.benchmark_vs_rvt

    # Register Python representations module in sys.modules
    sys.modules[__name__ + ".representations"] = representations

    if os.environ.get("DEBUG_EVLIB"):
        print(
            f"DEBUG: globals() now has preprocess_for_detection: {'preprocess_for_detection' in globals()}"
        )

# Register the RVT preprocessing package in sys.modules so it is reachable both
# as `evlib.rvt` and via `import evlib.rvt`.
if rvt is not None:
    sys.modules[__name__ + ".rvt"] = rvt

# Filtering is the pure-Python Polars implementation (the single implementation).
if python_filtering is None:
    raise ImportError("Failed to import evlib.filtering Python module")
filtering = python_filtering
sys.modules[__name__ + ".filtering"] = python_filtering

if os.environ.get("DEBUG_EVLIB"):
    print("DEBUG: Using Python filtering module")


try:
    from . import streaming_utils
except ImportError:
    streaming_utils = None

try:
    from . import pytorch
except ImportError:
    pytorch = None

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

                _cargo_toml_path = (
                    pathlib.Path(__file__).parent.parent.parent / "Cargo.toml"
                )
                with open(_cargo_toml_path, "r") as f:
                    content = f.read()
                version_match = re.search(
                    r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE
                )
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


def _save_events_to_hdf5_python(xs, ys, ts, ps, path):
    """
    Python fallback for HDF5 save using h5py.

    This function is used on Windows or when the Rust HDF5 feature is unavailable.

    Args:
        xs: NumPy array of x coordinates
        ys: NumPy array of y coordinates
        ts: NumPy array of timestamps
        ps: NumPy array of polarities
        path: Output HDF5 file path
    """
    try:
        import h5py
        import numpy as np
    except ImportError as e:
        raise ImportError(
            f"h5py is required for HDF5 save on this platform. Install with: pip install h5py\n"
            f"Original error: {e}"
        )

    # Validate array lengths
    n = len(ts)
    if len(xs) != n or len(ys) != n or len(ps) != n:
        raise ValueError("Arrays must have the same length")

    # Ensure arrays are NumPy arrays
    xs = np.asarray(xs, dtype=np.uint16)
    ys = np.asarray(ys, dtype=np.uint16)
    ts = np.asarray(ts, dtype=np.float64)
    ps = np.asarray(ps, dtype=np.int8)

    # Create HDF5 file and write datasets
    with h5py.File(path, "w") as f:
        grp = f.create_group("events")
        grp.create_dataset("xs", data=xs, compression="gzip", compression_opts=9)
        grp.create_dataset("ys", data=ys, compression="gzip", compression_opts=9)
        grp.create_dataset("ts", data=ts, compression="gzip", compression_opts=9)
        grp.create_dataset("ps", data=ps, compression="gzip", compression_opts=9)


def save_events_to_hdf5(xs, ys, ts, ps, path):
    """
    Save events to an HDF5 file.

    This function automatically uses the best available implementation:
    - Rust (hdf5-metno) on Linux/macOS with HDF5 feature enabled
    - Python (h5py) fallback on Windows or when Rust HDF5 is unavailable

    Args:
        xs: Array of x coordinates (NumPy array or compatible)
        ys: Array of y coordinates (NumPy array or compatible)
        ts: Array of timestamps (NumPy array or compatible)
        ps: Array of polarities (NumPy array or compatible)
        path: Output HDF5 file path
    """
    # Try Rust implementation first if available
    if hasattr(formats, "save_events_to_hdf5"):
        try:
            return formats.save_events_to_hdf5(xs, ys, ts, ps, path)
        except AttributeError:
            # Rust function not available, fall through to Python
            pass

    # Use Python fallback
    return _save_events_to_hdf5_python(xs, ys, ts, ps, path)


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
    print("HDF5 Plugin Diagnostic")
    print("=" * 50)

    # Check if hdf5plugin is available
    try:
        import hdf5plugin

        print("✓ hdf5plugin is installed")
        print(f"  Version: {hdf5plugin.version}")
        print(f"  Plugin path: {hdf5plugin.PLUGIN_PATH}")

        # Set up environment
        os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGIN_PATH
        if hasattr(hdf5plugin, "register"):
            hdf5plugin.register()
        print("✓ HDF5 plugins configured")

    except ImportError:
        print("✗ hdf5plugin not installed")
        print("  Fix: pip install hdf5plugin")
        return

    # Check h5py
    try:
        import h5py

        print(f"✓ h5py version: {h5py.version.version}")
    except ImportError:
        print("✗ h5py not installed")
        print("  Fix: pip install h5py")
        return

    # Test file if provided
    if file_path:
        try:
            with h5py.File(file_path, "r") as f:
                if "CD" in f and "events" in f["CD"]:
                    print(f"✓ Successfully opened Prophesee file: {file_path}")
                    print(f"  Events: {len(f['CD']['events']):,}")
                else:
                    print(f"✓ Opened HDF5 file (not Prophesee format): {file_path}")
        except Exception as e:
            print(f"✗ Cannot read file: {e}")

    print("\nFor Prophesee files, ensure:")
    print("  1. pip install hdf5plugin h5py")
    print("  2. Set HDF5_PLUGIN_PATH environment variable")
    print("  3. Use evlib.setup_hdf5_plugins() before loading")


def load_events(
    path,
    t_start=None,
    t_end=None,
    min_x=None,
    max_x=None,
    min_y=None,
    max_y=None,
    polarity=None,
    sort=True,
):
    """
    Load events as a Polars LazyFrame.

    The Rust loader performs the full decode and returns the complete frame;
    all load-time filters are applied here as Polars expressions so that the
    whole load+filter is a single GPU-collectable LazyFrame.

    Args:
        path: Path to event file.
        t_start: Inclusive lower time bound in seconds (optional).
        t_end: Inclusive upper time bound in seconds (optional).
        min_x: Inclusive lower x bound (optional).
        max_x: Inclusive upper x bound (optional).
        min_y: Inclusive lower y bound (optional).
        max_y: Inclusive upper y bound (optional).
        polarity: Keep only events with this polarity value (optional).
        sort: Sort by timestamp after filtering (default True).

    Returns:
        Polars LazyFrame with columns [x, y, t, polarity]
        - t is a Duration type in microseconds
        - polarity is already converted to -1/1

    Example:
        # Basic loading
        events = evlib.load_events("data.h5")

        # For validation, use the validation module explicitly:
        # from evlib.validation import quick_validate_events
        # is_valid = quick_validate_events(events)
    """
    import polars as pl

    # Full decode in Rust; no row filters passed to the readers.
    lf = formats.load_events(path)

    preds = []
    if t_start is not None:
        preds.append(pl.col("t").dt.total_microseconds() >= int(t_start * 1_000_000))
    if t_end is not None:
        preds.append(pl.col("t").dt.total_microseconds() <= int(t_end * 1_000_000))
    if min_x is not None:
        preds.append(pl.col("x") >= min_x)
    if max_x is not None:
        preds.append(pl.col("x") <= max_x)
    if min_y is not None:
        preds.append(pl.col("y") >= min_y)
    if max_y is not None:
        preds.append(pl.col("y") <= max_y)
    if polarity is not None:
        preds.append(pl.col("polarity") == polarity)

    if preds:
        from functools import reduce

        lf = lf.filter(reduce(lambda a, b: a & b, preds))

    if sort:
        lf = lf.sort("t")

    return lf


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
    __all__.append("filtering")
if visualization:
    __all__.append("visualization")


if streaming_utils:
    __all__.append("streaming_utils")
if pytorch:
    __all__.append("pytorch")
if simulation:
    __all__.append("simulation")
