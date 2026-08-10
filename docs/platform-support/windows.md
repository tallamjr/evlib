# Windows Support Guide

## Overview

evlib provides comprehensive Windows support with most features fully functional. Windows wheels are built and published to PyPI, allowing simple installation via `pip install evlib`.

## Installation

### Standard Installation

```bash
pip install evlib
```

That's it! Windows wheels are pre-compiled and include all Python bindings, Polars integration, and event processing functionality.

### Optional Dependencies

```bash
# For plotting and visualisation
pip install evlib[plot]

# For Jupyter notebook support
pip install evlib[jupyter]

# For PyTorch model integration
pip install evlib[torch]

# All optional dependencies
pip install evlib[all]
```

## Feature Availability

### Fully Supported Features ✅

All core functionality works identically on Windows as on Linux/macOS:

- **Event Loading**: EVT2, EVT3, AEDAT (1.0-4.0), AER, text formats. `load_events` returns a Polars `LazyFrame`.
- **Event Processing**: Filtering via `evlib.filtering` (lazy Polars)
- **Polars DataFrames**: High-performance lazy DataFrame operations
- **Event Representations**: Voxel grids, event frames, time surfaces, stacked histograms
- **Visualisation**: Plotting via the Python `evlib.visualization` module
- **PyTorch Integration**: E2VID and RVT model loading and inference via the Python `evlib.models` module

GPU acceleration note: the CUDA scatter-add backend is Linux-oriented (NVIDIA) and the Metal backend is Apple Silicon only; neither applies to Windows. On Windows, the RVT pipeline and representations run on the CPU Polars and Rust backends.

### Platform-Specific Limitations ⚠️

#### HDF5 Support

**What's affected**: Native Rust HDF5 read and write functions (including `save_events_to_hdf5()`) are not available on Windows. On Linux and macOS, the PyPI wheels include statically linked HDF5 support since 0.13.1, and source builds can opt in with `--features hdf5-static` (built from source, no system HDF5 needed) or `--features hdf5` (dynamically linked against a system install); none of this is available on Windows.

**Why**: HDF5 system libraries are complex to build on Windows with the MSVC toolchain.

**Workaround**: Use pure Python h5py instead:

```python
import evlib
import h5py

# Load events normally (load_events returns a LazyFrame; collect to a DataFrame)
df = evlib.load_events("data.evt2").collect()

# Save using h5py
with h5py.File("output.h5", "w") as f:
    grp = f.create_group("events")
    grp.create_dataset("x", data=df["x"].to_numpy())
    grp.create_dataset("y", data=df["y"].to_numpy())
    grp.create_dataset("t", data=df["t"].dt.total_microseconds().to_numpy())
    grp.create_dataset("p", data=df["polarity"].to_numpy())
```

#### HDF5 Reading with ECF Codec

**What's affected**: Native Rust ECF decoder for Prophesee HDF5 files.

**Why**: ECF codec requires HDF5 system libraries.

**Workaround**: Use h5py with hdf5plugin (already included in dependencies):

```python
import h5py
import hdf5plugin  # Automatically registers ECF codec
import polars as pl

# Read Prophesee HDF5 file
with h5py.File("prophesee_data.h5", "r") as f:
    cd_events = f["CD"]["events"][:]

# Convert to Polars DataFrame
events = pl.DataFrame({
    "x": cd_events["x"],
    "y": cd_events["y"],
    "t": cd_events["t"],
    "p": cd_events["p"]
})
```

## What You DON'T Need to Install

### ❌ System HDF5 Libraries

Windows users do **not** need to install system HDF5 libraries or set environment variables. The pure Python h5py package handles all HDF5 operations.

### ❌ Visual Studio Build Tools (for pip install)

Pre-built Windows wheels mean you don't need Visual Studio or Rust toolchain for installation. Build tools are only needed if building from source.

### ❌ Manual Environment Variable Configuration

No `HDF5_DIR`, `PKG_CONFIG_PATH`, or other environment variables are required.

## Building from Source (Optional)

If you want to build evlib from source on Windows:

### Requirements

- **Rust**: Nightly toolchain
- **Python**: ≥3.11 (supported: 3.11, 3.12, 3.13)
- **Visual Studio**: 2019 or later with C++ build tools

### Build Commands

```powershell
# Install Rust (if not already installed)
# Download from https://rustup.rs/

# Set up virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install maturin
pip install maturin

# Build with default features (HDF5 is not available on Windows)
maturin develop --release

# Install in development mode
pip install -e .
```

## Common Workflows

### Loading Event Camera Data

```python
import evlib

# Works with all formats on Windows
events = evlib.load_events("data.evt2")  # Prophesee EVT2
events = evlib.load_events("data.aedat4")  # AEDAT 4.0
events = evlib.load_events("data.txt")  # Text format
```

### High-Performance Processing

```python
import evlib
import polars as pl

# Load as a Polars LazyFrame
events = evlib.load_events("data.evt2")

# Filter by time window. t is a Duration column, so compare on total_microseconds.
events_filtered = events.filter(
    pl.col("t").dt.total_microseconds().is_between(1_000_000, 2_000_000)
)

# Filter by spatial region
events_roi = events.filter(
    pl.col("x").is_between(100, 500) & pl.col("y").is_between(100, 500)
)
```

### Creating Event Representations

```python
import evlib
import evlib.representations as evr

events = evlib.load_events("data.evt2")

# Create voxel grid (works on Windows)
voxel_grid = evr.create_voxel_grid(
    events,
    height=480,
    width=640,
    n_time_bins=5,
)

# Create event frame
frame = evr.create_event_frame(
    events,
    height=480,
    width=640,
    n_time_bins=10,
)
```

## Performance Considerations

Windows builds perform comparably to Linux/macOS:

| Operation | Windows Performance | Notes |
|-----------|---------------------|-------|
| EVT2/3 Loading | Full speed | Native Rust implementation |
| Polars Filtering | Full speed | Lazy Polars |
| Voxel Grid Creation | Full speed | Polars (CPU) |
| AEDAT Parsing | Full speed | Format-agnostic |
| HDF5 Reading (h5py) | Slightly slower | Pure Python overhead |

## Troubleshooting

### Import Errors

If you encounter import errors:

```python
# Verify installation
import evlib
print(evlib.__version__)

# Check available functions
import evlib.formats
print(dir(evlib.formats))
```

### Missing HDF5 Save Function

This is expected on Windows. Use the h5py workaround shown above.

### Performance Issues

```python
# Ensure Polars is installed and being used
import polars as pl
import evlib

# load_events returns a Polars LazyFrame
events = evlib.load_events("data.evt2")
print(type(events))  # Should be polars.lazyframe.frame.LazyFrame
```

## Frequently Asked Questions

### Q: Can I use HDF5 files on Windows?

**A**: Yes! You can **read** HDF5 files using h5py (included in dependencies). You cannot use the native Rust `save_events_to_hdf5()` function, but h5py works for saving as well.

### Q: Do I need to install anything besides pip install evlib?

**A**: No! All required dependencies are included in the wheel.

### Q: Will HDF5 support be added in the future?

**A**: HDF5 save functionality may be added if:
1. HDF5 libraries become easier to build on Windows with MSVC
2. A pure Rust HDF5 implementation becomes mature enough
3. Sufficient user demand for native HDF5 save on Windows

For now, h5py provides excellent HDF5 support for Windows users.

### Q: Can I build with HDF5 support on Windows?

**A**: Technically possible but not recommended. It requires:
- Building HDF5 from source with MSVC
- Setting multiple environment variables
- Potential compatibility issues
- The h5py workaround is simpler and equally functional

## Examples

### Complete Workflow: EVT2 to HDF5

```python
import evlib
import h5py
import numpy as np

# Step 1: Load EVT2 file (native Rust, very fast). load_events returns a LazyFrame.
events = evlib.load_events("recording.evt2")

# Step 2: Process events using Polars (t is a Duration column)
import polars as pl

df = events.filter(
    pl.col("t").dt.total_microseconds().is_between(1_000_000, 2_000_000)  # Time window
).filter(
    pl.col("polarity") == 1  # Only positive polarity
).collect()

# Step 3: Save to HDF5 using h5py
with h5py.File("processed.h5", "w") as f:
    grp = f.create_group("events")

    # Convert Polars to NumPy for h5py
    grp.create_dataset("x", data=df["x"].to_numpy(), compression="gzip")
    grp.create_dataset("y", data=df["y"].to_numpy(), compression="gzip")
    grp.create_dataset("t", data=df["t"].dt.total_microseconds().to_numpy(), compression="gzip")
    grp.create_dataset("p", data=df["polarity"].to_numpy(), compression="gzip")

print(f"Saved {len(df)} events to processed.h5")
```

### Loading Prophesee HDF5 with ECF

```python
import h5py
import hdf5plugin
import polars as pl
import evlib

# The hdf5plugin automatically registers ECF codec
with h5py.File("prophesee_recording.h5", "r") as f:
    # Prophesee format: CD/events dataset
    cd_events = f["CD"]["events"][:]

# Convert to evlib's column layout (t as a Duration in microseconds, polarity column)
events = pl.DataFrame({
    "x": cd_events["x"].astype(np.int16),
    "y": cd_events["y"].astype(np.int16),
    "t": pl.duration(microseconds=cd_events["t"].astype(np.int64)),
    "polarity": cd_events["p"].astype(np.int8),
})

# Now use evlib for processing
import evlib.representations as evr
voxel_grid = evr.create_voxel_grid(events, height=720, width=1280, n_time_bins=5)
```

## Support

For Windows-specific issues:
- GitHub Issues: https://github.com/tallamjr/evlib/issues (label: `platform: windows`)
- Documentation: https://tallamjr.github.io/evlib/
- Examples: https://github.com/tallamjr/evlib/tree/master/examples

---

*evlib is fully committed to Windows support. While there are minor platform differences (HDF5 save), all core functionality works excellently on Windows.*
