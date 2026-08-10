# Installation

## Requirements

- **Python**: ≥ 3.11 (supported: 3.11, 3.12, 3.13; 3.12 recommended)
- **Operating System**: Linux, macOS, or Windows

Since 0.13.1, the PyPI wheels for macOS and Linux statically link HDF5, so `pip install evlib` reads HDF5 files (including Prophesee ECF-compressed data) through the Rust path with no system HDF5 install and no `hdf5plugin` needed. Windows wheels do not include HDF5; use `h5py` directly for HDF5 I/O there.

When building from source, HDF5 support is **opt-in**: EVT2/3, AEDAT, AER and text formats all work without it. Two Cargo features add HDF5 support on Linux and macOS: `--features hdf5-static` builds HDF5 from source, so no system HDF5 install is needed (this is what the release wheels use); `--features hdf5` links dynamically against a system HDF5 install instead. Windows source builds have neither feature available.

## System Dependencies

`pkg-config` is the only system dependency for a default build. System HDF5 libraries are needed only for a dynamically linked `--features hdf5` build; a `--features hdf5-static` build needs no system HDF5.

### Ubuntu/Debian
```bash
sudo apt update
sudo apt install pkg-config
# Only if building with --features hdf5 (dynamic linking):
sudo apt install libhdf5-dev
```

### macOS
```bash
brew install pkg-config
# Only if building with --features hdf5 (dynamic linking):
brew install hdf5
```

## Python Installation

### From PyPI (Recommended)
```bash
pip install evlib
```

### Development Installation
```bash
# Clone the repository
git clone https://github.com/tallamjr/evlib.git
cd evlib

# Install in development mode
pip install -e ".[dev]"
```

## Feature-Specific Installation

### Core Functionality Only
```bash
pip install evlib
```

### With Visualization Support
```bash
pip install evlib[plot]
```

### With PyTorch Integration
```bash
pip install evlib[torch]
```

### With Jupyter Notebook Support
```bash
pip install evlib[jupyter]
```

### Complete Installation
```bash
pip install evlib[all]
```

## Build from Source

### Prerequisites
- **Rust**: nightly toolchain (see [rustup.rs](https://rustup.rs/))
- **Maturin**: Python-Rust build tool

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install maturin
pip install maturin

# Clone and build
git clone https://github.com/tallamjr/evlib.git
cd evlib
maturin develop                           # default minimal build, no HDF5
maturin develop --features hdf5-static    # HDF5 built from source, no system HDF5 needed
maturin develop --features hdf5           # HDF5 dynamically linked against a system install
```

See the [main README](https://github.com/tallamjr/evlib#installation) for the Homebrew HDF5 2.x incompatibility warning and the conda-forge workaround for the dynamic `--features hdf5` build.

### GPU scatter-add kernels (optional)

The RVT preprocessing pipeline (`evlib.rvt.process_sequence`) can use native GPU scatter-add kernels:

- **CUDA** (NVIDIA, Linux-oriented): build the nvcc kernel and point `EVLIB_CUDA_LIB` at the resulting `librvt_scatter.so`, then call `process_sequence(..., backend="cuda")`.
- **Metal** (Apple Silicon): build with `CC=clang maturin develop --features metal`, then call `process_sequence(..., backend="metal")`.

Without these, the `polars` (CPU or cudf GPU) and `rust` (CPU) backends are always available.

## Verification

Test your installation:

```python
import evlib

# Test basic functionality
print(f"Available modules: {[m for m in dir(evlib) if not m.startswith('_')]}")

# Test with sample data
events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()
print(f"Loaded {len(df)} events successfully!")
```

## Troubleshooting

### Common Issues

#### HDF5 Library Not Found
```bash
# Error: HDF5 library not found
# Only relevant when building from source with the dynamic --features hdf5.
# Solution: install system HDF5 libraries (see above), or build with
# --features hdf5-static instead, which needs no system HDF5.
```

#### Import Error
```bash
# Error: ModuleNotFoundError: No module named 'evlib'
# Solution: Ensure proper installation and Python environment
pip install --upgrade evlib
```

#### Build Failures
```bash
# Error: maturin build failed
# Solution: Ensure Rust toolchain is installed
rustup update stable
```

### Performance Considerations

For optimal performance:

1. **Use Python 3.12**: Latest Python version with performance improvements
2. **Install NumPy optimized builds**: Use conda or optimized pip installations
3. **HDF5 optimization**: if building from source with `--features hdf5`, ensure the system HDF5 install is compiled with compression support

### Docker Installation

The PyPI wheel already includes statically linked HDF5, so no system HDF5 packages are needed:

```dockerfile
FROM python:3.12-slim

RUN pip install evlib[all]
```

## Next Steps

- [Quick Start Guide](quickstart.md)
- [Performance Guide](performance.md)
- [User Guide](../user-guide/loading-data.md)
