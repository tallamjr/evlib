# Installation

## Requirements

- **Python**: ≥ 3.11 (supported: 3.11, 3.12, 3.13; 3.12 recommended)
- **Operating System**: Linux, macOS, or Windows

HDF5 is **opt-in**: EVT2/3, AEDAT, AER and text formats all work without it. You only need HDF5 system libraries if you build the Rust HDF5 reader with `--features hdf5` (Linux and macOS only). On Windows, use `h5py` directly for HDF5 I/O.

## System Dependencies

`pkg-config` is the only system dependency for a default build. HDF5 system libraries are needed only for an opt-in `--features hdf5` build.

### Ubuntu/Debian
```bash
sudo apt update
sudo apt install pkg-config
# Only if building with --features hdf5:
sudo apt install libhdf5-dev
```

### macOS
```bash
brew install pkg-config
# Only if building with --features hdf5:
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
- **Rust**: Stable toolchain (see [rustup.rs](https://rustup.rs/))
- **Maturin**: Python-Rust build tool

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install maturin
pip install maturin

# Clone and build
git clone https://github.com/tallamjr/evlib.git
cd evlib
maturin develop                    # default minimal build
maturin develop --features hdf5    # opt-in HDF5 support (Linux/macOS)
```

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
# Solution: Install system HDF5 libraries (see above)
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
3. **HDF5 optimization**: Ensure HDF5 is compiled with compression support

### Docker Installation

```dockerfile
FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    pkg-config \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install evlib[all]
```

## Next Steps

- [Quick Start Guide](quickstart.md)
- [Performance Guide](performance.md)
- [User Guide](../user-guide/loading-data.md)
