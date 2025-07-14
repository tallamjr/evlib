# Installation

## Requirements

- **Python**: ≥ 3.10 (3.12 recommended)
- **Operating System**: Linux, macOS, or Windows
- **Dependencies**: HDF5 system libraries (for file I/O)

## System Dependencies

### Ubuntu/Debian
```bash
sudo apt update
sudo apt install libhdf5-dev pkg-config
```

### macOS
```bash
brew install hdf5 pkg-config
```

### Windows
```bash
# Using conda (recommended)
conda install -c conda-forge hdf5 pkg-config
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
maturin develop
```

## Verification

Test your installation:

```python
import evlib

# Test basic functionality
print(f"evlib version: {evlib.__version__}")
print(f"Available modules: {[m for m in dir(evlib) if not m.startswith('_')]}")

# Test with sample data
xs, ys, ts, ps = evlib.formats.load_events("data/slider_depth/events.txt")
print(f"Loaded {len(xs)} events successfully!")
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

- 📖 [Quick Start Guide](quickstart.md)
- 🎯 [Performance Guide](performance.md)
- 📚 [User Guide](../user-guide/loading-data.md)
