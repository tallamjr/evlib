# Contributing to evlib

Thank you for your interest in contributing to evlib! This guide will help you get started with development.

## Philosophy

evlib follows a **"robust over rapid"** philosophy:
- **Quality over quantity**: Every feature must be thoroughly tested
- **Real data validation**: No mock data or placeholder implementations
- **Honest performance**: Claims must be verified with benchmarks
- **Production ready**: Code must handle edge cases and errors gracefully

## Development Setup

### Prerequisites

- **Python**: ≥ 3.10 (3.12 recommended)
- **Rust**: Stable toolchain (see [rustup.rs](https://rustup.rs/))
- **System dependencies**: HDF5, pkg-config

### Installation

```bash
# Clone the repository
git clone https://github.com/tallamjr/evlib.git
cd evlib

# Create development environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install Rust if not already installed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install maturin for building
pip install maturin

# Build the project
maturin develop
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## Development Workflow

### Building and Testing

```bash
# Development build (fast iteration)
maturin develop

# Release build (for performance testing)
maturin develop --release

# Run all tests
pytest
cargo test

# Run specific test file
pytest tests/test_evlib.py
cargo test --test test_smooth_voxel

# Run with coverage
pytest tests/ --cov=evlib --cov-report=xml

# Test notebooks
pytest --nbmake examples/
```

### Code Quality

```bash
# Format Python code
black python/ tests/ examples/

# Format Rust code
cargo fmt

# Lint Rust code
cargo clippy -- -D warnings

# Type checking
cargo check
```

## Contribution Guidelines

### Code Standards

**Python Code:**
- Format with Black (line length: 110)
- Use descriptive variable names
- Add type hints where appropriate
- Follow PEP 8 conventions

**Rust Code:**
- Format with rustfmt (automatic via IDE)
- Use descriptive variable names
- Add comprehensive documentation
- Follow Rust conventions

### Testing Requirements

**New Features Must Include:**
1. **Unit tests** for all functions
2. **Integration tests** with real data
3. **Performance benchmarks** vs existing implementations
4. **Documentation** with examples

**Test Coverage:**
- Target: >80% test coverage
- All maintained features: 100% coverage
- Tests must use real data from `data/` directory

### Documentation

**Required for New Features:**
- **API documentation**: Docstrings for all public functions
- **User guide**: How to use the feature
- **Examples**: Jupyter notebook demonstrating usage
- **Benchmarks**: Performance comparison where relevant

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add smooth voxel grid implementation

- Implement bilinear interpolation for temporal smoothing
- Add comprehensive tests with slider_depth dataset
- Benchmark shows 2.5x speedup vs pure Python
- Add user guide documentation

References:
https://github.com/uzh-rpg/rpg_e2vid
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `perf`: Performance improvement
- `test`: Add or update tests
- `docs`: Documentation changes
- `refactor`: Code refactoring
- `style`: Code style changes

## Architecture Overview

### Project Structure

```
evlib/
├── src/                    # Rust source code
│   ├── ev_core/           # Core data structures
│   ├── ev_formats/        # File I/O (HDF5, text)
│   ├── ev_representations/ # Voxel grids, smooth voxel
│   ├── ev_processing/     # Neural networks, E2VID
│   ├── ev_transforms/     # Spatial transformations
│   ├── ev_visualization/  # Plotting utilities
│   └── lib.rs            # Python bindings
├── tests/                 # Python tests
├── examples/              # Jupyter notebooks
├── data/                  # Test datasets
└── docs/                  # Documentation
```

### Module Mapping

```
Rust Module              → Python Module
src/ev_core/            → evlib.core
src/ev_formats/         → evlib.formats
src/ev_representations/ → evlib.representations
src/ev_processing/      → evlib.processing
src/ev_transforms/      → evlib.augmentation
src/ev_visualization/   → evlib.visualization
```

## Adding New Features

### Step 1: Design and Planning

1. **Create an issue** describing the feature
2. **Research existing implementations** (academic papers, other libraries)
3. **Design API** with clear function signatures
4. **Plan tests** with specific datasets

### Step 2: Implementation

**Rust Implementation:**
```rust
// src/ev_representations/my_feature.rs
use ndarray::Array3;
use crate::ev_core::EventData;

/// Create custom representation from events
pub fn create_custom_representation(
    events: &EventData,
    width: usize,
    height: usize,
    parameter: f64,
) -> Array3<f32> {
    // Implementation here
}
```

**Python Bindings:**
```rust
// src/lib.rs
#[pyfunction]
fn create_custom_representation(
    xs: Vec<u16>,
    ys: Vec<u16>,
    ts: Vec<f64>,
    ps: Vec<i8>,
    width: usize,
    height: usize,
    parameter: f64,
) -> PyResult<Vec<Vec<Vec<f32>>>> {
    // Convert to Rust types and call implementation
}
```

### Step 3: Testing

**Unit Tests (Rust):**
```rust
// src/ev_representations/mod.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_custom_representation() {
        // Test with known data
        let events = create_test_events();
        let result = create_custom_representation(&events, 640, 480, 1.0);
        assert_eq!(result.shape(), &[640, 480, 5]);
    }
}
```

**Integration Tests (Python):**
```python
# tests/test_custom_representation.py
import evlib
import numpy as np

def test_custom_representation():
    # Load real data
    xs, ys, ts, ps = evlib.formats.load_events("data/slider_depth/events.txt")

    # Test function
    result = evlib.representations.create_custom_representation(
        xs, ys, ts, ps, 640, 480, 1.0
    )

    # Validate output
    assert result.shape == (640, 480, 5)
    assert result.dtype == np.float32

def test_custom_representation_benchmark():
    # Benchmark vs pure Python implementation
    pass
```

### Step 4: Documentation

**API Documentation:**
```python
def create_custom_representation(xs, ys, ts, ps, width, height, parameter):
    """
    Create custom event representation.

    Parameters
    ----------
    xs : np.ndarray
        X coordinates of events
    ys : np.ndarray
        Y coordinates of events
    ts : np.ndarray
        Timestamps of events
    ps : np.ndarray
        Polarities of events
    width : int
        Image width
    height : int
        Image height
    parameter : float
        Custom parameter

    Returns
    -------
    np.ndarray
        Custom representation with shape (width, height, 5)

    Examples
    --------
    >>> xs, ys, ts, ps = evlib.formats.load_events("data/slider_depth/events.txt")
    >>> custom = evlib.representations.create_custom_representation(
    ...     xs, ys, ts, ps, 640, 480, 1.0
    ... )
    >>> custom.shape
    (640, 480, 5)
    """
```

## Performance Guidelines

### Benchmarking

All new features must include benchmarks:

```python
# tests/test_benchmarks.py
import time
import evlib

def benchmark_custom_representation():
    xs, ys, ts, ps = evlib.formats.load_events("data/slider_depth/events.txt")

    # Benchmark evlib
    start = time.time()
    result_evlib = evlib.representations.create_custom_representation(
        xs, ys, ts, ps, 640, 480, 1.0
    )
    evlib_time = time.time() - start

    # Benchmark pure Python
    start = time.time()
    result_python = create_custom_representation_python(xs, ys, ts, ps, 640, 480, 1.0)
    python_time = time.time() - start

    print(f"evlib: {evlib_time:.3f}s")
    print(f"Python: {python_time:.3f}s")
    print(f"Speedup: {python_time/evlib_time:.2f}x")

    # Validate results are equivalent
    np.testing.assert_allclose(result_evlib, result_python, rtol=1e-6)
```

### Memory Profiling

```python
# Use memory_profiler for detailed analysis
from memory_profiler import profile

@profile
def test_memory_usage():
    xs, ys, ts, ps = evlib.formats.load_events("data/slider_depth/events.txt")
    result = evlib.representations.create_custom_representation(
        xs, ys, ts, ps, 640, 480, 1.0
    )
    return result
```

## Data Requirements

### Test Data

- **Primary dataset**: `data/slider_depth/events.txt` (1M+ events)
- **Validation**: All tests must use real event data
- **No mock data**: Avoid synthetic or placeholder data

### Adding New Test Data

1. **Add to `data/` directory**
2. **Include metadata**: Camera calibration, scene description
3. **Document format**: Column order, coordinate system
4. **Add to `.gitignore`** if file is large (>10MB)

## Neural Network Models

### Model Requirements

- **Working implementation**: Must have verified download URLs
- **Test data**: Include sample outputs for validation
- **Performance metrics**: Document inference time and accuracy
- **Multiple backends**: Support both PyTorch and ONNX where possible

### Model Integration

```rust
// src/ev_processing/models/custom_model.rs
use crate::ev_core::EventData;

pub struct CustomModel {
    // Model state
}

impl CustomModel {
    pub fn new(model_path: &str) -> Result<Self, ModelError> {
        // Load model
    }

    pub fn predict(&self, events: &EventData) -> Array3<f32> {
        // Inference
    }
}
```

## Testing Strategy

### Test Categories

1. **Unit Tests**: Individual function correctness
2. **Integration Tests**: End-to-end workflows
3. **Performance Tests**: Benchmarks vs baselines
4. **Notebook Tests**: Jupyter notebook execution
5. **Cross-platform Tests**: Linux, macOS, Windows

### Continuous Integration

Tests run automatically on:
- Pull requests
- Main branch commits
- Release tags

### Manual Testing

Before submitting:
```bash
# Run all tests
pytest
cargo test

# Test notebooks
pytest --nbmake examples/

# Check formatting
black --check python/ tests/ examples/
cargo fmt --check

# Run linting
cargo clippy -- -D warnings
```

## Release Process

### Version Numbering

Follow [Semantic Versioning](https://semver.org/):
- `MAJOR.MINOR.PATCH`
- Breaking changes: increment MAJOR
- New features: increment MINOR
- Bug fixes: increment PATCH

### Release Checklist

1. **Update version** in `Cargo.toml` and `pyproject.toml`
2. **Update CHANGELOG.md** with release notes
3. **Run full test suite** including benchmarks
4. **Build release wheels** with `maturin build --release`
5. **Test installation** from built wheels
6. **Create release tag** and GitHub release
7. **Publish to PyPI** (maintainers only)

## Getting Help

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: General questions and ideas
- **Documentation**: Comprehensive guides and API reference
- **Examples**: Jupyter notebooks demonstrating usage

## Code of Conduct

We follow a professional, inclusive environment:
- Be respectful and constructive
- Focus on the code, not the person
- Help newcomers learn and contribute
- Prioritize project goals over personal preferences

---

*Thank you for contributing to evlib! Your efforts help make event-based vision more accessible to everyone.*
