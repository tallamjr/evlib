# Architecture

evlib architecture overview: a high-performance event camera processing library with Rust core and Python bindings.

## Design Philosophy

### Core Principles

1. **Rust Backend, Python Frontend**: Leverage Rust's performance and safety with Python's ease of use
2. **Zero-Copy Operations**: Minimize memory allocations and data copying
3. **Real Data Validation**: All features tested with real event camera datasets
4. **Production Ready**: Robust error handling and edge case management

### Performance Strategy

- **Complex algorithms in Rust**: Voxel grids, neural networks, file I/O
- **Simple operations in Python**: Basic array manipulations, plotting
- **Honest benchmarking**: Document real performance characteristics

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Python Frontend                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │   evlib.    │ │   evlib.    │ │   evlib.    │          │
│  │ formats     │ │ represent-  │ │ processing  │   ...    │
│  │             │ │ ations      │ │             │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
           │                    │                    │
           │                    │                    │
           ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│                     PyO3 Bindings                           │
└─────────────────────────────────────────────────────────────┘
           │                    │                    │
           │                    │                    │
           ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│                     Rust Core                               │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │ ev_formats  │ │ ev_repre-   │ │ ev_process- │   ...    │
│  │             │ │ sentations  │ │ ing         │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

### Module Hierarchy

```
evlib/
├── src/                           # Rust source code
│   ├── ev_core/                  # Core data structures
│   │   ├── mod.rs               # Event arrays, validation
│   │   └── types.rs             # Type definitions
│   ├── ev_formats/               # File I/O operations
│   │   ├── mod.rs               # Public API
│   │   ├── text.rs              # Text file loading
│   │   └── hdf5.rs              # HDF5 file operations
│   ├── ev_representations/       # Event representations
│   │   ├── mod.rs               # Public API
│   │   ├── voxel_grid.rs        # Standard voxel grids
│   │   └── smooth_voxel.rs      # Smooth voxel grids
│   ├── ev_processing/            # Neural networks
│   │   ├── mod.rs               # Public API
│   │   ├── e2vid.rs             # E2VID implementations
│   │   └── pytorch_loader.rs    # PyTorch model loading
│   ├── ev_transforms/            # Spatial transformations
│   │   ├── mod.rs               # Public API
│   │   ├── flip.rs              # Flip operations
│   │   └── noise.rs             # Noise addition
│   ├── ev_visualization/         # Visualization
│   │   ├── mod.rs               # Public API
│   │   ├── terminal.rs          # Terminal visualization
│   │   └── web_server.rs        # Web visualization
│   ├── ev_tracking/              # Event tracking
│   │   ├── mod.rs               # Public API
│   │   └── etap.rs              # ETAP integration
│   └── lib.rs                    # Python bindings
└── python/                       # Python package structure
    └── evlib/                    # Python module
        ├── __init__.py          # Package initialization
        ├── formats.py           # File format wrappers
        ├── representations.py   # Representation wrappers
        ├── processing.py        # Processing wrappers
        ├── augmentation.py      # Augmentation wrappers
        └── visualization.py     # Visualization wrappers
```

## Data Flow Architecture

### Event Data Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Raw       │    │   Parsed    │    │   Filtered  │    │   Processed │
│   Files     │───►│   Events    │───►│   Events    │───►│   Output    │
│             │    │             │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
      │                    │                    │                    │
      │                    │                    │                    │
      ▼                    ▼                    ▼                    ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Text/HDF5   │    │ Event       │    │ Time/Space  │    │ Voxel Grid  │
│ Files       │    │ Arrays      │    │ Windowing   │    │ Representa- │
│             │    │ (xs,ys,ts,ps│    │             │    │ tions       │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### Memory Management

```
┌─────────────────────────────────────────────────────────────┐
│                     Memory Layout                            │
│                                                             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │ Event Data  │ │ Voxel Grid  │ │ Model       │          │
│  │ (Stack)     │ │ (Heap)      │ │ Weights     │          │
│  │             │ │             │ │ (Heap)      │          │
│  │ xs: u16[]   │ │ f32[][][]   │ │ PyTorch     │          │
│  │ ys: u16[]   │ │             │ │ Model       │          │
│  │ ts: f64[]   │ │             │ │             │          │
│  │ ps: i8[]    │ │             │ │             │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### Event Data Structure

```rust
// src/ev_core/types.rs
#[derive(Debug, Clone)]
pub struct EventData {
    pub xs: Vec<u16>,      // X coordinates
    pub ys: Vec<u16>,      // Y coordinates
    pub ts: Vec<f64>,      // Timestamps
    pub ps: Vec<i8>,       // Polarities
}

impl EventData {
    pub fn new(xs: Vec<u16>, ys: Vec<u16>, ts: Vec<f64>, ps: Vec<i8>) -> Self {
        Self { xs, ys, ts, ps }
    }

    pub fn len(&self) -> usize {
        self.xs.len()
    }

    pub fn validate(&self) -> Result<(), EventError> {
        // Comprehensive validation
    }
}
```

### File I/O Architecture

```rust
// src/ev_formats/mod.rs
pub trait EventLoader {
    fn load_events(&self, path: &str, config: &LoadConfig) -> Result<EventData, FormatError>;
}

pub struct TextLoader;
pub struct HDF5Loader;

impl EventLoader for TextLoader {
    fn load_events(&self, path: &str, config: &LoadConfig) -> Result<EventData, FormatError> {
        // Text file parsing with filtering
    }
}

impl EventLoader for HDF5Loader {
    fn load_events(&self, path: &str, config: &LoadConfig) -> Result<EventData, FormatError> {
        // HDF5 file loading with filtering
    }
}
```

### Representation Architecture

```rust
// src/ev_representations/mod.rs
pub trait EventRepresentation {
    type Output;

    fn create(&self, events: &EventData, config: &RepresentationConfig) -> Self::Output;
}

pub struct VoxelGrid;
pub struct SmoothVoxelGrid;

impl EventRepresentation for VoxelGrid {
    type Output = Array3<f32>;

    fn create(&self, events: &EventData, config: &RepresentationConfig) -> Self::Output {
        // Voxel grid creation
    }
}
```

## PyO3 Integration

### Binding Architecture

```rust
// src/lib.rs
use pyo3::prelude::*;

#[pymodule]
fn evlib(_py: Python, m: &PyModule) -> PyResult<()> {
    // Core module
    m.add_function(wrap_pyfunction!(load_events, m)?)?;

    // Representations
    m.add_function(wrap_pyfunction!(create_voxel_grid, m)?)?;
    m.add_function(wrap_pyfunction!(create_smooth_voxel_grid, m)?)?;

    // Processing
    m.add_function(wrap_pyfunction!(events_to_video, m)?)?;

    Ok(())
}
```

### Type Conversion

```rust
// Convert Python types to Rust types
#[pyfunction]
fn load_events(
    file_path: String,
    t_start: Option<f64>,
    t_end: Option<f64>,
    // ... other parameters
) -> PyResult<(Vec<u16>, Vec<u16>, Vec<f64>, Vec<i8>)> {

    // Create load configuration
    let config = LoadConfig {
        t_start,
        t_end,
        ..Default::default()
    };

    // Load events
    let events = ev_formats::load_events(&file_path, &config)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;

    // Convert to Python types
    Ok((events.xs, events.ys, events.ts, events.ps))
}
```

## Neural Network Integration

### Model Loading Architecture

```rust
// src/ev_processing/mod.rs
pub enum ModelBackend {
    PyTorch(PyTorchModel),
    ONNX(OnnxModel),
}

pub struct PyTorchModel {
    model: tch::CModule,
}

pub struct OnnxModel {
    session: ort::Session,
}

impl ModelBackend {
    pub fn load(path: &str, backend: &str) -> Result<Self, ModelError> {
        match backend {
            "pytorch" => Ok(ModelBackend::PyTorch(PyTorchModel::load(path)?)),
            "onnx" => Ok(ModelBackend::ONNX(OnnxModel::load(path)?)),
            _ => Err(ModelError::UnsupportedBackend(backend.to_string())),
        }
    }

    pub fn predict(&self, input: &Array3<f32>) -> Result<Array3<f32>, ModelError> {
        match self {
            ModelBackend::PyTorch(model) => model.predict(input),
            ModelBackend::ONNX(model) => model.predict(input),
        }
    }
}
```

### E2VID Integration

```rust
// src/ev_processing/e2vid.rs
pub struct E2VIDModel {
    backend: ModelBackend,
    config: E2VIDConfig,
}

impl E2VIDModel {
    pub fn new(model_path: &str) -> Result<Self, ModelError> {
        let backend = ModelBackend::load(model_path, "pytorch")?;
        let config = E2VIDConfig::default();
        Ok(Self { backend, config })
    }

    pub fn events_to_video(&self, events: &EventData) -> Result<Array3<f32>, ModelError> {
        // Convert events to voxel grid
        let voxel_grid = VoxelGrid.create(events, &self.config.voxel_config);

        // Run inference
        self.backend.predict(&voxel_grid)
    }
}
```

## Performance Optimization

### Memory Layout

```rust
// Optimize for cache locality
#[repr(C)]
pub struct EventBatch {
    pub count: usize,
    pub xs: *mut u16,
    pub ys: *mut u16,
    pub ts: *mut f64,
    pub ps: *mut i8,
}

// SIMD operations for bulk processing
use std::simd::*;

pub fn process_events_simd(events: &[Event]) -> Vec<ProcessedEvent> {
    // Vectorized processing
}
```

### Parallel Processing

```rust
use rayon::prelude::*;

pub fn create_voxel_grid_parallel(events: &EventData, config: &VoxelConfig) -> Array3<f32> {
    // Parallel voxel grid creation
    events.par_chunks(1000)
        .map(|chunk| process_chunk(chunk, config))
        .reduce(|| Array3::zeros((config.bins, config.height, config.width)),
                |acc, chunk| acc + chunk)
}
```

## Error Handling

### Error Types

```rust
// src/ev_core/error.rs
#[derive(Debug, thiserror::Error)]
pub enum EventError {
    #[error("Invalid event data: {0}")]
    InvalidData(String),

    #[error("Array length mismatch: expected {expected}, got {actual}")]
    ArrayLengthMismatch { expected: usize, actual: usize },

    #[error("Timestamp order violation at index {index}")]
    TimestampOrderViolation { index: usize },
}

#[derive(Debug, thiserror::Error)]
pub enum FormatError {
    #[error("File not found: {0}")]
    FileNotFound(String),

    #[error("Invalid file format: {0}")]
    InvalidFormat(String),

    #[error("HDF5 error: {0}")]
    HDF5Error(#[from] hdf5::Error),
}
```

### Error Propagation

```rust
// Consistent error handling across modules
pub fn load_events(path: &str, config: &LoadConfig) -> Result<EventData, FormatError> {
    let raw_data = read_file(path)?;
    let events = parse_events(raw_data)?;
    let validated = events.validate()
        .map_err(|e| FormatError::InvalidFormat(e.to_string()))?;
    Ok(validated)
}
```

## Testing Architecture

### Test Organization

```
tests/
├── integration/           # End-to-end tests
│   ├── test_pipeline.py  # Full pipeline tests
│   └── test_models.py    # Model integration tests
├── unit/                 # Unit tests
│   ├── test_formats.py   # File I/O tests
│   ├── test_reprs.py     # Representation tests
│   └── test_core.py      # Core functionality tests
├── benchmarks/           # Performance benchmarks
│   ├── test_benchmarks.py
│   └── benchmark_utils.py
└── data/                 # Test data
    ├── slider_depth/     # Primary test dataset
    └── synthetic/        # Generated test data
```

### Test Data Management

```rust
// src/testing/mod.rs
pub struct TestDataset {
    pub name: String,
    pub events: EventData,
    pub metadata: DatasetMetadata,
}

impl TestDataset {
    pub fn slider_depth() -> Self {
        // Load standard test dataset
    }

    pub fn synthetic(config: &SyntheticConfig) -> Self {
        // Generate synthetic events
    }
}
```

## Build System

### Maturin Configuration

```toml
# pyproject.toml
[build-system]
requires = ["maturin>=1.3.2"]
build-backend = "maturin"

[project]
name = "evlib"
dynamic = ["version"]
dependencies = [
    "numpy>=1.24.0",
]

[tool.maturin]
bindings = "pyo3"
features = ["pyo3/extension-module"]
```

### Feature Flags

```toml
# Cargo.toml
[features]
default = ["hdf5"]
hdf5 = ["dep:hdf5"]
pytorch = ["dep:tch"]
cuda = ["tch/cuda"]
mkl = ["tch/mkl"]
```

## Documentation Architecture

### API Documentation

```rust
/// Load events from file with filtering options
///
/// # Arguments
///
/// * `file_path` - Path to event file
/// * `t_start` - Start time filter (optional)
/// * `t_end` - End time filter (optional)
///
/// # Returns
///
/// Event arrays (xs, ys, ts, ps)
///
/// # Errors
///
/// Returns `FormatError` if file cannot be loaded or parsed
///
/// # Examples
///
/// ```rust
/// let events = load_events("data/slider_depth/events.txt", Some(0.0), Some(1.0))?;
/// ```
pub fn load_events(
    file_path: &str,
    t_start: Option<f64>,
    t_end: Option<f64>
) -> Result<EventData, FormatError> {
    // Implementation
}
```

### Cross-Language Documentation

- **Rust docs**: `cargo doc --open`
- **Python docs**: Generated from docstrings
- **User guides**: Markdown documentation
- **Examples**: Jupyter notebooks

## Deployment Architecture

### Package Distribution

```
┌─────────────────────────────────────────────────────────────┐
│                    PyPI Distribution                         │
│                                                             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │ Linux       │ │ macOS       │ │ Windows     │          │
│  │ x86_64      │ │ x86_64      │ │ x86_64      │          │
│  │ aarch64     │ │ aarch64     │ │             │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

### CI/CD Pipeline

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]

jobs:
  test:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          pip install maturin
          maturin develop
      - name: Run tests
        run: pytest
```

## Future Architecture Considerations

### Planned Enhancements

1. **GPU Acceleration**: CUDA/OpenCL support for voxel grid creation
2. **Distributed Processing**: Multi-node event processing
3. **Real-time Streaming**: Live event camera integration
4. **Advanced Models**: Transformer-based architectures
5. **Language Bindings**: C++, Julia, R support

### Scalability

- **Memory management**: Streaming large datasets
- **Parallel processing**: Multi-GPU support
- **Cloud deployment**: Containerized processing
- **Edge computing**: Embedded device support

---

*This architecture balances performance, maintainability, and ease of use while providing a solid foundation for event-based vision applications.*
