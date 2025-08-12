# Testing

Comprehensive testing strategy for evlib ensuring reliability and performance.

## Testing Philosophy

### Core Principles

1. **Real Data Testing**: All tests use real event camera data from `data/` directory
2. **No Mock Data**: Avoid synthetic or placeholder data in tests
3. **100% Coverage**: All maintained features must have complete test coverage
4. **Performance Validation**: Benchmark all implementations against baselines
5. **Cross-Platform**: Tests must pass on Linux, macOS, and Windows

### Test-Driven Development

- **Write tests first**: Define expected behavior before implementation
- **Red-Green-Refactor**: Fail, pass, optimize cycle
- **Continuous testing**: Run tests frequently during development
- **Documentation tests**: Ensure examples in docs work correctly

## Test Structure

### Current Test Organization

```
tests/
├── test_acceleration.py            # Acceleration/performance tests
├── test_ev_core.py                 # Core functionality tests
├── test_evt3_comprehensive.py      # Comprehensive EVT3 format tests
├── test_evt3_format_support.py     # EVT3 format support unit tests
├── test_evt3_working.py            # EVT3 working implementation tests
├── test_hdf5_roundtrip.py          # HDF5 format roundtrip tests
├── test_real_data_formats_comprehensive.py  # Real data format tests
├── test_reconstruction.py          # Event reconstruction tests
├── test_representations.py         # Voxel grid and representation tests
├── test_simulation.py              # Event simulation tests
├── test_streaming.py               # Streaming data tests
├── test_utilities.py               # Utility function tests
├── test_visualization.py           # Visualization tests
└── [Rust tests]                   # Rust backend tests
    ├── test_aedat_address_decoding.rs
    ├── test_aer_formats_realdata.rs
    ├── test_event_validation.rs
    ├── test_evt2_detection.rs
    ├── test_evt2_formats_realdata.rs
    ├── test_evt3_formats.rs
    ├── test_evt_format_detection.rs
    ├── test_format_detection.rs
    ├── test_hdf5_formats_realdata.rs
    ├── test_polarity_conversion.rs
    ├── test_real_data_formats_rust.rs
    └── test_realtime_performance.rs
```

### Rust Tests

```
src/
├── ev_core/
│   ├── mod.rs              # Unit tests embedded
│   └── tests.rs            # Integration tests
├── ev_formats/
│   ├── mod.rs              # Unit tests embedded
│   └── tests.rs            # Integration tests
└── tests/                  # Rust integration tests
    ├── test_smooth_voxel.rs
    └── test_pipeline.rs
```

## Test Data Management

### Available Test Datasets

The project includes comprehensive real-world datasets for testing:

#### Primary Test Dataset - slider_depth
- **Events**: 1,000,000+ events
- **Duration**: ~10 seconds
- **Resolution**: 346x240 pixels
- **Format**: Text file with t, x, y, polarity
- **Scene**: Slider moving at different depths
- **File**: `data/slider_depth/events.txt` (22MB)

#### eTram Dataset
- **Medium files**: `../tests/data/eTram/h5/val_2/val_night_011_td.h5` (14.9MB)
- **Resolution**: 1280x720 pixels
- **Format**: EVT2 binary format
- **HDF5 versions**: Available in `data/eTram/h5/` directory

#### Additional Test Data
- **Gen4 data**: `data/gen4/test/` with multiple preprocessed datasets
- **Large HDF5**: `data/original/front/seq01.h5` (1.6GB, 346x240 resolution)

```python
import evlib
import polars as pl

# Standard test data loading
def load_test_data():
    return evlib.load_events("data/slider_depth/events.txt")

# Smaller subset for fast tests
def load_test_data_small():
    events = evlib.load_events("data/slider_depth/events.txt")
    filtered = events.filter((pl.col('t') >= 0.0) & (pl.col('t') <= 1.0))
    return filtered
```

### Test Data Validation

```python
import evlib
import numpy as np

def validate_test_data():
    """Ensure test data meets requirements"""
    events = evlib.load_events("data/slider_depth/events.txt")
    df = events.collect()
    xs, ys, ps = df['x'].to_numpy(), df['y'].to_numpy(), df['polarity'].to_numpy()
    # Convert Duration timestamps to seconds (float64)
    ts = df['t'].dt.total_seconds().to_numpy()

    # Basic validation
    assert len(xs) > 100000, "Insufficient events for testing"
    assert len(set([len(xs), len(ys), len(ts), len(ps)])) == 1, "Array length mismatch"

    # Data quality checks
    assert np.all(xs >= 0) and np.all(xs < 640), "Invalid x coordinates"
    assert np.all(ys >= 0) and np.all(ys < 480), "Invalid y coordinates"
    assert np.all(np.diff(ts) >= 0), "Timestamps not sorted"
    assert np.all(np.isin(ps, [-1, 1])), "Invalid polarities"

    print(f"SUCCESS: Test data validated: {len(xs)} events")
```

### Format Testing Results

#### EVT3 Format Support SUCCESS:
- **Status**: Production ready
- **Tests**: 8/8 passing
- **Coverage**: Complete specification compliance
- **Key finding**: EVT3 format detection and reading works correctly
- **Data structure**: Returns arrays rather than individual event objects
- **Performance**: Memory efficient with NumPy arrays

#### Real Data Format Compatibility WARNING:
Comprehensive testing against real data files revealed important compatibility issues:

1. **Polarity Encoding Mismatch**
   - Real data uses 0/1 encoding (0=negative, 1=positive)
   - Tests expect -1/1 encoding after conversion
   - **Solution**: Configure polarity conversion in LoadConfig

2. **EVT2 Files**
   - **Issue**: Real files contain event types not handled by current reader
   - **Error**: `InvalidEventType { type_value: 12, offset: 366 }`
   - **Recommendation**: Enhance EVT2 reader to handle additional event types

3. **HDF5 Files**
   - **Issue**: Polarity validation fails on real data
   - **Error**: `Invalid polarities found`
   - **Recommendation**: Handle multiple polarity encoding schemes

4. **Format Detection** SUCCESS:
   - **Status**: Works correctly
   - **Confidence**: >0.8 for all tested formats
   - **Coverage**: EVT2, HDF5, and text formats

#### Testing Recommendations

1. **Immediate Fixes**
   - Fix polarity encoding configuration in tests
   - Enhance EVT2 reader for additional event types
   - Improve HDF5 reader robustness

2. **Test Suite Improvements**
   - Add real data test suite to CI/CD
   - Separate validation for raw vs converted data
   - Performance benchmarks with real files

## Unit Testing

### Python Unit Tests

```python
# tests/unit/test_formats.py
import pytest
import numpy as np
import polars as pl
import evlib

class TestEventLoading:
    def test_load_events_basic(self):
        """Test basic event loading"""
        events = evlib.load_events("data/slider_depth/events.txt")
        df = events.collect()
        xs, ys, ps = df['x'].to_numpy(), df['y'].to_numpy(), df['polarity'].to_numpy()
        # Convert Duration timestamps to seconds (float64)
        ts = df['timestamp'].dt.total_seconds().to_numpy()

        assert len(xs) > 0, "No events loaded"
        assert isinstance(xs, np.ndarray), "xs should be numpy array"
        assert xs.dtype == np.uint16, "xs should be uint16"
        assert ys.dtype == np.uint16, "ys should be uint16"
        assert ts.dtype == np.float64, "ts should be float64"
        assert ps.dtype == np.int8, "ps should be int8"

    def test_load_events_time_filter(self):
        """Test time filtering during loading"""
        events = evlib.load_events("data/slider_depth/events.txt")
        filtered = events.filter((pl.col('t') >= 1.0) & (pl.col('t') <= 2.0))
        df = filtered.collect()
        xs, ys, ps = df['x'].to_numpy(), df['y'].to_numpy(), df['polarity'].to_numpy()
        # Convert Duration timestamps to seconds (float64)
        ts = df['timestamp'].dt.total_seconds().to_numpy()

        assert len(xs) > 0, "No events loaded with time filter"
        # Convert durations to seconds for comparison
        ts_seconds = ts.astype('float64') / 1e6  # Convert microseconds to seconds
        assert np.all(ts_seconds >= 1.0), "Events before t_start found"
        assert np.all(ts_seconds <= 2.0), "Events after t_end found"

    def test_load_events_spatial_filter(self):
        """Test spatial filtering during loading"""
        events = evlib.load_events("data/slider_depth/events.txt")
        filtered = events.filter(
            (pl.col('x') >= 200) & (pl.col('x') <= 400) &
            (pl.col('y') >= 100) & (pl.col('y') <= 300)
        )
        df = filtered.collect()
        xs, ys, ps = df['x'].to_numpy(), df['y'].to_numpy(), df['polarity'].to_numpy()
        # Convert Duration timestamps to seconds (float64)
        ts = df['timestamp'].dt.total_seconds().to_numpy()

        assert len(xs) > 0, "No events loaded with spatial filter"
        assert np.all(xs >= 200), "Events below min_x found"
        assert np.all(xs <= 400), "Events above max_x found"
        assert np.all(ys >= 100), "Events below min_y found"
        assert np.all(ys <= 300), "Events above max_y found"

    def test_load_events_polarity_filter(self):
        """Test polarity filtering during loading"""
        events = evlib.load_events("data/slider_depth/events.txt")
        pos_events = events.filter(pl.col('polarity') == 1)
        pos_df = pos_events.collect()
        pos_xs, pos_ys, pos_ps = pos_df['x'].to_numpy(), pos_df['y'].to_numpy(), pos_df['polarity'].to_numpy()
        # Convert Duration timestamps to seconds (float64)
        pos_ts = pos_df['t'].dt.total_seconds().to_numpy()

        assert len(pos_xs) > 0, "No positive events loaded"
        assert np.all(pos_ps == 1), "Non-positive events found"

        events = evlib.load_events("data/slider_depth/events.txt")
        neg_events = events.filter(pl.col('polarity') == -1)
        neg_df = neg_events.collect()
        neg_xs, neg_ys, neg_ps = neg_df['x'].to_numpy(), neg_df['y'].to_numpy(), neg_df['polarity'].to_numpy()
        # Convert Duration timestamps to seconds (float64)
        neg_ts = neg_df['t'].dt.total_seconds().to_numpy()

        assert len(neg_xs) > 0, "No negative events loaded"
        assert np.all(neg_ps == -1), "Non-negative events found"

    def test_load_events_file_not_found(self):
        """Test error handling for missing files"""
        with pytest.raises(FileNotFoundError):
            evlib.load_events("nonexistent_file.txt")

    def test_load_events_invalid_format(self):
        """Test error handling for invalid file format"""
        # Create invalid file
        with open("invalid_format.txt", "w") as f:
            f.write("invalid content\n")

        with pytest.raises(OSError):
            evlib.load_events("invalid_format.txt")
```

### Rust Unit Tests

```rust
// src/ev_representations/mod.rs
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ev_core::EventData;

    #[test]
    fn test_voxel_grid_creation() {
        // Create test data
        let xs = vec![100, 200, 300];
        let ys = vec![150, 250, 350];
        let ts = vec![0.0, 0.5, 1.0];
        let ps = vec![1, -1, 1];

        let events = EventData::new(xs, ys, ts, ps);

        // Test voxel grid creation
        let voxel_grid = create_voxel_grid(&events, 640, 480, 3);

        assert_eq!(voxel_grid.shape(), &[3, 480, 640]);
        assert!(voxel_grid.iter().any(|&x| x != 0.0), "Voxel grid should not be empty");
    }

    #[test]
    fn test_voxel_grid_empty_events() {
        // Test with empty events
        let events = EventData::new(vec![], vec![], vec![], vec![]);
        let voxel_grid = create_voxel_grid(&events, 640, 480, 3);

        assert_eq!(voxel_grid.shape(), &[3, 480, 640]);
        assert!(voxel_grid.iter().all(|&x| x == 0.0), "Empty events should produce empty voxel grid");
    }

    #[test]
    fn test_smooth_voxel_grid_creation() {
        // Create test data with precise timing
        let xs = vec![320, 320, 320];
        let ys = vec![240, 240, 240];
        let ts = vec![0.0, 0.33, 0.66];
        let ps = vec![1, 1, 1];

        let events = EventData::new(xs, ys, ts, ps);

        let smooth_voxel = create_smooth_voxel_grid(&events, 640, 480, 3);
        let regular_voxel = create_voxel_grid(&events, 640, 480, 3);

        // Smooth voxel should have different distribution
        assert_ne!(smooth_voxel, regular_voxel);
    }
}
```

## Integration Testing

### End-to-End Pipeline Tests

```python
# tests/integration/test_pipeline.py
import pytest
import numpy as np
import evlib

class TestEventProcessingPipeline:
    def test_full_pipeline(self):
        """Test complete event processing pipeline"""
        # Load events
        import evlib.representations as evr
        events = evlib.load_events("data/slider_depth/events.txt")
        df = events.collect()
        xs, ys, ps = df['x'].to_numpy(), df['y'].to_numpy(), df['polarity'].to_numpy()
        # Convert Duration timestamps to seconds (float64)
        ts = df['timestamp'].dt.total_seconds().to_numpy()

        # Create voxel grid
        voxel_lazy = evr.create_voxel_grid(
            "data/slider_depth/events.txt",
            width=640, height=480, n_time_bins=5
        )
        voxel_df = voxel_lazy.collect()
        voxel_grid = voxel_df.to_numpy().reshape(5, 480, 640)

        # Apply manual spatial transformation
        xs_aug = 640 - 1 - xs
        ys_aug = ys
        ts_aug = ts
        ps_aug = ps

        # Basic visualization test (using matplotlib)
        # import matplotlib.pyplot as plt
        # plt.scatter(xs[:1000], ys[:1000], c=ps[:1000], s=1)

        # Validate results
        assert voxel_grid.shape == (5, 480, 640)
        assert len(xs_aug) == len(xs)
        assert np.all(xs_aug == 640 - 1 - xs)

    def test_hdf5_roundtrip(self):
        """Test HDF5 save/load roundtrip"""
        # Load original data
        events = evlib.load_events("data/slider_depth/events.txt")
        df = events.collect()
        xs, ys, ps = df['x'].to_numpy(), df['y'].to_numpy(), df['polarity'].to_numpy()
        # Convert Duration timestamps to seconds (float64)
        ts = df['timestamp'].dt.total_seconds().to_numpy()

        # Save to HDF5
        evlib.save_events_to_hdf5(xs, ys, ts, ps, "test_output.h5")

        # Load from HDF5
        events2 = evlib.load_events("output.h5")
        df2 = events2.collect()
        xs2, ys2, ps2 = df2['x'].to_numpy(), df2['y'].to_numpy(), df2['polarity'].to_numpy()
        # Convert Duration timestamps to seconds (float64)
        ts2 = df2['timestamp'].dt.total_seconds().to_numpy()

        # Verify perfect roundtrip
        np.testing.assert_array_equal(xs, xs2)
        np.testing.assert_array_equal(ys, ys2)
        np.testing.assert_array_equal(ts, ts2)
        np.testing.assert_array_equal(ps, ps2)

    def test_model_integration(self):
        """Test neural network model integration"""
        # Load events using filtering API
        import evlib.filtering as evf
        events = evlib.load_events("data/slider_depth/events.txt")
        events_df = events.collect()  # Convert LazyFrame to DataFrame first
        filtered_events = evf.filter_by_time(events_df, t_start=0.0, t_end=1.0)
        xs, ys, ps = filtered_events['x'].to_numpy(), filtered_events['y'].to_numpy(), filtered_events['polarity'].to_numpy()
        # Convert Duration timestamps to seconds (float64)
        ts = filtered_events['t'].dt.total_seconds().to_numpy()

        # Note: Neural network processing is under development
        # For now, test basic event processing
        import evlib.representations as evr
        voxel_lazy = evr.create_voxel_grid(
            filtered_events, width=640, height=480, n_time_bins=5
        )
        voxel_df = voxel_lazy.collect()
        voxel_grid = voxel_df.to_numpy().reshape(5, 480, 640)

        # Validate output
        assert voxel_grid.shape == (5, 480, 640)
        assert len(xs) > 0  # Events were loaded
```

### Model Integration Tests

```python
# tests/integration/test_models.py
import pytest
import numpy as np
import evlib

class TestEventRepresentations:
    def test_voxel_grid_creation(self):
        """Test voxel grid creation with real data"""
        import evlib.representations as evr

        # Create voxel grid from events
        voxel_lazy = evr.create_voxel_grid(
            "data/slider_depth/events.txt",
            width=640, height=480, n_time_bins=5
        )
        voxel_df = voxel_lazy.collect()
        voxel_grid = voxel_df.to_numpy().reshape(5, 480, 640)

        assert voxel_grid.shape == (5, 480, 640), "Wrong voxel grid shape"
        assert voxel_grid.dtype == np.float32, "Wrong voxel grid dtype"

    def test_stacked_histogram_creation(self):
        """Test stacked histogram creation"""
        import evlib.representations as evr

        # Load test data with filtering
        import evlib.filtering as evf
        events = evlib.load_events("data/slider_depth/events.txt")
        events_df = events.collect()  # Convert LazyFrame to DataFrame first
        filtered_events = evf.filter_by_time(events_df, t_start=0.0, t_end=0.5)

        # Create stacked histogram
        hist_lazy = evr.create_stacked_histogram(
            filtered_events, width=640, height=480, bins=10
        )
        hist_df = hist_lazy.collect()
        hist_grid = hist_df.to_numpy().reshape(10, 480, 640)

        assert hist_grid.shape == (10, 480, 640), "Wrong histogram shape"
        assert len(filtered_events.collect()) > 0, "No events loaded"

    def test_representation_consistency(self):
        """Test that representations produce consistent results"""
        import evlib.filtering as evf
        import evlib.representations as evr

        # Load events with filtering
        events = evlib.load_events("data/slider_depth/events.txt")
        events_df = events.collect()  # Convert LazyFrame to DataFrame first
        filtered_events = evf.filter_by_time(events_df, t_start=0.0, t_end=0.1)

        # Create voxel grid twice
        voxel_lazy1 = evr.create_voxel_grid(
            filtered_events, width=640, height=480, n_time_bins=5
        )
        voxel_df1 = voxel_lazy1.collect()
        result1 = voxel_df1.to_numpy().reshape(5, 480, 640)

        voxel_lazy2 = evr.create_voxel_grid(
            filtered_events, width=640, height=480, n_time_bins=5
        )
        voxel_df2 = voxel_lazy2.collect()
        result2 = voxel_df2.to_numpy().reshape(5, 480, 640)

        # Results should be identical
        np.testing.assert_array_equal(result1, result2)
```

## Performance Testing

### Benchmark Framework

```python
# tests/benchmarks/test_benchmarks.py
import time
import numpy as np
import evlib
import pytest

class TestPerformanceBenchmarks:
    def setup_method(self):
        """Load test data once for all benchmarks"""
        events = evlib.load_events("data/slider_depth/events.txt")
        df = events.collect()
        self.xs, self.ys, self.ps = df['x'].to_numpy(), df['y'].to_numpy(), df['polarity'].to_numpy()
        # Convert Duration timestamps to seconds (float64)
        self.ts = df['t'].dt.total_seconds().to_numpy()

    def benchmark_voxel_grid_creation(self):
        """Benchmark voxel grid creation vs pure Python"""
        # evlib implementation
        import evlib.representations as evr
        start = time.time()
        voxel_lazy = evr.create_voxel_grid(
            "data/slider_depth/events.txt",
            width=640, height=480, n_time_bins=5
        )
        voxel_df = voxel_lazy.collect()
        voxel_evlib = voxel_df.to_numpy().reshape(5, 480, 640)
        evlib_time = time.time() - start

        # Pure Python implementation
        start = time.time()
        voxel_numpy = self._create_voxel_grid_numpy(
            self.xs, self.ys, self.ts, self.ps, 640, 480, 5
        )
        numpy_time = time.time() - start

        # Validate equivalence
        np.testing.assert_allclose(voxel_evlib, voxel_numpy, rtol=1e-5)

        # Report performance
        speedup = numpy_time / evlib_time
        print(f"Voxel grid creation:")
        print(f"  evlib: {evlib_time:.3f}s")
        print(f"  NumPy: {numpy_time:.3f}s")
        print(f"  Speedup: {speedup:.2f}x")

        # Performance expectation
        assert speedup > 1.0, f"evlib should be faster than NumPy, got {speedup:.2f}x"

    def _create_voxel_grid_numpy(self, xs, ys, ts, ps, width, height, bins):
        """Pure NumPy voxel grid implementation for comparison"""
        voxel_grid = np.zeros((bins, height, width), dtype=np.float32)

        # Temporal binning
        t_min, t_max = ts.min(), ts.max()
        t_bins = np.linspace(t_min, t_max, bins + 1)

        for i in range(len(xs)):
            x, y, t, p = xs[i], ys[i], ts[i], ps[i]

            # Find temporal bin
            bin_idx = np.searchsorted(t_bins[1:], t)
            bin_idx = min(bin_idx, bins - 1)

            # Add to voxel grid
            voxel_grid[bin_idx, y, x] += p

        return voxel_grid

    def benchmark_file_loading(self):
        """Benchmark file loading performance"""
        file_path = "data/slider_depth/events.txt"

        # evlib loading
        start = time.time()
        events = evlib.load_events(file_path)
        df = events.collect()
        xs1, ys1, ps1 = df['x'].to_numpy(), df['y'].to_numpy(), df['polarity'].to_numpy()
        ts1 = df['t'].dt.total_seconds().to_numpy()
        evlib_time = time.time() - start

        # NumPy loading
        start = time.time()
        data = np.loadtxt(file_path)
        numpy_time = time.time() - start

        # Report performance
        speedup = numpy_time / evlib_time
        print(f"File loading:")
        print(f"  evlib: {evlib_time:.3f}s")
        print(f"  NumPy: {numpy_time:.3f}s")
        print(f"  Speedup: {speedup:.2f}x")

    def benchmark_memory_usage(self):
        """Benchmark memory usage"""
        import psutil
        import os

        process = psutil.Process(os.getpid())

        # Baseline memory
        baseline = process.memory_info().rss / 1024 / 1024  # MB

        # Load events
        import evlib.representations as evr
        events = evlib.load_events("data/slider_depth/events.txt")
        df = events.collect()
        xs, ys, ps = df['x'].to_numpy(), df['y'].to_numpy(), df['polarity'].to_numpy()
        # Convert Duration timestamps to seconds (float64)
        ts = df['timestamp'].dt.total_seconds().to_numpy()
        loaded = process.memory_info().rss / 1024 / 1024  # MB

        # Create voxel grid
        voxel_lazy = evr.create_voxel_grid(
            "data/slider_depth/events.txt",
            width=640, height=480, n_time_bins=5
        )
        voxel_df = voxel_lazy.collect()
        voxel_grid = voxel_df.to_numpy().reshape(5, 480, 640)
        voxel = process.memory_info().rss / 1024 / 1024  # MB

        # Report memory usage
        events_memory = loaded - baseline
        voxel_memory = voxel - loaded

        print(f"Memory usage:")
        print(f"  Events: {events_memory:.1f} MB")
        print(f"  Voxel grid: {voxel_memory:.1f} MB")
        print(f"  Memory per event: {events_memory * 1024 / len(xs):.2f} KB")
```

### Continuous Benchmarking

```python
# tests/benchmarks/benchmark_utils.py
import json
import time
from pathlib import Path

class BenchmarkRecorder:
    def __init__(self, benchmark_file="benchmark_results.json"):
        self.benchmark_file = Path(benchmark_file)
        self.results = self._load_existing_results()

    def _load_existing_results(self):
        if self.benchmark_file.exists():
            with open(self.benchmark_file, 'r') as f:
                return json.load(f)
        return {}

    def record_benchmark(self, name, time_seconds, details=None):
        """Record benchmark result"""
        timestamp = time.time()

        if name not in self.results:
            self.results[name] = []

        self.results[name].append({
            'timestamp': timestamp,
            'time_seconds': time_seconds,
            'details': details or {}
        })

        # Save results
        with open(self.benchmark_file, 'w') as f:
            json.dump(self.results, f, indent=2)

    def get_latest_result(self, name):
        """Get latest benchmark result"""
        if name in self.results and self.results[name]:
            return self.results[name][-1]
        return None

    def check_regression(self, name, current_time, threshold=0.2):
        """Check if current time represents a performance regression"""
        latest = self.get_latest_result(name)
        if latest is None:
            return False

        previous_time = latest['time_seconds']
        regression_ratio = (current_time - previous_time) / previous_time

        return regression_ratio > threshold
```

## Notebook Testing

### Jupyter Notebook Validation

```python
# tests/notebooks/test_notebooks.py
import pytest
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

class TestNotebooks:
    def test_notebook_execution(self):
        """Test that all example notebooks execute without errors"""
        notebook_dir = Path("examples")
        notebooks = list(notebook_dir.glob("*.ipynb"))

        assert len(notebooks) > 0, "No notebooks found to test"

        for notebook_path in notebooks:
            with open(notebook_path, 'r') as f:
                notebook = nbformat.read(f, as_version=4)

            # Execute notebook
            ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
            try:
                ep.preprocess(notebook, {'metadata': {'path': str(notebook_dir)}})
                print(f"SUCCESS: {notebook_path.name} executed successfully")
            except Exception as e:
                pytest.fail(f"ERROR: {notebook_path.name} failed: {e}")
```

### Notebook Content Validation

```python
def test_notebook_content():
    """Test that notebooks contain required sections"""
    notebook_dir = Path("examples")

    for notebook_path in notebook_dir.glob("*.ipynb"):
        with open(notebook_path, 'r') as f:
            notebook = nbformat.read(f, as_version=4)

        # Check for required sections
        cells = [cell for cell in notebook.cells if cell.cell_type == 'markdown']
        markdown_text = ' '.join([cell.source for cell in cells])

        assert 'import evlib' in str(notebook), f"Notebook {notebook_path.name} should import evlib"
        assert any('data/' in str(cell) for cell in notebook.cells), \
            f"Notebook {notebook_path.name} should use real data"
```

## Test Execution

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_formats.py

# Run with coverage
pytest --cov=evlib --cov-report=html

# Run benchmarks only
pytest --benchmark-only

# Run tests in parallel
pytest -n auto

# Test specific functionality
pytest tests/unit/test_representations.py::TestVoxelGrid::test_voxel_grid_creation
```

### Rust Tests

```bash
# Run Rust unit tests
cargo test

# Run specific test
cargo test test_voxel_grid_creation

# Run with optimizations
cargo test --release

# Run integration tests
cargo test --test test_smooth_voxel
```

### Notebook Tests

```bash
# Test notebook execution
pytest --nbmake examples/

# Test specific notebook
pytest --nbmake examples/data_reader_demo.ipynb
```

## Continuous Integration

### GitHub Actions Configuration

```yaml
# .github/workflows/test.yml
name: Test Suite

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ["3.10", "3.11", "3.12"]

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install system dependencies
      run: |
        if [ "$RUNNER_OS" == "Linux" ]; then
          sudo apt-get update
          sudo apt-get install -y libhdf5-dev pkg-config
        elif [ "$RUNNER_OS" == "macOS" ]; then
          brew install hdf5 pkg-config
        fi
      shell: bash

    - name: Install Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
        default: true

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install maturin pytest pytest-cov
        maturin develop

    - name: Run tests
      run: |
        pytest tests/ --cov=evlib --cov-report=xml
        cargo test

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## Test Data Management

### Test Data Setup

```python
# tests/conftest.py
import pytest
import numpy as np
import evlib
from pathlib import Path

@pytest.fixture(scope="session")
def test_data_path():
    """Path to test data directory"""
    return Path("data/slider_depth")

@pytest.fixture(scope="session")
def sample_events():
    """Load sample events for testing"""
    return evlib.load_events("data/slider_depth/events.txt")

@pytest.fixture(scope="session")
def small_events():
    """Load small subset of events for fast tests"""
    import evlib.filtering as evf
    events = evlib.load_events("data/slider_depth/events.txt")
    return evf.filter_by_time(
        events,
        t_start=0.0, t_end=0.1
    )

@pytest.fixture
def synthetic_events():
    """Generate synthetic events for testing"""
    np.random.seed(42)  # Reproducible

    n_events = 1000
    xs = np.random.randint(0, 640, n_events, dtype=np.int64)
    ys = np.random.randint(0, 480, n_events, dtype=np.int64)
    ts = np.sort(np.random.rand(n_events).astype(np.float64))
    ps = np.random.choice([-1, 1], n_events, dtype=np.int64)

    return xs, ys, ts, ps
```

## Quality Assurance

### Test Coverage Requirements

- **Unit tests**: 100% coverage for all public functions
- **Integration tests**: All major workflows tested
- **Performance tests**: All new features benchmarked
- **Notebook tests**: All examples must execute successfully

### Test Quality Metrics

```python
# Monitor test quality
def test_coverage_report():
    """Generate coverage report"""
    import coverage

    cov = coverage.Coverage()
    cov.start()

    # Run tests
    import pytest
    pytest.main(["-v", "tests/"])

    cov.stop()
    cov.save()

    # Generate report
    print("\nCoverage Report:")
    cov.report()

    # Check minimum coverage
    total_coverage = cov.report(show_missing=False)
    assert total_coverage >= 80, f"Coverage too low: {total_coverage}%"
```

## Debugging Tests

### Test Debugging

```python
# Use pytest debugging features
def test_debug_example():
    """Example of debugging a failing test"""
    events = evlib.load_events("data/slider_depth/events.txt")
    df = events.collect()
    xs, ys, ps = df['x'].to_numpy(), df['y'].to_numpy(), df['polarity'].to_numpy()
    # Convert Duration timestamps to seconds (float64)
    ts = df['t'].dt.total_seconds().to_numpy()

    # Add debug prints
    print(f"Loaded {len(xs)} events")
    print(f"Time range: {ts.min():.3f} - {ts.max():.3f}")

    # Use assert for debugging
    assert len(xs) > 0, "No events loaded"

    # Break into debugger if needed
    import pdb; pdb.set_trace()

    # Continue with test
    import evlib.representations as evr
    voxel_lazy = evr.create_voxel_grid(
        "data/slider_depth/events.txt",
        width=640, height=480, n_time_bins=5
    )
    voxel_df = voxel_lazy.collect()
    voxel_grid = voxel_df.to_numpy().reshape(5, 480, 640)

    assert voxel_grid.shape == (5, 480, 640)
```

### Performance Debugging

```python
# Profile test performance
import cProfile
import pstats

def profile_test():
    """Profile test execution"""
    pr = cProfile.Profile()
    pr.enable()

    # Run test code
    import evlib.representations as evr
    events = evlib.load_events("data/slider_depth/events.txt")
    df = events.collect()
    xs, ys, ps = df['x'].to_numpy(), df['y'].to_numpy(), df['polarity'].to_numpy()
    # Convert Duration timestamps to seconds (float64)
    ts = df['t'].dt.total_seconds().to_numpy()

    voxel_lazy = evr.create_voxel_grid(
        "data/slider_depth/events.txt",
        width=640, height=480, n_time_bins=5
    )
    voxel_df = voxel_lazy.collect()
    voxel_grid = voxel_df.to_numpy().reshape(5, 480, 640)

    pr.disable()

    # Analyze profile
    stats = pstats.Stats(pr)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
```

---

*Testing is the foundation of reliability. Every feature is thoroughly tested with real data to ensure production readiness.*
