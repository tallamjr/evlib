# ev_filtering Module Testing with Pandera Validation

This directory contains comprehensive pytest tests for the `evlib.ev_filtering` module that incorporate pandera validation to ensure data integrity throughout the filtering pipeline.

## Test Files

- `test_ev_filtering_pandera.py` - Main pytest file with pandera validation
- `validate_filtering_with_pandera.py` - Standalone validation script (in examples/)

## Features

### Pandera Schema Validation
- **EventDataSchema**: Validates raw event data structure and constraints
- **FilteredEventSchema**: Validates filtered events with relaxed constraints
- **SpatialFilterSchema**: Validates spatially filtered events within ROI bounds
- **TemporalFilterSchema**: Validates temporally filtered events within time windows

### Comprehensive Test Coverage
- Individual filter validation (temporal, spatial, polarity, hot pixel, noise)
- Combined filtering pipeline testing
- Progressive filtering validation
- Performance benchmarking
- Edge case handling (empty results, data integrity)
- Real data validation using eTram dataset

### Data Integrity Checks
- Schema validation at each filtering stage
- Coordinate range validation
- Timestamp monotonicity verification
- Polarity value consistency
- Data type preservation

## Running the Tests

### Prerequisites
```bash
# Install test dependencies
pip install pytest pandera polars

# Build evlib with filtering support
maturin develop --features polars
```

### Basic Usage
```bash
# Run all filtering tests
pytest tests/test_ev_filtering_pandera.py -v

# Run specific test class
pytest tests/test_ev_filtering_pandera.py::TestTemporalFiltering -v

# Run with detailed output
pytest tests/test_ev_filtering_pandera.py -v -s

# Skip tests requiring test data
pytest tests/test_ev_filtering_pandera.py -m "not requires_data"
```

### Test Markers
- `requires_pandera` - Tests requiring pandera library
- `requires_data` - Tests requiring real test data files
- `slow` - Performance/benchmark tests

### Expected Output
When tests pass, you'll see validation confirmations like:
```
✅ Schema validation passed for input events
✅ Timestamps are monotonic for temporally filtered events
✅ Coordinate ranges valid for spatially filtered events
✅ All filtered events have positive polarity
```

## Test Data Requirements

Tests look for data files in the following locations:
- `tests/data/eTram/h5/val_2/*.h5`
- `tests/data/gen4_1mpx_processed_RVT/test/**/*.h5`
- `tests/data/slider_depth/events.txt`

If no test data is found, data-dependent tests will be skipped.

## Schema Validation Benefits

1. **Early Error Detection**: Catch data corruption or invalid transformations immediately
2. **Data Contract Enforcement**: Ensure filtering operations maintain expected data structure
3. **Regression Prevention**: Detect when code changes break data integrity
4. **Documentation**: Schemas serve as living documentation of data expectations
5. **Production Readiness**: Same validation logic can be used in production pipelines

## Integration with CI/CD

Add to your pytest configuration:
```ini
[tool:pytest]
markers =
    requires_pandera: tests requiring pandera validation library
    requires_data: tests requiring real test data files
    slow: performance/benchmark tests
```

Run in CI with appropriate markers:
```bash
# Fast tests only
pytest tests/test_ev_filtering_pandera.py -m "not slow and not requires_data"

# With real data
pytest tests/test_ev_filtering_pandera.py -m "not slow"

# Full test suite
pytest tests/test_ev_filtering_pandera.py
```

## Example Test Output

```
TEMPORAL FILTERING TEST
============================================================
Loading test data from: tests/data/eTram/h5/val_2/val_night_011_td.h5
Loaded 1,234,567 events
✅ Schema validation passed for input events
Original time range: [0.000, 2.456]s (duration: 2.456s)
Filtering completed in 0.123s
Events: 1,234,567 → 617,284 (50.0% reduction)
Filtered time range: [0.614, 1.842]s
✅ All events within specified time window
✅ Temporal filter schema validation passed
✅ Timestamps are monotonic for temporally filtered events
```

This comprehensive test suite ensures the Polars-first filtering implementation maintains data integrity while providing excellent performance for event camera data processing.
