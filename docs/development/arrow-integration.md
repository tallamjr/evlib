# Apache Arrow Integration

## Overview

Apache Arrow integration in evlib provides zero-copy data transfer and high-performance interoperability with Arrow-based systems like PyArrow, DuckDB, and other Arrow ecosystem tools. This **complements** rather than **replaces** the existing efficient Polars infrastructure.

## When to Use Arrow vs Polars

### Current Polars Architecture (Optimal for Internal Processing)

```
File → Rust Format Reader → Rust Events Vec → Polars DataFrame → Python Dict → Polars LazyFrame
                                          ↑                      ↑
                                   Copy happens here      Copy happens here
                                   (~13 bytes/event)      (serialization)
```

**Use Polars for:**
- All `evlib.filtering` operations (most efficient)
- Internal evlib processing and analysis
- When working within the evlib ecosystem

### New Arrow Architecture (Optimal for External Integration)

```
File → Rust Format Reader → Rust Events Vec → Arrow RecordBatch → Python PyArrow Table
                                          ↑                     ↑
                                   Copy happens here       ZERO COPY!
                                   (~15 bytes/event)       (shared memory)
```

**Use Arrow for:**
- Exporting data to external Arrow-compatible systems
- Integration with DuckDB, Parquet files, or other Arrow tools
- When you need zero-copy data sharing with external libraries

## Python API

### Loading Events as Arrow

```python
# Arrow integration requires the 'arrow' feature to be enabled during build
# Build evlib with: maturin develop --features arrow

# For standard installations, use Polars which provides Arrow internally:
import evlib

# Load events as Polars DataFrame (uses Arrow columnar format internally)
events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()

# Convert to PyArrow if needed (Polars uses Arrow internally)
arrow_table = df.to_arrow()

# With filtering options
events_filtered = evlib.load_events(
    "data/slider_depth/events.txt",
    t_start=0.1,
    t_end=0.5,
    min_x=100, max_x=500,
    polarity=1
)
df_filtered = events_filtered.collect()
arrow_table_filtered = df_filtered.to_arrow()

print(f"Loaded {len(arrow_table)} events to Arrow format via Polars")
print(f"Filtered to {len(arrow_table_filtered)} events")
```

### Converting Arrow Back to Events

```python
# Convert PyArrow back to event data dictionary via Polars
import pyarrow as pa
import polars as pl
import evlib

# First create an Arrow table from evlib data
events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()
arrow_table = df.to_arrow()

# Convert Arrow table back to Polars DataFrame
df_from_arrow = pl.from_arrow(arrow_table)

# Extract as dictionary with NumPy arrays
events_dict = {
    "x": df_from_arrow['x'].to_numpy(),
    "y": df_from_arrow['y'].to_numpy(),
    "t": df_from_arrow['t'].to_numpy(),
    "polarity": df_from_arrow['polarity'].to_numpy()
}

print(f"Converted back to events dict with {len(events_dict['x'])} events")
print(f"Event types: {list(events_dict.keys())}")
```

### Integration with External Tools

```python
# External tool integration via Arrow is under development
# For now, use Polars DataFrames:

import evlib
import duckdb
import polars as pl

# Load as Polars DataFrame
events_df = evlib.load_events("data/slider_depth/events.txt")

# Convert to Arrow for DuckDB (Polars handles this efficiently)
con = duckdb.connect()
con.register("events", events_df.collect().to_arrow())

# SQL queries with DuckDB
result = con.execute("SELECT COUNT(*) FROM events WHERE polarity = 1").fetchone()
print(f"Positive events: {result[0]}")

# Export to Parquet using Polars
events_df.collect().write_parquet("events.parquet")

# Convert to Pandas if needed (not recommended for large datasets)
df = events_df.collect().to_pandas()
```

## Schema Specification

### Arrow Schema Definition

The Arrow schema exactly matches the existing Polars schema for perfect compatibility:

```rust
Schema::new(vec![
    Field::new("x", DataType::Int16, false),
    Field::new("y", DataType::Int16, false),
    Field::new("t", DataType::Duration(TimeUnit::Microsecond), false),
    Field::new("polarity", DataType::Int8, false),
])
```

### Memory Layout

| Field     | Type                         | Bytes | Description |
|-----------|------------------------------|-------|-------------|
| x         | Int16                        | 2     | Pixel x coordinate |
| y         | Int16                        | 2     | Pixel y coordinate |
| t | Duration(Microseconds)       | 8     | Event time |
| polarity  | Int8                         | 1     | Event polarity |
| **Total** |                              | **13** | Core data per event |
| **Arrow Overhead** |                   | **~2** | Array metadata |
| **Total with Arrow** |                 | **~15** | Bytes per event |

### Format-Specific Polarity Encoding

Arrow implementation maintains the same polarity encoding as the existing Polars implementation:

- **EVT2/EVT3/HDF5**: `0/1 → -1/1` (true polarity representation)
- **Text/Other**: `0/1 → 0/1` (matches file format)

### Timestamp Conversion

Automatic time handling:
- **If time value ≥ 1,000,000**: Assume microseconds, use directly
- **If time value < 1,000,000**: Assume seconds, multiply by 1,000,000
- **Output**: Always Duration(Microseconds) for consistency

## Building with Arrow Support

### Installation

Arrow support requires the `arrow` feature flag:

```bash
# Install with Arrow support
maturin develop --features arrow,polars,python

# Or build for distribution
maturin build --release --features arrow,polars,python
```

### Feature Configuration

```toml
# Cargo.toml
[features]
default = ["polars", "python"]
arrow = ["dep:arrow", "dep:arrow-array", "dep:pyo3-arrow"]
zero-copy = ["arrow"]  # Alias for clarity

[dependencies]
arrow = { version = "55.0", default-features = false, optional = true }
arrow-array = { version = "55.0", optional = true }
pyo3-arrow = { version = "0.10.1", optional = true }
```

## Performance Characteristics

### Memory Efficiency Comparison

| Implementation | Bytes per Event | Notes |
|----------------|-----------------|-------|
| Raw Event struct | 24 | Rust representation |
| **Polars DataFrame** | **13** | **Most efficient for processing** |
| Arrow RecordBatch | 15 | Slightly more overhead than Polars |

### Zero-Copy Benefits

The real benefit of Arrow is **eliminating data duplication**:

- **Before**: Rust events → copy to Python lists → copy to NumPy → copy to external tool = **3x memory**
- **After**: Rust Arrow → zero-copy share with Python → direct external tool access = **1x memory**

### Performance Targets

- **Direct loading**: Match Polars performance (baseline)
- **Zero-copy scenarios**: 10-50% improvement vs dictionary conversion
- **Streaming**: Support datasets >100M events without memory issues

## Use Cases and Examples

### Scientific Computing with DuckDB

```python
import evlib
import duckdb

# Load large event dataset and convert to Arrow
events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()
arrow_table = df.to_arrow()

# Register with DuckDB
con = duckdb.connect()
con.register("events", arrow_table)

# Complex SQL analysis (impossible with pure Python/Polars)
analysis = con.execute("""
    SELECT
        x // 50 as tile_x,
        y // 50 as tile_y,
        COUNT(*) as event_count
    FROM events
    WHERE polarity = 1
    GROUP BY tile_x, tile_y
    ORDER BY event_count DESC
    LIMIT 10
""").fetchdf()

print("Top 10 most active tiles:")
print(analysis)
```

### Data Export Pipeline

```python
import evlib

# Load and export to multiple formats efficiently
events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()

# Export to Parquet (compressed, columnar storage)
df.write_parquet("output.parquet")

# Export to CSV (if needed) - convert Duration to numeric first
import polars as pl
df_for_csv = df.with_columns([
    pl.col("t").dt.total_microseconds().alias("time_us")
]).drop("t")
df_for_csv.write_csv("output.csv")

# Export to HDF5 via PyTables (if needed)
# import tables
# events_df.collect().to_pandas().to_hdf("output.h5", key="events")
print("HDF5 export example - uncomment if pytables is available")
```

### Integration with Machine Learning

```python
import evlib
import numpy as np
# sklearn imported conditionally below

# Load events and convert to Arrow
events = evlib.load_events("data/slider_depth/events.txt")
df = events.collect()

# Convert to NumPy for ML (efficient columnar access)
coords = np.column_stack([
    df['x'].to_numpy(),
    df['y'].to_numpy()
])

# Note: sklearn not available in test environment
try:
    from sklearn.cluster import KMeans
    # Spatial clustering
    kmeans = KMeans(n_clusters=10)
    clusters = kmeans.fit_predict(coords)
    print(f"Found {len(np.unique(clusters))} spatial clusters")
except ImportError:
    print("sklearn not available - skipping clustering example")
```

## Architecture Details

### Core Components

1. **ArrowEventBuilder**: High-performance Arrow array construction
2. **ArrowEventStreamer**: Chunked processing for large datasets
3. **Schema Management**: Exact Polars compatibility
4. **Python Bindings**: pyo3-arrow integration for zero-copy transfer

### Integration Points

The Arrow implementation integrates at several levels:

```rust
// File loading with Arrow output
#[cfg(feature = "arrow")]
pub fn load_events_to_arrow(
    path: &str,
    config: &LoadConfig
) -> Result<RecordBatch, Box<dyn std::error::Error>>

// Python bindings for PyArrow
#[cfg(all(feature = "python", feature = "arrow"))]
#[pyfunction]
pub fn load_events_to_pyarrow(
    py: Python<'_>,
    path: &str,
    // ... filtering parameters
) -> PyResult<PyObject>  // Returns PyArrow Table
```

### Error Handling

Comprehensive error handling for Arrow operations:

```rust
#[derive(Debug, thiserror::Error)]
pub enum ArrowBuilderError {
    #[error("Arrow array construction failed: {0}")]
    ArrayConstruction(String),

    #[error("Invalid event data: {message}")]
    InvalidData { message: String },

    #[error("Feature not enabled: Arrow support requires 'arrow' feature flag")]
    FeatureNotEnabled,
}
```

## Testing and Validation

### Test Coverage

The Arrow integration includes comprehensive tests:

- **Schema validation**: Ensure exact match with Polars schema
- **Polarity encoding**: Verify format-specific encoding (EVT2 vs Text)
- **Timestamp conversion**: Test seconds ↔ microseconds conversion
- **Round-trip conversion**: Arrow → Events → Arrow consistency
- **Streaming**: Large dataset processing validation
- **Zero-copy verification**: Memory usage validation

### Running Tests

```bash
# Test Arrow functionality (without Python dependencies)
cargo test test_arrow_integration --no-default-features --features arrow

# Test specific Arrow features
cargo test test_arrow_schema_creation --no-default-features --features arrow
cargo test test_arrow_round_trip_conversion --no-default-features --features arrow
```

## Best Practices

### When to Use Arrow

✅ **Use Arrow when:**
- Exporting data to external systems (DuckDB, Parquet, etc.)
- Integrating with non-Polars data science tools
- Need zero-copy data sharing
- Working with Arrow-native applications

❌ **Don't use Arrow when:**
- Using `evlib.filtering` operations (Polars is more efficient)
- Working purely within the evlib ecosystem
- Memory usage is more critical than interoperability

### Performance Tips

1. **Prefer Polars for internal processing**: The existing Polars pipeline is optimized for evlib operations
2. **Use Arrow for export/import**: Convert to Arrow only when interfacing with external systems
3. **Consider streaming**: For large datasets (>5M events), streaming provides memory efficiency
4. **Batch operations**: Process multiple files together when possible

### Migration Guidelines

The Arrow integration is **additive** - no existing code needs to change:

```python
# Standard API (recommended for all use cases)
import evlib
events = evlib.load_events("data/slider_depth/events.txt")  # Returns Polars LazyFrame
df = events.collect()

# Convert to Arrow when needed for external integration
arrow_events = df.to_arrow()  # Returns PyArrow Table

# Note: Direct Arrow API (evlib.formats.load_events_to_pyarrow) requires
# building evlib with --features arrow flag
```

## Conclusion

The Apache Arrow integration provides powerful interoperability capabilities while maintaining the efficiency of the existing Polars-based architecture. Use Arrow when you need to interface with external systems, and continue using the Polars pipeline for internal evlib operations.

The zero-copy benefits are realized specifically when sharing data with external Arrow-compatible systems, enabling efficient data export, SQL analysis, and integration with the broader data science ecosystem.
