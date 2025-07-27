# DuckDB Integration with evlib

This document demonstrates how to use evlib's zero-copy Arrow integration with DuckDB for high-performance event data analysis.

## Overview

evlib provides seamless integration with the Apache Arrow ecosystem through zero-copy data transfer. This enables efficient interoperability with analytics engines like DuckDB, allowing you to perform complex SQL queries on event camera data without expensive data copies.

## Basic Usage

### Loading Events to Arrow Format

```python
import evlib
import polars as pl

# Load events as Polars DataFrame (Arrow integration coming soon)
events_df = evlib.load_events("data/slider_depth/events.txt")
# Convert to Arrow for DuckDB integration
events = events_df.collect().to_arrow()
```

### DuckDB Integration

```python
import evlib
import duckdb

# Load events and convert to Arrow format for DuckDB
events_df = evlib.load_events("data/slider_depth/events.txt")

# Register with DuckDB
con = duckdb.connect()
con.register("events", events_df.collect().to_arrow())

# Basic queries
print("=== Basic Event Statistics ===")

# Total event count
total_events = con.execute("SELECT COUNT(*) as total_events FROM events").fetchone()[0]
print(f"Total events: {total_events:,}")

# Event distribution by polarity
polarity_dist = con.execute("""
    SELECT polarity, COUNT(*) as count,
           ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
    FROM events
    GROUP BY polarity
    ORDER BY polarity
""").fetchall()

print(f"\\nPolarity distribution:")
for pol, count, pct in polarity_dist:
    print(f"  Polarity {pol}: {count:,} events ({pct}%)")

# Temporal statistics
temporal_stats = con.execute("""
    SELECT
        MIN(EXTRACT(microseconds FROM timestamp)) as min_timestamp,
        MAX(EXTRACT(microseconds FROM timestamp)) as max_timestamp,
        AVG(EXTRACT(microseconds FROM timestamp)) as avg_timestamp,
        STDDEV(EXTRACT(microseconds FROM timestamp)) as std_timestamp
    FROM events
""").fetchone()

print(f"\\nTemporal statistics:")
print(f"  Time range: {temporal_stats[0]:,.0f} to {temporal_stats[1]:,.0f} microseconds")
print(f"  Average timestamp: {temporal_stats[2]:,.0f} microseconds")
print(f"  Standard deviation: {temporal_stats[3]:,.0f} microseconds")

# Spatial statistics
spatial_stats = con.execute("""
    SELECT
        MIN(x) as min_x, MAX(x) as max_x,
        MIN(y) as min_y, MAX(y) as max_y,
        COUNT(DISTINCT x) as unique_x_coords,
        COUNT(DISTINCT y) as unique_y_coords
    FROM events
""").fetchone()

print(f"\\nSpatial statistics:")
print(f"  X range: {spatial_stats[0]} to {spatial_stats[1]} ({spatial_stats[4]} unique values)")
print(f"  Y range: {spatial_stats[2]} to {spatial_stats[3]} ({spatial_stats[5]} unique values)")
```

## Advanced Analytics

### Time-based Analysis

```python
# Events per second
import duckdb
con = duckdb.connect()
events_df = evlib.load_events("data/slider_depth/events.txt")
con.register("events", events_df.collect().to_arrow())

events_per_second = con.execute("""
    SELECT
        EXTRACT(microseconds FROM timestamp) // 1000000 as second,  -- Convert microseconds to seconds
        COUNT(*) as events_count
    FROM events
    GROUP BY EXTRACT(microseconds FROM timestamp) // 1000000
    ORDER BY second
    LIMIT 10
""").fetchall()

print("Events per second (first 10 seconds):")
for second, count in events_per_second:
    print(f"  Second {second}: {count:,} events")
```

### Spatial Density Analysis

```python
# Spatial density heatmap data
import duckdb
con = duckdb.connect()
events_df = evlib.load_events("data/slider_depth/events.txt")
con.register("events", events_df.collect().to_arrow())

density_map = con.execute("""
    SELECT
        x // 10 * 10 as x_bin,  -- 10x10 pixel bins
        y // 10 * 10 as y_bin,
        COUNT(*) as event_density
    FROM events
    GROUP BY x_bin, y_bin
    HAVING event_density > 100  -- Filter low-density areas
    ORDER BY event_density DESC
    LIMIT 20
""").fetchall()

print("\\nTop 20 highest density regions (10x10 pixel bins):")
for x_bin, y_bin, density in density_map:
    print(f"  Region ({x_bin},{y_bin}): {density:,} events")
```

### Performance Comparison

The zero-copy Arrow integration provides significant performance benefits:

- **Memory efficiency**: No data copying between evlib and DuckDB
- **Query performance**: DuckDB's columnar engine optimized for Arrow data
- **Ecosystem compatibility**: Works seamlessly with Polars, Pandas, and other Arrow-compatible tools

### Alternative Ecosystems

The same Arrow data can be used with other analytics engines:

```python
# With Polars (already loaded as DataFrame)
import polars as pl
events_df = evlib.load_events("data/slider_depth/events.txt")
result = events_df.group_by("polarity").agg(pl.len())

# With Pandas (includes copy overhead)
import pandas as pd
df = events_df.collect().to_pandas()
result = df.groupby("polarity").size()
```

## Supported File Formats

evlib's Arrow integration supports all major event camera formats:

- **HDF5**: Prophesee, iniVation datasets
- **EVT2/EVT3**: Prophesee RAW format
- **AEDAT**: iniVation format (v1.0, v2.0, v3.1, v4.0)
- **AER**: Address-Event Representation
- **Text**: Custom CSV-like formats

## Performance Notes

- Arrow integration is enabled with the `arrow` feature flag
- Zero-copy transfer requires compatible data layouts
- For maximum performance, use file formats that align with Arrow's columnar structure
- Large datasets benefit most from the zero-copy approach

## Dependencies

To use DuckDB integration:

```bash
pip install evlib[polars] duckdb
```

Or for development:

```bash
maturin develop --features polars
pip install duckdb
```
