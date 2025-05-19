# evlib Examples

This directory contains examples demonstrating how to use the `evlib` library for event camera data processing.

## Module Structure

The module structure has been updated:
- `evlib.core` - Core functionality for event handling (formerly `evlib.events`)
- `evlib.augmentation` - Event augmentation functions
- `evlib.formats` - Data loading and saving
- `evlib.representations` - Event representations (voxel grid, etc.)
- `evlib.visualization` - Visualization utilities

## Examples

### Basic Usage
```
python basic_usage.py
```
Demonstrates basic functionality:
- Creating event data arrays
- Converting to block representation
- Using basic utility functions

### Event Augmentation
```
python event_augmentation.py
```
Demonstrates event augmentation techniques:
- Adding random events
- Adding correlated events (noise near existing events)
- Removing events
- Visualizing the results

### Event Transformations
```
python event_transformations.py
```
Demonstrates spatial transformations:
- Flipping events along x and y axes
- Rotating events by an angle
- Clipping events to bounds
- Visualizing transformations

### Synthetic DVS Data
```
python synthetic_dvs_data.py
```
Demonstrates creating and visualizing synthetic event data:
- Generating events from moving patterns
- Applying transformations to event streams
- Visualizing events in 3D space (x, y, t)
- Creating animations from event data

### Benchmark
```
python benchmark.py
```
Benchmarks the Rust-backed implementation against pure Python:
- Measures performance for common operations
- Compares execution times
- Verifies correctness of results

## Requirements

These examples require:
- NumPy
- Matplotlib
- evlib

## Usage Notes

1. Make sure you have installed evlib:
```
pip install evlib
```

2. For development:
```
pip install -e ".[dev]"
```

3. Run any example directly:
```
python examples/basic_usage.py
```
