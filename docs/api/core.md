# Core API Reference

Core data structures and utilities for event camera data processing.

## Event Data Structures

### Event Arrays

Event data is represented as four separate arrays:

```python
xs: np.ndarray[np.uint16]    # X coordinates (0-639 for 640x480)
ys: np.ndarray[np.uint16]    # Y coordinates (0-479 for 640x480)
ts: np.ndarray[np.float64]   # Timestamps in seconds
ps: np.ndarray[np.int8]      # Polarities (+1 for ON, -1 for OFF)
```

### Data Validation

```python
def validate_events(xs, ys, ts, ps):
    """Validate event arrays for consistency"""

    # Check array lengths match
    lengths = [len(xs), len(ys), len(ts), len(ps)]
    if len(set(lengths)) != 1:
        raise ValueError(f"Array lengths don't match: {lengths}")

    # Check data types
    assert xs.dtype == np.uint16, f"xs must be uint16, got {xs.dtype}"
    assert ys.dtype == np.uint16, f"ys must be uint16, got {ys.dtype}"
    assert ts.dtype == np.float64, f"ts must be float64, got {ts.dtype}"
    assert ps.dtype == np.int8, f"ps must be int8, got {ps.dtype}"

    # Check coordinate bounds
    assert np.all(xs >= 0), "Negative x coordinates found"
    assert np.all(ys >= 0), "Negative y coordinates found"

    # Check polarity values
    assert np.all(np.isin(ps, [-1, 1])), "Invalid polarity values found"

    # Check temporal ordering
    assert np.all(ts[:-1] <= ts[1:]), "Timestamps not in ascending order"

    return True
```

## Utility Functions

### Array Operations

```python
def event_count_summary(xs, ys, ts, ps):
    """Get summary statistics for event arrays"""

    return {
        'total_events': len(xs),
        'duration': ts.max() - ts.min(),
        'event_rate': len(xs) / (ts.max() - ts.min()),
        'positive_events': np.sum(ps > 0),
        'negative_events': np.sum(ps < 0),
        'spatial_extent': {
            'x_range': (xs.min(), xs.max()),
            'y_range': (ys.min(), ys.max())
        }
    }
```

### Memory Usage

```python
def estimate_memory_usage(xs, ys, ts, ps):
    """Estimate memory usage of event arrays"""

    bytes_per_event = xs.itemsize + ys.itemsize + ts.itemsize + ps.itemsize
    total_bytes = len(xs) * bytes_per_event

    return {
        'bytes_per_event': bytes_per_event,
        'total_bytes': total_bytes,
        'megabytes': total_bytes / (1024 * 1024)
    }
```

## Constants

```python
# Standard event camera resolutions
RESOLUTION_VGA = (640, 480)
RESOLUTION_QVGA = (320, 240)
RESOLUTION_HD = (1280, 720)

# Polarity values
POLARITY_POSITIVE = 1
POLARITY_NEGATIVE = -1

# Data type specifications
DTYPE_COORDINATES = np.uint16
DTYPE_TIMESTAMPS = np.float64
DTYPE_POLARITIES = np.int8
```
