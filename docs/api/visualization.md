# Visualization API Reference

Functions for visualizing event camera data.

## Basic Plotting

### Event Scatter Plots

```python
def plot_events(xs: np.ndarray, ys: np.ndarray, ts: np.ndarray, ps: np.ndarray,
               figsize: Tuple[int, int] = (10, 8),
               alpha: float = 0.6,
               point_size: float = 1.0) -> None:
    """Plot events as scatter plot

    Args:
        xs: X coordinates
        ys: Y coordinates
        ts: Timestamps
        ps: Polarities
        figsize: Figure size
        alpha: Point transparency
        point_size: Point size
    """
    pass

def plot_events_temporal(xs: np.ndarray, ys: np.ndarray, ts: np.ndarray, ps: np.ndarray,
                        time_window: float = 0.1,
                        figsize: Tuple[int, int] = (12, 8)) -> None:
    """Plot events with temporal color coding

    Args:
        xs: X coordinates
        ys: Y coordinates
        ts: Timestamps
        ps: Polarities
        time_window: Time window for color coding
        figsize: Figure size
    """
    pass
```

### Event Images

```python
def plot_event_image(voxel_grid: np.ndarray,
                    bin_index: int = 0,
                    cmap: str = 'RdBu_r',
                    figsize: Tuple[int, int] = (10, 8)) -> None:
    """Plot event voxel grid as image

    Args:
        voxel_grid: Voxel grid (T, H, W)
        bin_index: Temporal bin to display
        cmap: Colormap
        figsize: Figure size
    """
    pass

def plot_accumulated_events(xs: np.ndarray, ys: np.ndarray, ps: np.ndarray,
                          width: int = 640, height: int = 480,
                          figsize: Tuple[int, int] = (10, 8)) -> None:
    """Plot accumulated event image

    Args:
        xs: X coordinates
        ys: Y coordinates
        ps: Polarities
        width: Image width
        height: Image height
        figsize: Figure size
    """
    pass
```

## Advanced Visualizations

### Multi-Panel Figures

```python
def plot_voxel_grid_panels(voxel_grid: np.ndarray,
                          figsize: Tuple[int, int] = (15, 3),
                          cmap: str = 'RdBu_r') -> None:
    """Plot all temporal bins of voxel grid

    Args:
        voxel_grid: Voxel grid (T, H, W)
        figsize: Figure size
        cmap: Colormap
    """
    pass

def plot_event_statistics(xs: np.ndarray, ys: np.ndarray, ts: np.ndarray, ps: np.ndarray,
                         figsize: Tuple[int, int] = (18, 12)) -> None:
    """Create comprehensive event statistics plot

    Args:
        xs: X coordinates
        ys: Y coordinates
        ts: Timestamps
        ps: Polarities
        figsize: Figure size
    """
    pass
```

### Temporal Analysis

```python
def plot_event_rate(ts: np.ndarray,
                   time_window: float = 0.01,
                   figsize: Tuple[int, int] = (12, 6)) -> Tuple[np.ndarray, np.ndarray]:
    """Plot event rate over time

    Args:
        ts: Timestamps
        time_window: Time window for rate calculation
        figsize: Figure size

    Returns:
        Time bins and event rates
    """
    pass

def plot_activity_heatmap(xs: np.ndarray, ys: np.ndarray, ts: np.ndarray,
                         time_window: float = 0.1,
                         figsize: Tuple[int, int] = (12, 8)) -> None:
    """Plot spatio-temporal activity heatmap

    Args:
        xs: X coordinates
        ys: Y coordinates
        ts: Timestamps
        time_window: Time window for binning
        figsize: Figure size
    """
    pass
```

## Interactive Visualizations

### Time-Sliced Viewer

```python
def interactive_time_viewer(xs: np.ndarray, ys: np.ndarray, ts: np.ndarray, ps: np.ndarray,
                           window_duration: float = 0.1,
                           figsize: Tuple[int, int] = (12, 10)) -> None:
    """Create interactive time-sliced event viewer

    Args:
        xs: X coordinates
        ys: Y coordinates
        ts: Timestamps
        ps: Polarities
        window_duration: Time window duration
        figsize: Figure size
    """
    pass
```

### Animation

```python
def create_event_animation(xs: np.ndarray, ys: np.ndarray, ts: np.ndarray, ps: np.ndarray,
                          window_duration: float = 0.1,
                          fps: int = 10,
                          save_path: Optional[str] = None) -> None:
    """Create animated event visualization

    Args:
        xs: X coordinates
        ys: Y coordinates
        ts: Timestamps
        ps: Polarities
        window_duration: Time window duration
        fps: Animation frame rate
        save_path: Optional path to save animation
    """
    pass
```

## Terminal Visualization

### ASCII Display

```python
def terminal_viz(xs: np.ndarray, ys: np.ndarray, ts: np.ndarray, ps: np.ndarray,
                width: int = 640, height: int = 480,
                downsample: int = 8) -> None:
    """Ultra-fast terminal visualization

    Args:
        xs: X coordinates
        ys: Y coordinates
        ts: Timestamps
        ps: Polarities
        width: Image width
        height: Image height
        downsample: Downsampling factor
    """
    pass

def terminal_live_viz(event_stream: Iterator[Tuple[np.ndarray, ...]],
                     refresh_rate: float = 0.1) -> None:
    """Live terminal visualization of event stream

    Args:
        event_stream: Iterator yielding (xs, ys, ts, ps) tuples
        refresh_rate: Display refresh rate in seconds
    """
    pass
```

## Publication-Quality Figures

### Professional Styling

```python
def create_publication_figure(xs: np.ndarray, ys: np.ndarray, ts: np.ndarray, ps: np.ndarray,
                             style: str = 'seaborn-v0_8-paper',
                             figsize: Tuple[int, int] = (16, 10),
                             dpi: int = 300) -> None:
    """Create publication-quality event visualization

    Args:
        xs: X coordinates
        ys: Y coordinates
        ts: Timestamps
        ps: Polarities
        style: Matplotlib style
        figsize: Figure size
        dpi: Resolution
    """
    pass

def save_high_quality_figure(filename: str,
                           dpi: int = 300,
                           bbox_inches: str = 'tight',
                           facecolor: str = 'white') -> None:
    """Save current figure with high quality settings

    Args:
        filename: Output filename
        dpi: Resolution
        bbox_inches: Bounding box setting
        facecolor: Background color
    """
    pass
```

## Color Schemes

### Predefined Palettes

```python
# Color schemes for different use cases
COLOR_SCHEMES = {
    'scientific': {'on': '#D62728', 'off': '#1F77B4'},
    'colorblind': {'on': '#E69F00', 'off': '#56B4E9'},
    'high_contrast': {'on': '#FF0000', 'off': '#0000FF'},
    'grayscale': {'on': '#000000', 'off': '#666666'}
}

def get_color_scheme(scheme_name: str) -> Dict[str, str]:
    """Get color scheme for event visualization

    Args:
        scheme_name: Name of color scheme

    Returns:
        Dictionary with 'on' and 'off' colors
    """
    return COLOR_SCHEMES.get(scheme_name, COLOR_SCHEMES['scientific'])
```

## Performance Utilities

### Optimization

```python
def subsample_events_for_visualization(xs: np.ndarray, ys: np.ndarray,
                                     ts: np.ndarray, ps: np.ndarray,
                                     max_events: int = 50000) -> Tuple[np.ndarray, ...]:
    """Subsample events for visualization performance

    Args:
        xs: X coordinates
        ys: Y coordinates
        ts: Timestamps
        ps: Polarities
        max_events: Maximum number of events to plot

    Returns:
        Subsampled event arrays
    """
    pass

def optimize_plot_parameters(n_events: int) -> Dict[str, Any]:
    """Get optimized plot parameters based on event count

    Args:
        n_events: Number of events

    Returns:
        Dictionary with optimized parameters
    """
    if n_events > 100000:
        return {'point_size': 0.5, 'alpha': 0.3, 'rasterized': True}
    elif n_events > 10000:
        return {'point_size': 1.0, 'alpha': 0.6, 'rasterized': True}
    else:
        return {'point_size': 2.0, 'alpha': 0.8, 'rasterized': False}
```

## Validation

### Plot Validation

```python
def validate_visualization_data(xs: np.ndarray, ys: np.ndarray,
                              ts: np.ndarray, ps: np.ndarray) -> None:
    """Validate data for visualization

    Args:
        xs: X coordinates
        ys: Y coordinates
        ts: Timestamps
        ps: Polarities

    Raises:
        ValueError: If data is invalid for visualization
    """
    if len(xs) == 0:
        raise ValueError("No events to visualize")

    if not np.all(np.isfinite(xs)) or not np.all(np.isfinite(ys)):
        raise ValueError("Invalid coordinates detected")

    if not np.all(np.isfinite(ts)):
        raise ValueError("Invalid timestamps detected")
```
