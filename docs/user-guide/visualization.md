# Visualization

Learn how to create effective visualizations of event camera data using evlib's built-in tools and integration with popular plotting libraries.

## Overview

Event camera data visualization serves multiple purposes:
- **Data exploration**: Understanding dataset characteristics
- **Algorithm debugging**: Verifying processing results
- **Publication**: Creating figures for papers and presentations
- **Real-time monitoring**: Live visualization of event streams

evlib provides several visualization approaches optimized for different use cases.

## Quick Event Plotting

### Basic Event Scatter Plot

```python
import evlib
import matplotlib.pyplot as plt

# Load events
xs, ys, ts, ps = evlib.formats.load_events("data/slider_depth/events.txt")

# Basic scatter plot
# evlib.visualization.plot_events  # Not available in current version(xs, ys, ts, ps)
plt.title("Event Visualization")
plt.show()
```

### Customized Event Plot

```python
# Plot with custom styling
plt.figure(figsize=(12, 8))

# Separate positive and negative events
pos_mask = ps > 0
neg_mask = ps < 0

# Plot with different colors and sizes
plt.scatter(xs[pos_mask], ys[pos_mask], c='red', s=1, alpha=0.6, label='ON events')
plt.scatter(xs[neg_mask], ys[neg_mask], c='blue', s=1, alpha=0.6, label='OFF events')

plt.xlabel('X coordinate (pixels)')
plt.ylabel('Y coordinate (pixels)')
plt.title(f'Event Data: {len(xs):,} events over {ts.max()-ts.min():.2f}s')
plt.legend()
plt.axis('equal')
plt.gca().invert_yaxis()  # Image coordinates
plt.show()
```

## Event Representations

### Voxel Grid Visualization

```python
# Create voxel grid
voxel_data, voxel_shape_data, voxel_shape_shape = evlib.representations.events_to_voxel_grid(xs, ys, ts, ps, 5, (640, 480))

# Visualize all temporal bins
fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for i in range(5):
    im = axes[i].imshow(voxel_grid[i], cmap='RdBu_r', vmin=-10, vmax=10)
    axes[i].set_title(f'Temporal Bin {i}')
    axes[i].axis('off')

plt.tight_layout()
plt.colorbar(im, ax=axes, shrink=0.6, label='Event Count')
plt.show()
```

### Accumulated Event Image

```python
# Create accumulated image
event_image_data, event_image_shape = evlib.representations.events_to_voxel_grid(xs, ys, ts, ps, 1, (640, 480))[0]

# Visualize
plt.figure(figsize=(10, 8))
plt.imshow(event_image, cmap='RdBu_r')
plt.title('Accumulated Event Image')
plt.colorbar(label='Accumulated Event Count')
plt.axis('off')
plt.show()
```

## Temporal Analysis

### Event Rate Over Time

```python
import numpy as np

def plot_event_rate(ts, time_window=0.01):
    """Plot event rate over time"""

    # Create time bins
    t_min, t_max = ts.min(), ts.max()
    time_bins = np.arange(t_min, t_max, time_window)

    # Count events in each bin
    event_counts, _ = np.histogram(ts, bins=time_bins)
    event_rate = event_counts / time_window  # Events per second

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(time_bins[:-1], event_rate, linewidth=1)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Event Rate (events/second)')
    plt.title(f'Event Rate Over Time (window: {time_window*1000:.1f}ms)')
    plt.grid(True, alpha=0.3)
    plt.show()

    return time_bins[:-1], event_rate

# Analyze event rate
time_axis, rates = plot_event_rate(ts, time_window=0.01)
print(f"Average event rate: {rates.mean():.0f} events/sec")
print(f"Peak event rate: {rates.max():.0f} events/sec")
```

### Event Activity Heatmap

```python
def plot_activity_heatmap(xs, ys, ts, time_window=0.1):
    """Create heatmap of event activity over time"""

    # Time bins
    t_min, t_max = ts.min(), ts.max()
    n_time_bins = int((t_max - t_min) / time_window)

    # Create 2D histogram over time
    activity_map = np.zeros((n_time_bins, 480 // 4))  # Downsample for visualization

    for i in range(n_time_bins):
        t_start = t_min + i * time_window
        t_end = t_start + time_window

        # Events in this time window
        mask = (ts >= t_start) & (ts < t_end)
        if mask.sum() > 0:
            # Create spatial histogram
            xs_window = xs[mask] // 4  # Downsample
            ys_window = ys[mask] // 4

            for x, y in zip(xs_window, ys_window):
                if 0 <= y < activity_map.shape[1]:
                    activity_map[i, y] += 1

    # Plot
    plt.figure(figsize=(12, 8))
    plt.imshow(activity_map.T, aspect='auto', cmap='hot', origin='lower')
    plt.xlabel('Time Bin')
    plt.ylabel('Y Coordinate (downsampled)')
    plt.title(f'Event Activity Heatmap (window: {time_window*1000:.0f}ms)')
    plt.colorbar(label='Event Count')
    plt.show()

# Create activity heatmap
plot_activity_heatmap(xs, ys, ts, time_window=0.05)
```

## Ultra-Fast Terminal Visualization

For real-time applications, evlib provides ultra-fast terminal-based visualization:

```python
# Real-time terminal visualization
# evlib.visualization.terminal_viz  # Not available in current version(xs, ys, ts, ps)
```

This creates a live ASCII visualization in your terminal, perfect for:
- Real-time monitoring
- Embedded systems with no GUI
- Debugging during development
- Maximum performance visualization

## Statistical Visualizations

### Event Distribution Analysis

```python
def analyze_event_statistics(xs, ys, ts, ps):
    """Create comprehensive statistical analysis plots"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Spatial distribution
    axes[0, 0].hist2d(xs, ys, bins=50, cmap='viridis')
    axes[0, 0].set_title('Spatial Event Distribution')
    axes[0, 0].set_xlabel('X coordinate')
    axes[0, 0].set_ylabel('Y coordinate')

    # Temporal distribution
    axes[0, 1].hist(ts, bins=100, alpha=0.7, color='blue')
    axes[0, 1].set_title('Temporal Event Distribution')
    axes[0, 1].set_xlabel('Time (seconds)')
    axes[0, 1].set_ylabel('Event Count')

    # Polarity distribution
    polarity_counts = [np.sum(ps > 0), np.sum(ps < 0)]
    axes[0, 2].bar(['Positive (ON)', 'Negative (OFF)'], polarity_counts,
                   color=['red', 'blue'], alpha=0.7)
    axes[0, 2].set_title('Polarity Distribution')
    axes[0, 2].set_ylabel('Event Count')

    # Inter-event intervals
    inter_intervals = np.diff(ts)
    axes[1, 0].hist(inter_intervals, bins=100, alpha=0.7, color='green')
    axes[1, 0].set_xlabel('Inter-event Interval (seconds)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Inter-event Interval Distribution')
    axes[1, 0].set_yscale('log')

    # Event rate over time
    time_bins = np.linspace(ts.min(), ts.max(), 100)
    event_counts, _ = np.histogram(ts, bins=time_bins)
    bin_centers = (time_bins[:-1] + time_bins[1:]) / 2
    axes[1, 1].plot(bin_centers, event_counts, linewidth=1)
    axes[1, 1].set_xlabel('Time (seconds)')
    axes[1, 1].set_ylabel('Events per bin')
    axes[1, 1].set_title('Event Rate Over Time')

    # Pixel activity distribution
    pixel_activity = np.zeros((480, 640))
    for x, y in zip(xs, ys):
        if 0 <= x < 640 and 0 <= y < 480:
            pixel_activity[y, x] += 1

    im = axes[1, 2].imshow(pixel_activity, cmap='hot')
    axes[1, 2].set_title('Pixel Activity Map')
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], label='Event Count')

    plt.tight_layout()
    plt.show()

    # Print statistics
    print(f"Total events: {len(xs):,}")
    print(f"Duration: {ts.max() - ts.min():.3f} seconds")
    print(f"Average event rate: {len(xs) / (ts.max() - ts.min()):.0f} events/sec")
    print(f"Positive events: {np.sum(ps > 0):,} ({np.mean(ps > 0)*100:.1f}%)")
    print(f"Negative events: {np.sum(ps < 0):,} ({np.mean(ps < 0)*100:.1f}%)")
    print(f"Average inter-event interval: {np.mean(inter_intervals)*1000:.3f} ms")

# Analyze dataset
analyze_event_statistics(xs, ys, ts, ps)
```

## Interactive Visualizations

### Time-Sliced Visualization

```python
from matplotlib.widgets import Slider

def interactive_time_visualization(xs, ys, ts, ps):
    """Create interactive time-sliced visualization"""

    fig, ax = plt.subplots(figsize=(12, 10))
    plt.subplots_adjust(bottom=0.25)

    # Initial time window
    window_duration = 0.1  # 100ms
    t_start = ts.min()

    def update_plot(t_start_val):
        ax.clear()

        # Filter events in time window
        mask = (ts >= t_start_val) & (ts < t_start_val + window_duration)

        if mask.sum() > 0:
            xs_window = xs[mask]
            ys_window = ys[mask]
            ps_window = ps[mask]

            # Plot events
            pos_mask = ps_window > 0
            neg_mask = ps_window < 0

            if pos_mask.sum() > 0:
                ax.scatter(xs_window[pos_mask], ys_window[pos_mask],
                          c='red', s=2, alpha=0.7, label='ON')
            if neg_mask.sum() > 0:
                ax.scatter(xs_window[neg_mask], ys_window[neg_mask],
                          c='blue', s=2, alpha=0.7, label='OFF')

        ax.set_xlim(0, 640)
        ax.set_ylim(480, 0)  # Invert y-axis
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_title(f'Events: {t_start_val:.3f} - {t_start_val + window_duration:.3f}s '
                    f'({mask.sum()} events)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Create slider
    ax_slider = plt.axes([0.1, 0.1, 0.8, 0.03])
    slider = Slider(ax_slider, 'Time Start', ts.min(), ts.max() - window_duration,
                   valinit=t_start, valfmt='%.3f s')

    # Connect slider to update function
    slider.on_changed(update_plot)

    # Initial plot
    update_plot(t_start)

    plt.show()

# Create interactive visualization
interactive_time_visualization(xs, ys, ts, ps)
```

## Publication-Quality Figures

### Professional Event Visualization

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

def create_publication_figure(xs, ys, ts, ps):
    """Create publication-quality figure"""

    # Set style
    plt.style.use('seaborn-v0_8-paper')
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    # Main event visualization
    ax_main = fig.add_subplot(gs[:2, :2])

    # Subsample for cleaner visualization
    subsample = slice(None, None, max(1, len(xs) // 50000))
    xs_sub = xs[subsample]
    ys_sub = ys[subsample]
    ps_sub = ps[subsample]

    pos_mask = ps_sub > 0
    neg_mask = ps_sub < 0

    ax_main.scatter(xs_sub[pos_mask], ys_sub[pos_mask], c='#FF4444', s=0.5,
                   alpha=0.8, label='ON events', rasterized=True)
    ax_main.scatter(xs_sub[neg_mask], ys_sub[neg_mask], c='#4444FF', s=0.5,
                   alpha=0.8, label='OFF events', rasterized=True)

    ax_main.set_xlabel('X coordinate (pixels)', fontsize=12)
    ax_main.set_ylabel('Y coordinate (pixels)', fontsize=12)
    ax_main.set_title('Event Camera Data Visualization', fontsize=14, fontweight='bold')
    ax_main.legend(fontsize=10)
    ax_main.set_xlim(0, 640)
    ax_main.set_ylim(480, 0)
    ax_main.grid(True, alpha=0.3)

    # Temporal distribution
    ax_temporal = fig.add_subplot(gs[0, 2:])
    time_bins = np.linspace(ts.min(), ts.max(), 200)
    counts, _ = np.histogram(ts, bins=time_bins)
    bin_centers = (time_bins[:-1] + time_bins[1:]) / 2
    ax_temporal.fill_between(bin_centers, counts, alpha=0.7, color='purple')
    ax_temporal.set_xlabel('Time (seconds)', fontsize=10)
    ax_temporal.set_ylabel('Event Count', fontsize=10)
    ax_temporal.set_title('Temporal Distribution', fontsize=12)
    ax_temporal.grid(True, alpha=0.3)

    # Polarity pie chart
    ax_pie = fig.add_subplot(gs[1, 2])
    polarity_counts = [np.sum(ps > 0), np.sum(ps < 0)]
    colors = ['#FF4444', '#4444FF']
    wedges, texts, autotexts = ax_pie.pie(polarity_counts, labels=['ON', 'OFF'],
                                         colors=colors, autopct='%1.1f%%',
                                         startangle=90)
    ax_pie.set_title('Polarity Distribution', fontsize=12)

    # Statistics table
    ax_stats = fig.add_subplot(gs[1, 3])
    ax_stats.axis('off')

    stats_data = [
        ['Total Events', f'{len(xs):,}'],
        ['Duration', f'{ts.max() - ts.min():.2f} s'],
        ['Event Rate', f'{len(xs) / (ts.max() - ts.min()):.0f} evt/s'],
        ['Resolution', '640 × 480 px'],
        ['ON Events', f'{np.sum(ps > 0):,} ({np.mean(ps > 0)*100:.1f}%)'],
        ['OFF Events', f'{np.sum(ps < 0):,} ({np.mean(ps < 0)*100:.1f}%)'],
    ]

    table = ax_stats.table(cellText=stats_data,
                          colLabels=['Metric', 'Value'],
                          cellLoc='left',
                          loc='center',
                          colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax_stats.set_title('Dataset Statistics', fontsize=12, pad=20)

    # Voxel grid visualization
    ax_voxel = fig.add_subplot(gs[2, :])
    voxel_data, voxel_shape_data, voxel_shape_shape = evlib.representations.events_to_voxel_grid(xs, ys, ts, ps, 5, (640, 480))

    # Create subplot for each temporal bin
    for i in range(5):
        ax_sub = plt.subplot(gs[2, i])
        im = ax_sub.imshow(voxel_grid[i], cmap='RdBu_r', vmin=-5, vmax=5)
        ax_sub.set_title(f'Bin {i}', fontsize=10)
        ax_sub.axis('off')

    # Add colorbar
    cbar = plt.colorbar(im, ax=fig.get_axes()[-5:], shrink=0.8,
                       orientation='horizontal', pad=0.1)
    cbar.set_label('Event Count', fontsize=10)

    plt.suptitle('Event Camera Data Analysis', fontsize=16, fontweight='bold', y=0.95)

    # Save high-quality figure
    plt.savefig('event_analysis.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

# Create publication figure
create_publication_figure(xs, ys, ts, ps)
```

## Real-Time Visualization

### Live Event Stream Visualization

```python
import time
from IPython.display import clear_output

def live_event_visualization(xs, ys, ts, ps, window_duration=0.1, update_rate=10):
    """Simulate live event visualization"""

    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(10, 8))

    t_start = ts.min()
    t_end = ts.max()
    current_time = t_start

    while current_time < t_end:
        # Clear previous plot
        ax.clear()

        # Get events in current window
        mask = (ts >= current_time) & (ts < current_time + window_duration)

        if mask.sum() > 0:
            xs_window = xs[mask]
            ys_window = ys[mask]
            ps_window = ps[mask]

            # Plot events
            pos_mask = ps_window > 0
            neg_mask = ps_window < 0

            if pos_mask.sum() > 0:
                ax.scatter(xs_window[pos_mask], ys_window[pos_mask],
                          c='red', s=3, alpha=0.8, label='ON')
            if neg_mask.sum() > 0:
                ax.scatter(xs_window[neg_mask], ys_window[neg_mask],
                          c='blue', s=3, alpha=0.8, label='OFF')

        # Set plot properties
        ax.set_xlim(0, 640)
        ax.set_ylim(480, 0)
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_title(f'Live Events: {current_time:.3f} - {current_time + window_duration:.3f}s '
                    f'({mask.sum()} events)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Update display
        plt.draw()
        plt.pause(1.0 / update_rate)

        # Advance time
        current_time += window_duration / 2  # 50% overlap

    plt.ioff()  # Turn off interactive mode
    plt.show()

# Run live visualization (comment out for non-interactive environments)
# live_event_visualization(xs, ys, ts, ps, window_duration=0.05, update_rate=20)
```

## Best Practices

### 1. Performance Optimization

```python
# For large datasets, subsample for visualization
def subsample_events(xs, ys, ts, ps, max_events=50000):
    """Subsample events for visualization performance"""
    if len(xs) <= max_events:
        return xs, ys, ts, ps

    # Random sampling
    indices = np.random.choice(len(xs), max_events, replace=False)
    indices = np.sort(indices)  # Maintain temporal order

    return xs[indices], ys[indices], ts[indices], ps[indices]

# Use subsampled data for plotting
xs_sub, ys_sub, ts_sub, ps_sub = subsample_events(xs, ys, ts, ps)
plt.scatter(xs_sub, ys_sub, c=ps_sub, cmap='RdBu_r', s=1)
plt.show()
```

### 2. Color Schemes

```python
# Recommended color schemes for different purposes

# Scientific publications
colors_scientific = {'on': '#D62728', 'off': '#1F77B4'}  # Red/Blue

# Colorblind-friendly
colors_colorblind = {'on': '#E69F00', 'off': '#56B4E9'}  # Orange/Sky Blue

# High contrast
colors_contrast = {'on': '#FF0000', 'off': '#0000FF'}  # Pure Red/Blue

# Grayscale for print
colors_grayscale = {'on': '#000000', 'off': '#666666'}  # Black/Gray
```

### 3. Figure Export

```python
# High-quality figure export
plt.figure(figsize=(12, 8), dpi=300)
# ... create your plot ...
plt.savefig('events.pdf', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('events.png', dpi=300, bbox_inches='tight', facecolor='white')
```

## Next Steps

- [Neural Networks](models.md): Use visualized data with deep learning
- [API Reference](../api/visualization.md): Detailed visualization functions
- [Examples](../examples/notebooks.md): More visualization examples
