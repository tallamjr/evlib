# Python Scripts

Standalone Python scripts demonstrating evlib usage patterns and common workflows.

## Basic Scripts

### Event File Converter

Convert between different event file formats:

```python
#!/usr/bin/env python3
"""
Convert event files between different formats.
Usage: python convert_events.py input.txt output.h5
"""
import sys
import evlib

def main():
    if len(sys.argv) != 3:
        print("Usage: python convert_events.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Load events
    print(f"Loading events from {input_file}...")
    xs, ys, ts, ps = evlib.formats.load_events(input_file)
    print(f"Loaded {len(xs):,} events")

    # Save in new format
    print(f"Saving events to {output_file}...")
    if output_file.endswith('.h5'):
        evlib.formats.save_events_to_hdf5(xs, ys, ts, ps, output_file)
    else:
        evlib.formats.save_events_to_text(xs, ys, ts, ps, output_file)

    print("Conversion complete!")

if __name__ == "__main__":
    main()
```

### Event Statistics

Generate comprehensive statistics for event files:

```python
#!/usr/bin/env python3
"""
Generate statistics for event camera data.
Usage: python event_stats.py events.txt
"""
import sys
import evlib
import numpy as np

def analyze_events(xs, ys, ts, ps):
    """Generate comprehensive event statistics"""

    n_events = len(xs)
    duration = ts.max() - ts.min()

    stats = {
        'total_events': n_events,
        'duration': duration,
        'event_rate': n_events / duration,
        'positive_events': np.sum(ps > 0),
        'negative_events': np.sum(ps < 0),
        'spatial_extent': {
            'x_range': (xs.min(), xs.max()),
            'y_range': (ys.min(), ys.max()),
            'width': xs.max() - xs.min() + 1,
            'height': ys.max() - ys.min() + 1
        },
        'temporal_stats': {
            'start_time': ts.min(),
            'end_time': ts.max(),
            'mean_interval': np.mean(np.diff(ts)),
            'std_interval': np.std(np.diff(ts))
        }
    }

    return stats

def print_stats(stats):
    """Print formatted statistics"""

    print("=== Event Camera Data Statistics ===")
    print(f"Total events: {stats['total_events']:,}")
    print(f"Duration: {stats['duration']:.3f} seconds")
    print(f"Event rate: {stats['event_rate']:.0f} events/sec")
    print()

    print("Polarity distribution:")
    pos_pct = 100 * stats['positive_events'] / stats['total_events']
    neg_pct = 100 * stats['negative_events'] / stats['total_events']
    print(f"  Positive (ON):  {stats['positive_events']:,} ({pos_pct:.1f}%)")
    print(f"  Negative (OFF): {stats['negative_events']:,} ({neg_pct:.1f}%)")
    print()

    print("Spatial extent:")
    spatial = stats['spatial_extent']
    print(f"  X range: {spatial['x_range'][0]} - {spatial['x_range'][1]} (width: {spatial['width']})")
    print(f"  Y range: {spatial['y_range'][0]} - {spatial['y_range'][1]} (height: {spatial['height']})")
    print()

    print("Temporal statistics:")
    temporal = stats['temporal_stats']
    print(f"  Start time: {temporal['start_time']:.6f} seconds")
    print(f"  End time: {temporal['end_time']:.6f} seconds")
    print(f"  Mean inter-event interval: {temporal['mean_interval']*1000:.3f} ms")
    print(f"  Std inter-event interval: {temporal['std_interval']*1000:.3f} ms")

def main():
    if len(sys.argv) != 2:
        print("Usage: python event_stats.py <event_file>")
        sys.exit(1)

    event_file = sys.argv[1]

    # Load events
    xs, ys, ts, ps = evlib.formats.load_events(event_file)

    # Analyze
    stats = analyze_events(xs, ys, ts, ps)

    # Display results
    print_stats(stats)

if __name__ == "__main__":
    main()
```

## Processing Scripts

### Batch Voxel Grid Creation

Process multiple event files to create voxel grids:

```python
#!/usr/bin/env python3
"""
Batch process event files to create voxel grids.
Usage: python batch_voxel.py input_dir output_dir
"""
import os
import sys
import glob
import evlib
import numpy as np

def process_file(input_path, output_path, bins=5):
    """Process single event file to voxel grid"""

    try:
        # Load events
        xs, ys, ts, ps = evlib.formats.load_events(input_path)

        # Create voxel grid
        voxel_data, voxel_shape_data, voxel_shape_shape = evlib.representations.events_to_voxel_grid(
            xs, ys, ts, ps, 640, 480, bins
        )

        # Save as numpy array
        np.save(output_path, voxel_grid)

        return True, len(xs)

    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False, 0

def main():
    if len(sys.argv) != 3:
        print("Usage: python batch_voxel.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find event files
    pattern = os.path.join(input_dir, "*.txt")
    event_files = glob.glob(pattern)

    if not event_files:
        print(f"No event files found in {input_dir}")
        sys.exit(1)

    print(f"Processing {len(event_files)} files...")

    successful = 0
    total_events = 0

    for event_file in event_files:
        basename = os.path.splitext(os.path.basename(event_file))[0]
        output_file = os.path.join(output_dir, f"{basename}_voxel.npy")

        print(f"Processing {event_file}...")
        success, n_events = process_file(event_file, output_file)

        if success:
            successful += 1
            total_events += n_events
            print(f"  Created {output_file} ({n_events:,} events)")

    print(f"\nProcessing complete:")
    print(f"  Successful: {successful}/{len(event_files)}")
    print(f"  Total events processed: {total_events:,}")

if __name__ == "__main__":
    main()
```

### Event Filtering

Filter events based on various criteria:

```python
#!/usr/bin/env python3
"""
Filter event data based on specified criteria.
Usage: python filter_events.py input.txt output.txt --time-start 1.0 --time-end 5.0
"""
import argparse
import evlib

def parse_args():
    parser = argparse.ArgumentParser(description="Filter event camera data")
    parser.add_argument("input_file", help="Input event file")
    parser.add_argument("output_file", help="Output event file")
    parser.add_argument("--time-start", type=float, help="Start time (seconds)")
    parser.add_argument("--time-end", type=float, help="End time (seconds)")
    parser.add_argument("--min-x", type=int, help="Minimum X coordinate")
    parser.add_argument("--max-x", type=int, help="Maximum X coordinate")
    parser.add_argument("--min-y", type=int, help="Minimum Y coordinate")
    parser.add_argument("--max-y", type=int, help="Maximum Y coordinate")
    parser.add_argument("--polarity", type=int, choices=[-1, 1], help="Filter by polarity")

    return parser.parse_args()

def main():
    args = parse_args()

    # Create load configuration
    load_config = evlib.formats.LoadConfig()

    if args.time_start is not None:
        load_config.t_start = args.time_start
    if args.time_end is not None:
        load_config.t_end = args.time_end
    if args.min_x is not None:
        load_config.min_x = args.min_x
    if args.max_x is not None:
        load_config.max_x = args.max_x
    if args.min_y is not None:
        load_config.min_y = args.min_y
    if args.max_y is not None:
        load_config.max_y = args.max_y
    if args.polarity is not None:
        load_config.polarity = args.polarity

    # Load filtered events
    print(f"Loading events from {args.input_file}...")
    xs, ys, ts, ps = evlib.formats.load_events(args.input_file, config=load_config)

    print(f"Loaded {len(xs):,} events after filtering")

    # Save filtered events
    print(f"Saving filtered events to {args.output_file}...")
    evlib.formats.save_events_to_text(xs, ys, ts, ps, args.output_file)

    print("Filtering complete!")

if __name__ == "__main__":
    main()
```

## Visualization Scripts

### Event Movie Creator

Create video from event data:

```python
#!/usr/bin/env python3
"""
Create video visualization from event data.
Usage: python event_movie.py events.txt output.mp4
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import evlib

def create_event_movie(xs, ys, ts, ps, output_file, fps=30, duration=None):
    """Create animated visualization of events"""

    if duration is None:
        duration = ts.max() - ts.min()

    window_duration = 0.05  # 50ms windows
    n_frames = int(duration / window_duration)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 640)
    ax.set_ylim(480, 0)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")

    def animate(frame):
        ax.clear()

        # Current time window
        t_start = ts.min() + frame * window_duration
        t_end = t_start + window_duration

        # Filter events in window
        mask = (ts >= t_start) & (ts < t_end)

        if mask.sum() > 0:
            xs_frame = xs[mask]
            ys_frame = ys[mask]
            ps_frame = ps[mask]

            # Plot events
            pos_mask = ps_frame > 0
            neg_mask = ps_frame < 0

            if pos_mask.sum() > 0:
                ax.scatter(xs_frame[pos_mask], ys_frame[pos_mask],
                          c='red', s=2, alpha=0.8, label='ON')
            if neg_mask.sum() > 0:
                ax.scatter(xs_frame[neg_mask], ys_frame[neg_mask],
                          c='blue', s=2, alpha=0.8, label='OFF')

        ax.set_xlim(0, 640)
        ax.set_ylim(480, 0)
        ax.set_title(f'Events: {t_start:.3f} - {t_end:.3f}s ({mask.sum()} events)')
        ax.legend()

    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=1000/fps)

    # Save video
    anim.save(output_file, writer='ffmpeg', fps=fps)
    plt.close()

def main():
    if len(sys.argv) != 3:
        print("Usage: python event_movie.py <event_file> <output_video>")
        sys.exit(1)

    event_file = sys.argv[1]
    output_file = sys.argv[2]

    # Load events
    print(f"Loading events from {event_file}...")
    xs, ys, ts, ps = evlib.formats.load_events(event_file)

    # Create movie
    print("Creating event movie...")
    create_event_movie(xs, ys, ts, ps, output_file)

    print(f"Movie saved to {output_file}")

if __name__ == "__main__":
    main()
```

## Utility Scripts

### Performance Benchmark

Comprehensive performance testing:

```python
#!/usr/bin/env python3
"""
Benchmark evlib performance against alternatives.
Usage: python benchmark.py [data_file]
"""
import sys
import time
import numpy as np
import evlib

def benchmark_loading(file_path, n_runs=5):
    """Benchmark event loading performance"""

    times = []
    for _ in range(n_runs):
        start = time.time()
        xs, ys, ts, ps = evlib.formats.load_events(file_path)
        times.append(time.time() - start)

    avg_time = np.mean(times)
    std_time = np.std(times)

    return avg_time, std_time, len(xs)

def benchmark_voxel_creation(xs, ys, ts, ps, n_runs=10):
    """Benchmark voxel grid creation"""

    times = []
    for _ in range(n_runs):
        start = time.time()
        voxel_data, voxel_shape = evlib.representations.events_to_voxel_grid(xs, ys, ts, ps, 5, (640, 480))
        times.append(time.time() - start)

    avg_time = np.mean(times)
    std_time = np.std(times)

    return avg_time, std_time

def main():
    if len(sys.argv) == 2:
        data_file = sys.argv[1]
    else:
        # Use default test data
        data_file = "data/slider_depth/events.txt"

    print("=== evlib Performance Benchmark ===")
    print(f"Data file: {data_file}")
    print()

    # Benchmark loading
    print("Benchmarking event loading...")
    load_time, load_std, n_events = benchmark_loading(data_file)
    print(f"Loading time: {load_time:.3f}s ± {load_std:.3f}s")
    print(f"Events loaded: {n_events:,}")
    print(f"Loading rate: {n_events/load_time:.0f} events/sec")
    print()

    # Load for further benchmarks
    xs, ys, ts, ps = evlib.formats.load_events(data_file)

    # Benchmark voxel creation
    print("Benchmarking voxel grid creation...")
    voxel_time, voxel_std = benchmark_voxel_creation(xs, ys, ts, ps)
    print(f"Voxel creation time: {voxel_time:.3f}s ± {voxel_std:.3f}s")
    print(f"Processing rate: {len(xs)/voxel_time:.0f} events/sec")
    print()

    # Memory usage estimate
    bytes_per_event = 2 + 2 + 8 + 1  # uint16 + uint16 + float64 + int8
    memory_mb = len(xs) * bytes_per_event / (1024 * 1024)
    print(f"Memory usage: {memory_mb:.1f} MB")

if __name__ == "__main__":
    main()
```

## Running Scripts

### Prerequisites

Ensure evlib is installed:

```bash
pip install evlib
```

### Make Scripts Executable

```bash
chmod +x *.py
```

### Example Usage

```bash
# Convert file format
python convert_events.py events.txt events.h5

# Generate statistics
python event_stats.py data/slider_depth/events.txt

# Filter events
python filter_events.py input.txt filtered.txt --time-start 1.0 --time-end 3.0

# Create batch voxel grids
python batch_voxel.py event_files/ voxel_grids/

# Run benchmarks
python benchmark.py data/slider_depth/events.txt
```

## Script Templates

### Basic Template

```python
#!/usr/bin/env python3
"""
Script description.
Usage: python script.py arguments
"""
import sys
import evlib

def main():
    # Parse arguments
    if len(sys.argv) != 2:
        print("Usage: python script.py <event_file>")
        sys.exit(1)

    event_file = sys.argv[1]

    try:
        # Load and process events
        xs, ys, ts, ps = evlib.formats.load_events(event_file)

        # Your processing code here

        print("Processing complete!")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## See Also

- [Jupyter Notebooks](notebooks.md): Interactive examples
- [API Reference](../api/core.md): Function documentation
- [Performance Guide](../getting-started/performance.md): Optimization tips
