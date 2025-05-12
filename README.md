# evlib: Event Camera Utilities in Rust

A high-performance implementation of event camera utilities using Rust with Python bindings via PyO3.

This library is based on the [event_utils](https://github.com/TimoStoff/event_utils) Python library but reimplemented in Rust for significantly better performance.

## Installation

```bash
pip install evlib
```

Or for development:

```bash
pip install -e ".[dev]"
```

## Features

- Event data structure manipulation
- Event augmentation
  - Random event addition
  - Correlated event addition
  - Event removal
- Event transformations
  - Flipping events along x and y axes
  - Clipping events to bounds
  - Rotating events

## Usage Examples

### Basic Usage

```python
import numpy as np
import evlib

# Create example event data
xs = np.array([10, 20, 30, 40], dtype=np.int64)
ys = np.array([50, 60, 70, 80], dtype=np.int64)
ts = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
ps = np.array([1, -1, 1, -1], dtype=np.int64)

# Convert to block representation
block = evlib.events_to_block(xs, ys, ts, ps)
print(f"Block shape: {block.shape}")  # (4, 4)
```

### Event Augmentation

```python
import numpy as np
import evlib
import matplotlib.pyplot as plt

# Create sample event data
xs = np.array([50, 60, 70, 80, 90], dtype=np.int64)
ys = np.array([50, 60, 70, 80, 90], dtype=np.int64)
ts = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
ps = np.array([1, -1, 1, -1, 1], dtype=np.int64)

# Add random events
to_add = 20
new_xs, new_ys, new_ts, new_ps = evlib.add_random_events(xs, ys, ts, ps, to_add)
print(f"Original events: {len(xs)}, After adding random events: {len(new_xs)}")

# Visualize original vs augmented events
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(xs, ys, c=ps, cmap='coolwarm', s=50, alpha=0.8)
plt.title('Original Events')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(new_xs, new_ys, c=new_ps, cmap='coolwarm', s=50, alpha=0.8)
plt.title('After Adding Random Events')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)

plt.tight_layout()
# plt.savefig('random_events.png')
# plt.show()
```

### Correlated Event Addition

```python
import numpy as np
import evlib
import matplotlib.pyplot as plt

# Create sample event data
xs = np.array([50, 60, 70, 80, 90], dtype=np.int64)
ys = np.array([50, 60, 70, 80, 90], dtype=np.int64)
ts = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
ps = np.array([1, -1, 1, -1, 1], dtype=np.int64)

# Add correlated events (events near existing ones)
to_add = 15
xy_std = 2.0  # Standard deviation for x,y coordinates
ts_std = 0.005  # Standard deviation for timestamps

new_xs, new_ys, new_ts, new_ps = evlib.add_correlated_events(
    xs, ys, ts, ps, to_add, 
    xy_std=xy_std, 
    ts_std=ts_std
)

# Visualize original vs correlated events
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(xs, ys, c=ps, cmap='coolwarm', s=50, alpha=0.8)
plt.title('Original Events')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(new_xs, new_ys, c=new_ps, cmap='coolwarm', s=50, alpha=0.8)
plt.title('After Adding Correlated Events')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)

plt.tight_layout()
# plt.savefig('correlated_events.png')
# plt.show()
```

### Event Transformations

```python
import numpy as np
import evlib
import matplotlib.pyplot as plt

# Create sample event data
xs = np.array([10, 20, 30, 40, 50, 60, 70], dtype=np.int64)
ys = np.array([15, 25, 35, 45, 55, 65, 75], dtype=np.int64)
ts = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float64)
ps = np.array([1, -1, 1, -1, 1, -1, 1], dtype=np.int64)

# Set the sensor resolution
sensor_resolution = (100, 100)  # (height, width)

# Flip events along x-axis
flipped_x_xs, flipped_x_ys, _, _ = evlib.flip_events_x(xs, ys, ts, ps, sensor_resolution)

# Flip events along y-axis
flipped_y_xs, flipped_y_ys, _, _ = evlib.flip_events_y(xs, ys, ts, ps, sensor_resolution)

# Rotate events by 45 degrees
theta_radians = np.pi / 4  # 45 degrees
center_of_rotation = (50, 50)  # Center of rotation
rotated_xs, rotated_ys, _, _ = evlib.rotate_events(
    xs, ys, sensor_resolution, theta_radians, center_of_rotation
)

# Visualize transformations
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.scatter(xs, ys, c=ps, cmap='coolwarm', s=50, alpha=0.8)
plt.title('Original Events')
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.grid(True)

plt.subplot(2, 2, 2)
plt.scatter(flipped_x_xs, flipped_x_ys, c=ps, cmap='coolwarm', s=50, alpha=0.8)
plt.title('Flipped X')
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.grid(True)

plt.subplot(2, 2, 3)
plt.scatter(flipped_y_xs, flipped_y_ys, c=ps, cmap='coolwarm', s=50, alpha=0.8)
plt.title('Flipped Y')
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.grid(True)

plt.subplot(2, 2, 4)
plt.scatter(rotated_xs, rotated_ys, c=ps, cmap='coolwarm', s=50, alpha=0.8)
plt.title('Rotated 45°')
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.grid(True)

plt.tight_layout()
# plt.savefig('transformations.png')
# plt.show()
```

### Clip Events to Bounds

```python
import numpy as np
import evlib
import matplotlib.pyplot as plt

# Create sample event data (with some events outside the desired bounds)
xs = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=np.int64)
ys = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=np.int64)
ts = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float64)
ps = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1], dtype=np.int64)

# Define bounds [min_y, max_y, min_x, max_x]
bounds = [30, 70, 30, 70]

# Clip events to bounds
clipped_xs, clipped_ys, clipped_ts, clipped_ps = evlib.clip_events_to_bounds(
    xs, ys, ts, ps, bounds
)

# Visualize clipping
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(xs, ys, c=ps, cmap='coolwarm', s=50, alpha=0.8)
plt.axvline(x=bounds[2], color='r', linestyle='--')
plt.axvline(x=bounds[3], color='r', linestyle='--')
plt.axhline(y=bounds[0], color='r', linestyle='--')
plt.axhline(y=bounds[1], color='r', linestyle='--')
plt.title('Original Events')
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(clipped_xs, clipped_ys, c=clipped_ps, cmap='coolwarm', s=50, alpha=0.8)
plt.axvline(x=bounds[2], color='r', linestyle='--')
plt.axvline(x=bounds[3], color='r', linestyle='--')
plt.axhline(y=bounds[0], color='r', linestyle='--')
plt.axhline(y=bounds[1], color='r', linestyle='--')
plt.title('Clipped Events')
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.grid(True)

plt.tight_layout()
# plt.savefig('clipped_events.png')
# plt.show()
```

### Generating Realistic DVS Event Data

```python
import numpy as np
import evlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Function to generate a simple moving pattern that would trigger DVS events
def generate_dvs_sample(width=128, height=128, num_frames=10):
    """Generate a synthetic DVS event stream from moving bars."""
    events = []
    timestamps = np.linspace(0, 1, num_frames)
    
    # Create moving horizontal bar
    for i, t in enumerate(timestamps):
        y_pos = int(height * (i / num_frames))
        
        # Generate positive events at leading edge
        for x in range(width):
            if i > 0 and y_pos > 0:
                events.append((x, y_pos, t, 1))  # Positive events (brightness increase)
                events.append((x, y_pos-1, t, -1))  # Negative events (brightness decrease)
    
    # Convert to numpy arrays
    xs = np.array([e[0] for e in events], dtype=np.int64)
    ys = np.array([e[1] for e in events], dtype=np.int64)
    ts = np.array([e[2] for e in events], dtype=np.float64)
    ps = np.array([e[3] for e in events], dtype=np.int64)
    
    return xs, ys, ts, ps

# Generate sample DVS events
width, height = 128, 128
xs, ys, ts, ps = generate_dvs_sample(width, height, 20)

# Apply various transformations
sensor_resolution = (height, width)

# Add correlated noise events
new_xs, new_ys, new_ts, new_ps = evlib.add_correlated_events(
    xs, ys, ts, ps, 
    to_add=len(xs)//4,  # Add 25% more events 
    xy_std=1.0,         # Low spatial spread
    ts_std=0.01         # Low temporal spread
)

# Flip along both axes (rotate 180 degrees)
flipped_xs, flipped_ys, _, _ = evlib.flip_events_x(new_xs, new_ys, new_ts, new_ps, sensor_resolution)
flipped_xs, flipped_ys, flipped_ts, flipped_ps = evlib.flip_events_y(
    flipped_xs, flipped_ys, new_ts, new_ps, sensor_resolution
)

# Visualize events over time
def plot_dvs_events(xs, ys, ts, ps, title):
    """Create 3D visualization of DVS events over time."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color by polarity
    colors = np.array(['r' if p > 0 else 'b' for p in ps])
    
    # Plot 3D points
    ax.scatter(xs, ys, ts, c=colors, s=5, alpha=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Time')
    ax.set_title(title)
    
    plt.tight_layout()
    # plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    # plt.show()

# Plot original and transformed events
plot_dvs_events(xs, ys, ts, ps, "Original DVS Events")
plot_dvs_events(new_xs, new_ys, new_ts, new_ps, "DVS Events with Correlated Noise")
plot_dvs_events(flipped_xs, flipped_ys, flipped_ts, flipped_ps, "180° Rotated DVS Events")
```

## API Reference

### Event Manipulation

- `events_to_block(xs, ys, ts, ps)`: Convert event components into a block representation
- `merge_events(event_sets)`: Merge multiple sets of events

### Event Augmentation

- `add_random_events(xs, ys, ts, ps, to_add, sensor_resolution=None, sort=True, return_merged=True)`: Add random events
- `remove_events(xs, ys, ts, ps, to_remove, add_noise=0)`: Remove events by random selection
- `add_correlated_events(xs, ys, ts, ps, to_add, sort=True, return_merged=True, xy_std=1.5, ts_std=0.001, add_noise=0)`: Add events in the vicinity of existing events

### Event Transformations

- `flip_events_x(xs, ys, ts, ps, sensor_resolution=(180, 240))`: Flip events along x axis
- `flip_events_y(xs, ys, ts, ps, sensor_resolution=(180, 240))`: Flip events along y axis
- `clip_events_to_bounds(xs, ys, ts=None, ps=None, bounds=[180, 240], set_zero=False)`: Clip events to the given bounds
- `rotate_events(xs, ys, sensor_resolution=(180, 240), theta_radians=None, center_of_rotation=None, clip_to_range=False)`: Rotate events by a given angle

## Implementation Details

The library is implemented in Rust using PyO3 for Python bindings, organized into modules:

- `events_core.rs`: Core event data structures and functions
- `transforms.rs`: Event transformation functions
- `lib.rs`: PyO3 module definition

## License

MIT