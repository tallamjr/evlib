# Event Augmentation

Learn how to apply data augmentation techniques to event camera data for improved neural network training and robust processing pipelines.

## Overview

Event camera data requires specialized augmentation techniques that preserve the temporal and spatial relationships between events. evlib provides optimized implementations for:

- **Spatial transformations**: Rotation, scaling, cropping, flipping
- **Temporal transformations**: Time scaling, temporal jittering
- **Noise and filtering**: Adding noise, event filtering
- **Polarity transformations**: Polarity flipping, masking

## Spatial Augmentation

### Basic Spatial Transformations

```python
import evlib
import numpy as np

# Load events
xs, ys, ts, ps = evlib.formats.load_events("data/slider_depth/events.txt")

# Random horizontal flip
def random_horizontal_flip(xs, ys, ts, ps, width=640, probability=0.5):
    """Randomly flip events horizontally"""

    if np.random.random() < probability:
        xs_flipped = width - 1 - xs
        return xs_flipped, ys, ts, ps
    return xs, ys, ts, ps

# Apply augmentation
xs_aug, ys_aug, ts_aug, ps_aug = random_horizontal_flip(xs, ys, ts, ps)
```

### Rotation

```python
def rotate_events(xs, ys, ts, ps, angle_degrees, width=640, height=480):
    """Rotate events around image center"""

    # Convert to radians
    angle_rad = np.radians(angle_degrees)

    # Center coordinates
    cx, cy = width // 2, height // 2

    # Translate to origin
    xs_centered = xs - cx
    ys_centered = ys - cy

    # Apply rotation
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    xs_rotated = xs_centered * cos_angle - ys_centered * sin_angle
    ys_rotated = xs_centered * sin_angle + ys_centered * cos_angle

    # Translate back
    xs_final = xs_rotated + cx
    ys_final = ys_rotated + cy

    # Filter events within bounds
    valid_mask = (
        (xs_final >= 0) & (xs_final < width) &
        (ys_final >= 0) & (ys_final < height)
    )

    return (
        xs_final[valid_mask].astype(np.uint16),
        ys_final[valid_mask].astype(np.uint16),
        ts[valid_mask],
        ps[valid_mask]
    )

# Apply rotation
xs_rot, ys_rot, ts_rot, ps_rot = rotate_events(xs, ys, ts, ps, angle_degrees=15)
print(f"Events after rotation: {len(xs_rot)} (was {len(xs)})")
```

### Scaling and Cropping

```python
def random_scale_crop(xs, ys, ts, ps, scale_range=(0.8, 1.2),
                     output_size=(640, 480)):
    """Random scale and crop augmentation"""

    # Random scale factor
    scale = np.random.uniform(*scale_range)

    # Apply scaling
    xs_scaled = xs * scale
    ys_scaled = ys * scale

    # Random crop parameters
    output_w, output_h = output_size
    max_x = xs_scaled.max() if len(xs_scaled) > 0 else output_w
    max_y = ys_scaled.max() if len(ys_scaled) > 0 else output_h

    # Ensure we can crop
    if max_x > output_w:
        crop_x = np.random.randint(0, int(max_x - output_w) + 1)
    else:
        crop_x = 0

    if max_y > output_h:
        crop_y = np.random.randint(0, int(max_y - output_h) + 1)
    else:
        crop_y = 0

    # Apply crop
    xs_cropped = xs_scaled - crop_x
    ys_cropped = ys_scaled - crop_y

    # Filter events within output bounds
    valid_mask = (
        (xs_cropped >= 0) & (xs_cropped < output_w) &
        (ys_cropped >= 0) & (ys_cropped < output_h)
    )

    return (
        xs_cropped[valid_mask].astype(np.uint16),
        ys_cropped[valid_mask].astype(np.uint16),
        ts[valid_mask],
        ps[valid_mask]
    )

# Apply scale and crop
xs_sc, ys_sc, ts_sc, ps_sc = random_scale_crop(xs, ys, ts, ps)
```

## Temporal Augmentation

### Time Scaling

```python
def temporal_scaling(xs, ys, ts, ps, scale_factor=1.0):
    """Scale temporal dimension of events"""

    # Scale timestamps
    ts_scaled = ts * scale_factor

    # Maintain relative timing
    ts_scaled = ts_scaled - ts_scaled.min()

    return xs, ys, ts_scaled, ps

# Apply temporal scaling
xs_temp, ys_temp, ts_temp, ps_temp = temporal_scaling(xs, ys, ts, ps, scale_factor=0.8)
print(f"Duration changed from {ts.max()-ts.min():.3f}s to {ts_temp.max()-ts_temp.min():.3f}s")
```

### Temporal Jittering

```python
def temporal_jittering(xs, ys, ts, ps, jitter_std=0.001):
    """Add temporal jitter to events"""

    # Add Gaussian noise to timestamps
    jitter = np.random.normal(0, jitter_std, len(ts))
    ts_jittered = ts + jitter

    # Ensure monotonicity (optional)
    ts_jittered = np.maximum.accumulate(ts_jittered)

    return xs, ys, ts_jittered, ps

# Apply temporal jittering
xs_jit, ys_jit, ts_jit, ps_jit = temporal_jittering(xs, ys, ts, ps, jitter_std=0.0005)
```

### Temporal Windowing

```python
def random_temporal_window(xs, ys, ts, ps, window_duration=1.0):
    """Extract random temporal window"""

    duration = ts.max() - ts.min()

    if duration <= window_duration:
        return xs, ys, ts, ps

    # Random start time
    max_start = duration - window_duration
    start_offset = np.random.uniform(0, max_start)

    start_time = ts.min() + start_offset
    end_time = start_time + window_duration

    # Filter events in window
    window_mask = (ts >= start_time) & (ts < end_time)

    return xs[window_mask], ys[window_mask], ts[window_mask], ps[window_mask]

# Apply temporal windowing
xs_win, ys_win, ts_win, ps_win = random_temporal_window(xs, ys, ts, ps, window_duration=0.5)
```

## Noise and Filtering

### Event Noise

```python
def add_event_noise(xs, ys, ts, ps, noise_rate=0.1, width=640, height=480):
    """Add random noise events"""

    n_events = len(xs)
    n_noise = int(n_events * noise_rate)

    # Generate noise events
    noise_xs = np.random.randint(0, width, n_noise, dtype=np.uint16)
    noise_ys = np.random.randint(0, height, n_noise, dtype=np.uint16)
    noise_ts = np.random.uniform(ts.min(), ts.max(), n_noise)
    noise_ps = np.random.choice([-1, 1], n_noise, dtype=np.int8)

    # Combine with original events
    xs_noisy = np.concatenate([xs, noise_xs])
    ys_noisy = np.concatenate([ys, noise_ys])
    ts_noisy = np.concatenate([ts, noise_ts])
    ps_noisy = np.concatenate([ps, noise_ps])

    # Sort by timestamp
    sort_idx = np.argsort(ts_noisy)

    return (
        xs_noisy[sort_idx],
        ys_noisy[sort_idx],
        ts_noisy[sort_idx],
        ps_noisy[sort_idx]
    )

# Add noise events
xs_noise, ys_noise, ts_noise, ps_noise = add_event_noise(xs, ys, ts, ps, noise_rate=0.05)
print(f"Added {len(xs_noise) - len(xs)} noise events")
```

### Event Dropout

```python
def event_dropout(xs, ys, ts, ps, dropout_rate=0.1):
    """Randomly drop events"""

    n_events = len(xs)
    n_keep = int(n_events * (1 - dropout_rate))

    # Random indices to keep
    keep_idx = np.random.choice(n_events, n_keep, replace=False)
    keep_idx = np.sort(keep_idx)

    return xs[keep_idx], ys[keep_idx], ts[keep_idx], ps[keep_idx]

# Apply event dropout
xs_drop, ys_drop, ts_drop, ps_drop = event_dropout(xs, ys, ts, ps, dropout_rate=0.05)
print(f"Dropped {len(xs) - len(xs_drop)} events")
```

### Spatial Filtering

```python
def random_spatial_mask(xs, ys, ts, ps, mask_probability=0.1, mask_size=(50, 50)):
    """Apply random spatial masking"""

    width, height = 640, 480
    mask_w, mask_h = mask_size

    if np.random.random() < mask_probability:
        # Random mask position
        mask_x = np.random.randint(0, width - mask_w)
        mask_y = np.random.randint(0, height - mask_h)

        # Create mask
        mask = ~(
            (xs >= mask_x) & (xs < mask_x + mask_w) &
            (ys >= mask_y) & (ys < mask_y + mask_h)
        )

        return xs[mask], ys[mask], ts[mask], ps[mask]

    return xs, ys, ts, ps

# Apply spatial masking
xs_mask, ys_mask, ts_mask, ps_mask = random_spatial_mask(xs, ys, ts, ps)
```

## Polarity Augmentation

### Polarity Flipping

```python
def random_polarity_flip(xs, ys, ts, ps, probability=0.5):
    """Randomly flip event polarities"""

    if np.random.random() < probability:
        ps_flipped = -ps
        return xs, ys, ts, ps_flipped

    return xs, ys, ts, ps

# Apply polarity flipping
xs_pf, ys_pf, ts_pf, ps_pf = random_polarity_flip(xs, ys, ts, ps)
```

### Polarity Masking

```python
def polarity_masking(xs, ys, ts, ps, mask_probability=0.1):
    """Randomly mask one polarity"""

    if np.random.random() < mask_probability:
        # Choose polarity to mask
        polarity_to_mask = np.random.choice([-1, 1])

        # Keep only opposite polarity
        mask = ps != polarity_to_mask
        return xs[mask], ys[mask], ts[mask], ps[mask]

    return xs, ys, ts, ps

# Apply polarity masking
xs_pm, ys_pm, ts_pm, ps_pm = polarity_masking(xs, ys, ts, ps)
```

## Composite Augmentation Pipeline

### Augmentation Pipeline

```python
class EventAugmentationPipeline:
    """Comprehensive event augmentation pipeline"""

    def __init__(self, config):
        self.config = config

    def apply_spatial_augmentation(self, xs, ys, ts, ps):
        """Apply spatial augmentations"""

        # Horizontal flip
        if self.config.get('horizontal_flip', False):
            xs, ys, ts, ps = random_horizontal_flip(xs, ys, ts, ps,
                                                   probability=self.config.get('flip_prob', 0.5))

        # Rotation
        if self.config.get('rotation', False):
            max_angle = self.config.get('max_rotation', 15)
            angle = np.random.uniform(-max_angle, max_angle)
            xs, ys, ts, ps = rotate_events(xs, ys, ts, ps, angle)

        # Scale and crop
        if self.config.get('scale_crop', False):
            scale_range = self.config.get('scale_range', (0.8, 1.2))
            xs, ys, ts, ps = random_scale_crop(xs, ys, ts, ps, scale_range)

        return xs, ys, ts, ps

    def apply_temporal_augmentation(self, xs, ys, ts, ps):
        """Apply temporal augmentations"""

        # Temporal scaling
        if self.config.get('temporal_scaling', False):
            scale_range = self.config.get('temporal_scale_range', (0.8, 1.2))
            scale = np.random.uniform(*scale_range)
            xs, ys, ts, ps = temporal_scaling(xs, ys, ts, ps, scale)

        # Temporal jittering
        if self.config.get('temporal_jittering', False):
            jitter_std = self.config.get('jitter_std', 0.001)
            xs, ys, ts, ps = temporal_jittering(xs, ys, ts, ps, jitter_std)

        # Temporal windowing
        if self.config.get('temporal_window', False):
            window_duration = self.config.get('window_duration', 1.0)
            xs, ys, ts, ps = random_temporal_window(xs, ys, ts, ps, window_duration)

        return xs, ys, ts, ps

    def apply_noise_augmentation(self, xs, ys, ts, ps):
        """Apply noise and filtering augmentations"""

        # Add noise events
        if self.config.get('add_noise', False):
            noise_rate = self.config.get('noise_rate', 0.1)
            xs, ys, ts, ps = add_event_noise(xs, ys, ts, ps, noise_rate)

        # Event dropout
        if self.config.get('event_dropout', False):
            dropout_rate = self.config.get('dropout_rate', 0.1)
            xs, ys, ts, ps = event_dropout(xs, ys, ts, ps, dropout_rate)

        # Spatial masking
        if self.config.get('spatial_masking', False):
            mask_prob = self.config.get('mask_probability', 0.1)
            mask_size = self.config.get('mask_size', (50, 50))
            xs, ys, ts, ps = random_spatial_mask(xs, ys, ts, ps, mask_prob, mask_size)

        return xs, ys, ts, ps

    def apply_polarity_augmentation(self, xs, ys, ts, ps):
        """Apply polarity augmentations"""

        # Polarity flipping
        if self.config.get('polarity_flip', False):
            flip_prob = self.config.get('polarity_flip_prob', 0.5)
            xs, ys, ts, ps = random_polarity_flip(xs, ys, ts, ps, flip_prob)

        # Polarity masking
        if self.config.get('polarity_masking', False):
            mask_prob = self.config.get('polarity_mask_prob', 0.1)
            xs, ys, ts, ps = polarity_masking(xs, ys, ts, ps, mask_prob)

        return xs, ys, ts, ps

    def __call__(self, xs, ys, ts, ps):
        """Apply full augmentation pipeline"""

        # Apply augmentations in order
        xs, ys, ts, ps = self.apply_spatial_augmentation(xs, ys, ts, ps)
        xs, ys, ts, ps = self.apply_temporal_augmentation(xs, ys, ts, ps)
        xs, ys, ts, ps = self.apply_noise_augmentation(xs, ys, ts, ps)
        xs, ys, ts, ps = self.apply_polarity_augmentation(xs, ys, ts, ps)

        return xs, ys, ts, ps

# Configure augmentation pipeline
augmentation_config = {
    'horizontal_flip': True,
    'flip_prob': 0.5,
    'rotation': True,
    'max_rotation': 10,
    'scale_crop': True,
    'scale_range': (0.9, 1.1),
    'temporal_scaling': True,
    'temporal_scale_range': (0.9, 1.1),
    'temporal_jittering': True,
    'jitter_std': 0.0005,
    'add_noise': True,
    'noise_rate': 0.02,
    'event_dropout': True,
    'dropout_rate': 0.02,
    'polarity_flip': True,
    'polarity_flip_prob': 0.3
}

# Create and apply pipeline
augmenter = EventAugmentationPipeline(augmentation_config)
xs_aug, ys_aug, ts_aug, ps_aug = augmenter(xs, ys, ts, ps)

print(f"Original events: {len(xs)}")
print(f"Augmented events: {len(xs_aug)}")
```

## Neural Network Training Integration

### PyTorch Dataset with Augmentation

```python
import torch
from torch.utils.data import Dataset, DataLoader

class EventDataset(Dataset):
    """Event dataset with augmentation"""

    def __init__(self, event_files, augmentation_config=None,
                 representation_params=None):
        self.event_files = event_files
        self.augmenter = EventAugmentationPipeline(augmentation_config) if augmentation_config else None
        self.repr_params = representation_params or {'bins': 5, 'width': 640, 'height': 480}

    def __len__(self):
        return len(self.event_files)

    def __getitem__(self, idx):
        # Load events
        event_file = self.event_files[idx]
        xs, ys, ts, ps = evlib.formats.load_events(event_file)

        # Apply augmentation
        if self.augmenter:
            xs, ys, ts, ps = self.augmenter(xs, ys, ts, ps)

        # Create representation
        voxel_grid_data, voxel_grid_shape = evlib.representations.events_to_smooth_voxel_grid(
            xs, ys, ts, ps,
            self.repr_params['width'],
            self.repr_params['height'],
            self.repr_params['bins']
        )

        # Normalize
        voxel_grid = (voxel_grid - voxel_grid.mean()) / (voxel_grid.std() + 1e-8)

        return torch.from_numpy(voxel_grid).float()

# Create dataset with augmentation
train_files = ["event1.txt", "event2.txt", "event3.txt"]
train_dataset = EventDataset(train_files, augmentation_config)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Training loop with augmentation
for batch_idx, voxel_batch in enumerate(train_loader):
    # voxel_batch contains augmented data
    # ... training code ...
    pass
```

## Performance Optimization

### Efficient Augmentation

```python
def efficient_augmentation(xs, ys, ts, ps, augment_params):
    """Optimized augmentation for large datasets"""

    # Pre-compute common values
    width, height = 640, 480
    n_events = len(xs)

    # Batch spatial transformations
    if augment_params.get('spatial_transforms', False):
        # Apply all spatial transforms in one pass
        transform_matrix = np.eye(3)

        # Rotation
        if augment_params.get('rotation', False):
            angle = np.random.uniform(-augment_params['max_rotation'],
                                    augment_params['max_rotation'])
            cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))
            rotation_matrix = np.array([[cos_a, -sin_a, 0],
                                      [sin_a, cos_a, 0],
                                      [0, 0, 1]])
            transform_matrix = transform_matrix @ rotation_matrix

        # Scale
        if augment_params.get('scale', False):
            scale = np.random.uniform(*augment_params['scale_range'])
            scale_matrix = np.array([[scale, 0, 0],
                                   [0, scale, 0],
                                   [0, 0, 1]])
            transform_matrix = transform_matrix @ scale_matrix

        # Apply transformation
        coords = np.vstack([xs, ys, np.ones(n_events)])
        transformed = transform_matrix @ coords

        xs_new = transformed[0]
        ys_new = transformed[1]

        # Filter valid events
        valid_mask = (
            (xs_new >= 0) & (xs_new < width) &
            (ys_new >= 0) & (ys_new < height)
        )

        xs, ys, ts, ps = (
            xs_new[valid_mask].astype(np.uint16),
            ys_new[valid_mask].astype(np.uint16),
            ts[valid_mask],
            ps[valid_mask]
        )

    return xs, ys, ts, ps
```

### Memory-Efficient Processing

```python
def memory_efficient_augmentation(event_file, augment_config, chunk_size=100000):
    """Process large event files in chunks"""

    # Load events in chunks
    all_augmented = []

    for chunk_xs, chunk_ys, chunk_ts, chunk_ps in evlib.formats.load_events_chunked(
        event_file, chunk_size=chunk_size
    ):
        # Apply augmentation to chunk
        augmenter = EventAugmentationPipeline(augment_config)
        aug_xs, aug_ys, aug_ts, aug_ps = augmenter(chunk_xs, chunk_ys, chunk_ts, chunk_ps)

        all_augmented.append((aug_xs, aug_ys, aug_ts, aug_ps))

    # Combine chunks
    xs_final = np.concatenate([chunk[0] for chunk in all_augmented])
    ys_final = np.concatenate([chunk[1] for chunk in all_augmented])
    ts_final = np.concatenate([chunk[2] for chunk in all_augmented])
    ps_final = np.concatenate([chunk[3] for chunk in all_augmented])

    return xs_final, ys_final, ts_final, ps_final
```

## Best Practices

### 1. Augmentation Selection

```python
# Training augmentation (aggressive)
train_config = {
    'horizontal_flip': True,
    'rotation': True,
    'scale_crop': True,
    'temporal_scaling': True,
    'add_noise': True,
    'event_dropout': True,
    'polarity_flip': True
}

# Validation augmentation (conservative)
val_config = {
    'horizontal_flip': False,
    'rotation': False,
    'temporal_scaling': False,
    'add_noise': False
}

# Test augmentation (none)
test_config = {}
```

### 2. Parameter Tuning

```python
# Guidelines for augmentation parameters
AUGMENTATION_GUIDELINES = {
    'horizontal_flip': {'probability': 0.5},  # 50% chance
    'rotation': {'max_angle': 15},  # ±15 degrees
    'scale_crop': {'scale_range': (0.8, 1.2)},  # 80%-120% scaling
    'temporal_scaling': {'scale_range': (0.9, 1.1)},  # 90%-110% temporal
    'noise_rate': 0.05,  # 5% noise events
    'dropout_rate': 0.05,  # 5% event dropout
    'jitter_std': 0.001,  # 1ms temporal jitter
}
```

### 3. Validation

```python
def validate_augmentation(xs_orig, ys_orig, ts_orig, ps_orig,
                         xs_aug, ys_aug, ts_aug, ps_aug):
    """Validate augmentation quality"""

    # Check event count changes
    orig_count = len(xs_orig)
    aug_count = len(xs_aug)
    count_ratio = aug_count / orig_count

    # Check temporal ordering
    temporal_ordered = np.all(ts_aug[:-1] <= ts_aug[1:])

    # Check coordinate bounds
    coords_valid = (
        np.all(xs_aug >= 0) & np.all(xs_aug < 640) &
        np.all(ys_aug >= 0) & np.all(ys_aug < 480)
    )

    # Check polarity values
    polarity_valid = np.all(np.isin(ps_aug, [-1, 1]))

    print(f"Event count ratio: {count_ratio:.3f}")
    print(f"Temporal ordering: {temporal_ordered}")
    print(f"Coordinates valid: {coords_valid}")
    print(f"Polarity valid: {polarity_valid}")

    return count_ratio, temporal_ordered, coords_valid, polarity_valid
```

## Next Steps

- [Neural Networks](models.md): Use augmented data for training
- [Visualization](visualization.md): Visualize augmentation effects
- [API Reference](../api/augmentation.md): Detailed augmentation functions
