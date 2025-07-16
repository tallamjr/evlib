#!/usr/bin/env python3
"""
Simple example showing how to load eTram data and work with it.

This is the simplest possible example of how to use evlib with eTram data.
"""

import numpy as np
import evlib

# Optional: PyTorch for tensor conversion
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def simple_etram_example():
    """Simple example of loading eTram data and creating representations."""

    # Load events from any supported format
    file_path = "data/slider_depth/events.txt"  # or any .h5 or .raw file

    print(f"Loading events from: {file_path}")

    # Load events - returns tuple of (timestamps, x, y, polarity)
    events = evlib.load_events(file_path)
    timestamps, x_coords, y_coords, polarities = events

    print(f"Loaded {len(timestamps)} events")
    print(f"Time range: {timestamps.min():.3f} - {timestamps.max():.3f} seconds")
    print(f"Spatial range: x=[{x_coords.min()}-{x_coords.max()}], y=[{y_coords.min()}-{y_coords.max()}]")

    # Create voxel grid representation
    sensor_resolution = (346, 240)  # width, height for this dataset
    time_bins = 5

    # Clean coordinates to ensure they're within bounds
    x_coords = x_coords.astype(np.int32)
    y_coords = y_coords.astype(np.int32)

    # Filter valid coordinates
    valid_mask = (
        (x_coords >= 0)
        & (x_coords < sensor_resolution[0])
        & (y_coords >= 0)
        & (y_coords < sensor_resolution[1])
    )

    x_coords = x_coords[valid_mask]
    y_coords = y_coords[valid_mask]
    timestamps = timestamps[valid_mask]
    polarities = polarities[valid_mask]

    print(f"After filtering: {len(timestamps)} valid events")

    # Create voxel grid
    voxel_grid = evlib.create_voxel_grid(
        x_coords, y_coords, timestamps, polarities, sensor_resolution=sensor_resolution, num_bins=time_bins
    )

    print(f"Created voxel grid with shape: {voxel_grid.shape}")
    print(f"Voxel grid range: [{voxel_grid.min():.3f}, {voxel_grid.max():.3f}]")

    # Convert to PyTorch tensor (if available)
    if TORCH_AVAILABLE:
        tensor = torch.from_numpy(voxel_grid).float()
        print(f"Created PyTorch tensor with shape: {tensor.shape}")

        # Example: Add batch dimension for neural network
        tensor_batch = tensor.unsqueeze(0)  # Add batch dimension
        print(f"With batch dimension: {tensor_batch.shape}")

        # Example: Permute dimensions for CNN (channels first)
        tensor_cnn = tensor_batch.permute(0, 3, 1, 2)  # (batch, channels, height, width)
        print(f"CNN format: {tensor_cnn.shape}")

        return tensor_cnn

    return voxel_grid


if __name__ == "__main__":
    result = simple_etram_example()
    print(f"\nFinal result shape: {result.shape}")
    print("✓ Successfully loaded eTram data and created tensor representation!")
