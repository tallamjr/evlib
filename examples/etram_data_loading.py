#!/usr/bin/env python3
"""
Example: Loading eTram Event Data and Converting to NumPy/PyTorch

This script demonstrates how to load eTram event data in different formats
(HDF5, EVT2 raw, text) and convert to numpy arrays and PyTorch tensors.

The eTram dataset contains event camera data with these formats:
- HDF5 files (.h5) - structured format with multiple datasets
- EVT2 raw files (.raw) - binary format from Prophesee cameras
- Text files (.txt) - space-separated format: timestamp x y polarity

Usage:
    python etram_data_loading.py
"""

import sys
from pathlib import Path

import numpy as np

# Add the project root to the path so we can import evlib
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import evlib

    print("evlib imported successfully")
    print(f"Available functions: {[name for name in dir(evlib) if not name.startswith('_')]}")
except ImportError as e:
    print(f"Error importing evlib: {e}")
    print("Make sure you have built evlib with: maturin develop")
    sys.exit(1)

# Optional: PyTorch import for tensor conversion
try:
    import torch

    TORCH_AVAILABLE = True
    print(f"PyTorch version: {torch.__version__}")
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available - tensor conversion examples will be skipped")


def load_events_safely(file_path):
    """Load events from file using evlib with error handling."""
    try:
        events = evlib.load_events(str(file_path))
        print(f"Events type: {type(events)}")
        if hasattr(events, "shape"):
            print(f"Events shape: {events.shape}")
        if hasattr(events, "dtype"):
            print(f"Events dtype: {events.dtype}")

        # If it's a tuple, let's see what's inside
        if isinstance(events, tuple):
            print("Tuple contents:")
            for i, item in enumerate(events):
                print(
                    f"  [{i}]: {type(item)} - {item.shape if hasattr(item, 'shape') else len(item) if hasattr(item, '__len__') else item}"
                )

        return events
    except Exception as e:
        print(f"Error loading file: {e}")
        return None


def process_events_data(events, description=""):
    """Process events data and print information."""
    if events is None:
        return None

    print(f"\n=== Processing {description} ===")

    # Handle different data formats
    if isinstance(events, tuple) and len(events) == 4:
        # It looks like evlib returns a tuple of (timestamps, x, y, polarity)
        timestamps, x_coords, y_coords, polarities = events
        print("Tuple format detected:")
        print(f"  Timestamps: {type(timestamps)} shape={timestamps.shape}")
        print(f"  X coordinates: {type(x_coords)} shape={x_coords.shape}")
        print(f"  Y coordinates: {type(y_coords)} shape={y_coords.shape}")
        print(f"  Polarities: {type(polarities)} shape={polarities.shape}")

        # Print statistics
        print(f"  Event count: {len(timestamps)}")
        print(f"  Timestamp range: {timestamps.min():.6f} - {timestamps.max():.6f}")
        print(
            f"  Spatial range: x=[{x_coords.min()}-{x_coords.max()}], y=[{y_coords.min()}-{y_coords.max()}]"
        )
        try:
            print(f"  Polarity distribution: {np.bincount(polarities)}")
        except ValueError:
            print(f"  Polarity range: [{polarities.min()}, {polarities.max()}]")
            print(f"  Polarity unique values: {np.unique(polarities)}")

    elif isinstance(events, np.ndarray):
        print(f"Data is numpy array with shape: {events.shape}")

        # Check if it's a structured array
        if events.dtype.names:
            print(f"Structured array with fields: {events.dtype.names}")
            if "timestamp" in events.dtype.names:
                print(f"Timestamp range: {events['timestamp'].min():.6f} - {events['timestamp'].max():.6f}")
            if "x" in events.dtype.names and "y" in events.dtype.names:
                print(
                    f"Spatial range: x=[{events['x'].min()}-{events['x'].max()}], y=[{events['y'].min()}-{events['y'].max()}]"
                )
            if "polarity" in events.dtype.names:
                print(f"Polarity distribution: {np.bincount(events['polarity'])}")
        else:
            print(f"Regular array with shape: {events.shape}")
            print(f"Data range: [{events.min():.6f}, {events.max():.6f}]")
    else:
        print(f"Data type: {type(events)}")
        try:
            print(f"Length: {len(events)}")
        except (TypeError, AttributeError):
            print("Cannot determine length")

    return events


def create_voxel_grid_from_events(events, width=1280, height=720, time_bins=5):
    """Create a voxel grid representation from event data."""
    print("\n=== Creating voxel grid representation ===")

    if events is None:
        print("No events data provided")
        return None

    try:
        # Extract data based on the format
        if isinstance(events, tuple) and len(events) == 4:
            # Tuple format (timestamps, x, y, polarity)
            timestamps, x_coords, y_coords, polarities = events
        elif isinstance(events, np.ndarray) and events.dtype.names:
            # Structured array
            timestamps = events["timestamp"]
            x_coords = events["x"]
            y_coords = events["y"]
            polarities = events["polarity"]
        elif isinstance(events, np.ndarray) and events.shape[1] == 4:
            # Regular array with 4 columns
            timestamps = events[:, 0]
            x_coords = events[:, 1].astype(int)
            y_coords = events[:, 2].astype(int)
            polarities = events[:, 3].astype(int)
        else:
            print(
                f"Unsupported event format: {type(events)}, shape: {events.shape if hasattr(events, 'shape') else 'No shape'}"
            )
            return None

        # Clean up coordinates - ensure they're within sensor bounds
        x_coords = x_coords.astype(np.int32)
        y_coords = y_coords.astype(np.int32)

        # Filter out invalid coordinates
        valid_mask = (x_coords >= 0) & (x_coords < width) & (y_coords >= 0) & (y_coords < height)
        if not np.all(valid_mask):
            print(f"  Filtering out {np.sum(~valid_mask)} invalid coordinates")
            x_coords = x_coords[valid_mask]
            y_coords = y_coords[valid_mask]
            timestamps = timestamps[valid_mask]
            polarities = polarities[valid_mask]

        # Use evlib's voxel grid creation
        voxel_grid = evlib.create_voxel_grid(
            x_coords, y_coords, timestamps, polarities, sensor_resolution=(width, height), num_bins=time_bins
        )

        print(f"Created voxel grid with shape: {voxel_grid.shape}")
        print(f"Voxel grid range: [{voxel_grid.min():.3f}, {voxel_grid.max():.3f}]")

        return voxel_grid

    except Exception as e:
        print(f"Error creating voxel grid: {e}")
        return None


def convert_to_pytorch_tensor(numpy_array):
    """Convert numpy array to PyTorch tensor."""
    if not TORCH_AVAILABLE:
        print("PyTorch not available - skipping tensor conversion")
        return None

    if numpy_array is None:
        print("No numpy array provided")
        return None

    print("\n=== Converting to PyTorch tensor ===")

    try:
        # Convert to PyTorch tensor
        tensor = torch.from_numpy(numpy_array)

        # For voxel grids, you might want to ensure float32 type
        if tensor.dtype == torch.float64:
            tensor = tensor.float()

        print(f"Created PyTorch tensor with shape: {tensor.shape}")
        print(f"Tensor dtype: {tensor.dtype}")
        print(f"Tensor device: {tensor.device}")

        # If you have a GPU available, you can move to GPU
        if torch.cuda.is_available():
            tensor_gpu = tensor.cuda()
            print(f"Moved tensor to GPU: {tensor_gpu.device}")
            return tensor_gpu
        else:
            print("GPU not available, keeping tensor on CPU")
            return tensor

    except Exception as e:
        print(f"Error converting to PyTorch tensor: {e}")
        return None


def main():
    """Main function demonstrating eTram data loading and conversion."""

    # Define data paths
    data_dir = Path("data")

    print("eTram Event Data Loading and Conversion Example")
    print("=" * 50)

    # Test different data files
    test_files = [
        ("HDF5 (small)", data_dir / "eTram/h5/val_2/val_night_011_td.h5"),
        ("EVT2 raw (small)", data_dir / "eTram/raw/val_2/val_night_011.raw"),
        ("Text format", data_dir / "slider_depth/events.txt"),
        ("HDF5 (large)", data_dir / "eTram/h5/val_2/val_night_007_td.h5"),
        ("HDF5 (original)", data_dir / "original/front/seq01.h5"),
    ]

    for description, file_path in test_files:
        print(f"\n{'='*60}")
        print(f"Testing: {description}")
        print(f"File: {file_path}")
        print(f"Exists: {file_path.exists()}")

        if not file_path.exists():
            print("File not found, skipping...")
            continue

        # Load events
        events = load_events_safely(file_path)

        if events is not None:
            # Process and analyze events
            processed_events = process_events_data(events, description)

            # Create voxel grid
            if processed_events is not None:
                # Determine appropriate dimensions
                if "346x240" in str(file_path) or "slider_depth" in str(file_path):
                    width, height = 346, 240
                else:
                    width, height = 1280, 720

                voxel_grid = create_voxel_grid_from_events(processed_events, width=width, height=height)

                if voxel_grid is not None:
                    # Convert to PyTorch tensor
                    tensor = convert_to_pytorch_tensor(voxel_grid)

                    if tensor is not None:
                        print(f"SUCCESS: Successfully created tensor for {description}")
                        print(f"  Shape: {tensor.shape}")
                        print(f"  Device: {tensor.device}")
                        print("  Ready for neural network processing!")

    print("\n" + "=" * 60)
    print("Data loading example completed!")
    print("\nNext steps:")
    print("1. Use the numpy arrays for data analysis")
    print("2. Use PyTorch tensors for neural network training")
    print("3. Apply spatial/temporal filters using evlib")
    print("4. Create custom representations for your specific use case")
    print("\nExample usage in your code:")
    print("```python")
    print("import evlib")
    print("events = evlib.load_events('path/to/your/file.h5')")
    print(
        "voxel_grid = evlib.create_voxel_grid(events['timestamp'], events['x'], events['y'], events['polarity'])"
    )
    print("tensor = torch.from_numpy(voxel_grid)")
    print("```")


if __name__ == "__main__":
    main()
