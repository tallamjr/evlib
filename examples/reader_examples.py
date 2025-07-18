#!/usr/bin/env python3
"""
Simple Reader Examples - Different Ways to Load Event Data

This script shows various examples of using evlib to read different
types of event data files in your dataset.
"""

import sys
from pathlib import Path
import numpy as np

# Add the project root to the path so we can import evlib
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import evlib  # noqa: E402


def example_1_large_hdf5():
    """Example 1: Reading a large HDF5 file (288M events)."""
    print("EXAMPLE 1: Large HDF5 File")
    print("-" * 40)

    # Load a large HDF5 file from the original dataset
    events = evlib.load_events("data/original/front/seq01.h5")
    timestamps, x_coords, y_coords, polarities = events

    print(f"SUCCESS: Loaded {len(timestamps):,} events")
    print(f"  Duration: {timestamps.max():.1f} seconds")
    print(f"  Event rate: {len(timestamps)/timestamps.max():,.0f} events/sec")
    print(f"  Polarity split: {np.bincount(polarities)}")

    # Take a 1% sample for further processing
    sample_indices = np.random.choice(len(timestamps), len(timestamps) // 100, replace=False)
    sample_events = (
        timestamps[sample_indices],
        x_coords[sample_indices],
        y_coords[sample_indices],
        polarities[sample_indices],
    )
    print(f"  Created 1% sample: {len(sample_events[0]):,} events")
    return sample_events


def example_2_etram_hdf5():
    """Example 2: Reading eTram HDF5 format."""
    print("\nEXAMPLE 2: eTram HDF5 Format")
    print("-" * 40)

    # Load eTram HDF5 file
    events = evlib.load_events("data/eTram/h5/val_2/val_night_011_td.h5")
    timestamps, x_coords, y_coords, polarities = events

    print(f"SUCCESS: Loaded {len(timestamps):,} events")
    print("  Dataset: eTram validation night sequence")
    print(f"  Recording: {timestamps.max():.1f} seconds")
    print(
        f"  Spatial range: x=[{x_coords.min():.0f}-{x_coords.max():.0f}], y=[{y_coords.min():.0f}-{y_coords.max():.0f}]"
    )
    print(f"  Polarity distribution: {np.bincount(polarities)}")

    return events


def example_3_prophesee_raw():
    """Example 3: Reading Prophesee EVT2 raw binary."""
    print("\nEXAMPLE 3: Prophesee EVT2 Raw Binary")
    print("-" * 40)

    # Load EVT2 raw file
    events = evlib.load_events("data/eTram/raw/val_2/val_night_011.raw")
    timestamps, x_coords, y_coords, polarities = events

    print(f"SUCCESS: Loaded {len(timestamps):,} events")
    print("  Format: Prophesee EVT2 raw binary")
    print(f"  Duration: {timestamps.max():.1f} seconds")
    print(f"  Polarity encoding: {np.unique(polarities)} (signed)")
    print(
        f"  Coordinate ranges: x=[{x_coords.min():.0f}-{x_coords.max():.0f}], y=[{y_coords.min():.0f}-{y_coords.max():.0f}]"
    )

    return events


def example_4_text_format():
    """Example 4: Reading text format (DAVIS)."""
    print("\nEXAMPLE 4: Text Format (DAVIS)")
    print("-" * 40)

    # Load text format file
    events = evlib.load_events("data/slider_depth/events.txt")
    timestamps, x_coords, y_coords, polarities = events

    print(f"SUCCESS: Loaded {len(timestamps):,} events")
    print("  Format: Space-separated text")
    print("  Camera: DAVIS (346x240)")
    print("  Scene: Sliding objects with depth")
    print(f"  Polarity encoding: {np.unique(polarities)} (unsigned)")

    return events


def example_5_compare_datasets():
    """Example 5: Compare different datasets."""
    print("\nEXAMPLE 5: Dataset Comparison")
    print("-" * 40)

    datasets = [
        ("DAVIS Text", "data/slider_depth/events.txt"),
        ("eTram HDF5", "data/eTram/h5/val_2/val_night_011_td.h5"),
        ("eTram Raw", "data/eTram/raw/val_2/val_night_011.raw"),
        ("Original HDF5", "data/original/front/seq02.h5"),
    ]

    for name, path in datasets:
        if Path(path).exists():
            events = evlib.load_events(path)
            timestamps, x_coords, y_coords, polarities = events

            duration = timestamps.max() - timestamps.min()
            rate = len(timestamps) / duration if duration > 0 else 0

            print(f"{name}:")
            print(f"  Events: {len(timestamps):,}")
            print(f"  Duration: {duration:.1f}s")
            print(f"  Rate: {rate:,.0f} events/sec")
            print(f"  Polarity: {np.unique(polarities)}")
            print()


def example_6_temporal_slicing():
    """Example 6: Temporal slicing of data."""
    print("\nEXAMPLE 6: Temporal Slicing")
    print("-" * 40)

    # Load data
    events = evlib.load_events("data/slider_depth/events.txt")
    timestamps, x_coords, y_coords, polarities = events

    print(f"Full dataset: {len(timestamps):,} events over {timestamps.max():.1f}s")

    # Create 10-second slices
    slice_duration = 10.0
    start_time = timestamps.min()
    end_time = timestamps.max()

    print(f"Creating {slice_duration}s temporal slices:")

    current_time = start_time
    slice_num = 0

    while current_time < end_time and slice_num < 5:  # Limit to 5 slices
        slice_end = min(current_time + slice_duration, end_time)

        # Create mask for this time slice
        mask = (timestamps >= current_time) & (timestamps < slice_end)

        slice_events = np.sum(mask)
        if slice_events > 0:
            slice_rate = slice_events / (slice_end - current_time)
            print(
                f"  Slice {slice_num}: {current_time:.1f}-{slice_end:.1f}s → {slice_events:,} events ({slice_rate:,.0f} evt/s)"
            )

            # Extract slice data
            _slice_timestamps = timestamps[mask]
            _slice_x = x_coords[mask]
            _slice_y = y_coords[mask]
            _slice_pol = polarities[mask]

            # Could process this slice separately...

        current_time = slice_end
        slice_num += 1


def example_7_format_detection():
    """Example 7: Format detection."""
    print("\nEXAMPLE 7: Format Detection")
    print("-" * 40)

    test_files = [
        "data/slider_depth/events.txt",
        "data/eTram/h5/val_2/val_night_011_td.h5",
        "data/eTram/raw/val_2/val_night_011.raw",
    ]

    for file_path in test_files:
        if Path(file_path).exists():
            print(f"File: {Path(file_path).name}")

            # Detect format
            format_info = evlib.detect_format(file_path)
            print(f"  Detected format: {format_info}")

            # Load and show basic info
            events = evlib.load_events(file_path)
            timestamps, x_coords, y_coords, polarities = events
            print(f"  Events loaded: {len(timestamps):,}")
            print()


def example_8_memory_efficient():
    """Example 8: Memory-efficient processing."""
    print("\nEXAMPLE 8: Memory-Efficient Processing")
    print("-" * 40)

    # Load large file
    file_path = "data/original/front/seq01.h5"
    if Path(file_path).exists():
        events = evlib.load_events(file_path)
        timestamps, x_coords, y_coords, polarities = events

        print(f"Large dataset: {len(timestamps):,} events")

        # Process in chunks to save memory
        chunk_size = 1000000  # 1M events per chunk
        num_chunks = len(timestamps) // chunk_size + 1

        print(f"Processing in {num_chunks} chunks of {chunk_size:,} events each:")

        for i in range(min(3, num_chunks)):  # Show first 3 chunks
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(timestamps))

            chunk_timestamps = timestamps[start_idx:end_idx]
            _chunk_x = x_coords[start_idx:end_idx]
            _chunk_y = y_coords[start_idx:end_idx]
            _chunk_pol = polarities[start_idx:end_idx]

            chunk_duration = chunk_timestamps.max() - chunk_timestamps.min()
            chunk_rate = len(chunk_timestamps) / chunk_duration if chunk_duration > 0 else 0

            print(f"  Chunk {i}: {len(chunk_timestamps):,} events, {chunk_rate:,.0f} events/sec")

            # Process chunk here...
            # del chunk_timestamps, chunk_x, chunk_y, chunk_pol  # Free memory
    else:
        print("Large file not available for memory example")


def main():
    """Run all reader examples."""
    print("EVLIB Reader Examples")
    print("=" * 50)

    try:
        example_1_large_hdf5()
        example_2_etram_hdf5()
        example_3_prophesee_raw()
        example_4_text_format()
        example_5_compare_datasets()
        example_6_temporal_slicing()
        example_7_format_detection()
        example_8_memory_efficient()

        print("\n" + "=" * 50)
        print("SUCCESS: All examples completed!")
        print("\nKey Points:")
        print("• evlib handles HDF5, EVT2 raw, and text formats seamlessly")
        print("• All formats return: (timestamps, x, y, polarity)")
        print("• Different datasets use different polarity encodings")
        print("• Large files can be processed efficiently")
        print("• Format detection works automatically")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
