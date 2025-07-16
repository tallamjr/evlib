#!/usr/bin/env python3
"""
Showcasing evlib Reader Capabilities with Different Data Formats

This script demonstrates various ways to read event data using evlib
across different formats and datasets available in the data directory.
"""

import sys
from pathlib import Path
import numpy as np

# Add the project root to the path so we can import evlib
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import evlib


def read_large_hdf5_file():
    """Example 1: Reading a large HDF5 file from the original dataset."""
    print("=" * 60)
    print("EXAMPLE 1: Reading Large HDF5 File (Original Dataset)")
    print("=" * 60)

    file_path = "data/original/front/seq01.h5"
    print(f"Loading: {file_path}")

    # Load events
    events = evlib.load_events(file_path)
    timestamps, x_coords, y_coords, polarities = events

    print(f"✓ Loaded {len(timestamps):,} events")
    print(f"  Duration: {timestamps.max() - timestamps.min():.2f} seconds")
    print(f"  Event rate: {len(timestamps)/(timestamps.max() - timestamps.min()):,.0f} events/sec")
    print(f"  Spatial extent: {x_coords.max() - x_coords.min() + 1} x {y_coords.max() - y_coords.min() + 1}")
    print(f"  Polarity split: {np.bincount(polarities)}")

    return events


def read_etram_night_sequence():
    """Example 2: Reading eTram nighttime sequence (HDF5 format)."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Reading eTram Nighttime Sequence (HDF5)")
    print("=" * 60)

    file_path = "data/eTram/h5/val_2/val_night_007_td.h5"
    print(f"Loading: {file_path}")

    events = evlib.load_events(file_path)
    timestamps, x_coords, y_coords, polarities = events

    print(f"✓ Loaded {len(timestamps):,} events")
    print(f"  Time span: {timestamps.min():.3f} to {timestamps.max():.3f} seconds")
    print(
        f"  Coordinate ranges: x=[{x_coords.min():.0f}-{x_coords.max():.0f}], y=[{y_coords.min():.0f}-{y_coords.max():.0f}]"
    )
    print(f"  Polarity distribution: {np.bincount(polarities)}")

    # Sample a subset for analysis
    sample_size = 100000
    indices = np.random.choice(len(timestamps), sample_size, replace=False)
    sample_events = (timestamps[indices], x_coords[indices], y_coords[indices], polarities[indices])

    print(f"✓ Created random sample of {sample_size:,} events")
    return sample_events


def read_prophesee_raw_file():
    """Example 3: Reading Prophesee EVT2 raw binary file."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Reading Prophesee EVT2 Raw Binary")
    print("=" * 60)

    file_path = "data/eTram/raw/val_2/val_night_008.raw"
    print(f"Loading: {file_path}")

    events = evlib.load_events(file_path)
    timestamps, x_coords, y_coords, polarities = events

    print(f"✓ Loaded {len(timestamps):,} events from raw binary")
    print("  File format: EVT2 (Prophesee)")
    print(f"  Recording duration: {timestamps.max() - timestamps.min():.2f} seconds")
    print(f"  Sensor resolution inferred: {x_coords.max() + 1} x {y_coords.max() + 1}")
    print(f"  Polarity encoding: {np.unique(polarities)} (likely -1/+1)")

    return events


def read_davis_text_format():
    """Example 4: Reading DAVIS text format from slider_depth dataset."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Reading DAVIS Text Format")
    print("=" * 60)

    file_path = "data/slider_depth/events.txt"
    print(f"Loading: {file_path}")

    events = evlib.load_events(file_path)
    timestamps, x_coords, y_coords, polarities = events

    print(f"✓ Loaded {len(timestamps):,} events from text format")
    print("  Format: Space-separated text (timestamp x y polarity)")
    print("  Camera: DAVIS (346x240 resolution)")
    print("  Scene: Object motion with depth information")
    print(f"  Polarity encoding: {np.unique(polarities)} (0/1 format)")

    return events


def read_multiple_sequences():
    """Example 5: Reading multiple sequences and comparing statistics."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Comparing Multiple Sequences")
    print("=" * 60)

    sequences = [
        ("eTram Night 007", "data/eTram/h5/val_2/val_night_007_td.h5"),
        ("eTram Night 008", "data/eTram/h5/val_2/val_night_008_td.h5"),
        ("eTram Night 009", "data/eTram/h5/val_2/val_night_009_td.h5"),
        ("eTram Night 010", "data/eTram/h5/val_2/val_night_010_td.h5"),
        ("eTram Night 011", "data/eTram/h5/val_2/val_night_011_td.h5"),
    ]

    stats = []
    for name, file_path in sequences:
        if Path(file_path).exists():
            print(f"\nLoading: {name}")
            events = evlib.load_events(file_path)
            timestamps, x_coords, y_coords, polarities = events

            duration = timestamps.max() - timestamps.min()
            event_rate = len(timestamps) / duration

            stat = {
                "name": name,
                "events": len(timestamps),
                "duration": duration,
                "rate": event_rate,
                "pos_events": np.sum(polarities == 1),
                "neg_events": np.sum(polarities == 0),
            }
            stats.append(stat)

            print(f"  Events: {stat['events']:,}")
            print(f"  Duration: {stat['duration']:.2f}s")
            print(f"  Rate: {stat['rate']:,.0f} events/sec")
            print(f"  Pos/Neg: {stat['pos_events']:,}/{stat['neg_events']:,}")

    if stats:
        print(
            f"\n{'Sequence':<20} {'Events':<12} {'Duration':<10} {'Rate (evt/s)':<12} {'Pos/Neg Ratio':<12}"
        )
        print("-" * 75)
        for stat in stats:
            ratio = stat["pos_events"] / stat["neg_events"] if stat["neg_events"] > 0 else 0
            print(
                f"{stat['name']:<20} {stat['events']:<12,} {stat['duration']:<10.2f} {stat['rate']:<12,.0f} {ratio:<12.2f}"
            )

    return stats


def read_gen4_preprocessed_data():
    """Example 6: Reading Gen4 preprocessed data with labels."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Reading Gen4 Preprocessed Data")
    print("=" * 60)

    # Look for available Gen4 sequences
    gen4_dir = Path("data/gen4/test")
    if gen4_dir.exists():
        sequences = list(gen4_dir.glob("*/"))[:3]  # Take first 3 sequences

        for seq_dir in sequences:
            print(f"\nProcessing: {seq_dir.name}")

            # Check for event representations
            repr_dir = seq_dir / "event_representations_v2"
            if repr_dir.exists():
                print(f"  Found event representations in: {repr_dir}")
                # List available representation files
                repr_files = list(repr_dir.glob("*.npy"))
                for repr_file in repr_files[:3]:  # Show first 3
                    print(f"    - {repr_file.name}")

            # Check for labels
            labels_dir = seq_dir / "labels_v2"
            if labels_dir.exists():
                labels_file = labels_dir / "labels.npz"
                timestamps_file = labels_dir / "timestamps_us.npy"

                if labels_file.exists() and timestamps_file.exists():
                    print(f"  Found labels: {labels_file}")
                    print(f"  Found timestamps: {timestamps_file}")

                    # Load and examine labels
                    labels = np.load(labels_file)
                    timestamps = np.load(timestamps_file)

                    print(f"    Label arrays: {list(labels.keys())}")
                    print(f"    Timestamp shape: {timestamps.shape}")

                    # Show some label statistics
                    if "labels" in labels:
                        label_data = labels["labels"]
                        print(f"    Label data shape: {label_data.shape}")
                        print(f"    Unique labels: {np.unique(label_data)}")
    else:
        print("Gen4 test directory not found")


def read_with_format_detection():
    """Example 7: Using format detection before reading."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Format Detection and Reading")
    print("=" * 60)

    test_files = [
        "data/slider_depth/events.txt",
        "data/eTram/h5/val_2/val_night_011_td.h5",
        "data/eTram/raw/val_2/val_night_011.raw",
        "data/original/front/seq02.h5",
    ]

    for file_path in test_files:
        if Path(file_path).exists():
            print(f"\nAnalyzing: {file_path}")

            # Detect format
            detected_format = evlib.detect_format(file_path)
            print(f"  Detected format: {detected_format}")

            # Get format description
            description = evlib.get_format_description(detected_format)
            print(f"  Description: {description}")

            # Load with detected format
            events = evlib.load_events(file_path)
            timestamps, x_coords, y_coords, polarities = events

            print(f"  ✓ Loaded {len(timestamps):,} events")
            print(f"  Event rate: {len(timestamps)/(timestamps.max() - timestamps.min()):,.0f} events/sec")
        else:
            print(f"\nFile not found: {file_path}")


def read_temporal_chunks():
    """Example 8: Reading data in temporal chunks."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Temporal Chunking for Large Files")
    print("=" * 60)

    file_path = "data/original/front/seq01.h5"
    if Path(file_path).exists():
        print(f"Loading entire file: {file_path}")

        # Load full dataset
        events = evlib.load_events(file_path)
        timestamps, x_coords, y_coords, polarities = events

        print(f"✓ Full dataset: {len(timestamps):,} events")
        print(f"  Time range: {timestamps.min():.3f} to {timestamps.max():.3f} seconds")

        # Create temporal chunks
        chunk_duration = 10.0  # 10 second chunks
        start_time = timestamps.min()
        end_time = timestamps.max()

        print(f"\nCreating {chunk_duration}s temporal chunks:")

        chunk_count = 0
        current_time = start_time

        while current_time < end_time and chunk_count < 5:  # Limit to 5 chunks for demo
            chunk_end = min(current_time + chunk_duration, end_time)

            # Create mask for this time chunk
            chunk_mask = (timestamps >= current_time) & (timestamps < chunk_end)

            chunk_events = np.sum(chunk_mask)
            if chunk_events > 0:
                chunk_rate = chunk_events / (chunk_end - current_time)
                print(
                    f"  Chunk {chunk_count}: {current_time:.1f}-{chunk_end:.1f}s → {chunk_events:,} events ({chunk_rate:,.0f} evt/s)"
                )

                # Example: Extract chunk data
                chunk_timestamps = timestamps[chunk_mask]
                chunk_x = x_coords[chunk_mask]
                chunk_y = y_coords[chunk_mask]
                chunk_pol = polarities[chunk_mask]

                # Could process this chunk separately...

            current_time = chunk_end
            chunk_count += 1
    else:
        print("Large file not found for chunking example")


def main():
    """Main function running all reader examples."""
    print("EVLIB Reader Capabilities Showcase")
    print("=" * 60)
    print("Demonstrating various ways to read event data with evlib")

    try:
        # Run all examples
        read_large_hdf5_file()
        read_etram_night_sequence()
        read_prophesee_raw_file()
        read_davis_text_format()
        read_multiple_sequences()
        read_gen4_preprocessed_data()
        read_with_format_detection()
        read_temporal_chunks()

        print("\n" + "=" * 60)
        print("✓ All reader examples completed successfully!")
        print("\nKey takeaways:")
        print("• evlib can read HDF5, EVT2 raw, and text formats seamlessly")
        print("• All formats return the same tuple structure: (timestamps, x, y, polarity)")
        print("• Format detection works automatically")
        print("• Large files can be processed efficiently")
        print("• Different datasets have different characteristics and encodings")

    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
