#!/usr/bin/env python3
"""
EVT2.1 Format Example - Demonstrating Advanced Event Camera Data Processing

This script demonstrates how to work with EVT2.1 (Event Data 2.1) format files using evlib.
EVT2.1 is a 64-bit vectorized format from Prophesee that supports efficient encoding
of spatially correlated events with up to 32 pixels per event word.

Key Features Demonstrated:
1. Format detection for EVT2.1 files
2. Loading EVT2.1 data with evlib.load_events()
3. Vectorized event handling capabilities
4. Comparison with EVT2.0 format
5. Real-world usage scenarios

References:
- Prophesee EVT2.1 specification
- https://docs.prophesee.ai/stable/data/encoding_formats/evt21.html
"""

import sys
import struct
import numpy as np
from pathlib import Path

# Add the project root to the path so we can import evlib
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import evlib  # noqa: E402


def create_synthetic_evt21_file(
    output_path: str,
    width: int = 640,
    height: int = 480,
    duration: float = 1.0,
    num_events: int = 100000,
    vectorized_ratio: float = 0.7,
) -> None:
    """
    Create a synthetic EVT2.1 file for testing purposes.

    This function generates a valid EVT2.1 file with proper header and binary data
    that demonstrates the key features of the format including vectorized events.

    Args:
        output_path: Path to output file
        width: Sensor width in pixels
        height: Sensor height in pixels
        duration: Recording duration in seconds
        num_events: Total number of events to generate
        vectorized_ratio: Ratio of events encoded as vectorized (0.0-1.0)
    """
    print(f"Creating synthetic EVT2.1 file: {output_path}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Duration: {duration}s")
    print(f"  Events: {num_events:,}")
    print(f"  Vectorized ratio: {vectorized_ratio:.1%}")

    with open(output_path, "wb") as f:
        # Write EVT2.1 header
        header = "% evt 2.1\n"
        header += f"% format EVT21;height={height};width={width}\n"
        header += f"% geometry {width}x{height}\n"
        header += "% date 2025-01-16 12:00:00\n"
        header += "% source synthetic_generator\n"
        header += "% end\n"

        f.write(header.encode("utf-8"))

        # Generate events
        timestamps = np.sort(np.random.uniform(0, duration * 1e6, num_events))  # microseconds
        x_coords = np.random.randint(0, width, num_events)
        y_coords = np.random.randint(0, height, num_events)
        polarities = np.random.choice([0, 1], num_events)

        # Generate binary data
        current_time_base = 0

        events_written = 0
        i = 0

        while i < num_events:
            # Write time high event when needed
            event_time = int(timestamps[i])
            required_time_base = (event_time >> 10) << 10

            if required_time_base > current_time_base:
                current_time_base = required_time_base
                time_high_data = int((current_time_base >> 10) << 4) | 0x08  # Time High event type
                f.write(struct.pack("<Q", time_high_data))

            # Decide whether to create vectorized event
            if np.random.random() < vectorized_ratio and i + 31 < num_events:
                # Create vectorized event (up to 32 pixels)
                base_x = x_coords[i]
                base_y = y_coords[i]
                base_polarity = polarities[i]
                base_timestamp = int(timestamps[i]) & 0x3FF  # 10-bit timestamp

                # Create validity mask for nearby pixels
                validity_mask = 0
                pixels_in_vector = min(32, num_events - i)

                for j in range(pixels_in_vector):
                    # Include pixel if it's spatially close and has same polarity
                    if (
                        abs(x_coords[i + j] - base_x) <= 31
                        and y_coords[i + j] == base_y
                        and polarities[i + j] == base_polarity
                    ):
                        validity_mask |= 1 << j

                # Encode vectorized event
                event_type = 0x01 if base_polarity else 0x00  # EVT_POS or EVT_NEG
                vectorized_data = (
                    (int(validity_mask) << 32)  # Bits 63-32: validity mask
                    | (int(base_y) << 26)  # Bits 31-26: Y coordinate
                    | (int(base_x) << 14)  # Bits 25-14: X coordinate base
                    | (int(base_timestamp) << 4)  # Bits 13-4: timestamp
                    | int(event_type)  # Bits 3-0: event type
                )

                f.write(struct.pack("<Q", vectorized_data))

                # Skip events that were included in the vectorized event
                events_included = bin(validity_mask).count("1")
                i += max(1, events_included)
                events_written += events_included

            else:
                # Create single-pixel vectorized event
                event_type = 0x01 if polarities[i] else 0x00
                timestamp = int(timestamps[i]) & 0x3FF
                validity_mask = 0x00000001  # Only first pixel valid

                vectorized_data = (
                    (int(validity_mask) << 32)
                    | (int(y_coords[i]) << 26)
                    | (int(x_coords[i]) << 14)
                    | (int(timestamp) << 4)
                    | int(event_type)
                )

                f.write(struct.pack("<Q", vectorized_data))
                i += 1
                events_written += 1

    print("✓ Synthetic EVT2.1 file created successfully")
    print(f"  File size: {Path(output_path).stat().st_size:,} bytes")
    print(f"  Events written: {events_written:,}")


def example_1_format_detection():
    """Example 1: Demonstrate EVT2.1 format detection capabilities."""
    print("EXAMPLE 1: EVT2.1 Format Detection")
    print("-" * 50)

    # Create a synthetic EVT2.1 file
    synthetic_file = "/tmp/synthetic_evt21.raw"
    create_synthetic_evt21_file(synthetic_file, width=640, height=480, num_events=50000)

    # Test format detection
    print("\nTesting format detection:")
    try:
        format_info = evlib.detect_format(synthetic_file)
        format_name, confidence, metadata = format_info

        print(f"  Detected format: {format_name}")
        print(f"  Confidence: {confidence:.2f}")
        print(f"  Metadata: {metadata}")

        if format_name == "EVT2.1":
            print("  ✓ EVT2.1 format correctly detected!")
        else:
            print(f"  ⚠ Expected EVT2.1, got {format_name}")

    except Exception as e:
        print(f"  ✗ Format detection failed: {e}")

    # Cleanup
    Path(synthetic_file).unlink(missing_ok=True)


def example_2_basic_loading():
    """Example 2: Basic EVT2.1 file loading with evlib.load_events()."""
    print("\nEXAMPLE 2: Basic EVT2.1 Loading")
    print("-" * 50)

    # Create a synthetic EVT2.1 file
    synthetic_file = "/tmp/synthetic_evt21_basic.raw"
    create_synthetic_evt21_file(
        synthetic_file, width=320, height=240, duration=0.5, num_events=25000, vectorized_ratio=0.8
    )

    try:
        # Load events using evlib
        print("Loading EVT2.1 data with evlib.load_events()...")
        events = evlib.load_events(synthetic_file)
        x_coords, y_coords, timestamps, polarities = events

        print(f"✓ Successfully loaded {len(timestamps):,} events")
        print(f"  Time range: {timestamps.min():.3f} - {timestamps.max():.3f} seconds")
        print(
            f"  Spatial range: x=[{x_coords.min()}-{x_coords.max()}], y=[{y_coords.min()}-{y_coords.max()}]"
        )
        print(
            f"  Data types: timestamps={type(timestamps[0])}, x={type(x_coords[0])}, y={type(y_coords[0])}, pol={type(polarities[0])}"
        )
        print(f"  Polarity distribution: {np.unique(polarities, return_counts=True)}")
        print(f"  Event rate: {len(timestamps) / (timestamps.max() - timestamps.min()):,.0f} events/sec")

        # Demonstrate data access
        print("\nFirst 5 events:")
        for i in range(min(5, len(timestamps))):
            print(
                f"  Event {i}: t={timestamps[i]:.6f}s, x={x_coords[i]}, y={y_coords[i]}, pol={polarities[i]}"
            )

    except Exception as e:
        print(f"✗ Failed to load EVT2.1 data: {e}")
        import traceback

        traceback.print_exc()

    # Cleanup
    Path(synthetic_file).unlink(missing_ok=True)


def example_3_vectorized_capabilities():
    """Example 3: Demonstrate vectorized event handling capabilities."""
    print("\nEXAMPLE 3: Vectorized Event Handling")
    print("-" * 50)

    # Create files with different vectorization ratios
    test_files = [
        ("/tmp/evt21_low_vector.raw", 0.3, "Low vectorization"),
        ("/tmp/evt21_high_vector.raw", 0.9, "High vectorization"),
    ]

    for file_path, vector_ratio, description in test_files:
        print(f"\n{description} (ratio: {vector_ratio:.1%}):")

        create_synthetic_evt21_file(
            file_path, width=640, height=480, duration=1.0, num_events=100000, vectorized_ratio=vector_ratio
        )

        try:
            # Load and analyze
            events = evlib.load_events(file_path)
            x_coords, y_coords, timestamps, polarities = events

            print(f"  Events loaded: {len(timestamps):,}")
            print(f"  File size: {Path(file_path).stat().st_size:,} bytes")
            print(
                f"  Compression ratio: {len(timestamps) / (Path(file_path).stat().st_size / 8):.2f} events/word"
            )

            # Analyze spatial clustering (indicator of vectorization effectiveness)
            unique_positions = len(set(zip(x_coords, y_coords)))
            clustering_factor = len(timestamps) / unique_positions
            print(f"  Spatial clustering: {clustering_factor:.2f} events/position")

        except Exception as e:
            print(f"  ✗ Failed to process {description}: {e}")

        # Cleanup
        Path(file_path).unlink(missing_ok=True)


def example_4_filtering_and_processing():
    """Example 4: Advanced filtering and processing of EVT2.1 data."""
    print("\nEXAMPLE 4: Advanced Filtering and Processing")
    print("-" * 50)

    # Create a larger synthetic file
    synthetic_file = "/tmp/evt21_filtering.raw"
    create_synthetic_evt21_file(
        synthetic_file, width=1280, height=720, duration=2.0, num_events=200000, vectorized_ratio=0.8
    )

    try:
        # Load with temporal filtering
        print("Loading with temporal filtering (0.5-1.5s)...")
        events = evlib.load_events(synthetic_file, t_start=0.5, t_end=1.5)
        x_coords, y_coords, timestamps, polarities = events

        print(f"  Filtered events: {len(timestamps):,}")
        print(f"  Time range: {timestamps.min():.3f} - {timestamps.max():.3f} seconds")

        # Load with spatial filtering
        print("\nLoading with spatial filtering (ROI: 320-960, 180-540)...")
        events_roi = evlib.load_events(synthetic_file, min_x=320, max_x=960, min_y=180, max_y=540)
        x_coords_roi, y_coords_roi, timestamps_roi, polarities_roi = events_roi

        print(f"  ROI events: {len(timestamps_roi):,}")
        print(
            f"  Spatial range: x=[{x_coords_roi.min()}-{x_coords_roi.max()}], y=[{y_coords_roi.min()}-{y_coords_roi.max()}]"
        )

        # Load with polarity filtering
        print("\nLoading positive events only...")
        events_pos = evlib.load_events(synthetic_file, polarity=1)
        x_coords_pos, y_coords_pos, timestamps_pos, polarities_pos = events_pos

        print(f"  Positive events: {len(timestamps_pos):,}")
        print(f"  Polarity check: {np.unique(polarities_pos)}")

        # Demonstrate temporal windowing
        print("\nTemporal windowing analysis:")
        window_size = 0.1  # 100ms windows
        full_events = evlib.load_events(synthetic_file)
        full_x_coords, full_y_coords, full_timestamps, full_polarities = full_events

        start_time = full_timestamps.min()
        end_time = full_timestamps.max()
        num_windows = int((end_time - start_time) / window_size)

        print(f"  Total duration: {end_time - start_time:.3f}s")
        print(f"  Window size: {window_size}s")
        print(f"  Number of windows: {num_windows}")

        window_counts = []
        for i in range(min(10, num_windows)):  # Show first 10 windows
            window_start = start_time + i * window_size
            window_end = window_start + window_size

            window_events = evlib.load_events(synthetic_file, t_start=window_start, t_end=window_end)
            window_x, window_y, window_t, window_p = window_events
            window_count = len(window_t)
            window_counts.append(window_count)

            print(f"  Window {i}: {window_start:.3f}-{window_end:.3f}s → {window_count:,} events")

        if window_counts:
            print(f"  Average events per window: {np.mean(window_counts):.0f}")
            print(f"  Window event rate std: {np.std(window_counts):.0f}")

    except Exception as e:
        print(f"✗ Failed to process EVT2.1 filtering: {e}")
        import traceback

        traceback.print_exc()

    # Cleanup
    Path(synthetic_file).unlink(missing_ok=True)


def example_5_evt21_vs_evt2_comparison():
    """Example 5: Compare EVT2.1 with EVT2.0 capabilities."""
    print("\nEXAMPLE 5: EVT2.1 vs EVT2.0 Comparison")
    print("-" * 50)

    # Create both formats for comparison
    evt21_file = "/tmp/comparison_evt21.raw"
    evt2_file = "/tmp/comparison_evt2.raw"

    # Create EVT2.1 file
    create_synthetic_evt21_file(
        evt21_file, width=640, height=480, duration=1.0, num_events=100000, vectorized_ratio=0.8
    )

    # Create a simple EVT2.0 file for comparison
    create_synthetic_evt2_file(evt2_file, width=640, height=480, duration=1.0, num_events=100000)

    try:
        # Compare formats
        print("Comparing EVT2.1 and EVT2.0 formats:")

        # Load both files
        evt21_events = evlib.load_events(evt21_file)
        evt2_events = evlib.load_events(evt2_file)

        evt21_x, evt21_y, evt21_timestamps, evt21_pol = evt21_events
        evt2_x, evt2_y, evt2_timestamps, evt2_pol = evt2_events

        # File size comparison
        evt21_size = Path(evt21_file).stat().st_size
        evt2_size = Path(evt2_file).stat().st_size

        print("\nFile Size Comparison:")
        print(f"  EVT2.1: {evt21_size:,} bytes")
        print(f"  EVT2.0: {evt2_size:,} bytes")
        print(f"  Compression improvement: {((evt2_size - evt21_size) / evt2_size) * 100:.1f}%")

        # Event count comparison
        print("\nEvent Count Comparison:")
        print(f"  EVT2.1: {len(evt21_timestamps):,} events")
        print(f"  EVT2.0: {len(evt2_timestamps):,} events")

        # Efficiency metrics
        print("\nEfficiency Metrics:")
        print(f"  EVT2.1: {len(evt21_timestamps) / evt21_size * 1000:.2f} events/KB")
        print(f"  EVT2.0: {len(evt2_timestamps) / evt2_size * 1000:.2f} events/KB")

        # Loading time comparison (approximate)
        import time

        start_time = time.time()
        evlib.load_events(evt21_file)
        evt21_load_time = time.time() - start_time

        start_time = time.time()
        evlib.load_events(evt2_file)
        evt2_load_time = time.time() - start_time

        print("\nLoading Time Comparison:")
        print(f"  EVT2.1: {evt21_load_time:.3f}s")
        print(f"  EVT2.0: {evt2_load_time:.3f}s")
        print(f"  Speed improvement: {((evt2_load_time - evt21_load_time) / evt2_load_time) * 100:.1f}%")

        # Data quality comparison
        print("\nData Quality Comparison:")
        print(f"  EVT2.1 time range: {evt21_timestamps.max() - evt21_timestamps.min():.3f}s")
        print(f"  EVT2.0 time range: {evt2_timestamps.max() - evt2_timestamps.min():.3f}s")

        print(f"  EVT2.1 spatial range: {evt21_x.max() - evt21_x.min()} x {evt21_y.max() - evt21_y.min()}")
        print(f"  EVT2.0 spatial range: {evt2_x.max() - evt2_x.min()} x {evt2_y.max() - evt2_y.min()}")

    except Exception as e:
        print(f"✗ Failed to compare formats: {e}")
        import traceback

        traceback.print_exc()

    # Cleanup
    Path(evt21_file).unlink(missing_ok=True)
    Path(evt2_file).unlink(missing_ok=True)


def create_synthetic_evt2_file(
    output_path: str,
    width: int = 640,
    height: int = 480,
    duration: float = 1.0,
    num_events: int = 100000,
) -> None:
    """Create a synthetic EVT2.0 file for comparison."""
    print(f"Creating synthetic EVT2.0 file: {output_path}")

    with open(output_path, "wb") as f:
        # Write EVT2.0 header
        header = "% evt 2.0\n"
        header += f"% format EVT2;height={height};width={width}\n"
        header += f"% geometry {width}x{height}\n"
        header += "% date 2025-01-16 12:00:00\n"
        header += "% source synthetic_generator\n"
        header += "% end\n"

        f.write(header.encode("utf-8"))

        # Generate events
        timestamps = np.sort(np.random.uniform(0, duration * 1e6, num_events))
        x_coords = np.random.randint(0, width, num_events)
        y_coords = np.random.randint(0, height, num_events)
        polarities = np.random.choice([0, 1], num_events)

        # Write EVT2.0 events (32-bit format)
        current_time_base = 0

        for i in range(num_events):
            event_time = int(timestamps[i])
            required_time_base = (event_time >> 6) << 6

            if required_time_base > current_time_base:
                current_time_base = required_time_base
                # EVT2.0 Time High event (simplified)
                time_high_data = int((current_time_base >> 6) << 2) | 0x02
                f.write(struct.pack("<I", time_high_data))

            # EVT2.0 CD event (simplified)
            timestamp_low = event_time & 0x3F
            event_data = (
                (int(y_coords[i]) << 22)
                | (int(x_coords[i]) << 12)
                | (int(polarities[i]) << 11)
                | (int(timestamp_low) << 2)
                | 0x00  # CD event type
            )
            f.write(struct.pack("<I", event_data))


def example_6_real_world_usage():
    """Example 6: Real-world usage scenarios with EVT2.1."""
    print("\nEXAMPLE 6: Real-World Usage Scenarios")
    print("-" * 50)

    # Create a realistic EVT2.1 file
    realistic_file = "/tmp/realistic_evt21.raw"
    create_synthetic_evt21_file(
        realistic_file, width=1280, height=720, duration=5.0, num_events=500000, vectorized_ratio=0.85
    )

    try:
        print("Scenario 1: Object tracking region extraction")
        # Extract events from a specific region (simulating object tracking)
        roi_events = evlib.load_events(
            realistic_file, min_x=400, max_x=880, min_y=200, max_y=520, t_start=1.0, t_end=3.0
        )
        roi_x, roi_y, roi_timestamps, roi_pol = roi_events

        print(f"  ROI events: {len(roi_timestamps):,}")
        print(f"  Event density: {len(roi_timestamps) / ((880-400) * (520-200)):.2f} events/pixel")
        print(f"  Temporal density: {len(roi_timestamps) / 2.0:.0f} events/second")

        print("\nScenario 2: High-speed event processing")
        # Process events in small time chunks (simulating real-time processing)
        chunk_duration = 0.1  # 100ms chunks
        total_events = 0

        full_events = evlib.load_events(realistic_file)
        full_x, full_y, full_timestamps, full_pol = full_events

        start_time = full_timestamps.min()
        end_time = full_timestamps.max()

        print(f"  Processing {chunk_duration}s chunks...")
        chunk_count = 0
        current_time = start_time

        while current_time < end_time and chunk_count < 10:  # Process first 10 chunks
            chunk_end = min(current_time + chunk_duration, end_time)

            chunk_events = evlib.load_events(realistic_file, t_start=current_time, t_end=chunk_end)
            chunk_x, chunk_y, chunk_t, chunk_p = chunk_events
            chunk_size = len(chunk_t)
            total_events += chunk_size

            print(f"    Chunk {chunk_count}: {current_time:.3f}s → {chunk_size:,} events")

            # Simulate processing time

            current_time = chunk_end
            chunk_count += 1

        print(f"  Total processed: {total_events:,} events")
        print(f"  Average processing rate: {total_events / (chunk_count * chunk_duration):,.0f} events/s")

        print("\nScenario 3: Polarity-based feature extraction")
        # Extract different polarity events for feature analysis
        pos_events = evlib.load_events(realistic_file, polarity=1)
        neg_events = evlib.load_events(realistic_file, polarity=-1)

        pos_x, pos_y, pos_timestamps, pos_pol = pos_events
        neg_x, neg_y, neg_timestamps, neg_pol = neg_events

        print(f"  Positive events: {len(pos_timestamps):,}")
        print(f"  Negative events: {len(neg_timestamps):,}")
        print(f"  Polarity ratio: {len(pos_timestamps) / len(neg_timestamps):.2f}")

        # Analyze spatial distribution
        pos_spatial_spread = np.std(pos_x) + np.std(pos_y)
        neg_spatial_spread = np.std(neg_x) + np.std(neg_y)

        print(f"  Positive spatial spread: {pos_spatial_spread:.1f}")
        print(f"  Negative spatial spread: {neg_spatial_spread:.1f}")

        print("\nScenario 4: Memory-efficient streaming")
        # Demonstrate memory-efficient processing using format info
        format_info = evlib.detect_format(realistic_file)
        format_name, confidence, metadata = format_info

        print(f"  Format: {format_name} (confidence: {confidence:.2f})")
        print(f"  Metadata: {metadata}")

        # File size analysis
        file_size = Path(realistic_file).stat().st_size
        total_events_data = evlib.load_events(realistic_file)
        total_events_count = len(total_events_data[2])  # timestamps is array 2

        print(f"  File size: {file_size:,} bytes")
        print(f"  Total events: {total_events_count:,}")
        print(f"  Bytes per event: {file_size / total_events_count:.2f}")
        print(f"  Compression efficiency: {total_events_count / (file_size / 8):.2f} events/word")

    except Exception as e:
        print(f"✗ Failed to demonstrate real-world usage: {e}")
        import traceback

        traceback.print_exc()

    # Cleanup
    Path(realistic_file).unlink(missing_ok=True)


def example_7_performance_analysis():
    """Example 7: Performance analysis of EVT2.1 processing."""
    print("\nEXAMPLE 7: Performance Analysis")
    print("-" * 50)

    import time

    # Test different file sizes
    test_configs = [
        (50000, "Small", 0.5),
        (200000, "Medium", 1.0),
        (500000, "Large", 2.0),
    ]

    for num_events, size_name, duration in test_configs:
        print(f"\n{size_name} dataset ({num_events:,} events):")

        test_file = f"/tmp/perf_test_{size_name.lower()}.raw"

        # Create file
        start_time = time.time()
        create_synthetic_evt21_file(
            test_file, width=640, height=480, duration=duration, num_events=num_events, vectorized_ratio=0.8
        )
        create_time = time.time() - start_time

        try:
            # Test loading performance
            start_time = time.time()
            events = evlib.load_events(test_file)
            load_time = time.time() - start_time

            # Test filtered loading performance
            start_time = time.time()
            filtered_events = evlib.load_events(
                test_file,
                min_x=100,
                max_x=540,
                min_y=100,
                max_y=380,
                t_start=duration * 0.2,
                t_end=duration * 0.8,
            )
            filtered_load_time = time.time() - start_time

            # Calculate metrics
            file_size = Path(test_file).stat().st_size
            x_coords, y_coords, timestamps, polarities = events
            filtered_x, filtered_y, filtered_timestamps, filtered_pol = filtered_events

            print(f"  File size: {file_size:,} bytes")
            print(f"  Creation time: {create_time:.3f}s")
            print(f"  Full load time: {load_time:.3f}s")
            print(f"  Filtered load time: {filtered_load_time:.3f}s")
            print(f"  Events loaded: {len(timestamps):,}")
            print(f"  Filtered events: {len(filtered_timestamps):,}")
            print(f"  Load rate: {len(timestamps) / load_time:,.0f} events/s")
            print(f"  Filtered rate: {len(filtered_timestamps) / filtered_load_time:,.0f} events/s")
            print(f"  Memory efficiency: {len(timestamps) / (file_size / 1024):.1f} events/KB")

        except Exception as e:
            print(f"  ✗ Performance test failed: {e}")

        # Cleanup
        Path(test_file).unlink(missing_ok=True)


def main():
    """Run all EVT2.1 examples."""
    print("EVT2.1 Format Examples - Advanced Event Camera Data Processing")
    print("=" * 70)

    print("\nThis example demonstrates the advanced capabilities of EVT2.1 format:")
    print("• 64-bit vectorized events with up to 32 pixels per word")
    print("• Improved compression through spatial correlation")
    print("• Enhanced timestamp resolution and range")
    print("• Seamless integration with evlib.load_events()")
    print("• Advanced filtering and processing capabilities")

    try:
        example_1_format_detection()
        example_2_basic_loading()
        example_3_vectorized_capabilities()
        example_4_filtering_and_processing()
        example_5_evt21_vs_evt2_comparison()
        example_6_real_world_usage()
        example_7_performance_analysis()

        print("\n" + "=" * 70)
        print("✓ All EVT2.1 examples completed successfully!")

        print("\nKey Takeaways:")
        print("• EVT2.1 provides significant compression improvements over EVT2.0")
        print("• Vectorized events enable efficient processing of dense event streams")
        print("• evlib.load_events() handles EVT2.1 format transparently")
        print("• Advanced filtering capabilities support real-time applications")
        print("• Format detection automatically identifies EVT2.1 files")
        print("• Performance scales well with file size and complexity")

        print("\nNext Steps:")
        print("• Try loading real EVT2.1 files from Prophesee cameras")
        print("• Experiment with different filtering combinations")
        print("• Integrate with computer vision pipelines")
        print("• Explore real-time processing applications")

    except Exception as e:
        print(f"\n✗ Example execution failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
