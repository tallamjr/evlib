#!/usr/bin/env python3
"""
Simple RVT Inference Example

Demonstrates RVT inference using evlib's native capabilities.
This example shows how to:
1. Load events with evlib
2. Create stacked histograms with evlib
3. Run RVT inference
4. Display results
"""

import sys
from pathlib import Path
import time

# Add evlib to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent / "python"))

try:
    import evlib
    from evlib.models import RVT
    import torch
    import polars as pl
    import numpy as np

    print("✓ Successfully imported evlib, RVT, and polars")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)


def find_sample_data():
    """Find sample event data in the repository."""
    data_dir = Path(__file__).parent.parent / "data"

    # Look for event files
    sample_files = []

    # Check different data directories
    for subdir in ["slider_depth", "eTram", "gen4", "original"]:
        search_dir = data_dir / subdir
        if search_dir.exists():
            # Look for various event file formats
            for pattern in ["*.txt", "*.h5", "*.hdf5", "*.raw"]:
                sample_files.extend(search_dir.rglob(pattern))

    return sorted(sample_files)


def run_rvt_inference_example(event_file: Path):
    """Run RVT inference on a sample event file."""
    print("🚀 RVT Inference Example")
    print(f"Event file: {event_file}")
    print("=" * 60)

    # Create RVT model
    print("Loading RVT model...")
    model = RVT(variant="tiny", pretrained=False, num_classes=2)
    model.eval()

    print(f"✓ Model loaded: {model.variant}")

    # Load events using evlib
    print(f"\nLoading events from: {event_file.name}")
    start_time = time.time()

    try:
        # Method 1: Use evlib directly (returns polars data)
        events_lf = evlib.load_events(str(event_file))

        # Get basic statistics
        stats = events_lf.select(
            [
                pl.len().alias("count"),
                pl.col("x").min().alias("x_min"),
                pl.col("x").max().alias("x_max"),
                pl.col("y").min().alias("y_min"),
                pl.col("y").max().alias("y_max"),
                pl.col("timestamp").min().alias("t_min"),
                pl.col("timestamp").max().alias("t_max"),
            ]
        ).collect()

        info = stats.row(0, named=True)
        load_time = time.time() - start_time

        print(f"✓ Loaded {info['count']} events in {load_time:.3f}s")
        print(f"  - Spatial: x=[{info['x_min']}, {info['x_max']}], y=[{info['y_min']}, {info['y_max']}]")
        print(f"  - Temporal: {info['t_min']} - {info['t_max']} μs")

        # Determine image dimensions
        height = info["y_max"] + 1
        width = info["x_max"] + 1

        print(f"  - Inferred resolution: {width}x{height}")

    except Exception as e:
        print(f"✗ Failed to load with evlib LazyFrame: {e}")
        print("Falling back to numpy arrays...")

        # Method 2: Load as numpy arrays (fallback)
        try:
            t, x, y, polarity = evlib.formats.load_events(str(event_file))

            # Convert to expected format
            xs = x.astype(int)
            ys = y.astype(int)
            ts = t.astype(float)
            ps_raw = polarity.astype(int)

            # Convert polarity to 0/1 encoding if needed (-1/1 -> 0/1)
            if ps_raw.min() < 0:
                ps = (ps_raw + 1) // 2  # Convert -1/1 to 0/1
            else:
                ps = ps_raw

            events = (xs, ys, ts, ps)
            height = int(ys.max()) + 1
            width = int(xs.max()) + 1

            load_time = time.time() - start_time
            print(f"✓ Loaded {len(xs)} events in {load_time:.3f}s")
            print(f"  - Spatial: x=[{xs.min()}, {xs.max()}], y=[{ys.min()}, {ys.max()}]")
            print(f"  - Temporal: {ts.min():.6f} - {ts.max():.6f} seconds")
            print(f"  - Resolution: {width}x{height}")

        except Exception as e2:
            print(f"✗ Failed to load events: {e2}")
            return

    # Create temporal windows for processing
    print("\nCreating temporal windows...")

    if "events_lf" in locals():
        # Use polars LazyFrame approach with evlib's native filtering
        print("Using evlib native filtering and processing...")

        # Use evlib's native time filtering for efficient windowed processing
        # Get time range information
        time_stats = events_lf.select(
            [
                pl.col("timestamp").min().alias("t_min"),
                pl.col("timestamp").max().alias("t_max"),
            ]
        ).collect()

        t_min = time_stats["t_min"][0]
        t_max = time_stats["t_max"][0]

        # Convert to microseconds for consistent handling
        if hasattr(t_min, "total_seconds"):
            t_min_us = int(t_min.total_seconds() * 1e6)
            t_max_us = int(t_max.total_seconds() * 1e6)
        else:
            t_min_us = int(t_min)
            t_max_us = int(t_max)

        duration_us = t_max_us - t_min_us
        window_duration_ms = 50.0  # 50ms windows
        window_duration_us = int(window_duration_ms * 1000)  # Convert to microseconds
        num_windows = max(1, int(duration_us / window_duration_us))

        print(f"  - Duration: {duration_us/1e6:.3f} seconds")
        print(f"  - Creating {num_windows} windows of {window_duration_ms:.1f}ms each")

        all_detections = []
        model.reset_states()

        for i in range(min(num_windows, 5)):  # Process first 5 windows
            window_start_us = t_min_us + i * window_duration_us
            window_end_us = window_start_us + window_duration_us

            print(f"\nProcessing window {i+1}/{min(num_windows, 5)}")

            # Use evlib's native time filtering
            try:
                # Convert microseconds to seconds for evlib filtering
                window_start_s = window_start_us / 1e6
                window_end_s = window_end_us / 1e6

                # Filter events using evlib's native time filtering
                filtered_events_lf = evlib.filter_by_time(
                    events_lf, start_time=window_start_s, end_time=window_end_s
                )

                # Collect filtered events
                window_events = filtered_events_lf.collect()

            except Exception as e:
                print(f"  - Error with evlib time filtering: {e}")
                # Fallback to manual filtering
                if hasattr(t_min, "total_seconds"):
                    import datetime

                    window_start_td = datetime.timedelta(microseconds=window_start_us)
                    window_end_td = datetime.timedelta(microseconds=window_end_us)
                    window_events_lf = events_lf.filter(
                        (pl.col("timestamp") >= window_start_td) & (pl.col("timestamp") < window_end_td)
                    )
                else:
                    window_events_lf = events_lf.filter(
                        (pl.col("timestamp") >= window_start_us) & (pl.col("timestamp") < window_end_us)
                    )
                window_events = window_events_lf.collect()

            if len(window_events) == 0:
                print("  - No events in window")
                all_detections.append([])
                continue

            print(f"  - {len(window_events)} events in window")

            # Use evlib's stacked histogram creation instead of manual conversion
            try:
                # Create stacked histogram directly with evlib
                histogram_df = evlib.create_stacked_histogram(
                    window_events,
                    height=height,
                    width=width,
                    nbins=model.temporal_bins,
                    window_duration_ms=window_duration_ms,
                )

                # Convert histogram to tensor using RVT model's method
                histogram_tensor, _, _ = model.preprocess_events_to_histogram(
                    (
                        window_events["x"].to_numpy(),
                        window_events["y"].to_numpy(),
                        window_events["timestamp"].to_numpy().astype(float) / 1e6,  # Convert to seconds
                        window_events["polarity"].to_numpy(),
                    ),
                    height=height,
                    width=width,
                )

                # Run inference directly with histogram
                start_inference = time.time()
                with torch.no_grad():
                    predictions, _, _ = model.forward(
                        histogram_tensor.unsqueeze(0), retrieve_detections=True  # Add batch dimension
                    )

                # Post-process predictions
                detections = model._postprocess_predictions(
                    predictions, 0.01, 0.5  # confidence_threshold, nms_threshold
                )

            except Exception as e:
                print(f"  - Error with evlib histogram creation: {e}")
                # Fallback to model's detect method
                xs = window_events["x"].to_numpy()
                ys = window_events["y"].to_numpy()
                ts_raw = window_events["timestamp"].to_numpy()

                if hasattr(ts_raw[0], "total_seconds"):
                    ts = np.array([t.total_seconds() for t in ts_raw], dtype=float)
                else:
                    ts = ts_raw.astype(float) / 1e6

                ps_raw = window_events["polarity"].to_numpy()
                if ps_raw.min() < 0:
                    ps = (ps_raw + 1) // 2
                else:
                    ps = ps_raw

                start_inference = time.time()
                with torch.no_grad():
                    detections = model.detect(
                        (xs, ys, ts, ps),
                        height=height,
                        width=width,
                        confidence_threshold=0.01,
                        nms_threshold=0.5,
                        reset_states=False,
                    )

            inference_time = time.time() - start_inference
            all_detections.append(detections)

            print(f"  - Inference time: {inference_time*1000:.1f}ms")
            print(f"  - Found {len(detections)} detections")

            for j, det in enumerate(detections[:2]):  # Show first 2 detections
                bbox = det["bbox"]
                print(
                    f"    Det {j+1}: {det['class_name']} ({det['score']:.3f}) at [{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}]"
                )

    else:
        # Use numpy array approach
        print("Using numpy array processing...")

        # Create time windows
        t_min, t_max = ts.min(), ts.max()
        duration = t_max - t_min
        window_duration = 0.05  # 50ms
        num_windows = max(1, int(duration / window_duration))

        print(f"  - Duration: {duration:.3f} seconds")
        print(f"  - Creating {num_windows} windows of {window_duration*1000:.1f}ms each")

        all_detections = []
        model.reset_states()

        for i in range(min(num_windows, 5)):  # Process first 5 windows
            window_start = t_min + i * window_duration
            window_end = window_start + window_duration

            print(f"\nProcessing window {i+1}/{min(num_windows, 5)}")

            # Filter events in window
            mask = (ts >= window_start) & (ts < window_end)
            if not mask.any():
                print("  - No events in window")
                all_detections.append([])
                continue

            xs_win, ys_win, ts_win, ps_win = xs[mask], ys[mask], ts[mask], ps[mask]
            window_events = (xs_win, ys_win, ts_win, ps_win)
            print(f"  - {len(window_events[0])} events in window")

            # Run detection
            start_inference = time.time()
            with torch.no_grad():
                detections = model.detect(
                    window_events,
                    height=height,
                    width=width,
                    confidence_threshold=0.01,
                    nms_threshold=0.5,
                    reset_states=False,
                )

            inference_time = time.time() - start_inference
            all_detections.append(detections)

            print(f"  - Inference time: {inference_time*1000:.1f}ms")
            print(f"  - Found {len(detections)} detections")

            for j, det in enumerate(detections[:2]):  # Show first 2 detections
                bbox = det["bbox"]
                print(
                    f"    Det {j+1}: {det['class_name']} ({det['score']:.3f}) at [{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}]"
                )

    # Summary
    total_detections = sum(len(dets) for dets in all_detections)
    processed_windows = len(all_detections)

    print("\n✅ Inference completed!")
    print(f"  - Processed {processed_windows} temporal windows")
    print(f"  - Total detections: {total_detections}")
    print(f"  - Average detections per window: {total_detections/max(processed_windows,1):.1f}")

    if total_detections == 0:
        print("\nNote: No detections found. This is normal with:")
        print("  - Random/synthetic data")
        print("  - Untrained model weights")
        print("  - Low confidence threshold needed for real detections")


def main():
    """Main function to run the example."""
    print("🔍 Looking for sample event data...")

    sample_files = find_sample_data()

    if not sample_files:
        print("✗ No sample event data found in the data directory")
        print("Please ensure you have event data files in the data/ directory")
        return

    print(f"✓ Found {len(sample_files)} event files")

    # Use the first suitable file, prefer events.txt for faster testing
    event_txt_files = [f for f in sample_files if f.name == "events.txt"]
    if event_txt_files:
        event_file = event_txt_files[0]
        print(f"Selected: {event_file}")
        run_rvt_inference_example(event_file)
        return

    # Otherwise use first suitable file
    for event_file in sample_files:
        if event_file.suffix in [".h5", ".hdf5"]:
            print(f"Selected: {event_file}")
            run_rvt_inference_example(event_file)
            return

    print("✗ No suitable event files found")
    print("Supported formats: .txt, .h5, .hdf5, .raw")


if __name__ == "__main__":
    main()
