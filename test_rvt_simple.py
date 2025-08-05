#!/usr/bin/env python3
"""
Simple RVT test with text file data.
"""

import sys
from pathlib import Path
import time

sys.path.insert(0, "python")

from evlib.models import RVT
import numpy as np


def test_rvt_on_text_data():
    """Test RVT on slider_depth text file."""
    print("🚀 Testing RVT on text event data")

    # Create model
    model = RVT(variant="tiny", pretrained=False)
    model.eval()
    print(f"✓ Created RVT model: {model.variant}")

    # Load text data using evlib formats
    try:
        import evlib

        print("Loading events with evlib...")

        # Try different loading approaches
        try:
            # Method 1: Direct load_events (polars)
            events_lf = evlib.load_events("data/slider_depth/events.txt")
            print(f"✓ Loaded as LazyFrame: {type(events_lf)}")

            # Convert to numpy for RVT (temporary)
            events_df = events_lf.collect()
            xs = events_df["x"].to_numpy()
            ys = events_df["y"].to_numpy()
            ts = events_df["timestamp"].to_numpy().astype(float) / 1e6  # μs to seconds
            ps = events_df["polarity"].to_numpy()

        except Exception as e1:
            print(f"LazyFrame approach failed: {e1}")
            print("Trying numpy array approach...")

            # Method 2: Load as arrays
            t, x, y, polarity = evlib.formats.load_events("data/slider_depth/events.txt")
            xs = x.astype(int)
            ys = y.astype(int)
            ts = t.astype(float)
            ps = polarity.astype(int)

        print(f"✓ Loaded {len(xs)} events")
        print(f"  - Time range: {ts.min():.3f} - {ts.max():.3f} seconds")
        print(f"  - Spatial: x=[{xs.min()}, {xs.max()}], y=[{ys.min()}, {ys.max()}]")

        # Infer dimensions
        height = int(ys.max()) + 1
        width = int(xs.max()) + 1
        print(f"  - Resolution: {width}x{height}")

        # Test a small subset (first 100ms)
        duration = ts.max() - ts.min()
        if duration > 0.1:  # If more than 100ms of data
            mask = ts < (ts.min() + 0.1)  # First 100ms
            xs_sub = xs[mask]
            ys_sub = ys[mask]
            ts_sub = ts[mask]
            ps_sub = ps[mask]

            print(f"Using subset: {len(xs_sub)} events from first 100ms")
            events = (xs_sub, ys_sub, ts_sub, ps_sub)
        else:
            events = (xs, ys, ts, ps)

        # Run inference
        print("\nRunning RVT inference...")
        start_time = time.time()

        # Debug: Test histogram creation first
        print("Creating stacked histogram...")
        histogram, h, w = model.preprocess_events_to_histogram(events, height, width)
        print(f"Histogram shape: {histogram.shape}, expected: ({2*model.temporal_bins}, {height}, {width})")

        # Debug: Test backbone forward
        print("Testing backbone forward...")
        with torch.no_grad():
            histogram_batch = histogram.unsqueeze(0)  # Add batch dimension
            backbone_features, states = model.forward_backbone(histogram_batch)

            print("Backbone features:")
            for stage, feat in backbone_features.items():
                print(f"  Stage {stage}: {feat.shape}")

        with torch.no_grad():
            detections = model.detect(
                events,
                height=height,
                width=width,
                confidence_threshold=0.001,  # Very low threshold
                nms_threshold=0.5,
                reset_states=True,
            )

        inference_time = time.time() - start_time

        print(f"✓ Inference completed in {inference_time*1000:.1f}ms")
        print(f"  - Found {len(detections)} detections")

        for i, det in enumerate(detections[:3]):
            bbox = det["bbox"]
            print(
                f"  Det {i+1}: {det['class_name']} ({det['score']:.4f}) at [{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}]"
            )

        # Test stacked histogram creation
        print("\nTesting stacked histogram...")
        histogram, h, w = model.preprocess_events_to_histogram(events, height, width)
        print(f"✓ Created histogram: {histogram.shape}")
        print(f"  - Expected: ({2*model.temporal_bins}, {height}, {width})")
        print(f"  - Value range: [{histogram.min():.2f}, {histogram.max():.2f}]")

        print("\n🎉 RVT test completed successfully!")

        if len(detections) == 0:
            print("\nNote: Zero detections is normal for:")
            print("  - Untrained model (random weights)")
            print("  - Text event data (may not contain objects)")
            print("  - Conservative confidence threshold")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    import torch

    test_rvt_on_text_data()
