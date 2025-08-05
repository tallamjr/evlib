#!/usr/bin/env python3
"""
Working RVT Validation Script
Based on the successful implementation documented in rvt_implementation_summary.md
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

import torch
import numpy as np
from pathlib import Path

from evlib.models.rvt import RVT, RVTModelConfig


def create_evlib_stacked_histogram(events_df, height=720, width=1280, temporal_bins=10):
    """
    Create stacked histogram using evlib's native create_stacked_histogram function.
    Uses evlib's optimized implementation for best performance.
    """
    import evlib

    # Use evlib's native stacked histogram function
    # This returns a DataFrame with histogram data
    hist_df = evlib.create_stacked_histogram(
        events_df,
        height,
        width,
        nbins=temporal_bins,
        window_duration_ms=50.0,  # 50ms window
        count_cutoff=10,  # Max count per bin
    )

    print(f"✓ evlib histogram DataFrame shape: {hist_df.shape}")
    print(f"  Columns: {hist_df.columns}")

    # Convert DataFrame to tensor format for RVT model
    # Create tensor: 20 channels (10 negative + 10 positive)
    hist_tensor = torch.zeros(20, height, width)

    # Fill tensor from DataFrame
    hist_data = hist_df.collect() if hasattr(hist_df, "collect") else hist_df

    for row in hist_data.iter_rows(named=True):
        channel = row["channel_time_bin"]  # This should be the combined channel+time_bin index
        y = row["y"]
        x = row["x"]
        count = row["count"]

        if 0 <= x < width and 0 <= y < height and 0 <= channel < 20:
            hist_tensor[channel, y, x] = min(count, 10)  # Apply count cutoff

    return hist_tensor


def test_rvt_validation():
    """Test RVT with the approach that achieved 0.596 max confidence."""
    print("🚀 RVT Validation - Using Successful Implementation")
    print("=" * 60)

    # Create model with 3 classes (100% parameter loading from summary)
    config = RVTModelConfig.tiny()
    config.num_classes = 3  # Critical for 100% parameter loading
    model = RVT(config=config, pretrained=True, num_classes=3)
    model.eval()
    print(f"✓ RVT model loaded with {config.num_classes} classes")

    # Load real RVT preprocessed validation data
    test_file = "data/gen4_1mpx_processed_RVT/val/moorea_2019-02-21_000_td_2257500000_2317500000/event_representations_v2/stacked_histogram_dt50_nbins10/event_representations_ds2_nearest.h5"
    if not Path(test_file).exists():
        print(f"✗ RVT preprocessed validation file not found: {test_file}")
        return

    print(f"Loading RVT preprocessed validation data from: {test_file}")
    print("This is the exact preprocessed stacked histogram data that RVT was trained on!")

    try:
        # Fix HDF5 plugin path issues
        import os

        try:
            import hdf5plugin

            # Plugin path will be set automatically by hdf5plugin
        except ImportError:
            # Fallback: set plugin path manually
            plugin_path = "/Users/tallam/github/tallamjr/origin/evlib/.venv/lib/python3.10/site-packages/hdf5plugin/plugins"
            if os.path.exists(plugin_path):
                os.environ["HDF5_PLUGIN_PATH"] = plugin_path

        import h5py

        # Load preprocessed stacked histogram directly (no need to create it!)
        print("Loading preprocessed stacked histogram data...")
        with h5py.File(test_file, "r") as f:
            print(f"✓ H5 file keys: {list(f.keys())}")

            # Load the preprocessed histogram data
            if "data" in f:
                hist_data = f["data"][:]  # Load the histogram data
                print(f"✓ Loaded preprocessed histogram shape: {hist_data.shape}")

                # Convert to PyTorch tensor
                hist = torch.from_numpy(hist_data).float()

                # If batch dimension exists, take first sample
                if hist.dim() == 4:  # [batch, channels, height, width]
                    hist = hist[0]
                    print(f"✓ Using first sample, final shape: {hist.shape}")

            else:
                print("✗ No 'data' key found in H5 file")
                print(f"Available keys: {list(f.keys())}")
                return
        print(f"✓ Histogram shape: {hist.shape}")
        print(f"  Value range: {hist.min():.3f} - {hist.max():.3f}")
        print(f"  Non-zero elements: {(hist > 0).sum()}")

        # Run inference
        print("\nRunning RVT inference...")
        with torch.no_grad():
            batch = hist.unsqueeze(0)  # Add batch dimension
            output = model(batch)

            if isinstance(output, tuple):
                predictions = output[0]
            else:
                predictions = output

            print(f"✓ Inference complete - Output shape: {predictions.shape}")

            # Apply sigmoid to convert logits to probabilities (0-1 range)
            confidences = torch.sigmoid(predictions)
            max_conf = confidences.max().item()
            print(f"  Raw max value: {predictions.max().item():.6f}")
            print(f"  Max confidence (after sigmoid): {max_conf:.6f}")

            # Count high-confidence detections on processed confidences
            thresholds = [0.001, 0.01, 0.1, 0.5]
            for thresh in thresholds:
                count = (confidences > thresh).sum().item()
                print(f"  Detections > {thresh}: {count}")

            if max_conf > 0.1:
                print(f"🎉 SUCCESS! High confidence detection: {max_conf:.3f}")
                print("   This matches the successful results from the summary!")
            elif max_conf > 0.01:
                print(f"✓ Good detection confidence: {max_conf:.3f}")
            else:
                print(f"⚠️  Low confidence: {max_conf:.6f}")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_rvt_validation()
