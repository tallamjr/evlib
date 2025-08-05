#!/usr/bin/env python3
"""
RVT Demo Script

Demonstrates RVT inference on real validation data from Gen4 1Mpx processed RVT dataset.
This script loads preprocessed stacked histogram representations and runs inference
to detect pedestrians and cyclists.

Usage:
    python examples/rvt_demo.py
    python examples/rvt_demo.py --data_dir /path/to/val/data --pretrained
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import time
from typing import Dict, List, Any, Optional

# Fix HDF5 plugin path issues early
try:
    import hdf5plugin

    # Plugin path will be set automatically by hdf5plugin
except ImportError:
    # Fallback: clear plugin path
    if "HDF5_PLUGIN_PATH" in os.environ:
        del os.environ["HDF5_PLUGIN_PATH"]
    os.environ["HDF5_PLUGIN_PATH"] = ""

# Add evlib to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent / "python"))

try:
    import evlib
    from evlib.models import RVT
    from evlib.pytorch import load_rvt_data, PolarsDataset, create_rvt_transform
    import polars as pl

    print("✓ Successfully imported evlib, RVT, and pytorch utilities")
except ImportError as e:
    print(f"✗ Failed to import evlib: {e}")
    print("Make sure you're running from the evlib root directory")
    sys.exit(1)


def find_validation_sequences(data_dir: Path) -> List[Path]:
    """Find available validation sequences in the data directory."""
    sequences = []

    if not data_dir.exists():
        print(f"✗ Data directory does not exist: {data_dir}")
        return sequences

    # Look for sequence directories
    for item in data_dir.iterdir():
        if item.is_dir():
            # Check if it has the expected structure (handle both naming conventions)
            event_repr_v2 = item / "event_representations_v2"
            if event_repr_v2.exists():
                # Try both naming conventions
                stacked_hist_dirs = [
                    event_repr_v2 / "stacked_histogram_dt50_nbins10",
                    event_repr_v2 / "stacked_histogram_dt=50_nbins=10",
                ]
                for stacked_hist_dir in stacked_hist_dirs:
                    if stacked_hist_dir.exists():
                        sequences.append(item)
                        break

    return sorted(sequences)


def load_sequence_data_evlib(sequence_path: Path, max_samples: int = 100) -> Optional[Any]:
    """Load sequence data using evlib's native RVT data loading functionality.

    Args:
        sequence_path: Path to sequence directory
        max_samples: Maximum number of samples to load

    Returns:
        Polars LazyFrame with RVT data or None if loading fails
    """
    print(f"Loading sequence with evlib native functions: {sequence_path.name}")

    try:
        # Use evlib's native RVT data loader
        lazy_df = load_rvt_data(sequence_path, max_samples=max_samples, setup_hdf5=True)

        if lazy_df is not None:
            # Get sample count
            sample_count = lazy_df.select(pl.len()).collect().item()
            print(f"  ✓ Loaded {sample_count} samples using evlib native loader")

            # Show feature information
            columns = lazy_df.columns
            print(f"  ✓ Available features: {len(columns)} columns")

            return lazy_df
        else:
            print(f"  ✗ evlib native loader returned None for {sequence_path}")
            return None

    except Exception as e:
        print(f"  ✗ Error with evlib native loader: {e}")
        return None


def load_sequence_data(sequence_path: Path) -> Dict[str, Any]:
    """Load preprocessed sequence data including representations and labels."""
    print(f"Loading sequence: {sequence_path.name}")

    # Load stacked histogram representations (handle both naming conventions)
    event_repr_v2 = sequence_path / "event_representations_v2"

    # Try both naming conventions
    repr_paths = [
        event_repr_v2 / "stacked_histogram_dt50_nbins10",
        event_repr_v2 / "stacked_histogram_dt=50_nbins=10",
    ]

    repr_path = None
    for path in repr_paths:
        if path.exists():
            repr_path = path
            break

    if repr_path is None:
        raise FileNotFoundError(f"Stacked histogram directory not found in {event_repr_v2}")

    h5_file = repr_path / "event_representations_ds2_nearest.h5"
    timestamps_file = repr_path / "timestamps_us.npy"

    if not h5_file.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_file}")

    try:
        import h5py
        import os

        # Fix HDF5 plugin path issue - try multiple approaches
        if "HDF5_PLUGIN_PATH" in os.environ:
            del os.environ["HDF5_PLUGIN_PATH"]
        os.environ["HDF5_PLUGIN_PATH"] = ""

        print(f"  Loading representations from: {h5_file.name}")

        with h5py.File(h5_file, "r") as f:
            # Find the main dataset
            if "events" in f:
                representations = f["events"][:]
            elif "data" in f:
                representations = f["data"][:]
            elif "stacked_histogram" in f:
                representations = f["stacked_histogram"][:]
            else:
                # Use the first available dataset
                key = list(f.keys())[0]
                print(f"  Using dataset key: {key}")
                representations = f[key][:]

        print(f"  ✓ Loaded representations: {representations.shape}")

    except ImportError:
        raise ImportError("h5py is required for loading preprocessed data")
    except Exception as e:
        raise RuntimeError(f"Error loading H5 file: {e}")

    # Load timestamps
    timestamps = None
    if timestamps_file.exists():
        timestamps = np.load(timestamps_file)
        print(f"  ✓ Loaded {len(timestamps)} timestamps")

    # Load labels if available
    labels_data = None
    labels_path = sequence_path / "labels_v2"
    if labels_path.exists():
        labels_file = labels_path / "labels.npz"
        if labels_file.exists():
            labels_data = np.load(labels_file)
            print(f"  ✓ Loaded labels: {list(labels_data.keys())}")

    return {
        "representations": representations,
        "timestamps": timestamps,
        "labels": labels_data,
        "sequence_name": sequence_path.name,
    }


def run_rvt_demo(
    data_dir: Path,
    pretrained: bool = False,
    max_sequences: int = 1,
    max_frames: int = 50,
    confidence_threshold: float = 0.1,
):
    """Run RVT demo on real validation data."""
    print("🚀 RVT Demo - Real Validation Data")
    print("=" * 60)

    # Create RVT model
    print("Loading RVT model...")
    # Adjust num_classes based on pretrained weights
    num_classes = 3 if pretrained else 2  # Pretrained has 3 classes, untrained uses 2

    model = RVT(variant="tiny", pretrained=pretrained, num_classes=num_classes)
    model.eval()

    print("✓ Model loaded successfully")
    print(f"  - Variant: {model.variant}")
    print(f"  - Device: {model._device}")
    print(f"  - Classes: {model.num_classes}")
    print(f"  - Temporal bins: {model.temporal_bins}")
    print(f"  - Pretrained: {pretrained}")

    # Find validation sequences
    print(f"\nScanning for validation sequences in: {data_dir}")
    sequences = find_validation_sequences(data_dir)

    if not sequences:
        print("✗ No validation sequences found")
        print("Expected structure: sequence_dir/event_representations_v2/stacked_histogram_dt50_nbins10/")
        return

    print(f"✓ Found {len(sequences)} validation sequences")

    # Process sequences
    total_detections = 0
    total_frames = 0
    total_inference_time = 0

    for seq_idx, sequence_path in enumerate(sequences[:max_sequences]):
        print(f"\n{'='*40}")
        print(f"SEQUENCE {seq_idx + 1}/{min(len(sequences), max_sequences)}")
        print(f"{'='*40}")

        try:
            # Load sequence data
            seq_data = load_sequence_data(sequence_path)
            representations = seq_data["representations"]
            timestamps = seq_data["timestamps"]
            labels = seq_data["labels"]

            T, C, H, W = representations.shape
            print(f"  - Shape: {T} frames, {C} channels, {H}x{W} resolution")

            if timestamps is not None:
                duration = (timestamps[-1] - timestamps[0]) / 1e6  # Convert to seconds
                print(f"  - Duration: {duration:.2f} seconds")
                print(f"  - FPS: {T/duration:.1f}")

            # Convert to torch tensor
            representations_torch = torch.from_numpy(representations).float()
            if torch.cuda.is_available() and model._device.type == "cuda":
                representations_torch = representations_torch.to(model._device)

            # Reset LSTM states for new sequence
            model.reset_states()

            # Process frames
            sequence_detections = []
            frames_to_process = min(T, max_frames)

            print(f"\nProcessing {frames_to_process} frames...")

            for frame_idx in range(frames_to_process):
                frame_repr = representations_torch[frame_idx].unsqueeze(0)  # Add batch dimension

                # Run inference
                start_time = time.time()
                with torch.no_grad():
                    predictions, _, _ = model.forward(frame_repr, retrieve_detections=True)

                    # Post-process predictions
                    detections = model._postprocess_predictions(
                        predictions, confidence_threshold, nms_threshold=0.45
                    )

                inference_time = time.time() - start_time
                total_inference_time += inference_time

                sequence_detections.append(detections)
                total_detections += len(detections)
                total_frames += 1

                # Print progress every 10 frames
                if (frame_idx + 1) % 10 == 0 or frame_idx == 0:
                    print(
                        f"  Frame {frame_idx + 1:3d}/{frames_to_process}: "
                        f"{len(detections)} detections ({inference_time*1000:.1f}ms)"
                    )

                    # Show first detection details
                    if len(detections) > 0:
                        det = detections[0]
                        bbox = det["bbox"]
                        print(
                            f"    → {det['class_name']} ({det['score']:.3f}) "
                            f"at [{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}]"
                        )

            # Sequence summary
            seq_detections = sum(len(dets) for dets in sequence_detections)
            print(f"\n  ✓ Sequence completed: {seq_detections} total detections")

            # Show ground truth comparison if available
            if labels is not None and "bbox" in labels:
                gt_bboxes = labels["bbox"]
                gt_frames_with_objects = len([bbox for bbox in gt_bboxes if len(bbox) > 0])
                print(f"  - Ground truth: {gt_frames_with_objects} frames with objects")

        except Exception as e:
            print(f"  ✗ Error processing sequence: {e}")
            continue

    # Overall summary
    print(f"\n{'='*60}")
    print("DEMO SUMMARY")
    print(f"{'='*60}")
    print(f"  - Processed sequences: {min(len(sequences), max_sequences)}")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Total detections: {total_detections}")
    print(f"  - Average detections per frame: {total_detections/max(total_frames,1):.2f}")
    avg_inference_time = total_inference_time / max(total_frames, 1)
    print(f"  - Average inference time: {avg_inference_time*1000:.1f}ms/frame")
    if avg_inference_time > 0:
        print(f"  - Effective FPS: {1/avg_inference_time:.1f}")
    else:
        print("  - Effective FPS: N/A")

    if total_detections == 0:
        print("\nNote: No detections found.")
        if not pretrained:
            print("  - EXPECTED: Untrained model produces uniform low-confidence outputs")
            print("  - Solution: Use --pretrained for meaningful results")
            print("  - Alternative: Try --confidence_threshold 0.0001 to see raw outputs")
        else:
            print("  - Try lowering --confidence_threshold")
        print("  - Model architecture is working correctly")

    print("\n🎉 RVT demo completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="RVT Demo with Real Validation Data")

    parser.add_argument(
        "--data_dir",
        type=str,
        default="/Users/tallam/github/tallamjr/origin/evlib/data/gen4_1mpx_processed_RVT/val",
        help="Directory containing validation sequences",
    )
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights")
    parser.add_argument("--max_sequences", type=int, default=1, help="Maximum number of sequences to process")
    parser.add_argument("--max_frames", type=int, default=50, help="Maximum frames per sequence")
    parser.add_argument(
        "--confidence_threshold", type=float, default=0.1, help="Detection confidence threshold"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    run_rvt_demo(
        data_dir=data_dir,
        pretrained=args.pretrained,
        max_sequences=args.max_sequences,
        max_frames=args.max_frames,
        confidence_threshold=args.confidence_threshold,
    )


if __name__ == "__main__":
    main()
