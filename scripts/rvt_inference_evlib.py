#!/usr/bin/env python3
"""
RVT Inference Script - Native evlib version

Run RVT object detection inference using evlib's native polars-based data structures.
This version leverages evlib's efficient event loading and representation creation.

Usage:
    python scripts/rvt_inference_evlib.py --event_file /path/to/events.h5 --model_variant tiny
    python scripts/rvt_inference_evlib.py --data_dir /path/to/data --pretrained
"""

import argparse
import os
import sys
from pathlib import Path
import time
from typing import List, Dict, Any, Optional, Tuple

# Add evlib to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent / "python"))

try:
    import evlib
    from evlib.models import RVT, RVTModelConfig
    import polars as pl
    import torch

    print("✓ Successfully imported evlib, RVT, and polars")
except ImportError as e:
    print(f"✗ Failed to import required modules: {e}")
    print("Make sure you have evlib built with polars support")
    sys.exit(1)


class RVTInferenceEvlib:
    """RVT inference engine using native evlib data structures."""

    def __init__(
        self,
        model_variant: str = "tiny",
        pretrained: bool = True,
        confidence_threshold: float = 0.1,
        nms_threshold: float = 0.45,
        device: Optional[str] = None,
    ):
        """Initialize RVT inference engine.

        Args:
            model_variant: Model variant ("tiny", "small", "base")
            pretrained: Whether to load pretrained weights
            confidence_threshold: Detection confidence threshold
            nms_threshold: NMS threshold
            device: Device to run on ("cpu", "cuda", or None for auto)
        """
        self.model_variant = model_variant
        self.pretrained = pretrained
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Load model
        print(f"Loading RVT-{model_variant.upper()} model...")
        self.model = RVT(
            variant=model_variant,
            pretrained=pretrained,
            num_classes=2,  # pedestrian + cyclist
        )
        self.model.to(self.device)
        self.model.eval()

        print("✓ Model loaded successfully")
        print(f"  - Variant: {self.model.variant}")
        print(f"  - Classes: {self.model.num_classes}")
        print(f"  - Temporal bins: {self.model.temporal_bins}")
        print(f"  - Device: {self.model._device}")

    def load_events_evlib(self, file_path: str) -> pl.LazyFrame:
        """Load events using evlib's native readers.

        Args:
            file_path: Path to event file

        Returns:
            Polars LazyFrame with events
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Event file not found: {file_path}")

        print(f"Loading events with evlib from: {file_path}")

        try:
            # Use evlib to load events - this returns polars data
            events_lf = evlib.load_events(str(file_path))

            # Get basic info about the events
            events_info = events_lf.select(
                [
                    pl.len().alias("count"),
                    pl.col("x").min().alias("x_min"),
                    pl.col("x").max().alias("x_max"),
                    pl.col("y").min().alias("y_min"),
                    pl.col("y").max().alias("y_max"),
                    pl.col("timestamp").min().alias("t_min"),
                    pl.col("timestamp").max().alias("t_max"),
                    pl.col("polarity").n_unique().alias("polarities"),
                ]
            ).collect()

            info = events_info.row(0, named=True)

            print(f"✓ Loaded {info['count']} events using evlib")
            print(
                f"  - Spatial range: x=[{info['x_min']}, {info['x_max']}], y=[{info['y_min']}, {info['y_max']}]"
            )
            print(f"  - Time range: {info['t_min']} - {info['t_max']} μs")
            print(f"  - Unique polarities: {info['polarities']}")

            return events_lf

        except Exception as e:
            print(f"✗ Failed to load events with evlib: {e}")
            raise

    def create_temporal_windows(
        self,
        events_lf: pl.LazyFrame,
        window_duration_us: int = 50_000,  # 50ms in microseconds
        stride_us: Optional[int] = None,
    ) -> List[pl.LazyFrame]:
        """Create temporal windows from events.

        Args:
            events_lf: Events LazyFrame
            window_duration_us: Window duration in microseconds
            stride_us: Stride between windows (default: same as window_duration_us)

        Returns:
            List of windowed event LazyFrames
        """
        if stride_us is None:
            stride_us = window_duration_us

        # Get time range
        time_info = events_lf.select(
            [
                pl.col("timestamp").min().alias("t_min"),
                pl.col("timestamp").max().alias("t_max"),
            ]
        ).collect()

        t_min = time_info["t_min"][0]
        t_max = time_info["t_max"][0]

        print("Creating temporal windows:")
        print(f"  - Window duration: {window_duration_us/1000:.1f} ms")
        print(f"  - Stride: {stride_us/1000:.1f} ms")
        print(f"  - Time range: {t_min} - {t_max} μs ({(t_max-t_min)/1e6:.3f} seconds)")

        # Create windows
        windows = []
        current_time = t_min

        while current_time < t_max:
            window_end = current_time + window_duration_us

            # Filter events in this window
            window_events = events_lf.filter(
                (pl.col("timestamp") >= current_time) & (pl.col("timestamp") < window_end)
            )

            windows.append(window_events)
            current_time += stride_us

        print(f"✓ Created {len(windows)} temporal windows")
        return windows

    def run_inference_on_windows(
        self,
        event_windows: List[pl.LazyFrame],
        height: int = 360,  # RVT preprocessed data height
        width: int = 640,  # RVT preprocessed data width
        temporal_bins: int = 10,
    ) -> List[List[Dict[str, Any]]]:
        """Run inference on temporal windows using evlib representations.

        Args:
            event_windows: List of event windows
            height: Image height
            width: Image width
            temporal_bins: Number of temporal bins for stacked histogram

        Returns:
            List of detection results for each window
        """
        print(f"Running inference on {len(event_windows)} windows")
        print(f"  - Resolution: {height}x{width}")
        print(f"  - Temporal bins: {temporal_bins}")

        all_detections = []
        total_inference_time = 0

        # Reset LSTM states
        self.model.reset_states()

        for i, window_lf in enumerate(event_windows):
            print(f"Processing window {i+1}/{len(event_windows)}")

            # Collect events for this window
            try:
                window_events = window_lf.collect()

                if len(window_events) == 0:
                    print("  - Empty window, skipping")
                    all_detections.append([])
                    continue

                print(f"  - {len(window_events)} events in window")

                # Create stacked histogram using evlib
                start_time = time.time()

                # Use evlib's stacked histogram representation
                histogram_df = evlib.create_stacked_histogram(
                    window_events, height=height, width=width, nbins=temporal_bins, window_duration_ms=50.0
                )

                # Convert to tensor format expected by RVT
                histogram_tensor = self._convert_histogram_to_tensor(
                    histogram_df, height, width, temporal_bins
                )

                # Run RVT inference
                with torch.no_grad():
                    predictions, _, _ = self.model.forward(
                        histogram_tensor.unsqueeze(0), retrieve_detections=True  # Add batch dimension
                    )

                # Post-process predictions
                detections = self.model._postprocess_predictions(
                    predictions, self.confidence_threshold, self.nms_threshold
                )

                inference_time = time.time() - start_time
                total_inference_time += inference_time

                all_detections.append(detections)

                if len(detections) > 0:
                    print(f"  - Found {len(detections)} detections")
                    for j, det in enumerate(detections[:3]):  # Show first 3
                        bbox = det["bbox"]
                        print(
                            f"    Det {j+1}: {det['class_name']} ({det['score']:.3f}) at [{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}]"
                        )

            except Exception as e:
                print(f"  - Error processing window: {e}")
                all_detections.append([])
                continue

        avg_inference_time = total_inference_time / len(event_windows)
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        total_detections = sum(len(dets) for dets in all_detections)

        print("\n✓ Inference completed!")
        print(f"  - Average inference time: {avg_inference_time*1000:.2f} ms/window")
        print(f"  - Effective FPS: {fps:.1f}")
        print(f"  - Total detections: {total_detections}")

        return all_detections

    def _convert_histogram_to_tensor(
        self, histogram_df: pl.DataFrame, height: int, width: int, bins: int
    ) -> torch.Tensor:
        """Convert evlib stacked histogram to PyTorch tensor using native .to_torch().

        Args:
            histogram_df: Histogram DataFrame (not LazyFrame)
            height: Image height
            width: Image width
            bins: Number of temporal bins

        Returns:
            PyTorch tensor of shape (2*bins, height, width)
        """
        # Create tensor
        channels = 2 * bins
        tensor = torch.zeros(channels, height, width, dtype=torch.float32, device=self.device)

        if len(histogram_df) > 0:
            # Use evlib's efficient tensor conversion via polars .to_torch()
            # Convert relevant columns to torch tensors directly
            channel_tensor = histogram_df.select(pl.col("channel")).to_torch().squeeze()
            time_bin_tensor = histogram_df.select(pl.col("time_bin")).to_torch().squeeze()
            y_tensor = histogram_df.select(pl.col("y")).to_torch().squeeze()
            x_tensor = histogram_df.select(pl.col("x")).to_torch().squeeze()
            count_tensor = histogram_df.select(pl.col("count")).to_torch().squeeze().float()

            # Calculate channel indices: channel + time_bin * 2
            # This interleaves pos/neg for each time bin: [t0_neg, t0_pos, t1_neg, t1_pos, ...]
            channel_indices = channel_tensor + time_bin_tensor * 2

            # Use torch advanced indexing for efficient tensor filling
            # Create masks for valid indices
            valid_mask = (
                (channel_indices >= 0)
                & (channel_indices < channels)
                & (y_tensor >= 0)
                & (y_tensor < height)
                & (x_tensor >= 0)
                & (x_tensor < width)
            )

            if valid_mask.any():
                valid_channels = channel_indices[valid_mask]
                valid_y = y_tensor[valid_mask]
                valid_x = x_tensor[valid_mask]
                valid_counts = count_tensor[valid_mask]

                # Use scatter_add for efficient accumulation
                tensor.index_put_((valid_channels, valid_y, valid_x), valid_counts, accumulate=True)

        return tensor

    def run_continuous_inference(
        self,
        events_lf: pl.LazyFrame,
        window_duration_ms: float = 50.0,  # 50ms windows
        stride_ms: Optional[float] = None,
        height: int = 360,
        width: int = 640,
    ) -> List[List[Dict[str, Any]]]:
        """Run continuous inference on event stream.

        Args:
            events_lf: Events LazyFrame
            window_duration_ms: Window duration in milliseconds
            stride_ms: Stride between windows in milliseconds
            height: Image height
            width: Image width

        Returns:
            List of detection results for each window
        """
        window_duration_us = int(window_duration_ms * 1000)
        stride_us = int(stride_ms * 1000) if stride_ms else window_duration_us

        # Create temporal windows
        event_windows = self.create_temporal_windows(events_lf, window_duration_us, stride_us)

        # Run inference on windows
        return self.run_inference_on_windows(event_windows, height, width, self.model.temporal_bins)

    def save_results(self, detections: List, output_path: str, metadata: Optional[Dict] = None):
        """Save detection results."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        import json

        # Prepare results
        results = {
            "detections": detections,
            "model_variant": self.model_variant,
            "confidence_threshold": self.confidence_threshold,
            "nms_threshold": self.nms_threshold,
            "num_windows": len(detections),
            "total_detections": sum(len(dets) for dets in detections),
        }

        if metadata:
            results["metadata"] = metadata

        # Save as JSON for better readability
        with open(output_path.with_suffix(".json"), "w") as f:
            json.dump(results, f, indent=2)

        print(f"✓ Results saved to: {output_path.with_suffix('.json')}")


def main():
    parser = argparse.ArgumentParser(description="RVT Inference Script - Native evlib version")

    # Model configuration
    parser.add_argument(
        "--model_variant",
        type=str,
        default="tiny",
        choices=["tiny", "small", "base"],
        help="RVT model variant",
    )
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights")
    parser.add_argument(
        "--confidence_threshold", type=float, default=0.01, help="Detection confidence threshold"
    )
    parser.add_argument("--nms_threshold", type=float, default=0.45, help="NMS threshold")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cpu/cuda)")

    # Data configuration
    parser.add_argument("--event_file", type=str, required=True, help="Path to event file (H5, raw, etc.)")
    parser.add_argument(
        "--data_dir", type=str, default=None, help="Directory containing multiple event files"
    )

    # Inference configuration
    parser.add_argument(
        "--window_duration_ms", type=float, default=50.0, help="Window duration in milliseconds"
    )
    parser.add_argument(
        "--stride_ms", type=float, default=None, help="Stride between windows in milliseconds"
    )
    parser.add_argument(
        "--height", type=int, default=360, help="Image height (360 for RVT preprocessed data)"
    )
    parser.add_argument("--width", type=int, default=640, help="Image width")

    # Output configuration
    parser.add_argument(
        "--output_dir", type=str, default="outputs/rvt_inference_evlib", help="Output directory for results"
    )
    parser.add_argument("--max_files", type=int, default=None, help="Maximum number of files to process")

    args = parser.parse_args()

    # Create inference engine
    inference_engine = RVTInferenceEvlib(
        model_variant=args.model_variant,
        pretrained=args.pretrained,
        confidence_threshold=args.confidence_threshold,
        nms_threshold=args.nms_threshold,
        device=args.device,
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.data_dir:
        # Process multiple files
        data_dir = Path(args.data_dir)
        event_files = []

        # Look for common event file formats
        for ext in ["*.h5", "*.hdf5", "*.raw", "*.dat", "*.aedat"]:
            event_files.extend(data_dir.rglob(ext))

        event_files = sorted(event_files)

        if args.max_files:
            event_files = event_files[: args.max_files]

        print(f"Found {len(event_files)} event files to process")

        for i, event_file in enumerate(event_files):
            print(f"\n--- Processing file {i+1}/{len(event_files)}: {event_file.name} ---")

            try:
                # Load events with evlib
                events_lf = inference_engine.load_events_evlib(event_file)

                # Run inference
                detections = inference_engine.run_continuous_inference(
                    events_lf,
                    window_duration_ms=args.window_duration_ms,
                    stride_ms=args.stride_ms,
                    height=args.height,
                    width=args.width,
                )

                # Save results
                output_file = output_dir / f"{event_file.stem}_results_{args.model_variant}"
                inference_engine.save_results(detections, output_file)

            except Exception as e:
                print(f"✗ Failed to process {event_file.name}: {e}")
                continue

    else:
        # Process single file
        print(f"\n{'='*60}")
        print("PROCESSING SINGLE EVENT FILE")
        print(f"{'='*60}")

        # Load events with evlib
        events_lf = inference_engine.load_events_evlib(args.event_file)

        # Run inference
        detections = inference_engine.run_continuous_inference(
            events_lf,
            window_duration_ms=args.window_duration_ms,
            stride_ms=args.stride_ms,
            height=args.height,
            width=args.width,
        )

        # Save results
        event_file = Path(args.event_file)
        output_file = output_dir / f"{event_file.stem}_results_{args.model_variant}"
        inference_engine.save_results(detections, output_file)

    print(f"\n🎉 Inference completed! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
