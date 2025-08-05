#!/usr/bin/env python3
"""
RVT Inference Script

Run RVT object detection inference on event data.
Supports both preprocessed RVT dataset format and raw event files.

Usage:
    python scripts/rvt_inference.py --data_path /path/to/data --model_variant tiny
    python scripts/rvt_inference.py --sequence_path /path/to/sequence --pretrained
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import time
from typing import List, Dict, Any, Optional, Tuple

# Add evlib to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent / "python"))

try:
    import evlib
    from evlib.models import RVT, RVTModelConfig

    print("✓ Successfully imported evlib and RVT")
except ImportError as e:
    print(f"✗ Failed to import evlib: {e}")
    print("Make sure you're running from the evlib root directory")
    sys.exit(1)


class RVTInference:
    """RVT inference engine for event-based object detection."""

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

    def load_preprocessed_sequence(self, sequence_path: str) -> Tuple[np.ndarray, Dict]:
        """Load preprocessed RVT sequence data.

        Args:
            sequence_path: Path to sequence directory

        Returns:
            Tuple of (event_representations, metadata)
        """
        sequence_path = Path(sequence_path)

        # Load event representations
        repr_path = sequence_path / "event_representations_v2"
        if not repr_path.exists():
            raise FileNotFoundError(f"Event representations not found: {repr_path}")

        # Look for stacked histogram directory
        stacked_hist_path = repr_path / "stacked_histogram_dt50_nbins10"
        if stacked_hist_path.exists():
            print(f"Found stacked histogram directory: {stacked_hist_path}")

            # Load H5 file
            h5_file = stacked_hist_path / "event_representations_ds2_nearest.h5"
            if h5_file.exists():
                try:
                    import h5py
                    import os

                    # Fix HDF5 plugin path issue
                    os.environ["HDF5_PLUGIN_PATH"] = ""

                    print(f"Loading H5 file: {h5_file}")

                    with h5py.File(h5_file, "r") as f:
                        print(f"H5 file keys: {list(f.keys())}")

                        # Look for the main data key
                        if "events" in f:
                            representations = f["events"][:]
                        elif "data" in f:
                            representations = f["data"][:]
                        elif "stacked_histogram" in f:
                            representations = f["stacked_histogram"][:]
                        else:
                            # Use the first dataset
                            key = list(f.keys())[0]
                            print(f"Using dataset key: {key}")
                            representations = f[key][:]

                    print(f"Loaded representations shape: {representations.shape}")

                    # Load timestamps
                    timestamps_file = stacked_hist_path / "timestamps_us.npy"
                    timestamps = None
                    if timestamps_file.exists():
                        timestamps = np.load(timestamps_file)
                        print(f"Loaded {len(timestamps)} timestamps")

                except ImportError:
                    print("h5py not available, trying alternative loading...")
                    raise FileNotFoundError("h5py required for loading H5 files")
                except Exception as e:
                    print(f"Error loading H5 file: {e}")
                    raise
            else:
                raise FileNotFoundError(f"H5 file not found: {h5_file}")
        else:
            # Fall back to looking for .npy files
            repr_files = sorted(repr_path.glob("*.npy"))
            if not repr_files:
                raise FileNotFoundError(f"No representation files found in {repr_path}")

            print(f"Found {len(repr_files)} .npy representation files")

            # Load representations (should be stacked histograms)
            representations = []
            for file_path in repr_files:
                repr_data = np.load(file_path)
                representations.append(repr_data)

            representations = np.stack(representations, axis=0)
            print(f"Loaded representations shape: {representations.shape}")
            timestamps = None

        # Load labels if available
        labels_path = sequence_path / "labels_v2"
        metadata = {"sequence_path": str(sequence_path)}

        if labels_path.exists():
            labels_file = labels_path / "labels.npz"

            if labels_file.exists():
                labels_data = np.load(labels_file)
                metadata["labels"] = labels_data
                print(f"Loaded labels: {list(labels_data.keys())}")

            # Also check for timestamps in labels directory
            if timestamps is None:
                timestamps_file = labels_path / "timestamps_us.npy"
                if timestamps_file.exists():
                    timestamps = np.load(timestamps_file)
                    print(f"Loaded {len(timestamps)} timestamps from labels")

        if timestamps is not None:
            metadata["timestamps"] = timestamps

        return representations, metadata

    def load_raw_events(self, file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load raw event data from various formats.

        Args:
            file_path: Path to event file

        Returns:
            Tuple of (xs, ys, ts, ps)
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Event file not found: {file_path}")

        print(f"Loading events from: {file_path}")

        # Use evlib to load events
        try:
            # Try loading with evlib
            t, x, y, polarity = evlib.formats.load_events(str(file_path))

            # Convert to expected format
            xs = x.astype(np.int64)
            ys = y.astype(np.int64)
            ts = t.astype(np.float64)
            ps = polarity.astype(np.int64)

            print(f"✓ Loaded {len(xs)} events")
            print(f"  - Time range: {ts.min():.6f} - {ts.max():.6f} seconds")
            print(f"  - Spatial range: x=[{xs.min()}, {xs.max()}], y=[{ys.min()}, {ys.max()}]")
            print(f"  - Polarities: {np.unique(ps)}")

            return xs, ys, ts, ps

        except Exception as e:
            print(f"✗ Failed to load events with evlib: {e}")
            raise

    def run_inference_on_representations(
        self,
        representations: np.ndarray,
        sequence_length: int = 5,
        reset_states_every: int = 10,
    ) -> List[List[Dict[str, Any]]]:
        """Run inference on preprocessed representations.

        Args:
            representations: Preprocessed representations (T, C, H, W)
            sequence_length: Length of temporal sequences
            reset_states_every: Reset LSTM states every N sequences

        Returns:
            List of detection results for each timestep
        """
        T, C, H, W = representations.shape
        print(f"Running inference on {T} timesteps, {C} channels, {H}x{W} resolution")

        # Convert to torch tensors
        representations_torch = torch.from_numpy(representations).float().to(self.device)

        all_detections = []
        total_inference_time = 0

        # Process in sequences
        for seq_start in range(0, T, sequence_length):
            seq_end = min(seq_start + sequence_length, T)
            seq_repr = representations_torch[seq_start:seq_end]

            # Reset states periodically
            if seq_start % reset_states_every == 0:
                self.model.reset_states()

            print(
                f"Processing sequence {seq_start//sequence_length + 1}/{(T + sequence_length - 1)//sequence_length}"
            )

            # Run inference on sequence
            sequence_detections = []
            for t_idx in range(seq_repr.shape[0]):
                frame_repr = seq_repr[t_idx].unsqueeze(0)  # Add batch dimension

                start_time = time.time()
                with torch.no_grad():
                    predictions, _, _ = self.model.forward(frame_repr, retrieve_detections=True)

                inference_time = time.time() - start_time
                total_inference_time += inference_time

                # Post-process predictions
                detections = self.model._postprocess_predictions(
                    predictions, self.confidence_threshold, self.nms_threshold
                )

                sequence_detections.append(detections)

                if len(detections) > 0:
                    print(f"  Frame {seq_start + t_idx}: {len(detections)} detections")

            all_detections.extend(sequence_detections)

        avg_inference_time = total_inference_time / T
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0

        print("\n✓ Inference completed!")
        print(f"  - Average inference time: {avg_inference_time*1000:.2f} ms/frame")
        print(f"  - Effective FPS: {fps:.1f}")
        print(f"  - Total detections: {sum(len(dets) for dets in all_detections)}")

        return all_detections

    def run_inference_on_events(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        ts: np.ndarray,
        ps: np.ndarray,
        height: int = 480,
        width: int = 640,
        time_window: float = 0.05,  # 50ms windows
    ) -> List[Dict[str, Any]]:
        """Run inference on raw event data.

        Args:
            xs, ys, ts, ps: Event arrays
            height: Image height
            width: Image width
            time_window: Time window for each inference step (seconds)

        Returns:
            List of detection results
        """
        print("Running inference on raw events")
        print(f"  - Events: {len(xs)}")
        print(f"  - Time range: {ts.min():.6f} - {ts.max():.6f} seconds")
        print(f"  - Window size: {time_window:.3f} seconds")

        # Create time windows
        t_start = ts.min()
        t_end = ts.max()
        time_steps = np.arange(t_start, t_end, time_window)

        all_detections = []
        total_inference_time = 0

        # Reset states
        self.model.reset_states()

        for i, t_window_start in enumerate(time_steps[:-1]):
            t_window_end = time_steps[i + 1]

            # Get events in time window
            mask = (ts >= t_window_start) & (ts < t_window_end)
            window_events = (xs[mask], ys[mask], ts[mask], ps[mask])

            if len(window_events[0]) == 0:
                # No events in window
                all_detections.append([])
                continue

            print(f"Window {i+1}/{len(time_steps)-1}: {len(window_events[0])} events")

            # Run detection
            start_time = time.time()
            detections = self.model.detect(
                window_events,
                height=height,
                width=width,
                confidence_threshold=self.confidence_threshold,
                nms_threshold=self.nms_threshold,
                reset_states=False,  # Keep temporal continuity
            )
            inference_time = time.time() - start_time
            total_inference_time += inference_time

            all_detections.append(detections)

            if len(detections) > 0:
                print(f"  → {len(detections)} detections")

        avg_inference_time = total_inference_time / len(time_steps)
        fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0

        print("\n✓ Inference completed!")
        print(f"  - Average inference time: {avg_inference_time*1000:.2f} ms/window")
        print(f"  - Effective FPS: {fps:.1f}")
        print(f"  - Total detections: {sum(len(dets) for dets in all_detections)}")

        return all_detections

    def save_results(self, detections: List, output_path: str, metadata: Optional[Dict] = None):
        """Save detection results to file.

        Args:
            detections: Detection results
            output_path: Output file path
            metadata: Optional metadata to save
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare results
        results = {
            "detections": detections,
            "model_variant": self.model_variant,
            "confidence_threshold": self.confidence_threshold,
            "nms_threshold": self.nms_threshold,
            "num_timesteps": len(detections),
            "total_detections": sum(len(dets) for dets in detections),
        }

        if metadata:
            results["metadata"] = metadata

        # Save as numpy archive
        np.savez_compressed(output_path, **results)
        print(f"✓ Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="RVT Inference Script")

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
        "--confidence_threshold", type=float, default=0.1, help="Detection confidence threshold"
    )
    parser.add_argument("--nms_threshold", type=float, default=0.45, help="NMS threshold")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cpu/cuda)")

    # Data configuration
    parser.add_argument(
        "--data_path",
        type=str,
        default="/Users/tallam/github/tallamjr/origin/evlib/data/gen4_1mpx_processed_RVT/val",
        help="Path to RVT validation data directory",
    )
    parser.add_argument("--sequence_path", type=str, default=None, help="Path to specific sequence directory")
    parser.add_argument("--event_file", type=str, default=None, help="Path to raw event file")

    # Inference configuration
    parser.add_argument("--sequence_length", type=int, default=5, help="Temporal sequence length")
    parser.add_argument(
        "--time_window", type=float, default=0.05, help="Time window for raw event processing (seconds)"
    )
    parser.add_argument("--height", type=int, default=480, help="Image height")
    parser.add_argument("--width", type=int, default=640, help="Image width")

    # Output configuration
    parser.add_argument(
        "--output_dir", type=str, default="outputs/rvt_inference", help="Output directory for results"
    )
    parser.add_argument(
        "--max_sequences", type=int, default=None, help="Maximum number of sequences to process"
    )

    args = parser.parse_args()

    # Create inference engine
    inference_engine = RVTInference(
        model_variant=args.model_variant,
        pretrained=args.pretrained,
        confidence_threshold=args.confidence_threshold,
        nms_threshold=args.nms_threshold,
        device=args.device,
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.event_file:
        # Process single raw event file
        print(f"\n{'='*60}")
        print("PROCESSING RAW EVENT FILE")
        print(f"{'='*60}")

        xs, ys, ts, ps = inference_engine.load_raw_events(args.event_file)
        detections = inference_engine.run_inference_on_events(
            xs, ys, ts, ps, args.height, args.width, args.time_window
        )

        # Save results
        output_file = output_dir / f"raw_events_results_{args.model_variant}.npz"
        inference_engine.save_results(detections, output_file)

    elif args.sequence_path:
        # Process single preprocessed sequence
        print(f"\n{'='*60}")
        print("PROCESSING SINGLE SEQUENCE")
        print(f"{'='*60}")

        representations, metadata = inference_engine.load_preprocessed_sequence(args.sequence_path)
        detections = inference_engine.run_inference_on_representations(representations, args.sequence_length)

        # Save results
        sequence_name = Path(args.sequence_path).name
        output_file = output_dir / f"{sequence_name}_results_{args.model_variant}.npz"
        inference_engine.save_results(detections, output_file, metadata)

    else:
        # Process validation dataset
        print(f"\n{'='*60}")
        print("PROCESSING VALIDATION DATASET")
        print(f"{'='*60}")

        data_path = Path(args.data_path)
        if not data_path.exists():
            print(f"✗ Data path does not exist: {data_path}")
            return

        # Find all sequences
        sequences = [d for d in data_path.iterdir() if d.is_dir()]
        sequences = sorted(sequences)

        if args.max_sequences:
            sequences = sequences[: args.max_sequences]

        print(f"Found {len(sequences)} sequences to process")

        for i, sequence_path in enumerate(sequences):
            print(f"\n--- Processing sequence {i+1}/{len(sequences)}: {sequence_path.name} ---")

            try:
                representations, metadata = inference_engine.load_preprocessed_sequence(sequence_path)
                detections = inference_engine.run_inference_on_representations(
                    representations, args.sequence_length
                )

                # Save results
                output_file = output_dir / f"{sequence_path.name}_results_{args.model_variant}.npz"
                inference_engine.save_results(detections, output_file, metadata)

            except Exception as e:
                print(f"✗ Failed to process sequence {sequence_path.name}: {e}")
                continue

    print(f"\n🎉 Inference completed! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
