#!/usr/bin/env python3
"""
ESIM Video-to-Events Converter

Convert video files to event camera data using the ESIM algorithm.
This is a modernized version of the original process_frames.py script
using the new evlib.simulation module.
"""

import argparse
import sys
from pathlib import Path
import time


def check_dependencies():
    """Check if required dependencies are available."""
    missing = []

    try:
        import torch

        # Check available acceleration
        available_devices = ["CPU"]
        if torch.cuda.is_available():
            available_devices.append("CUDA")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            available_devices.append("MPS")

        print(f"Available PyTorch devices: {', '.join(available_devices)}")

        if len(available_devices) == 1:  # Only CPU
            print(
                "Warning: No GPU acceleration available. Processing will run on CPU only."
            )
    except ImportError:
        missing.append("torch")

    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")

    if missing:
        print(f"Error: Missing required dependencies: {', '.join(missing)}")
        print("\nInstallation instructions:")
        for dep in missing:
            if dep == "torch":
                print("  PyTorch: https://pytorch.org/get-started/locally/")
            elif dep == "opencv-python":
                print("  OpenCV: pip install opencv-python")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert a video file to event-based data using ESIM algorithm with GPU acceleration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input/Output
    parser.add_argument("video_file", help="Path to the input video file")
    parser.add_argument(
        "-o",
        "--output_file",
        default="events_esim.h5",
        help="Path to the output HDF5 event file",
    )

    # ESIM Configuration
    parser.add_argument(
        "--cp",
        "--positive_threshold",
        type=float,
        default=0.4,
        help="Positive contrast threshold",
    )
    parser.add_argument(
        "--cn",
        "--negative_threshold",
        type=float,
        default=0.4,
        help="Negative contrast threshold",
    )
    parser.add_argument(
        "--refractory_period",
        type=float,
        default=0.1,
        help="Refractory period in milliseconds",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Computing device",
    )

    # Video Configuration
    parser.add_argument(
        "--width", type=int, default=640, help="Width to resize video frames"
    )
    parser.add_argument(
        "--height", type=int, default=480, help="Height to resize video frames"
    )
    parser.add_argument(
        "--fps", type=float, help="Override video FPS (use original if not specified)"
    )
    parser.add_argument("--start_time", type=float, help="Start time in seconds")
    parser.add_argument("--end_time", type=float, help="End time in seconds")
    parser.add_argument(
        "--frame_skip",
        type=int,
        default=0,
        help="Number of frames to skip between processed frames",
    )

    # Output options
    parser.add_argument(
        "--output_dir", default="h5", help="Output directory for results"
    )
    parser.add_argument(
        "--estimate_only",
        action="store_true",
        help="Only estimate event count without processing entire video",
    )
    parser.add_argument(
        "--sample_frames",
        type=int,
        default=100,
        help="Number of frames to sample for estimation (when --estimate_only)",
    )
    parser.add_argument(
        "--video_info",
        action="store_true",
        help="Show video information and processing parameters",
    )

    # Performance options
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming processing (lower memory usage)",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        default=True,
        help="Show progress information",
    )

    args = parser.parse_args()

    # Check dependencies first
    if not check_dependencies():
        return 1

    # Import evlib after dependency check
    try:
        import evlib
        from evlib.simulation import (
            ESIMConfig,
            VideoConfig,
            VideoToEvents,
            estimate_event_count,
        )
    except ImportError as e:
        print(f"Error importing evlib: {e}")
        print("Make sure evlib is properly installed with simulation support.")
        return 1

    # Validate input file
    video_path = Path(args.video_file)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return 1

    # Create configurations
    esim_config = ESIMConfig(
        positive_threshold=args.cp,
        negative_threshold=args.cn,
        refractory_period_ms=args.refractory_period,
        device=args.device,
    )

    video_config = VideoConfig(
        width=args.width,
        height=args.height,
        fps=args.fps,
        start_time=args.start_time,
        end_time=args.end_time,
        frame_skip=args.frame_skip,
    )

    # Create processor
    processor = VideoToEvents(esim_config, video_config)

    # Show video information if requested
    if args.video_info:
        try:
            info = processor.get_video_info(video_path)
            print("\n=== Video Information ===")
            print(f"File: {info['path']}")
            print(f"Resolution: {info['width']}x{info['height']}")
            print(f"FPS: {info['fps']:.2f}")
            print(f"Frame count: {info['frame_count']:,}")
            print(f"Duration: {info['duration_seconds']:.2f} seconds")

            print("\n=== Processing Configuration ===")
            proc = info["processing"]
            print(f"Target resolution: {proc['target_width']}x{proc['target_height']}")
            print(f"Target FPS: {proc['target_fps']:.2f}")
            print(f"Effective FPS: {proc['effective_fps']:.2f}")
            print(f"Frame skip: {proc['frame_skip']}")
            print(f"Grayscale: {proc['grayscale']}")

            print("\n=== ESIM Configuration ===")
            print(f"Positive threshold: {esim_config.positive_threshold}")
            print(f"Negative threshold: {esim_config.negative_threshold}")
            print(f"Refractory period: {esim_config.refractory_period_ms} ms")
            print(f"Device: {esim_config.device}")

            # Show actual device that will be used
            temp_processor = VideoToEvents(esim_config, VideoConfig())
            actual_device = temp_processor.simulator.device
            if str(actual_device) != esim_config.device:
                print(f"Actual device: {actual_device}")
            print()

        except Exception as e:
            print(f"Error getting video info: {e}")

    # Estimate event count if requested
    if args.estimate_only:
        print("Estimating event count...")
        try:
            start_time = time.time()
            estimate = estimate_event_count(
                video_path, esim_config, video_config, args.sample_frames
            )
            estimate_time = time.time() - start_time

            print("\n=== Event Count Estimation ===")
            if estimate["estimated"]:
                print(f"Estimated total events: {estimate['estimated_total_events']:,}")
                print(f"Sample frames: {estimate['sample_frames']:,}")
                print(f"Sample events: {estimate['sample_events']:,}")
            else:
                print(f"Actual total events: {estimate['total_events']:,}")
                print("(Processed entire video)")

            print(f"Events per frame: {estimate['events_per_frame']:.1f}")
            print(f"Estimation time: {estimate_time:.2f} seconds")

        except Exception as e:
            print(f"Error during estimation: {e}")
            return 1

        return 0

    # Process video
    print("Converting video to events using ESIM algorithm...")
    print(f"Input: {video_path}")
    print(f"Output: {args.output_file}")

    try:
        import numpy as np

        start_time = time.time()

        if args.streaming:
            # Streaming mode (lower memory usage)
            print("Using streaming mode...")
            all_events = [[], [], [], []]  # x, y, t, polarity

            event_count = 0
            for frame_events in processor.process_frames_streaming(video_path):
                if len(frame_events[0]) > 0:
                    for i in range(4):
                        all_events[i].extend(frame_events[i])
                    event_count += len(frame_events[0])

                    if args.progress and event_count % 100000 == 0:
                        print(f"Processed {event_count:,} events...")

            # Convert to numpy arrays
            x_np = np.array(all_events[0], dtype=np.int64)
            y_np = np.array(all_events[1], dtype=np.int64)
            t_np = np.array(all_events[2], dtype=np.float64)
            p_np = np.array(all_events[3], dtype=np.int64)

        else:
            # Batch mode (faster but higher memory usage)
            x_np, y_np, t_np, p_np = processor.process_video(video_path)

        processing_time = time.time() - start_time

        if len(x_np) == 0:
            print("\nNo events were generated. Nothing to save.")
            print("Try adjusting the contrast thresholds (--cp and --cn).")
            return 0

        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)

        # Save results
        output_path = output_dir / args.output_file
        print(f"\nSaving {len(x_np):,} events to {output_path}...")

        evlib.formats.save_events_to_hdf5(x_np, y_np, t_np, p_np, str(output_path))

        # Print summary
        print("\n=== Processing Complete ===")
        print(f"Total events generated: {len(x_np):,}")
        print(f"Processing time: {processing_time:.2f} seconds")
        if processing_time > 0:
            print(f"Events per second: {len(x_np) / processing_time:,.0f}")

        print("\n=== Event Statistics ===")
        print(f"Time range: {t_np.min():.6f} - {t_np.max():.6f} seconds")
        print(f"Duration: {t_np.max() - t_np.min():.6f} seconds")
        print(f"X range: {x_np.min()} - {x_np.max()}")
        print(f"Y range: {y_np.min()} - {y_np.max()}")

        positive_events = np.sum(p_np == 1)
        negative_events = np.sum(p_np == -1)
        print(
            f"Positive events: {positive_events:,} ({positive_events / len(x_np) * 100:.1f}%)"
        )
        print(
            f"Negative events: {negative_events:,} ({negative_events / len(x_np) * 100:.1f}%)"
        )

        print(f"\nOutput saved to: {output_path}")

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
        return 1
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
