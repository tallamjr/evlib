#!/usr/bin/env python3
"""
E2VID Video Reconstruction CLI

Command-line tool for reconstructing videos from event camera data using the E2VID model.
Supports multiple event formats (EVT2, H5, text) and provides configurable reconstruction parameters.

Example usage:
    python scripts/e2vid.py --input data/prophersee/samples/evt2/80_balls.raw --output 80_balls_reconstructed.mp4
    python scripts/e2vid.py --input data/slider_depth/events.txt --output slider_depth.mp4 --fps 30 --duration 2.0
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, List, Tuple
import logging

import numpy as np
import cv2

try:
    import evlib
    import evlib.models
    import torch
except ImportError as e:
    print(f"Error: Could not import required packages: {e}")
    print("Please ensure evlib and torch are installed: pip install -e . torch")
    sys.exit(1)


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    return logging.getLogger(__name__)


def find_active_periods(
    input_path: str, stats: dict, logger: logging.Logger, min_events: int = 1000
) -> List[Tuple[float, float]]:
    """Find periods with significant event activity using sampling."""

    duration = stats["end_time"] - stats["start_time"]
    window_size = 0.5  # 500ms windows for faster analysis
    max_windows = 40  # Limit analysis to 40 windows maximum

    if duration > max_windows * window_size:
        # Sample windows across the duration
        step_size = duration / max_windows
        sample_times = [stats["start_time"] + i * step_size for i in range(max_windows)]
    else:
        # Use regular intervals
        num_windows = int(duration / window_size)
        sample_times = [stats["start_time"] + i * window_size for i in range(num_windows)]

    logger.info(f"Sampling {len(sample_times)} windows across {duration:.1f}s to find active periods...")

    active_periods = []
    window_activities = []

    # Quick sampling to find activity levels
    for t_start in sample_times:
        t_end = t_start + window_size
        window_events = evlib.load_events(input_path, t_start=t_start, t_end=t_end)
        event_count = len(window_events.collect())
        window_activities.append((t_start, event_count))

        if len(window_activities) % 10 == 0:
            logger.info(f"  Analyzed {len(window_activities)}/{len(sample_times)} windows...")

    # Find continuous active regions
    import numpy as np

    active_threshold = max(min_events, np.percentile([count for _, count in window_activities], 75))
    logger.info(f"Using activity threshold: {active_threshold:,} events per {window_size*1000:.0f}ms window")

    current_start = None
    for t_start, count in window_activities:
        if count >= active_threshold:
            if current_start is None:
                current_start = t_start
        else:
            if current_start is not None:
                active_periods.append((current_start, t_start))
                current_start = None

    # Handle case where file ends during active period
    if current_start is not None:
        active_periods.append((current_start, stats["end_time"]))

    # Merge nearby periods and extend slightly
    merged_periods = []
    for start, end in active_periods:
        # Extend periods slightly
        extended_start = max(stats["start_time"], start - 0.5)
        extended_end = min(stats["end_time"], end + 0.5)

        # Merge with previous if close
        if merged_periods and extended_start - merged_periods[-1][1] < 1.0:
            merged_periods[-1] = (merged_periods[-1][0], extended_end)
        else:
            merged_periods.append((extended_start, extended_end))

    total_active_duration = sum(end - start for start, end in merged_periods)
    logger.info(f"Found {len(merged_periods)} active periods totaling {total_active_duration:.1f}s")

    for i, (start, end) in enumerate(merged_periods):
        logger.info(f"  Period {i+1}: {start:.3f} - {end:.3f}s ({end-start:.1f}s)")

    return merged_periods


def load_and_analyze_events(input_path: str, logger: logging.Logger) -> Tuple[object, dict]:
    """Load events and analyze temporal/spatial properties."""
    logger.info(f"Loading events from: {input_path}")

    # Load all events to analyze properties
    all_events = evlib.load_events(input_path)
    events_df = all_events.collect()

    # Analyze properties
    num_events = len(events_df)
    timestamps_sec = events_df["timestamp"].to_numpy().astype(np.float64) / 1e6
    duration = timestamps_sec.max() - timestamps_sec.min()

    width = int(events_df["x"].max()) + 1
    height = int(events_df["y"].max()) + 1

    # Polarity distribution
    polarities = events_df["polarity"].to_numpy()
    pos_events = np.sum(polarities == 1)
    neg_events = np.sum(polarities == 0) + np.sum(polarities == -1)

    stats = {
        "num_events": num_events,
        "duration": duration,
        "width": width,
        "height": height,
        "event_rate": num_events / duration if duration > 0 else 0,
        "start_time": timestamps_sec.min(),
        "end_time": timestamps_sec.max(),
        "pos_events": pos_events,
        "neg_events": neg_events,
    }

    logger.info("Dataset analysis:")
    logger.info(f"  Events: {num_events:,}")
    logger.info(f"  Duration: {duration:.3f} seconds")
    logger.info(f"  Resolution: {width}x{height}")
    logger.info(f"  Event rate: {stats['event_rate']:.0f} events/second")
    logger.info(f"  Polarity: {pos_events:,} positive, {neg_events:,} negative")

    return all_events, stats


def detect_and_setup_device(device: Optional[str], logger: logging.Logger) -> str:
    """Detect and set up the best available device for inference."""

    if device is not None:
        # User specified device
        if device == "auto":
            device = None  # Will be auto-detected below
        else:
            # Validate user-specified device
            try:
                torch.device(device)
                logger.info(f"Using user-specified device: {device}")
                return device
            except Exception as e:
                logger.warning(f"Invalid device '{device}': {e}. Auto-detecting...")
                device = None

    if device is None:
        # Auto-detect best device
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"Using CUDA GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using Metal Performance Shaders (MPS) for GPU acceleration")
        else:
            device = "cpu"
            logger.info("Using CPU (no GPU acceleration available)")

    return device


def create_model(pretrained: bool, device: str, logger: logging.Logger) -> object:
    """Create and initialize E2VID model."""
    logger.info("Initializing E2VID model...")

    # Create model
    model = evlib.models.E2VID(pretrained=pretrained)

    # Move model to specified device
    if hasattr(model, "_model") and model._model is not None:
        model._model = model._model.to(device)
        model._device = torch.device(device)
        logger.info(f"Model moved to device: {device}")

    logger.info(f"Model: {model}")

    return model


def reconstruct_frames(
    input_path: str,
    model: object,
    stats: dict,
    fps: float,
    duration: Optional[float],
    start_time: Optional[float],
    max_resolution: Optional[int],
    logger: logging.Logger,
) -> List[np.ndarray]:
    """Reconstruct video frames from events."""

    # Calculate temporal parameters
    frame_duration = 1.0 / fps
    data_start_time = stats["start_time"]
    data_end_time = stats["end_time"]

    # Determine reconstruction window
    if start_time is not None:
        recon_start = data_start_time + start_time
    else:
        recon_start = data_start_time

    if duration is not None:
        recon_end = recon_start + duration
    else:
        recon_end = data_end_time

    # Ensure we don't exceed data bounds
    recon_start = max(recon_start, data_start_time)
    recon_end = min(recon_end, data_end_time)

    if recon_start >= recon_end:
        logger.error(f"Invalid time range: {recon_start:.3f} - {recon_end:.3f}")
        return []

    actual_duration = recon_end - recon_start
    num_frames = int(actual_duration / frame_duration)

    # Resolution limiting for performance
    original_width, original_height = stats["width"], stats["height"]
    if max_resolution and max(original_width, original_height) > max_resolution:
        # Calculate scale factor to limit resolution
        scale_factor = max_resolution / max(original_width, original_height)
        target_width = int(original_width * scale_factor)
        target_height = int(original_height * scale_factor)
        logger.info(
            f"Limiting resolution: {original_width}x{original_height} → {target_width}x{target_height}"
        )
    else:
        target_width, target_height = original_width, original_height

    logger.info("Reconstruction parameters:")
    logger.info(f"  Time range: {recon_start:.3f} - {recon_end:.3f} seconds")
    logger.info(f"  Duration: {actual_duration:.3f} seconds")
    logger.info(f"  Resolution: {target_width}x{target_height}")
    logger.info(f"  FPS: {fps}")
    logger.info(f"  Frame duration: {frame_duration*1000:.0f}ms")
    logger.info(f"  Target frames: {num_frames}")

    # Analyze event distribution to warn about sparse periods
    logger.info("Analyzing event distribution...")
    sample_windows = min(20, num_frames)
    empty_windows = 0
    for i in range(sample_windows):
        t_start_sample = recon_start + (i / sample_windows) * actual_duration
        t_end_sample = t_start_sample + frame_duration
        sample_events = evlib.load_events(input_path, t_start=t_start_sample, t_end=t_end_sample)
        if len(sample_events.collect()) == 0:
            empty_windows += 1

    if empty_windows > sample_windows * 0.3:  # >30% empty
        logger.warning(
            f"Detected sparse event data: {empty_windows}/{sample_windows} sample windows are empty"
        )
        logger.warning("This is normal for event cameras during static periods")
        logger.warning("Consider using longer frame duration (lower FPS) or selecting active time segments")

    # Reconstruct frames
    frames = []
    successful_frames = 0

    logger.info("Starting frame reconstruction...")
    start_recon_time = time.time()

    for i in range(num_frames):
        t_start = recon_start + i * frame_duration
        t_end = t_start + frame_duration

        try:
            # Load events for this time window
            frame_events = evlib.load_events(input_path, t_start=t_start, t_end=t_end)
            events_count = len(frame_events.collect())

            if events_count == 0:
                logger.warning(f"Frame {i+1}/{num_frames}: No events, using previous frame or black")
                # Use previous frame or create black frame
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(np.zeros((target_height, target_width), dtype=np.float32))
                continue

            # Reconstruct frame
            frame = model.reconstruct(frame_events)

            # Apply resolution limiting if specified
            if max_resolution and max(original_width, original_height) > max_resolution:
                import cv2

                frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

            # Enhance contrast for better visualization
            frame_enhanced = (frame - frame.min()) / (frame.max() - frame.min())
            frames.append(frame_enhanced)
            successful_frames += 1

            # Progress update
            if (i + 1) % max(1, num_frames // 20) == 0 or i == num_frames - 1:
                elapsed = time.time() - start_recon_time
                progress = (i + 1) / num_frames
                eta = elapsed / progress - elapsed if progress > 0 else 0

                logger.info(
                    f"Progress: {i+1}/{num_frames} ({progress*100:.1f}%) - "
                    f"{events_count:,} events - ETA: {eta:.1f}s"
                )

        except Exception as e:
            logger.error(f"Error reconstructing frame {i+1}: {e}")
            # Use previous frame or black frame as fallback
            if frames:
                frames.append(frames[-1].copy())
            else:
                frames.append(np.zeros((target_height, target_width), dtype=np.float32))

    total_recon_time = time.time() - start_recon_time
    logger.info(
        f"Reconstruction complete: {successful_frames}/{num_frames} frames in {total_recon_time:.1f}s"
    )
    logger.info(f"Average: {total_recon_time/len(frames):.3f}s per frame")

    return frames


def save_video(
    frames: List[np.ndarray], output_path: str, fps: float, quality: str, logger: logging.Logger
) -> bool:
    """Save frames as MP4 video."""

    if not frames:
        logger.error("No frames to save")
        return False

    logger.info(f"Saving video to: {output_path}")

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Video parameters
    height, width = frames[0].shape

    # Quality settings
    quality_settings = {
        "low": {"bitrate": 1000, "crf": 28},
        "medium": {"bitrate": 2000, "crf": 23},
        "high": {"bitrate": 5000, "crf": 18},
        "lossless": {"bitrate": 10000, "crf": 0},
    }

    # Get quality settings (for future bitrate configuration)
    _ = quality_settings.get(quality, quality_settings["medium"])

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

    if not video_writer.isOpened():
        logger.error("Failed to open video writer")
        return False

    logger.info("Video parameters:")
    logger.info(f"  Resolution: {width}x{height}")
    logger.info(f"  FPS: {fps}")
    logger.info(f"  Frames: {len(frames)}")
    logger.info(f"  Quality: {quality}")

    # Write frames
    for i, frame in enumerate(frames):
        # Convert to 8-bit grayscale
        frame_8bit = (frame * 255).astype(np.uint8)
        video_writer.write(frame_8bit)

        if (i + 1) % max(1, len(frames) // 10) == 0:
            logger.info(f"Writing: {i+1}/{len(frames)} frames")

    video_writer.release()

    # Verify output file
    if Path(output_path).exists():
        file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        logger.info(f"Video saved successfully: {file_size_mb:.1f} MB")
        return True
    else:
        logger.error("Failed to save video file")
        return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="E2VID Video Reconstruction from Event Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic reconstruction
  python scripts/e2vid.py --input data/events.txt --output video.mp4

  # High quality, 30 FPS reconstruction
  python scripts/e2vid.py --input data/80_balls.raw --output 80_balls.mp4 --fps 30 --quality high

  # Reconstruct specific time segment
  python scripts/e2vid.py --input data/events.h5 --output segment.mp4 --start 1.0 --duration 5.0

  # Fast reconstruction without pretrained weights
  python scripts/e2vid.py --input data/events.txt --output video.mp4 --no-pretrained --fps 15

  # GPU-accelerated reconstruction (auto-detects CUDA/MPS)
  python scripts/e2vid.py --input data/80_balls.raw --output 80_balls.mp4 --device auto

  # Force CPU inference
  python scripts/e2vid.py --input data/events.txt --output video.mp4 --device cpu

  # Auto-detect most active period (good for sparse data)
  python scripts/e2vid.py --input data/pedestrians.raw --output pedestrians.mp4 --auto-active
        """,
    )

    # Required arguments
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=str,
        help="Input event data file (EVT2, H5, text formats supported)",
    )
    parser.add_argument("--output", "-o", required=True, type=str, help="Output video file path (.mp4)")

    # Reconstruction parameters
    parser.add_argument("--fps", type=float, default=20.0, help="Output video frame rate (default: 20)")
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Duration to reconstruct in seconds (default: entire file)",
    )
    parser.add_argument(
        "--start", type=float, default=None, help="Start time offset in seconds (default: beginning of file)"
    )

    # Model parameters
    parser.add_argument(
        "--no-pretrained", action="store_true", help="Use randomly initialized weights instead of pretrained"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for inference: auto, cpu, cuda, mps (default: auto)",
    )
    parser.add_argument(
        "--max-resolution",
        type=int,
        default=None,
        help="Limit maximum resolution for performance (e.g., 1024)",
    )
    parser.add_argument(
        "--auto-active", action="store_true", help="Automatically detect and use most active time period"
    )

    # Output parameters
    parser.add_argument(
        "--quality",
        choices=["low", "medium", "high", "lossless"],
        default="medium",
        help="Video quality setting (default: medium)",
    )

    # General options
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.verbose)

    # Validate arguments
    if not Path(args.input).exists():
        logger.error(f"Input file does not exist: {args.input}")
        sys.exit(1)

    if not args.output.endswith(".mp4"):
        logger.warning("Output file should have .mp4 extension for best compatibility")

    if args.fps <= 0 or args.fps > 120:
        logger.error("FPS must be between 0 and 120")
        sys.exit(1)

    # Start processing
    logger.info("=== E2VID Video Reconstruction ===")
    total_start_time = time.time()

    try:
        # Step 1: Load and analyze events
        all_events, stats = load_and_analyze_events(args.input, logger)

        # Step 1b: Handle auto-active period detection
        if args.auto_active and args.start is None and args.duration is None:
            logger.info("Auto-detecting most active period...")
            active_periods = find_active_periods(args.input, stats, logger)

            if active_periods:
                # Use the longest active period
                longest_period = max(active_periods, key=lambda x: x[1] - x[0])
                args.start = longest_period[0] - stats["start_time"]  # Convert to offset
                args.duration = min(longest_period[1] - longest_period[0], 30.0)  # Max 30s
                logger.info(
                    f"Using most active period: {args.start:.3f}s offset, {args.duration:.1f}s duration"
                )
            else:
                logger.warning("No active periods found, using default time range")

        # Step 2: Setup device and create model
        device = detect_and_setup_device(args.device, logger)
        model = create_model(not args.no_pretrained, device, logger)

        # Step 3: Reconstruct frames
        frames = reconstruct_frames(
            args.input, model, stats, args.fps, args.duration, args.start, args.max_resolution, logger
        )

        if not frames:
            logger.error("No frames were reconstructed")
            sys.exit(1)

        # Step 4: Save video
        success = save_video(frames, args.output, args.fps, args.quality, logger)

        if not success:
            logger.error("Failed to save video")
            sys.exit(1)

        # Summary
        total_time = time.time() - total_start_time
        logger.info("=== Reconstruction Complete ===")
        logger.info(f"Total time: {total_time:.1f} seconds")
        logger.info(f"Output: {args.output}")

    except KeyboardInterrupt:
        logger.info("Reconstruction interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Reconstruction failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
