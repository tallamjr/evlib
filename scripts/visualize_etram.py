#!/usr/bin/env python3
"""
eTram Event Data Visualization CLI

Command-line tool for visualizing eTram processed event camera data as videos.
Provides similar visualization to the WASM demo with red/blue polarity rendering.

Example usage:
    python scripts/visualize_etram.py --input data/eTram_processed/test/test_day_001 --output test_day_001.mp4
    python scripts/visualize_etram.py --batch data/eTram_processed/test --output-dir outputs/videos/
    python scripts/visualize_etram.py --input data/eTram_processed/test/test_day_001 --fps 60 --decay 50
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional
import time

try:
    import evlib.visualization as viz
    import cv2

    # Ensure HDF5 plugins are available for eTram data
    try:
        import hdf5plugin
    except ImportError:
        print("Warning: hdf5plugin not available - may have issues with compressed eTram HDF5 files")
        print("Install with: pip install hdf5plugin")

except ImportError as e:
    print(f"Error: Could not import required packages: {e}")
    print("Please ensure evlib is installed with visualization dependencies: pip install -e .[plot]")
    sys.exit(1)


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    return logging.getLogger(__name__)


def parse_resolution(resolution_str: str) -> tuple[int, int]:
    """Parse resolution string like '640x480' to (width, height)."""
    try:
        width, height = map(int, resolution_str.split("x"))
        return width, height
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Resolution must be in format 'WIDTHxHEIGHT', got '{resolution_str}'"
        )


def parse_color(color_str: str) -> tuple[int, int, int]:
    """Parse color string like '255,0,0' to (B, G, R) tuple for OpenCV."""
    try:
        r, g, b = map(int, color_str.split(","))
        return (b, g, r)  # Convert RGB to BGR for OpenCV
    except ValueError:
        raise argparse.ArgumentTypeError(f"Color must be in format 'R,G,B', got '{color_str}'")


def validate_paths(args) -> None:
    """Validate input and output paths."""
    if args.batch:
        if not Path(args.batch).is_dir():
            raise FileNotFoundError(f"Batch input directory does not exist: {args.batch}")
        if not args.output_dir:
            raise ValueError("--output-dir is required when using --batch")
    else:
        if not args.input:
            raise ValueError("--input is required when not using --batch")
        if not Path(args.input).exists():
            raise FileNotFoundError(f"Input path does not exist: {args.input}")
        if not args.output:
            raise ValueError("--output is required when not using --batch")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize eTram event camera data as videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file processing
  python scripts/visualize_etram.py --input data/eTram_processed/test/test_day_001 --output test.mp4

  # Batch processing
  python scripts/visualize_etram.py --batch data/eTram_processed/test --output-dir outputs/

  # Custom parameters
  python scripts/visualize_etram.py --input data/eTram_processed/test/test_day_001 \\
    --output test.mp4 --fps 60 --decay 50 --resolution 1280x720

  # Time range selection
  python scripts/visualize_etram.py --input data/eTram_processed/test/test_day_001 \\
    --output test.mp4 --start-time 10.0 --duration 5.0
        """,
    )

    # Input/output options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", "-i", type=str, help="Path to eTram data directory or HDF5 file")
    input_group.add_argument(
        "--batch", "-b", type=str, help="Process all eTram data directories in this path"
    )

    parser.add_argument(
        "--output", "-o", type=str, help="Output video file path (required for single file processing)"
    )
    parser.add_argument(
        "--output-dir", type=str, help="Output directory for batch processing (required for batch mode)"
    )

    # Video parameters
    parser.add_argument("--fps", type=float, default=30.0, help="Output video frame rate (default: 30.0)")
    parser.add_argument(
        "--resolution",
        type=parse_resolution,
        default=(640, 360),
        help="Output video resolution as WIDTHxHEIGHT (default: 640x360)",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="mp4v",
        choices=["mp4v", "XVID", "MJPG", "H264"],
        help="Video codec (default: mp4v)",
    )

    # Visualization parameters
    parser.add_argument(
        "--decay", type=float, default=100.0, help="Event decay time in milliseconds (default: 100.0)"
    )
    parser.add_argument(
        "--positive-color",
        type=parse_color,
        default=(0, 0, 255),  # Red in BGR
        help="Color for positive events as R,G,B (default: 255,0,0)",
    )
    parser.add_argument(
        "--negative-color",
        type=parse_color,
        default=(255, 0, 0),  # Blue in BGR
        help="Color for negative events as R,G,B (default: 0,0,255)",
    )
    parser.add_argument(
        "--background-color",
        type=parse_color,
        default=(0, 0, 0),  # Black in BGR
        help="Background color as R,G,B (default: 0,0,0)",
    )

    # Time selection
    parser.add_argument("--start-time", type=float, help="Start time in seconds (default: from beginning)")
    parser.add_argument("--duration", type=float, help="Duration in seconds (default: entire file)")

    # Display options
    parser.add_argument("--no-stats", action="store_true", help="Disable statistics overlay")
    parser.add_argument(
        "--stats-color",
        type=parse_color,
        default=(255, 255, 255),  # White in BGR
        help="Statistics text color as R,G,B (default: 255,255,255)",
    )

    # Processing options
    parser.add_argument(
        "--pattern",
        type=str,
        default="*/event_representations_v2",
        help="Pattern to match data directories in batch mode (default: */event_representations_v2)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")

    # Logging
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")

    args = parser.parse_args()

    # Set up logging
    if args.quiet:
        log_level = logging.ERROR
    elif args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )
    logger = logging.getLogger(__name__)

    try:
        # Validate arguments
        validate_paths(args)

        # Create visualization configuration
        config = viz.VisualizationConfig(
            width=args.resolution[0],
            height=args.resolution[1],
            fps=args.fps,
            positive_color=args.positive_color,
            negative_color=args.negative_color,
            background_color=args.background_color,
            decay_ms=args.decay,
            show_stats=not args.no_stats,
            stats_color=args.stats_color,
            codec=args.codec,
        )

        # Create visualizer
        visualizer = viz.eTramVisualizer(config)

        logger.info("eTram Event Visualization")
        logger.info("=" * 50)
        logger.info(f"Resolution: {config.width}x{config.height}")
        logger.info(f"FPS: {config.fps}")
        logger.info(f"Decay: {config.decay_ms}ms")
        logger.info(f"Codec: {config.codec}")

        start_time = time.time()

        if args.batch:
            # Batch processing mode
            logger.info(f"Processing batch: {args.batch}")
            logger.info(f"Output directory: {args.output_dir}")
            logger.info(f"Pattern: {args.pattern}")

            # Check if output directory exists and create if needed
            output_dir = Path(args.output_dir)
            if not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created output directory: {output_dir}")

            # Process batch
            successful_outputs = visualizer.process_directory(
                args.batch, args.output_dir, pattern=args.pattern
            )

            # Report results
            total_time = time.time() - start_time
            logger.info("=" * 50)
            logger.info(f"Batch processing complete in {total_time:.1f}s")
            logger.info(f"Successfully processed: {len(successful_outputs)} files")

            if successful_outputs:
                logger.info("Output files:")
                for output_path in successful_outputs:
                    file_size_mb = output_path.stat().st_size / (1024 * 1024)
                    logger.info(f"  {output_path.name}: {file_size_mb:.1f} MB")

        else:
            # Single file processing mode
            logger.info(f"Processing: {args.input}")
            logger.info(f"Output: {args.output}")

            if args.start_time is not None:
                logger.info(f"Start time: {args.start_time}s")
            if args.duration is not None:
                logger.info(f"Duration: {args.duration}s")

            # Check if output file exists
            output_path = Path(args.output)
            if output_path.exists() and not args.overwrite:
                logger.error(f"Output file already exists: {args.output}")
                logger.error("Use --overwrite to replace existing files")
                sys.exit(1)

            # Process single file
            success = visualizer.process_file(
                args.input, args.output, start_time_s=args.start_time, duration_s=args.duration
            )

            # Report results
            total_time = time.time() - start_time
            if success:
                file_size_mb = output_path.stat().st_size / (1024 * 1024)
                logger.info("=" * 50)
                logger.info(f"Processing complete in {total_time:.1f}s")
                logger.info(f"Output: {args.output} ({file_size_mb:.1f} MB)")
            else:
                logger.error("Processing failed")
                sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
