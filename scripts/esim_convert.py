#!/usr/bin/env python3
"""Convert a video file to events with the ESIM kernel (evlib.simulation).

Run: .venv/bin/python scripts/esim_convert.py clip.mp4 -o events.h5
Output format follows the extension: .h5/.hdf5, .parquet, or text (.txt/.csv).
"""

import argparse
import sys
import time
from pathlib import Path

import polars as pl

import evlib
from evlib.simulation import (
    ESIMConfig,
    VideoConfig,
    VideoToEvents,
    estimate_event_count,
)

_FLOAT_OPTIONS = [
    ("--cp", 0.2, "Positive contrast threshold"),
    ("--cn", 0.2, "Negative contrast threshold"),
    ("--refractory-period", 0.0, "Refractory period in ms"),
    ("--log-eps", 1e-3, "Log intensity epsilon"),
    ("--threshold-sigma", 0.0, "Per-pixel threshold std"),
    ("--fps", None, "Override the video frame rate"),
    ("--start-time", None, "Start time in seconds"),
    ("--end-time", None, "End time in seconds"),
]
_INT_OPTIONS = [
    ("--seed", 0, "Threshold map seed"),
    ("--width", None, "Resize width"),
    ("--height", None, "Resize height"),
    ("--frame-skip", 0, "Frames to skip between kept frames"),
    ("--chunk-frames", 64, "Frames per streaming chunk"),
    ("--sample-frames", 100, "Frames sampled for estimation"),
]
_FLAGS = [
    ("--streaming", "Chunked processing"),
    ("--estimate-only", "Estimate the event count only"),
    ("--video-info", "Print video properties"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a video file to event data with the ESIM algorithm.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video_file", help="Path to the input video file")
    parser.add_argument("-o", "--output", default="events_esim.h5", help="Output file")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    for flag, default, help_text in _FLOAT_OPTIONS:
        parser.add_argument(flag, type=float, default=default, help=help_text)
    for flag, default, help_text in _INT_OPTIONS:
        parser.add_argument(flag, type=int, default=default, help=help_text)
    for flag, help_text in _FLAGS:
        parser.add_argument(flag, action="store_true", help=help_text)
    return parser


def save_events(df, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".parquet":
        df.write_parquet(output)
        return
    x = df["x"].to_numpy().astype("int64")
    y = df["y"].to_numpy().astype("int64")
    t = df["t"].dt.total_microseconds().to_numpy().astype("float64") / 1e6
    p = df["polarity"].to_numpy().astype("int64")
    if suffix in (".h5", ".hdf5"):
        evlib.save_events_to_hdf5(x, y, t, p, str(output))
    else:
        evlib.save_events_to_text(x, y, t, p, str(output))


def main() -> int:
    args = build_parser().parse_args()
    video_path = Path(args.video_file)
    if not video_path.exists():
        print(f"Error: video file not found: {video_path}")
        return 1
    esim_config = ESIMConfig(
        positive_threshold=args.cp,
        negative_threshold=args.cn,
        refractory_period_ms=args.refractory_period,
        log_eps=args.log_eps,
        threshold_sigma=args.threshold_sigma,
        seed=args.seed,
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
    processor = VideoToEvents(esim_config, video_config)

    if args.video_info:
        info = processor.get_video_info(video_path)
        proc = info["processing"]
        print(f"Source: {info['width']}x{info['height']} @ {info['fps']:.2f} fps")
        print(f"Frames: {info['frame_count']:,} ({info['duration_seconds']:.2f} s)")
        print(f"Target: {proc['target_width']}x{proc['target_height']}")
        print(f"ESIM: {esim_config}")

    if args.estimate_only:
        started = time.time()
        estimate = estimate_event_count(
            video_path, esim_config, video_config, args.sample_frames
        )
        print(f"Estimated total events: {estimate['estimated_total_events']:,}")
        print(f"Events per frame: {estimate['events_per_frame']:.1f}")
        print(f"Estimation time: {time.time() - started:.2f} s")
        return 0

    print(f"Converting {video_path} -> {args.output}")
    started = time.time()
    if args.streaming:
        df = pl.concat(
            processor.process_frames_streaming(video_path, args.chunk_frames)
        )
    else:
        df = processor.process_video(video_path)
    elapsed = time.time() - started
    if df.height == 0:
        print("No events were generated. Try lower thresholds (--cp, --cn).")
        return 0

    output = Path(args.output)
    save_events(df, output)
    t_us = df["t"].dt.total_microseconds()
    positive = int((df["polarity"] == 1).sum())
    print(f"Events: {df.height:,} in {elapsed:.2f} s")
    print(f"Time range: {t_us.min() / 1e6:.6f} - {t_us.max() / 1e6:.6f} s")
    print(f"Positive: {positive:,}, negative: {df.height - positive:,}")
    print(f"Saved to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
