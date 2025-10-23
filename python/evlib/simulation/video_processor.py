"""
Video processing utilities for event camera simulation.

This module provides high-level interfaces for converting video files
to event camera data using various simulation algorithms.
"""

import numpy as np
from typing import Tuple, Optional, Union, Generator
from pathlib import Path

try:
    import cv2

    _opencv_available = True
except ImportError:
    _opencv_available = False
    cv2 = None

try:
    import torch

    _torch_available = True
except ImportError:
    _torch_available = False

from .config import ESIMConfig, VideoConfig

if _torch_available:
    from .esim import ESIMSimulator


class VideoToEvents:
    """
    High-level interface for converting video files to event camera data.

    Handles video I/O, frame processing, and integration with simulation algorithms.

    Args:
        esim_config: Configuration for the ESIM algorithm
        video_config: Configuration for video processing
    """

    def __init__(self, esim_config: ESIMConfig, video_config: VideoConfig):
        if not _opencv_available:
            raise ImportError(
                "OpenCV is required for video processing. Install with: pip install opencv-python"
            )

        if not _torch_available:
            raise ImportError(
                "PyTorch is required for ESIM simulation. Install with: pip install torch"
            )

        self.esim_config = esim_config
        self.video_config = video_config
        self.simulator = ESIMSimulator(esim_config)

        # Video processing state
        self._cap: Optional[cv2.VideoCapture] = None
        self._video_fps: Optional[float] = None
        self._total_frames: Optional[int] = None
        self._current_frame: int = 0

    def process_video(
        self, video_path: Union[str, Path]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Process entire video file and return all generated events.

        Args:
            video_path: Path to input video file

        Returns:
            Tuple of (x, y, t, polarity) arrays containing all events

        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video cannot be opened
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Open video
        self._cap = cv2.VideoCapture(str(video_path))
        if not self._cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        try:
            # Get video properties
            self._setup_video_properties()

            # Reset simulator
            self.simulator.reset()

            # Process all frames
            all_events = []

            print(f"Processing {self._total_frames} frames from {video_path.name}...")
            print(f"Using device: {self.simulator.device}")

            for events in self._process_frames_generator():
                if len(events[0]) > 0:  # Check if any events were generated
                    all_events.append(events)

            # Combine all events
            if all_events:
                x_arrays, y_arrays, t_arrays, p_arrays = zip(*all_events)

                x_combined = np.concatenate(x_arrays)
                y_combined = np.concatenate(y_arrays)
                t_combined = np.concatenate(t_arrays)
                p_combined = np.concatenate(p_arrays)

                # Sort by timestamp
                sort_indices = np.argsort(t_combined)

                return (
                    x_combined[sort_indices],
                    y_combined[sort_indices],
                    t_combined[sort_indices],
                    p_combined[sort_indices],
                )
            else:
                print("No events were generated")
                return np.array([]), np.array([]), np.array([]), np.array([])

        finally:
            if self._cap:
                self._cap.release()
                self._cap = None

    def process_frames_streaming(
        self, video_path: Union[str, Path]
    ) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
        """
        Process video frames and yield events in streaming fashion.

        Args:
            video_path: Path to input video file

        Yields:
            Tuple of (x, y, t, polarity) arrays for each frame's events
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self._cap = cv2.VideoCapture(str(video_path))
        if not self._cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        try:
            self._setup_video_properties()
            self.simulator.reset()

            for events in self._process_frames_generator():
                yield events

        finally:
            if self._cap:
                self._cap.release()
                self._cap = None

    def _setup_video_properties(self) -> None:
        """Set up video properties and handle configuration."""
        if not self._cap:
            return

        # Get original video properties
        original_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Use configured FPS or original
        self._video_fps = self.video_config.fps or original_fps

        # Handle start/end time
        if self.video_config.start_time is not None:
            start_frame = int(self.video_config.start_time * original_fps)
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self._current_frame = start_frame

        # Adjust total frames for end time
        if self.video_config.end_time is not None:
            end_frame = int(self.video_config.end_time * original_fps)
            self._total_frames = min(self._total_frames, end_frame)

    def _process_frames_generator(
        self,
    ) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
        """Generator that processes video frames and yields events."""
        if not self._cap:
            return

        frame_count = 0

        while True:
            ret, frame = self._cap.read()
            if not ret:
                break

            # Check if we've reached the end time
            if (
                self.video_config.end_time is not None
                and self._current_frame >= self._total_frames
            ):
                break

            # Handle frame skipping
            if frame_count % (self.video_config.frame_skip + 1) != 0:
                frame_count += 1
                self._current_frame += 1
                continue

            # Preprocess frame
            processed_frame = self._preprocess_frame(frame)

            # Calculate timestamp
            timestamp = self._current_frame / self._video_fps

            # Generate events
            events = self.simulator.process_frame(processed_frame, timestamp)

            yield events

            frame_count += 1
            self._current_frame += 1

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame according to video configuration."""
        # Resize if needed
        if self.video_config.width is not None and self.video_config.height is not None:
            frame = cv2.resize(
                frame, (self.video_config.width, self.video_config.height)
            )

        # Convert to grayscale if needed
        if self.video_config.grayscale and len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return frame

    def get_video_info(self, video_path: Union[str, Path]) -> dict:
        """
        Get information about a video file.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with video information
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        try:
            info = {
                "path": str(video_path),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "duration_seconds": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                / cap.get(cv2.CAP_PROP_FPS),
            }

            # Add processing info based on configuration
            target_width = self.video_config.width or info["width"]
            target_height = self.video_config.height or info["height"]
            target_fps = self.video_config.fps or info["fps"]

            info.update(
                {
                    "processing": {
                        "target_width": target_width,
                        "target_height": target_height,
                        "target_fps": target_fps,
                        "frame_skip": self.video_config.frame_skip,
                        "grayscale": self.video_config.grayscale,
                        "effective_fps": target_fps
                        / (self.video_config.frame_skip + 1),
                    }
                }
            )

            return info

        finally:
            cap.release()

    def reset(self) -> None:
        """Reset the video processor and simulator state."""
        self.simulator.reset()
        self._current_frame = 0
        if self._cap:
            self._cap.release()
            self._cap = None


# Convenience functions
def video_to_events_simple(
    video_path: Union[str, Path],
    width: int = 640,
    height: int = 480,
    positive_threshold: float = 0.4,
    negative_threshold: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple interface for converting video to events with basic parameters.

    Args:
        video_path: Path to input video file
        width: Target width for frames
        height: Target height for frames
        positive_threshold: Positive contrast threshold
        negative_threshold: Negative contrast threshold

    Returns:
        Tuple of (x, y, t, polarity) arrays containing events
    """
    esim_config = ESIMConfig(
        positive_threshold=positive_threshold, negative_threshold=negative_threshold
    )
    video_config = VideoConfig(width=width, height=height)

    processor = VideoToEvents(esim_config, video_config)
    return processor.process_video(video_path)


def estimate_event_count(
    video_path: Union[str, Path],
    esim_config: Optional[ESIMConfig] = None,
    video_config: Optional[VideoConfig] = None,
    sample_frames: int = 100,
) -> dict:
    """
    Estimate the number of events that will be generated from a video.

    Args:
        video_path: Path to input video file
        esim_config: ESIM configuration
        video_config: Video configuration
        sample_frames: Number of frames to sample for estimation

    Returns:
        Dictionary with estimation results
    """
    processor = VideoToEvents(
        esim_config or ESIMConfig(), video_config or VideoConfig()
    )

    video_info = processor.get_video_info(video_path)
    total_frames = video_info["frame_count"]

    if sample_frames >= total_frames:
        # Process entire video
        events = processor.process_video(video_path)
        return {
            "total_events": len(events[0]),
            "estimated": False,
            "sample_frames": total_frames,
            "events_per_frame": len(events[0]) / total_frames
            if total_frames > 0
            else 0,
        }

    # Sample frames evenly throughout the video
    sample_interval = max(1, total_frames // sample_frames)

    # Configure to process only sampled frames
    sample_config = VideoConfig(
        width=video_config.width if video_config else None,
        height=video_config.height if video_config else None,
        frame_skip=sample_interval - 1,
    )
    sample_processor = VideoToEvents(esim_config or ESIMConfig(), sample_config)

    events = sample_processor.process_video(video_path)
    sample_event_count = len(events[0])

    # Estimate total events
    actual_sample_frames = (
        sample_event_count // sample_interval
        if sample_event_count > 0
        else sample_frames
    )
    events_per_frame = (
        sample_event_count / actual_sample_frames if actual_sample_frames > 0 else 0
    )
    estimated_total = events_per_frame * total_frames

    return {
        "estimated_total_events": int(estimated_total),
        "estimated": True,
        "sample_frames": actual_sample_frames,
        "sample_events": sample_event_count,
        "events_per_frame": events_per_frame,
        "total_frames": total_frames,
    }
