"""
Event Visualization Module

Provides functionality to visualize event camera data with video output capabilities.
Specifically designed for eTram processed data but extensible for other formats.

Features:
- Load and render eTram stacked histogram data
- Polarity-based coloring (red/blue) similar to WASM visualization
- Temporal decay effects
- Statistics overlay
- Video output using OpenCV
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, List
import time
from dataclasses import dataclass

import numpy as np
import h5py
import cv2

# Set up HDF5 plugins for ECF codec support
try:
    import hdf5plugin

    os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGIN_PATH
    if hasattr(hdf5plugin, "register"):
        hdf5plugin.register()
except ImportError:
    logging.warning(
        "hdf5plugin not available - may have issues reading compressed HDF5 files"
    )


@dataclass
class VisualizationConfig:
    """Configuration for event visualization."""

    # Display parameters
    width: int = 640
    height: int = 360
    fps: float = 30.0

    # Color configuration
    positive_color: Tuple[int, int, int] = (0, 0, 255)  # Red (BGR format for OpenCV)
    negative_color: Tuple[int, int, int] = (
        255,
        128,
        0,
    )  # Bright blue (BGR format for OpenCV)
    background_color: Tuple[int, int, int] = (
        200,
        180,
        150,
    )  # Light blue background (BGR format for OpenCV)

    # Colormap visualization mode
    use_colormap: bool = False  # Enable thermal/jet-like colormap visualization
    colormap_type: str = "jet"  # OpenCV colormap: jet, hot, plasma, viridis, etc.

    # Temporal effects
    decay_ms: float = 100.0  # Event decay time in milliseconds
    frame_duration_ms: float = 33.33  # Frame duration (1000/fps)

    # Statistics overlay
    show_stats: bool = True
    stats_color: Tuple[int, int, int] = (255, 255, 255)  # White
    stats_font_scale: float = 0.8
    stats_thickness: int = 1

    # Video output
    codec: str = "mp4v"
    quality: int = 90

    def __post_init__(self) -> None:
        """Calculate derived parameters."""
        self.frame_duration_ms = 1000.0 / self.fps


class eTramDataLoader:
    """Loader for eTram processed event data."""

    def __init__(self, data_path: Union[str, Path]):
        """
        Initialize data loader for eTram processed data.

        Args:
            data_path: Path to eTram data directory or HDF5 file
        """
        self.data_path = Path(data_path)
        self.logger = logging.getLogger(__name__)

        # Find the HDF5 file
        self.h5_file_path = self._find_h5_file()
        if not self.h5_file_path:
            raise FileNotFoundError(f"No HDF5 file found in {data_path}")

        # Load metadata
        self._load_metadata()

    def _find_h5_file(self) -> Optional[Path]:
        """Find the event representations HDF5 file."""
        if self.data_path.is_file() and self.data_path.suffix == ".h5":
            return self.data_path

        # Look for stacked histogram data
        pattern_paths = [
            "event_representations_v2/stacked_histogram_dt=50_nbins=10/event_representations_ds2_nearest.h5",
            "event_representations_v2/*.h5",
        ]

        for pattern in pattern_paths:
            if "/" in pattern:
                full_path = self.data_path / pattern
                if full_path.exists():
                    return full_path
            else:
                # Use glob for patterns
                matches = list(self.data_path.glob(pattern))
                if matches:
                    return matches[0]

        return None

    def _load_metadata(self) -> None:
        """Load metadata from HDF5 file and timestamps."""
        try:
            # Ensure HDF5 plugins are set up before metadata loading
            self._setup_hdf5_plugins_static()

            with h5py.File(self.h5_file_path, "r") as f:
                # Check for both data formats: "data" key (eTram/representations) or "events" group (raw events)
                if "data" in f:
                    # Standard format: 4D representation data
                    data_shape = f["data"].shape
                    self.num_frames = data_shape[0]
                    self.num_bins = data_shape[1]
                    self.height = data_shape[2]
                    self.width = data_shape[3]
                    self.dtype = f["data"].dtype
                    self.data_format = "representation"
                elif "events" in f:
                    # Raw events format: need to convert to representation format on-demand
                    events_group = f["events"]
                    if not all(key in events_group for key in ["xs", "ys", "ts", "ps"]):
                        raise ValueError(
                            "Events group missing required datasets (xs, ys, ts, ps)"
                        )

                    # Load event data to determine dimensions
                    xs = events_group["xs"][:]
                    ys = events_group["ys"][:]
                    ts = events_group["ts"][:]
                    ps = events_group["ps"][:]

                    # Determine spatial dimensions from event coordinates
                    self.height = int(ys.max()) + 1
                    self.width = int(xs.max()) + 1

                    # Create temporal bins (default 10 bins)
                    self.num_bins = 10
                    t_min, t_max = ts.min(), ts.max()
                    duration = t_max - t_min
                    frame_duration = 0.05  # 50ms frames
                    self.num_frames = max(1, int(duration / frame_duration))

                    self.dtype = np.float32
                    self.data_format = "raw_events"

                    # Cache event data for on-demand conversion
                    self._cached_events = {"xs": xs, "ys": ys, "ts": ts, "ps": ps}
                else:
                    raise ValueError(
                        "HDF5 file must contain either 'data' dataset or 'events' group"
                    )

            # Load timestamps
            timestamp_file = (
                self.data_path
                / "event_representations_v2/stacked_histogram_dt=50_nbins=10/timestamps_us.npy"
            )
            if not timestamp_file.exists():
                # Try alternative path
                timestamp_file = self.data_path.parent / "timestamps_us.npy"

            if timestamp_file.exists():
                self.timestamps_us = np.load(timestamp_file)
                self.start_time_s = self.timestamps_us[0] / 1_000_000
                self.end_time_s = self.timestamps_us[-1] / 1_000_000
                self.duration_s = self.end_time_s - self.start_time_s
            else:
                self.logger.warning("No timestamps file found, using frame indices")
                self.timestamps_us = np.arange(self.num_frames) * 50000  # 50ms default
                self.start_time_s = 0.0
                self.end_time_s = self.num_frames * 0.05
                self.duration_s = self.end_time_s

            self.logger.info(
                f"Loaded eTram data: {self.num_frames} frames, {self.width}x{self.height}, "
                f"{self.duration_s:.2f}s duration"
            )

        except Exception as e:
            raise RuntimeError(f"Failed to load metadata from {self.h5_file_path}: {e}")

    def get_frame_data(self, frame_idx: int) -> np.ndarray:
        """
        Get event data for a specific frame.

        Args:
            frame_idx: Frame index to load

        Returns:
            Event data array with shape (num_bins, height, width)
        """
        if frame_idx < 0 or frame_idx >= self.num_frames:
            raise ValueError(
                f"Frame index {frame_idx} out of range [0, {self.num_frames})"
            )

        try:
            if self.data_format == "representation":
                # Standard format: read directly from "data" dataset
                self._setup_hdf5_plugins()
                with h5py.File(self.h5_file_path, "r") as f:
                    frame_data = f["data"][frame_idx]
                    return frame_data
            elif self.data_format == "raw_events":
                # Raw events format: convert to representation on-demand
                return self._convert_events_to_frame(frame_idx)
            else:
                raise ValueError(f"Unknown data format: {self.data_format}")
        except Exception as e:
            raise RuntimeError(f"Failed to load frame {frame_idx}: {e}")

    def _setup_hdf5_plugins(self) -> None:
        """Ensure HDF5 plugins are properly configured."""
        try:
            import hdf5plugin

            os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGIN_PATH
            if hasattr(hdf5plugin, "register"):
                hdf5plugin.register()
        except ImportError:
            pass  # Already warned during module import

    def _convert_events_to_frame(self, frame_idx: int) -> np.ndarray:
        """Convert raw events to a single representation frame."""
        if not hasattr(self, "_cached_events"):
            raise RuntimeError("No cached events available for conversion")

        # Extract event data
        xs, ys, ts, ps = (
            self._cached_events["xs"],
            self._cached_events["ys"],
            self._cached_events["ts"],
            self._cached_events["ps"],
        )

        # Calculate time window for this frame
        t_min, t_max = ts.min(), ts.max()
        duration = t_max - t_min
        frame_duration = duration / self.num_frames
        frame_start = t_min + frame_idx * frame_duration
        frame_end = t_min + (frame_idx + 1) * frame_duration

        # Filter events for this time window
        mask = (ts >= frame_start) & (ts < frame_end)
        frame_xs = xs[mask]
        frame_ys = ys[mask]
        frame_ts = ts[mask]
        frame_ps = ps[mask]

        # Create representation frame with polarity separation
        # Even bins = positive events, odd bins = negative events (to match renderer expectations)
        frame_data = np.zeros(
            (self.num_bins, self.height, self.width), dtype=np.float32
        )

        if len(frame_xs) > 0:
            # Separate positive and negative events
            pos_mask = frame_ps == 1
            neg_mask = frame_ps == -1

            pos_xs, pos_ys, pos_ts = (
                frame_xs[pos_mask],
                frame_ys[pos_mask],
                frame_ts[pos_mask],
            )
            neg_xs, neg_ys, neg_ts = (
                frame_xs[neg_mask],
                frame_ys[neg_mask],
                frame_ts[neg_mask],
            )

            # Create temporal bins within the frame
            temporal_bins = self.num_bins // 2  # Half for positive, half for negative
            bin_duration = frame_duration / temporal_bins

            # Process positive events (even bins: 0, 2, 4, ...)
            for bin_idx in range(temporal_bins):
                bin_start = frame_start + bin_idx * bin_duration
                bin_end = frame_start + (bin_idx + 1) * bin_duration

                bin_mask = (pos_ts >= bin_start) & (pos_ts < bin_end)
                bin_xs = pos_xs[bin_mask]
                bin_ys = pos_ys[bin_mask]

                # Accumulate positive events in even bins
                for x, y in zip(bin_xs, bin_ys):
                    if 0 <= x < self.width and 0 <= y < self.height:
                        frame_data[bin_idx * 2, y, x] += 1.0  # Even bins for positive

            # Process negative events (odd bins: 1, 3, 5, ...)
            for bin_idx in range(temporal_bins):
                bin_start = frame_start + bin_idx * bin_duration
                bin_end = frame_start + (bin_idx + 1) * bin_duration

                bin_mask = (neg_ts >= bin_start) & (neg_ts < bin_end)
                bin_xs = neg_xs[bin_mask]
                bin_ys = neg_ys[bin_mask]

                # Accumulate negative events in odd bins
                for x, y in zip(bin_xs, bin_ys):
                    if 0 <= x < self.width and 0 <= y < self.height:
                        frame_data[bin_idx * 2 + 1, y, x] += (
                            1.0  # Odd bins for negative
                        )

        return frame_data

    def _convert_events_to_frames(self, start_idx: int, end_idx: int) -> np.ndarray:
        """Convert raw events to multiple representation frames."""
        frames = []
        for frame_idx in range(start_idx, end_idx):
            frame_data = self._convert_events_to_frame(frame_idx)
            frames.append(frame_data)
        return np.stack(frames, axis=0)

    @staticmethod
    def _setup_hdf5_plugins_static() -> None:
        """Static method for setting up HDF5 plugins."""
        try:
            import hdf5plugin

            os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGIN_PATH
            if hasattr(hdf5plugin, "register"):
                hdf5plugin.register()
        except ImportError:
            pass

    def get_frame_range(self, start_idx: int, end_idx: int) -> np.ndarray:
        """
        Get event data for a range of frames.

        Args:
            start_idx: Start frame index (inclusive)
            end_idx: End frame index (exclusive)

        Returns:
            Event data array with shape (num_frames, num_bins, height, width)
        """
        if start_idx < 0 or end_idx > self.num_frames or start_idx >= end_idx:
            raise ValueError(
                f"Invalid frame range [{start_idx}, {end_idx}) for {self.num_frames} frames"
            )

        try:
            # Ensure HDF5 plugins are set up before each file access
            self._setup_hdf5_plugins()

            with h5py.File(self.h5_file_path, "r") as f:
                frame_data = f["data"][start_idx:end_idx]
                return frame_data
        except Exception as e:
            raise RuntimeError(
                f"Failed to load frame range [{start_idx}, {end_idx}): {e}"
            )


class EventFrameRenderer:
    """Renders event data to RGB frames for video output."""

    def __init__(self, config: VisualizationConfig):
        """
        Initialize frame renderer.

        Args:
            config: Visualization configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize data dimensions and simple decay buffer
        self.data_height = None
        self.data_width = None
        self.frame_count = 0
        self.previous_frame = None

    def render_frame(
        self,
        event_data: np.ndarray,
        timestamp_s: float = 0.0,
        show_stats: Optional[Dict] = None,
    ) -> np.ndarray:
        """
        Render a single frame from event data.

        Args:
            event_data: Event data with shape (num_bins, height, width)
            timestamp_s: Current timestamp in seconds
            show_stats: Optional statistics to overlay

        Returns:
            RGB frame as uint8 array with shape (height, width, 3)
        """
        # Initialize dimensions on first frame
        if self.data_height is None:
            self.data_height, self.data_width = event_data.shape[1], event_data.shape[2]

        # Process event data - sum positive and negative events across bins
        positive_events = np.sum(event_data[::2], axis=0).astype(
            np.float32
        )  # Even bins = positive
        negative_events = np.sum(event_data[1::2], axis=0).astype(
            np.float32
        )  # Odd bins = negative

        # Normalize event intensities (0-255 range)
        if positive_events.max() > 0:
            positive_norm = (positive_events / positive_events.max()) * 255
        else:
            positive_norm = positive_events

        if negative_events.max() > 0:
            negative_norm = (negative_events / negative_events.max()) * 255
        else:
            negative_norm = negative_events

        # Choose rendering mode based on configuration
        if self.config.use_colormap:
            frame = self._render_colormap_frame(positive_events, negative_events)
        else:
            # Simplified polarity-based rendering
            frame = self._render_polarity_frame(positive_norm, negative_norm)

        # Apply simple temporal decay if we have a previous frame
        if self.previous_frame is not None and self.config.decay_ms > 0:
            decay_factor = np.exp(-self.config.frame_duration_ms / self.config.decay_ms)
            # Blend current frame with decayed previous frame
            frame = frame + (self.previous_frame * decay_factor)

        # Store current frame for next iteration
        self.previous_frame = frame.copy()

        # Convert to uint8
        frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8)

        # Resize to target resolution if different from data resolution
        if (
            frame_uint8.shape[0] != self.config.height
            or frame_uint8.shape[1] != self.config.width
        ):
            frame_uint8 = cv2.resize(
                frame_uint8, (self.config.width, self.config.height)
            )

        # Add statistics overlay if requested
        if show_stats and self.config.show_stats:
            frame_uint8 = self._add_stats_overlay(frame_uint8, show_stats, timestamp_s)

        self.frame_count += 1
        return frame_uint8

    def _add_stats_overlay(
        self, frame: np.ndarray, stats: Dict, timestamp_s: float
    ) -> np.ndarray:
        """Add compact statistics overlay to frame."""
        overlay_frame = frame.copy()

        # Prepare compact statistics text
        y_offset = 18
        stats_lines = [
            f"FPS: {stats.get('fps', 0.0):.1f}",
            f"Events/s: {stats.get('events_per_sec', 0):,}",
            f"Total Events: {stats.get('total_events', 0):,}",
            f"Time: {timestamp_s:.3f}s",
            f"Frame: {self.frame_count}",
        ]

        # Calculate smaller background rectangle
        line_height = 14  # Reduced from 20
        bg_height = len(stats_lines) * line_height + 8  # Reduced padding
        bg_width = 160  # Reduced from 200

        # Add smaller background rectangle
        cv2.rectangle(overlay_frame, (8, 3), (bg_width, bg_height), (0, 0, 0), -1)
        cv2.rectangle(overlay_frame, (8, 3), (bg_width, bg_height), (64, 64, 64), 1)

        # Add text with smaller font (DUPLEX for better readability at small sizes)
        for i, line in enumerate(stats_lines):
            cv2.putText(
                overlay_frame,
                line,
                (12, y_offset + i * line_height),
                cv2.FONT_HERSHEY_PLAIN,
                self.config.stats_font_scale,
                self.config.stats_color,
                self.config.stats_thickness,
            )

        return overlay_frame

    def _render_polarity_frame(
        self, positive_norm: np.ndarray, negative_norm: np.ndarray
    ) -> np.ndarray:
        """Simplified polarity-based rendering with red/blue colors."""
        # Start with clean background frame
        frame = np.full(
            (self.data_height, self.data_width, 3),
            self.config.background_color,
            dtype=np.float32,
        )

        # Add positive events (RED) - pure red on background
        pos_mask = positive_norm > 0
        if np.any(pos_mask):
            frame[pos_mask, 0] = 0  # Blue = 0
            frame[pos_mask, 1] = 0  # Green = 0
            frame[pos_mask, 2] = 255  # Red = 255 (BGR format)

        # Add negative events (BLUE) - pure blue on background
        neg_mask = negative_norm > 0
        if np.any(neg_mask):
            frame[neg_mask, 0] = 255  # Blue = 255 (BGR format)
            frame[neg_mask, 1] = 0  # Green = 0
            frame[neg_mask, 2] = 0  # Red = 0

        return frame

    def _render_colormap_frame(
        self, positive_events: np.ndarray, negative_events: np.ndarray
    ) -> np.ndarray:
        """Enhanced colormap-based rendering that preserves polarity information."""
        # Start with configured background color
        frame = np.full(
            (self.data_height, self.data_width, 3),
            self.config.background_color,
            dtype=np.float32,
        )

        # Get colormap configuration
        colormap_dict = {
            "jet": cv2.COLORMAP_JET,
            "hot": cv2.COLORMAP_HOT,
            "plasma": cv2.COLORMAP_PLASMA,
            "viridis": cv2.COLORMAP_VIRIDIS,
            "inferno": cv2.COLORMAP_INFERNO,
            "magma": cv2.COLORMAP_MAGMA,
            "rainbow": cv2.COLORMAP_RAINBOW,
            "ocean": cv2.COLORMAP_OCEAN,
            "summer": cv2.COLORMAP_SUMMER,
            "spring": cv2.COLORMAP_SPRING,
            "cool": cv2.COLORMAP_COOL,
            "hsv": cv2.COLORMAP_HSV,
            "pink": cv2.COLORMAP_PINK,
            "bone": cv2.COLORMAP_BONE,
        }
        colormap = colormap_dict.get(
            self.config.colormap_type.lower(), cv2.COLORMAP_JET
        )

        # Process positive events (warm colors - red/yellow side of colormap)
        if positive_events.max() > 0:
            pos_norm = (positive_events / positive_events.max() * 255).astype(np.uint8)
            pos_colored = cv2.applyColorMap(pos_norm, colormap).astype(np.float32)

            # Keep only warm colors (shift colormap to red/yellow range)
            pos_mask = pos_norm > 0
            frame[pos_mask] = pos_colored[pos_mask]

        # Process negative events (cool colors - blue/cyan side of colormap)
        if negative_events.max() > 0:
            neg_norm = (negative_events / negative_events.max() * 255).astype(np.uint8)
            neg_colored = cv2.applyColorMap(neg_norm, colormap).astype(np.float32)

            # Shift negative events to cool side of colormap
            neg_mask = neg_norm > 0
            if self.config.colormap_type.lower() in ["jet", "rainbow", "hsv"]:
                # For jet/rainbow: use blue side for negative events
                neg_colored[neg_mask, 2] = neg_colored[neg_mask, 2] * 0.3  # Reduce red
                neg_colored[neg_mask, 1] = (
                    neg_colored[neg_mask, 1] * 0.6
                )  # Reduce green
                neg_colored[neg_mask, 0] = np.minimum(
                    255, neg_colored[neg_mask, 0] * 1.5
                )  # Enhance blue
            elif self.config.colormap_type.lower() in ["hot", "inferno", "magma"]:
                # For hot/inferno: use different intensity mapping
                neg_colored[neg_mask, 0] = (
                    neg_colored[neg_mask, 0] * 1.2
                )  # More blue/purple
                neg_colored[neg_mask, 1] = neg_colored[neg_mask, 1] * 0.8  # Less green
                neg_colored[neg_mask, 2] = neg_colored[neg_mask, 2] * 0.5  # Less red

            # Blend negative events with existing frame
            alpha = 0.8
            frame[neg_mask] = (
                frame[neg_mask] * (1 - alpha) + neg_colored[neg_mask] * alpha
            )

        # Note: Temporal decay removed for simplicity - each frame is independent

        return frame

    def reset(self) -> None:
        """Reset the renderer state."""
        self.frame_count = 0
        self.previous_frame = None


class eTramVisualizer:
    """Main visualizer class for eTram event data."""

    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize eTram visualizer.

        Args:
            config: Visualization configuration, uses defaults if None
        """
        self.config = config or VisualizationConfig()
        self.logger = logging.getLogger(__name__)
        self.renderer = EventFrameRenderer(self.config)

    def process_file(
        self,
        data_path: Union[str, Path],
        output_path: Union[str, Path],
        start_time_s: Optional[float] = None,
        duration_s: Optional[float] = None,
    ) -> bool:
        """
        Process a single eTram data file to video.

        Args:
            data_path: Path to eTram data directory or HDF5 file
            output_path: Output video file path
            start_time_s: Start time in seconds (None = from beginning)
            duration_s: Duration in seconds (None = entire file)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load data
            self.logger.info(f"Loading eTram data from {data_path}")
            loader = eTramDataLoader(data_path)

            # Calculate frame range
            if start_time_s is not None:
                start_frame = int(
                    (start_time_s - loader.start_time_s) * self.config.fps
                )
                start_frame = max(0, min(start_frame, loader.num_frames - 1))
            else:
                start_frame = 0
                start_time_s = loader.start_time_s

            if duration_s is not None:
                num_frames = int(duration_s * self.config.fps)
                end_frame = min(start_frame + num_frames, loader.num_frames)
            else:
                end_frame = loader.num_frames

            actual_duration = (end_frame - start_frame) / self.config.fps

            self.logger.info(
                f"Rendering {end_frame - start_frame} frames "
                f"({actual_duration:.2f}s) at {self.config.fps} FPS"
            )

            # Initialize video writer
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            fourcc = cv2.VideoWriter_fourcc(*self.config.codec)
            video_writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                self.config.fps,
                (self.config.width, self.config.height),
                isColor=True,
            )

            if not video_writer.isOpened():
                self.logger.error("Failed to open video writer")
                return False

            # Reset renderer
            self.renderer.reset()

            # Process frames
            start_time = time.time()

            for frame_idx in range(start_frame, end_frame):
                try:
                    # Load frame data
                    event_data = loader.get_frame_data(frame_idx)

                    # Calculate timestamp
                    if frame_idx < len(loader.timestamps_us):
                        timestamp_s = loader.timestamps_us[frame_idx] / 1_000_000
                    else:
                        timestamp_s = (
                            start_time_s + (frame_idx - start_frame) / self.config.fps
                        )

                    # Calculate statistics
                    total_events = int(np.sum(event_data))
                    elapsed_time = time.time() - start_time
                    processing_fps = (
                        (frame_idx - start_frame + 1) / elapsed_time
                        if elapsed_time > 0
                        else 0
                    )

                    stats = {
                        "fps": processing_fps,
                        "events_per_sec": int(total_events * self.config.fps),
                        "total_events": total_events,
                    }

                    # Render frame (resizing is handled internally)
                    rgb_frame = self.renderer.render_frame(
                        event_data, timestamp_s, stats
                    )

                    # Write frame
                    video_writer.write(rgb_frame)

                    # Progress update
                    if (frame_idx - start_frame + 1) % max(
                        1, (end_frame - start_frame) // 20
                    ) == 0:
                        progress = (frame_idx - start_frame + 1) / (
                            end_frame - start_frame
                        )
                        eta = (
                            elapsed_time / progress - elapsed_time
                            if progress > 0
                            else 0
                        )
                        self.logger.info(
                            f"Progress: {frame_idx - start_frame + 1}/{end_frame - start_frame} "
                            f"({progress * 100:.1f}%) - ETA: {eta:.1f}s"
                        )

                except Exception as e:
                    self.logger.error(f"Error processing frame {frame_idx}: {e}")
                    continue

            # Cleanup
            video_writer.release()

            # Verify output
            if output_path.exists():
                file_size_mb = output_path.stat().st_size / (1024 * 1024)
                total_time = time.time() - start_time
                self.logger.info(
                    f"Video saved successfully: {file_size_mb:.1f} MB in {total_time:.1f}s"
                )
                return True
            else:
                self.logger.error("Failed to save video file")
                return False

        except Exception as e:
            self.logger.error(f"Failed to process file {data_path}: {e}")
            return False

    def process_directory(
        self,
        data_dir: Union[str, Path],
        output_dir: Union[str, Path],
        pattern: str = "*/event_representations_v2",
    ) -> List[Path]:
        """
        Process all eTram data files in a directory.

        Args:
            data_dir: Directory containing eTram data
            output_dir: Output directory for videos
            pattern: Pattern to match data directories

        Returns:
            List of successfully created video files
        """
        data_dir = Path(data_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all data directories
        data_paths = list(data_dir.glob(pattern))
        if not data_paths:
            self.logger.warning(
                f"No data directories found matching pattern: {pattern}"
            )
            return []

        self.logger.info(f"Found {len(data_paths)} data directories to process")

        successful_outputs = []

        for i, data_path in enumerate(data_paths):
            # Generate output filename
            output_name = data_path.parent.name + ".mp4"
            output_path = output_dir / output_name

            self.logger.info(
                f"Processing {i + 1}/{len(data_paths)}: {data_path.parent.name}"
            )

            if self.process_file(data_path.parent, output_path):
                successful_outputs.append(output_path)
            else:
                self.logger.error(f"Failed to process {data_path}")

        self.logger.info(
            f"Completed batch processing: {len(successful_outputs)}/{len(data_paths)} successful"
        )
        return successful_outputs
