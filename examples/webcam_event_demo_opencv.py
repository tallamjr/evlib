#!/usr/bin/env python3
"""
Real-time webcam event stream demonstration with Rust-optimized visualization

This script demonstrates real-time event generation from webcam input using
evlib's GStreamer integration and ESIM event simulation, with high-performance
Rust-based visualization that minimizes Python overhead.

Key optimizations:
- Rust-based event visualization pipeline
- Minimal data copying between Rust and Python
- Real-time buffer management
- Automatic latency control

Requirements:
- evlib compiled with GStreamer support
- Webcam connected to the system
- opencv-python (cv2)
- numpy

Usage:
    python webcam_event_demo_opencv.py [--device_id 0] [--fps 60] [--threshold 0.25]

Controls:
    Press 'q' to quit
    Press 'r' to reset statistics
    Press '+'/'-' to adjust contrast threshold
    Press 'p' to pause/unpause
    Press 'm' to toggle real-time mode
    Press 'v' to toggle event visualization
    Press 'f' to toggle fast visualization mode
"""

import argparse
import sys
import time
from collections import deque

import cv2
import numpy as np

try:
    import evlib  # noqa: F401

    is_realtime_available = evlib.simulation.is_realtime_available
    create_realtime_stream_py = evlib.simulation.create_realtime_stream_py
except ImportError as e:
    print(f"Error importing evlib: {e}")
    print("Please ensure evlib is installed with: pip install -e .")
    sys.exit(1)


class WebcamEventDemoOpenCV:
    """High-performance webcam event streaming demonstration using OpenCV"""

    def __init__(
        self,
        device_id=0,
        target_fps=60.0,  # Increased default FPS
        contrast_threshold=0.25,  # Increased to reduce event count
        resolution=(640, 480),
        max_buffer_size=1000,  # Reduced buffer to prevent latency buildup
        realtime_mode=True,  # Prioritize low latency over completeness
    ):
        """
        Initialize the webcam event demo

        Args:
            device_id: Camera device ID (default: 0)
            target_fps: Target processing frame rate
            contrast_threshold: Event generation threshold
            resolution: Camera resolution tuple (width, height)
            max_buffer_size: Maximum event buffer size
        """
        self.device_id = device_id
        self.target_fps = target_fps
        self.contrast_threshold = contrast_threshold
        self.resolution = resolution
        self.max_buffer_size = max_buffer_size
        self.realtime_mode = realtime_mode

        # Initialize stream
        self.stream = None
        self.is_running = False
        self.is_paused = False
        self.show_events = True

        # Visualization state
        self.event_buffer = deque(maxlen=500)  # Reduced from 2000 for better performance
        self.fps_history = deque(maxlen=30)  # Reduced from 100
        self.event_count_history = deque(maxlen=30)  # Reduced from 100

        # Performance tracking
        self.frame_times = deque(maxlen=30)
        self.last_frame_time = time.time()
        self.buffer_high_water_mark = max_buffer_size * 0.8  # 80% threshold
        self.consecutive_high_buffer = 0

        # Rust-based visualization pipeline
        self.use_rust_viz = True
        self.fast_viz_mode = False
        if self.use_rust_viz:
            viz_config = PyRealtimeVisualizationConfig(
                display_width=resolution[0],
                display_height=resolution[1],
                event_decay_ms=30.0,
                max_events=2000,
                show_fps=True,
                background_color=(255, 255, 255),
                positive_color=(255, 0, 0),
                negative_color=(0, 0, 255),
            )
            self.rust_visualizer = PyEventVisualizationPipeline(viz_config)

        # Create visualization canvas (only needed for non-Rust visualization)
        if not self.use_rust_viz:
            self.canvas_height = resolution[1] + 200  # Extra space for stats
            self.canvas_width = resolution[0]
            self.canvas = np.ones((self.canvas_height, self.canvas_width, 3), dtype=np.uint8) * 255
        else:
            self.canvas_height = resolution[1]
            self.canvas_width = resolution[0]

    def check_realtime_support(self):
        """Check if real-time streaming is available"""
        if not is_realtime_available():
            print("ERROR: Real-time streaming not available!")
            print("evlib was not compiled with GStreamer support.")
            print("Please compile with: maturin develop --features gstreamer")
            return False
        return True

    def initialize_stream(self):
        """Initialize the real-time event stream"""
        try:
            print("Initializing webcam event stream...")
            print(f"  Device ID: {self.device_id}")
            print(f"  Target FPS: {self.target_fps}")
            print(f"  Contrast threshold: {self.contrast_threshold}")
            print(f"  Buffer size: {self.max_buffer_size}")
            print(f"  Real-time mode: {'ON' if self.realtime_mode else 'OFF'}")
            print(f"  Resolution: {self.resolution}")

            # Create stream
            self.stream = create_realtime_stream_py(
                target_fps=self.target_fps,
                contrast_threshold=self.contrast_threshold,
                device_id=self.device_id,
                max_buffer_size=self.max_buffer_size,
                resolution=self.resolution,
            )

            print("Stream created successfully!")
            return True

        except Exception as e:
            print(f"ERROR: Failed to initialize stream: {e}")
            return False

    def start_streaming(self):
        """Start the webcam streaming"""
        try:
            self.stream.start_streaming()
            self.is_running = True
            print("Webcam streaming started!")
            return True
        except Exception as e:
            print(f"ERROR: Failed to start streaming: {e}")
            return False

    def stop_streaming(self):
        """Stop the webcam streaming"""
        if self.stream and self.is_running:
            try:
                self.stream.stop_streaming()
                self.is_running = False
                print("Webcam streaming stopped.")
            except Exception as e:
                print(f"ERROR: Failed to stop streaming: {e}")

    def process_frame(self):
        """Process a single frame and return events"""
        if not self.is_running or self.is_paused:
            return [], 0

        try:
            # In realtime mode, check buffer size first
            if self.realtime_mode:
                stats = self.stream.get_stats()

                # If buffer is getting too full, clear it to catch up
                if stats.buffer_size > self.buffer_high_water_mark:
                    self.consecutive_high_buffer += 1
                    if self.consecutive_high_buffer > 3:
                        # Clear buffer by consuming all events
                        self.stream.get_events(max_count=None)
                        self.consecutive_high_buffer = 0
                        print(f"Buffer cleared to reduce latency (was {stats.buffer_size} events)")
                        return [], 0
                else:
                    self.consecutive_high_buffer = 0

            # Process next frame
            frame_available = self.stream.process_next_frame()

            if frame_available:
                # In realtime mode, get all available events to prevent buildup
                if self.realtime_mode:
                    xs, ys, ts, ps = self.stream.get_events(max_count=None)
                    # But only visualize a subset for performance
                    if len(xs) > 500:
                        # Sample events uniformly
                        indices = np.linspace(0, len(xs) - 1, 500, dtype=int)
                        xs = [xs[i] for i in indices]
                        ys = [ys[i] for i in indices]
                        ts = [ts[i] for i in indices]
                        ps = [ps[i] for i in indices]
                else:
                    # Normal mode - get limited events
                    xs, ys, ts, ps = self.stream.get_events(max_count=500)

                # Convert to Python lists if numpy arrays
                if hasattr(xs, "__len__") and len(xs) > 0:
                    events = list(zip(xs, ys, ts, ps))
                    return events, len(events)

            return [], 0

        except Exception as e:
            print(f"ERROR: Frame processing failed: {e}")
            return [], 0

    def draw_events(self, event_frame, events):
        """Draw events on the frame efficiently"""
        if not events or not self.show_events:
            return

        # Convert events to numpy arrays for efficient processing
        if len(events) > 0:
            # Split positive and negative events
            events_array = np.array(events)
            xs = events_array[:, 0].astype(int)
            ys = events_array[:, 1].astype(int)
            ps = events_array[:, 3]

            # Filter valid coordinates
            valid_mask = (xs >= 0) & (xs < self.resolution[0]) & (ys >= 0) & (ys < self.resolution[1])
            xs = xs[valid_mask]
            ys = ys[valid_mask]
            ps = ps[valid_mask]

            if len(xs) > 0:
                # Draw events directly on array (faster than cv2.circle for many points)
                pos_mask = ps > 0
                # Positive events - red
                event_frame[ys[pos_mask], xs[pos_mask]] = [0, 0, 255]
                # Negative events - blue
                event_frame[ys[~pos_mask], xs[~pos_mask]] = [255, 0, 0]

    def draw_stats(self, frame, stats, event_count):
        """Draw statistics overlay on the frame"""
        # Create semi-transparent overlay for stats
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 180), (240, 240, 240), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Define text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        color = (0, 0, 0)  # Black text

        # Status
        status = "PAUSED" if self.is_paused else "RUNNING"
        status_color = (0, 0, 255) if self.is_paused else (0, 128, 0)
        cv2.putText(frame, f"Status: {status}", (20, 30), font, font_scale, status_color, thickness)

        # Stats
        y_offset = 50
        cv2.putText(
            frame, f"FPS: {stats.current_fps:.1f}", (20, y_offset), font, font_scale, color, thickness
        )
        cv2.putText(
            frame, f"Events/Frame: {event_count}", (20, y_offset + 20), font, font_scale, color, thickness
        )
        cv2.putText(
            frame,
            f"Total Events: {stats.events_generated:,}",
            (20, y_offset + 40),
            font,
            font_scale,
            color,
            thickness,
        )
        cv2.putText(
            frame,
            f"Threshold: {self.contrast_threshold:.3f}",
            (20, y_offset + 60),
            font,
            font_scale,
            color,
            thickness,
        )
        cv2.putText(
            frame, f"Buffer: {stats.buffer_size:,}", (20, y_offset + 80), font, font_scale, color, thickness
        )
        cv2.putText(
            frame,
            f"Latency: {stats.avg_latency_ms:.1f}ms",
            (20, y_offset + 100),
            font,
            font_scale,
            color,
            thickness,
        )

        # Controls help (bottom of frame)
        help_y = frame.shape[0] - 60
        cv2.putText(
            frame,
            "Controls: Q=Quit R=Reset P=Pause +/-=Threshold S=Save V=Toggle Events",
            (10, help_y),
            font,
            0.4,
            (64, 64, 64),
            1,
        )

    def draw_graphs(self):
        """Draw simple performance graphs in the bottom area"""
        # Clear graph area
        graph_start_y = self.resolution[1]
        self.canvas[graph_start_y:, :] = 255  # White background

        # Draw FPS graph
        if len(self.fps_history) > 1:
            # Normalize FPS values to fit in graph area
            fps_values = list(self.fps_history)
            max_fps = max(fps_values) if fps_values and max(fps_values) > 0 else self.target_fps
            graph_height = 90
            graph_width = self.canvas_width - 40

            # Draw FPS line
            points = []
            if max_fps > 0:  # Prevent division by zero
                for i, fps in enumerate(fps_values):
                    x = int(20 + (i / len(fps_values)) * graph_width)
                    y = int(graph_start_y + 20 + graph_height - (fps / max_fps) * graph_height)
                    points.append([x, y])

            if len(points) > 1:
                points_array = np.array(points, np.int32)
                cv2.polylines(self.canvas, [points_array], False, (0, 255, 0), 2)

            # Draw target FPS line
            if max_fps > 0:  # Prevent division by zero
                target_y = int(graph_start_y + 20 + graph_height - (self.target_fps / max_fps) * graph_height)
                cv2.line(self.canvas, (20, target_y), (graph_width + 20, target_y), (0, 0, 255), 1)

            # Labels
            cv2.putText(
                self.canvas,
                f"FPS (max: {max_fps:.0f})",
                (20, graph_start_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
            )

    def run(self):
        """Run the main demo loop"""
        print("WebCam Event Stream Demo (OpenCV)")
        print("=" * 50)

        # Check if real-time streaming is supported
        if not self.check_realtime_support():
            return False

        # Initialize stream
        if not self.initialize_stream():
            return False

        # Start streaming
        if not self.start_streaming():
            return False

        print("\nStarting visualization...")
        print("Press 'q' to quit, 'p' to pause, '+'/'-' to adjust threshold")
        print(f"Real-time mode: {'ON' if self.realtime_mode else 'OFF'} (prioritizes low latency)")

        # Create window
        cv2.namedWindow("Event Stream", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Event Stream", self.canvas_width, self.canvas_height)

        try:
            while True:
                # Timing for FPS calculation
                current_time = time.time()
                self.frame_times.append(current_time - self.last_frame_time)
                self.last_frame_time = current_time

                # Process events
                events, event_count = self.process_frame()

                # Update buffers
                if events:
                    self.event_buffer.extend(events[-200:])  # Keep only most recent events

                # Get stats
                try:
                    stats = self.stream.get_stats()
                    self.fps_history.append(stats.current_fps)
                    self.event_count_history.append(event_count)
                except Exception:

                    class MockStats:
                        current_fps = 0
                        frames_processed = 0
                        events_generated = 0
                        buffer_size = 0
                        dropped_frames = 0
                        avg_latency_ms = 0

                    stats = MockStats()

                # Clear canvas
                self.canvas[: self.resolution[1], :] = 255  # White background for event area

                # Draw events
                event_frame = self.canvas[: self.resolution[1], :]
                self.draw_events(event_frame, list(self.event_buffer))

                # Draw stats overlay
                self.draw_stats(self.canvas, stats, event_count)

                # Draw performance graphs
                self.draw_graphs()

                # Show frame
                cv2.imshow("Event Stream", self.canvas)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    # Reset statistics
                    if self.stream:
                        self.stream.reset()
                    self.event_buffer.clear()
                    self.fps_history.clear()
                    self.event_count_history.clear()
                    print("Statistics reset!")
                elif key == ord("p"):
                    # Pause/unpause
                    self.is_paused = not self.is_paused
                    status = "paused" if self.is_paused else "resumed"
                    print(f"Streaming {status}")
                elif key == ord("+") or key == ord("="):
                    # Increase threshold
                    self.contrast_threshold = min(1.0, self.contrast_threshold + 0.01)
                    if self.stream:
                        self.stream.update_params(contrast_threshold=self.contrast_threshold)
                    print(f"Contrast threshold: {self.contrast_threshold:.3f}")
                elif key == ord("-"):
                    # Decrease threshold
                    self.contrast_threshold = max(0.01, self.contrast_threshold - 0.01)
                    if self.stream:
                        self.stream.update_params(contrast_threshold=self.contrast_threshold)
                    print(f"Contrast threshold: {self.contrast_threshold:.3f}")
                elif key == ord("s"):
                    # Save current frame
                    filename = f"event_frame_{int(time.time())}.png"
                    cv2.imwrite(filename, self.canvas)
                    print(f"Frame saved to {filename}")
                elif key == ord("v"):
                    # Toggle event visualization
                    self.show_events = not self.show_events
                    print(f"Event visualization: {'ON' if self.show_events else 'OFF'}")
                elif key == ord("m"):
                    # Toggle realtime mode
                    self.realtime_mode = not self.realtime_mode
                    print(
                        f"Real-time mode: {'ON (low latency)' if self.realtime_mode else 'OFF (all events)'}"
                    )

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"ERROR: Visualization failed: {e}")
        finally:
            cv2.destroyAllWindows()
            self.stop_streaming()

        return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Real-time webcam event stream demonstration (OpenCV)")
    parser.add_argument("--device_id", type=int, default=0, help="Camera device ID (default: 0)")
    parser.add_argument("--fps", type=float, default=60.0, help="Target processing FPS (default: 60.0)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        help="Contrast threshold (default: 0.25, higher = fewer events)",
    )
    parser.add_argument("--width", type=int, default=640, help="Camera width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Camera height (default: 480)")
    parser.add_argument("--buffer_size", type=int, default=1000, help="Event buffer size (default: 1000)")
    parser.add_argument(
        "--no-realtime", action="store_true", help="Disable realtime mode (process all events)"
    )

    args = parser.parse_args()

    # Create and run demo
    demo = WebcamEventDemoOpenCV(
        device_id=args.device_id,
        target_fps=args.fps,
        contrast_threshold=args.threshold,
        resolution=(args.width, args.height),
        max_buffer_size=args.buffer_size,
        realtime_mode=not args.no_realtime,
    )

    success = demo.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
