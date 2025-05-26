#!/usr/bin/env python3
"""
Real-time webcam event stream demonstration

This script demonstrates real-time event generation from webcam input using
evlib's GStreamer integration and ESIM event simulation.

Requirements:
- evlib compiled with GStreamer support
- Webcam connected to the system
- matplotlib for visualization
- numpy

Usage:
    python webcam_event_demo.py [--device_id 0] [--fps 30] [--threshold 0.15]

Controls:
    Press 'q' to quit
    Press 'r' to reset statistics
    Press '+'/'-' to adjust contrast threshold
    Press 'p' to pause/unpause
"""

import argparse
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

try:
    import evlib  # noqa: F401
    from evlib.simulation import (
        is_realtime_available,
        create_realtime_stream_py,
    )
except ImportError as e:
    print(f"Error importing evlib: {e}")
    print("Please ensure evlib is installed with: pip install -e .")
    sys.exit(1)


class WebcamEventDemo:
    """Interactive webcam event streaming demonstration"""

    def __init__(
        self,
        device_id=0,
        target_fps=30.0,
        contrast_threshold=0.15,
        resolution=(640, 480),
        max_buffer_size=10000,
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

        # Initialize stream
        self.stream = None
        self.is_running = False
        self.is_paused = False

        # Visualization state
        self.event_history = deque(maxlen=1000)  # Keep last 1000 events for visualization
        self.fps_history = deque(maxlen=50)  # FPS tracking
        self.event_count_history = deque(maxlen=50)  # Event count tracking

        # Setup matplotlib
        self.setup_visualization()

    def setup_visualization(self):
        """Setup matplotlib visualization with subplots"""
        plt.style.use("dark_background")
        self.fig, ((self.ax_events, self.ax_stats), (self.ax_fps, self.ax_count)) = plt.subplots(
            2, 2, figsize=(12, 8)
        )

        # Event visualization subplot
        self.ax_events.set_xlim(0, self.resolution[0])
        self.ax_events.set_ylim(self.resolution[1], 0)  # Flip Y axis for image coordinates
        self.ax_events.set_title("Real-time Event Stream")
        self.ax_events.set_xlabel("X (pixels)")
        self.ax_events.set_ylabel("Y (pixels)")
        self.ax_events.set_aspect("equal")

        # Statistics text subplot
        self.ax_stats.axis("off")
        self.stats_text = self.ax_stats.text(
            0.05,
            0.95,
            "",
            transform=self.ax_stats.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )

        # FPS history subplot
        self.ax_fps.set_title("Processing FPS")
        self.ax_fps.set_xlabel("Time (samples)")
        self.ax_fps.set_ylabel("FPS")
        self.ax_fps.grid(True, alpha=0.3)

        # Event count history subplot
        self.ax_count.set_title("Events per Frame")
        self.ax_count.set_xlabel("Time (samples)")
        self.ax_count.set_ylabel("Event Count")
        self.ax_count.grid(True, alpha=0.3)

        # Connect keyboard events
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

        plt.tight_layout()

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
            # Process next frame
            frame_available = self.stream.process_next_frame()

            if frame_available:
                # Get events from buffer
                xs, ys, ts, ps = self.stream.get_events(max_count=1000)  # Limit for visualization

                # Convert to Python lists if numpy arrays
                if hasattr(xs, "__len__") and len(xs) > 0:
                    events = list(zip(xs, ys, ts, ps))
                    return events, len(events)

            return [], 0

        except Exception as e:
            print(f"ERROR: Frame processing failed: {e}")
            return [], 0

    def update_visualization(self, frame_num):
        """Update the matplotlib visualization"""
        if not self.is_running:
            return

        # Process events
        events, event_count = self.process_frame()

        # Get streaming statistics
        try:
            stats = self.stream.get_stats()
            current_fps = stats.current_fps
            total_frames = stats.frames_processed
            total_events = stats.events_generated
            buffer_size = stats.buffer_size
            dropped_frames = stats.dropped_frames
            latency = stats.avg_latency_ms
        except Exception:
            current_fps = 0
            total_frames = 0
            total_events = 0
            buffer_size = 0
            dropped_frames = 0
            latency = 0

        # Update event history
        self.event_history.extend(events)
        self.fps_history.append(current_fps)
        self.event_count_history.append(event_count)

        # Update event visualization
        self.ax_events.clear()
        self.ax_events.set_xlim(0, self.resolution[0])
        self.ax_events.set_ylim(self.resolution[1], 0)
        self.ax_events.set_title(f"Real-time Event Stream (Threshold: {self.contrast_threshold:.3f})")
        self.ax_events.set_xlabel("X (pixels)")
        self.ax_events.set_ylabel("Y (pixels)")

        # Plot recent events
        if self.event_history:
            recent_events = list(self.event_history)[-500:]  # Show last 500 events
            pos_events = [(x, y) for x, y, t, p in recent_events if p > 0]
            neg_events = [(x, y) for x, y, t, p in recent_events if p < 0]

            if pos_events:
                pos_x, pos_y = zip(*pos_events)
                self.ax_events.scatter(
                    pos_x, pos_y, c="red", s=1, alpha=0.7, label=f"Positive ({len(pos_events)})"
                )

            if neg_events:
                neg_x, neg_y = zip(*neg_events)
                self.ax_events.scatter(
                    neg_x, neg_y, c="blue", s=1, alpha=0.7, label=f"Negative ({len(neg_events)})"
                )

            if pos_events or neg_events:
                self.ax_events.legend()

        # Update statistics text
        status = "PAUSED" if self.is_paused else "RUNNING"
        stats_text = f"""
Status: {status}
Frames Processed: {total_frames:,}
Total Events: {total_events:,}
Current FPS: {current_fps:.1f}
Events/Frame: {event_count}
Buffer Size: {buffer_size:,}
Dropped Frames: {dropped_frames:,}
Latency: {latency:.1f} ms

Controls:
'q' - Quit
'r' - Reset stats
'p' - Pause/Resume
'+'/'-' - Adjust threshold
        """.strip()

        self.stats_text.set_text(stats_text)

        # Update FPS plot
        if self.fps_history:
            self.ax_fps.clear()
            self.ax_fps.plot(list(self.fps_history), "g-", linewidth=2)
            self.ax_fps.set_title("Processing FPS")
            self.ax_fps.set_xlabel("Time (samples)")
            self.ax_fps.set_ylabel("FPS")
            self.ax_fps.grid(True, alpha=0.3)
            self.ax_fps.axhline(y=self.target_fps, color="r", linestyle="--", alpha=0.7, label="Target")
            self.ax_fps.legend()

        # Update event count plot
        if self.event_count_history:
            self.ax_count.clear()
            self.ax_count.plot(list(self.event_count_history), "b-", linewidth=2)
            self.ax_count.set_title("Events per Frame")
            self.ax_count.set_xlabel("Time (samples)")
            self.ax_count.set_ylabel("Event Count")
            self.ax_count.grid(True, alpha=0.3)

    def on_key_press(self, event):
        """Handle keyboard input"""
        if event.key == "q":
            self.stop_streaming()
            plt.close("all")
            sys.exit(0)
        elif event.key == "r":
            # Reset statistics
            if self.stream:
                self.stream.reset()
            self.event_history.clear()
            self.fps_history.clear()
            self.event_count_history.clear()
            print("Statistics reset!")
        elif event.key == "p":
            # Pause/unpause
            self.is_paused = not self.is_paused
            status = "paused" if self.is_paused else "resumed"
            print(f"Streaming {status}")
        elif event.key == "+" or event.key == "=":
            # Increase threshold
            self.contrast_threshold = min(1.0, self.contrast_threshold + 0.01)
            if self.stream:
                self.stream.update_params(contrast_threshold=self.contrast_threshold)
            print(f"Contrast threshold: {self.contrast_threshold:.3f}")
        elif event.key == "-":
            # Decrease threshold
            self.contrast_threshold = max(0.01, self.contrast_threshold - 0.01)
            if self.stream:
                self.stream.update_params(contrast_threshold=self.contrast_threshold)
            print(f"Contrast threshold: {self.contrast_threshold:.3f}")

    def run(self):
        """Run the main demo loop"""
        print("WebCam Event Stream Demo")
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

        try:
            # Start animation
            _ani = animation.FuncAnimation(
                self.fig,
                self.update_visualization,
                interval=33,  # ~30 FPS updates
                cache_frame_data=False,
                blit=False,
            )

            plt.show()

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"ERROR: Visualization failed: {e}")
        finally:
            self.stop_streaming()

        return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Real-time webcam event stream demonstration")
    parser.add_argument("--device_id", type=int, default=0, help="Camera device ID (default: 0)")
    parser.add_argument("--fps", type=float, default=30.0, help="Target processing FPS (default: 30.0)")
    parser.add_argument("--threshold", type=float, default=0.15, help="Contrast threshold (default: 0.15)")
    parser.add_argument("--width", type=int, default=640, help="Camera width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Camera height (default: 480)")
    parser.add_argument("--buffer_size", type=int, default=10000, help="Event buffer size (default: 10000)")

    args = parser.parse_args()

    # Create and run demo
    demo = WebcamEventDemo(
        device_id=args.device_id,
        target_fps=args.fps,
        contrast_threshold=args.threshold,
        resolution=(args.width, args.height),
        max_buffer_size=args.buffer_size,
    )

    success = demo.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
