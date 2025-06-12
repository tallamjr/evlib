#!/usr/bin/env python3
"""
Ultra-high-performance terminal-based webcam event stream demonstration

This script demonstrates real-time event generation from webcam input using
evlib's GStreamer integration and displays events directly in the terminal
using Ratatui for maximum performance.

Key advantages of terminal visualization:
- Zero GUI overhead (no OpenCV, no X11/Wayland)
- Ultra-fast terminal rendering
- Works over SSH and in headless environments
- Minimal resource usage
- Very high frame rates possible

Requirements:
- evlib compiled with GStreamer and terminal support
- Webcam connected to the system
- Terminal with good Unicode support

Usage:
    python webcam_event_demo_terminal.py [--device_id 0] [--fps 60] [--threshold 0.25]

Terminal Controls:
    q, Esc     - Quit
    p, Space   - Pause/Resume
    r          - Reset statistics
    s          - Toggle statistics display
    +/-        - Adjust event decay time
    h, F1      - Toggle help

Build with terminal support:
    maturin develop --features "python terminal gstreamer"
"""

import argparse
import sys
import time
import signal
from threading import Thread, Event as ThreadEvent
from queue import Queue, Empty

try:
    import evlib  # noqa: F401

    # Check for real-time streaming support
    is_realtime_available = evlib.simulation.is_realtime_available
    create_realtime_stream_py = evlib.simulation.create_realtime_stream_py

    # Check for terminal visualization support
    try:
        PyTerminalVisualizationConfig = evlib.visualization.PyTerminalVisualizationConfig
        PyTerminalEventVisualizer = evlib.visualization.PyTerminalEventVisualizer
        create_terminal_event_viewer = evlib.visualization.create_terminal_event_viewer
        terminal_support = True
    except AttributeError:
        terminal_support = False
        print("ERROR: Terminal visualization not available!")
        print("Please compile evlib with terminal support:")
        print("  maturin develop --features 'python terminal gstreamer'")
        sys.exit(1)

except ImportError as e:
    print(f"Error importing evlib: {e}")
    print("Please ensure evlib is installed with: pip install -e .")
    sys.exit(1)


class TerminalWebcamEventDemo:
    """Ultra-high-performance terminal-based webcam event streaming demonstration"""

    def __init__(
        self,
        device_id=0,
        target_fps=60.0,
        contrast_threshold=0.25,
        resolution=(640, 480),
        max_buffer_size=500,  # Smaller buffer for ultra-low latency
        event_decay_ms=50.0,  # Fast decay for terminal
    ):
        self.device_id = device_id
        self.target_fps = target_fps
        self.contrast_threshold = contrast_threshold
        self.resolution = resolution
        self.max_buffer_size = max_buffer_size
        self.event_decay_ms = event_decay_ms

        # Initialize stream
        self.stream = None
        self.is_running = False

        # Terminal visualizer
        self.terminal_viz = None

        # Threading for event processing
        self.event_queue = Queue(maxsize=1000)
        self.shutdown_event = ThreadEvent()
        self.event_thread = None

        # Statistics
        self.stats_interval = 2.0
        self.last_stats_time = time.time()

    def check_terminal_support(self):
        """Check if terminal visualization is available"""
        return terminal_support

    def check_realtime_support(self):
        """Check if real-time streaming is available"""
        if not is_realtime_available():
            print("ERROR: Real-time streaming not available!")
            print("evlib was not compiled with GStreamer support.")
            print("Please compile with: maturin develop --features 'python terminal gstreamer'")
            return False
        return True

    def initialize_stream(self):
        """Initialize the real-time event stream"""
        try:
            print("Initializing ultra-fast terminal event stream...")
            print(f"  Device ID: {self.device_id}")
            print(f"  Target FPS: {self.target_fps}")
            print(f"  Contrast threshold: {self.contrast_threshold}")
            print(f"  Resolution: {self.resolution}")
            print(f"  Buffer size: {self.max_buffer_size}")
            print(f"  Event decay: {self.event_decay_ms}ms")

            # Create stream with aggressive real-time settings
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

    def initialize_terminal_visualizer(self):
        """Initialize the terminal visualizer"""
        try:
            print("Initializing terminal visualizer...")

            # Create terminal visualization config
            config = PyTerminalVisualizationConfig(
                event_decay_ms=self.event_decay_ms,
                max_events=2000,  # Reasonable for terminal
                target_fps=self.target_fps,
                show_stats=True,
                canvas_scale=1.0,
            )

            # Create terminal visualizer
            self.terminal_viz = PyTerminalEventVisualizer(config)

            print("Terminal visualizer ready!")
            return True

        except Exception as e:
            print(f"ERROR: Failed to initialize terminal visualizer: {e}")
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

    def event_producer_thread(self):
        """Background thread that produces events from the camera"""
        print("Event producer thread started")

        consecutive_empty = 0
        max_consecutive_empty = 30  # Reset if no events for too long

        while not self.shutdown_event.is_set() and self.is_running:
            try:
                # Process next frame
                frame_available = self.stream.process_next_frame()

                if frame_available:
                    # Get events (consume all to prevent buffer buildup)
                    xs, ys, ts, ps = self.stream.get_events(max_count=None)

                    if len(xs) > 0:
                        consecutive_empty = 0

                        # Limit events for terminal rendering performance
                        if len(xs) > 500:
                            # Sample events uniformly
                            import numpy as np

                            indices = np.linspace(0, len(xs) - 1, 500, dtype=int)
                            xs = [xs[i] for i in indices]
                            ys = [ys[i] for i in indices]
                            ts = [ts[i] for i in indices]
                            ps = [ps[i] for i in indices]

                        # Put events in queue (non-blocking)
                        try:
                            self.event_queue.put((xs, ys, ts, ps), block=False)
                        except:
                            # Queue full, skip these events to maintain real-time
                            pass
                    else:
                        consecutive_empty += 1
                        if consecutive_empty > max_consecutive_empty:
                            # Reset stream to clear any stuck state
                            self.stream.reset()
                            consecutive_empty = 0
                else:
                    consecutive_empty += 1

            except Exception as e:
                print(f"Event producer error: {e}")
                break

            # Brief sleep to prevent busy waiting
            time.sleep(0.001)

        print("Event producer thread finished")

    def run(self):
        """Run the ultra-fast terminal demo"""
        print("Ultra-Fast Terminal Event Stream Demo")
        print("=" * 50)

        # Check prerequisites
        if not self.check_terminal_support():
            return False

        if not self.check_realtime_support():
            return False

        # Initialize components
        if not self.initialize_stream():
            return False

        if not self.initialize_terminal_visualizer():
            return False

        # Start streaming
        if not self.start_streaming():
            return False

        print("\\nStarting terminal visualization...")
        print("Terminal will take over in 2 seconds...")
        print("Press 'h' for help once in terminal mode")
        time.sleep(2)

        # Set up signal handling for clean shutdown
        def signal_handler(signum, frame):
            self.shutdown_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start event producer thread
        self.event_thread = Thread(target=self.event_producer_thread, daemon=True)
        self.event_thread.start()

        success = True
        try:
            # Main terminal loop
            while not self.shutdown_event.is_set():
                # Handle input
                if not self.terminal_viz.handle_input():
                    break

                # Get events from queue
                try:
                    xs, ys, ts, ps = self.event_queue.get_nowait()

                    # Add events to terminal visualizer
                    import numpy as np

                    self.terminal_viz.add_events(
                        np.array(xs, dtype=np.int64),
                        np.array(ys, dtype=np.int64),
                        np.array(ts, dtype=np.float64),
                        np.array(ps, dtype=np.int64),
                    )

                except Empty:
                    # No new events, that's okay
                    pass
                except Exception as e:
                    print(f"Event processing error: {e}")

                # Render frame
                try:
                    self.terminal_viz.render_frame()
                except Exception as e:
                    print(f"Render error: {e}")
                    break

                # Print stats periodically to stderr (so it doesn't interfere with terminal UI)
                current_time = time.time()
                if current_time - self.last_stats_time > self.stats_interval:
                    frames, events, fps = self.terminal_viz.get_stats()
                    # Note: This will be hidden by the terminal UI, but useful for debugging
                    # sys.stderr.write(f"\\rStats: {frames} frames, {events} events, {fps:.1f} FPS\\n")
                    self.last_stats_time = current_time

        except KeyboardInterrupt:
            print("\\nInterrupted by user")
        except Exception as e:
            print(f"\\nERROR: Terminal visualization failed: {e}")
            success = False
        finally:
            # Clean shutdown
            self.shutdown_event.set()

            if self.event_thread:
                self.event_thread.join(timeout=1.0)

            self.stop_streaming()

            # Terminal cleanup is handled automatically by the Rust Drop implementation

        return success


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Ultra-fast terminal webcam event stream demonstration")
    parser.add_argument("--device_id", type=int, default=0, help="Camera device ID (default: 0)")
    parser.add_argument("--fps", type=float, default=60.0, help="Target processing FPS (default: 60.0)")
    parser.add_argument("--threshold", type=float, default=0.25, help="Contrast threshold (default: 0.25)")
    parser.add_argument("--width", type=int, default=640, help="Camera width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Camera height (default: 480)")
    parser.add_argument("--buffer_size", type=int, default=500, help="Event buffer size (default: 500)")
    parser.add_argument("--decay", type=float, default=50.0, help="Event decay time in ms (default: 50.0)")

    args = parser.parse_args()

    # Create and run demo
    demo = TerminalWebcamEventDemo(
        device_id=args.device_id,
        target_fps=args.fps,
        contrast_threshold=args.threshold,
        resolution=(args.width, args.height),
        max_buffer_size=args.buffer_size,
        event_decay_ms=args.decay,
    )

    success = demo.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
