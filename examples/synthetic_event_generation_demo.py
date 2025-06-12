#!/usr/bin/env python3
"""
Synthetic Event Generation Demo

This demonstration shows how to:
1. Generate synthetic video patterns
2. Simulate event camera data from video frames
3. Analyze and save event data

NOTE: This does NOT use real video files or GStreamer.
For real video processing, use external tools to extract frames
and then apply the event simulation pipeline shown here.

Author: evlib contributors
"""

import sys
import numpy as np
from pathlib import Path

try:
    import evlib  # noqa: F401

    print("✅ evlib imported successfully")
except ImportError as e:
    print(f"❌ Failed to import evlib: {e}")
    print("Please build evlib with: maturin develop")
    exit(1)


def check_video_file(video_path):
    """Check if video file exists and has supported format"""
    path = Path(video_path)

    if not path.exists():
        print(f"❌ Video file not found: {path}")
        return False

    # Check file extension
    supported_formats = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
    if path.suffix.lower() not in supported_formats:
        print(f"⚠️ Warning: Unsupported format {path.suffix}")
        print(f"Supported formats: {', '.join(supported_formats)}")

    print(f"✅ Video file found: {path}")
    print(f"  📂 Size: {path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  📝 Format: {path.suffix}")

    return True


def generate_synthetic_video(video_path_hint, max_frames=None):
    """
    Generate synthetic video frames for event simulation demonstration

    Args:
        video_path_hint: Video file path (used only for metadata simulation)
        max_frames: Maximum number of frames to generate (None for default)

    Returns:
        List of synthetic frames and metadata
    """
    print(f"🎬 Generating synthetic video patterns (inspired by: {video_path_hint})")

    try:
        # Create device (CPU for this demo)
        _device = "cpu"  # Will be used with actual GStreamer integration

        print("🎥 Initializing synthetic video generator...")

        # Video processing configuration
        config = {"output_resolution": (640, 480), "force_grayscale": True, "frame_rate": 30.0}

        print("⚙️ Configuration:")
        print(f"  📏 Resolution: {config['output_resolution']}")
        print(f"  🎨 Grayscale: {config['force_grayscale']}")
        print(f"  🎞️ Frame rate: {config['frame_rate']} fps")

        # Generate synthetic video frames
        print("📸 Generating synthetic video frames...")
        frames = []
        metadata = {
            "fps": 30.0,
            "duration": 5.0,  # Placeholder
            "total_frames": 150,  # 5 seconds at 30fps
            "resolution": config["output_resolution"],
        }

        # Process frames (simulated)
        num_frames = min(metadata["total_frames"], max_frames or metadata["total_frames"])

        for frame_idx in range(num_frames):
            # PLACEHOLDER: This creates synthetic frames instead of loading from GStreamer
            # TODO: Replace with actual GStreamer pipeline when implemented
            height, width = config["output_resolution"][1], config["output_resolution"][0]

            # Create moving pattern based on frame index
            x_grid, y_grid = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
            time_factor = frame_idx / metadata["fps"]

            # Moving sine wave pattern
            frame = 0.5 + 0.3 * np.sin(2 * np.pi * (x_grid * 2 + time_factor)) * np.cos(
                2 * np.pi * (y_grid * 2 + time_factor * 0.7)
            )

            frame = frame.astype(np.float32)
            frames.append(frame)

            # Progress update
            if (frame_idx + 1) % 30 == 0:
                progress = (frame_idx + 1) / num_frames * 100
                print(f"  📊 Progress: {progress:.1f}% ({frame_idx + 1}/{num_frames} frames)")

        print(f"✅ Synthetic video generation complete: {len(frames)} frames created")

        return frames, metadata

    except Exception as e:
        print(f"❌ Synthetic video generation failed: {e}")
        return [], {}


def simulate_events_from_video(frames, metadata, simulation_config=None):
    """
    Simulate events from video frames using ESIM-style algorithm

    Args:
        frames: List of video frames
        metadata: Video metadata
        simulation_config: Event simulation configuration

    Returns:
        Events dictionary with x, y, t, p arrays
    """
    print("⚡ Simulating events from video frames...")

    if not frames:
        print("❌ No frames to process")
        return None

    # Default simulation configuration
    if simulation_config is None:
        simulation_config = {
            "contrast_threshold_pos": 0.15,
            "contrast_threshold_neg": 0.15,
            "sigma_contrast": 0.03,
            "refractory_period_ns": 1000,  # 1 microsecond
        }

    print("⚙️ Simulation configuration:")
    for key, value in simulation_config.items():
        print(f"  {key}: {value}")

    try:
        # Simulate ESIM event generation
        all_events = {"x": [], "y": [], "t": [], "p": []}

        # Previous frame for comparison
        prev_frame = None
        frame_time_step = 1.0 / metadata["fps"]

        for frame_idx, frame in enumerate(frames):
            frame_time = frame_idx * frame_time_step

            if prev_frame is not None:
                # Calculate intensity change
                log_intensity_diff = np.log(frame + 1e-6) - np.log(prev_frame + 1e-6)

                # Positive events (brightness increase)
                pos_events = log_intensity_diff > simulation_config["contrast_threshold_pos"]
                pos_coords = np.where(pos_events)

                # Negative events (brightness decrease)
                neg_events = log_intensity_diff < -simulation_config["contrast_threshold_neg"]
                neg_coords = np.where(neg_events)

                # Add positive events
                if len(pos_coords[0]) > 0:
                    all_events["x"].extend(pos_coords[1])  # x = column
                    all_events["y"].extend(pos_coords[0])  # y = row
                    all_events["t"].extend([frame_time] * len(pos_coords[0]))
                    all_events["p"].extend([1] * len(pos_coords[0]))  # positive polarity

                # Add negative events
                if len(neg_coords[0]) > 0:
                    all_events["x"].extend(neg_coords[1])  # x = column
                    all_events["y"].extend(neg_coords[0])  # y = row
                    all_events["t"].extend([frame_time] * len(neg_coords[0]))
                    all_events["p"].extend([0] * len(neg_coords[0]))  # negative polarity

            prev_frame = frame.copy()

            # Progress update
            if (frame_idx + 1) % 30 == 0:
                total_events = len(all_events["x"])
                print(f"  📊 Frame {frame_idx + 1}/{len(frames)}: {total_events} events generated")

        # Convert to numpy arrays
        for key in all_events:
            all_events[key] = np.array(all_events[key])

        total_events = len(all_events["x"])
        duration = all_events["t"][-1] - all_events["t"][0] if total_events > 0 else 0
        event_rate = total_events / duration if duration > 0 else 0

        print("✅ Event simulation complete:")
        print(f"  ⚡ Total events: {total_events:,}")
        print(f"  ⏱️ Duration: {duration:.2f} seconds")
        print(f"  📊 Event rate: {event_rate:.0f} events/second")

        return all_events

    except Exception as e:
        print(f"❌ Event simulation failed: {e}")
        return None


def save_events_to_formats(events, output_prefix="video_events"):
    """Save events in multiple formats"""
    if events is None or len(events["x"]) == 0:
        print("❌ No events to save")
        return False

    try:
        # Save as text file (compatible with most event camera tools)
        txt_file = f"{output_prefix}.txt"
        print(f"💾 Saving events to {txt_file}")

        with open(txt_file, "w") as f:
            f.write("# x y t p\n")
            for i in range(len(events["x"])):
                f.write(f"{events['x'][i]} {events['y'][i]} {events['t'][i]:.6f} {events['p'][i]}\n")

        # Save as numpy arrays
        npz_file = f"{output_prefix}.npz"
        print(f"💾 Saving events to {npz_file}")
        np.savez(npz_file, **events)

        print("✅ Events saved in multiple formats")
        return True

    except Exception as e:
        print(f"❌ Failed to save events: {e}")
        return False


def analyze_events(events):
    """Analyze generated events and print statistics"""
    if events is None or len(events["x"]) == 0:
        print("❌ No events to analyze")
        return

    print("\n📈 Event Analysis")
    print("=" * 30)

    total_events = len(events["x"])
    pos_events = np.sum(events["p"] == 1)
    neg_events = np.sum(events["p"] == 0)

    duration = events["t"][-1] - events["t"][0]
    event_rate = total_events / duration

    print("📊 Event Statistics:")
    print(f"  Total events: {total_events:,}")
    print(f"  Positive events: {pos_events:,} ({pos_events/total_events*100:.1f}%)")
    print(f"  Negative events: {neg_events:,} ({neg_events/total_events*100:.1f}%)")
    print(f"  Duration: {duration:.3f} seconds")
    print(f"  Average rate: {event_rate:.0f} events/second")

    print("\n🎯 Spatial Distribution:")
    print(f"  X range: {np.min(events['x'])} - {np.max(events['x'])}")
    print(f"  Y range: {np.min(events['y'])} - {np.max(events['y'])}")

    print("\n⏰ Temporal Distribution:")
    print(f"  Time range: {events['t'][0]:.3f} - {events['t'][-1]:.3f} seconds")
    print(f"  First event: t={events['t'][0]:.6f}s at ({events['x'][0]}, {events['y'][0]})")
    print(f"  Last event: t={events['t'][-1]:.6f}s at ({events['x'][-1]}, {events['y'][-1]})")


def main():
    """Main demo function"""
    print("🎬 Synthetic Event Generation Demo")
    print("=" * 50)

    # Get video file path
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        # Use sample video or create placeholder
        video_path = "sample_video.mp4"
        print(f"ℹ️ No video file specified, using: {video_path}")
        print("Usage: python synthetic_event_generation_demo.py <pattern_name>")

    # Check video file
    if not check_video_file(video_path):
        print("💡 This demo generates synthetic video patterns.")
        print("  For real video processing, extract frames using external tools.")
        print("\n🎬 Proceeding with synthetic pattern generation...")
        video_path = "synthetic_video"

    # Step 1: Generate synthetic video
    print("\n🎬 Step 1: Synthetic Video Generation")
    frames, metadata = generate_synthetic_video(video_path, max_frames=100)  # Limit for demo

    if not frames:
        print("❌ Failed to generate synthetic video")
        return

    # Step 2: Simulate events
    print("\n⚡ Step 2: Event Simulation")
    events = simulate_events_from_video(frames, metadata)

    if events is None:
        print("❌ Failed to simulate events")
        return

    # Step 3: Analyze results
    analyze_events(events)

    # Step 4: Save results
    print("\n💾 Step 4: Save Results")
    save_events_to_formats(events, "synthetic_demo_events")

    print("\n✅ Demo completed successfully!")
    print("\n💡 Next steps:")
    print("  - Try with different video files")
    print("  - Adjust event simulation parameters")
    print("  - Use events for reconstruction with E2VID")
    print("  - Visualize events with evlib.visualization")


if __name__ == "__main__":
    main()
