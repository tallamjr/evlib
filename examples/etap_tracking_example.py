#!/usr/bin/env python3
"""
ETAP Point Tracking Example using evlib

This example demonstrates how to use the new ETAP integration in evlib
for tracking arbitrary points through event streams.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import evlib

    print(f"Using evlib version: {evlib.__version__}")
except ImportError:
    print("Error: evlib not found. Please install evlib first.")
    exit(1)


def create_synthetic_events(width=640, height=480, num_events=10000):
    """Create synthetic event data for demonstration."""

    # Create a moving circle of events
    center_x = width // 2
    center_y = height // 2
    radius = 50

    # Generate timestamps
    timestamps = np.linspace(0, 1.0, num_events)

    # Generate coordinates for a moving circle
    angles = np.random.uniform(0, 2 * np.pi, num_events)
    radii = np.random.uniform(0, radius, num_events)

    # Add circular motion to the center
    motion_freq = 2.0  # frequency of circular motion
    center_motion_x = 50 * np.cos(2 * np.pi * motion_freq * timestamps)
    center_motion_y = 30 * np.sin(2 * np.pi * motion_freq * timestamps)

    # Calculate event coordinates
    xs = (center_x + center_motion_x + radii * np.cos(angles)).astype(np.int64)
    ys = (center_y + center_motion_y + radii * np.sin(angles)).astype(np.int64)

    # Clip to bounds
    xs = np.clip(xs, 0, width - 1)
    ys = np.clip(ys, 0, height - 1)

    # Generate polarities (random for this example)
    polarities = np.random.choice([-1, 1], num_events).astype(np.int64)

    return xs, ys, timestamps, polarities


def create_object_mask(width=640, height=480):
    """Create a simple circular mask representing an object to track."""

    mask = np.zeros((height, width), dtype=bool)

    # Create circular mask
    center_x, center_y = width // 2, height // 2
    radius = 40

    y, x = np.ogrid[:height, :width]
    mask_circle = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2
    mask[mask_circle] = True

    return mask


def main():
    """Main demonstration function."""

    print("🎯 ETAP Point Tracking Example with evlib")
    print("=" * 50)

    # Parameters
    width, height = 640, 480
    resolution = (width, height)
    num_events = 50000

    # 1. Create synthetic event data
    print("📊 Creating synthetic event data...")
    xs, ys, timestamps, polarities = create_synthetic_events(width, height, num_events)

    print(f"   Generated {len(xs)} events")
    print(f"   Time range: {timestamps.min():.3f} - {timestamps.max():.3f}s")
    print(f"   Spatial range: x=[{xs.min()}, {xs.max()}], y=[{ys.min()}, {ys.max()}]")

    # 2. Create object mask for keypoint extraction
    print("\n🎭 Creating object mask...")
    object_mask = create_object_mask(width, height)
    print(f"   Mask contains {object_mask.sum()} pixels")

    # 3. Extract keypoints from the mask
    print("\n🔍 Extracting keypoints...")
    try:
        keypoints = evlib.tracking.extract_keypoints_from_mask(
            object_mask,
            method="contour",  # options: "contour", "grid", "skeleton", "corners"
            num_points=8,
            min_distance=10.0,
        )

        print(f"   Extracted {len(keypoints)} keypoints:")
        for i, kp in enumerate(keypoints):
            print(f"     Point {i}: ({kp.x:.1f}, {kp.y:.1f})")

    except Exception as e:
        print(f"   Error extracting keypoints: {e}")
        return

    # 4. Create query points for tracking
    print("\n🎯 Setting up tracking queries...")
    query_points = []
    for i, kp in enumerate(keypoints[:5]):  # Track first 5 keypoints
        query_point = evlib.tracking.PyQueryPoint(frame_idx=0, x=kp.x, y=kp.y)  # Start tracking from frame 0
        query_points.append(query_point)
        print(f"   Query {i}: {query_point}")

    # 5. Prepare event representation for ETAP
    print("\n🧠 Preparing event representation...")
    try:
        event_repr = evlib.tracking.prepare_event_representation(
            xs,
            ys,
            timestamps,
            polarities,
            resolution=resolution,
            window_length=8,
            num_bins=5,
            voxel_method="count",
        )

        print(f"   Event representation shape: {event_repr.shape}")
        print(f"   Data type: {event_repr.dtype}")
        print(f"   Value range: [{event_repr.min():.3f}, {event_repr.max():.3f}]")

    except Exception as e:
        print(f"   Error preparing representation: {e}")
        return

    # 6. Track points using mock ETAP
    print("\n🔄 Tracking points...")
    try:
        # Flatten the event representation for the mock function
        event_repr_flat = event_repr.flatten().astype(np.float32)

        track_results = evlib.tracking.track_points_mock(event_repr_flat, query_points, num_frames=10)

        print(f"   Tracked {len(track_results)} point trajectories")

        # Display tracking results
        for track_id, result in track_results.items():
            coords = result.coords
            visibility = result.visibility
            print(f"   Track {track_id}: {len(coords)} frames")
            print(f"     Start: ({coords[0].x:.1f}, {coords[0].y:.1f})")
            print(f"     End: ({coords[-1].x:.1f}, {coords[-1].y:.1f})")
            print(f"     Avg visibility: {np.mean(visibility):.3f}")

            # Get visible points
            visible_pts = result.visible_points(threshold=0.5)
            print(f"     Visible points (>0.5): {len(visible_pts)}")

    except Exception as e:
        print(f"   Error during tracking: {e}")
        return

    # 7. Visualisation
    print("\n📊 Creating visualisation...")
    try:
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("ETAP Point Tracking with evlib", fontsize=16)

        # Plot 1: Event scatter plot
        ax1 = axes[0, 0]
        scatter = ax1.scatter(xs[::100], ys[::100], c=timestamps[::100], s=1, alpha=0.6, cmap="viridis")
        ax1.set_title("Event Data (subsampled)")
        ax1.set_xlabel("X coordinate")
        ax1.set_ylabel("Y coordinate")
        ax1.set_aspect("equal")
        plt.colorbar(scatter, ax=ax1, label="Time (s)")

        # Plot 2: Object mask with keypoints
        ax2 = axes[0, 1]
        ax2.imshow(object_mask, cmap="gray", alpha=0.7)
        for i, kp in enumerate(keypoints):
            ax2.plot(kp.x, kp.y, "ro", markersize=8)
            ax2.text(kp.x + 5, kp.y + 5, str(i), color="red", fontweight="bold")
        ax2.set_title("Object Mask with Keypoints")
        ax2.set_xlabel("X coordinate")
        ax2.set_ylabel("Y coordinate")

        # Plot 3: Event representation (first channel)
        ax3 = axes[1, 0]
        if len(event_repr.shape) == 5:  # [B, T, C, H, W]
            slice_data = event_repr[0, 0, 0, :, :]  # First batch, time, channel
        else:
            slice_data = event_repr[0, :, :]  # Fallback
        ax3.imshow(slice_data, cmap="hot", aspect="auto")
        ax3.set_title("Event Representation (Voxel Grid)")
        ax3.set_xlabel("X coordinate")
        ax3.set_ylabel("Y coordinate")

        # Plot 4: Tracking trajectories
        ax4 = axes[1, 1]
        colors = plt.cm.tab10(np.linspace(0, 1, len(track_results)))

        for (track_id, result), color in zip(track_results.items(), colors):
            coords = result.coords
            visibility = result.visibility

            # Plot trajectory
            xs_track = [c.x for c in coords]
            ys_track = [c.y for c in coords]

            # Use visibility for alpha
            for i in range(len(coords) - 1):
                alpha = visibility[i]
                ax4.plot(
                    [xs_track[i], xs_track[i + 1]],
                    [ys_track[i], ys_track[i + 1]],
                    color=color,
                    alpha=alpha,
                    linewidth=2,
                )

            # Mark start and end
            ax4.plot(
                xs_track[0], ys_track[0], "o", color=color, markersize=8, label=f"Track {track_id} start"
            )
            ax4.plot(xs_track[-1], ys_track[-1], "s", color=color, markersize=8)

        ax4.set_title("Tracked Point Trajectories")
        ax4.set_xlabel("X coordinate")
        ax4.set_ylabel("Y coordinate")
        ax4.legend()
        ax4.set_aspect("equal")

        plt.tight_layout()

        # Save plot
        output_path = Path("etap_tracking_example.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"   Saved visualisation to: {output_path}")

        # Show plot
        plt.show()

    except Exception as e:
        print(f"   Error creating visualisation: {e}")
        import traceback

        traceback.print_exc()

    print("\n✅ ETAP tracking example completed successfully!")
    print("\nNext steps:")
    print("- Replace mock tracking with real ETAP model")
    print("- Integrate with actual event camera data")
    print("- Add real-time tracking capabilities")
    print("- Combine with object detection for full pipeline")


if __name__ == "__main__":
    main()
