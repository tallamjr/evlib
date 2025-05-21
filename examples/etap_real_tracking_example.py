#!/usr/bin/env python3
"""
Real ETAP Point Tracking Example using evlib

This example demonstrates how to use the real ETAP model integration
for tracking arbitrary points through event streams.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import evlib
    from evlib.etap_integration import create_etap_tracker, ETAP_AVAILABLE, TORCH_AVAILABLE

    print(f"Using evlib version: {evlib.__version__}")
except ImportError:
    print("Error: evlib not found. Please install evlib first.")
    exit(1)


def create_synthetic_events(width=640, height=480, num_events=10000, duration=1.0):
    """Create synthetic event data with moving patterns."""

    # Create multiple moving objects
    num_objects = 3
    events_per_object = num_events // num_objects

    all_xs, all_ys, all_ts, all_ps = [], [], [], []

    for obj_id in range(num_objects):
        # Different motion patterns for each object
        if obj_id == 0:
            # Circular motion
            center_x, center_y = width // 4, height // 2
            radius = 50
            timestamps = np.linspace(obj_id * duration / 3, (obj_id + 1) * duration / 3, events_per_object)
            angles = np.linspace(0, 4 * np.pi, events_per_object)

            xs = (center_x + radius * np.cos(angles)).astype(np.int64)
            ys = (center_y + radius * np.sin(angles)).astype(np.int64)

        elif obj_id == 1:
            # Linear motion
            timestamps = np.linspace(obj_id * duration / 3, (obj_id + 1) * duration / 3, events_per_object)
            start_x, start_y = width // 2, height // 4
            end_x, end_y = width // 2, 3 * height // 4

            progress = np.linspace(0, 1, events_per_object)
            xs = (start_x + progress * (end_x - start_x)).astype(np.int64)
            ys = (start_y + progress * (end_y - start_y)).astype(np.int64)

        else:
            # Spiral motion
            center_x, center_y = 3 * width // 4, height // 2
            timestamps = np.linspace(obj_id * duration / 3, (obj_id + 1) * duration / 3, events_per_object)
            angles = np.linspace(0, 6 * np.pi, events_per_object)
            radius = np.linspace(10, 60, events_per_object)

            xs = (center_x + radius * np.cos(angles)).astype(np.int64)
            ys = (center_y + radius * np.sin(angles)).astype(np.int64)

        # Add noise to events
        noise_x = np.random.randint(-5, 6, len(xs))
        noise_y = np.random.randint(-5, 6, len(ys))
        xs = np.clip(xs + noise_x, 0, width - 1)
        ys = np.clip(ys + noise_y, 0, height - 1)

        # Generate polarities
        polarities = np.random.choice([-1, 1], len(xs)).astype(np.int64)

        all_xs.extend(xs)
        all_ys.extend(ys)
        all_ts.extend(timestamps)
        all_ps.extend(polarities)

    # Combine all events and sort by time
    combined = list(zip(all_xs, all_ys, all_ts, all_ps))
    combined.sort(key=lambda x: x[2])  # Sort by timestamp

    xs, ys, ts, ps = zip(*combined)
    return (
        np.array(xs, dtype=np.int64),
        np.array(ys, dtype=np.int64),
        np.array(ts, dtype=np.float64),
        np.array(ps, dtype=np.int64),
    )


def create_tracking_queries(width=640, height=480):
    """Create query points for tracking at key locations."""

    query_points = []

    # Query points for each moving object
    queries_config = [
        # Object 1 (circular): center and edge points
        (width // 4, height // 2),  # center
        (width // 4 + 50, height // 2),  # right edge
        (width // 4, height // 2 - 50),  # top edge
        # Object 2 (linear): start and middle
        (width // 2, height // 4),  # start
        (width // 2, height // 2),  # middle
        # Object 3 (spiral): center
        (3 * width // 4, height // 2),  # center
    ]

    for i, (x, y) in enumerate(queries_config):
        query = evlib.tracking.PyQueryPoint(
            frame_idx=0, x=float(x), y=float(y)  # Start tracking from first frame
        )
        query_points.append(query)

    return query_points


def main():
    """Main demonstration function."""

    print("🎯 Real ETAP Point Tracking Example with evlib")
    print("=" * 55)

    # Check availability
    if not TORCH_AVAILABLE:
        print("❌ PyTorch not available. Please install PyTorch.")
        return

    if not ETAP_AVAILABLE:
        print("❌ ETAP model not available. Using mock implementation.")
        use_real_etap = False
    else:
        print("✅ ETAP model available!")
        use_real_etap = True

    # Parameters
    width, height = 640, 480
    resolution = (width, height)
    num_events = 20000
    duration = 1.0

    # 1. Create synthetic event data
    print("\n📊 Creating synthetic event data...")
    xs, ys, timestamps, polarities = create_synthetic_events(width, height, num_events, duration)

    print(f"   Generated {len(xs)} events")
    print(f"   Time range: {timestamps.min():.3f} - {timestamps.max():.3f}s")
    print(f"   Spatial range: x=[{xs.min()}, {xs.max()}], y=[{ys.min()}, {ys.max()}]")

    # 2. Create tracking queries
    print("\n🎯 Setting up tracking queries...")
    query_points = create_tracking_queries(width, height)

    print(f"   Created {len(query_points)} query points:")
    for i, qp in enumerate(query_points):
        print(f"     Query {i}: {qp}")

    # 3. Track points using ETAP
    print(f"\n🧠 Tracking points with {'real ETAP' if use_real_etap else 'mock implementation'}...")

    if use_real_etap:
        try:
            # Look for ETAP model weights
            model_paths = [
                "/Users/tallam/github/tallamjr/clones/ETAP/weights/ETAP_v1_cvpr25.pth",
                "./ETAP_v1_cvpr25.pth",
                "./etap_model.pth",
            ]

            model_path = None
            for path in model_paths:
                if Path(path).exists():
                    model_path = path
                    break

            if model_path:
                print(f"   Using model: {model_path}")
            else:
                print("   No model weights found. Using untrained model.")

            # Create ETAP tracker
            tracker = create_etap_tracker(
                model_path=model_path,
                device="auto",  # Will automatically select best device
                window_len=8,
                model_resolution=(512, 512),
                num_bins=5,
            )

            # Track points
            track_results = tracker.track_points(
                (xs, ys, timestamps, polarities),
                query_points,
                resolution=resolution,
                iters=6,  # Number of optimization iterations
            )

            print(f"   Successfully tracked {len(track_results)} point trajectories")

        except Exception as e:
            print(f"   Error with real ETAP: {e}")
            print("   Falling back to mock implementation...")
            use_real_etap = False

    if not use_real_etap:
        # Use evlib's built-in mock tracking
        try:
            # Prepare event representation
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

            # Track using mock implementation
            track_results = evlib.tracking.track_points_mock(event_repr, query_points, num_frames=20)

            print(f"   Tracked {len(track_results)} point trajectories (mock)")

        except Exception as e:
            print(f"   Error with mock tracking: {e}")
            return

    # 4. Analyze results
    print("\n📈 Analyzing tracking results...")
    for track_id, result in track_results.items():
        coords = result.coords
        visibility = result.visibility

        if len(coords) > 0:
            start_point = coords[0]
            end_point = coords[-1]
            avg_visibility = np.mean(visibility)

            # Calculate total displacement
            displacement = np.sqrt((end_point.x - start_point.x) ** 2 + (end_point.y - start_point.y) ** 2)

            print(f"   Track {track_id}:")
            print(f"     Frames: {len(coords)}")
            print(f"     Start: ({start_point.x:.1f}, {start_point.y:.1f})")
            print(f"     End: ({end_point.x:.1f}, {end_point.y:.1f})")
            print(f"     Displacement: {displacement:.1f} pixels")
            print(f"     Avg visibility: {avg_visibility:.3f}")

            # Get visible points
            visible_points = result.visible_points(threshold=0.5)
            print(f"     Visible points (>0.5): {len(visible_points)}")

    # 5. Visualisation
    print("\n📊 Creating visualisation...")
    try:
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{"Real ETAP" if use_real_etap else "Mock"} Point Tracking with evlib', fontsize=16)

        # Plot 1: Event scatter plot with temporal coloring
        ax1 = axes[0, 0]
        scatter = ax1.scatter(xs[::50], ys[::50], c=timestamps[::50], s=2, alpha=0.7, cmap="viridis")
        ax1.set_title("Event Data (subsampled)")
        ax1.set_xlabel("X coordinate")
        ax1.set_ylabel("Y coordinate")
        ax1.set_aspect("equal")
        plt.colorbar(scatter, ax=ax1, label="Time (s)")

        # Add query points
        for i, qp in enumerate(query_points):
            ax1.plot(qp.point.x, qp.point.y, "ro", markersize=8, markeredgecolor="white", markeredgewidth=2)
            ax1.text(qp.point.x + 10, qp.point.y + 10, f"Q{i}", color="red", fontweight="bold")

        # Plot 2: Event density heatmap
        ax2 = axes[0, 1]
        hist, xedges, yedges = np.histogram2d(xs, ys, bins=50, range=[[0, width], [0, height]])
        ax2.imshow(hist.T, origin="lower", aspect="auto", cmap="hot", extent=[0, width, 0, height])
        ax2.set_title("Event Density Heatmap")
        ax2.set_xlabel("X coordinate")
        ax2.set_ylabel("Y coordinate")

        # Plot 3: Tracking trajectories
        ax3 = axes[1, 0]
        colors = plt.cm.tab10(np.linspace(0, 1, len(track_results)))

        for (track_id, result), color in zip(track_results.items(), colors):
            coords = result.coords
            visibility = result.visibility

            if len(coords) > 1:
                # Extract trajectory
                xs_track = [c.x for c in coords]
                ys_track = [c.y for c in coords]

                # Plot trajectory with visibility-based alpha
                for i in range(len(coords) - 1):
                    alpha = max(0.3, visibility[i]) if i < len(visibility) else 0.5
                    ax3.plot(
                        [xs_track[i], xs_track[i + 1]],
                        [ys_track[i], ys_track[i + 1]],
                        color=color,
                        alpha=alpha,
                        linewidth=2,
                    )

                # Mark start and end
                ax3.plot(
                    xs_track[0],
                    ys_track[0],
                    "o",
                    color=color,
                    markersize=10,
                    markeredgecolor="white",
                    markeredgewidth=2,
                    label=f"Track {track_id}",
                )
                ax3.plot(
                    xs_track[-1],
                    ys_track[-1],
                    "s",
                    color=color,
                    markersize=8,
                    markeredgecolor="white",
                    markeredgewidth=2,
                )

        ax3.set_title("Tracked Point Trajectories")
        ax3.set_xlabel("X coordinate")
        ax3.set_ylabel("Y coordinate")
        ax3.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax3.set_aspect("equal")

        # Plot 4: Visibility over time
        ax4 = axes[1, 1]
        for (track_id, result), color in zip(track_results.items(), colors):
            visibility = result.visibility
            if len(visibility) > 0:
                frames = range(len(visibility))
                ax4.plot(
                    frames,
                    visibility,
                    color=color,
                    linewidth=2,
                    marker="o",
                    markersize=4,
                    label=f"Track {track_id}",
                )

        ax4.set_title("Point Visibility Over Time")
        ax4.set_xlabel("Frame")
        ax4.set_ylabel("Visibility Score")
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.set_ylim(0, 1.1)

        plt.tight_layout()

        # Save plot
        output_path = Path(f"{'etap_real' if use_real_etap else 'etap_mock'}_tracking_example.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"   Saved visualisation to: {output_path}")

        # Show plot
        plt.show()

    except Exception as e:
        print(f"   Error creating visualisation: {e}")
        import traceback

        traceback.print_exc()

    print(f"\n✅ {'Real ETAP' if use_real_etap else 'Mock'} tracking example completed!")

    if use_real_etap:
        print("\n🎉 Successfully demonstrated real ETAP integration!")
        print("Benefits of real ETAP:")
        print("- Transformer-based tracking with attention mechanisms")
        print("- Superior handling of occlusions and fast motion")
        print("- State-of-the-art accuracy on event camera data")
    else:
        print("\nTo use real ETAP:")
        print("1. Install PyTorch: pip install torch")
        print("2. Download ETAP model weights")
        print("3. Ensure ETAP repository is accessible")

    print("\nNext steps:")
    print("- Use with real event camera data")
    print("- Integrate with object detection pipelines")
    print("- Add real-time streaming capabilities")


if __name__ == "__main__":
    main()
