"""
Temporal Event-to-Video Reconstruction Demo

This example demonstrates the use of E2VID+ and FireNet+ models
for temporal event reconstruction with ConvLSTM processing.
"""

import numpy as np
import matplotlib.pyplot as plt
import evlib


def create_moving_edge_events(width=128, height=128, duration=0.5, speed=50):
    """Create synthetic events simulating a moving vertical edge"""
    events = []
    num_timesteps = 1000

    for t_idx in range(num_timesteps):
        t = t_idx * duration / num_timesteps
        # Vertical edge position
        edge_x = int((t / duration) * width * 0.8 + width * 0.1)

        if edge_x < width - 1:
            # Generate events at the edge
            for y in range(height):
                # Positive events on the leading edge
                if np.random.random() < 0.7:
                    events.append([edge_x, y, t, 1])
                # Negative events on the trailing edge
                if edge_x > 1 and np.random.random() < 0.7:
                    events.append([edge_x - 1, y, t, -1])

    events = np.array(events)
    if len(events) > 0:
        xs = events[:, 0].astype(np.int64)
        ys = events[:, 1].astype(np.int64)
        ts = events[:, 2].astype(np.float64)
        ps = events[:, 3].astype(np.int64)
        return xs, ys, ts, ps
    else:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.int64),
        )


def visualize_temporal_reconstruction(xs, ys, ts, ps, height=128, width=128):
    """Visualize temporal reconstruction using E2VID+ and FireNet+"""

    # Number of frames to reconstruct
    num_frames = 8

    # E2VID+ reconstruction
    print("Running E2VID+ temporal reconstruction...")
    e2vid_frames = evlib.processing.events_to_video_temporal(
        xs,
        ys,
        ts,
        ps,
        height=height,
        width=width,
        num_frames=num_frames,
        num_bins=5,
        model_type="e2vid_plus_small",  # Use small variant for faster demo
    )

    # FireNet+ reconstruction
    print("Running FireNet+ temporal reconstruction...")
    firenet_frames = evlib.processing.events_to_video_temporal(
        xs,
        ys,
        ts,
        ps,
        height=height,
        width=width,
        num_frames=num_frames,
        num_bins=5,
        model_type="firenet_plus",
    )

    # Visualize results
    fig, axes = plt.subplots(2, num_frames, figsize=(16, 5))

    for i in range(num_frames):
        # E2VID+ frames
        axes[0, i].imshow(e2vid_frames[i, :, :, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"E2VID+ t={i+1}")
        axes[0, i].axis("off")

        # FireNet+ frames
        axes[1, i].imshow(firenet_frames[i, :, :, 0], cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_title(f"FireNet+ t={i+1}")
        axes[1, i].axis("off")

    plt.suptitle("Temporal Event Reconstruction: E2VID+ vs FireNet+")
    plt.tight_layout()
    plt.show()

    # Compare temporal consistency
    print("\nTemporal consistency analysis:")

    # E2VID+ frame differences
    e2vid_diffs = []
    for i in range(1, num_frames):
        diff = np.mean(np.abs(e2vid_frames[i] - e2vid_frames[i - 1]))
        e2vid_diffs.append(diff)

    # FireNet+ frame differences
    firenet_diffs = []
    for i in range(1, num_frames):
        diff = np.mean(np.abs(firenet_frames[i] - firenet_frames[i - 1]))
        firenet_diffs.append(diff)

    print(f"E2VID+ average frame difference: {np.mean(e2vid_diffs):.4f}")
    print(f"FireNet+ average frame difference: {np.mean(firenet_diffs):.4f}")

    return e2vid_frames, firenet_frames


def main():
    """Run the temporal reconstruction demo"""
    print("Temporal Event Reconstruction Demo")
    print("==================================")

    # Generate synthetic events
    print("\nGenerating synthetic moving edge events...")
    xs, ys, ts, ps = create_moving_edge_events()
    print(f"Generated {len(xs)} events over {ts[-1]:.3f} seconds")

    # Visualize event data
    plt.figure(figsize=(10, 4))

    # Plot events over time
    plt.subplot(1, 2, 1)
    plt.scatter(xs[ps > 0], ys[ps > 0], c="red", s=1, alpha=0.5, label="Positive")
    plt.scatter(xs[ps < 0], ys[ps < 0], c="blue", s=1, alpha=0.5, label="Negative")
    plt.xlim(0, 128)
    plt.ylim(0, 128)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Event Spatial Distribution")
    plt.legend()

    # Plot temporal distribution
    plt.subplot(1, 2, 2)
    plt.hist(ts, bins=50, alpha=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Event Count")
    plt.title("Event Temporal Distribution")

    plt.tight_layout()
    plt.show()

    # Run temporal reconstruction
    e2vid_frames, firenet_frames = visualize_temporal_reconstruction(xs, ys, ts, ps)

    print("\nDemo completed!")
    print("\nKey observations:")
    print("- E2VID+ uses ConvLSTM for temporal memory across frames")
    print("- FireNet+ provides faster inference with temporal gating")
    print("- Both models maintain temporal consistency better than frame-by-frame methods")


if __name__ == "__main__":
    main()
