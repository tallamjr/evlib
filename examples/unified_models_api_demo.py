#!/usr/bin/env python3
"""
Demonstration of the unified models API in evlib.

This example shows how to use the high-level API to load
different event-to-video reconstruction models and process events.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import evlib
from evlib import models


def load_sample_events(path: Path = None):
    """Load sample event data."""
    if path is None:
        path = Path("data/slider_depth/events.txt")

    if path.exists():
        print(f"Loading events from {path}")
        xs, ys, ts, ps = evlib.formats.load_events(str(path))
        # Use only first 10k events for demo
        n = min(10000, len(xs))
        return xs[:n], ys[:n], ts[:n], ps[:n]
    else:
        # Generate synthetic events
        print("Generating synthetic events for demo")
        n_events = 10000
        width, height = 240, 180

        # Create events along a moving edge
        t = np.linspace(0, 1, n_events)
        x = (50 + 140 * t + 20 * np.sin(10 * t)).astype(np.int64)
        y = (90 + 30 * np.sin(5 * t)).astype(np.int64)
        p = np.random.choice([-1, 1], size=n_events).astype(np.int64)

        # Add some noise
        x = np.clip(x + np.random.randint(-5, 6, n_events), 0, width - 1)
        y = np.clip(y + np.random.randint(-5, 6, n_events), 0, height - 1)

        return x, y, t, p


def demo_basic_usage():
    """Demonstrate basic model usage."""
    print("\n=== Basic Model Usage ===")

    # Load events
    events = load_sample_events()

    # Create a model with default configuration
    model = models.E2VID(pretrained=False)  # Set to True when weights are available
    print(f"Created model: {model}")

    # Reconstruct a single frame
    frame = model.reconstruct(events)
    print(f"Reconstructed frame shape: {frame.shape}")

    # Display the result
    plt.figure(figsize=(8, 6))
    plt.imshow(frame, cmap="gray")
    plt.title("E2VID Reconstruction")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("examples/figures/unified_api_basic.png", dpi=150)
    plt.close()


def demo_different_models():
    """Compare different models."""
    print("\n=== Comparing Different Models ===")

    # Load events
    events = load_sample_events()
    xs, ys, ts, ps = events
    height = int(np.max(ys)) + 1
    width = int(np.max(xs)) + 1

    # Define models to compare
    model_configs = [
        ("E2VID", models.E2VID()),
        ("FireNet", models.FireNet()),
        # ("SPADE", models.SPADE(variant="lite")),
        # ("SSL", models.SSL()),
    ]

    # Create comparison plot
    fig, axes = plt.subplots(1, len(model_configs), figsize=(5 * len(model_configs), 5))
    if len(model_configs) == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, model_configs):
        print(f"Processing with {name}...")
        frame = model.reconstruct(events, height=height, width=width)

        if frame.ndim == 3:
            # If multiple frames, show the last one
            frame = frame[-1]

        ax.imshow(frame, cmap="gray")
        ax.set_title(name)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("examples/figures/unified_api_comparison.png", dpi=150)
    plt.close()
    print("Saved comparison to examples/figures/unified_api_comparison.png")


def demo_custom_config():
    """Demonstrate custom model configurations."""
    print("\n=== Custom Model Configuration ===")

    # Load events
    events = load_sample_events()

    # Create custom configuration
    custom_config = models.ModelConfig(
        in_channels=5,
        out_channels=1,
        base_channels=128,  # More channels for better quality
        num_bins=10,  # More time bins
        use_gpu=True,
    )

    # Create model with custom config
    model = models.E2VID(config=custom_config)
    print(f"Created model with custom config: {custom_config}")

    # Reconstruct
    frame = model.reconstruct(events)

    # Compare with default config
    default_model = models.E2VID()
    default_frame = default_model.reconstruct(events)

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.imshow(default_frame, cmap="gray")
    ax1.set_title("Default Configuration")
    ax1.axis("off")

    ax2.imshow(frame, cmap="gray")
    ax2.set_title("Custom Configuration\n(more channels & time bins)")
    ax2.axis("off")

    plt.tight_layout()
    plt.savefig("examples/figures/unified_api_custom_config.png", dpi=150)
    plt.close()


def demo_temporal_models():
    """Demonstrate temporal reconstruction models."""
    print("\n=== Temporal Models (E2VID+ and FireNet+) ===")

    # Load events
    events = load_sample_events()

    # Create temporal models
    e2vid_plus = models.E2VIDPlus()
    firenet_plus = models.FireNetPlus()

    # Reconstruct multiple frames
    num_frames = 5
    print(f"Reconstructing {num_frames} frames with temporal models...")

    frames_e2vid = e2vid_plus.reconstruct(events, num_frames=num_frames)
    frames_firenet = firenet_plus.reconstruct(events, num_frames=num_frames)

    # Display results
    fig, axes = plt.subplots(2, num_frames, figsize=(15, 6))

    for i in range(num_frames):
        axes[0, i].imshow(frames_e2vid[i], cmap="gray")
        axes[0, i].set_title(f"E2VID+ Frame {i+1}")
        axes[0, i].axis("off")

        axes[1, i].imshow(frames_firenet[i], cmap="gray")
        axes[1, i].set_title(f"FireNet+ Frame {i+1}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig("examples/figures/unified_api_temporal.png", dpi=150)
    plt.close()
    print("Saved temporal reconstruction to examples/figures/unified_api_temporal.png")


def demo_model_utilities():
    """Demonstrate model utility functions."""
    print("\n=== Model Utilities ===")

    # List available models
    available_models = models.list_models()
    print(f"Available models: {available_models}")

    # Get model information
    for model_name in available_models[:3]:  # Show first 3
        info = models.utils.get_model_info(model_name)
        print(f"\n{model_name}:")
        print(f"  - Name: {info['name']}")
        print(f"  - Description: {info['description']}")
        print(f"  - Size: {info['size_mb']} MB")
        print(f"  - Architecture: {info['architecture']}")

    # Check model paths
    model_name = "e2vid_unet"
    path = models.get_model_path(model_name)
    if path:
        print(f"\nModel '{model_name}' cached at: {path}")
    else:
        print(f"\nModel '{model_name}' not cached")

    # Pre-defined configurations
    print("\n\nAvailable configurations:")
    for name in ["default", "high_res", "fast", "temporal"]:
        config = models.config.get_config(name)
        print(f"  {name}: base_channels={config.base_channels}, num_bins={config.num_bins}")


def main():
    """Run all demonstrations."""
    # Create output directory
    Path("examples/figures").mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("evlib Unified Models API Demonstration")
    print("=" * 60)

    # Run demos
    demo_basic_usage()
    demo_different_models()
    demo_custom_config()
    demo_temporal_models()
    demo_model_utilities()

    print("\n" + "=" * 60)
    print("Demo completed! Check examples/figures/ for output images.")
    print("=" * 60)


if __name__ == "__main__":
    main()
