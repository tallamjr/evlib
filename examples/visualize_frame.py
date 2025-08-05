#!/usr/bin/env python3
"""
Simple frame visualization script for E2VID reconstructed frames.
"""

import matplotlib.pyplot as plt
import numpy as np


def visualize_frame(frame, title="Reconstructed Frame"):
    """
    Visualize a reconstructed frame from E2VID.

    Args:
        frame: numpy array of shape (height, width)
        title: Title for the plot
    """
    plt.figure(figsize=(10, 6))

    # Main image
    plt.subplot(1, 2, 1)
    plt.imshow(frame, cmap="gray")
    plt.title(title)
    plt.colorbar(label="Intensity")
    plt.axis("off")

    # Histogram
    plt.subplot(1, 2, 2)
    plt.hist(frame.flatten(), bins=50, alpha=0.7, color="blue")
    plt.title("Pixel Intensity Distribution")
    plt.xlabel("Intensity")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def visualize_frame_enhanced(frame, title="Enhanced Reconstruction"):
    """
    Visualize frame with enhanced contrast and multiple colormaps.
    """
    # Normalize to [0, 1] for better contrast
    frame_norm = (frame - frame.min()) / (frame.max() - frame.min())

    plt.figure(figsize=(15, 5))

    # Original
    plt.subplot(1, 3, 1)
    plt.imshow(frame, cmap="gray")
    plt.title("Original")
    plt.colorbar()
    plt.axis("off")

    # Enhanced contrast
    plt.subplot(1, 3, 2)
    plt.imshow(frame_norm, cmap="gray")
    plt.title("Enhanced Contrast")
    plt.colorbar()
    plt.axis("off")

    # Color map for better detail
    plt.subplot(1, 3, 3)
    plt.imshow(frame_norm, cmap="viridis")
    plt.title("Viridis Colormap")
    plt.colorbar()
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# Example usage:
if __name__ == "__main__":
    import evlib
    import evlib.models

    # Load and reconstruct
    events = evlib.load_events("data/slider_depth/events.txt", t_start=0.0, t_end=0.1)
    model = evlib.models.E2VID()
    frame = model.reconstruct(events)

    # Visualize
    print("Basic visualization:")
    visualize_frame(frame)

    print("\nEnhanced visualization:")
    visualize_frame_enhanced(frame)
