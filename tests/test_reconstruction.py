#!/usr/bin/env python3
"""
Test script for event reconstruction functionality in evlib

This script tests the event-to-video reconstruction functions to diagnose
the issue in the evlib_event_reconstruction.ipynb notebook.
"""
import os
import time

import matplotlib.pyplot as plt
import numpy as np

import evlib

# Set data path to slider_depth dataset
data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/slider_depth/")

test_path = os.path.dirname(os.path.realpath(__file__))


def load_events(filepath):
    """Load events using evlib's formats module"""
    print(f"Loading events from {filepath} using evlib...")
    xs, ys, ts, ps = evlib.formats.load_events(filepath)
    print(f"Loaded {len(xs)} events")
    return xs, ys, ts, ps


def test_single_frame_reconstruction():
    """Test single frame reconstruction function"""
    print("\nTesting single frame reconstruction...")

    # Load the dataset
    events_path = os.path.join(data_path, "events.txt")
    xs, ys, ts, ps = load_events(events_path)

    # Determine sensor resolution
    sensor_width = int(xs.max() + 1)
    sensor_height = int(ys.max() + 1)
    print(f"Sensor resolution: {sensor_width} x {sensor_height}")

    # Select a small subset of events
    subset_size = min(10000, len(xs))
    print(f"Using subset of {subset_size} events")

    # Ensure we use a sorted subset
    indices = np.arange(subset_size)
    subset_xs = xs[indices].copy()
    subset_ys = ys[indices].copy()
    subset_ts = ts[indices].copy()
    subset_ps = ps[indices].copy()

    # Create debug flag to print more details about event types
    debug_dtypes = True
    if debug_dtypes:
        print(f"xs dtype: {subset_xs.dtype}")
        print(f"ys dtype: {subset_ys.dtype}")
        print(f"ts dtype: {subset_ts.dtype}")
        print(f"ps dtype: {subset_ps.dtype}")

    # Test with different num_bins
    for num_bins in [3, 5]:
        try:
            print(f"\nTrying with num_bins={num_bins}")
            start_time = time.time()
            reconstructed_frame = evlib.processing.events_to_video(
                subset_xs,
                subset_ys,
                subset_ts,
                subset_ps,
                height=sensor_height,
                width=sensor_width,
                num_bins=num_bins,
            )
            end_time = time.time()

            # Print shape and type
            print(
                f"Success! Returned frame with shape {reconstructed_frame.shape}, dtype {reconstructed_frame.dtype}"
            )
            print(f"Reconstruction took {end_time - start_time:.3f} seconds")

            # Visualize result
            plt.figure(figsize=(10, 8))
            plt.imshow(reconstructed_frame[:, :, 0], cmap="gray")
            plt.title(f"Reconstructed Frame (num_bins={num_bins})")
            plt.colorbar()
            plt.savefig(f"{test_path}/reconstructed_frame_bins{num_bins}.png")
            print(f"Saved visualization to reconstructed_frame_bins{num_bins}.png")

        except Exception as e:
            print(f"Error: {e}")
            print(f"Traceback: {type(e).__name__}")


def custom_multi_frame_reconstruction(xs, ys, ts, ps, height, width, num_frames, num_bins):
    """
    Custom function to reconstruct multiple frames from events

    This is a workaround for the type mismatch issue in the built-in function.
    It uses the single-frame reconstruction function repeatedly.
    """
    # Get time range
    t_min, t_max = ts.min(), ts.max()
    time_step = (t_max - t_min) / num_frames

    # List to store frames
    frames = []

    for i in range(num_frames):
        # Define time window for this frame
        t_end = t_min + time_step * (i + 1)

        # Get events up to this time
        mask = ts <= t_end
        frame_xs = xs[mask]
        frame_ys = ys[mask]
        frame_ts = ts[mask]
        frame_ps = ps[mask]

        # Skip if no events
        if len(frame_xs) == 0:
            # Empty frame
            empty_frame = np.zeros((height, width, 1), dtype=np.float32)
            frames.append(empty_frame)
            continue

        # Process events using the working single-frame function
        frame = evlib.processing.events_to_video(
            frame_xs,
            frame_ys,
            frame_ts,
            frame_ps,
            height=height,
            width=width,
            num_bins=num_bins,
        )

        frames.append(frame)

    return frames


def test_multi_frame_reconstruction():
    """Test multi-frame reconstruction function"""
    print("\nTesting workaround for multi-frame reconstruction...")

    # Load the dataset
    events_path = os.path.join(data_path, "events.txt")
    xs, ys, ts, ps = load_events(events_path)

    # Determine sensor resolution
    sensor_width = int(xs.max() + 1)
    sensor_height = int(ys.max() + 1)

    # Create smaller subset for testing
    subset_size = min(50000, len(xs))
    indices = np.arange(subset_size)
    subset_xs = xs[indices].copy()
    subset_ys = ys[indices].copy()
    subset_ts = ts[indices].copy()
    subset_ps = ps[indices].copy()

    # Try with different parameters
    for num_frames in [3, 5]:
        for num_bins in [3, 5]:
            try:
                print(f"\nTrying with num_frames={num_frames}, num_bins={num_bins}")
                start_time = time.time()

                # Use custom implementation instead of evlib.processing.reconstruct_events_to_frames_py
                reconstructed_frames = custom_multi_frame_reconstruction(
                    subset_xs,
                    subset_ys,
                    subset_ts,
                    subset_ps,
                    height=sensor_height,
                    width=sensor_width,
                    num_frames=num_frames,
                    num_bins=num_bins,
                )

                end_time = time.time()

                print(f"Success! Reconstructed {len(reconstructed_frames)} frames")
                print(f"First frame shape: {reconstructed_frames[0].shape}")
                print(f"Reconstruction took {end_time - start_time:.3f} seconds")

                # Visualize first and last frames
                fig, axes = plt.subplots(1, num_frames, figsize=(15, 5))
                if num_frames == 1:
                    axes = [axes]

                for i in range(num_frames):
                    frame = reconstructed_frames[i]
                    if len(frame.shape) == 3 and frame.shape[2] == 1:
                        frame = frame[:, :, 0]
                    axes[i].imshow(frame, cmap="gray")
                    axes[i].set_title(f"Frame {i+1}")
                    axes[i].axis("off")

                plt.tight_layout()
                plt.savefig(f"{test_path}/multi_frame_recon_f{num_frames}_b{num_bins}.png")
                print(f"Saved visualization to multi_frame_recon_f{num_frames}_b{num_bins}.png")

            except Exception as e:
                print(f"Error: {e}")
                print(f"Traceback: {type(e).__name__}")


if __name__ == "__main__":
    test_single_frame_reconstruction()
    test_multi_frame_reconstruction()
