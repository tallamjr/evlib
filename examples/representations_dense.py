"""Representations to dense tensor demo.

Loads the tracked slider_depth recording, builds two tonic-validated event
representations as long-format Polars DataFrames, then densifies each into a
model-ready numpy tensor. Finally it saves a small figure summarising one
time bin of each representation.

Run headless with:

    MPLBACKEND=Agg .venv/bin/python examples/representations_dense.py
"""

import matplotlib

# Force a non-interactive backend so the script runs headless (no display).
matplotlib.use("Agg")

import os

import matplotlib.pyplot as plt

import evlib
import evlib.representations as evr

# slider_depth is a DAVIS240 recording: 240 x 180 sensor.
DATA_PATH = "data/slider_depth/events.txt"
WIDTH = 240
HEIGHT = 180
N_TIME_BINS = 5
# Saved next to this script under examples/ (examples/*.png is gitignored).
OUTPUT_FIGURE = os.path.join(os.path.dirname(__file__), "representations_dense.png")


def main() -> None:
    # Load events as a Polars LazyFrame (columns: x, y, t, polarity).
    events = evlib.load_events(DATA_PATH)

    # --- Voxel grid (Zhu et al. 2019 event volume, full bilinear interpolation).
    # create_voxel_grid returns a long-format frame: one row per non-empty
    # (x, y, time_bin) cell with the summed signed contribution.
    voxel_df = evr.create_voxel_grid(
        events, height=HEIGHT, width=WIDTH, n_time_bins=N_TIME_BINS
    )

    # densify_voxel_grid scatters that into the dense, model-ready tensor of
    # shape (n_time_bins, 1, H, W).
    voxel = evr.densify_voxel_grid(voxel_df, N_TIME_BINS, HEIGHT, WIDTH)
    print(f"Voxel grid long rows: {voxel_df.height}")
    print(f"Dense voxel grid shape: {voxel.shape}")  # (5, 1, 180, 240)

    # --- Event frame (tonic to_frame semantics: equal-width time bins, per
    # polarity counts). Long format columns: [time_bin, polarity, y, x, count].
    frame_df = evr.create_event_frame(
        events, height=HEIGHT, width=WIDTH, n_time_bins=N_TIME_BINS
    )

    # Two polarity channels (0 negative, 1 positive) -> (n_time_bins, 2, H, W).
    frame = evr.densify_event_frame(frame_df, N_TIME_BINS, 2, HEIGHT, WIDTH)
    print(f"Event frame long rows: {frame_df.height}")
    print(f"Dense event frame shape: {frame.shape}")  # (5, 2, 180, 240)

    # --- Save a small figure: first time bin of the voxel grid and of each
    # event-frame polarity channel.
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(voxel[0, 0], cmap="seismic")
    axes[0].set_title("Voxel grid, bin 0\n(signed contribution)")

    axes[1].imshow(frame[0, 1], cmap="hot")
    axes[1].set_title("Event frame, bin 0\n(positive count)")

    axes[2].imshow(frame[0, 0], cmap="hot")
    axes[2].set_title("Event frame, bin 0\n(negative count)")

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle("evlib representations -> dense tensors (slider_depth)")
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURE, dpi=100)
    plt.close(fig)
    print(f"Saved figure to {OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()
