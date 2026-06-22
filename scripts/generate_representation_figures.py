"""Generate the README/docs event-representation figures.

Two figure types are produced for each input sequence:

1. A transformation figure (``*_sidebyside.png`` / ``*_pedestrians.png``): the
   raw event stream on the left, and the output of
   ``evlib.representations.create_stacked_histogram`` for the same 250 ms window
   on the right, rendered as its five 50 ms temporal bins.

2. A representation gallery (``representation_gallery.png``): the same window
   passed through four different evlib representations side by side, so readers
   can see how event frames, voxel grids, time surfaces and stacked histograms
   each encode the same events.

The 80_balls EVT2 sample is tracked in the repository, so its figures are fully
reproducible. The pedestrians sequence is a gitignored large sample; its figures
are only generated when the file is present locally.

Usage:
    python scripts/generate_representation_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# House style: Tahoma for all figure text.
matplotlib.rcParams["font.family"] = "Tahoma"

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch

import evlib
import evlib.representations as evr

BINS = 5
WINDOW_DURATION_MS = 50.0
SPAN_US = int(BINS * WINDOW_DURATION_MS * 1000)  # total temporal span: 250 ms

# Diverging blue -> dark -> red map on a near-black background. Sparse event
# data reads far better on a dark canvas than on white.
POLARITY_CMAP = LinearSegmentedColormap.from_list(
    "evlib_polarity", ["#3a8bff", "#0b0b12", "#ff4d4d"]
)

# Perceptual "rainbow" used to encode event time within the window: dark blue is
# the start of the window (old), red/yellow is the end (recent). Moving objects
# therefore leave a colour-graded motion trail.
TIME_CMAP = matplotlib.colormaps["turbo"]

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "docs" / "images"


def _robust_vmax(array: np.ndarray, percentile: float = 98.0) -> float:
    """Percentile of the non-zero magnitudes, so a few hot pixels do not crush
    the visible contrast of the thin moving edges."""
    nonzero = np.abs(array[array != 0])
    return float(np.percentile(nonzero, percentile)) if nonzero.size else 1.0


def _relative_microseconds(events: pl.DataFrame) -> pl.DataFrame:
    return events.with_columns(
        (
            pl.col("t").dt.total_microseconds()
            - pl.col("t").dt.total_microseconds().min()
        ).alias("us")
    )


def _densest_window(events: pl.DataFrame) -> pl.DataFrame:
    """Return the events of the busiest 250 ms window, for a lively figure."""
    with_us = _relative_microseconds(events)
    busiest = (
        with_us.with_columns((pl.col("us") // SPAN_US).alias("window"))
        .group_by("window")
        .len()
        .sort("len", descending=True)
    )
    window_index = int(busiest["window"][0])
    lo, hi = window_index * SPAN_US, (window_index + 1) * SPAN_US
    return with_us.filter((pl.col("us") >= lo) & (pl.col("us") < hi)).drop("us")


def _signed_scatter(
    events: pl.DataFrame, height: int, width: int, count_col: str | None = None
) -> np.ndarray:
    """Scatter events into a signed (red = +1, blue = -1) frame."""
    frame = np.zeros((height, width), np.float64)
    sign = np.where(events["polarity"].to_numpy() > 0, 1.0, -1.0)
    weight = sign if count_col is None else events[count_col].to_numpy() * sign
    np.add.at(frame, (events["y"].to_numpy(), events["x"].to_numpy()), weight)
    return frame


# --------------------------------------------------------------------------- #
# Figure 1: raw stream -> stacked histogram temporal bins
# --------------------------------------------------------------------------- #
def render_transformation_figure(
    events_path: str, scene: str, title: str, output_name: str
) -> None:
    events = evlib.load_events(events_path).collect(engine="streaming")
    width = int(events["x"].max()) + 1
    height = int(events["y"].max()) + 1
    window = _densest_window(events)

    accumulation = _signed_scatter(window, height, width)

    histogram = evr.create_stacked_histogram(
        window,
        height=height,
        width=width,
        bins=BINS,
        window_duration_ms=WINDOW_DURATION_MS,
    )
    dense = np.zeros((BINS, height, width), np.float32)
    for row in histogram.iter_rows(named=True):
        sign = 1 if row["polarity"] == 1 else -1
        dense[row["time_bin"], row["y"], row["x"]] += row["count"] * sign

    fig = plt.figure(figsize=(13.5, 3.6), dpi=160)
    grid = fig.add_gridspec(1, 7, width_ratios=[2.4, 0.28, 1, 1, 1, 1, 1], wspace=0.07)

    raw_ax = fig.add_subplot(grid[0, 0])
    raw_vmax = _robust_vmax(accumulation)
    raw_ax.imshow(
        np.clip(accumulation, -raw_vmax, raw_vmax),
        cmap=POLARITY_CMAP,
        vmin=-raw_vmax,
        vmax=raw_vmax,
        interpolation="nearest",
    )
    raw_ax.set_title(
        f"Raw events  ({len(window):,} events, {int(BINS * WINDOW_DURATION_MS)}ms)",
        fontsize=9,
    )
    raw_ax.set_xticks([])
    raw_ax.set_yticks([])
    raw_ax.legend(
        handles=[
            Patch(color="#ff4d4d", label="+1"),
            Patch(color="#3a8bff", label="-1"),
        ],
        loc="upper right",
        fontsize=7,
        framealpha=0.9,
        title="polarity",
        title_fontsize=7,
        labelcolor="white",
        facecolor="#222",
        edgecolor="#555",
    )

    arrow_ax = fig.add_subplot(grid[0, 1])
    arrow_ax.axis("off")
    arrow_ax.annotate(
        "",
        xy=(1, 0.5),
        xytext=(0, 0.5),
        arrowprops=dict(arrowstyle="-|>", lw=1.7, color="#444"),
    )
    arrow_ax.text(
        0.5,
        0.60,
        "stacked\nhistogram",
        ha="center",
        va="bottom",
        fontsize=7,
        color="#444",
    )

    hist_vmax = _robust_vmax(dense)
    for bin_index in range(BINS):
        ax = fig.add_subplot(grid[0, 2 + bin_index])
        ax.imshow(
            np.clip(dense[bin_index], -hist_vmax, hist_vmax),
            cmap=POLARITY_CMAP,
            vmin=-hist_vmax,
            vmax=hist_vmax,
            interpolation="nearest",
        )
        start = int(bin_index * WINDOW_DURATION_MS)
        end = int((bin_index + 1) * WINDOW_DURATION_MS)
        ax.set_title(f"t-bin {bin_index}  ({start}-{end}ms)", fontsize=7.5)
        ax.set_xticks([])
        ax.set_yticks([])
        if bin_index == 2:
            ax.set_xlabel(
                f"create_stacked_histogram(bins={BINS}, window_duration_ms={int(WINDOW_DURATION_MS)})"
                f"  ->  {BINS} bins x 2 polarity = {BINS * 2} channels",
                fontsize=8.5,
            )

    fig.suptitle(title, fontsize=10.5, y=1.04)
    output_path = OUTPUT_DIR / output_name
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(
        f"wrote {output_path.relative_to(REPO_ROOT)}  ({width}x{height}, {len(window):,} events)"
    )


# --------------------------------------------------------------------------- #
# Figure 2: gallery of representations on one window
# --------------------------------------------------------------------------- #
def _accumulate(height: int, width: int, ys, xs, weights) -> np.ndarray:
    out = np.zeros((height, width), np.float64)
    np.add.at(out, (ys, xs), weights)
    return out


def _time_coded_rgb(
    activity: np.ndarray, time01: np.ndarray | None, gamma: float = 0.6
) -> np.ndarray:
    """Map per-pixel time (`time01` in [0, 1]) to the TIME_CMAP hue and per-pixel
    activity to brightness, on a black background. When ``time01`` is None the
    panel is rendered achromatically (grayscale): the representation carries no
    time axis, so it gets no colour."""
    vmax = _robust_vmax(activity)
    brightness = (
        np.clip(activity / vmax, 0.0, 1.0) ** gamma if vmax else np.zeros_like(activity)
    )
    if time01 is None:
        return np.repeat(brightness[..., None], 3, axis=2)
    rgb = TIME_CMAP(np.clip(time01, 0.0, 1.0))[..., :3]
    return rgb * brightness[..., None]


def _time_coded_panels(window: pl.DataFrame, height: int, width: int):
    """Return (title, rgb_image) per panel, each colouring pixels by the event
    time they encode so the temporal resolution of each representation is
    visible. Brightness is event density; hue is time (old -> recent)."""
    ys = window["y"].to_numpy()
    xs = window["x"].to_numpy()
    absolute_us = (
        window.select(pl.col("t").dt.total_microseconds()).to_series().to_numpy()
    )
    start_t = float(absolute_us.min())
    span = float(absolute_us.max() - absolute_us.min()) or 1.0
    t01 = (absolute_us - start_t) / span

    # Raw events: full microsecond time, mean event time per pixel.
    counts = _accumulate(height, width, ys, xs, np.ones_like(t01))
    time_sum = _accumulate(height, width, ys, xs, t01)
    raw_time = np.divide(
        time_sum, counts, out=np.zeros_like(time_sum), where=counts > 0
    )
    raw_rgb = _time_coded_rgb(counts, raw_time)

    # Event frame: counts only, no time axis -> grayscale.
    event_frame = evr.create_event_frame(
        window, height=height, width=width, n_time_bins=1
    )
    ef_counts = _accumulate(
        height,
        width,
        event_frame["y"].to_numpy(),
        event_frame["x"].to_numpy(),
        event_frame["count"].to_numpy(),
    )
    ef_rgb = _time_coded_rgb(ef_counts, None)

    # Voxel grid: time quantised into BINS bilinear bins -> banded colour.
    voxel = evr.create_voxel_grid(window, height=height, width=width, n_time_bins=BINS)
    vy, vx = voxel["y"].to_numpy(), voxel["x"].to_numpy()
    weight = np.abs(voxel["contribution"].to_numpy())
    vox_act = _accumulate(height, width, vy, vx, weight)
    vox_tw = _accumulate(height, width, vy, vx, voxel["time_bin"].to_numpy() * weight)
    vox_time = np.divide(
        vox_tw, vox_act, out=np.zeros_like(vox_tw), where=vox_act > 0
    ) / max(BINS - 1, 1)
    vox_rgb = _time_coded_rgb(vox_act, vox_time)

    # Time surface: continuous recency in [0, 1] is itself the time encoding.
    surface_df = evr.create_time_surface(
        window, height=height, width=width, dt=span, tau=span / 3.0
    )
    surface = evr.densify_time_surface(
        surface_df,
        n_slices=1,
        n_polarities=2,
        height=height,
        width=width,
        dt=span,
        tau=span / 3.0,
        start_t=start_t,
    )
    recency = surface[0].max(axis=0)
    ts_rgb = _time_coded_rgb(recency, recency)

    # Stacked histogram: time quantised into BINS bins -> banded colour.
    stacked = evr.create_stacked_histogram(
        window,
        height=height,
        width=width,
        bins=BINS,
        window_duration_ms=WINDOW_DURATION_MS,
    )
    sy, sx = stacked["y"].to_numpy(), stacked["x"].to_numpy()
    sc = stacked["count"].to_numpy().astype(np.float64)
    sh_act = _accumulate(height, width, sy, sx, sc)
    sh_tw = _accumulate(height, width, sy, sx, stacked["time_bin"].to_numpy() * sc)
    sh_time = np.divide(
        sh_tw, sh_act, out=np.zeros_like(sh_tw), where=sh_act > 0
    ) / max(BINS - 1, 1)
    sh_rgb = _time_coded_rgb(sh_act, sh_time)

    return [
        ("Raw events\n(full microsecond time)", raw_rgb),
        ("create_event_frame\n(grayscale: no time axis)", ef_rgb),
        (f"create_voxel_grid\n({BINS} bilinear time bins)", vox_rgb),
        ("create_time_surface\n(continuous recency)", ts_rgb),
        (f"create_stacked_histogram\n({BINS} time bins)", sh_rgb),
    ]


def render_gallery(events_path: str, scene: str, output_name: str) -> None:
    events = evlib.load_events(events_path).collect(engine="streaming")
    width = int(events["x"].max()) + 1
    height = int(events["y"].max()) + 1
    window = _densest_window(events)

    panels = _time_coded_panels(window, height, width)

    fig = plt.figure(figsize=(15, 4.0), dpi=160)
    grid = fig.add_gridspec(
        2, len(panels), height_ratios=[1.0, 0.05], hspace=0.32, wspace=0.05
    )
    for col, (title, rgb) in enumerate(panels):
        ax = fig.add_subplot(grid[0, col])
        ax.imshow(rgb, interpolation="nearest")
        ax.set_title(title, fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])

    cax = fig.add_subplot(grid[1, :])
    bar = fig.colorbar(
        cm.ScalarMappable(norm=Normalize(0.0, 1.0), cmap=TIME_CMAP),
        cax=cax,
        orientation="horizontal",
    )
    bar.set_ticks([0.0, 1.0])
    bar.set_ticklabels(["window start (old)", "window end (recent)"], fontsize=7.5)
    bar.set_label(
        "Hue = event time within the 250ms window; brightness = event density. "
        "Moving objects leave colour-graded motion trails; the event frame is grayscale because it keeps no time.",
        fontsize=7.5,
    )

    fig.suptitle(
        f"evlib representations, time encoded as colour ({scene}, {len(window):,} events, 250ms)",
        fontsize=11,
        y=0.99,
    )
    output_path = OUTPUT_DIR / output_name
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(
        f"wrote {output_path.relative_to(REPO_ROOT)}  ({width}x{height}, {len(window):,} events)"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    balls = "data/prophesee/samples/evt2/80_balls.raw"
    render_transformation_figure(
        balls,
        "80_balls",
        "evlib (80_balls): raw event stream  ->  RVT-style stacked-histogram representation",
        "representations_sidebyside.png",
    )

    # The representation gallery reads most clearly on a scene with recognisable
    # moving objects, so prefer the pedestrians sequence when it is available
    # locally and fall back to the tracked 80_balls sample otherwise.
    pedestrians = (
        REPO_ROOT / "data" / "prophesee" / "samples" / "evt3" / "pedestrians.raw"
    )
    if pedestrians.exists():
        render_transformation_figure(
            str(pedestrians),
            "pedestrians",
            "evlib (pedestrians): raw event stream  ->  stacked-histogram representation",
            "representations_pedestrians.png",
        )
        render_gallery(str(pedestrians), "pedestrians", "representation_gallery.png")
    else:
        print(
            "pedestrians sample absent: rendering the gallery from the tracked 80_balls sample"
        )
        render_gallery(balls, "80_balls", "representation_gallery.png")


if __name__ == "__main__":
    main()
