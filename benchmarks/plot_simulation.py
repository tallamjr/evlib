"""Render the event simulator benchmark figures from the committed JSON.

Reads benchmarks/out/simulation_bench_results.json (evlib rows) and
simulation_reference_vid2e.json (esim_torch and esim_py rows); no network, no GPU.
Writes benchmarks/out/simulation_throughput.png, simulation_breakdown.png,
simulation_cuda_levers.png and the transparent SVG twins of the throughput
figure under docs/images/ (light and -dark). Tahoma, 300 dpi.

    .venv/bin/python -m benchmarks.plot_simulation
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "out"
DOCS_IMAGES = Path(__file__).resolve().parents[1] / "docs" / "images"

CUDA, CUDA_LIGHT = "#2a78d6", "#86b6ef"
CPU, CPU_LIGHT = "#1baf7a", "#86d9b8"
GREY = "#bdbdbd"
# Stage colours: the dataviz reference palette, slots 1 to 6.
STAGE_COLOURS = {
    "upload": "#2a78d6",
    "kernel": "#eb6834",
    "sort": "#1baf7a",
    "download": "#eda100",
    "copy-out": "#e87ba4",
    "DataFrame": "#008300",
}
# (ink, muted, axes surface, grid) per mode; the SVG modes have no surface.
INK = {
    "light": ("#0b0b0b", "#52514e", "#f7f7f7", "white"),
    "light-svg": ("#0b0b0b", "#52514e", "none", "#e1e0d9"),
    "dark-svg": ("#ffffff", "#c3c2b7", "none", "#3a3a38"),
}
plt.rcParams["font.family"] = "Tahoma"
plt.rcParams["svg.fonttype"] = "none"
# Stable ids and no date so a regenerated SVG is byte-identical.
plt.rcParams["svg.hashsalt"] = "evlib"


def load():
    results = json.loads((OUT_DIR / "simulation_bench_results.json").read_text())
    reference = json.loads((OUT_DIR / "simulation_reference_vid2e.json").read_text())
    return results, reference


def pick(rows, **match):
    hits = [r for r in rows if all(r.get(k) == v for k, v in match.items())]
    if len(hits) != 1:
        raise KeyError(f"{len(hits)} rows match {match}")
    return hits[0]


def style_axis(ax, surface: str, muted: str, grid: str):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(muted)
    ax.tick_params(colors=muted, labelcolor=muted)
    ax.set_facecolor(surface)
    ax.grid(axis="x", color=grid, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)


def throughput_rows(results, reference, resolution: str):
    """(label, M events/s, colour) rows for one raw30 resolution."""
    ev = [
        r
        for r in results["rows"]
        if r["input"] == "raw30" and r["resolution"] == resolution
    ]
    ref = [
        r
        for r in reference["rows"]
        if r["input"] == "raw30" and r["resolution"] == resolution
    ]
    rows = []
    for batch, tag in ((32, "batch 32"), ("whole", "whole stack")):
        cuda = pick(ev, backend="cuda", batch=batch)
        rows.append(
            (
                f"evlib CUDA kernel ceiling ({tag})",
                cuda["device_events_per_s"],
                CUDA_LIGHT,
            )
        )
    for batch, tag in ((32, "batch 32"), ("whole", "whole stack")):
        cuda = pick(ev, backend="cuda", batch=batch)
        rows.append(
            (f"evlib CUDA sorted DataFrame ({tag})", cuda["events_per_s"], CUDA)
        )
    for batch, tag in ((32, "batch 32"), ("whole", "whole stack")):
        cpu = pick(ev, backend="cpu", batch=batch)
        rows.append(
            (
                f"evlib CPU kernel, 48 threads ({tag})",
                cpu["kernel_events_per_s"],
                CPU_LIGHT,
            )
        )
    for batch, tag in ((32, "batch 32"), ("whole", "whole stack")):
        cpu = pick(ev, backend="cpu", batch=batch)
        rows.append(
            (
                f"evlib CPU sorted DataFrame, 48 threads ({tag})",
                cpu["events_per_s"],
                CPU,
            )
        )
    for batch in (32, 1):
        k = pick(ref, system="esim_torch kernel-only", batch=batch)
        rows.append(
            (f"esim_torch kernel-only (batch {batch})", k["events_per_s"], GREY)
        )
    rows.append(
        (
            "esim_torch as shipped (PNG to npz)",
            pick(ref, system="esim_torch as shipped")["events_per_s"],
            GREY,
        )
    )
    rows.append(
        ("esim_py C++, 1 thread", pick(ref, system="esim_py")["events_per_s"], GREY)
    )
    return [(label, value / 1e6, colour) for label, value, colour in rows]


def plot_throughput(results, reference, mode: str, out: Path, transparent: bool):
    ink, muted, surface, grid = INK[mode]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8.6), sharex=True)
    for ax, resolution in zip(axes, ("320x320", "640x480")):
        rows = throughput_rows(results, reference, resolution)
        y = np.arange(len(rows))
        values = [r[1] for r in rows]
        ax.barh(y, values, color=[r[2] for r in rows], height=0.66, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows], fontsize=8.5, color=ink)
        ax.invert_yaxis()
        for yi, v in zip(y, values):
            tag = f"{v:,.0f} M" if v >= 100 else f"{v:.1f} M"
            ax.text(v + 15, yi, tag, va="center", ha="left", fontsize=8, color=ink)
        events = pick(
            results["rows"],
            input="raw30",
            resolution=resolution,
            backend="cuda",
            batch=32,
        )["events"]
        ax.set_title(
            f"raw30 {resolution}, 900 frames, {events / 1e6:.1f} M events",
            loc="left",
            fontsize=10,
            color=ink,
        )
        style_axis(ax, surface, muted, grid)
    axes[1].set_xlabel("million events per second (higher is better)", color=ink)
    axes[1].set_xlim(
        0, max(r[1] for r in throughput_rows(results, reference, "640x480")) * 1.15
    )
    fig.suptitle(
        "Event simulator throughput on an RTX 4090 host: evlib against rpg_vid2e",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
        color=ink,
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c)
        for c in (CUDA_LIGHT, CUDA, CPU_LIGHT, CPU, GREY)
    ]
    labels = [
        "CUDA kernel ceiling (device resident)",
        "CUDA sorted DataFrame",
        "CPU kernel (unsorted arrays)",
        "CPU sorted DataFrame",
        "rpg_vid2e reference",
    ]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        fontsize=8,
        frameon=False,
        labelcolor=ink,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(out, dpi=300, transparent=transparent, metadata={"Date": None})
    plt.close(fig)
    if out.suffix == ".svg":
        # Matplotlib leaves trailing spaces in path data; the pre-commit hook strips them.
        lines = [line.rstrip() for line in out.read_text().splitlines()]
        out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")


def breakdown_row(r):
    """Stage seconds for one row at batch 32; CUDA stages come from the sorted C ABI run."""
    if r["backend"] == "cpu":
        return {"kernel": r["kernel_s"], "sort": r["sort_s"], "DataFrame": r["df_s"]}
    s = r["stages_sorted"]
    inside = (
        s["stage_s"] + s["upload_s"] + s["device_s"] + s["sort_s"] + s["download_s"]
    )
    return {
        "upload": s["stage_s"] + s["upload_s"],
        "kernel": s["device_s"],
        "sort": s["sort_s"],
        "download": s["download_s"],
        "copy-out": r["sorted_s"] - inside,
        "DataFrame": r["df_s"],
    }


def plot_breakdown(results, out: Path):
    ink, muted, surface, grid = INK["light"]
    rows = []
    for backend in ("cuda", "cpu"):
        for input_kind in ("raw30", "upsampled"):
            for resolution in ("320x320", "640x480"):
                r = pick(
                    results["rows"],
                    backend=backend,
                    input=input_kind,
                    resolution=resolution,
                    batch=32,
                )
                frames = "900 frames" if input_kind == "raw30" else "8,000 frames"
                rows.append(
                    (
                        f"{backend.upper()} {input_kind} {resolution} ({frames})",
                        breakdown_row(r),
                        r["wall_s"],
                    )
                )
    fig, ax = plt.subplots(figsize=(10, 4.6))
    y = np.arange(len(rows))
    left = np.zeros(len(rows))
    for stage, colour in STAGE_COLOURS.items():
        share = np.array(
            [100 * parts.get(stage, 0.0) / wall for _, parts, wall in rows]
        )
        ax.barh(
            y,
            share,
            left=left,
            color=colour,
            height=0.66,
            zorder=3,
            label=stage,
            edgecolor="white",
            linewidth=1,
        )
        left += share
    for yi, (_, _, wall) in zip(y, rows):
        ax.text(
            101,
            yi,
            f"{wall * 1000:,.0f} ms",
            va="center",
            ha="left",
            fontsize=8,
            color=ink,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5, color=ink)
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.set_xlabel(
        "share of the sorted DataFrame wall time at batch 32 (%), wall in ms at the bar end",
        color=ink,
    )
    ax.set_title(
        "Where the time goes: evlib simulator stages per backend and input (RTX 4090 host)",
        loc="left",
        fontsize=11,
        fontweight="bold",
        color=ink,
    )
    style_axis(ax, surface, muted, grid)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.3),
        ncol=6,
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_levers(results, reference, out: Path):
    ink, muted, surface, grid = INK["light"]
    steps = [
        ("baseline", "baseline\nf5b9c72"),
        ("lever1", "lever 1\nu8 upload,\ndevice LUT"),
        ("lever2", "lever 2\npinned, async,\nretained buffers"),
        ("lever3", "lever 3\ndevice sort"),
        ("final", "final\nparallel copy-out,\nin-place t"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    width = 0.38
    for ax, resolution in zip(axes, ("320x320", "640x480")):
        x = np.arange(len(steps))
        rows = [
            pick(results["cuda_levers"], step=s, resolution=resolution, batch=32)
            for s, _ in steps
        ]
        sorted_df = [r["events_per_s"] / 1e6 for r in rows]
        kernel = [r["kernel_events_per_s"] / 1e6 for r in rows]
        ax.bar(
            x - width / 2,
            sorted_df,
            width,
            color=CUDA,
            zorder=3,
            label="sorted DataFrame (public path)",
        )
        ax.bar(
            x + width / 2,
            kernel,
            width,
            color=CUDA_LIGHT,
            zorder=3,
            label="unsorted arrays on the host",
        )
        for xi, v in zip(x, sorted_df):
            ax.text(
                xi - width / 2,
                v + 8,
                f"{v:,.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=ink,
            )
        for xi, v in zip(x, kernel):
            ax.text(
                xi + width / 2,
                v + 8,
                f"{v:,.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=muted,
            )
        torch32 = pick(
            reference["rows"],
            system="esim_torch kernel-only",
            input="raw30",
            resolution=resolution,
            batch=32,
        )
        ax.axhline(300, color=muted, linewidth=1, zorder=2)
        ax.text(
            -0.45,
            306,
            "spec bar 300 M",
            ha="left",
            va="bottom",
            fontsize=7.5,
            color=muted,
        )
        ax.axhline(torch32["events_per_s"] / 1e6, color=muted, linewidth=1, zorder=2)
        ax.text(
            len(steps) - 0.5,
            torch32["events_per_s"] / 1e6 + 6,
            f"esim_torch kernel-only batch 32: {torch32['events_per_s'] / 1e6:,.0f} M",
            ha="right",
            va="bottom",
            fontsize=7.5,
            color=muted,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in steps], fontsize=8, color=ink)
        ax.set_title(
            f"raw30 {resolution}, batch 32", loc="left", fontsize=10, color=ink
        )
        style_axis(ax, surface, muted, grid)
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color=grid, linewidth=1.0, zorder=0)
    axes[0].set_ylabel("million events per second", color=ink)
    axes[0].set_ylim(0, 720)
    fig.suptitle(
        "CUDA path after each Task 10 lever (RTX 4090, one commit per lever)",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
        color=ink,
    )
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> int:
    results, reference = load()
    plot_throughput(
        results,
        reference,
        "light",
        OUT_DIR / "simulation_throughput.png",
        transparent=False,
    )
    plot_throughput(
        results,
        reference,
        "light-svg",
        DOCS_IMAGES / "simulation_throughput.svg",
        transparent=True,
    )
    plot_throughput(
        results,
        reference,
        "dark-svg",
        DOCS_IMAGES / "simulation_throughput-dark.svg",
        transparent=True,
    )
    plot_breakdown(results, OUT_DIR / "simulation_breakdown.png")
    plot_levers(results, reference, OUT_DIR / "simulation_cuda_levers.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
