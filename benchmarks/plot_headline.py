"""Render the headline 'evlib vs RVT' chart from the committed RVT dataset results.

A focused head-to-head: evlib's two competitive backends (CUDA on the GPU, Rust on the CPU)
against the RVT torch reference on the same hardware tier. This deliberately omits evlib's own
Polars-GPU path (which is not competitive for this kernel and whose 1169s bar wrecks the scale);
the full five-backend chart lives in rvt_final_time.png. Reads benchmarks/out/rvt_final_results.json.

    python -m benchmarks.plot_headline
"""

from __future__ import annotations

import json
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "out"


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    results = json.loads((OUT_DIR / "rvt_final_results.json").read_text())
    totals = {b: sum(r["wall_s"] for r in runs) for b, runs in results.items()}

    # Two head-to-head pairs on matched hardware tiers (GPU, then CPU).
    rows = [
        ("evlib CUDA (GPU)", totals["evlib_cuda"], "#2a9d5c"),
        ("RVT torch (GPU)", totals["rvt_gpu"], "#bdbdbd"),
        ("evlib Rust (CPU)", totals["evlib_rust"], "#2a9d5c"),
        ("RVT torch (CPU)", totals["rvt_cpu"], "#bdbdbd"),
    ]
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colours = [r[2] for r in rows]

    plt.rcParams["font.family"] = "Tahoma"
    fig, ax = plt.subplots(figsize=(8, 3.4))
    y = np.arange(len(rows))
    ax.barh(y, vals, color=colours, zorder=3, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("total wall-clock over 18 gen4 sequences (s), lower is better")
    ax.set_title(
        "evlib vs RVT preprocessing (RTX 4090): faster on GPU and CPU, with matching output",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )
    # Win annotations on the evlib bars.
    wins = {
        0: f"{vals[1] / vals[0]:.2f}x vs RVT-GPU",
        2: f"{vals[3] / vals[2]:.2f}x vs RVT-CPU",
    }
    for yi, v in zip(y, vals):
        tag = f" {v:.0f}s"
        if yi in wins:
            tag += f"  ({wins[yi]})"
        ax.text(v, yi, tag, va="center", ha="left", fontsize=9)
    ax.set_xlim(0, max(vals) * 1.32)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("#f7f7f7")
    ax.grid(axis="x", color="white", linewidth=1.2, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = OUT_DIR / "rvt_headline.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
