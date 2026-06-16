"""Benchmark harness: evlib RVT-preprocessing pipeline vs the genuine RVT torch pipeline.

This module measures wall-clock time and peak resident memory for two pipelines that
turn the raw Gen4 validation h5 into the stacked-histogram event-representation h5:

1. evlib streaming pipeline (``evlib.rvt.process_sequence(engine="streaming")``).
2. RVT's own reference pipeline, reusing the genuine RVT modules under ``lib/RVT``
   (``H5Reader``, ``StackedHistogram``, ``downsample_ev_repr``, ``H5Writer``).

Both pipelines consume the same raw input and the same reference window-end grid, and
both outputs are asserted bit-identical to the committed reference output before any
timing is recorded.

Peak memory is measured by running each pipeline in a fresh subprocess and reading
``resource.getrusage(RUSAGE_CHILDREN).ru_maxrss`` after the child exits. tracemalloc
only sees Python allocations and would miss Polars and torch native buffers, so the
subprocess+rusage approach is used for the real peak.

Run the full benchmark and render the plots::

    .venv/bin/python -m benchmarks.bench_rvt_pipeline --repeats 3
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / "out"

# Reference grid + reference output live with the test fixtures.
sys.path.insert(0, str(ROOT))


def ru_maxrss_bytes(usage_value: int) -> int:
    """Normalise ``ru_maxrss`` to bytes for the current platform.

    macOS (darwin) reports ru_maxrss in bytes; Linux reports it in kibibytes.
    """
    if sys.platform == "darwin":
        return int(usage_value)
    # linux and most other unices report KiB
    return int(usage_value) * 1024


def summarize(times: Sequence[float]) -> Dict[str, float]:
    """Return min/median/max of a sequence of measurements."""
    values = list(times)
    if not values:
        raise ValueError("summarize requires at least one measurement")
    return {
        "min": float(min(values)),
        "median": float(statistics.median(values)),
        "max": float(max(values)),
    }


def time_and_memory(fn: Callable[[], object], repeats: int = 1) -> Dict[str, object]:
    """Time ``fn`` ``repeats`` times in-process and capture child peak RSS.

    Returns a dict with per-run wall-clock ``times`` (seconds) and the high-water
    ``peak_rss_bytes`` observed across ``RUSAGE_SELF`` and ``RUSAGE_CHILDREN`` after
    the calls. For pipelines that spawn subprocesses, prefer
    :func:`run_pipeline_subprocess`, which isolates each run's peak cleanly.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    times: List[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    self_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    child_usage = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    peak = max(ru_maxrss_bytes(self_usage), ru_maxrss_bytes(child_usage))
    return {"times": times, "peak_rss_bytes": peak}


@dataclass
class RunResult:
    """Per-run measurement for a single pipeline invocation in a subprocess."""

    wall_s: float
    peak_rss_bytes: int


@dataclass
class PipelineResult:
    """Aggregated measurements across repeats for one pipeline."""

    label: str
    note: str
    runs: List[RunResult] = field(default_factory=list)

    @property
    def times(self) -> List[float]:
        return [r.wall_s for r in self.runs]

    @property
    def peak_rss_bytes(self) -> int:
        return max(r.peak_rss_bytes for r in self.runs)

    def time_summary(self) -> Dict[str, float]:
        return summarize(self.times)

    @property
    def peak_gb(self) -> float:
        return self.peak_rss_bytes / (1024**3)


def run_pipeline_subprocess(
    child_args: Sequence[str], timeout: float = 1800.0
) -> RunResult:
    """Run one pipeline invocation as a child process and read its peak RSS.

    ``child_args`` are appended to ``python -m benchmarks.bench_rvt_pipeline``. The child
    is expected to print a single JSON line ``{"wall_s": <float>}`` on stdout reporting
    its own internal wall-clock for the pipeline body (excluding interpreter start-up).
    Peak RSS for the child is read via ``RUSAGE_CHILDREN`` deltas around the call.
    """
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    start = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "benchmarks.bench_rvt_pipeline", *child_args],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - start
    after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if proc.returncode != 0:
        raise RuntimeError(
            f"child {' '.join(child_args)} failed (rc={proc.returncode})\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    # ru_maxrss under RUSAGE_CHILDREN is the max over all terminated children, so the
    # delta is dominated by this child when runs are sequential. Use the absolute peak
    # of this child as reported after it exits.
    wall_s = elapsed
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and "wall_s" in line:
            try:
                payload = json.loads(line)
                wall_s = float(payload["wall_s"])
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
    peak = ru_maxrss_bytes(after) - ru_maxrss_bytes(before)
    if peak <= 0:
        # First child or non-increasing high-water mark: fall back to the absolute peak.
        peak = ru_maxrss_bytes(after)
    return RunResult(wall_s=wall_s, peak_rss_bytes=int(peak))


# --------------------------------------------------------------------------------------
# Pipeline bodies (run inside child subprocesses via the --child dispatch below).
# --------------------------------------------------------------------------------------


def _ref_paths():
    from tests.rvt_fixtures import raw_input_path, ref_repr_h5, ref_timestamps

    grid = np.load(str(ref_timestamps())).astype(np.int64)
    return raw_input_path(), ref_repr_h5(), grid


def run_evlib_full(out_dir: Path) -> float:
    """Run the real evlib pipeline end-to-end (h5 -> parquet -> representation h5).

    Returns internal wall-clock seconds for the pipeline body.
    """
    import evlib.rvt as rvt

    raw, _ref, grid = _ref_paths()
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    rvt.process_sequence(
        raw,
        out_dir,
        dataset="gen4",
        height=720,
        width=1280,
        ev_repr_timestamps_us=grid,
        downsample_by_2=True,
        engine="streaming",
    )
    return time.perf_counter() - start


def run_evlib_build_only(out_dir: Path, parquet: Path) -> float:
    """Run only the evlib representation build from a pre-built parquet (no conversion).

    Reuses the public pipeline primitives so this is the genuine build path, just with
    the one-time h5->parquet conversion removed. Returns internal wall-clock seconds.
    """
    import polars as pl

    from evlib.rvt import REPR_NAME
    from evlib.rvt.pipeline import (
        build_sparse_histogram,
        scatter_window_dense,
    )
    from evlib.rvt.writer import H5RepresentationWriter

    _raw, _ref, grid = _ref_paths()
    grid = np.asarray(grid, dtype=np.int64)
    height, width = 720, 1280
    nbins, count_cutoff, delta_t_us = 10, 10, 50_000
    downsample_by_2 = True
    window_batch_size = 10

    repr_dir = out_dir / "event_representations_v2" / REPR_NAME
    repr_dir.mkdir(parents=True, exist_ok=True)
    out_h, out_w = height // 2, width // 2
    channels = 2 * nbins
    num_windows = len(grid)
    out_h5 = repr_dir / "event_representations_ds2_nearest.h5"

    start = time.perf_counter()
    with H5RepresentationWriter(
        out_h5,
        num_windows=num_windows,
        channels=channels,
        height=out_h,
        width=out_w,
    ) as writer:
        for a in range(0, num_windows, window_batch_size):
            b = min(a + window_batch_size - 1, num_windows - 1)
            t_lo = int(grid[a] - delta_t_us)
            t_hi = int(grid[b])
            batch_events = pl.scan_parquet(str(parquet)).filter(
                pl.col("t").is_between(t_lo, t_hi)
            )
            sparse = build_sparse_histogram(
                batch_events,
                ev_repr_timestamps_us=grid[a : b + 1],
                delta_t_us=delta_t_us,
                nbins=nbins,
                count_cutoff=count_cutoff,
                height=height,
                width=width,
                downsample_by_2=downsample_by_2,
                engine="streaming",
            )
            if sparse.height:
                parts = sparse.partition_by("window_id", as_dict=True)
                for k, wdf in parts.items():
                    local = k[0] if isinstance(k, tuple) else k
                    dense = scatter_window_dense(wdf, channels, out_h, out_w)
                    writer.write_window(a + int(local), dense)
    return time.perf_counter() - start


def _add_rvt_paths() -> None:
    rvt_root = ROOT / "lib" / "RVT"
    genx = rvt_root / "scripts" / "genx"
    for p in (str(rvt_root), str(genx)):
        if p not in sys.path:
            sys.path.insert(0, p)


def run_rvt_reference(out_h5: Path) -> float:
    """Run the genuine RVT torch pipeline for the full sequence using RVT's own modules.

    This reuses RVT's ``H5Reader`` (numba cum-max time correction), ``StackedHistogram``
    (count_cutoff=10, fastmode=True), ``downsample_ev_repr`` (nearest-exact 0.5), and
    ``H5Writer`` exactly as in ``write_event_representations``. The only adaptation is
    that the window-end grid is supplied directly (the committed reference grid) instead
    of being recomputed from labels, so both pipelines window over the identical grid.
    Returns internal wall-clock seconds for the pipeline body.
    """
    _add_rvt_paths()
    import torch  # noqa: F401  (imported by RVT modules)
    from data.utils.representations import StackedHistogram
    from preprocess_dataset import H5Reader, H5Writer, downsample_ev_repr

    raw, _ref, grid = _ref_paths()
    grid = np.asarray(grid, dtype=np.int64)
    height, width = 720, 1280
    nbins, count_cutoff = 10, 10
    delta_t_us = 50_000

    out_h5.parent.mkdir(parents=True, exist_ok=True)
    if out_h5.exists():
        out_h5.unlink()

    ev_repr = StackedHistogram(
        bins=nbins, height=height, width=width, count_cutoff=count_cutoff, fastmode=True
    )
    ev_repr_shape = tuple(ev_repr.get_shape())
    ev_repr_shape = (ev_repr_shape[0], ev_repr_shape[1] // 2, ev_repr_shape[2] // 2)
    ev_repr_dtype = ev_repr.get_numpy_dtype()

    start = time.perf_counter()
    with (
        H5Reader(raw, dataset="gen4") as h5_reader,
        H5Writer(
            out_h5, key="data", ev_repr_shape=ev_repr_shape, numpy_dtype=ev_repr_dtype
        ) as h5_writer,
    ):
        ev_ts_us = h5_reader.time
        end_indices = np.searchsorted(ev_ts_us, grid, side="right")
        start_indices = np.searchsorted(ev_ts_us, grid - delta_t_us, side="left")
        for idx_start, idx_end in zip(start_indices, end_indices):
            ev_window = h5_reader.get_event_slice(
                idx_start=int(idx_start), idx_end=int(idx_end)
            )
            rep = ev_repr.construct(
                x=ev_window["x"],
                y=ev_window["y"],
                pol=ev_window["p"],
                time=ev_window["t"],
            )
            rep = rep.unsqueeze(0)
            rep = downsample_ev_repr(x=rep, scale_factor=0.5)
            h5_writer.add_data(rep.numpy()[0])
    return time.perf_counter() - start


# --------------------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------------------


def verify_against_reference(out_h5: Path) -> None:
    """Assert ``out_h5`` is bit-identical to the committed reference output."""
    import h5py

    from tests.rvt_fixtures import ref_repr_h5

    with (
        h5py.File(str(ref_repr_h5()), "r") as fref,
        h5py.File(str(out_h5), "r") as fout,
    ):
        ref = fref["data"][:]
        got = fout["data"][:]
    if ref.shape != got.shape:
        raise AssertionError(f"shape mismatch: ref {ref.shape} vs got {got.shape}")
    if not np.array_equal(ref, got):
        diff = int(np.count_nonzero(ref != got))
        raise AssertionError(
            f"{out_h5} not bit-identical to reference: {diff} differing elements "
            f"of {ref.size}"
        )


# --------------------------------------------------------------------------------------
# Benchmark orchestration
# --------------------------------------------------------------------------------------

CACHED_PARQUET = Path("/tmp/moorea_events.parquet")


def run_benchmark(
    repeats: int = 3, rvt_repeats: Optional[int] = None, timeout: float = 1800.0
) -> Dict[str, PipelineResult]:
    """Run all pipelines ``repeats`` times in subprocesses, returning measurements.

    Verifies each pipeline's output bit-identical to the reference once before timing.
    ``rvt_repeats`` lets the (slower) RVT reference use fewer repeats than evlib.
    """
    if rvt_repeats is None:
        rvt_repeats = repeats

    work = OUT_DIR / "work"
    work.mkdir(parents=True, exist_ok=True)

    specs = [
        ("evlib_full", "evlib streaming\n(h5 to parquet + build)", repeats),
        ("evlib_build", "evlib build only\n(from cached parquet)", repeats),
        ("rvt", "RVT torch\n(reference)", rvt_repeats),
    ]

    # Verification pass: run each pipeline once and check the output.
    print("Verification pass (one run each, asserting bit-identical output)...")
    for key, _label, _n in specs:
        out_h5 = _verify_one(key, work, timeout)
        verify_against_reference(out_h5)
        print(f"  {key}: output verified bit-identical to reference -> {out_h5}")

    results: Dict[str, PipelineResult] = {}
    for key, label, n in specs:
        note = label
        pr = PipelineResult(label=label, note=note)
        print(f"\nTiming {key} ({n} repeats)...")
        for i in range(n):
            res = run_pipeline_subprocess(
                ["--child", key, "--work", str(work)], timeout=timeout
            )
            pr.runs.append(res)
            print(
                f"  run {i + 1}/{n}: {res.wall_s:.2f}s, peak {res.peak_rss_bytes / 1024**3:.2f} GB"
            )
        results[key] = pr
    return results


def _verify_one(key: str, work: Path, timeout: float) -> Path:
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.bench_rvt_pipeline",
            "--child",
            key,
            "--work",
            str(work),
        ],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"verification child {key} failed:\n{res.stdout}\n{res.stderr}"
        )
    out_h5 = None
    for line in res.stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and "out_h5" in line:
            out_h5 = Path(json.loads(line)["out_h5"])
    if out_h5 is None:
        raise RuntimeError(f"child {key} did not report out_h5:\n{res.stdout}")
    return out_h5


def _child_main(key: str, work: Path) -> None:
    """Execute a single pipeline body and report wall_s + out_h5 as JSON on stdout."""
    if key == "evlib_full":
        out_dir = work / "evlib_full"
        wall = run_evlib_full(out_dir)
        from evlib.rvt import REPR_NAME

        out_h5 = (
            out_dir
            / "event_representations_v2"
            / REPR_NAME
            / "event_representations_ds2_nearest.h5"
        )
    elif key == "evlib_build":
        out_dir = work / "evlib_build"
        if not CACHED_PARQUET.exists():
            raise FileNotFoundError(f"cached parquet not found: {CACHED_PARQUET}")
        wall = run_evlib_build_only(out_dir, CACHED_PARQUET)
        from evlib.rvt import REPR_NAME

        out_h5 = (
            out_dir
            / "event_representations_v2"
            / REPR_NAME
            / "event_representations_ds2_nearest.h5"
        )
    elif key == "rvt":
        out_h5 = work / "rvt" / "event_representations_ds2_nearest.h5"
        wall = run_rvt_reference(out_h5)
    else:
        raise ValueError(f"unknown child key: {key}")
    print(json.dumps({"wall_s": wall, "out_h5": str(out_h5)}))


# --------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------


def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("#f7f7f7")
    ax.grid(axis="x", color="white", linewidth=1.2, zorder=0)
    ax.set_axisbelow(True)


def plot(results: Dict[str, PipelineResult], out_time: Path, out_mem: Path) -> None:
    """Render two horizontal-bar charts: wall-clock seconds and peak memory in GB."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["evlib_full", "evlib_build", "rvt"]
    order = [k for k in order if k in results]
    labels = [results[k].note for k in order]
    colours = ["#3b7dd8", "#7eb6f0", "#d86b3b"][: len(order)]

    # Time chart
    medians = [results[k].time_summary()["median"] for k in order]
    mins = [results[k].time_summary()["min"] for k in order]
    maxs = [results[k].time_summary()["max"] for k in order]
    xerr = [
        [m - lo for m, lo in zip(medians, mins)],
        [hi - m for m, hi in zip(medians, maxs)],
    ]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    y = np.arange(len(order))
    ax.barh(y, medians, color=colours, zorder=3, height=0.55)
    ax.errorbar(
        medians, y, xerr=xerr, fmt="none", ecolor="#333333", capsize=4, zorder=4, lw=1.2
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("wall-clock time (s), median with min..max")
    ax.set_title(
        "RVT preprocessing: wall-clock time", loc="left", fontsize=12, fontweight="bold"
    )
    for yi, m in zip(y, medians):
        ax.text(m, yi, f" {m:.1f}s", va="center", ha="left", fontsize=9)
    ax.set_xlim(0, max(maxs) * 1.18)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_time, dpi=150)
    plt.close(fig)

    # Memory chart
    gbs = [results[k].peak_gb for k in order]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.barh(y, gbs, color=colours, zorder=3, height=0.55)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("peak resident memory (GB)")
    ax.set_title(
        "RVT preprocessing: peak memory", loc="left", fontsize=12, fontweight="bold"
    )
    for yi, g in zip(y, gbs):
        ax.text(g, yi, f" {g:.2f} GB", va="center", ha="left", fontsize=9)
    ax.set_xlim(0, max(gbs) * 1.18)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_mem, dpi=150)
    plt.close(fig)


def write_summary_md(results: Dict[str, PipelineResult], path: Path) -> None:
    order = [k for k in ["evlib_full", "evlib_build", "rvt"] if k in results]
    lines = [
        "# evlib vs RVT preprocessing benchmark",
        "",
        "Full Gen4 validation sequence, raw h5 to stacked-histogram representation h5.",
        "All outputs verified bit-identical to the committed reference (1198 windows,",
        "shape (1198, 20, 360, 640) uint8).",
        "",
        "| pipeline | min (s) | median (s) | max (s) | peak RSS (GB) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for k in order:
        r = results[k]
        s = r.time_summary()
        label = r.note.replace("\n", " ")
        lines.append(
            f"| {label} | {s['min']:.2f} | {s['median']:.2f} | {s['max']:.2f} | {r.peak_gb:.2f} |"
        )
    lines.append("")
    if "evlib_full" in results and "rvt" in results:
        ev = results["evlib_full"].time_summary()["median"]
        rv = results["rvt"].time_summary()["median"]
        ev_mem = results["evlib_full"].peak_gb
        rv_mem = results["rvt"].peak_gb
        lines.append(
            f"Headline: evlib full pipeline is {rv / ev:.1f}x faster "
            f"({ev:.1f}s vs {rv:.1f}s median) and uses {rv_mem / ev_mem:.1f}x less peak "
            f"memory ({ev_mem:.2f} GB vs {rv_mem:.2f} GB) than RVT's torch reference."
        )
    path.write_text("\n".join(lines) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repeats", type=int, default=3, help="repeats for evlib pipelines"
    )
    parser.add_argument(
        "--rvt-repeats",
        type=int,
        default=None,
        help="repeats for RVT (default = --repeats)",
    )
    parser.add_argument(
        "--timeout", type=float, default=1800.0, help="per-run subprocess timeout (s)"
    )
    # Hidden child dispatch:
    parser.add_argument("--child", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--work", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.child is not None:
        _child_main(args.child, Path(args.work))
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = run_benchmark(
        repeats=args.repeats, rvt_repeats=args.rvt_repeats, timeout=args.timeout
    )

    out_time = OUT_DIR / "rvt_pipeline_time.png"
    out_mem = OUT_DIR / "rvt_pipeline_memory.png"
    plot(results, out_time, out_mem)
    md = OUT_DIR / "rvt_pipeline_bench.md"
    write_summary_md(results, md)

    print(f"\nWrote {out_time}")
    print(f"Wrote {out_mem}")
    print(f"Wrote {md}")
    print("\n" + md.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
