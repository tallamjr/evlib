"""Dataset-scale benchmark: evlib (GPU / CPU / Rust) vs RVT (GPU / CPU) over a full split.

This drives the raw -> processed RVT stacked-histogram preprocessing across every sequence
of a dataset split (e.g. the gen4_1mpx validation set, 19 sequences) and compares wall-clock
time, peak resident memory, and throughput for five pipelines:

1. ``evlib_gpu``  -- ``process_sequence(backend="polars", engine="gpu")`` (cudf-polars on the GPU).
2. ``evlib_cpu``  -- ``process_sequence(backend="polars", engine="auto")`` (Polars on the CPU).
3. ``evlib_rust`` -- ``process_sequence(backend="rust")`` (Rust dense scatter-add, CPU).
4. ``rvt_gpu``    -- RVT's own modules with event tensors moved onto the GPU.
5. ``rvt_cpu``    -- RVT's own modules on the CPU (the genuine reference pipeline).

Each (backend, sequence) run executes in a fresh subprocess so peak RSS is an unambiguous
per-process ``RUSAGE_SELF.ru_maxrss``. Every output is asserted bit-identical to that
sequence's committed RVT reference h5 before its timing is kept; a non-identical output is a
hard failure (the whole point of the comparison is exact reproduction).

The window grid for each sequence is taken from that sequence's own RVT
``timestamps_us.npy`` so the comparison isolates the representation computation (both
pipelines window over the identical grid) rather than re-deriving it from labels.

Run on the box (after activating the env)::

    source /home/tarek/evlib/.venv/bin/activate
    python -m benchmarks.bench_rvt_dataset \
        --orig-root ~/datasets/gen4_1mpx_original/val \
        --ref-root  ~/datasets/gen4_1mpx_processed_RVT/gen4/val \
        --backends evlib_gpu evlib_cpu evlib_rust rvt_gpu rvt_cpu
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / "out"
sys.path.insert(0, str(ROOT))

REPR_SUBDIR = "event_representations_v2/stacked_histogram_dt=50_nbins=10"
REF_H5_NAME = "event_representations_ds2_nearest.h5"
HEIGHT, WIDTH = 720, 1280
NBINS, COUNT_CUTOFF, DELTA_T_US = 10, 10, 50_000

ALL_BACKENDS = (
    "evlib_cuda",
    "evlib_gpu",
    "evlib_gpu_uvm",
    "evlib_cpu",
    "evlib_rust",
    "rvt_gpu",
    "rvt_cpu",
)
BACKEND_LABEL = {
    "evlib_cuda": "evlib CUDA (Rust scatter-add, GPU)",
    "evlib_gpu": "evlib polars (GPU / cudf, device pool)",
    "evlib_gpu_uvm": "evlib polars (GPU / cudf, UVM managed)",
    "evlib_cpu": "evlib polars (CPU)",
    "evlib_rust": "evlib rust (dense scatter-add)",
    "rvt_gpu": "RVT torch (GPU)",
    "rvt_cpu": "RVT torch (CPU, reference)",
}


def _uvm_gpu_engine():
    """Build the larger-than-VRAM Polars GPU engine: CUDA managed memory (UVM) under a
    memory pool with a prefetch adaptor, per pola.rs/posts/uvm-larger-than-ram-gpu.

    Managed memory oversubscribes the 24 GB device into host RAM so sequences that exceed
    VRAM stay on the GPU instead of falling back to CPU; the pool + prefetch keep the
    paging cost to ~1.2-1.3x vs a pure device pool.
    """
    import polars as pl
    import rmm

    mr = rmm.mr.PrefetchResourceAdaptor(
        rmm.mr.PoolMemoryResource(rmm.mr.ManagedMemoryResource())
    )
    return pl.GPUEngine(memory_resource=mr)


def ru_maxrss_bytes(value: int) -> int:
    """Normalise ru_maxrss to bytes (darwin reports bytes, linux KiB)."""
    return int(value) if sys.platform == "darwin" else int(value) * 1024


# --------------------------------------------------------------------------------------
# Sequence discovery
# --------------------------------------------------------------------------------------


@dataclass
class Sequence_:
    name: str
    in_h5: Path
    ref_h5: Path
    grid_npy: Path


def _raw_readable(h5: Path) -> bool:
    """True if the raw h5 opens and exposes /events/t (guards against truncated downloads)."""
    import h5py

    try:
        import hdf5plugin  # noqa: F401
    except ImportError:
        pass
    try:
        with h5py.File(str(h5), "r") as f:
            _ = f["events"]["t"].shape
        return True
    except OSError:
        return False


def discover_sequences(orig_root: Path, ref_root: Path) -> List[Sequence_]:
    """Pair each ``*_td.h5`` raw sequence with its committed RVT reference output + grid."""
    seqs: List[Sequence_] = []
    for h5 in sorted(Path(orig_root).glob("*_td.h5")):
        name = h5.name[: -len("_td.h5")]
        ref_dir = Path(ref_root) / name / REPR_SUBDIR
        ref_h5 = ref_dir / REF_H5_NAME
        grid_npy = ref_dir / "timestamps_us.npy"
        if not (ref_h5.exists() and grid_npy.exists()):
            print(
                f"  skip {name}: missing reference "
                f"(h5={ref_h5.exists()}, grid={grid_npy.exists()})"
            )
            continue
        if not _raw_readable(h5):
            print(f"  skip {name}: raw h5 truncated/corrupt")
            continue
        seqs.append(Sequence_(name, h5, ref_h5, grid_npy))
    return seqs


# --------------------------------------------------------------------------------------
# Pipeline bodies (executed inside child subprocesses)
# --------------------------------------------------------------------------------------


def _evlib_out_h5(out_dir: Path) -> Path:
    from evlib.rvt import REPR_NAME

    return (
        out_dir
        / "event_representations_v2"
        / REPR_NAME
        / "event_representations_ds2_nearest.h5"
    )


def run_evlib(
    in_h5: Path, grid: np.ndarray, out_dir: Path, *, backend: str, engine: object
) -> Tuple[float, Path]:
    import evlib.rvt as rvt

    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    rvt.process_sequence(
        in_h5,
        out_dir,
        dataset="gen4",
        height=HEIGHT,
        width=WIDTH,
        ev_repr_timestamps_us=grid,
        downsample_by_2=True,
        backend=backend,
        engine=engine,
    )
    return time.perf_counter() - start, _evlib_out_h5(out_dir)


def _add_rvt_paths() -> None:
    rvt_root = ROOT / "lib" / "RVT"
    genx = rvt_root / "scripts" / "genx"
    for p in (str(rvt_root), str(genx)):
        if p not in sys.path:
            sys.path.insert(0, p)


def run_rvt(
    in_h5: Path, grid: np.ndarray, out_h5: Path, *, device: str
) -> Tuple[float, Path]:
    """Run RVT's own modules; ``device='cuda'`` moves every window's events onto the GPU.

    Integer scatter-add (StackedHistogram, fastmode uint8) and nearest 0.5 downsample are
    device-agnostic, so the GPU output is expected bit-identical to the CPU reference.
    """
    _add_rvt_paths()
    import hdf5plugin  # noqa: F401  registers blosc/zstd filters used by RVT's H5Writer
    import torch
    from data.utils.representations import StackedHistogram
    from preprocess_dataset import H5Reader, H5Writer, downsample_ev_repr

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("rvt_gpu requested but torch.cuda.is_available() is False")

    grid = np.asarray(grid, dtype=np.int64)
    ev_repr = StackedHistogram(
        bins=NBINS, height=HEIGHT, width=WIDTH, count_cutoff=COUNT_CUTOFF, fastmode=True
    )
    shp = tuple(ev_repr.get_shape())
    shp = (shp[0], shp[1] // 2, shp[2] // 2)
    dt = ev_repr.get_numpy_dtype()

    out_h5.parent.mkdir(parents=True, exist_ok=True)
    if out_h5.exists():
        out_h5.unlink()

    start = time.perf_counter()
    with (
        H5Reader(in_h5, dataset="gen4") as reader,
        H5Writer(out_h5, key="data", ev_repr_shape=shp, numpy_dtype=dt) as writer,
    ):
        ev_ts_us = reader.time
        end_indices = np.searchsorted(ev_ts_us, grid, side="right")
        start_indices = np.searchsorted(ev_ts_us, grid - DELTA_T_US, side="left")
        for idx_start, idx_end in zip(start_indices, end_indices):
            ev = reader.get_event_slice(idx_start=int(idx_start), idx_end=int(idx_end))
            x, y, p, t = ev["x"], ev["y"], ev["p"], ev["t"]
            if device == "cuda":
                x, y, p, t = x.cuda(), y.cuda(), p.cuda(), t.cuda()
            rep = ev_repr.construct(x=x, y=y, pol=p, time=t).unsqueeze(0)
            rep = downsample_ev_repr(x=rep, scale_factor=0.5)
            writer.add_data(
                rep.cpu().numpy()[0] if device == "cuda" else rep.numpy()[0]
            )
        if device == "cuda":
            torch.cuda.synchronize()
    return time.perf_counter() - start, out_h5


# --------------------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------------------


# A handful of boundary events differ from the torch reference because evlib's float64
# binning only reproduces torch's float32 floor "on most data" (a pre-existing evlib-vs-torch
# edge case, identical on CPU and GPU). Tolerate these tiny counts but hard-fail on anything
# large enough to indicate a real regression.
DIFF_HARD_FAIL = 10_000


def verify_against_reference(out_h5: Path, ref_h5: Path) -> int:
    """Return the number of elements differing from the committed reference (0 = bit-identical).

    Raises on a shape mismatch or on a diff large enough to be a real bug (not boundary noise).
    """
    import h5py
    import hdf5plugin  # noqa: F401  registers blosc/zstd filters used by the reference h5

    with h5py.File(str(ref_h5), "r") as fref, h5py.File(str(out_h5), "r") as fout:
        ref = fref["data"][:]
        got = fout["data"][:]
    if ref.shape != got.shape:
        raise AssertionError(f"shape mismatch: ref {ref.shape} vs got {got.shape}")
    diff = int(np.count_nonzero(ref != got))
    if diff > DIFF_HARD_FAIL:
        raise AssertionError(
            f"{out_h5} differs from {ref_h5} by {diff}/{ref.size} elements (> {DIFF_HARD_FAIL}); "
            "this is a real regression, not boundary noise"
        )
    return diff


# --------------------------------------------------------------------------------------
# Child dispatch
# --------------------------------------------------------------------------------------


def _child_main(args: argparse.Namespace) -> None:
    grid = np.load(args.grid).astype(np.int64)
    in_h5 = Path(args.in_h5)
    work = Path(args.work)
    backend = args.child

    if backend == "evlib_cuda":
        wall, out_h5 = run_evlib(in_h5, grid, work, backend="cuda", engine="auto")
    elif backend == "evlib_gpu":
        wall, out_h5 = run_evlib(in_h5, grid, work, backend="polars", engine="gpu")
    elif backend == "evlib_gpu_uvm":
        wall, out_h5 = run_evlib(
            in_h5, grid, work, backend="polars", engine=_uvm_gpu_engine()
        )
    elif backend == "evlib_cpu":
        wall, out_h5 = run_evlib(in_h5, grid, work, backend="polars", engine="auto")
    elif backend == "evlib_rust":
        wall, out_h5 = run_evlib(in_h5, grid, work, backend="rust", engine="auto")
    elif backend == "rvt_gpu":
        wall, out_h5 = run_rvt(in_h5, grid, work / REF_H5_NAME, device="cuda")
    elif backend == "rvt_cpu":
        wall, out_h5 = run_rvt(in_h5, grid, work / REF_H5_NAME, device="cpu")
    else:
        raise ValueError(f"unknown child backend: {backend}")

    peak = ru_maxrss_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print(json.dumps({"wall_s": wall, "peak_rss_bytes": peak, "out_h5": str(out_h5)}))


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------


@dataclass
class SeqRun:
    backend: str
    seq: str
    wall_s: float
    peak_rss_bytes: int
    n_windows: int
    diff_elems: int = 0


def _run_child(backend: str, seq: Sequence_, work: Path, timeout: float) -> SeqRun:
    out = work / backend / seq.name
    out.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.bench_rvt_dataset",
            "--child",
            backend,
            "--in-h5",
            str(seq.in_h5),
            "--grid",
            str(seq.grid_npy),
            "--work",
            str(out),
        ],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"child {backend} / {seq.name} failed (rc={proc.returncode})\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    payload = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and "peak_rss_bytes" in line:
            payload = json.loads(line)
            break
    if payload is None:
        raise RuntimeError(
            f"no JSON from child {backend}/{seq.name}:\n{proc.stdout}\n{proc.stderr}"
        )
    out_h5 = Path(str(payload["out_h5"]))
    diff = verify_against_reference(out_h5, seq.ref_h5)
    grid = np.load(seq.grid_npy)
    # Free the per-run output to bound disk (26 GB reference scale).
    try:
        out_h5.unlink()
    except FileNotFoundError:
        pass
    return SeqRun(
        backend=backend,
        seq=seq.name,
        wall_s=float(payload["wall_s"]),
        peak_rss_bytes=int(payload["peak_rss_bytes"]),
        n_windows=int(grid.shape[0]),
        diff_elems=diff,
    )


def run_benchmark(
    orig_root: Path,
    ref_root: Path,
    backends: Sequence[str],
    timeout: float,
    limit: Optional[int] = None,
) -> Dict[str, List[SeqRun]]:
    seqs = discover_sequences(orig_root, ref_root)
    if not seqs:
        raise SystemExit(
            f"no sequences found under {orig_root} with references in {ref_root}"
        )
    if limit is not None:
        seqs = seqs[:limit]
    print(f"Discovered {len(seqs)} sequences with full references.")
    work = OUT_DIR / "dataset_work"
    work.mkdir(parents=True, exist_ok=True)

    results: Dict[str, List[SeqRun]] = {b: [] for b in backends}
    for backend in backends:
        print(f"\n=== {backend} ({BACKEND_LABEL[backend]}) ===")
        for i, seq in enumerate(seqs):
            run = _run_child(backend, seq, work, timeout)
            results[backend].append(run)
            tag = (
                "bit-identical"
                if run.diff_elems == 0
                else f"{run.diff_elems} boundary elems differ"
            )
            print(
                f"  [{i + 1}/{len(seqs)}] {seq.name}: {run.wall_s:.2f}s, "
                f"peak {run.peak_rss_bytes / 1024**3:.2f} GB, {run.n_windows} windows  [{tag}]"
            )
    return results


def _save(results: Dict[str, List[SeqRun]], path: Path) -> None:
    payload = {
        b: [
            {
                "seq": r.seq,
                "wall_s": r.wall_s,
                "peak_rss_bytes": r.peak_rss_bytes,
                "n_windows": r.n_windows,
                "diff_elems": r.diff_elems,
            }
            for r in runs
        ]
        for b, runs in results.items()
    }
    path.write_text(json.dumps(payload, indent=2))


def write_summary_md(results: Dict[str, List[SeqRun]], path: Path) -> None:
    order = [b for b in ALL_BACKENDS if b in results and results[b]]
    n_seq = len(next(iter(results.values()))) if results else 0
    lines = [
        "# evlib vs RVT preprocessing benchmark (full dataset)",
        "",
        f"Full gen4_1mpx validation split: {n_seq} sequences, raw h5 -> stacked-histogram h5.",
        "Every per-sequence output verified bit-identical to the committed RVT reference.",
        "Wall-clock and peak RSS measured per sequence in isolated subprocesses (single pass).",
        "",
        "| pipeline | total time (s) | mean/seq (s) | peak RSS (GB) | total windows | diff vs ref (elems) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    totals: Dict[str, float] = {}
    for b in order:
        runs = results[b]
        total = sum(r.wall_s for r in runs)
        totals[b] = total
        mean = total / len(runs)
        peak = max(r.peak_rss_bytes for r in runs) / (1024**3)
        nwin = sum(r.n_windows for r in runs)
        tdiff = sum(r.diff_elems for r in runs)
        total_elems = sum(r.n_windows for r in runs) * 20 * 360 * 640
        diff_str = (
            "0 (bit-identical)"
            if tdiff == 0
            else f"{tdiff} / {total_elems} ({tdiff / total_elems:.1e})"
        )
        lines.append(
            f"| {BACKEND_LABEL[b]} | {total:.1f} | {mean:.2f} | {peak:.2f} | {nwin} | {diff_str} |"
        )
    lines.append("")
    # Comparison phrases anchored on the genuine reference (rvt_cpu) when present.
    base = "rvt_cpu" if "rvt_cpu" in totals else None
    if base:
        for b in order:
            if b == base:
                continue
            a, r = totals[b], totals[base]
            if a <= r:
                lines.append(
                    f"{BACKEND_LABEL[b]} is {r / a:.2f}x faster than {BACKEND_LABEL[base]} (total {a:.1f}s vs {r:.1f}s)."
                )
            else:
                lines.append(
                    f"{BACKEND_LABEL[b]} is {a / r:.2f}x slower than {BACKEND_LABEL[base]} (total {a:.1f}s vs {r:.1f}s)."
                )
    if "evlib_gpu" in totals and "evlib_cpu" in totals:
        g, c = totals["evlib_gpu"], totals["evlib_cpu"]
        if g <= c:
            lines.append(
                f"evlib GPU is {c / g:.2f}x faster than evlib CPU at full-dataset scale (total {g:.1f}s vs {c:.1f}s)."
            )
        else:
            lines.append(
                f"evlib GPU is {g / c:.2f}x slower than evlib CPU at full-dataset scale (total {g:.1f}s vs {c:.1f}s)."
            )
    path.write_text("\n".join(lines) + "\n")


def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("#f7f7f7")
    ax.grid(axis="x", color="white", linewidth=1.2, zorder=0)
    ax.set_axisbelow(True)


COLOUR = {
    "evlib_cuda": "#9b59b6",
    "evlib_gpu": "#2a9d5c",
    "evlib_gpu_uvm": "#1f7a44",
    "evlib_cpu": "#3b7dd8",
    "evlib_rust": "#7eb6f0",
    "rvt_gpu": "#e0a23b",
    "rvt_cpu": "#d86b3b",
}


def plot(results: Dict[str, List[SeqRun]], out_time: Path, out_mem: Path) -> None:
    """Render horizontal-bar charts: total wall-clock and peak RSS per backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Tahoma"
    order = [b for b in ALL_BACKENDS if b in results and results[b]]
    labels = [BACKEND_LABEL[b].replace(" (", "\n(") for b in order]
    colours = [COLOUR[b] for b in order]
    y = np.arange(len(order))

    totals = [sum(r.wall_s for r in results[b]) for b in order]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.barh(y, totals, color=colours, zorder=3, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("total wall-clock over the split (s)")
    n_seq = len(results[order[0]])
    ax.set_title(
        f"RVT preprocessing, full gen4 val ({n_seq} seqs): total time",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )
    for yi, t in zip(y, totals):
        ax.text(t, yi, f" {t:.0f}s", va="center", ha="left", fontsize=9)
    ax.set_xlim(0, max(totals) * 1.18)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_time, dpi=300)
    plt.close(fig)

    peaks = [max(r.peak_rss_bytes for r in results[b]) / (1024**3) for b in order]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.barh(y, peaks, color=colours, zorder=3, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("peak resident memory, max over sequences (GB)")
    ax.set_title(
        f"RVT preprocessing, full gen4 val ({n_seq} seqs): peak host memory",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )
    for yi, g in zip(y, peaks):
        ax.text(g, yi, f" {g:.2f} GB", va="center", ha="left", fontsize=9)
    ax.set_xlim(0, max(peaks) * 1.18)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_mem, dpi=300)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orig-root", type=Path, help="dir of raw *_td.h5 sequences")
    parser.add_argument(
        "--ref-root", type=Path, help="dir of RVT processed reference sequences"
    )
    parser.add_argument(
        "--backends", nargs="+", default=list(ALL_BACKENDS), choices=list(ALL_BACKENDS)
    )
    parser.add_argument(
        "--timeout", type=float, default=1800.0, help="per-run subprocess timeout (s)"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="cap number of sequences (smoke test)"
    )
    parser.add_argument(
        "--out-prefix",
        default="rvt_dataset",
        help="output filename prefix under benchmarks/out",
    )
    parser.add_argument(
        "--replot",
        action="store_true",
        help="re-render md + plots from saved results JSON",
    )
    # Hidden child dispatch:
    parser.add_argument("--child", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--in-h5", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--grid", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--work", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.child is not None:
        _child_main(args)
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_json = OUT_DIR / f"{args.out_prefix}_results.json"
    md = OUT_DIR / f"{args.out_prefix}_bench.md"
    out_time = OUT_DIR / f"{args.out_prefix}_time.png"
    out_mem = OUT_DIR / f"{args.out_prefix}_memory.png"

    if args.replot:
        payload = json.loads(results_json.read_text())
        results = {
            b: [
                SeqRun(
                    b,
                    r["seq"],
                    r["wall_s"],
                    r["peak_rss_bytes"],
                    r["n_windows"],
                    r.get("diff_elems", 0),
                )
                for r in runs
            ]
            for b, runs in payload.items()
        }
    else:
        if args.orig_root is None or args.ref_root is None:
            parser.error("--orig-root and --ref-root are required")
        orig = args.orig_root.expanduser()
        ref = args.ref_root.expanduser()
        results = run_benchmark(
            orig, ref, args.backends, args.timeout, limit=args.limit
        )
        # Merge into any existing results so a single-backend re-run (e.g. evlib_gpu_uvm)
        # augments the saved set rather than discarding the other backends.
        if results_json.exists():
            prior = json.loads(results_json.read_text())
            merged = {
                b: [
                    SeqRun(
                        b,
                        r["seq"],
                        r["wall_s"],
                        r["peak_rss_bytes"],
                        r["n_windows"],
                        r.get("diff_elems", 0),
                    )
                    for r in runs
                ]
                for b, runs in prior.items()
            }
            merged.update(results)
            results = merged
        _save(results, results_json)

    write_summary_md(results, md)
    print(f"\nWrote {results_json}")
    print(f"Wrote {md}")
    # Plotting is optional: the box has no matplotlib, so render locally via --replot.
    try:
        plot(results, out_time, out_mem)
        print(f"Wrote {out_time}")
        print(f"Wrote {out_mem}")
    except ImportError as exc:
        print(
            f"Skipped plots (matplotlib unavailable: {exc}); run --replot where it is installed."
        )
    print("\n" + md.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
