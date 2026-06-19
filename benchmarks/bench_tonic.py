"""Benchmark evlib (Polars CPU + Polars GPU/cudf UVM) against tonic for event representations.

tonic (pure NumPy) is the natural baseline for evlib's general representation/filtering surface
(unlike RVT, which is a single-model preprocessing script). This harness loads one real event
stream once, feeds the IDENTICAL events to both libraries, and compares wall-clock and peak RSS
for the representations both implement: voxel grid, event frame (ToFrame n_time_bins), and HOTS
time surface. evlib's outputs are already bit/closely validated against tonic, so this measures
speed and memory, not correctness.

Each (op, backend) runs in a fresh subprocess for an unambiguous peak RSS. evlib runs on the CPU
Polars engine and on the cudf GPU engine with a CUDA managed-memory (UVM) resource so it can
oversubscribe VRAM on large streams.

    source /home/tarek/evlib/.venv/bin/activate
    python -m benchmarks.bench_tonic --raw ~/datasets/eTram/raw/test_1/test_day_001.raw --n-events 30000000
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / "out"
sys.path.insert(0, str(ROOT))

HEIGHT, WIDTH = 720, 1280
N_TIME_BINS_VOXEL = 5
N_TIME_BINS_FRAME = 10
OPS = ("voxel_grid", "event_frame", "time_surface")
TIME_SURFACE_SLICES = 20  # dt chosen so the recording yields ~this many surfaces
BACKENDS = ("tonic", "evlib_cpu", "evlib_gpu_uvm")
LABEL = {
    "tonic": "tonic (NumPy)",
    "evlib_cpu": "evlib Polars (CPU)",
    "evlib_gpu_uvm": "evlib Polars (GPU / cudf UVM)",
}


def ru_maxrss_bytes(v: int) -> int:
    return int(v) if sys.platform == "darwin" else int(v) * 1024


def _uvm_engine():
    import polars as pl
    import rmm

    mr = rmm.mr.PrefetchResourceAdaptor(
        rmm.mr.PoolMemoryResource(rmm.mr.ManagedMemoryResource())
    )
    return pl.GPUEngine(memory_resource=mr)


def _load_events(raw: Path, n_events: int):
    """Load n_events from the raw stream once; return a Polars DataFrame (evlib schema)."""
    import evlib

    return evlib.load_events(str(raw)).limit(n_events).collect()


def _tonic_array(df):
    """Build a tonic structured event array (x,y,t-us,p in {0,1}) from the evlib DataFrame."""
    x = df["x"].to_numpy().astype(np.int16)
    y = df["y"].to_numpy().astype(np.int16)
    t = df["t"].dt.total_microseconds().to_numpy().astype(np.int64)
    pol = df["polarity"].to_numpy()
    p = ((pol > 0).astype(np.int8))  # evlib -1/+1 -> tonic 0/1
    dtype = np.dtype([("x", np.int16), ("y", np.int16), ("t", np.int64), ("p", np.int8)])
    arr = np.empty(len(x), dtype=dtype)
    arr["x"], arr["y"], arr["t"], arr["p"] = x, y, t, p
    return arr


def _child(args: argparse.Namespace) -> None:
    op, backend = args.child, args.backend
    df = _load_events(Path(args.raw), args.n_events)
    n = df.height
    sensor = (WIDTH, HEIGHT, 2)

    # Prepare each library's input BEFORE the timer so we measure only the representation, not the
    # one-off conversion into tonic's structured-array format (tonic users load straight into it).
    if backend == "tonic":
        import tonic.transforms as T

        arr = _tonic_array(df)
    else:
        import evlib.representations as evr

        engine = _uvm_engine() if backend == "evlib_gpu_uvm" else "auto"
        lf = df.lazy()

    dt = tau = 0.0
    if op == "time_surface":
        t_us = df["t"].dt.total_microseconds().to_numpy()
        span = float(t_us[-1] - t_us[0])
        dt = span / TIME_SURFACE_SLICES
        tau = 2.0 * dt

    start = time.perf_counter()
    if backend == "tonic":
        if op == "voxel_grid":
            out = T.ToVoxelGrid(sensor_size=sensor, n_time_bins=N_TIME_BINS_VOXEL)(arr)
        elif op == "event_frame":
            out = T.ToFrame(sensor_size=sensor, n_time_bins=N_TIME_BINS_FRAME)(arr)
        elif op == "time_surface":
            out = T.ToTimesurface(sensor_size=sensor, dt=dt, tau=tau)(arr)
        else:
            raise ValueError(op)
        sig = float(np.asarray(out).sum())
    else:
        if op == "voxel_grid":
            res = evr.create_voxel_grid(lf, HEIGHT, WIDTH, N_TIME_BINS_VOXEL, engine=engine)
        elif op == "event_frame":
            res = evr.create_event_frame(lf, HEIGHT, WIDTH, N_TIME_BINS_FRAME, engine=engine)
        elif op == "time_surface":
            res = evr.create_time_surface(lf, HEIGHT, WIDTH, dt=dt, tau=tau, engine=engine)
        else:
            raise ValueError(op)
        sig = float(res.height)
    wall = time.perf_counter() - start
    peak = ru_maxrss_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print(json.dumps({"wall_s": wall, "peak_rss_bytes": peak, "n_events": n, "sig": sig}))


def _run(op: str, backend: str, raw: Path, n_events: int, timeout: float) -> Dict:
    proc = subprocess.run(
        [
            sys.executable, "-m", "benchmarks.bench_tonic",
            "--child", op, "--backend", backend, "--raw", str(raw),
            "--n-events", str(n_events),
        ],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{op}/{backend} failed (rc={proc.returncode})\n{proc.stdout}\n{proc.stderr}")
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and "peak_rss_bytes" in line:
            return json.loads(line)
    raise RuntimeError(f"no JSON from {op}/{backend}:\n{proc.stdout}\n{proc.stderr}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--n-events", type=int, default=30_000_000)
    parser.add_argument("--backends", nargs="+", default=list(BACKENDS), choices=list(BACKENDS))
    parser.add_argument("--ops", nargs="+", default=list(OPS), choices=list(OPS))
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--out-prefix", default="tonic_bench")
    parser.add_argument("--child", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--backend", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.child is not None:
        _child(args)
        return 0
    if args.raw is None:
        parser.error("--raw is required")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict[str, Dict]] = {}
    for op in args.ops:
        results[op] = {}
        print(f"\n=== {op} ({args.n_events:,} events) ===")
        for backend in args.backends:
            r = _run(op, backend, args.raw, args.n_events, args.timeout)
            results[op][backend] = r
            print(f"  {LABEL[backend]:30s} {r['wall_s']:8.2f}s  peak {r['peak_rss_bytes']/1024**3:6.2f} GB")

    md = OUT_DIR / f"{args.out_prefix}.md"
    lines = [
        "# evlib vs tonic representation benchmark",
        "",
        f"Single event stream, {args.n_events:,} events, eTram (1280x720). Identical events to both.",
        "Wall-clock and peak RSS per (op, backend) in isolated subprocesses (single pass).",
        "",
    ]
    for op in args.ops:
        lines += [f"## {op}", "", "| backend | time (s) | peak RSS (GB) | events/s |", "| --- | --- | --- | --- |"]
        for backend in args.backends:
            r = results[op][backend]
            eps = r["n_events"] / r["wall_s"]
            lines.append(f"| {LABEL[backend]} | {r['wall_s']:.2f} | {r['peak_rss_bytes']/1024**3:.2f} | {eps/1e6:.1f}M |")
        # speedup vs tonic
        if "tonic" in results[op]:
            tw = results[op]["tonic"]["wall_s"]
            for backend in args.backends:
                if backend == "tonic":
                    continue
                bw = results[op][backend]["wall_s"]
                rel = f"{tw/bw:.2f}x faster" if bw <= tw else f"{bw/tw:.2f}x slower"
                lines.append(f"")
                lines.append(f"{LABEL[backend]} is {rel} than tonic for {op} ({bw:.2f}s vs {tw:.2f}s).")
        lines.append("")
    (OUT_DIR / f"{args.out_prefix}_results.json").write_text(json.dumps(results, indent=2))
    md.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {md}")
    print("\n" + md.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
