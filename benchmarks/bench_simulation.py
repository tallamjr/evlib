"""Benchmark evlib.simulation on a PNG frame folder (VID2E layout).

Run: .venv/bin/python -m benchmarks.bench_simulation --frames-dir ~/vid2e-bench/raw30/320x320/seq \
         --backend cuda --batches 32,256,all --out benchmarks/out/evlib_raw30_320x320.json
The folder holds timestamps.txt (seconds) and imgs/*.png. Frames load once; one
persistent ESIMSimulator then runs the stack in slices of each batch size. Per
mode the median of --repeats after one warm-up: kernel (unsorted raw arrays),
sorted (raw arrays sorted by time) and wall (sorted Polars DataFrame, the public
path). Reports events/s, frames/s and video-hours per GPU-day (24 * video_s /
wall_s). --stages (CUDA) adds per-stage CUDA event times through the libevsim.so
C ABI and the device-resident kernel ceiling. Acceptance: every mode and backend
gives the same event count; the script raises otherwise.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

import evlib
from evlib.simulation import ESIMConfig, ESIMSimulator


def load_sequence(seq: Path, max_frames: int | None):
    lines = (seq / "timestamps.txt").read_text().split()
    pngs = sorted((seq / "imgs").glob("*.png"))
    if len(lines) != len(pngs):
        raise RuntimeError(f"{seq}: {len(lines)} timestamps but {len(pngs)} PNGs")
    if max_frames is not None:
        lines, pngs = lines[:max_frames], pngs[:max_frames]
    t_ns = np.asarray([round(float(v) * 1e9) for v in lines], dtype=np.int64)
    first = np.asarray(Image.open(pngs[0]).convert("L"), dtype=np.uint8)
    frames = np.empty((len(pngs),) + first.shape, dtype=np.uint8)
    frames[0] = first
    for k, path in enumerate(pngs[1:], start=1):
        frames[k] = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    return frames, t_ns


def batch_bounds(n_frames: int, batch: int | None):
    if batch is None:
        return [(0, n_frames)]
    return [(a, min(a + batch, n_frames)) for a in range(0, n_frames, batch)]


def run_raw(sim: ESIMSimulator, frames, t_ns, bounds, sort: bool):
    sim.reset()
    return [sim._inner.run(frames[a:b], t_ns[a:b], sort=sort) for a, b in bounds]


def run_api(sim: ESIMSimulator, frames, t_ns, bounds):
    sim.reset()
    return [sim.simulate(frames[a:b], t_ns[a:b], sort=True) for a, b in bounds]


def timed(fn, repeats: int):
    """One warm-up call, then the median wall of `repeats` calls and the last result."""
    fn()
    walls = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        walls.append(time.perf_counter() - start)
    return statistics.median(walls), walls, result


class EvsimLib:
    """ctypes view of libevsim.so: evsim_run2 with CUDA event timing per stage."""

    STAGE_NAMES = "call stage upload count scan write sort decode download".split()
    DEVICE_STAGES = ["count", "scan", "write", "decode"]
    FLAG_SORT, FLAG_RESIDENT, FLAG_NO_DOWNLOAD = 1, 4, 8

    def __init__(self, path: str, sim: ESIMSimulator, refractory_ns: int):
        self.lib = lib = ctypes.CDLL(path)
        vp, ll, ci = ctypes.c_void_p, ctypes.c_longlong, ctypes.c_int
        pvp = ctypes.POINTER(vp)
        signatures = {
            "evsim_create": [ci, ci, vp, vp, ll, pvp],
            "evsim_reset": [vp],
            "evsim_destroy": [vp],
            "evsim_set_lut": [vp, vp],
            "evsim_set_timing": [vp, ci],
            "evsim_last_timings": [vp, ctypes.POINTER(ctypes.c_double), ci],
            "evsim_run2": [
                vp,
                ci,
                vp,
                vp,
                ci,
                ci,
                pvp,
                pvp,
                pvp,
                pvp,
                ctypes.POINTER(ll),
            ],
        }
        for name, argtypes in signatures.items():
            getattr(lib, name).argtypes = argtypes
        c_pos, c_neg = (
            np.ascontiguousarray(a, dtype=np.float32) for a in sim.thresholds()
        )
        # The exact LUT of the product path; np.log differs from Rust f32::ln by an ulp on a few entries.
        self.lut = np.ascontiguousarray(sim._inner.log_lut(), dtype=np.float32)
        self.handle = vp()
        args = (
            sim.width,
            sim.height,
            c_pos.ctypes.data,
            c_neg.ctypes.data,
            refractory_ns,
        )
        self._check(lib.evsim_create(*args, ctypes.byref(self.handle)), "evsim_create")
        self._check(lib.evsim_set_timing(self.handle, 1), "evsim_set_timing")
        self._check(
            lib.evsim_set_lut(self.handle, self.lut.ctypes.data), "evsim_set_lut"
        )
        self.timings = (ctypes.c_double * len(self.STAGE_NAMES))()

    @staticmethod
    def _check(rc: int, name: str):
        if rc != 0:
            raise RuntimeError(f"{name} rc {rc}")

    def reset(self):
        self.lib.evsim_reset(self.handle)

    def run2(self, frames, t_ns, flags: int):
        """(T, H, W) uint8 slice through evsim_run2; returns (n_events, stage seconds by name)."""
        outs = [ctypes.byref(ctypes.c_void_p()) for _ in range(4)]
        n_events = ctypes.c_longlong(0)
        args = (
            self.handle,
            1,
            frames.ctypes.data,
            t_ns.ctypes.data,
            frames.shape[0],
            flags,
        )
        self._check(
            self.lib.evsim_run2(*args, *outs, ctypes.byref(n_events)), "evsim_run2"
        )
        self.lib.evsim_last_timings(self.handle, self.timings, len(self.STAGE_NAMES))
        return int(n_events.value), {
            k: ms / 1e3 for k, ms in zip(self.STAGE_NAMES, self.timings)
        }

    def breakdown(self, frames, t_ns, bounds, flags: int):
        """Summed stage seconds over the slices plus the device-stage sum."""
        self.reset()
        total = {k: 0.0 for k in self.STAGE_NAMES}
        events = 0
        for a, b in bounds:
            n, stage = self.run2(frames[a:b], np.ascontiguousarray(t_ns[a:b]), flags)
            events += n
            for k, v in stage.items():
                total[k] += v
        out = {"events": events, **{k + "_s": v for k, v in total.items()}}
        out["device_s"] = sum(total[k] for k in self.DEVICE_STAGES)
        return out

    def resident_ceiling(self, frames, t_ns, repeats: int):
        """Upload the whole stack once, then time reruns on the resident frames without download."""
        t = np.ascontiguousarray(t_ns)
        self.reset()
        self.run2(frames, t, self.FLAG_NO_DOWNLOAD)
        walls = []
        for _ in range(repeats):
            self.reset()
            start = time.perf_counter()
            self.run2(frames, t, self.FLAG_RESIDENT | self.FLAG_NO_DOWNLOAD)
            walls.append(time.perf_counter() - start)
        return statistics.median(walls)

    def close(self):
        self.lib.evsim_destroy(self.handle)


def gpu_name() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return f"unavailable ({exc})"
    return out.stdout.strip().splitlines()[0]


def median_stages(evsim: EvsimLib, frames, t_ns, bounds, flags: int, repeats: int):
    evsim.breakdown(frames, t_ns, bounds, flags)
    runs = [evsim.breakdown(frames, t_ns, bounds, flags) for _ in range(repeats)]
    return {k: statistics.median(r[k] for r in runs) for k in runs[0]}


def bench_backend(backend: str, frames, t_ns, batches, args, video_s: float):
    n_frames, height, width = frames.shape
    cfg = ESIMConfig(
        positive_threshold=args.c_pos, negative_threshold=args.c_neg, device=backend
    )
    sim = ESIMSimulator(cfg, width=width, height=height)
    evsim = None
    if backend == "cuda" and args.stages:
        evsim = EvsimLib(
            os.environ.get("EVLIB_CUDA_SIM_LIB", "libevsim.so"), sim, cfg.refractory_ns
        )
    rows = []
    for batch in batches:
        bounds = batch_bounds(n_frames, batch)
        kernel_s, kernel_walls, raw = timed(
            lambda: run_raw(sim, frames, t_ns, bounds, False), args.repeats
        )
        n_events = int(sum(len(q[0]) for q in raw))
        sorted_s, sorted_walls, _ = timed(
            lambda: run_raw(sim, frames, t_ns, bounds, True), args.repeats
        )
        wall_s, api_walls, dfs = timed(
            lambda: run_api(sim, frames, t_ns, bounds), args.repeats
        )
        api_events = sum(len(d) for d in dfs)
        if api_events != n_events:
            raise RuntimeError(
                f"{backend} batch {batch}: DataFrame path gave {api_events} events != {n_events}"
            )
        row = {
            "backend": backend,
            "batch": "whole" if batch is None else batch,
            "n_batches": len(bounds),
            "events": n_events,
            "kernel_s": kernel_s,
            "sorted_s": sorted_s,
            "sort_s": sorted_s - kernel_s,
            "df_s": max(wall_s - sorted_s, 0.0),
            "wall_s": wall_s,
            "kernel_walls": kernel_walls,
            "sorted_walls": sorted_walls,
            "api_walls": api_walls,
            "events_per_s": n_events / wall_s,
            "kernel_events_per_s": n_events / kernel_s,
            "frames_per_s": n_frames / wall_s,
            "video_hours_per_gpu_day": 24.0 * video_s / wall_s,
        }
        if evsim is not None:
            row["stages"] = median_stages(evsim, frames, t_ns, bounds, 0, args.repeats)
            row["stages_sorted"] = median_stages(
                evsim, frames, t_ns, bounds, EvsimLib.FLAG_SORT, args.repeats
            )
            if batch is None:
                row["resident_ceiling_s"] = evsim.resident_ceiling(
                    frames, t_ns, args.repeats
                )
        rows.append(row)
        print(
            f"  {backend:4s} batch {row['batch']!s:5s}: {n_events:,} ev  kernel {kernel_s:.3f} s  "
            f"sort {row['sort_s']:.3f} s  df {row['df_s']:.3f} s  wall {wall_s:.3f} s  "
            f"{row['events_per_s'] / 1e6:.1f} M ev/s  {row['video_hours_per_gpu_day']:.0f} vh/day"
        )
        del raw, dfs
    if evsim is not None:
        evsim.close()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--frames-dir",
        required=True,
        type=Path,
        help="folder with timestamps.txt and imgs/*.png",
    )
    ap.add_argument("--backend", default="cpu,cuda", help="comma list of cpu, cuda")
    ap.add_argument(
        "--batches",
        default="32,256,all",
        help="comma list of frames per call; all = whole stack",
    )
    ap.add_argument("--out", required=True, type=Path, help="JSON output path")
    ap.add_argument("--label", default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--c-pos", type=float, default=0.2)
    ap.add_argument("--c-neg", type=float, default=0.2)
    ap.add_argument(
        "--stages",
        action="store_true",
        help="CUDA per-stage timing through EVLIB_CUDA_SIM_LIB",
    )
    args = ap.parse_args()

    start = time.perf_counter()
    frames, t_ns = load_sequence(args.frames_dir.expanduser(), args.max_frames)
    load_s = time.perf_counter() - start
    n_frames, height, width = frames.shape
    video_s = float(t_ns[-1] - t_ns[0]) / 1e9
    label = args.label or args.frames_dir.expanduser().parent.name
    print(
        f"{label}: {n_frames} frames {width}x{height}, {video_s:.2f} s video, load {load_s:.2f} s"
    )

    backends = [b for b in args.backend.split(",") if b]
    if "cuda" in backends and not evlib.simulation_rs.cuda_available():
        raise RuntimeError(
            "cuda requested but evlib.simulation_rs.cuda_available() is False"
        )
    batches = [
        None if b in ("all", "whole") else int(b) for b in args.batches.split(",") if b
    ]

    rows = []
    for backend in backends:
        rows.extend(bench_backend(backend, frames, t_ns, batches, args, video_s))
    counts = {r["events"] for r in rows}
    if len(counts) != 1:
        raise RuntimeError(f"event counts differ across modes: {sorted(counts)}")

    out = {
        "label": label,
        "seq": str(args.frames_dir),
        "frames": n_frames,
        "width": width,
        "height": height,
        "video_s": video_s,
        "load_s": load_s,
        "c_pos": args.c_pos,
        "c_neg": args.c_neg,
        "repeats": args.repeats,
        "host": platform.node(),
        "cpu_count": os.cpu_count(),
        "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS"),
        "gpu": gpu_name() if "cuda" in backends else None,
        "evlib_version": evlib.__version__,
        "rows": rows,
    }
    args.out.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.out.expanduser().write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
