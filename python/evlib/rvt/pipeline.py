"""Orchestrate the RVT-identical preprocessing for one sequence."""

from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

# h5py and the hdf5plugin blosc filter are imported lazily inside the function that reads the
# raw h5 (below), so that ``import evlib.rvt`` works on platforms without h5py (e.g. Windows CI).

from evlib.rvt.downsample import selected_source_indices
from evlib.rvt.representation import (
    build_sparse_histogram,
    build_sparse_histogram_assigned,
)
from evlib.rvt.writer import H5RepresentationWriter, scatter_window_dense

REPR_NAME = "stacked_histogram_dt50_nbins10"


def _assign_window_ids(t_batch: np.ndarray, local_grid: np.ndarray, delta_t_us: int):
    """Window assignment via searchsorted (O(n_events)), exactly matching RVT membership.

    For each event time ``t`` the windows kept are ``{ i : t <= T_i <= t + delta_t }`` where the
    contiguous index range is ``[lo, hi]`` with
        lo = first i with T_i >= t        = searchsorted(grid, t, "left")
        hi = last  i with T_i <= t + dt   = searchsorted(grid, t + dt, "right") - 1
    Because consecutive grid steps are >= delta_t, an interval of length delta_t spans at most two
    windows, so ``hi - lo`` is 0 (single window) or 1 (shared-boundary double-count). Returns
    ``(event_index_into_batch, local_window_id)`` arrays, one row per event-window membership.
    """
    lo = np.searchsorted(local_grid, t_batch, side="left")
    hi = np.searchsorted(local_grid, t_batch + delta_t_us, side="right") - 1
    valid = (lo <= hi) & (hi >= 0)
    if valid.any():
        span = (hi[valid] - lo[valid]).max()
        if span > 1:
            # Would require >2 windows per event (grid step < delta_t); the searchsorted
            # fast path assumes <= 2. Fail loudly rather than silently drop windows.
            raise ValueError(
                f"event spans {span + 1} windows; searchsorted assignment assumes <= 2 "
                "(grid step must be >= delta_t)"
            )
    ev_idx = np.nonzero(valid)[0]
    win = lo[ev_idx].astype(np.int64)
    dup = valid & (hi == lo + 1)
    dup_idx = np.nonzero(dup)[0]
    dup_win = (lo[dup_idx] + 1).astype(np.int64)
    return (
        np.concatenate([ev_idx, dup_idx]),
        np.concatenate([win, dup_win]),
    )


def _build_index_maps(full_size: int, out_size: int) -> np.ndarray:
    """Source -> output index map of length ``full_size``; ``-1`` where dropped."""
    sel = selected_source_indices(full_size, out_size)
    index_map = np.full(full_size, -1, dtype=np.int64)
    index_map[np.asarray(sel, dtype=np.int64)] = np.arange(out_size, dtype=np.int64)
    return index_map


def _process_sequence_rust(
    in_h5: Path,
    grid: np.ndarray,
    height: int,
    width: int,
    out_h: int,
    out_w: int,
    nbins: int,
    count_cutoff: int,
    delta_t_us: int,
    downsample_by_2: bool,
    writer: H5RepresentationWriter,
    window_batch_size: int,
    dataset_group: str = "events",
    gpu: Optional[str] = None,
) -> None:
    """Rust dense scatter-add backend (CPU, or GPU when ``use_cuda``).

    Reads the raw h5 ``/{group}/{t,x,y,p}`` directly (no parquet conversion), applies the
    RVT non-decreasing time correction, computes per-window slices via ``np.searchsorted``
    on the global corrected time, and processes windows in bounded batches. Each batch
    slices the needed event range from h5py and hands it to the Rust ``stacked_histogram_dense``
    function, which returns the dense uint8 windows that are written straight to the h5.
    """
    if downsample_by_2:
        row_map = _build_index_maps(height, out_h)
        col_map = _build_index_maps(width, out_w)
    else:
        row_map = np.arange(height, dtype=np.int64)
        col_map = np.arange(width, dtype=np.int64)

    import evlib

    try:
        import hdf5plugin  # noqa: F401  (registers the blosc filter for the raw h5)
    except ImportError:
        pass
    import h5py

    with h5py.File(str(in_h5), "r") as f:
        grp = f[dataset_group]
        # Read the raw time column into a single preallocated uint32 buffer in chunks and
        # correct it to non-decreasing in place. The global searchsorted needs the whole time
        # column resident, but uint32 (the raw h5 dtype, ~2.16 GB for 540M events) holds the
        # microsecond timestamps exactly (up to ~71 minutes) and halves the footprint versus an
        # int64 copy, matching RVT's own memory profile. The small per-batch slice is cast to
        # int64 only where the Rust function needs it.
        t_ds = grp["t"]
        n_total = t_ds.shape[0]
        t_full = np.empty(n_total, dtype=np.uint32)
        chunk = 16_000_000
        for c0 in range(0, n_total, chunk):
            c1 = min(c0 + chunk, n_total)
            t_full[c0:c1] = t_ds[c0:c1]
        np.maximum.accumulate(t_full, out=t_full)
        # RVT window slices over the global, corrected, non-decreasing time array.
        starts = np.searchsorted(t_full, grid - delta_t_us, side="left")
        ends = np.searchsorted(t_full, grid, side="right")

        num_windows = len(grid)
        for a in range(0, num_windows, window_batch_size):
            b = min(a + window_batch_size - 1, num_windows - 1)
            ev_lo = int(starts[a])
            ev_hi = int(ends[b])
            if ev_hi <= ev_lo:
                continue
            t_batch = np.asarray(t_full[ev_lo:ev_hi], dtype=np.int64)
            # The GPU kernels (CUDA/Metal) take x/y/p as int32 (their native h5 dtype) to halve the
            # host->device transfer; the CPU Rust kernel takes int64.
            coord_dt = np.int32 if gpu else np.int64
            x_batch = np.asarray(grp["x"][ev_lo:ev_hi], dtype=coord_dt)
            y_batch = np.asarray(grp["y"][ev_lo:ev_hi], dtype=coord_dt)
            p_batch = np.asarray(grp["p"][ev_lo:ev_hi], dtype=coord_dt)

            if gpu == "cuda":
                dense_fn = evlib.representations_rs.stacked_histogram_dense_cuda
            elif gpu == "metal":
                dense_fn = evlib.representations_rs.stacked_histogram_dense_metal
            else:
                dense_fn = evlib.representations_rs.stacked_histogram_dense
            dense = dense_fn(
                t_batch,
                x_batch,
                y_batch,
                p_batch,
                np.asarray(grid[a : b + 1], dtype=np.int64),
                delta_t_us,
                nbins,
                count_cutoff,
                row_map,
                col_map,
                out_h,
                out_w,
            )
            for local in range(b - a + 1):
                writer.write_window(a + local, dense[local])


def _process_sequence_polars(
    in_h5: Path,
    grid: np.ndarray,
    height: int,
    width: int,
    out_h: int,
    out_w: int,
    channels: int,
    nbins: int,
    count_cutoff: int,
    delta_t_us: int,
    downsample_by_2: bool,
    writer: H5RepresentationWriter,
    window_batch_size: int,
    engine: str,
    dataset_group: str = "events",
) -> None:
    """Polars/GPU backend: searchsorted window assignment + one large group_by per batch.

    Reads the raw h5 directly (no parquet round-trip), corrects time to non-decreasing, and for
    each batch of ``window_batch_size`` windows assigns events to windows with ``np.searchsorted``
    (instead of a cross-join, which would blow up at large batch sizes). The tagged events are
    aggregated in a single ``build_sparse_histogram_assigned`` collect that runs entirely on the
    selected engine (the cudf-polars GPU engine when ``engine`` is "gpu"/a GPUEngine), then the
    sparse per-window counts are scattered to dense and written. Larger batches mean far fewer,
    larger GPU collects than the old per-10-window cross-join path.
    """
    try:
        import hdf5plugin  # noqa: F401  (registers the blosc filter for the raw h5)
    except ImportError:
        pass
    import h5py

    with h5py.File(str(in_h5), "r") as f:
        grp = f[dataset_group]
        t_ds = grp["t"]
        n_total = t_ds.shape[0]
        t_full = np.empty(n_total, dtype=np.uint32)
        chunk = 16_000_000
        for c0 in range(0, n_total, chunk):
            c1 = min(c0 + chunk, n_total)
            t_full[c0:c1] = t_ds[c0:c1]
        np.maximum.accumulate(t_full, out=t_full)
        starts = np.searchsorted(t_full, grid - delta_t_us, side="left")
        ends = np.searchsorted(t_full, grid, side="right")

        num_windows = len(grid)
        for a in range(0, num_windows, window_batch_size):
            b = min(a + window_batch_size - 1, num_windows - 1)
            ev_lo = int(starts[a])
            ev_hi = int(ends[b])
            if ev_hi <= ev_lo:
                continue
            t_batch = np.asarray(t_full[ev_lo:ev_hi], dtype=np.int64)
            local_grid = np.asarray(grid[a : b + 1], dtype=np.int64)
            ev_idx, win_local = _assign_window_ids(t_batch, local_grid, delta_t_us)
            if ev_idx.size == 0:
                continue
            x_batch = np.asarray(grp["x"][ev_lo:ev_hi], dtype=np.int64)
            y_batch = np.asarray(grp["y"][ev_lo:ev_hi], dtype=np.int64)
            p_batch = np.clip(
                np.asarray(grp["p"][ev_lo:ev_hi], dtype=np.int64), 0, None
            )
            df = pl.DataFrame(
                {
                    "t": t_batch[ev_idx],
                    "x": x_batch[ev_idx],
                    "y": y_batch[ev_idx],
                    "p": p_batch[ev_idx],
                    "window_id": win_local,
                }
            )
            sparse = build_sparse_histogram_assigned(
                df,
                delta_t_us=delta_t_us,
                nbins=nbins,
                count_cutoff=count_cutoff,
                height=height,
                width=width,
                downsample_by_2=downsample_by_2,
                engine=engine,
            )
            if sparse.height:
                parts = sparse.partition_by("window_id", as_dict=True)
                for k, wdf in parts.items():
                    local = k[0] if isinstance(k, tuple) else k
                    dense = scatter_window_dense(wdf, channels, out_h, out_w)
                    writer.write_window(a + int(local), dense)


def process_sequence(
    in_h5: Path,
    out_dir: Path,
    dataset: str,
    height: int,
    width: int,
    ev_repr_timestamps_us: np.ndarray,
    downsample_by_2: bool,
    nbins: int = 10,
    count_cutoff: int = 10,
    delta_t_us: int = 50_000,
    engine: str = "auto",
    backend: str = "polars",
    labels_npy: Optional[Path] = None,
    split: str = "val",
    window_batch_size: int = 10,
    polars_batch_windows: int = 16,
    cuda_batch_windows: int = 128,
) -> Path:
    if backend not in ("polars", "rust", "cuda", "metal"):
        raise ValueError(
            f"backend must be 'polars', 'rust', 'cuda' or 'metal', got {backend!r}"
        )
    out_dir = Path(out_dir)
    repr_dir = out_dir / "event_representations_v2" / REPR_NAME
    repr_dir.mkdir(parents=True, exist_ok=True)

    grid = np.asarray(ev_repr_timestamps_us, dtype=np.int64)
    num_windows = len(grid)
    out_h, out_w = (height // 2, width // 2) if downsample_by_2 else (height, width)
    channels = 2 * nbins
    suffix = "_ds2_nearest" if downsample_by_2 else ""
    out_h5 = repr_dir / f"event_representations{suffix}.h5"

    if backend in ("rust", "cuda", "metal"):
        with H5RepresentationWriter(
            out_h5,
            num_windows=num_windows,
            channels=channels,
            height=out_h,
            width=out_w,
        ) as writer:
            _process_sequence_rust(
                in_h5,
                grid,
                height=height,
                width=width,
                out_h=out_h,
                out_w=out_w,
                nbins=nbins,
                count_cutoff=count_cutoff,
                delta_t_us=delta_t_us,
                downsample_by_2=downsample_by_2,
                writer=writer,
                # The CUDA scatter does one big launch per batch, so use large window batches to
                # amortise the host<->device transfer and kernel launch over many windows (the CPU
                # Rust backend stays at its small default).
                window_batch_size=(
                    cuda_batch_windows if backend in ("cuda", "metal") else window_batch_size
                ),
                gpu=(backend if backend in ("cuda", "metal") else None),
            )
        np.save(
            str(repr_dir / "timestamps_us.npy"),
            np.asarray(ev_repr_timestamps_us, dtype=np.int64),
        )
        _write_labels(
            labels_npy, split, dataset, repr_dir, out_dir, ev_repr_timestamps_us
        )
        return out_h5

    # Polars/GPU backend: read the raw h5 directly (no parquet round-trip), assign windows with
    # searchsorted, and aggregate each large batch of windows in a single engine collect.
    with H5RepresentationWriter(
        out_h5,
        num_windows=num_windows,
        channels=channels,
        height=out_h,
        width=out_w,
    ) as writer:
        _process_sequence_polars(
            in_h5,
            grid,
            height=height,
            width=width,
            out_h=out_h,
            out_w=out_w,
            channels=channels,
            nbins=nbins,
            count_cutoff=count_cutoff,
            delta_t_us=delta_t_us,
            downsample_by_2=downsample_by_2,
            writer=writer,
            window_batch_size=polars_batch_windows,
            engine=engine,
        )
    np.save(
        str(repr_dir / "timestamps_us.npy"),
        np.asarray(ev_repr_timestamps_us, dtype=np.int64),
    )

    _write_labels(labels_npy, split, dataset, repr_dir, out_dir, ev_repr_timestamps_us)
    return out_h5


def _write_labels(
    labels_npy: Optional[Path],
    split: str,
    dataset: str,
    repr_dir: Path,
    out_dir: Path,
    ev_repr_timestamps_us: np.ndarray,
) -> None:
    if labels_npy is None:
        return
    try:
        from evlib.rvt.labels import build_timeline  # deferred module; optional
    except ImportError:
        return
    tl = build_timeline(labels_npy, split=split, dataset=dataset)
    np.save(
        str(repr_dir / "objframe_idx_2_repr_idx.npy"),
        tl.objframe_idx_2_repr_idx,
    )
    labels_dir = out_dir / "labels_v2"
    labels_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(labels_dir / "labels.npz"),
        labels=tl.labels_v2,
        objframe_idx_2_label_idx=tl.objframe_idx_2_label_idx,
    )
    np.save(str(labels_dir / "timestamps_us.npy"), tl.frame_timestamps_us)


def main(argv: Optional[list] = None) -> int:
    """CLI entry point: preprocess one raw event sequence into the RVT layout.

    Requires one of --grid-npy (a precomputed ev_repr_timestamps_us .npy) or
    --labels-npy. If only --labels-npy is given and evlib.rvt.labels is importable,
    the grid is derived from it; the labels file is also passed through to
    process_sequence, which guards its own (deferred) label handling.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="evlib-rvt-preprocess",
        description="Preprocess a raw event sequence into the RVT stacked-histogram layout.",
    )
    parser.add_argument("--in-h5", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--dataset", default="gen4", choices=["gen1", "gen4"])
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument(
        "--grid-npy",
        type=Path,
        default=None,
        help="Path to a precomputed ev_repr_timestamps_us .npy file.",
    )
    parser.add_argument(
        "--labels-npy",
        type=Path,
        default=None,
        help="Optional labels .npy; used to derive the grid when evlib.rvt.labels "
        "is available, and passed through to process_sequence.",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--no-downsample", action="store_true")
    parser.add_argument("--engine", default="auto")
    args = parser.parse_args(argv)

    if args.grid_npy is None and args.labels_npy is None:
        raise SystemExit("one of --grid-npy or --labels-npy is required")

    grid = None
    if args.grid_npy is not None:
        grid = np.load(args.grid_npy)
    elif args.labels_npy is not None:
        try:
            from evlib.rvt.labels import build_timeline  # deferred module; optional
        except ImportError:
            build_timeline = None
        if build_timeline is not None:
            tl = build_timeline(args.labels_npy, split=args.split, dataset=args.dataset)
            grid = np.asarray(tl.frame_timestamps_us, dtype=np.int64)

    if grid is None:
        raise SystemExit("one of --grid-npy or --labels-npy is required")

    process_sequence(
        args.in_h5,
        args.out_dir,
        dataset=args.dataset,
        height=args.height,
        width=args.width,
        ev_repr_timestamps_us=grid,
        downsample_by_2=not args.no_downsample,
        engine=args.engine,
        labels_npy=args.labels_npy,
        split=args.split,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
