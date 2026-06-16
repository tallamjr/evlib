"""Orchestrate the RVT-identical preprocessing for one sequence."""

from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

from evlib.rvt.events import convert_h5_to_parquet
from evlib.rvt.representation import build_sparse_histogram
from evlib.rvt.writer import H5RepresentationWriter, scatter_window_dense

REPR_NAME = "stacked_histogram_dt50_nbins10"


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
    labels_npy: Optional[Path] = None,
    split: str = "val",
    tmp_parquet: Optional[Path] = None,
    window_batch_size: int = 10,
) -> Path:
    out_dir = Path(out_dir)
    repr_dir = out_dir / "event_representations_v2" / REPR_NAME
    repr_dir.mkdir(parents=True, exist_ok=True)

    pq = Path(tmp_parquet) if tmp_parquet else repr_dir / "_events.parquet"
    convert_h5_to_parquet(in_h5, pq)

    grid = np.asarray(ev_repr_timestamps_us, dtype=np.int64)
    num_windows = len(grid)
    out_h, out_w = (height // 2, width // 2) if downsample_by_2 else (height, width)
    channels = 2 * nbins
    suffix = "_ds2_nearest" if downsample_by_2 else ""
    out_h5 = repr_dir / f"event_representations{suffix}.h5"

    # Window-batched processing for bounded memory. Each batch covers global window
    # indices [a, b]. RVT assigns every event with T_i - delta_t <= t <= T_i to window i,
    # so the events needed for windows [a, b] are exactly those with
    #   grid[a] - delta_t_us <= t <= grid[b].
    # Predicate pushdown on the sorted-by-t parquet keeps each batch read bounded.
    # An event sitting on a shared boundary at grid[b] is also read by the next batch
    # (its range starts at grid[b+1] - delta_t_us == grid[b] when the step == delta_t),
    # which reproduces RVT's boundary double-count exactly.
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
            batch_events = pl.scan_parquet(str(pq)).filter(
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
                engine=engine,
            )
            if sparse.height:
                parts = sparse.partition_by("window_id", as_dict=True)
                for k, wdf in parts.items():
                    local = k[0] if isinstance(k, tuple) else k
                    dense = scatter_window_dense(wdf, channels, out_h, out_w)
                    writer.write_window(a + int(local), dense)
    np.save(
        str(repr_dir / "timestamps_us.npy"),
        np.asarray(ev_repr_timestamps_us, dtype=np.int64),
    )

    if labels_npy is not None:
        try:
            from evlib.rvt.labels import build_timeline  # deferred module; optional
        except ImportError:
            build_timeline = None
        if build_timeline is not None:
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

    pq.unlink(missing_ok=True)
    return out_h5
