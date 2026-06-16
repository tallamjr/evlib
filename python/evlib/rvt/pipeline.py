"""Orchestrate the RVT-identical preprocessing for one sequence."""

from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

from evlib.rvt.events import convert_h5_to_parquet
from evlib.rvt.representation import build_sparse_histogram
from evlib.rvt.writer import write_event_representation_h5

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
) -> Path:
    out_dir = Path(out_dir)
    repr_dir = out_dir / "event_representations_v2" / REPR_NAME
    repr_dir.mkdir(parents=True, exist_ok=True)

    pq = Path(tmp_parquet) if tmp_parquet else repr_dir / "_events.parquet"
    convert_h5_to_parquet(in_h5, pq)
    events = pl.scan_parquet(str(pq))

    sparse = build_sparse_histogram(
        events,
        ev_repr_timestamps_us=ev_repr_timestamps_us,
        delta_t_us=delta_t_us,
        nbins=nbins,
        count_cutoff=count_cutoff,
        height=height,
        width=width,
        downsample_by_2=downsample_by_2,
        engine=engine,
    )
    out_h, out_w = (height // 2, width // 2) if downsample_by_2 else (height, width)
    suffix = "_ds2_nearest" if downsample_by_2 else ""
    out_h5 = repr_dir / f"event_representations{suffix}.h5"
    write_event_representation_h5(
        out_h5,
        sparse,
        num_windows=len(ev_repr_timestamps_us),
        channels=2 * nbins,
        height=out_h,
        width=out_w,
    )
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
