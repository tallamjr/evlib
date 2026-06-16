"""Convert raw Prophesee/RVT event HDF5 to Parquet for streaming, with RVT time correction."""

from pathlib import Path
from typing import Optional
import h5py
import numpy as np
import polars as pl
import pyarrow.parquet as pq

try:
    import hdf5plugin  # noqa: F401  (registers the blosc filter needed to read the raw h5)
except ImportError:
    pass


def correct_time_nondecreasing(t: np.ndarray, prev_max: int = 0) -> np.ndarray:
    """Vectorized equivalent of RVT H5Reader._correct_time: enforce running maximum.

    RVT clamps each timestamp up to the max seen so far so the stream is non-decreasing.
    np.maximum.accumulate is exactly that, with prev_max carrying across chunks.
    """
    assert t[0] >= 0
    out = np.maximum.accumulate(t)
    if prev_max:
        out = np.maximum(out, prev_max)
    return out


def convert_h5_to_parquet(
    in_h5: Path,
    out_parquet: Path,
    dataset_group: str = "events",
    chunk_rows: int = 16_000_000,
    max_rows: Optional[int] = None,
) -> int:
    """Stream the raw h5 in row chunks, correct time, write a single Parquet file.

    Returns the number of rows written.
    """
    with h5py.File(str(in_h5), "r") as f:
        grp = f[dataset_group]
        n_total = grp["t"].shape[0]
        n = n_total if max_rows is None else min(max_rows, n_total)
        writer = None
        prev_max = 0
        written = 0
        try:
            for start in range(0, n, chunk_rows):
                end = min(start + chunk_rows, n)
                t = np.asarray(grp["t"][start:end], dtype=np.int64)
                t = correct_time_nondecreasing(t, prev_max=prev_max)
                prev_max = int(t[-1])
                x = np.asarray(grp["x"][start:end], dtype=np.int16)
                y = np.asarray(grp["y"][start:end], dtype=np.int16)
                p = np.clip(np.asarray(grp["p"][start:end], dtype=np.int8), 0, None)
                table = pl.DataFrame({"t": t, "x": x, "y": y, "p": p}).to_arrow()
                if writer is None:
                    writer = pq.ParquetWriter(str(out_parquet), table.schema)
                writer.write_table(table)
                written += end - start
        finally:
            if writer is not None:
                writer.close()
    return written
