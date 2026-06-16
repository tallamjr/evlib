"""Dense scatter and RVT-layout HDF5 writer."""

from pathlib import Path

import numpy as np
import polars as pl

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass
import h5py


def _blosc_opts(complevel: int = 1, shuffle: str = "byte", complib: str = "blosc:zstd"):
    shuffle_code = 2 if shuffle == "bit" else 1 if shuffle == "byte" else 0
    compressors = ["blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"]
    complib_code = ["blosc:" + c for c in compressors].index(complib)
    args = {
        "compression": 32001,
        "compression_opts": (0, 0, 0, 0, complevel, shuffle_code, complib_code),
    }
    if shuffle_code > 0:
        args["shuffle"] = False
    return args


def scatter_window_dense(
    window_sparse: pl.DataFrame, channels: int, height: int, width: int
) -> np.ndarray:
    dense = np.zeros((channels, height, width), dtype=np.uint8)
    if window_sparse.height == 0:
        return dense
    c = window_sparse["channel"].to_numpy()
    y = window_sparse["y"].to_numpy()
    x = window_sparse["x"].to_numpy()
    v = window_sparse["count"].to_numpy().astype(np.uint8)
    dense[c, y, x] = v
    return dense


class H5RepresentationWriter:
    """Incremental writer for the RVT-layout event representation HDF5.

    Creates the ``data`` dataset once with the same shape, chunking and blosc-zstd
    compression as :func:`write_event_representation_h5`, then fills individual
    window slices via :meth:`write_window`. Use as a context manager (or call
    :meth:`close`) to ensure the file is flushed.
    """

    def __init__(
        self,
        out_path: Path,
        num_windows: int,
        channels: int,
        height: int,
        width: int,
    ) -> None:
        self.channels = channels
        self.height = height
        self.width = width
        self.num_windows = num_windows
        shape = (channels, height, width)
        self._h5f = h5py.File(str(out_path), "w")
        self._dset = self._h5f.create_dataset(
            "data",
            dtype="uint8",
            shape=(num_windows,) + shape,
            chunks=(1,) + shape,
            maxshape=(None,) + shape,
            **_blosc_opts(),
        )

    def write_window(self, global_window_index: int, dense_array: np.ndarray) -> None:
        self._dset[global_window_index] = dense_array

    def close(self) -> None:
        if self._h5f is not None:
            self._h5f.close()
            self._h5f = None

    def __enter__(self) -> "H5RepresentationWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def write_event_representation_h5(
    out_path: Path,
    sparse: pl.DataFrame,
    num_windows: int,
    channels: int,
    height: int,
    width: int,
) -> None:
    shape = (channels, height, width)
    with h5py.File(str(out_path), "w") as h5f:
        dset = h5f.create_dataset(
            "data",
            dtype="uint8",
            shape=(num_windows,) + shape,
            chunks=(1,) + shape,
            maxshape=(None,) + shape,
            **_blosc_opts(),
        )
        parts = sparse.partition_by("window_id", as_dict=True) if sparse.height else {}
        norm = {}
        for k, v in parts.items():
            wid = k[0] if isinstance(k, tuple) else k
            norm[int(wid)] = v
        for w in range(num_windows):
            wdf = norm.get(w)
            if wdf is None:
                continue
            dset[w] = scatter_window_dense(wdf, channels, height, width)
