"""Window sources behind the dataset seam. PreprocessedH5Source reads RVT .h5 layout."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Protocol, Tuple

import numpy as np
import torch

import evlib
from evlib.data.labels import boxes_to_yolox
from evlib.rvt.pipeline import _build_index_maps

REPR_NAME = "stacked_histogram_dt50_nbins10"


def _import_h5():
    """Import h5py, first registering the optional blosc filter if available.

    ``hdf5plugin`` is an optional dependency that registers the blosc HDF5
    filter on import; its absence is the one legitimate optional-import guard.
    """
    try:
        import hdf5plugin  # noqa: F401  registers the blosc filter
    except ImportError:
        pass
    import h5py

    return h5py


class ReprSource(Protocol):
    def window_count(self) -> int: ...
    def read_windows(
        self, lo: int, hi: int
    ) -> Tuple[List[torch.Tensor], List[Optional[torch.Tensor]]]: ...


def _nbins_from_repr_name(repr_name: str) -> int:
    m = re.search(r"nbins(\d+)", repr_name)
    if not m:
        raise ValueError(f"cannot parse nbins from repr_name {repr_name!r}")
    return int(m.group(1))


class PreprocessedH5Source:
    """Reads one preprocessed RVT sequence directory. Opens the h5 lazily (fork-safe)."""

    def __init__(
        self, seq_dir, repr_name: str = REPR_NAME, downsample_by_2: bool = True
    ) -> None:
        self.seq_dir = Path(seq_dir)
        self.repr_name = repr_name
        self.downsample_by_2 = downsample_by_2
        self.nbins = _nbins_from_repr_name(repr_name)
        # Live handles are created lazily and never persist across pickling so
        # that DataLoader workers (fork or spawn) each open their own h5 file.
        self._h5 = None
        self._data = None
        # Cheap, picklable metadata loaded once via _ensure_meta().
        self._n_windows = None
        self._labels = None
        self._objframe_2_repr = None
        self._repr_2_objframe = None

    @property
    def _repr_dir(self) -> Path:
        return self.seq_dir / "event_representations_v2" / self.repr_name

    @property
    def _h5_path(self) -> Path:
        name = (
            "event_representations_ds2_nearest.h5"
            if self.downsample_by_2
            else "event_representations.h5"
        )
        return self._repr_dir / name

    def _ensure_meta(self) -> None:
        """Load cheap metadata and labels, holding no persistent h5 handle.

        Opens the h5 only long enough to read ``data.shape`` (validating the
        channel count and recording the window count), then closes it. The numpy
        label arrays are picklable, so the source stays picklable under spawn.
        """
        if self._n_windows is not None:
            return
        h5py = _import_h5()

        for p in (
            self._h5_path,
            self._repr_dir / "objframe_idx_2_repr_idx.npy",
            self.seq_dir / "labels_v2" / "labels.npz",
        ):
            if not p.exists():
                raise FileNotFoundError(
                    f"preprocessed sequence missing required file: {p}"
                )

        with h5py.File(str(self._h5_path), "r") as h5:
            shape = h5["data"].shape
        expected_c = 2 * self.nbins
        if shape[1] != expected_c:
            raise ValueError(
                f"on-disk channel count {shape[1]} != 2*nbins {expected_c} at {self._h5_path}"
            )
        self._n_windows = int(shape[0])

        self._objframe_2_repr = np.load(self._repr_dir / "objframe_idx_2_repr_idx.npy")
        npz = np.load(self.seq_dir / "labels_v2" / "labels.npz")
        self._labels = npz["labels"]
        objframe_2_label = npz["objframe_idx_2_label_idx"]
        # build repr_idx -> (label_lo, label_hi) for fast per-window lookup
        self._repr_2_objframe = {}
        n_obj = len(self._objframe_2_repr)
        for obj_i in range(n_obj):
            repr_i = int(self._objframe_2_repr[obj_i])
            lo = int(objframe_2_label[obj_i])
            hi = (
                int(objframe_2_label[obj_i + 1])
                if obj_i + 1 < len(objframe_2_label)
                else len(self._labels)
            )
            self._repr_2_objframe[repr_i] = (lo, hi)

    def _ensure_data(self) -> None:
        """Open and keep the ``data`` dataset handle for reading.

        Called only from read_windows, so the persistent (fork-unsafe) handle is
        created in the process that actually reads, post-fork or post-unpickle.
        """
        self._ensure_meta()
        if self._data is not None:
            return
        h5py = _import_h5()

        self._h5 = h5py.File(str(self._h5_path), "r")
        self._data = self._h5["data"]

    def __getstate__(self) -> dict:
        # Drop the live h5/data handles so the source pickles cleanly under
        # spawn; they are re-opened lazily in the unpickling process.
        state = self.__dict__.copy()
        state["_h5"] = None
        state["_data"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._h5 = None
        self._data = None

    def window_count(self) -> int:
        self._ensure_meta()
        return int(self._n_windows)

    def read_windows(self, lo: int, hi: int):
        self._ensure_data()
        if lo < 0 or hi > self._data.shape[0] or lo >= hi:
            raise ValueError(
                f"window range [{lo},{hi}) out of bounds for {self._data.shape[0]} windows"
            )
        block = np.asarray(self._data[lo:hi])  # [hi-lo, C, H, W] uint8
        ev = [
            torch.from_numpy(np.ascontiguousarray(block[i]))
            for i in range(block.shape[0])
        ]
        labels: List[Optional[torch.Tensor]] = []
        for repr_i in range(lo, hi):
            span = self._repr_2_objframe.get(repr_i)
            if span is None:
                labels.append(None)
            else:
                l0, l1 = span
                labels.append(boxes_to_yolox(self._labels[l0:l1]))
        return ev, labels


class EvlibStreamSource:
    """Build dense ``[C, H, W]`` uint8 windows on the fly from a raw event h5.

    Reuses ``evlib.rvt``'s window assignment (the same global non-decreasing time
    correction and searchsorted window slices as
    ``evlib.rvt.pipeline._process_sequence_rust``) and the Rust
    ``evlib.representations_rs.stacked_histogram_dense`` densify kernel, instead of
    reading the precomputed representation h5. The window-end grid and the labels
    are taken from the same on-disk processed sequence directory (labels are read
    via ``PreprocessedH5Source``; phase 1 does not synthesise labels).
    """

    def __init__(
        self,
        raw_h5,
        seq_dir,
        repr_name: str = REPR_NAME,
        downsample_by_2: bool = True,
        nbins: int = 10,
        count_cutoff: int = 10,
        delta_t_us: int = 50_000,
        height: int = 720,
        width: int = 1280,
        gpu: Optional[str] = None,
        dataset_group: str = "events",
    ) -> None:
        parsed_nbins = _nbins_from_repr_name(repr_name)
        if parsed_nbins != nbins:
            raise ValueError(
                f"nbins mismatch: explicit nbins={nbins} disagrees with nbins "
                f"parsed from repr_name {repr_name!r} (={parsed_nbins}); the "
                "windows would not match the on-disk labels grid"
            )
        self.raw_h5 = Path(raw_h5)
        self.label_source = PreprocessedH5Source(seq_dir, repr_name, downsample_by_2)
        self.repr_name = repr_name
        self.downsample_by_2 = downsample_by_2
        self.nbins = nbins
        self.count_cutoff = count_cutoff
        self.delta_t_us = delta_t_us
        self.height = height
        self.width = width
        self.gpu = gpu
        self.dataset_group = dataset_group
        self.engine_used = (
            "cuda" if gpu == "cuda" else "metal" if gpu == "metal" else "rust-cpu"
        )
        # The window-end grid is the same one process_sequence wrote next to the
        # representation h5; loaded lazily so construction is cheap and picklable.
        self._grid = None
        # The corrected global time array and its derived window-start/-end index
        # arrays are built once per worker (see _ensure_time) and never pickled:
        # _t_full is ~2 GB on a real gen4 sequence, so re-reading and re-correcting
        # it on every read_windows call is impractical in a training loop, and
        # carrying it through pickle would break fork/spawn DataLoader workers.
        self._t_full = None
        self._starts = None
        self._ends = None

    def _ensure_grid(self) -> None:
        if self._grid is None:
            grid_path = self.label_source._repr_dir / "timestamps_us.npy"
            self._grid = np.load(grid_path).astype(np.int64)

    def _ensure_time(self) -> None:
        """Build and cache the corrected global time and window index arrays.

        Reads the whole raw uint32 time column once, corrects it to non-decreasing
        in place (exactly as ``_process_sequence_rust`` does), then searchsorts the
        window-start/-end indices over the grid. All three arrays are cached so
        subsequent ``read_windows`` calls reuse them rather than re-reading ~2 GB.
        Requires ``_ensure_grid`` to have run, as the index arrays need the grid.
        """
        if self._t_full is not None:
            return
        self._ensure_grid()
        h5py = _import_h5()
        with h5py.File(str(self.raw_h5), "r") as f:
            t_ds = f[self.dataset_group]["t"]
            n_total = t_ds.shape[0]
            t_full = np.empty(n_total, dtype=np.uint32)
            chunk = 16_000_000
            for c0 in range(0, n_total, chunk):
                c1 = min(c0 + chunk, n_total)
                t_full[c0:c1] = t_ds[c0:c1]
        np.maximum.accumulate(t_full, out=t_full)
        self._t_full = t_full
        self._starts = np.searchsorted(
            t_full, self._grid - self.delta_t_us, side="left"
        )
        self._ends = np.searchsorted(t_full, self._grid, side="right")

    def __getstate__(self) -> dict:
        # Drop the large corrected-time array and its derived index arrays so the
        # source pickles cheaply; each spawned/forked worker rebuilds its own via
        # _ensure_time. label_source and the small _grid carry through intact.
        state = self.__dict__.copy()
        state["_t_full"] = None
        state["_starts"] = None
        state["_ends"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._t_full = None
        self._starts = None
        self._ends = None

    def window_count(self) -> int:
        self._ensure_grid()
        return len(self._grid)

    def read_windows(self, lo: int, hi: int):
        self._ensure_grid()
        self._ensure_time()
        if lo < 0 or hi > len(self._grid) or lo >= hi:
            raise ValueError(
                f"window range [{lo},{hi}) out of bounds for {len(self._grid)} windows"
            )

        out_h, out_w = (
            (self.height // 2, self.width // 2)
            if self.downsample_by_2
            else (self.height, self.width)
        )
        if self.downsample_by_2:
            row_map = _build_index_maps(self.height, out_h)
            col_map = _build_index_maps(self.width, out_w)
        else:
            row_map = np.arange(self.height, dtype=np.int64)
            col_map = np.arange(self.width, dtype=np.int64)

        h5py = _import_h5()

        grid = self._grid
        # starts/ends/_t_full are built once by _ensure_time and reused here.
        ev_lo = int(self._starts[lo])
        ev_hi = int(self._ends[hi - 1])
        n_windows = hi - lo
        if ev_hi <= ev_lo:
            # No events cover any requested window; emit empty dense windows.
            channels = 2 * self.nbins
            ev = [
                torch.zeros((channels, out_h, out_w), dtype=torch.uint8)
                for _ in range(n_windows)
            ]
            _, labels = self.label_source.read_windows(lo, hi)
            return ev, labels

        t_batch = np.asarray(self._t_full[ev_lo:ev_hi], dtype=np.int64)
        coord_dt = np.int32 if self.gpu else np.int64
        with h5py.File(str(self.raw_h5), "r") as f:
            grp = f[self.dataset_group]
            # Only the small per-batch x/y/p slices are read per call (cheap); the
            # global corrected time is already cached in self._t_full.
            x_batch = np.asarray(grp["x"][ev_lo:ev_hi], dtype=coord_dt)
            y_batch = np.asarray(grp["y"][ev_lo:ev_hi], dtype=coord_dt)
            p_batch = np.asarray(grp["p"][ev_lo:ev_hi], dtype=coord_dt)

        if self.gpu == "cuda":
            dense_fn = evlib.representations_rs.stacked_histogram_dense_cuda
        elif self.gpu == "metal":
            dense_fn = evlib.representations_rs.stacked_histogram_dense_metal
        else:
            dense_fn = evlib.representations_rs.stacked_histogram_dense
        dense = dense_fn(
            t_batch,
            x_batch,
            y_batch,
            p_batch,
            np.asarray(grid[lo:hi], dtype=np.int64),
            self.delta_t_us,
            self.nbins,
            self.count_cutoff,
            row_map,
            col_map,
            out_h,
            out_w,
        )
        ev = [
            torch.from_numpy(np.ascontiguousarray(dense[k])) for k in range(n_windows)
        ]
        _, labels = self.label_source.read_windows(lo, hi)
        return ev, labels
