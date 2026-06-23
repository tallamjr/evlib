"""Window sources behind the dataset seam. PreprocessedH5Source reads RVT .h5 layout."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Protocol, Tuple

import numpy as np
import torch

from evlib.data.labels import boxes_to_yolox

REPR_NAME = "stacked_histogram_dt50_nbins10"


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
        try:
            import hdf5plugin  # noqa: F401  registers the blosc filter
        except ImportError:
            pass
        import h5py

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
        try:
            import hdf5plugin  # noqa: F401  registers the blosc filter
        except ImportError:
            pass
        import h5py

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
