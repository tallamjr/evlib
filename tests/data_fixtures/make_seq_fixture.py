"""Generate a tiny RVT-layout preprocessed sequence under tests/data_fixtures/mini_seq."""

from pathlib import Path

import h5py
import hdf5plugin
import numpy as np

REPR_NAME = "stacked_histogram_dt50_nbins10"
N, C, H, W = 6, 20, 8, 12  # 6 windows, C=2*nbins(=10), tiny 8x12 sensor


def main() -> None:
    root = Path(__file__).resolve().parent / "mini_seq"
    repr_dir = root / "event_representations_v2" / REPR_NAME
    lab_dir = root / "labels_v2"
    repr_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    data = rng.integers(0, 4, size=(N, C, H, W), dtype=np.uint8)
    with h5py.File(repr_dir / "event_representations_ds2_nearest.h5", "w") as f:
        f.create_dataset("data", data=data, **hdf5plugin.Blosc())
    np.save(repr_dir / "timestamps_us.npy", (np.arange(N) * 50_000).astype(np.int64))

    # two object frames, mapped to repr indices 1 and 4
    objframe_idx_2_repr_idx = np.array([1, 4], dtype=np.int64)
    np.save(repr_dir / "objframe_idx_2_repr_idx.npy", objframe_idx_2_repr_idx)

    fields = [
        ("t", "<f4"),
        ("x", "<f4"),
        ("y", "<f4"),
        ("w", "<f4"),
        ("h", "<f4"),
        ("class_id", "<u4"),
        ("class_confidence", "<f4"),
    ]
    labels = np.zeros(3, dtype=np.dtype(fields))
    # Boxes are stored at FULL resolution (2x the ds2 representation), matching
    # the real RVT labels.npz contract: the loader scales them to ds2 space by
    # 0.5, so the loaded box values are the ds2-space (x,y,w,h) the tests expect.
    # objframe 0 -> 1 box ; objframe 1 -> 2 boxes (ds2 targets: (1,1,2,2),
    # (3,3,2,2), (5,1,1,1)).
    labels[0] = (0, 2, 2, 4, 4, 0, 1.0)
    labels[1] = (0, 6, 6, 4, 4, 1, 1.0)
    labels[2] = (0, 10, 2, 2, 2, 0, 1.0)
    objframe_idx_2_label_idx = np.array(
        [0, 1], dtype=np.int64
    )  # frame0 -> [0:1], frame1 -> [1:3]
    np.savez(
        lab_dir / "labels.npz",
        labels=labels,
        objframe_idx_2_label_idx=objframe_idx_2_label_idx,
    )
    np.save(lab_dir / "timestamps_us.npy", (np.array([1, 4]) * 50_000).astype(np.int64))
    print(f"wrote fixture to {root}")


if __name__ == "__main__":
    main()
