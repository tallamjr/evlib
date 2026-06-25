"""Generate a tiny 2-window RVT-layout sequence under tests/data_fixtures/short_seq.

Deliberately shorter than mini_seq (2 windows vs 6) so a streaming dataset can
exhaust one slot before another and exercise order-independent slot padding.
Real arrays only, a few KB, tracked in the repo.
"""

from pathlib import Path

import h5py
import hdf5plugin
import numpy as np

REPR_NAME = "stacked_histogram_dt50_nbins10"
N, C, H, W = 2, 20, 8, 12  # 2 windows, C=2*nbins(=10), tiny 8x12 sensor


def main() -> None:
    root = Path(__file__).resolve().parent / "short_seq"
    repr_dir = root / "event_representations_v2" / REPR_NAME
    lab_dir = root / "labels_v2"
    repr_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(1)
    data = rng.integers(0, 4, size=(N, C, H, W), dtype=np.uint8)
    with h5py.File(repr_dir / "event_representations_ds2_nearest.h5", "w") as f:
        f.create_dataset("data", data=data, **hdf5plugin.Blosc())
    np.save(repr_dir / "timestamps_us.npy", (np.arange(N) * 50_000).astype(np.int64))

    # one object frame, mapped to repr index 1
    objframe_idx_2_repr_idx = np.array([1], dtype=np.int64)
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
    labels = np.zeros(1, dtype=np.dtype(fields))
    # Stored at FULL resolution (2x ds2); the loader scales by 0.5 so the loaded
    # box is the ds2-space (1,1,2,2) the tests expect.
    labels[0] = (0, 2, 2, 4, 4, 0, 1.0)
    objframe_idx_2_label_idx = np.array([0], dtype=np.int64)
    np.savez(
        lab_dir / "labels.npz",
        labels=labels,
        objframe_idx_2_label_idx=objframe_idx_2_label_idx,
    )
    np.save(lab_dir / "timestamps_us.npy", (np.array([1]) * 50_000).astype(np.int64))
    print(f"wrote fixture to {root}")


if __name__ == "__main__":
    main()
