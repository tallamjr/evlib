from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = (
    ROOT
    / "data/gen4_1mpx_processed_RVT/val/moorea_2019-02-21_000_td_2257500000_2317500000"
)
RAW = (
    ROOT
    / "data/gen4_1mpx_original/val/moorea_2019-02-21_000_td_2257500000_2317500000_td.h5"
)
REPR_SUBDIR = "event_representations_v2/stacked_histogram_dt50_nbins10"


def raw_input_path() -> Path:
    return RAW


def ref_repr_dir() -> Path:
    return REF / REPR_SUBDIR


def ref_repr_h5() -> Path:
    return ref_repr_dir() / "event_representations_ds2_nearest.h5"


def ref_timestamps() -> Path:
    return ref_repr_dir() / "timestamps_us.npy"


def ref_objframe_idx() -> Path:
    return ref_repr_dir() / "objframe_idx_2_repr_idx.npy"


def ref_labels_npz() -> Path:
    return REF / "labels_v2/labels.npz"


def ref_labels_timestamps() -> Path:
    return REF / "labels_v2/timestamps_us.npy"
