"""RVT bbox filter chain for label preprocessing (build-side, task B1).

Reads a raw Prophesee ``*_bbox.npy`` structured array and applies RVT's exact
bbox filter chain, producing a filtered structured array. The arithmetic mirrors
``lib/RVT/scripts/genx/preprocess_dataset.py`` precisely (the right/bottom crop
bounds use ``W-1``/``H-1``) so the later byte-identical gate (B3) holds.

The ``track_id`` and ``class_confidence`` fields are carried through every
filter, matching the real on-disk schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

# RVT's on-disk representation directory name carries ``=`` separators
# (``stacked_histogram_dt=50_nbins=10``).  Pass this constant as
# ``repr_dir_name`` when reproducing RVT's upstream on-disk layout.
RVT_REPR_DIR_NAME = "stacked_histogram_dt=50_nbins=10"

# evlib-native form: no ``=`` separators.  This is the default for
# ``write_preprocessed`` and ``preprocess_sequence`` so that output written by
# those functions is readable by the default ``PreprocessedH5Source`` and
# ``EvlibStreamSource`` without any extra configuration.
# Must match evlib.data.sources.REPR_NAME and evlib.rvt.pipeline.REPR_NAME --
# kept as a standalone literal to avoid pulling torch into this module.
EVLIB_REPR_DIR_NAME = "stacked_histogram_dt50_nbins10"

# Verified on-disk schema from the staged real data. The field order is
# load-bearing: B2/B3 write ``labels.npz`` from arrays of exactly this dtype.
BBOX_DTYPE = np.dtype(
    [
        ("t", "<u8"),
        ("x", "<f4"),
        ("y", "<f4"),
        ("w", "<f4"),
        ("h", "<f4"),
        ("class_id", "u1"),
        ("class_confidence", "<f4"),
        ("track_id", "<u4"),
    ]
)

LABEL_NPZ_FIELDS = tuple(BBOX_DTYPE.names)

# gen1/gen4 frame geometry (RVT preprocess_dataset.py lines 58-59).
_DATASET_HEIGHT = {"gen1": 240, "gen4": 720}
_DATASET_WIDTH = {"gen1": 304, "gen4": 1280}


class NoLabelsError(Exception):
    """Raised when a sequence filters down to zero boxes.

    Mirrors RVT's ``NoLabelsException`` intent: zero surviving labels is a named
    failure, never a silently returned empty array.
    """


def read_raw_bbox(path: Union[str, Path]) -> np.ndarray:
    """Load a raw structured ``*_bbox.npy`` and validate its fields.

    Raises ``ValueError`` naming the path if the array is not structured with
    exactly the expected fields. Does not coerce silently.
    """
    path = Path(path)
    labels = np.load(str(path))
    if labels.dtype.names is None:
        raise ValueError(
            f"{path} is not a structured bbox array (got dtype {labels.dtype})"
        )
    actual = tuple(labels.dtype.names)
    if actual != LABEL_NPZ_FIELDS:
        raise ValueError(
            f"{path} has unexpected bbox fields {actual}; expected {LABEL_NPZ_FIELDS}"
        )
    return labels


def keep_classes_gen4(labels: np.ndarray) -> np.ndarray:
    """Keep gen4 classes pedestrian/two-wheeler/car (``class_id <= 2``).

    Drops truck, bus, traffic sign, traffic light. gen4 only.
    (RVT ``prophesee_remove_labels_filter_gen4``, lines 263-271.)
    """
    keep = labels["class_id"] <= 2
    return labels[keep]


def crop_to_fov(labels: np.ndarray, height: int, width: int) -> np.ndarray:
    """Clamp box edges to the FOV, recompute w/h, drop degenerate boxes.

    Matches RVT ``crop_to_fov_filter`` (lines 232-260): clamp left/top/right/
    bottom edges to ``[0, W-1]`` / ``[0, H-1]``, recompute w/h from the clamped
    edges, then drop boxes with ``w <= 0`` or ``h <= 0``. Operates on a copy so
    the caller's array is not mutated.
    """
    labels = labels.copy()
    x_left = labels["x"]
    y_top = labels["y"]
    x_right = x_left + labels["w"]
    y_bottom = y_top + labels["h"]

    x_left_cropped = np.clip(x_left, a_min=0, a_max=width - 1)
    y_top_cropped = np.clip(y_top, a_min=0, a_max=height - 1)
    x_right_cropped = np.clip(x_right, a_min=0, a_max=width - 1)
    y_bottom_cropped = np.clip(y_bottom, a_min=0, a_max=height - 1)

    w_cropped = x_right_cropped - x_left_cropped
    h_cropped = y_bottom_cropped - y_top_cropped
    assert np.all(w_cropped >= 0)
    assert np.all(h_cropped >= 0)

    labels["x"] = x_left_cropped
    labels["y"] = y_top_cropped
    labels["w"] = w_cropped
    labels["h"] = h_cropped

    keep = (labels["w"] > 0) & (labels["h"] > 0)
    return labels[keep]


def conservative_size_filter(labels: np.ndarray) -> np.ndarray:
    """Keep boxes with ``w >= 5 AND h >= 5`` (RVT ``conservative_bbox_filter``).

    Used when ``apply_psee_bbox_filter`` is False (the gen4 config).
    """
    min_box_side = 5
    side_ok = (labels["w"] >= min_box_side) & (labels["h"] >= min_box_side)
    return labels[side_ok]


def prophesee_size_filter(labels: np.ndarray, dataset: str) -> np.ndarray:
    """Prophesee diag/side size filter (RVT ``prophesee_bbox_filter``).

    Keeps boxes whose diagonal is at least ``min_box_diag`` and whose width and
    height each reach ``min_box_side``. gen4: diag 60, side 20; gen1: 30, 10.
    Used when ``apply_psee_bbox_filter`` is True (the gen1 branch).
    """
    if dataset not in {"gen1", "gen4"}:
        raise ValueError(f"unknown dataset {dataset!r}")
    min_box_diag = 60 if dataset == "gen4" else 30
    min_box_side = 20 if dataset == "gen4" else 10

    w_lbl = labels["w"]
    h_lbl = labels["h"]
    diag_ok = w_lbl**2 + h_lbl**2 >= min_box_diag**2
    side_ok = (w_lbl >= min_box_side) & (h_lbl >= min_box_side)
    return labels[diag_ok & side_ok]


def remove_faulty_huge_bbox(labels: np.ndarray, width: int) -> np.ndarray:
    """Drop boxes wider than ``(9 * width) // 10`` (RVT ``remove_faulty_huge``).

    These span the frame horizontally without covering an object. TRAIN-split
    only; the caller gates by split.
    """
    max_width = (9 * width) // 10
    side_ok = labels["w"] <= max_width
    return labels[side_ok]


def apply_filters(
    labels: np.ndarray,
    *,
    dataset: str = "gen4",
    split: str,
    height: int,
    width: int,
    apply_psee_bbox_filter: bool = False,
    apply_faulty_bbox_filter: bool = True,
) -> np.ndarray:
    """Run RVT's filter chain in order, returning the surviving boxes.

    Order (RVT ``apply_filters``, lines 274-288):
    1. gen4 class removal (gen4 only),
    2. crop-to-FOV,
    3. size filter: prophesee diag/side if ``apply_psee_bbox_filter`` else
       conservative (gen4 uses conservative),
    4. faulty-huge removal, TRAIN split only and only if
       ``apply_faulty_bbox_filter``.

    Raises ``NoLabelsError`` if zero boxes survive (never returns empty silently).
    """
    if dataset not in {"gen1", "gen4"}:
        raise ValueError(f"unknown dataset {dataset!r}")

    if dataset == "gen4":
        labels = keep_classes_gen4(labels)
    labels = crop_to_fov(labels, height, width)
    if apply_psee_bbox_filter:
        labels = prophesee_size_filter(labels, dataset)
    else:
        labels = conservative_size_filter(labels)
    if split == "train" and apply_faulty_bbox_filter:
        labels = remove_faulty_huge_bbox(labels, width)

    if labels.shape[0] == 0:
        raise NoLabelsError(
            f"all boxes removed by the filter chain (dataset={dataset}, split={split})"
        )
    return labels


def get_base_delta_ts_us(unique_ts_us: np.ndarray, dataset: str = "gen4") -> int:
    """Base label cadence in microseconds (RVT ``get_base_delta_ts_for_labels_us``).

    gen1 is a fixed 4 Hz cadence (``250000`` us). gen4 infers the dense label
    cadence from ``median(diff(unique_ts))``, asserts it is 30 or 60 Hz, then
    scales by 3 (30 Hz) or 6 (60 Hz) to approximate the 10 Hz object-frame rate.
    (Lines 291-303.)
    """
    if dataset == "gen1":
        return 250000
    if dataset != "gen4":
        raise ValueError(f"unknown dataset {dataset!r}")
    diff_us = np.diff(unique_ts_us)
    median_diff_us = np.median(diff_us)
    hz = int(np.rint(10**6 / median_diff_us))
    if hz not in {30, 60}:
        raise ValueError(f"inferred label cadence {hz} Hz; expected 30 or 60")
    return int(6 * median_diff_us if hz == 60 else 3 * median_diff_us)


@dataclass
class ObjframeGridResult:
    """Aligned object-frame and event-representation-grid artifacts.

    All index/timestamp arrays are int64, matching RVT's on-disk dtypes.

    - ``labels``: the per-object-frame-grouped boxes, concatenated in frame
      order (structured ``BBOX_DTYPE`` array).
    - ``objframe_idx_2_label_idx``: cumulative per-frame start offsets into
      ``labels`` (one entry per object frame).
    - ``frame_timestamps_us``: the selected object-frame timestamps.
    - ``ev_repr_timestamps_us_end``: the window-end grid for event reprs.
    - ``objframe_idx_2_repr_idx``: ``searchsorted`` of the frame timestamps into
      the grid; ``frame_timestamps_us == grid[objframe_idx_2_repr_idx]`` exactly.
    """

    labels: np.ndarray
    objframe_idx_2_label_idx: np.ndarray
    frame_timestamps_us: np.ndarray
    ev_repr_timestamps_us_end: np.ndarray
    objframe_idx_2_repr_idx: np.ndarray


def build_objframes_and_grid(
    filtered_labels: np.ndarray,
    *,
    dataset: str = "gen4",
    delta_t_us: int = 50000,
    align_t_us: int = 100000,
    ts_step_frame_ms: int = 100,
    ts_step_ev_repr_ms: int = 50,
    jitter_us: int = 2000,
) -> ObjframeGridResult:
    """Select object frames, build the window-end grid, and align the two.

    Transcribes RVT's ``labels_and_ev_repr_timestamps`` (lines 340-432) minus the
    file IO. ``filtered_labels`` must be the output of :func:`apply_filters` and,
    like RVT's input, be sorted by ``t`` (the grouping uses ``searchsorted`` on
    ``t``, lines 390-391).
    """
    if ts_step_frame_ms < ts_step_ev_repr_ms or ts_step_ev_repr_ms <= 0:
        raise ValueError("require ts_step_frame_ms >= ts_step_ev_repr_ms > 0")
    if ts_step_frame_ms % ts_step_ev_repr_ms != 0:
        raise ValueError("ts_step_frame_ms must be a multiple of ts_step_ev_repr_ms")

    label_ts = np.asarray(filtered_labels["t"], dtype="int64")
    unique_ts_us = np.unique(label_ts)

    base_delta_us = get_base_delta_ts_us(unique_ts_us, dataset)

    # First unique timestamp at or after the alignment time (line 371).
    first_idx = int(np.searchsorted(unique_ts_us, align_t_us, side="left"))

    # Walk forward, accepting a timestamp when its gap to the previously accepted
    # frame rounds to an integer multiple of base_delta within the jitter
    # tolerance (lines 374-388). The per-gap event-repr count uses the
    # frame/ev-repr step ratio (ts_step_frame_ms // ts_step_ev_repr_ms == 2).
    repr_step_ratio = ts_step_frame_ms // ts_step_ev_repr_ms
    num_ev_reprs_between = []
    frame_timestamps = [int(unique_ts_us[first_idx])]
    for idx in range(first_idx + 1, len(unique_ts_us)):
        reference_time = frame_timestamps[-1]
        ts = int(unique_ts_us[idx])
        diff_to_ref = ts - reference_time
        base_delta_count = round(diff_to_ref / base_delta_us)
        diff_to_ref_rounded = base_delta_count * base_delta_us
        if abs(diff_to_ref - diff_to_ref_rounded) <= jitter_us:
            if base_delta_count <= 0:
                raise ValueError(
                    "accepted frame with non-positive base_delta_count "
                    f"(diff_to_ref={diff_to_ref}, base_delta_us={base_delta_us})"
                )
            frame_timestamps.append(ts)
            num_ev_reprs_between.append(base_delta_count * repr_step_ratio)
    frame_timestamps_us = np.asarray(frame_timestamps, dtype="int64")

    # Group boxes into object frames via the searchsorted boundaries on the
    # sorted label timestamps (lines 390-399). RVT relies on the labels being
    # time-sorted; all boxes inside a frame share a single timestamp.
    start_indices = np.searchsorted(label_ts, frame_timestamps_us, side="left")
    end_indices = np.searchsorted(label_ts, frame_timestamps_us, side="right")
    labels_per_frame = []
    objframe_idx_2_label_idx = []
    start_offset = 0
    for idx_start, idx_end in zip(start_indices, end_indices):
        boxes = filtered_labels[idx_start:idx_end]
        frame_time = int(boxes["t"][0])
        if not np.all(np.asarray(boxes["t"], dtype="int64") == frame_time):
            raise ValueError("boxes grouped into a frame have differing timestamps")
        labels_per_frame.append(boxes)
        objframe_idx_2_label_idx.append(start_offset)
        start_offset += len(boxes)
    grouped_labels = np.concatenate(labels_per_frame)
    objframe_idx_2_label_idx = np.asarray(objframe_idx_2_label_idx, dtype="int64")

    if len(frame_timestamps_us) > 1:
        min_gap = int(np.diff(frame_timestamps_us).min())
        if min_gap <= 98000:
            raise ValueError(f"selected frames too close together (min gap {min_gap})")

    # Window-end grid: pre-fill from the first frame backwards toward 0 in
    # -delta_t_us steps, dropping the 0 and first-frame edges (line 405), then
    # linspace each consecutive frame pair, dropping the shared edge between
    # segments (lines 408-419).
    grid = list(reversed(range(int(frame_timestamps_us[0]), 0, -delta_t_us)))[1:-1]
    if len(num_ev_reprs_between) != len(frame_timestamps_us) - 1:
        raise ValueError("num_ev_reprs_between length inconsistent with frames")
    for idx, (num_between, frame_start, frame_end) in enumerate(
        zip(num_ev_reprs_between, frame_timestamps_us[:-1], frame_timestamps_us[1:])
    ):
        edges = np.asarray(
            np.linspace(frame_start, frame_end, num_between + 1), dtype="int64"
        ).tolist()
        is_last = idx == len(num_ev_reprs_between) - 1
        if not is_last:
            edges = edges[:-1]
        grid.extend(edges)
    if len(frame_timestamps_us) == 1:
        # No linspace iterations run; append the single frame edge directly.
        grid.append(int(frame_timestamps_us[0]))
    ev_repr_timestamps_us_end = np.asarray(grid, dtype="int64")

    objframe_idx_2_repr_idx = np.searchsorted(
        ev_repr_timestamps_us_end, frame_timestamps_us, side="left"
    ).astype("int64")

    # Sanity checks (lines 426-430): the grid must land exactly on every frame.
    if not np.array_equal(
        frame_timestamps_us, ev_repr_timestamps_us_end[objframe_idx_2_repr_idx]
    ):
        raise ValueError("grid does not land exactly on the frame timestamps")

    return ObjframeGridResult(
        labels=grouped_labels,
        objframe_idx_2_label_idx=objframe_idx_2_label_idx,
        frame_timestamps_us=frame_timestamps_us,
        ev_repr_timestamps_us_end=ev_repr_timestamps_us_end,
        objframe_idx_2_repr_idx=objframe_idx_2_repr_idx,
    )


def write_preprocessed(
    out_dir: Union[str, Path],
    result: ObjframeGridResult,
    *,
    repr_dir_name: str = EVLIB_REPR_DIR_NAME,
) -> None:
    """Lay down the RVT directory tree for one sequence.

    Mirrors RVT's ``save_labels`` (lines 306-337) and ``write_event_data``
    (lines 435-464), writing:

    - ``labels_v2/labels.npz`` with ``labels`` and ``objframe_idx_2_label_idx``,
    - ``labels_v2/timestamps_us.npy`` (the frame timestamps),
    - ``event_representations_v2/<repr_dir_name>/objframe_idx_2_repr_idx.npy``,
    - ``event_representations_v2/<repr_dir_name>/timestamps_us.npy`` (the grid).
    """
    out_dir = Path(out_dir)
    labels_dir = out_dir / "labels_v2"
    repr_dir = out_dir / "event_representations_v2" / repr_dir_name
    labels_dir.mkdir(parents=True, exist_ok=True)
    repr_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        str(labels_dir / "labels.npz"),
        labels=result.labels,
        objframe_idx_2_label_idx=result.objframe_idx_2_label_idx,
    )
    np.save(str(labels_dir / "timestamps_us.npy"), result.frame_timestamps_us)
    np.save(
        str(repr_dir / "objframe_idx_2_repr_idx.npy"),
        result.objframe_idx_2_repr_idx,
    )
    np.save(
        str(repr_dir / "timestamps_us.npy"),
        result.ev_repr_timestamps_us_end,
    )


def preprocess_sequence(
    bbox_path: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    dataset: str = "gen4",
    split: str = "val",
    height: int = 720,
    width: int = 1280,
    repr_dir_name: str = EVLIB_REPR_DIR_NAME,
) -> ObjframeGridResult:
    """Read a raw bbox file, filter, align, and write the RVT artifacts.

    Ties :func:`read_raw_bbox` -> :func:`apply_filters` -> the train-split faulty
    filter gate -> :func:`build_objframes_and_grid` -> :func:`write_preprocessed`.
    """
    raw_labels = read_raw_bbox(bbox_path)
    filtered = apply_filters(
        raw_labels,
        dataset=dataset,
        split=split,
        height=height,
        width=width,
    )
    result = build_objframes_and_grid(filtered, dataset=dataset)
    write_preprocessed(out_dir, result, repr_dir_name=repr_dir_name)
    return result
