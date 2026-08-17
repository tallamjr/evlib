"""Load raw frame stacks written by lumin vfi's StackWriter.

Layout: `frames.u8` (T*H*W bytes, frame-major, row-major), `timestamps_ns.i64`
(T little-endian int64), `meta.json` (width, height, frames, source,
source_fps, model, provider, policy, factor, generated_by, date).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

FRAMES_FILE = "frames.u8"
TIMESTAMPS_FILE = "timestamps_ns.i64"
META_FILE = "meta.json"


@dataclass
class FrameStack:
    """A frame stack loaded from a lumin vfi stack directory."""

    frames: np.memmap
    timestamps_ns: np.ndarray
    meta: dict


def load_frame_stack(path: Union[str, Path]) -> FrameStack:
    """Load a stack directory written by lumin's StackWriter.

    `frames` is memory-mapped read-only as (T, H, W) uint8; `timestamps_ns`
    is loaded fully as int64. Raises FileNotFoundError if a file is missing,
    ValueError if meta.json does not match the file sizes or timestamps are
    not strictly increasing.
    """
    directory = Path(path)
    meta_path = directory / META_FILE
    frames_path = directory / FRAMES_FILE
    timestamps_path = directory / TIMESTAMPS_FILE
    for file_path in (meta_path, frames_path, timestamps_path):
        if not file_path.exists():
            raise FileNotFoundError(str(file_path))

    meta = json.loads(meta_path.read_text())
    for key in ("width", "height", "frames"):
        if key not in meta:
            raise ValueError(f"{meta_path}: missing key {key!r}")

    width = int(meta["width"])
    height = int(meta["height"])
    frame_count = int(meta["frames"])

    frames_bytes = frames_path.stat().st_size
    expected_frames_bytes = width * height * frame_count
    if frames_bytes != expected_frames_bytes:
        raise ValueError(
            f"{frames_path}: holds {frames_bytes} bytes, meta.json implies "
            f"{expected_frames_bytes} bytes ({frame_count} frames of {height}x{width})"
        )

    timestamps_bytes = timestamps_path.stat().st_size
    expected_timestamps_bytes = frame_count * 8
    if timestamps_bytes != expected_timestamps_bytes:
        raise ValueError(
            f"{timestamps_path}: holds {timestamps_bytes} bytes, meta.json implies "
            f"{expected_timestamps_bytes} bytes ({frame_count} frames x 8)"
        )

    frames = np.memmap(
        frames_path, dtype=np.uint8, mode="r", shape=(frame_count, height, width)
    )
    timestamps_ns = np.fromfile(timestamps_path, dtype="<i8").astype(
        np.int64, copy=False
    )
    if frame_count > 1:
        deltas = np.diff(timestamps_ns)
        if not np.all(deltas > 0):
            bad = int(np.argmax(deltas <= 0))
            raise ValueError(
                f"{timestamps_path}: timestamps must strictly increase, "
                f"got {timestamps_ns[bad]} then {timestamps_ns[bad + 1]} at index {bad + 1}"
            )

    return FrameStack(frames=frames, timestamps_ns=timestamps_ns, meta=meta)
