"""Reader for the N-Caltech101 ATIS ``.bin`` event format.

N-Caltech101 recordings are the standard ATIS binary event stream: 5 bytes
per event, sensor 240 (width) x 180 (height). Each 5-byte group
``(b0, b1, b2, b3, b4)`` decodes as::

    x = b0
    y = b1
    polarity = b2 >> 7
    t_us = ((b2 & 0x7F) << 16) | (b3 << 8) | b4

with the timestamp in microseconds (a 23-bit value spread across the low 7
bits of ``b2`` and all of ``b3`` and ``b4``).
"""

from os import PathLike
from pathlib import Path

import numpy as np
import polars as pl

from evlib.rvt.representation import build_sparse_histogram
from evlib.rvt.writer import scatter_window_dense

NCALTECH_HEIGHT = 180
NCALTECH_WIDTH = 240

_BYTES_PER_EVENT = 5


def read_atis_bin(path: str | PathLike) -> pl.DataFrame:
    """Decode an N-Caltech101 ATIS ``.bin`` file into an event frame.

    Returns a Polars DataFrame with columns ``x``, ``y``, ``t``, ``polarity``
    (all ``Int64``), in file order. ``t`` is the event timestamp in
    microseconds. ``polarity`` is the raw ATIS polarity bit (0 or 1).

    Raises ``ValueError`` if the file length is not a multiple of 5 bytes.
    """
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size % _BYTES_PER_EVENT != 0:
        raise ValueError(
            f"ATIS .bin file length {raw.size} is not a multiple of "
            f"{_BYTES_PER_EVENT} bytes: {path}"
        )

    events = raw.reshape(-1, _BYTES_PER_EVENT).astype(np.int64)
    b0, b1, b2, b3, b4 = (events[:, i] for i in range(_BYTES_PER_EVENT))

    x = b0
    y = b1
    polarity = b2 >> 7
    t_us = ((b2 & 0x7F) << 16) | (b3 << 8) | b4

    return pl.DataFrame(
        {
            "x": x,
            "y": y,
            "t": t_us,
            "polarity": polarity,
        },
        schema={
            "x": pl.Int64,
            "y": pl.Int64,
            "t": pl.Int64,
            "polarity": pl.Int64,
        },
    )


def representation_from_events(
    events: pl.DataFrame,
    *,
    nbins: int = 10,
    count_cutoff: int = 10,
    height: int = NCALTECH_HEIGHT,
    width: int = NCALTECH_WIDTH,
) -> np.ndarray:
    """Build one full-recording stacked-histogram representation.

    Each N-Caltech recording maps to a single classification sample: one
    stacked-histogram window spanning the whole recording. The window end is
    ``T = t_max`` with ``delta_t_us = t_max - t_min`` so the membership rule
    ``T - delta_t <= t <= T`` (read from
    :func:`evlib.rvt.representation.build_sparse_histogram`) admits every event
    in ``[t_min, t_max]`` inclusive. For a degenerate single-timestamp
    recording (``t_max == t_min``) a ``delta_t_us`` of 1 keeps the lone event
    in range without dividing by a zero span (the backend already clamps the
    binning denominator to >= 1, so the event lands in temporal bin 0).

    Returns a dense ``[2 * nbins, height, width]`` ``uint8`` array. Channels
    are polarity-major: ``channel = polarity * nbins + temporal_bin``.
    """
    if events.height == 0:
        raise ValueError("cannot build a representation from an empty event frame")

    backend_events = events.rename({"polarity": "p"})

    t_min = int(backend_events["t"].min())
    t_max = int(backend_events["t"].max())
    delta_t_us = max(t_max - t_min, 1)

    sparse = build_sparse_histogram(
        backend_events,
        ev_repr_timestamps_us=np.array([t_max], dtype=np.int64),
        delta_t_us=delta_t_us,
        nbins=nbins,
        count_cutoff=count_cutoff,
        height=height,
        width=width,
        downsample_by_2=False,
    )
    window = sparse.filter(pl.col("window_id") == 0)
    return scatter_window_dense(window, channels=2 * nbins, height=height, width=width)


def convert_ncaltech(
    root_dir: str | PathLike,
    out_dir: str | PathLike,
    *,
    nbins: int = 10,
    count_cutoff: int = 10,
) -> tuple[list[Path], list[int]]:
    """Convert an N-Caltech101 class tree into ``SampleDataset`` inputs.

    Walks ``root_dir/<class_name>/*.bin``, builds the class->int label map as
    ``sorted(class_dir_names)`` mapped to ``0..K-1``, decodes each recording
    via :func:`read_atis_bin`, builds its full-recording representation, and
    writes ``<out_dir>/<idx>.npy`` (uint8) per recording plus a parallel
    ``<out_dir>/labels.npy`` (int64). Returns the per-sample paths and labels.

    Raises ``ValueError`` if ``root_dir`` has no class subdirectories or a
    class subdirectory contains no ``.bin`` files.
    """
    root = Path(root_dir)
    out = Path(out_dir)

    class_dirs = sorted(entry for entry in root.iterdir() if entry.is_dir())
    if not class_dirs:
        raise ValueError(f"no class subdirectories found under {root}")

    out.mkdir(parents=True, exist_ok=True)

    sample_paths: list[Path] = []
    labels: list[int] = []
    sample_index = 0
    for label, class_dir in enumerate(class_dirs):
        bin_files = sorted(class_dir.glob("*.bin"))
        if not bin_files:
            raise ValueError(
                f"class directory {class_dir.name!r} contains no .bin files"
            )
        for bin_file in bin_files:
            events = read_atis_bin(bin_file)
            rep = representation_from_events(
                events, nbins=nbins, count_cutoff=count_cutoff
            )
            sample_path = out / f"{sample_index}.npy"
            np.save(sample_path, rep)
            sample_paths.append(sample_path)
            labels.append(label)
            sample_index += 1

    np.save(out / "labels.npy", np.array(labels, dtype=np.int64))
    return sample_paths, labels
