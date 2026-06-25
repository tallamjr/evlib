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

import numpy as np
import polars as pl

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
