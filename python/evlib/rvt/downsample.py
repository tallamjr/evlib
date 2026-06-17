"""Nearest-exact downsample index map (pure gather, matches torch 'nearest-exact')."""

import math
from typing import List


def selected_source_indices(in_size: int, out_size: int) -> List[int]:
    """For each output index d in [0, out_size), the source index it samples.

    torch 'nearest-exact' maps dst -> floor((dst + 0.5) * in_size / out_size).
    """
    assert in_size >= out_size >= 1
    scale = in_size / out_size
    return [int(math.floor((d + 0.5) * scale)) for d in range(out_size)]


def source_to_output_map(in_size: int, out_size: int) -> dict:
    """Map a selected source index back to its output index (inverse of the gather)."""
    return {src: d for d, src in enumerate(selected_source_indices(in_size, out_size))}
