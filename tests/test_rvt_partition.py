"""Round-trip tests for the MaxViT non-square partition primitives.

The reference RVT derives a resolution-dependent, possibly NON-SQUARE partition
size ``(h_part, w_part)`` from the padded input HW (gen4 ds2: padded 384x640 ->
partition (6, 10)). evlib's MaxViT primitives must accept that 2-tuple and match
the reference semantics exactly:

- ``window_partition`` treats the tuple as the WINDOW SIZE: each window is
  ``h_part x w_part`` and there are ``H // h_part`` by ``W // w_part`` windows.
- ``grid_partition`` treats the tuple as the GRID SIZE: there are ``h_part`` by
  ``w_part`` windows and each window is ``H // h_part`` by ``W // w_part``.

Both must reconstruct the input exactly under their reverse op, and the evlib
output must equal the reference implementation's output element-for-element.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from evlib.models.rvt_layers import (
    window_partition,
    window_reverse,
    grid_partition,
    grid_reverse,
)

# The reference RVT lives under lib/ and imports its submodules relatively
# (``from models.layers...``), so its package root must be on sys.path. It is
# gitignored in CI, so these reference-comparison tests skip when it is absent.
_REF_ROOT = Path(__file__).resolve().parents[1] / "lib" / "ssms_event_cameras" / "RVT"
if _REF_ROOT.is_dir() and str(_REF_ROOT) not in sys.path:
    sys.path.insert(0, str(_REF_ROOT))

try:
    from models.layers.maxvit.maxvit import (  # type: ignore
        window_partition as ref_window_partition,
        window_reverse as ref_window_reverse,
        grid_partition as ref_grid_partition,
        grid_reverse as ref_grid_reverse,
    )

    _HAVE_REF = True
except Exception:  # reference checkout absent or its deps unavailable
    _HAVE_REF = False

requires_reference = pytest.mark.skipif(
    not _HAVE_REF,
    reason="reference RVT checkout (lib/ssms_event_cameras) not available",
)


PARTITION = (6, 10)
H, W = 12, 20  # divisible by the partition both ways
B, C = 2, 8


def _input() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(B, H, W, C)


def test_window_partition_round_trip_non_square():
    """window_partition -> window_reverse reconstructs a (6,10)-windowed input."""
    x = _input()
    parts = window_partition(x, PARTITION)
    recon = window_reverse(parts, PARTITION, (H, W))
    assert recon.shape == x.shape
    assert torch.equal(recon, x)


def test_grid_partition_round_trip_non_square():
    """grid_partition -> grid_reverse reconstructs a (6,10)-grid input."""
    x = _input()
    parts = grid_partition(x, PARTITION)
    recon = grid_reverse(parts, PARTITION, (H, W))
    assert recon.shape == x.shape
    assert torch.equal(recon, x)


@requires_reference
def test_window_partition_matches_reference():
    """evlib window_partition output equals the reference element-for-element."""
    x = _input()
    ours = window_partition(x, PARTITION)
    ref = ref_window_partition(x, PARTITION)
    # Reference keeps windows as (-1, h, w, C); evlib may flatten the spatial dims
    # into (-1, h*w, C). Compare on a common (-1, h*w, C) view.
    assert torch.equal(
        ours.reshape(-1, PARTITION[0] * PARTITION[1], C),
        ref.reshape(-1, PARTITION[0] * PARTITION[1], C),
    )


@requires_reference
def test_grid_partition_matches_reference():
    """evlib grid_partition output equals the reference element-for-element."""
    x = _input()
    ours = grid_partition(x, PARTITION)
    ref = ref_grid_partition(x, PARTITION)
    win_h = H // PARTITION[0]
    win_w = W // PARTITION[1]
    assert torch.equal(
        ours.reshape(-1, win_h * win_w, C), ref.reshape(-1, win_h * win_w, C)
    )


@requires_reference
def test_window_reverse_matches_reference():
    """evlib window_reverse equals the reference after attention-shaped round-trip."""
    x = _input()
    ref_parts = ref_window_partition(x, PARTITION)
    ref_back = ref_window_reverse(ref_parts, PARTITION, (H, W))
    our_parts = window_partition(x, PARTITION)
    our_back = window_reverse(our_parts, PARTITION, (H, W))
    assert torch.equal(our_back, ref_back)


@requires_reference
def test_grid_reverse_matches_reference():
    """evlib grid_reverse equals the reference after attention-shaped round-trip."""
    x = _input()
    ref_parts = ref_grid_partition(x, PARTITION)
    ref_back = ref_grid_reverse(ref_parts, PARTITION, (H, W))
    our_parts = grid_partition(x, PARTITION)
    our_back = grid_reverse(our_parts, PARTITION, (H, W))
    assert torch.equal(our_back, ref_back)
