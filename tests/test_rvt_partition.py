"""Round-trip tests for the MaxViT non-square partition primitives.

The reference RVT derives a resolution-dependent, possibly NON-SQUARE partition
size ``(h_part, w_part)`` from the padded input HW (gen4 ds2: padded 384x640 ->
partition (6, 10)). evlib's MaxViT primitives must accept that 2-tuple and match
the reference semantics exactly:

- ``window_partition`` treats the tuple as the WINDOW SIZE: each window is
  ``h_part x w_part`` and there are ``H // h_part`` by ``W // w_part`` windows.
- ``grid_partition`` treats the tuple as the GRID-WINDOW SIZE the same way: each
  attention window is ``h_part x w_part`` tokens sampled on a stride-``(H //
  h_part, W // w_part)`` lattice, and there are ``(H // h_part) * (W // w_part)``
  windows. (The earlier convention that flattened ``(H // h_part) * (W //
  w_part)`` into the token axis scrambled the grid tokens versus the reference.)

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
        SelfAttentionCl as RefSelfAttentionCl,
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
    """evlib grid_partition output equals the reference element-for-element.

    The grid attention window is ``h_part x w_part`` tokens, so the token axis
    has length ``h_part * w_part`` (not ``win_h * win_w``); comparing on that
    common view catches the scrambled-token regression.
    """
    x = _input()
    ours = grid_partition(x, PARTITION)
    ref = ref_grid_partition(x, PARTITION)
    assert torch.equal(
        ours.reshape(-1, PARTITION[0] * PARTITION[1], C),
        ref.reshape(-1, PARTITION[0] * PARTITION[1], C),
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


def test_attention_derives_heads_from_dim_head():
    """evlib Attention derives num_heads = dim // dim_head (reference behaviour).

    The reference fixes dim_head=32, so a 64-wide stage uses 2 heads and a
    256-wide stage uses 8. A fixed head count would diverge from the trained
    weights on every stage except the widest.
    """
    from evlib.models.rvt_layers import Attention

    for dim, expected_heads in ((32, 1), (64, 2), (128, 4), (256, 8)):
        attn = Attention(dim, dim_head=32)
        assert attn.num_heads == expected_heads, (dim, attn.num_heads)


@requires_reference
@pytest.mark.parametrize("dim", [32, 64, 128, 256])
def test_attention_matches_reference_self_attention(dim: int):
    """evlib Attention equals the reference SelfAttentionCl for a multi-head stage.

    Catches the qkv head/channel split-order regression: the reference reads the
    qkv projection as (num_heads, dim_head * 3) and only then splits q/k/v, which
    differs from a (3, num_heads, dim_head) split whenever num_heads > 1.
    """
    from evlib.models.rvt_layers import Attention

    torch.manual_seed(0)
    ours = Attention(dim, dim_head=32)
    ref = RefSelfAttentionCl(dim, dim_head=32, bias=True)
    # share weights
    ref.qkv.weight.data.copy_(ours.qkv.weight.data)
    ref.qkv.bias.data.copy_(ours.qkv.bias.data)
    ref.proj.weight.data.copy_(ours.proj.weight.data)
    ref.proj.bias.data.copy_(ours.proj.bias.data)
    ours.eval()
    ref.eval()

    x = torch.randn(4, 60, dim)  # (B*windows, tokens, C)
    with torch.no_grad():
        out_ours = ours(x)
        out_ref = ref(x)
    assert torch.allclose(out_ours, out_ref, atol=1e-6), (
        dim,
        (out_ours - out_ref).abs().max().item(),
    )
