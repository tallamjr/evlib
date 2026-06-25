"""Slow RVT bit-equivalence gate for the bbox-aware SequenceAugmentor.

evlib's ``SequenceAugmentor`` uses a numpy RNG; RVT's
``RandomSpatialAugmentorGenX`` uses a torch RNG. The two RNG streams are NOT
bit-identical, so a shared seed cannot prove equivalence. Instead this test
FORCES identical augmentation params (flip flag, rotation angle, zoom
direction/factor and zoom offsets) into BOTH implementations and asserts their
outputs match: the augmented ``[C, H, W]`` uint8 tensor is byte-identical
(``np.array_equal``) and the transformed yolox boxes agree, after converting
RVT's top-left-stored ``ObjectLabels`` to the yolox CENTRE form evlib produces.

The reference math is the vendored RVT source itself, not a re-implementation:
``lib/RVT/data/utils/augmentor.py`` (tensor ops) and
``lib/RVT/data/genx_utils/labels.py`` (``ObjectLabels`` box ops). It is run as a
LOCAL-ONLY slow gate; on machines where RVT or its deps (einops/omegaconf) are
not importable the whole module skips cleanly via ``importorskip`` rather than
weakening into a non-equivalence check.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.slow

# The vendored RVT tree is imported as a top-level package rooted at lib/RVT, so
# its internal ``from data...`` / ``from utils...`` imports resolve. Inserted
# before importorskip so the guarded imports below see it on sys.path.
_RVT_ROOT = Path(__file__).resolve().parent.parent / "lib" / "RVT"
if _RVT_ROOT.is_dir() and str(_RVT_ROOT) not in sys.path:
    sys.path.insert(0, str(_RVT_ROOT))

# RVT's labels module imports einops at top level; the augmentor imports
# omegaconf. Skip the whole gate cleanly if either (or the RVT tree) is absent.
pytest.importorskip("einops", reason="RVT labels require einops")
pytest.importorskip("omegaconf", reason="RVT augmentor requires omegaconf")
rvt_labels = pytest.importorskip(
    "data.genx_utils.labels", reason="vendored RVT tree not importable"
)
rvt_augmentor = pytest.importorskip(
    "data.utils.augmentor", reason="vendored RVT tree not importable"
)
rvt_types = pytest.importorskip(
    "data.utils.types", reason="vendored RVT tree not importable"
)

from evlib.data.augment import SequenceAugmentor  # noqa: E402

ObjectLabels = rvt_labels.ObjectLabels
RandomSpatialAugmentorGenX = rvt_augmentor.RandomSpatialAugmentorGenX
DataType = rvt_types.DataType

HEIGHT = WIDTH = 8


def _window() -> torch.Tensor:
    """A deterministic [2, 8, 8] uint8 window; channel 1 is channel 0 doubled."""
    base = torch.arange(HEIGHT * WIDTH, dtype=torch.int64).reshape(HEIGHT, WIDTH)
    return torch.stack([base, (base * 2) % 256], dim=0).to(torch.uint8)


def _evlib_centre_box() -> torch.Tensor:
    """One box, evlib yolox CENTRE form [class_id, cx, cy, w, h].

    Top-left rectangle (x=2, y=2, w=2, h=2) -> centre (cx=3, cy=3, w=2, h=2).
    """
    return torch.tensor([[0.0, 3.0, 3.0, 2.0, 2.0]], dtype=torch.float32)


def _rvt_object_labels() -> "ObjectLabels":
    """The SAME box as RVT 7-field rows [t, x, y, w, h, class_id, class_conf].

    RVT stores TOP-LEFT (x=2, y=2); ``get_labels_as_tensors('yolox')`` converts
    to the centre form evlib produces, so the two are directly comparable.
    """
    rows = torch.tensor([[0.0, 2.0, 2.0, 2.0, 2.0, 0.0, 1.0]], dtype=torch.float32)
    return ObjectLabels(object_labels=rows, input_size_hw=(HEIGHT, WIDTH))


def _aug() -> SequenceAugmentor:
    # The RNG is irrelevant here: every test calls the per-transform static
    # methods with explicitly forced params, so no sampling happens.
    return SequenceAugmentor(rng=np.random.default_rng(0))


def test_hflip_bit_equivalent_to_rvt():
    aug = _aug()
    window = _window()

    evlib_tensor = aug._flip_tensor(window)
    rvt_tensor = RandomSpatialAugmentorGenX._flip_tensor(
        window.clone(), flip_type="h", datatype=DataType.EV_REPR
    )
    assert np.array_equal(evlib_tensor.numpy(), rvt_tensor.numpy())
    assert evlib_tensor.dtype == torch.uint8

    evlib_box = aug._flip_box(_evlib_centre_box(), WIDTH)
    labels = _rvt_object_labels()
    labels.flip_lr_()
    rvt_box = labels.get_labels_as_tensors("yolox")
    assert torch.allclose(evlib_box, rvt_box)


def test_rotate_bit_equivalent_to_rvt():
    aug = _aug()
    window = _window()
    angle_deg = 90.0  # forced identical angle into both

    evlib_tensor = aug._rotate_tensor(window, angle_deg)
    rvt_tensor = RandomSpatialAugmentorGenX._rotate_tensor(
        window.clone(), angle_deg=angle_deg, datatype=DataType.EV_REPR
    )
    assert np.array_equal(evlib_tensor.numpy(), rvt_tensor.numpy())

    evlib_box = aug._rotate_box(_evlib_centre_box(), (HEIGHT, WIDTH), angle_deg)
    labels = _rvt_object_labels()
    labels.rotate_(angle_deg=angle_deg)
    rvt_box = labels.get_labels_as_tensors("yolox")
    assert torch.allclose(evlib_box, rvt_box)


def test_zoom_out_bit_equivalent_to_rvt():
    aug = _aug()
    window = _window()
    factor = 2.0
    x0, y0 = 1, 1  # forced identical placement into both

    evlib_tensor = aug._zoom_out_tensor(window, x0, y0, factor)
    rvt_tensor = RandomSpatialAugmentorGenX._zoom_out_and_rescale_tensor(
        window.clone(),
        zoom_coordinates_x0y0=(x0, y0),
        zoom_out_factor=factor,
        datatype=DataType.EV_REPR,
    )
    assert np.array_equal(evlib_tensor.numpy(), rvt_tensor.numpy())

    evlib_box = aug._zoom_out_box(_evlib_centre_box(), (HEIGHT, WIDTH), x0, y0, factor)
    labels = _rvt_object_labels()
    labels.zoom_out_and_rescale_(zoom_coordinates_x0y0=(x0, y0), zoom_out_factor=factor)
    rvt_box = labels.get_labels_as_tensors("yolox")
    assert torch.allclose(evlib_box, rvt_box)


def test_zoom_in_bit_equivalent_to_rvt():
    aug = _aug()
    window = _window()
    factor = 2.0
    zoom_x0, zoom_y0 = 1, 1  # forced identical crop window into both

    evlib_tensor = aug._zoom_in_tensor(window, zoom_x0, zoom_y0, factor)
    rvt_tensor = RandomSpatialAugmentorGenX._zoom_in_and_rescale_tensor(
        window.clone(),
        zoom_coordinates_x0y0=(zoom_x0, zoom_y0),
        zoom_in_factor=factor,
        datatype=DataType.EV_REPR,
    )
    assert np.array_equal(evlib_tensor.numpy(), rvt_tensor.numpy())

    evlib_box = aug._zoom_in_box(
        _evlib_centre_box(), (HEIGHT, WIDTH), zoom_x0, zoom_y0, factor
    )
    labels = _rvt_object_labels()
    labels.zoom_in_and_rescale_(
        zoom_coordinates_x0y0=(zoom_x0, zoom_y0), zoom_in_factor=factor
    )
    rvt_box = labels.get_labels_as_tensors("yolox")
    assert evlib_box is not None and rvt_box.numel() > 0
    assert torch.allclose(evlib_box, rvt_box)
