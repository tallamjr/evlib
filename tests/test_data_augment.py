"""Deterministic per-transform tests for the bbox-aware SequenceAugmentor.

These hand-computed tests are the primary correctness evidence for Task C-core
(the slow RVT bit-equivalence gate is a separate later task that may skip when
RVT deps are absent). Each transform is checked on a tiny [2, 4, 4] window with
one or two known boxes so BOTH the tensor and the box land at exact coordinates
that are verified against the vendored RVT math
(lib/RVT/data/utils/augmentor.py + lib/RVT/data/genx_utils/labels.py).
"""

from __future__ import annotations

import numpy as np
import torch

from evlib.data.augment import (
    SequenceAugmentor,
    _AugmentationState,
    _ZoomOutState,
)
from evlib.data.sequence import SequenceSample


def _arange_window(channels: int = 2, height: int = 4, width: int = 4) -> torch.Tensor:
    """A deterministic uint8 window whose values double on the second channel."""
    base = torch.arange(height * width, dtype=torch.int64).reshape(height, width)
    second = (base * 2) % 256
    return torch.stack([base, second], dim=0).to(torch.uint8)


def _centre_box(class_id, cx, cy, w, h) -> torch.Tensor:
    return torch.tensor([[class_id, cx, cy, w, h]], dtype=torch.float32)


def _sample(ev_repr, labels, is_padded_mask=None, is_first_sample=True):
    if is_padded_mask is None:
        is_padded_mask = [False] * len(ev_repr)
    return SequenceSample(
        ev_repr=ev_repr,
        labels=labels,
        is_first_sample=is_first_sample,
        is_padded_mask=is_padded_mask,
    )


# ---------------------------------------------------------------------------
# Horizontal flip
# ---------------------------------------------------------------------------


def test_hflip_tensor_and_box_exact():
    window = _arange_window()
    # top-left box (x=1, y=1, w=1, h=1) -> centre (cx=1.5, cy=1.5, w=1, h=1)
    box = _centre_box(0.0, 1.5, 1.5, 1.0, 1.0)
    sample = _sample([window.clone()], [box.clone()])

    aug = SequenceAugmentor(
        prob_hflip=1.0, rotate_prob=0.0, zoom_prob=0.0, rng=np.random.default_rng(0)
    )
    out = aug(sample)

    expected_tensor = torch.flip(window, dims=[-1])
    assert torch.equal(out.ev_repr[0], expected_tensor)
    assert out.ev_repr[0].dtype == torch.uint8

    # RVT flip_lr_: x_new = W - 1 - x - w = 3 - 1 - 1 = 1 (top-left), so
    # centre cx = 1 + 0.5 = 1.5; y, w, h unchanged.
    expected_box = _centre_box(0.0, 1.5, 1.5, 1.0, 1.0)
    assert torch.allclose(out.labels[0], expected_box)


def test_hflip_box_moves_for_offcentre_box():
    window = _arange_window()
    # top-left box (x=0, y=0, w=2, h=2) -> centre (1.0, 1.0, 2, 2)
    box = _centre_box(3.0, 1.0, 1.0, 2.0, 2.0)
    sample = _sample([window.clone()], [box.clone()])

    aug = SequenceAugmentor(
        prob_hflip=1.0, rotate_prob=0.0, zoom_prob=0.0, rng=np.random.default_rng(0)
    )
    out = aug(sample)

    # x_new = 3 - 0 - 2 = 1 (top-left) -> centre cx = 1 + 1 = 2.0
    expected_box = _centre_box(3.0, 2.0, 1.0, 2.0, 2.0)
    assert torch.allclose(out.labels[0], expected_box)


# ---------------------------------------------------------------------------
# Rotate
# ---------------------------------------------------------------------------


def test_rotate_90deg_tensor_and_box_exact():
    window = _arange_window()
    # top-left box (x=1, y=1, w=1, h=1) -> centre (1.5, 1.5, 1, 1)
    box = _centre_box(0.0, 1.5, 1.5, 1.0, 1.0)
    sample = _sample([window.clone()], [box.clone()])

    aug = SequenceAugmentor(
        prob_hflip=0.0,
        rotate_prob=1.0,
        rotate_min_deg=90.0,
        rotate_max_deg=90.0,
        zoom_prob=0.0,
        # seed 2 draws a positive sign so the rotation is +90 deg (CCW).
        rng=np.random.default_rng(2),
    )
    out = aug(sample)

    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.functional import rotate as tv_rotate

    expected_tensor = tv_rotate(window, 90.0, interpolation=InterpolationMode.NEAREST)
    assert torch.equal(out.ev_repr[0], expected_tensor)

    # Hand-computed: corners of top-left box rotated CCW 90deg about (2, 2)
    # give axis-aligned bbox top-left (x0=1, y0=2, w=1, h=1) -> centre (1.5, 2.5).
    expected_box = _centre_box(0.0, 1.5, 2.5, 1.0, 1.0)
    assert torch.allclose(out.labels[0], expected_box)


def test_rotate_uses_random_sign():
    # rotate_min == rotate_max (=90) so the only randomness is the sign. A 90deg
    # rotation moves the box to distinguishable centres: +90 -> (1.5, 2.5),
    # -90 -> (2.5, 1.5). Across seeds we must see BOTH signs, proving the random
    # sign is exercised. (A 4deg NEAREST rotation is too small to move a tiny
    # tensor, so we assert via the box coordinate, not the tensor.)
    window = _arange_window()
    box = _centre_box(0.0, 1.5, 1.5, 1.0, 1.0)

    pos_centre = _centre_box(0.0, 1.5, 2.5, 1.0, 1.0)
    neg_centre = _centre_box(0.0, 2.5, 1.5, 1.0, 1.0)

    seen = set()
    for seed in range(20):
        aug = SequenceAugmentor(
            prob_hflip=0.0,
            rotate_prob=1.0,
            rotate_min_deg=90.0,
            rotate_max_deg=90.0,
            zoom_prob=0.0,
            rng=np.random.default_rng(seed),
        )
        out = aug(_sample([window.clone()], [box.clone()]))
        if torch.allclose(out.labels[0], pos_centre):
            seen.add("pos")
        elif torch.allclose(out.labels[0], neg_centre):
            seen.add("neg")
        else:
            raise AssertionError(f"rotate produced unexpected box {out.labels[0]}")
    assert seen == {"pos", "neg"}


# ---------------------------------------------------------------------------
# Zoom out
# ---------------------------------------------------------------------------


def test_zoom_out_factor2_tensor_and_box_exact():
    window = _arange_window()
    # top-left box (x=0, y=0, w=2, h=2) -> centre (1.0, 1.0, 2, 2)
    box = _centre_box(5.0, 1.0, 1.0, 2.0, 2.0)
    sample = _sample([window.clone()], [box.clone()])

    # Force a zoom-out at factor 2 with placement at (x0=0, y0=0).
    aug = SequenceAugmentor(
        prob_hflip=0.0,
        rotate_prob=0.0,
        zoom_prob=1.0,
        zoom_in_weight=0,
        zoom_out_weight=1,
        zoom_out_range=(2.0, 2.0),
        rng=np.random.default_rng(0),
    )
    out = aug(sample)

    # Tensor: shrink 4x4 -> 2x2 via nearest-exact, paste at sampled (x0, y0).
    # The placement offset is RNG-sampled in [0, W - 2]; discover it from the
    # tensor, then assert both tensor and box agree on the same offset.
    from torch.nn.functional import interpolate

    shrunk = interpolate(
        window.unsqueeze(0).float(), size=(2, 2), mode="nearest-exact"
    )[0].to(torch.uint8)

    found = None
    for y0 in range(0, 4 - 2 + 1):
        for x0 in range(0, 4 - 2 + 1):
            canvas = torch.zeros_like(window)
            canvas[:, y0 : y0 + 2, x0 : x0 + 2] = shrunk
            if torch.equal(out.ev_repr[0], canvas):
                found = (x0, y0)
                break
        if found is not None:
            break
    assert found is not None, "zoom-out tensor did not match any valid offset"
    x0, y0 = found

    # Box per RVT zoom_out: scale(1/2) -> top-left (0, 0, 1, 1); +(x0, y0).
    # centre cx = x0 + 0.5, cy = y0 + 0.5, w = 1, h = 1.
    expected_box = _centre_box(5.0, x0 + 0.5, y0 + 0.5, 1.0, 1.0)
    assert torch.allclose(out.labels[0], expected_box)


def test_zoom_out_forced_params_hand_pinned_box():
    # FORCE the zoom-out state directly (no RNG discovery): factor 2 with the
    # shrunk canvas pasted at the KNOWN offset (x0=1, y0=1). Both the tensor and
    # the box are pinned to independently hand-computed coordinates.
    window = _arange_window()
    # top-left box (x=0, y=0, w=2, h=2) -> centre (1.0, 1.0, 2, 2)
    box = _centre_box(5.0, 1.0, 1.0, 2.0, 2.0)
    sample = _sample([window.clone()], [box.clone()])

    aug = SequenceAugmentor(
        prob_hflip=0.0, rotate_prob=0.0, zoom_prob=0.0, rng=np.random.default_rng(0)
    )
    state = _AugmentationState(
        apply_h_flip=False,
        rotation_active=False,
        rotation_angle_deg=0.0,
        apply_zoom_in=False,
        zoom_in_factor=1.0,
        zoom_out=_ZoomOutState(active=True, x0=1, y0=1, factor=2.0),
    )
    out = aug._apply(sample, state, (4, 4))

    # Tensor: shrink 4x4 -> 2x2 nearest-exact, paste at the FORCED (x0=1, y0=1).
    from torch.nn.functional import interpolate

    shrunk = interpolate(
        window.unsqueeze(0).float(), size=(2, 2), mode="nearest-exact"
    )[0].to(torch.uint8)
    expected_tensor = torch.zeros_like(window)
    expected_tensor[:, 1:3, 1:3] = shrunk
    assert torch.equal(out.ev_repr[0], expected_tensor)

    # Box per RVT zoom_out: scale(1/2) of top-left (0,0,2,2) with clamp to
    # new_img-1 = 1 gives (0, 0, 1, 1); translate by (x0=1, y0=1) -> (1, 1, 1, 1).
    # centre cx = 1 + 0.5 = 1.5, cy = 1.5, w = 1, h = 1.
    expected_box = _centre_box(5.0, 1.5, 1.5, 1.0, 1.0)
    assert torch.allclose(out.labels[0], expected_box)


# ---------------------------------------------------------------------------
# Zoom in
# ---------------------------------------------------------------------------


def test_zoom_in_forced_params_hand_pinned_box():
    # FORCE zoom-in factor 2 and a box geometry whose label-aware valid window
    # COLLAPSES to a single point (x0_valid == x1_valid == 2), so _uniform(2, 2)
    # returns 2 deterministically: the crop offset is (2, 2) with no RNG draw.
    window = torch.arange(64, dtype=torch.uint8).reshape(8, 8)
    window = torch.stack([window, window], dim=0)
    # top-left box (x=2, y=2, w=4, h=4) -> centre (4.0, 4.0, 4, 4).
    box = _centre_box(1.0, 4.0, 4.0, 4.0, 4.0)
    sample = _sample([window.clone()], [box.clone()])

    aug = SequenceAugmentor(
        prob_hflip=0.0, rotate_prob=0.0, zoom_prob=0.0, rng=np.random.default_rng(0)
    )
    state = _AugmentationState(
        apply_h_flip=False,
        rotation_active=False,
        rotation_angle_deg=0.0,
        apply_zoom_in=True,
        zoom_in_factor=2.0,
        zoom_out=_ZoomOutState(active=False, x0=0, y0=0, factor=1.0),
    )
    out = aug._apply(sample, state, (8, 8))

    # Tensor: crop window[2:6, 2:6] (4x4) upscaled to 8x8 nearest-exact, with the
    # FORCED crop offset (x0=2, y0=2).
    from torch.nn.functional import interpolate

    crop = window[:, 2:6, 2:6].unsqueeze(0).float()
    expected_tensor = interpolate(crop, size=(8, 8), mode="nearest-exact")[0].to(
        torch.uint8
    )
    assert torch.equal(out.ev_repr[0], expected_tensor)

    # Box per RVT zoom_in: clamp into the crop -> (new_x=0, new_y=0, new_w=3,
    # new_h=3) after subtracting (z_x0=2, z_y0=2) and clamping x1,y1 to z*1-1=5;
    # then scale_ by factor 2 with clamp to new_img-1=7 -> (0, 0, 6, 6).
    # centre cx = 0 + 3 = 3.0, cy = 3.0, w = 6, h = 6.
    expected_box = _centre_box(1.0, 3.0, 3.0, 6.0, 6.0)
    assert torch.allclose(out.labels[0], expected_box)


def test_zoom_in_factor_exactly_one_is_noop():
    # A zoom-in factor of exactly 1.0 is a documented no-op: _apply short-circuits
    # on `state.zoom_in_factor != 1.0`, matching RVT's behaviour, so the tensor
    # and boxes pass through unchanged.
    window = _arange_window()
    box = _centre_box(2.0, 1.5, 1.5, 1.0, 1.0)
    sample = _sample([window.clone()], [box.clone()])

    aug = SequenceAugmentor(
        prob_hflip=0.0, rotate_prob=0.0, zoom_prob=0.0, rng=np.random.default_rng(0)
    )
    state = _AugmentationState(
        apply_h_flip=False,
        rotation_active=False,
        rotation_angle_deg=0.0,
        apply_zoom_in=True,
        zoom_in_factor=1.0,
        zoom_out=_ZoomOutState(active=False, x0=0, y0=0, factor=1.0),
    )
    out = aug._apply(sample, state, (4, 4))

    assert torch.equal(out.ev_repr[0], window)
    assert torch.allclose(out.labels[0], box)


def test_zoom_in_factor2_label_aware_tensor_and_box_exact():
    # Use an 8x8 window so a factor-2 zoom-in crop (4x4) can fully contain a
    # box and the label-aware window sampling has a deterministic solution.
    window = torch.arange(64, dtype=torch.uint8).reshape(8, 8)
    window = torch.stack([window, window], dim=0)
    # top-left box (x=2, y=2, w=2, h=2) -> centre (3.0, 3.0, 2, 2)
    box = _centre_box(1.0, 3.0, 3.0, 2.0, 2.0)
    sample = _sample([window.clone()], [box.clone()])

    aug = SequenceAugmentor(
        prob_hflip=0.0,
        rotate_prob=0.0,
        zoom_prob=1.0,
        zoom_in_weight=1,
        zoom_out_weight=0,
        zoom_in_range=(2.0, 2.0),
        rng=np.random.default_rng(0),
    )
    out = aug(sample)

    # crop window is 4x4. label-aware sampling must keep box (x1=4 <= crop)
    # fully inside. With this seed we capture the realised (x0, y0) from the
    # tensor and assert the box matches the same crop deterministically.
    from torch.nn.functional import interpolate

    # Discover the realised crop offset by matching against all valid offsets.
    realised = out.ev_repr[0]
    found = None
    for y0 in range(0, 8 - 4 + 1):
        for x0 in range(0, 8 - 4 + 1):
            crop = window[:, y0 : y0 + 4, x0 : x0 + 4].unsqueeze(0).float()
            up = interpolate(crop, size=(8, 8), mode="nearest-exact")[0].to(torch.uint8)
            if torch.equal(up, realised):
                found = (x0, y0)
                break
        if found is not None:
            break
    assert found is not None, "zoom-in tensor did not match any valid crop"
    x0, y0 = found

    # The crop must fully contain the box [2,4]x[2,4]; label-aware sampling
    # guarantees x0 <= 2 and x0 + 4 >= 4 (=> 0 <= x0 <= 2), same for y0.
    assert 0 <= x0 <= 2
    assert 0 <= y0 <= 2

    # Box per RVT zoom_in_and_rescale_: subtract (x0, y0), scale by factor 2.
    bx = (2 - x0) * 2.0
    by = (2 - y0) * 2.0
    bw = 2 * 2.0  # 4
    bh = 2 * 2.0
    expected_centre = torch.tensor(
        [[1.0, bx + bw / 2.0, by + bh / 2.0, bw, bh]], dtype=torch.float32
    )
    assert torch.allclose(out.labels[0], expected_centre)


# ---------------------------------------------------------------------------
# Sequence-wide consistency, padding, None labels, dropped boxes
# ---------------------------------------------------------------------------


def test_params_drawn_once_per_call():
    windows = [_arange_window().clone() for _ in range(3)]
    boxes = [_centre_box(0.0, 1.5, 1.5, 1.0, 1.0) for _ in range(3)]
    sample = _sample(windows, boxes)

    aug = SequenceAugmentor(
        prob_hflip=1.0, rotate_prob=0.0, zoom_prob=0.0, rng=np.random.default_rng(0)
    )
    out = aug(sample)

    expected = torch.flip(_arange_window(), dims=[-1])
    for window in out.ev_repr:
        assert torch.equal(window, expected)
    for box in out.labels:
        assert torch.allclose(box, _centre_box(0.0, 1.5, 1.5, 1.0, 1.0))


def test_padded_windows_untouched():
    real = _arange_window()
    padded = torch.zeros(2, 4, 4, dtype=torch.uint8)
    box = _centre_box(0.0, 1.5, 1.5, 1.0, 1.0)
    sample = _sample(
        [real.clone(), padded.clone()],
        [box.clone(), None],
        is_padded_mask=[False, True],
    )

    aug = SequenceAugmentor(
        prob_hflip=1.0, rotate_prob=0.0, zoom_prob=0.0, rng=np.random.default_rng(0)
    )
    out = aug(sample)

    # Padded slot unchanged, its label stays None.
    assert torch.equal(out.ev_repr[1], padded)
    assert out.labels[1] is None
    # Real slot was flipped.
    assert torch.equal(out.ev_repr[0], torch.flip(real, dims=[-1]))


def test_none_labels_survive_each_transform():
    window = _arange_window()
    sample = _sample([window.clone()], [None])

    aug = SequenceAugmentor(
        prob_hflip=1.0,
        rotate_prob=1.0,
        rotate_min_deg=90.0,
        rotate_max_deg=90.0,
        zoom_prob=0.0,
        rng=np.random.default_rng(0),
    )
    out = aug(sample)

    assert out.labels[0] is None
    # tensor was still transformed.
    assert not torch.equal(out.ev_repr[0], window)


def test_box_pushed_out_of_frame_is_dropped():
    window = _arange_window()
    # A degenerate-after-flip box at the far right edge that flip pushes to the
    # left edge but then a 90deg rotation collapses. Instead test a box that
    # rotation maps to a degenerate (w<=0) bbox after clamping: place a thin box
    # on the boundary. Use a box at the very corner that clamps to zero area.
    box = _centre_box(0.0, 3.0, 3.0, 0.5, 0.5)  # top-left (2.75, 2.75) w=h=0.5
    sample = _sample([window.clone()], [box.clone()])

    aug = SequenceAugmentor(
        prob_hflip=0.0,
        rotate_prob=0.0,
        zoom_prob=1.0,
        zoom_in_weight=0,
        zoom_out_weight=1,
        zoom_out_range=(4.0, 4.0),  # shrink to 1x1, scale boxes by 1/4
        rng=np.random.default_rng(0),
    )
    out = aug(sample)

    # After scaling by 1/4 the 0.5-wide box becomes 0.125 wide and clamps to a
    # degenerate (w<=0 after RVT clamp logic) box, which RVT drops -> None.
    assert out.labels[0] is None


def test_sequence_augmentor_exported_from_package():
    import evlib.data as data

    assert data.SequenceAugmentor is SequenceAugmentor
    assert "SequenceAugmentor" in data.__all__
