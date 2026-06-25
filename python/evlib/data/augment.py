"""Bbox-aware geometric augmentation for event-representation sequences.

`SequenceAugmentor` reproduces RVT's training-time spatial augmentation
(hflip -> rotate -> zoom_in -> zoom_out) on a single `SequenceSample`,
transforming the `[C, H, W]` uint8 representation tensors AND their yolox boxes
together. The exact tensor and box math mirrors the vendored RVT source:

  - lib/RVT/data/utils/augmentor.py  (RandomSpatialAugmentorGenX)
  - lib/RVT/data/genx_utils/labels.py (ObjectLabels in-place ops)

evlib stores boxes in yolox CENTRE form ``[class_id, cx, cy, w, h]`` while RVT's
math is written in TOP-LEFT ``[x, y, w, h]``. Each transform therefore converts
centre -> top-left, applies RVT's exact op, then converts back to centre. NEAREST
interpolation is used everywhere so the uint8 tensors are preserved exactly.
Padded windows (``is_padded_mask[t]`` true) are returned untouched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.nn.functional import interpolate
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import rotate as tv_rotate

from evlib.data.sequence import SequenceSample

__all__ = ["SequenceAugmentor"]


@dataclass
class _ZoomOutState:
    active: bool
    x0: int
    y0: int
    factor: float


@dataclass
class _AugmentationState:
    apply_h_flip: bool
    rotation_active: bool
    rotation_angle_deg: float
    apply_zoom_in: bool
    zoom_in_factor: float
    zoom_out: _ZoomOutState


def _centre_to_topleft(box: torch.Tensor) -> torch.Tensor:
    """[class_id, cx, cy, w, h] -> column stack [class_id, x, y, w, h] (top-left)."""
    cls = box[:, 0]
    cx = box[:, 1]
    cy = box[:, 2]
    w = box[:, 3]
    h = box[:, 4]
    x = cx - 0.5 * w
    y = cy - 0.5 * h
    return torch.stack([cls, x, y, w, h], dim=1)


def _topleft_to_centre(cls, x, y, w, h) -> Optional[torch.Tensor]:
    """Top-left arrays -> yolox centre tensor; None when no boxes remain."""
    if x.numel() == 0:
        return None
    cx = x + 0.5 * w
    cy = y + 0.5 * h
    return torch.stack([cls, cx, cy, w, h], dim=1).to(torch.float32)


class SequenceAugmentor:
    """Sample augmentation params once per sequence and apply them to every window.

    Parameter ranges default to RVT's ``sampler="random"`` config
    (lib/RVT/config/dataset/base.yaml). Pass ``sampler="stream"`` for the
    streaming preset (zoom-out only, lower zoom probability). All randomness flows
    through a ``numpy.random.Generator`` so tests are deterministic.
    """

    def __init__(
        self,
        *,
        sampler: str = "random",
        prob_hflip: float = 0.5,
        rotate_prob: float = 0.0,
        rotate_min_deg: float = 2.0,
        rotate_max_deg: float = 6.0,
        zoom_prob: Optional[float] = None,
        zoom_in_weight: int = 8,
        zoom_out_weight: int = 2,
        zoom_in_range: Tuple[float, float] = (1.0, 1.5),
        zoom_out_range: Tuple[float, float] = (1.0, 1.2),
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if sampler not in ("random", "stream"):
            raise ValueError(f"sampler must be 'random' or 'stream', got {sampler!r}")
        self.sampler = sampler

        if sampler == "stream":
            # Streaming preset: zoom-out only, lower zoom probability.
            if zoom_prob is None:
                zoom_prob = 0.5
            zoom_in_weight = 0
            zoom_out_range = (1.0, 1.2)
        elif zoom_prob is None:
            zoom_prob = 0.8

        if not 0.0 <= prob_hflip <= 1.0:
            raise ValueError("prob_hflip must be in [0, 1]")
        if not 0.0 <= rotate_prob <= 1.0:
            raise ValueError("rotate_prob must be in [0, 1]")
        if not 0.0 <= zoom_prob <= 1.0:
            raise ValueError("zoom_prob must be in [0, 1]")
        if not 0.0 <= rotate_min_deg <= rotate_max_deg:
            raise ValueError("require 0 <= rotate_min_deg <= rotate_max_deg")
        if not zoom_in_range[1] >= zoom_in_range[0] >= 1.0:
            raise ValueError("require zoom_in_range[1] >= zoom_in_range[0] >= 1")
        if not zoom_out_range[1] >= zoom_out_range[0] >= 1.0:
            raise ValueError("require zoom_out_range[1] >= zoom_out_range[0] >= 1")
        if zoom_in_weight < 0 or zoom_out_weight < 0:
            raise ValueError("zoom weights must be non-negative")
        if zoom_in_weight + zoom_out_weight <= 0:
            raise ValueError("at least one zoom weight must be positive")

        self.prob_hflip = float(prob_hflip)
        self.rotate_prob = float(rotate_prob)
        self.rotate_min_deg = float(rotate_min_deg)
        self.rotate_max_deg = float(rotate_max_deg)
        self.zoom_prob = float(zoom_prob)
        self.zoom_in_weight = int(zoom_in_weight)
        self.zoom_out_weight = int(zoom_out_weight)
        self.zoom_in_range = (float(zoom_in_range[0]), float(zoom_in_range[1]))
        self.zoom_out_range = (float(zoom_out_range[0]), float(zoom_out_range[1]))
        self.rng = rng if rng is not None else np.random.default_rng()

    # -- parameter sampling -------------------------------------------------

    def _uniform(self, low: float, high: float) -> float:
        if high == low:
            return low
        return float(self.rng.uniform(low, high))

    def _randomize(self, sensor_hw: Tuple[int, int]) -> _AugmentationState:
        height, width = sensor_hw

        apply_h_flip = self.prob_hflip > float(self.rng.random())

        rotation_active = self.rotate_prob > float(self.rng.random())
        rotation_angle_deg = 0.0
        if rotation_active:
            sign = 1.0 if self.rng.random() >= 0.5 else -1.0
            rotation_angle_deg = sign * self._uniform(
                self.rotate_min_deg, self.rotate_max_deg
            )

        do_zoom = self.zoom_prob > float(self.rng.random())
        # Categorical over [zoom_in, zoom_out]; index 0 is zoom-in.
        weights = np.array(
            [self.zoom_in_weight, self.zoom_out_weight], dtype=np.float64
        )
        probs = weights / weights.sum()
        do_zoom_in = int(self.rng.choice(2, p=probs)) == 0
        do_zoom_out = not do_zoom_in
        do_zoom_in = do_zoom_in and do_zoom
        do_zoom_out = do_zoom_out and do_zoom

        zoom_out = _ZoomOutState(active=False, x0=0, y0=0, factor=1.0)
        if do_zoom_out:
            factor = self._uniform(*self.zoom_out_range)
            zoom_window_h = int(height / factor)
            zoom_window_w = int(width / factor)
            x0 = int(self._uniform(0, width - zoom_window_w))
            y0 = int(self._uniform(0, height - zoom_window_h))
            zoom_out = _ZoomOutState(active=True, x0=x0, y0=y0, factor=factor)

        # zoom-in factor sampled here; its label-aware window is sampled later,
        # once we know the most-recent non-empty frame (mirrors RVT).
        zoom_in_factor = 1.0
        if do_zoom_in:
            zoom_in_factor = self._uniform(*self.zoom_in_range)

        return _AugmentationState(
            apply_h_flip=apply_h_flip,
            rotation_active=rotation_active,
            rotation_angle_deg=rotation_angle_deg,
            apply_zoom_in=do_zoom_in,
            zoom_in_factor=zoom_in_factor,
            zoom_out=zoom_out,
        )

    # -- per-window tensor ops ---------------------------------------------

    @staticmethod
    def _flip_tensor(window: torch.Tensor) -> torch.Tensor:
        return torch.flip(window, dims=[-1])

    @staticmethod
    def _rotate_tensor(window: torch.Tensor, angle_deg: float) -> torch.Tensor:
        return tv_rotate(
            window, angle=angle_deg, interpolation=InterpolationMode.NEAREST
        )

    @staticmethod
    def _zoom_out_tensor(
        window: torch.Tensor, x0: int, y0: int, factor: float
    ) -> torch.Tensor:
        height, width = window.shape[-2:]
        zoom_window_h = int(height / factor)
        zoom_window_w = int(width / factor)
        shrunk = interpolate(
            window.unsqueeze(0).float(),
            size=(zoom_window_h, zoom_window_w),
            mode="nearest-exact",
        )[0].to(window.dtype)
        out = torch.zeros_like(window)
        out[:, y0 : y0 + zoom_window_h, x0 : x0 + zoom_window_w] = shrunk
        return out

    @staticmethod
    def _zoom_in_tensor(
        window: torch.Tensor, x0: int, y0: int, factor: float
    ) -> torch.Tensor:
        height, width = window.shape[-2:]
        zoom_window_h = int(height / factor)
        zoom_window_w = int(width / factor)
        crop = window[..., y0 : y0 + zoom_window_h, x0 : x0 + zoom_window_w].unsqueeze(
            0
        )
        out = interpolate(crop.float(), size=(height, width), mode="nearest-exact")[
            0
        ].to(window.dtype)
        return out

    # -- per-window box ops (top-left math, mirrors RVT ObjectLabels) -------

    @staticmethod
    def _flip_box(box: torch.Tensor, width: int) -> Optional[torch.Tensor]:
        tl = _centre_to_topleft(box)
        cls, x, y, w, h = tl[:, 0], tl[:, 1], tl[:, 2], tl[:, 3], tl[:, 4]
        x = (width - 1) - x - w  # flip_lr_
        return _topleft_to_centre(cls, x, y, w, h)

    @staticmethod
    def _rotate_box(
        box: torch.Tensor, sensor_hw: Tuple[int, int], angle_deg: float
    ) -> Optional[torch.Tensor]:
        height, width = sensor_hw
        tl = _centre_to_topleft(box)
        cls, x, y, w, h = tl[:, 0], tl[:, 1], tl[:, 2], tl[:, 3], tl[:, 4]

        p00 = torch.stack((x, y), dim=1)
        p10 = torch.stack((x + w, y), dim=1)
        p01 = torch.stack((x, y + h), dim=1)
        p11 = torch.stack((x + w, y + h), dim=1)
        points = torch.stack((p00, p10, p01, p11), dim=0)  # 4 x N x 2

        cx = width // 2
        cy = height // 2
        centre = torch.tensor([cx, cy], dtype=points.dtype)

        angle_rad = angle_deg / 180.0 * math.pi
        rot = torch.tensor(
            [
                [math.cos(angle_rad), math.sin(angle_rad)],
                [-math.sin(angle_rad), math.cos(angle_rad)],
            ],
            dtype=points.dtype,
        )
        points = points - centre
        points = torch.einsum("ij,pnj->pni", rot, points)
        points = points + centre

        x0 = torch.clamp(torch.min(points[..., 0], dim=0)[0], min=0, max=width - 1)
        y0 = torch.clamp(torch.min(points[..., 1], dim=0)[0], min=0, max=height - 1)
        x1 = torch.clamp(torch.max(points[..., 0], dim=0)[0], min=0, max=width - 1)
        y1 = torch.clamp(torch.max(points[..., 1], dim=0)[0], min=0, max=height - 1)

        new_x = x0
        new_y = y0
        new_w = x1 - x0
        new_h = y1 - y0
        keep = (new_w > 0) & (new_h > 0)  # remove_flat_labels_
        return _topleft_to_centre(
            cls[keep], new_x[keep], new_y[keep], new_w[keep], new_h[keep]
        )

    @staticmethod
    def _scale_box(
        cls, x, y, w, h, sensor_hw: Tuple[float, float], multiplier: float
    ) -> Tuple:
        """RVT ObjectLabels.scale_: scale in place and drop flat boxes.

        Returns the new (cls, x, y, w, h, new_sensor_hw).
        """
        img_ht, img_wd = sensor_hw
        new_img_ht = multiplier * img_ht
        new_img_wd = multiplier * img_wd
        x1 = torch.clamp((x + w) * multiplier, max=new_img_wd - 1)
        y1 = torch.clamp((y + h) * multiplier, max=new_img_ht - 1)
        x = x * multiplier
        y = y * multiplier
        w = x1 - x
        h = y1 - y
        keep = (w > 0) & (h > 0)
        return (
            cls[keep],
            x[keep],
            y[keep],
            w[keep],
            h[keep],
            (new_img_ht, new_img_wd),
        )

    @classmethod
    def _zoom_out_box(
        cls_self,
        box: torch.Tensor,
        sensor_hw: Tuple[int, int],
        x0: int,
        y0: int,
        factor: float,
    ) -> Optional[torch.Tensor]:
        tl = _centre_to_topleft(box)
        cls, x, y, w, h = tl[:, 0], tl[:, 1], tl[:, 2], tl[:, 3], tl[:, 4]
        h_orig, w_orig = sensor_hw
        cls, x, y, w, h, _ = cls_self._scale_box(
            cls, x, y, w, h, (h_orig, w_orig), 1.0 / factor
        )
        # input_size restored to (h_orig, w_orig); translate by (x0, y0).
        x = x + x0
        y = y + y0
        return _topleft_to_centre(cls, x, y, w, h)

    @classmethod
    def _zoom_in_box(
        cls_self,
        box: torch.Tensor,
        sensor_hw: Tuple[int, int],
        z_x0: int,
        z_y0: int,
        factor: float,
    ) -> Optional[torch.Tensor]:
        tl = _centre_to_topleft(box)
        cls, x, y, w, h = tl[:, 0], tl[:, 1], tl[:, 2], tl[:, 3], tl[:, 4]
        h_orig, w_orig = sensor_hw
        zoom_window_h = h_orig / factor
        zoom_window_w = w_orig / factor
        z_x1 = min(z_x0 + zoom_window_w, w_orig - 1)
        z_y1 = min(z_y0 + zoom_window_h, h_orig - 1)

        x0 = torch.clamp(x, min=z_x0, max=z_x1 - 1)
        y0 = torch.clamp(y, min=z_y0, max=z_y1 - 1)
        x1 = torch.clamp(x + w, min=z_x0, max=z_x1 - 1)
        y1 = torch.clamp(y + h, min=z_y0, max=z_y1 - 1)

        new_x = x0 - z_x0
        new_y = y0 - z_y0
        new_w = x1 - x0
        new_h = y1 - y0
        keep = (new_w > 0) & (new_h > 0)  # remove_flat_labels_
        cls, new_x, new_y, new_w, new_h = (
            cls[keep],
            new_x[keep],
            new_y[keep],
            new_w[keep],
            new_h[keep],
        )
        # scale_ back to full resolution by factor.
        cls, new_x, new_y, new_w, new_h, _ = cls_self._scale_box(
            cls, new_x, new_y, new_w, new_h, (zoom_window_h, zoom_window_w), factor
        )
        return _topleft_to_centre(cls, new_x, new_y, new_w, new_h)

    # -- label-aware zoom-in window sampling (mirrors RVT) ------------------

    def _sample_zoom_in_window(
        self,
        box: torch.Tensor,
        sensor_hw: Tuple[int, int],
        zoom_window_h: int,
        zoom_window_w: int,
    ) -> Tuple[int, int]:
        height, width = sensor_hw
        tl = _centre_to_topleft(box)
        candidates: List[Tuple[int, int]] = []
        for idx in range(tl.shape[0]):
            x0_l = float(tl[idx, 1])
            y0_l = float(tl[idx, 2])
            w_l = float(tl[idx, 3])
            h_l = float(tl[idx, 4])
            candidates.append(
                self._zoom_window_from_label_rectangle(
                    x0_l,
                    y0_l,
                    w_l,
                    h_l,
                    height,
                    width,
                    zoom_window_h,
                    zoom_window_w,
                )
            )
        assert len(candidates) > 0
        if len(candidates) == 1:
            sample_idx = 0
        else:
            # RVT: th.randint(low=0, high=len(candidates) - 1). The upper bound
            # is exclusive, matching the (arguably off-by-one) RVT behaviour.
            sample_idx = int(self.rng.integers(0, len(candidates) - 1))
        return candidates[sample_idx]

    def _zoom_window_from_label_rectangle(
        self,
        x0_l: float,
        y0_l: float,
        w_l: float,
        h_l: float,
        input_height: int,
        input_width: int,
        zoom_window_height: int,
        zoom_window_width: int,
    ) -> Tuple[int, int]:
        x1_l = x0_l + w_l
        y1_l = y0_l + h_l

        x0_valid = max(x1_l - max(zoom_window_width, w_l), 0)
        y0_valid = max(y1_l - max(zoom_window_height, h_l), 0)
        x1_valid = min(x0_l + max(zoom_window_width, w_l), input_width - 1)
        y1_valid = min(y0_l + max(zoom_window_height, h_l), input_height - 1)

        x1_valid = max(x1_valid - zoom_window_width, x0_valid)
        y1_valid = max(y1_valid - zoom_window_height, y0_valid)

        x_topleft = int(self._uniform(x0_valid, x1_valid))
        y_topleft = int(self._uniform(y0_valid, y1_valid))
        return x_topleft, y_topleft

    @staticmethod
    def _most_recent_nonempty(
        sample: SequenceSample,
    ) -> Optional[Tuple[int, torch.Tensor]]:
        for idx in range(len(sample.labels) - 1, -1, -1):
            if sample.is_padded_mask[idx]:
                continue
            box = sample.labels[idx]
            if box is not None and box.shape[0] > 0:
                return idx, box
        return None

    # -- public entry point -------------------------------------------------

    def __call__(self, sample: SequenceSample) -> SequenceSample:
        if len(sample.ev_repr) == 0:
            return sample
        sensor_hw = tuple(sample.ev_repr[0].shape[-2:])
        state = self._randomize(sensor_hw)

        # The zoom-in window is label-aware: sample it once from the most-recent
        # non-empty frame and reuse it for every window (mirrors RVT).
        zoom_in_window: Optional[Tuple[int, int]] = None
        if state.apply_zoom_in and state.zoom_in_factor != 1.0:
            latest = self._most_recent_nonempty(sample)
            if latest is None:
                # RVT warns and skips zoom-in when no labels are present.
                state.apply_zoom_in = False
            else:
                _, latest_box = latest
                height, width = sensor_hw
                zoom_window_h = int(height / state.zoom_in_factor)
                zoom_window_w = int(width / state.zoom_in_factor)
                zoom_in_window = self._sample_zoom_in_window(
                    latest_box, sensor_hw, zoom_window_h, zoom_window_w
                )

        new_ev_repr: List[torch.Tensor] = []
        new_labels: List[Optional[torch.Tensor]] = []

        for idx, window in enumerate(sample.ev_repr):
            if sample.is_padded_mask[idx]:
                new_ev_repr.append(window)
                new_labels.append(sample.labels[idx])
                continue

            box = sample.labels[idx]
            win = window

            if state.apply_h_flip:
                win = self._flip_tensor(win)
                if box is not None:
                    box = self._flip_box(box, sensor_hw[1])

            if state.rotation_active:
                win = self._rotate_tensor(win, state.rotation_angle_deg)
                if box is not None:
                    box = self._rotate_box(box, sensor_hw, state.rotation_angle_deg)

            if state.apply_zoom_in and zoom_in_window is not None:
                z_x0, z_y0 = zoom_in_window
                win = self._zoom_in_tensor(win, z_x0, z_y0, state.zoom_in_factor)
                if box is not None:
                    box = self._zoom_in_box(
                        box, sensor_hw, z_x0, z_y0, state.zoom_in_factor
                    )
            elif state.zoom_out.active:
                z = state.zoom_out
                win = self._zoom_out_tensor(win, z.x0, z.y0, z.factor)
                if box is not None:
                    box = self._zoom_out_box(box, sensor_hw, z.x0, z.y0, z.factor)

            new_ev_repr.append(win)
            new_labels.append(box)

        return SequenceSample(
            ev_repr=new_ev_repr,
            labels=new_labels,
            is_first_sample=sample.is_first_sample,
            is_padded_mask=list(sample.is_padded_mask),
        )
