"""RVT gen4 sequence-eval harness: run the detector over a val split, score mAP.

This wires the streaming loader, the RVT forward, the yolox postprocess, the
Task-2 converters and the ported Prophesee evaluator into one function,
``evaluate_rvt_gen4``, mirroring the reference validation step
(``lib/ssms_event_cameras/RVT/modules/detection.py:272-385``):

- a streaming sampler over the val sequences (every GT-bearing frame is scored,
  not just the last of each chunk);
- recurrent state reset on ``is_first_sample`` and detached/carried across steps;
- the input padded BOTTOM/RIGHT to the backbone's required multiple so box
  coordinates keep their top-left origin and stay aligned with the unpadded GT;
- only frames carrying GT are scored (the reference ``len(labels) > 0`` gate);
- ``forward_detect`` + ``postprocess(num_classes, conf, nms)`` on the selected
  frames, converted to Prophesee ``BBOX_DTYPE`` with each GT frame's timestamp;
- ``evaluate_buffer`` called with the PADDED HW, returning the metrics dict.

Model construction (loading ``rvt-t.ckpt``) is the caller's job; this function
takes a constructed model so a tiny random model can drive the loop in tests.

GT pairing: the streamed batch dict carries the centre-yolox training labels but
not the on-disk timestamp the evaluator needs, nor source/window provenance. We
therefore drive the per-slot streams here (mirroring
``evlib.data.dataset_stream`` ordering exactly) and carry, per timestep per slot,
the originating ``(source, repr_idx)`` so the raw GT rows (with ``t``) can be
fetched via ``source.read_window_gt`` and converted with their true timestamp.

Backbone multiple: evlib's ``RVT.forward_backbone`` does NOT pad internally (the
reference pads via ``InputPadderFromShape`` before calling it). We pad here,
bottom/right, to ``padded_hw`` and pass that HW to ``evaluate_buffer``. For gen4
ds2 the data HW is 360x640 and ``padded_hw`` is 384x640.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from evlib.data.collate import custom_collate_stream
from evlib.data.sequence import DataKey, SequenceSample
from evlib.data.sources import ReprSource
from evlib.eval.convert import gt_rows_to_prophesee, preds_to_prophesee
from evlib.eval.prophesee import PropheseeEvaluator
from evlib.models.yolox_head import postprocess

# Per timestep per slot: the originating source and repr window index, or None
# for a padded/exhausted slot frame that has no on-disk GT.
Provenance = Optional[Tuple[ReprSource, int]]


def _stream_with_provenance(
    sources: List[ReprSource],
    sequence_length: int,
    batch_size: int,
):
    """Yield ``(batch_slot, provenance)`` steps mirroring SequenceStreamDataset.

    ``batch_slot`` is the list of ``batch_size`` ``SequenceSample`` (real or
    padded) for one streaming step; ``provenance`` is a parallel list of length
    ``batch_size``, each a list of length ``sequence_length`` of ``(source,
    repr_idx)`` for a real frame or ``None`` for a padded frame.

    Ordering is identical to ``evlib.data.dataset_stream``: sources are assigned
    round-robin to slots and each slot's sources are streamed in order in chunks
    of ``sequence_length``; a trailing short chunk is bottom-padded; an exhausted
    slot is filled with an all-padded placeholder so every step is full width.
    """
    L = sequence_length
    slots: List[List[ReprSource]] = [[] for _ in range(batch_size)]
    for idx, s in enumerate(sources):
        slots[idx % batch_size].append(s)

    def slot_iter(slot_sources):
        for s in slot_sources:
            n = s.window_count()
            first = True
            for start in range(0, n, L):
                hi = min(start + L, n)
                ev, labels = s.read_windows(start, hi)
                real = hi - start
                pad = L - real
                is_padded = [False] * real + [True] * pad
                prov: List[Provenance] = [(s, start + i) for i in range(real)] + [
                    None
                ] * pad
                if pad:
                    zero = torch.zeros_like(ev[0])
                    ev = ev + [zero.clone() for _ in range(pad)]
                    labels = labels + [None] * pad
                sample = SequenceSample(
                    ev, labels, is_first_sample=first, is_padded_mask=is_padded
                )
                yield sample, prov
                first = False

    iters = [slot_iter(s) for s in slots]
    pad_zeros: Optional[List[torch.Tensor]] = None

    while True:
        batch_slot: List[Optional[SequenceSample]] = [None] * batch_size
        prov_slot: List[Optional[List[Provenance]]] = [None] * batch_size
        any_real = False
        for j in range(batch_size):
            try:
                sample, prov = next(iters[j])
            except StopIteration:
                continue
            batch_slot[j] = sample
            prov_slot[j] = prov
            any_real = True
            if pad_zeros is None:
                pad_zeros = [torch.zeros_like(t) for t in sample.ev_repr]
        if not any_real:
            return
        assert pad_zeros is not None
        n = len(pad_zeros)
        for j in range(batch_size):
            if batch_slot[j] is None:
                batch_slot[j] = SequenceSample(
                    [z.clone() for z in pad_zeros],
                    [None] * n,
                    is_first_sample=False,
                    is_padded_mask=[True] * n,
                )
                prov_slot[j] = [None] * n
        yield batch_slot, prov_slot


def _pad_bottom_right(tensor: torch.Tensor, padded_hw: Tuple[int, int]) -> torch.Tensor:
    """Pad ``tensor`` (``[..., H, W]``) bottom/right to ``padded_hw`` with zeros.

    Padding is bottom/right only (corner type), so box coordinates keep their
    top-left origin and stay aligned with the unpadded GT, matching the reference
    ``InputPadderFromShape(type="corner")``.
    """
    h, w = tensor.shape[-2:]
    des_h, des_w = padded_hw
    if h > des_h or w > des_w:
        raise ValueError(
            f"input HW ({h},{w}) exceeds padded HW ({des_h},{des_w}); "
            "padded HW must be >= the data HW"
        )
    if des_h % 4 != 0 or des_w % 4 != 0:
        raise ValueError(f"padded HW {padded_hw} must be divisible by 4")
    # F.pad order is (left, right, top, bottom): pad right and bottom only.
    return F.pad(tensor, [0, des_w - w, 0, des_h - h], mode="constant", value=0)


def evaluate_rvt_gen4(
    model,
    sources: List[ReprSource],
    *,
    sequence_length: int = 10,
    batch_size: int = 8,
    conf_thre: float = 0.001,
    nms_thre: float = 0.45,
    num_classes: int = 3,
    padded_hw: Tuple[int, int] = (384, 640),
    device: Optional[str] = None,
) -> Dict:
    """Run RVT over the gen4 val ``sources`` and return the Prophesee metrics.

    Args:
        model: a constructed ``evlib.models.rvt.RVT`` (weights loaded by the
            caller); only the loop is implemented here.
        sources: ``ReprSource`` objects (e.g. ``PreprocessedH5Source``) over the
            val sequence dirs, each also exposing ``read_window_gt``.
        sequence_length: streaming chunk length T (RVT eval uses 10).
        batch_size: number of parallel streaming slots.
        conf_thre, nms_thre, num_classes: yolox postprocess params (gen4 eval
            uses 0.001 / 0.45 / 3).
        padded_hw: the backbone-multiple HW the input is bottom/right padded to;
            also the HW passed to ``evaluate_buffer`` (gen4 ds2: 384x640).
        device: torch device string; defaults to the model's device.

    Returns:
        The Prophesee evaluator metrics dict (caller reads ``"AP"``).
    """
    dev = (
        torch.device(device) if device is not None else next(model.parameters()).device
    )
    model = model.to(dev)
    model.eval()

    evaluator = PropheseeEvaluator(dataset="gen4", downsample_by_2=True)

    prev_states: Optional[List] = None

    with torch.no_grad():
        for batch_slot, prov_slot in _stream_with_provenance(
            sources, sequence_length, batch_size
        ):
            batch = custom_collate_stream(batch_slot)
            ev_repr: List[torch.Tensor] = batch[DataKey.EV_REPR]  # T x [B, C, H, W]
            is_first: List[bool] = batch[DataKey.IS_FIRST_SAMPLE]

            # Reset recurrent state for slots whose stream just started. The first
            # step of a fresh source must not carry the previous source's state.
            if prev_states is None or any(is_first):
                if prev_states is None or all(is_first):
                    prev_states = None
                else:
                    prev_states = _reset_states_for_slots(prev_states, is_first)

            T = len(ev_repr)
            # Forward the T-stack one timestep at a time, carrying states. evlib's
            # forward_backbone takes [B, C, H, W]; the reference stacks (L,B,...)
            # because its backbone unrolls internally. We unroll here.
            for tidx in range(T):
                step = ev_repr[tidx].to(dev).to(torch.float32)
                step = _pad_bottom_right(step, padded_hw)
                features, prev_states = model.forward_backbone(
                    step, previous_states=prev_states
                )

                # Score only slots whose frame at this timestep carries GT.
                gt_by_slot: List[Optional] = []
                selected_slots: List[int] = []
                for slot in range(len(prov_slot)):
                    prov = prov_slot[slot]
                    entry = prov[tidx] if prov is not None else None
                    if entry is None:
                        continue
                    src, repr_idx = entry
                    rows = src.read_window_gt(repr_idx)
                    if rows is None or len(rows) == 0:
                        continue
                    selected_slots.append(slot)
                    gt_by_slot.append(rows)

                if not selected_slots:
                    continue

                # Run detection only on the selected slots' features.
                selected_features = {
                    stage: feat[selected_slots] for stage, feat in features.items()
                }
                predictions, _ = model.forward_detect(selected_features)
                processed = postprocess(
                    predictions,
                    num_classes=num_classes,
                    conf_thre=conf_thre,
                    nms_thre=nms_thre,
                )

                labels_proph = []
                preds_proph = []
                for local_idx, rows in enumerate(gt_by_slot):
                    frame_t = int(rows["t"][0])
                    labels_proph.append(gt_rows_to_prophesee(rows, frame_t=frame_t))
                    preds_proph.append(
                        preds_to_prophesee(processed[local_idx], frame_t=frame_t)
                    )
                evaluator.add_labels(labels_proph)
                evaluator.add_predictions(preds_proph)

            # Detach carried states so the graph is not retained across steps.
            prev_states = _detach_states(prev_states)

    metrics = evaluator.evaluate_buffer(img_height=padded_hw[0], img_width=padded_hw[1])
    return metrics


def _reset_states_for_slots(prev_states, is_first: List[bool]):
    """Zero the recurrent state for the batch slots flagged is_first.

    ``prev_states`` is a list (one per backbone stage) of ``(hidden, cell)``
    tuples shaped ``[B, ...]``. The reference resets per-slot by index; we zero
    the rows of the flagged slots so a fresh source starts from zero state while
    continuing slots keep theirs.
    """
    mask = torch.tensor(is_first, dtype=torch.bool)
    new_states = []
    for stage_state in prev_states:
        if stage_state is None:
            new_states.append(None)
            continue
        hidden, cell = stage_state
        hidden = hidden.clone()
        cell = cell.clone()
        slot_mask = mask.to(hidden.device)
        hidden[slot_mask] = 0
        cell[slot_mask] = 0
        new_states.append((hidden, cell))
    return new_states


def _detach_states(prev_states):
    """Detach carried recurrent states to break the autograd graph across steps."""
    if prev_states is None:
        return None
    detached = []
    for stage_state in prev_states:
        if stage_state is None:
            detached.append(None)
            continue
        hidden, cell = stage_state
        detached.append((hidden.detach(), cell.detach()))
    return detached
