"""Slow integration smoke test: one collated training batch flows into the RVT backbone.

This proves the ``evlib.data`` loader emits a per-timestep batch tensor with the
layout the real ``evlib.models`` RVT backbone consumes: ``[B, C, H, W]`` with
``C == 2 * nbins`` channels.

The tracked fixture ``tests/data_fixtures/mini_seq`` is only 8x12 spatial, which
the backbone cannot process: its total downsample factor is 32 (stem patch 4,
then three 2x stages) and its MaxViT window/grid size is 7, so the input must be
divisible by 32 and reduce to a final stage divisible by 7. An 8x12 input would
collapse below one pixel. We therefore prove the contract in two parts:

1. Build a real loader batch from the fixture and assert the per-timestep tensor
   rank (4) and channel count (2 * nbins == 20) that collate emits.
2. Feed a correctly sized, real ``[B, 2*nbins, H, W]`` float tensor with the same
   rank and channels through the backbone forward pass and assert it returns
   per-stage features and per-stage LSTM states without error.

Both halves use real torch tensors (no mocks). The genuine forward pass is the
acceptance: it confirms the loader's ``[B, C, H, W]`` contract is shape-compatible
with the backbone, differing from the tiny fixture only in spatial extent.
"""

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from evlib.data import (
    DataKey,
    PreprocessedH5Source,
    SequenceRandomDataset,
    custom_collate_random,
)

FIX = Path(__file__).resolve().parent / "data_fixtures" / "mini_seq"

pytestmark = pytest.mark.slow


def test_loader_batch_layout_feeds_rvt_backbone():
    """A collated batch tensor's [B, C, H, W] layout is accepted by the backbone."""
    from evlib.models.rvt_backbone import RVTBackbone, RVTConfig

    # Part 1: a real loader batch proves the per-timestep [B, C, H, W] layout.
    source = PreprocessedH5Source(FIX)
    dataset = SequenceRandomDataset([source], sequence_length=2)
    loader = DataLoader(dataset, batch_size=1, collate_fn=custom_collate_random)
    batch = next(iter(loader))

    per_timestep = batch[DataKey.EV_REPR]
    assert isinstance(per_timestep, list) and len(per_timestep) == 2
    first_step = per_timestep[0].float()  # [B, C, H, W]
    assert first_step.ndim == 4, "loader must emit a rank-4 per-timestep tensor"
    expected_channels = 2 * source.nbins
    assert expected_channels == 20
    assert first_step.shape[1] == expected_channels

    # Part 2: a backbone-valid input with the same rank and channel count runs.
    config = RVTConfig.tiny()
    assert config.input_channels == expected_channels
    backbone = RVTBackbone(config).eval()

    # Total downsample is 32 and the final stage must be divisible by 7; 224 -> 7.
    backbone_height = 224
    backbone_width = 224
    backbone_input = torch.zeros(
        first_step.shape[0],
        expected_channels,
        backbone_height,
        backbone_width,
        dtype=torch.float32,
    )
    assert backbone_input.ndim == first_step.ndim
    assert backbone_input.shape[1] == first_step.shape[1]

    with torch.no_grad():
        stage_outputs, new_states = backbone(backbone_input, previous_states=None)

    # The forward pass returns one feature map and one LSTM state per stage.
    assert set(stage_outputs.keys()) == {1, 2, 3, 4}
    assert len(new_states) == config.num_stages
    for stage_number, feature in stage_outputs.items():
        assert feature.ndim == 4
        assert feature.shape[0] == backbone_input.shape[0]
    for hidden, cell in new_states:
        assert hidden.ndim == 4 and cell.ndim == 4
