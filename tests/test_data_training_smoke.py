"""Slow end-to-end smoke test for the phase-2 data loader through the backbone.

The phase-1 test (``tests/test_data_rvt_backbone_smoke.py``) already proves the
plain loader path emits a ``[B, C, H, W]`` (``C == 2*nbins``) per-timestep tensor
the RVT backbone accepts. This test does NOT duplicate that; it adds the three
phase-2 integration points that wire the augmentor and the Lightning DataModule
into the loader path, and that drive the backbone recurrently across timesteps:

1. ``SequenceAugmentor`` composed in the ``SequenceRandomDataset`` path: a
   post-augmentation collated batch is still rank-4 ``[B, C, H, W]`` with
   ``C == 2*nbins``, so augmentation preserves the backbone contract.
2. ``EventDataModule`` wiring: a batch pulled from ``train_dataloader()`` (built
   with a train-only augmentor) carries the same per-timestep layout, proving the
   DataModule path composes end to end.
3. Recurrent LSTM state carried across two timesteps: two correctly-sized real
   ``[B, 2*nbins, 224, 224]`` tensors are fed through the backbone, threading the
   first step's per-stage states into the second forward, the genuine recurrent
   training-step shape.

The tracked fixture ``tests/data_fixtures/mini_seq`` is only 8x12 spatial, which
the backbone cannot process (total downsample 32, final stage divisible by 7), so
the augmentor and DataModule run on the tiny fixture while the genuine backbone
forward uses a correctly-sized real tensor. Real torch tensors throughout, no
mocks. h5-backed bits are guarded by ``importorskip("h5py")`` and the Lightning
bit by ``importorskip("pytorch_lightning")`` so CI and Windows skip cleanly.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from evlib.data import (
    DataKey,
    PreprocessedH5Source,
    SequenceAugmentor,
    SequenceRandomDataset,
    custom_collate_random,
)

FIX = Path(__file__).resolve().parent / "data_fixtures" / "mini_seq"

pytestmark = pytest.mark.slow


def _assert_per_timestep_layout(batch, expected_channels, sequence_length):
    """A collated batch carries rank-4 [B, C, H, W] per-timestep tensors."""
    per_timestep = batch[DataKey.EV_REPR]
    assert isinstance(per_timestep, list)
    assert len(per_timestep) == sequence_length
    first_step = per_timestep[0].float()
    assert first_step.ndim == 4, "loader must emit a rank-4 per-timestep tensor"
    assert first_step.shape[1] == expected_channels
    return first_step


def test_augmentor_in_loader_path_preserves_backbone_contract():
    """SequenceAugmentor composed with collate keeps the [B, C, H, W] contract."""
    pytest.importorskip("h5py")

    source = PreprocessedH5Source(FIX)
    expected_channels = 2 * source.nbins
    assert expected_channels == 20

    # A deterministic, spatial-transform-heavy augmentor; runs fine at 8x12.
    augmentor = SequenceAugmentor(
        sampler="random",
        prob_hflip=1.0,
        rotate_prob=1.0,
        zoom_prob=1.0,
        rng=np.random.default_rng(0),
    )
    dataset = SequenceRandomDataset([source], sequence_length=2, augmentor=augmentor)
    loader = DataLoader(dataset, batch_size=1, collate_fn=custom_collate_random)
    batch = next(iter(loader))

    _assert_per_timestep_layout(batch, expected_channels, sequence_length=2)


def test_event_datamodule_train_dataloader_layout():
    """EventDataModule.train_dataloader() composes the augmentor end to end."""
    pytest.importorskip("h5py")
    pytest.importorskip("pytorch_lightning")
    from evlib.data import EventDataModule

    source = PreprocessedH5Source(FIX)
    expected_channels = 2 * source.nbins

    # The random sampler draws fresh params per __getitem__, so a plain random
    # augmentor is API-correct here (no stream-safety requirement).
    augmentor = SequenceAugmentor(sampler="random", rng=np.random.default_rng(1))
    datamodule = EventDataModule(
        train_sources=[source],
        val_sources=[source],
        sequence_length=2,
        batch_size=1,
        num_workers=0,
        sampling="random",
        augmentor=augmentor,
    )
    batch = next(iter(datamodule.train_dataloader()))

    _assert_per_timestep_layout(batch, expected_channels, sequence_length=2)


def test_backbone_carries_recurrent_state_across_timesteps():
    """Two timesteps run recurrently: states0 thread into the second forward."""
    from evlib.models.rvt_backbone import RVTBackbone, RVTConfig

    config = RVTConfig.tiny()
    expected_channels = config.input_channels
    assert expected_channels == 20
    backbone = RVTBackbone(config).eval()

    # Total downsample is 32 and the final stage must be divisible by 7; 224 -> 7.
    batch_size = 1
    height = 224
    width = 224
    step_zero = torch.zeros(
        batch_size, expected_channels, height, width, dtype=torch.float32
    )
    step_one = torch.ones(
        batch_size, expected_channels, height, width, dtype=torch.float32
    )

    with torch.no_grad():
        out0, states0 = backbone(step_zero, previous_states=None)
        out1, states1 = backbone(step_one, previous_states=states0)

    for stage_outputs in (out0, out1):
        assert set(stage_outputs.keys()) == {1, 2, 3, 4}
        for feature in stage_outputs.values():
            assert feature.ndim == 4
            assert feature.shape[0] == batch_size

    for new_states in (states0, states1):
        assert len(new_states) == config.num_stages
        for hidden, cell in new_states:
            assert hidden.ndim == 4 and cell.ndim == 4
