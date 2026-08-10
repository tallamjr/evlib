"""E2VIDRecurrent must load the bundled recurrent checkpoint completely."""

import numpy as np
import pytest
import torch

from evlib.models.e2vid_recurrent import E2VIDRecurrent

# e2vid-lite.pth (bundled under python/evlib/models/weights/) is gitignored, so
# it is present locally but absent on CI runners. Skip the whole module when
# missing, matching tests/test_rvt_weight_load.py for the rvt-t.ckpt case.
pytestmark = pytest.mark.skipif(
    not E2VIDRecurrent.weights_path().exists(),
    reason="e2vid-lite.pth is gitignored (local-only); skipping weight-load checks",
)


def _weights_path():
    return E2VIDRecurrent.weights_path()


def _random_events(count, height, width, seed):
    rng = np.random.default_rng(seed)
    return (
        rng.integers(0, width, count),
        rng.integers(0, height, count),
        np.sort(rng.uniform(0, 0.05, count)),
        rng.choice([-1, 1], count),
    )


def test_checkpoint_loads_strict():
    model = E2VIDRecurrent(pretrained=True)
    ckpt = torch.load(_weights_path(), map_location="cpu", weights_only=True)
    state = ckpt.get("state_dict", ckpt)
    stripped = {k.removeprefix("unetrecurrent."): v for k, v in state.items()}
    missing, unexpected = model._model.load_state_dict(stripped, strict=False)
    assert missing == [] and unexpected == []


def test_reconstruct_carries_state():
    model = E2VIDRecurrent(pretrained=True)
    events = _random_events(5000, height=48, width=64, seed=7)
    frame1, state = model.reconstruct(events, height=48, width=64, state=None)
    frame2, state2 = model.reconstruct(events, height=48, width=64, state=state)
    assert frame1.shape == (48, 64)
    assert state is not None and state2 is not None
    assert not np.allclose(frame1, frame2)  # state changes the output


def test_reconstruct_returns_float32_in_unit_range():
    model = E2VIDRecurrent(pretrained=True)
    events = _random_events(5000, height=48, width=64, seed=11)
    frame, _ = model.reconstruct(events, height=48, width=64)
    assert frame.dtype == np.float32
    assert frame.min() >= 0.0 and frame.max() <= 1.0


def test_state_holds_one_hidden_cell_pair_per_encoder():
    model = E2VIDRecurrent(pretrained=True)
    events = _random_events(5000, height=48, width=64, seed=13)
    _, state = model.reconstruct(events, height=48, width=64)
    assert len(state) == model.num_encoders
    for encoder_index, (hidden, cell) in enumerate(state):
        channels = model.base_num_channels * 2 ** (encoder_index + 1)
        expected = (1, channels, 48 >> (encoder_index + 1), 64 >> (encoder_index + 1))
        assert tuple(hidden.shape) == expected
        assert tuple(cell.shape) == expected


def test_architecture_is_read_from_the_checkpoint():
    ckpt = torch.load(_weights_path(), map_location="cpu", weights_only=True)
    state = ckpt.get("state_dict", ckpt)
    head_weight = state["unetrecurrent.head.conv2d.weight"]

    model = E2VIDRecurrent(pretrained=True)
    assert model.num_bins == head_weight.shape[1]
    assert model.base_num_channels == head_weight.shape[0]
    assert model.num_encoders == sum(
        1 for key in state if key.endswith(".conv.conv2d.weight")
    )


def test_out_of_range_coordinates_raise():
    model = E2VIDRecurrent(pretrained=True)
    events = (
        np.array([0, 64], dtype=np.int64),
        np.array([0, 10], dtype=np.int64),
        np.array([0.0, 0.01]),
        np.array([1, -1], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="outside the 64x48 sensor"):
        model.reconstruct(events, height=48, width=64)
