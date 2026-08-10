"""E2VIDRecurrent must load the bundled recurrent checkpoint completely."""

from pathlib import Path

import numpy as np
import torch

import evlib.models.e2vid as e2vid_mod


def _weights_path():
    return Path(e2vid_mod.__file__).parent / "weights" / "e2vid-lite.pth"


def test_checkpoint_loads_strict():
    from evlib.models import E2VIDRecurrent

    model = E2VIDRecurrent(pretrained=True)
    ckpt = torch.load(_weights_path(), map_location="cpu", weights_only=True)
    state = ckpt.get("state_dict", ckpt)
    stripped = {k.removeprefix("unetrecurrent."): v for k, v in state.items()}
    missing, unexpected = model._model.load_state_dict(stripped, strict=False)
    assert missing == [] and unexpected == []


def test_reconstruct_carries_state():
    from evlib.models import E2VIDRecurrent

    model = E2VIDRecurrent(pretrained=True)
    rng = np.random.default_rng(7)
    n = 5000
    events = (
        rng.integers(0, 64, n),
        rng.integers(0, 48, n),
        np.sort(rng.uniform(0, 0.05, n)),
        rng.choice([-1, 1], n),
    )
    frame1, state = model.reconstruct(events, height=48, width=64, state=None)
    frame2, state2 = model.reconstruct(events, height=48, width=64, state=state)
    assert frame1.shape == (48, 64)
    assert state is not None and state2 is not None
    assert not np.allclose(frame1, frame2)  # state changes the output
