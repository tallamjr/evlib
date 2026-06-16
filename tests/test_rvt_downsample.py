import numpy as np
import pytest
from evlib.rvt.downsample import selected_source_indices


def test_selected_indices_match_torch_nearest_exact():
    torch = pytest.importorskip("torch")
    for in_size, out_size in [(720, 360), (1280, 640), (8, 4), (9, 3)]:
        ours = np.asarray(selected_source_indices(in_size, out_size))
        probe = torch.arange(in_size, dtype=torch.float32).reshape(1, 1, 1, in_size)
        out = torch.nn.functional.interpolate(
            probe, size=(1, out_size), mode="nearest-exact"
        )
        torch_idx = out.reshape(out_size).round().to(torch.int64).numpy()
        assert np.array_equal(ours, torch_idx), (in_size, out_size, ours, torch_idx)


def test_selected_indices_are_strictly_increasing_and_in_range():
    idx = selected_source_indices(720, 360)
    assert len(idx) == 360
    assert all(0 <= i < 720 for i in idx)
    assert all(idx[k] < idx[k + 1] for k in range(len(idx) - 1))
