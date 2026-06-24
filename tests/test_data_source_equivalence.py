"""Source-equivalence gate: on-the-fly windows must match the preprocessed h5.

This proves that ``EvlibStreamSource`` (which builds the dense ``[C, H, W]`` uint8
windows on the fly from the raw event h5 via ``evlib.rvt`` window assignment and the
Rust ``stacked_histogram_dense`` kernel) is byte-identical to ``PreprocessedH5Source``
(which reads the precomputed RVT representation h5) on the real gen4 sequence.

Slow- and local-only: the gen4 RVT datasets are gitignored and absent in CI, so the
test skips cleanly when the raw input is not present.
"""

import numpy as np
import pytest

pytest.importorskip("h5py")
pytest.importorskip("hdf5plugin")

from tests.rvt_fixtures import REF, raw_input_path, requires_reference_data


@pytest.mark.slow
@requires_reference_data
def test_evlib_stream_source_matches_preprocessed_h5():
    from evlib.data.sources import EvlibStreamSource, PreprocessedH5Source

    h5_src = PreprocessedH5Source(REF)
    stream_src = EvlibStreamSource(raw_h5=raw_input_path(), seq_dir=REF)

    assert stream_src.engine_used == "rust-cpu"
    assert stream_src.window_count() == h5_src.window_count()

    # The first few windows are enough to prove byte identity; reading the full
    # sequence is unnecessary for the gate.
    n = min(h5_src.window_count(), 8)
    assert n > 0

    ev_h5, _ = h5_src.read_windows(0, n)
    ev_stream, _ = stream_src.read_windows(0, n)

    assert len(ev_h5) == len(ev_stream) == n
    for i, (a, b) in enumerate(zip(ev_h5, ev_stream)):
        assert a.dtype == b.dtype
        assert a.shape == b.shape
        assert np.array_equal(a.numpy(), b.numpy()), f"window {i} differs"
