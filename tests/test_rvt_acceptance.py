import resource
import subprocess
import sys
import textwrap
import h5py
import numpy as np
import pytest
import hdf5plugin  # noqa: F401
from pathlib import Path
from tests.rvt_fixtures import raw_input_path, ref_repr_h5, ref_timestamps
from evlib.rvt.pipeline import process_sequence


@pytest.mark.slow
def test_fast_slice_first_20_windows(tmp_path):
    grid = np.load(ref_timestamps())[:20]
    out_h5 = process_sequence(
        raw_input_path(),
        tmp_path / "out20",
        dataset="gen4",
        height=720,
        width=1280,
        ev_repr_timestamps_us=grid,
        downsample_by_2=True,
        engine="streaming",
    )
    with h5py.File(out_h5, "r") as f_ours, h5py.File(ref_repr_h5(), "r") as f_ref:
        for i in range(20):
            assert np.array_equal(f_ours["data"][i], f_ref["data"][i]), (
                f"window {i} differs"
            )


@pytest.mark.slow
def test_full_sequence_matches_rvt_reference(tmp_path):
    grid = np.load(ref_timestamps())
    out_h5 = process_sequence(
        raw_input_path(),
        tmp_path / "out",
        dataset="gen4",
        height=720,
        width=1280,
        ev_repr_timestamps_us=grid,
        downsample_by_2=True,
        engine="streaming",
    )
    with h5py.File(out_h5, "r") as f_ours, h5py.File(ref_repr_h5(), "r") as f_ref:
        ours = f_ours["data"]
        ref = f_ref["data"]
        assert ours.shape == ref.shape == (1198, 20, 360, 640)
        assert ours.dtype == ref.dtype == np.uint8
        for i in range(ref.shape[0]):
            assert np.array_equal(ours[i], ref[i]), f"window {i} differs"


@pytest.mark.slow
def test_full_sequence_rust_backend_matches_reference(tmp_path):
    """The Rust dense scatter-add backend must be bit-identical to the RVT reference.

    Runs ``process_sequence(..., backend="rust")`` over the full Gen4 validation
    sequence (raw h5 read directly, no parquet) and asserts every one of the 1198
    windows equals the committed reference output element-for-element.
    """
    grid = np.load(ref_timestamps())
    out_h5 = process_sequence(
        raw_input_path(),
        tmp_path / "out_rust",
        dataset="gen4",
        height=720,
        width=1280,
        ev_repr_timestamps_us=grid,
        downsample_by_2=True,
        backend="rust",
    )
    with h5py.File(out_h5, "r") as f_ours, h5py.File(ref_repr_h5(), "r") as f_ref:
        ours = f_ours["data"]
        ref = f_ref["data"]
        assert ours.shape == ref.shape == (1198, 20, 360, 640)
        assert ours.dtype == ref.dtype == np.uint8
        for i in range(ref.shape[0]):
            assert np.array_equal(ours[i], ref[i]), f"window {i} differs"


@pytest.mark.slow
def test_streaming_peak_memory_bounded(tmp_path):
    """Full streaming run of process_sequence must stay within a bounded RSS.

    The window-batched orchestration reads only the events for each batch of
    windows, so peak memory is far below the ~14 GB of the old single global
    query. We spawn the work in a subprocess and read its high-water-mark RSS
    via RUSAGE_CHILDREN. ru_maxrss is bytes on darwin, KiB on Linux. With the
    default window_batch_size=10 the observed peak is ~3.6 GB, so we assert a
    6 GB ceiling with margin.
    """
    out_dir = tmp_path / "out_mem"
    script = textwrap.dedent(
        f"""
        import numpy as np
        from pathlib import Path
        from tests.rvt_fixtures import raw_input_path, ref_timestamps
        from evlib.rvt.pipeline import process_sequence

        grid = np.load(ref_timestamps())
        process_sequence(
            raw_input_path(),
            Path({str(out_dir)!r}),
            dataset="gen4",
            height=720,
            width=1280,
            ev_repr_timestamps_us=grid,
            downsample_by_2=True,
            engine="streaming",
        )
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stderr}"

    ru_maxrss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    peak_bytes = ru_maxrss if sys.platform == "darwin" else ru_maxrss * 1024
    peak_gb = peak_bytes / 1e9
    ceiling_gb = 6.0
    assert peak_gb < ceiling_gb, (
        f"peak RSS {peak_gb:.2f} GB exceeded {ceiling_gb} GB ceiling"
    )
