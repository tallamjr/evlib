"""CUDA event simulator against the CPU kernel on real slider_depth frames.
Run: .venv/bin/pytest tests/test_simulation_cuda.py. Skips unless the crate was built with
`--features cuda` and EVLIB_CUDA_SIM_LIB points at libevsim.so on a CUDA machine.
Acceptance: CPU and CUDA DataFrames are identical after sorting by (t, y, x, polarity).
"""

import ctypes
import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import evlib
from evlib.simulation import ESIMConfig, ESIMSimulator, simulate_frames

ROOT = Path(__file__).resolve().parents[1]
SLIDER = ROOT / "data" / "slider_depth"

pytestmark = pytest.mark.skipif(
    not evlib.simulation_rs.cuda_available(),
    reason="CUDA simulator library not available",
)

SORT_KEYS = ["t", "y", "x", "polarity"]


@pytest.fixture(scope="module")
def slider_frames():
    from PIL import Image

    lines = (SLIDER / "images.txt").read_text().splitlines()
    frames, t_ns = [], []
    for line in lines:
        secs, rel = line.split()
        frames.append(np.asarray(Image.open(SLIDER / rel).convert("L"), dtype=np.uint8))
        t_ns.append(round(float(secs) * 1e9))
    return np.stack(frames), np.asarray(t_ns, dtype=np.int64)


def _sorted(df: pl.DataFrame) -> pl.DataFrame:
    return df.sort(SORT_KEYS)


@pytest.mark.parametrize(
    "config",
    [
        ESIMConfig(),
        ESIMConfig(threshold_sigma=0.2, refractory_period_ms=0.25, seed=3),
    ],
    ids=["default", "mismatch_refractory"],
)
def test_simulate_frames_cuda_matches_cpu_on_slider_depth(slider_frames, config):
    frames, t_ns = slider_frames
    cpu_cfg = ESIMConfig(**{**config.__dict__, "device": "cpu"})
    cuda_cfg = ESIMConfig(**{**config.__dict__, "device": "cuda"})
    cpu = _sorted(simulate_frames(frames, t_ns, cpu_cfg))
    cuda = _sorted(simulate_frames(frames, t_ns, cuda_cfg))
    assert cpu.height > 100_000
    assert cpu.schema == cuda.schema
    assert cpu.equals(cuda)


def test_stateful_cuda_simulator_matches_cpu_step_by_step(slider_frames):
    frames, t_ns = slider_frames
    height, width = frames.shape[1:]
    cpu = ESIMSimulator(ESIMConfig(device="cpu"), width=width, height=height)
    cuda = ESIMSimulator(ESIMConfig(device="cuda"), width=width, height=height)
    assert cuda.device == "cuda"
    np.testing.assert_array_equal(cpu.thresholds()[0], cuda.thresholds()[0])
    for frame, t in zip(frames[:20], t_ns[:20]):
        got = cuda.step_ns(frame, int(t))
        want = cpu.step_ns(frame, int(t))
        got_key = np.lexsort((got[3], got[0], got[1], got[2]))
        want_key = np.lexsort((want[3], want[0], want[1], want[2]))
        for g, w, k_g, k_w in zip(got, want, [got_key] * 4, [want_key] * 4):
            np.testing.assert_array_equal(g[k_g], w[k_w])
    assert cuda.is_initialised
    cuda.reset()
    assert not cuda.is_initialised


def test_auto_device_picks_cuda(slider_frames):
    frames, t_ns = slider_frames
    auto = _sorted(simulate_frames(frames[:5], t_ns[:5], ESIMConfig(device="auto")))
    cuda = _sorted(simulate_frames(frames[:5], t_ns[:5], ESIMConfig(device="cuda")))
    assert auto.equals(cuda)


def test_cuda_run_in_batches_matches_cpu_whole_stack(slider_frames):
    frames, t_ns = slider_frames
    height, width = frames.shape[1:]
    cuda = ESIMSimulator(ESIMConfig(device="cuda"), width=width, height=height)
    parts = [
        cuda.simulate(frames[a:b], t_ns[a:b], sort=False)
        for a, b in ((0, 8), (8, 9), (9, 20))
    ]
    joined = _sorted(pl.concat(parts))
    whole = _sorted(simulate_frames(frames[:20], t_ns[:20], ESIMConfig(device="cpu")))
    assert joined.equals(whole)


def test_cuda_sorted_output_is_time_ordered_for_u8_and_f32(slider_frames):
    frames, t_ns = slider_frames
    height, width = frames.shape[1:]
    log_frames = np.log(
        frames[:30].astype(np.float32) / np.float32(255.0) + np.float32(1e-3)
    )
    for stack in (frames[:30], np.ascontiguousarray(log_frames)):
        cuda = evlib.simulation_rs.EventSimulatorCuda(width=width, height=height)
        cpu = evlib.simulation_rs.EventSimulator(width=width, height=height)
        got = cuda.run(stack, t_ns[:30], sort=True)
        want = cpu.run(stack, t_ns[:30], sort=True)
        assert len(got[2]) > 100_000
        assert np.all(np.diff(got[2]) >= 0)
        got_key = np.lexsort((got[3], got[0], got[1], got[2]))
        want_key = np.lexsort((want[3], want[0], want[1], want[2]))
        for g, w in zip(got, want):
            np.testing.assert_array_equal(g[got_key], w[want_key])


def test_legacy_evsim_run_abi_matches_cpu(slider_frames):
    """evsim_run (float32, caller buffers, capacity retry) still works through ctypes."""
    frames, t_ns = slider_frames
    height, width = frames.shape[1:]
    lib = ctypes.CDLL(os.environ["EVLIB_CUDA_SIM_LIB"])
    vp, ll = ctypes.c_void_p, ctypes.c_longlong
    lib.evsim_create.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        vp,
        vp,
        ll,
        ctypes.POINTER(vp),
    ]
    lib.evsim_run.argtypes = [
        vp,
        vp,
        vp,
        ctypes.c_int,
        vp,
        vp,
        vp,
        vp,
        ll,
        ctypes.POINTER(ll),
    ]
    lib.evsim_destroy.argtypes = [vp]
    cpu = evlib.simulation_rs.EventSimulator(width=width, height=height)
    c_pos, c_neg = (np.ascontiguousarray(m, dtype=np.float32) for m in cpu.thresholds())
    handle = vp()
    assert (
        lib.evsim_create(
            width, height, c_pos.ctypes.data, c_neg.ctypes.data, 0, ctypes.byref(handle)
        )
        == 0
    )
    log = np.ascontiguousarray(
        np.log(frames[:12].astype(np.float32) / np.float32(255.0) + np.float32(1e-3))
    )
    t = np.ascontiguousarray(t_ns[:12])
    n = ll(0)
    rc = lib.evsim_run(
        handle,
        log.ctypes.data,
        t.ctypes.data,
        12,
        None,
        None,
        None,
        None,
        0,
        ctypes.byref(n),
    )
    assert rc == 1 and n.value > 0
    cap = n.value
    x, y = np.empty(cap, np.uint16), np.empty(cap, np.uint16)
    tt, p = np.empty(cap, np.int64), np.empty(cap, np.int8)
    rc = lib.evsim_run(
        handle,
        log.ctypes.data,
        t.ctypes.data,
        12,
        x.ctypes.data,
        y.ctypes.data,
        tt.ctypes.data,
        p.ctypes.data,
        cap,
        ctypes.byref(n),
    )
    assert rc == 0 and n.value == cap
    assert lib.evsim_destroy(handle) == 0
    want = cpu.run(log, t, sort=True)
    got = (x.astype(np.int16), y.astype(np.int16), tt, p)
    got_key = np.lexsort((got[3], got[0], got[1], got[2]))
    want_key = np.lexsort((want[3], want[0], want[1], want[2]))
    for g, w in zip(got, want):
        np.testing.assert_array_equal(g[got_key], w[want_key])
