"""Bindings for evlib.simulation_rs. Run: .venv/bin/pytest tests/test_simulation_rs.py."""

import numpy as np
import pytest

import evlib


def test_simulate_frames_ramp_u8():
    frames = np.stack([np.full((2, 3), v, dtype=np.uint8) for v in (0, 128, 255)])
    t = np.array([0, 1_000_000, 2_000_000], dtype=np.int64)
    x, y, t_ns, p = evlib.simulation_rs.simulate_frames(
        frames,
        t,
        c_pos=0.5,
        c_neg=0.5,
        threshold_sigma=0.0,
        refractory_ns=0,
        log_eps=1e-3,
        seed=0,
        sort=True,
    )
    assert (
        x.dtype == np.int16
        and y.dtype == np.int16
        and t_ns.dtype == np.int64
        and p.dtype == np.int8
    )
    # ln((128/255+1e-3)/(1e-3)) = 6.22 -> 12 crossings at c=0.5 in the first step per pixel.
    assert len(t_ns) == 6 * 12 + 6 * 1
    assert np.all(p == 1)
    assert np.all(np.diff(t_ns) >= 0)


def test_simulate_frames_float32_is_log_input():
    frames = np.zeros((2, 1, 1), dtype=np.float32)
    frames[1] = -0.35
    t = np.array([0, 100], dtype=np.int64)
    _, _, t_ns, p = evlib.simulation_rs.simulate_frames(
        frames,
        t,
        c_pos=0.1,
        c_neg=0.1,
        threshold_sigma=0.0,
        refractory_ns=0,
        log_eps=1e-3,
        seed=0,
        sort=True,
    )
    assert list(p) == [-1, -1, -1]
    assert list(t_ns) == [29, 57, 86]


def test_errors_are_value_errors():
    frames = np.zeros((2, 2, 2), dtype=np.uint8)
    with pytest.raises(ValueError):
        evlib.simulation_rs.simulate_frames(
            frames,
            np.array([5, 5], dtype=np.int64),
            c_pos=0.2,
            c_neg=0.2,
            threshold_sigma=0.0,
            refractory_ns=0,
            log_eps=1e-3,
            seed=0,
            sort=True,
        )
    with pytest.raises(ValueError):
        evlib.simulation_rs.simulate_frames(
            frames,
            np.array([0], dtype=np.int64),
            c_pos=0.2,
            c_neg=0.2,
            threshold_sigma=0.0,
            refractory_ns=0,
            log_eps=1e-3,
            seed=0,
            sort=True,
        )
    with pytest.raises(TypeError):
        evlib.simulation_rs.simulate_frames(
            frames.astype(np.float64),
            np.array([0, 1], dtype=np.int64),
            c_pos=0.2,
            c_neg=0.2,
            threshold_sigma=0.0,
            refractory_ns=0,
            log_eps=1e-3,
            seed=0,
            sort=True,
        )


def test_stateful_simulator_matches_batch():
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, size=(5, 6, 7), dtype=np.uint8)
    t = np.arange(5, dtype=np.int64) * 10_000
    kw = dict(
        c_pos=0.3,
        c_neg=0.25,
        threshold_sigma=0.1,
        refractory_ns=2_000,
        log_eps=1e-3,
        seed=3,
    )
    batch = evlib.simulation_rs.simulate_frames(frames, t, sort=True, **kw)
    sim = evlib.simulation_rs.EventSimulator(width=7, height=6, **kw)
    assert not sim.is_initialised
    parts = [sim.step(frames[k], int(t[k])) for k in range(5)]
    assert sim.is_initialised
    assert all(len(a) == 0 for a in parts[0])
    stepped_t = np.sort(np.concatenate([q[2] for q in parts]))
    assert np.array_equal(stepped_t, batch[2])
    cp, cn = sim.thresholds()
    assert cp.shape == (6, 7) and cn.shape == (6, 7) and not np.allclose(cp, 0.3)
    sim.reset()
    assert not sim.is_initialised
