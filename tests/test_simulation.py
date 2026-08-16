"""evlib.simulation on real slider_depth frames. Run: .venv/bin/pytest tests/test_simulation.py.
Acceptance: no torch import, Polars output in the canonical schema, deterministic with a seed.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from evlib.simulation import ESIMConfig, ESIMSimulator, VideoConfig, simulate_frames

ROOT = Path(__file__).resolve().parents[1]
SLIDER = ROOT / "data" / "slider_depth"


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


def test_no_torch_import():
    import evlib.simulation

    package_dir = Path(evlib.simulation.__file__).parent
    for source in package_dir.glob("*.py"):
        text = source.read_text()
        assert "import torch" not in text and "from torch" not in text, source


def test_config_defaults_and_validation():
    c = ESIMConfig()
    assert (
        c.positive_threshold,
        c.negative_threshold,
        c.refractory_period_ms,
        c.log_eps,
    ) == (
        0.2,
        0.2,
        0.0,
        1e-3,
    )
    assert c.refractory_ns == 0
    assert ESIMConfig(refractory_period_ms=0.1).refractory_ns == 100_000
    with pytest.raises(ValueError):
        ESIMConfig(positive_threshold=0)
    with pytest.raises(ValueError):
        ESIMConfig(log_eps=0)
    with pytest.raises(ValueError):
        ESIMConfig(threshold_sigma=-1)
    with pytest.raises(TypeError):
        ESIMConfig(log_floor=1e-3)


def test_simulate_frames_schema_and_order(slider_frames):
    frames, t_ns = slider_frames
    df = simulate_frames(frames, t_ns, ESIMConfig())
    assert df.schema == {
        "x": pl.Int16,
        "y": pl.Int16,
        "t": pl.Duration("us"),
        "polarity": pl.Int8,
    }
    assert df.height > 100_000
    assert df["t"].is_sorted()
    assert set(df["polarity"].unique().to_list()) == {-1, 1}
    assert df["x"].max() < 240 and df["y"].max() < 180


def test_seed_determinism_and_sigma_effect(slider_frames):
    frames, t_ns = slider_frames
    a = simulate_frames(frames[:20], t_ns[:20], ESIMConfig(threshold_sigma=0.1, seed=1))
    b = simulate_frames(frames[:20], t_ns[:20], ESIMConfig(threshold_sigma=0.1, seed=1))
    c = simulate_frames(frames[:20], t_ns[:20], ESIMConfig(threshold_sigma=0.1, seed=2))
    assert a.equals(b)
    assert not a.equals(c)


def test_process_frame_streaming_matches_batch(slider_frames):
    frames, t_ns = slider_frames
    cfg = ESIMConfig()
    sim = ESIMSimulator(cfg, width=240, height=180)
    parts = [sim.process_frame(frames[k], t_ns[k] / 1e9) for k in range(10)]
    assert all(len(a) == 0 for a in parts[0])
    t_stream = np.sort(np.concatenate([p[2] for p in parts]))
    batch = simulate_frames(frames[:10], t_ns[:10], cfg)
    t_batch = batch["t"].dt.total_microseconds().to_numpy() / 1e6
    assert len(t_stream) == len(t_batch)
    # Batch t is floor(t_ns / 1000) us; stream t is t_ns / 1e9 s, so they differ by < 1 us.
    assert np.all(t_stream >= t_batch - 1e-12) and np.all(t_stream - t_batch < 1e-6)


def test_process_frame_rgb_uses_luma_weights():
    sim = ESIMSimulator(ESIMConfig(), width=1, height=1)
    sim.process_frame(np.zeros((1, 1, 3), dtype=np.uint8), 0.0)
    x, y, t, p = sim.process_frame(np.array([[[255, 0, 0]]], dtype=np.uint8), 0.01)
    # 0.299 * 255 = 76 -> ln((76/255 + 1e-3) / 1e-3) = 5.7 -> 28 crossings at c = 0.2.
    assert len(t) == 28 and np.all(p == 1)


def test_process_frame_float32_is_log_intensity(slider_frames):
    frames, t_ns = slider_frames
    log_frames = np.log(frames[:10].astype(np.float32) / 255 + 1e-3).astype(np.float32)
    cfg = ESIMConfig()
    sim = ESIMSimulator(cfg, width=240, height=180)
    parts = [sim.step_ns(log_frames[k], int(t_ns[k])) for k in range(10)]
    # Equal-t ties have no defined order, so compare in a canonical (t, y, x, p) order.
    stream = pl.concat(
        [
            pl.DataFrame({"x": p[0], "y": p[1], "t": p[2] // 1000, "polarity": p[3]})
            for p in parts
        ]
    ).sort(["t", "y", "x", "polarity"])
    batch = (
        simulate_frames(log_frames, t_ns[:10], cfg)
        .with_columns(pl.col("t").dt.total_microseconds())
        .sort(["t", "y", "x", "polarity"])
    )
    assert stream.height == batch.height > 0
    for column in ("t", "x", "y", "polarity"):
        assert np.array_equal(stream[column].to_numpy(), batch[column].to_numpy()), (
            column
        )
    with pytest.raises(TypeError):
        sim.step_ns(log_frames[0].astype(np.float64), 0)


def test_device_dispatch_follows_cuda_available():
    import evlib

    from evlib.simulation.esim import resolve_device

    assert isinstance(evlib.simulation_rs.cuda_available(), bool)
    assert resolve_device("cpu") == "cpu"
    if evlib.simulation_rs.cuda_available():
        assert resolve_device("auto") == "cuda"
        assert (
            ESIMSimulator(ESIMConfig(device="cuda"), width=4, height=4).device == "cuda"
        )
    else:
        assert resolve_device("auto") == "cpu"
        with pytest.raises(RuntimeError, match="CUDA backend unavailable"):
            ESIMSimulator(ESIMConfig(device="cuda"), width=4, height=4)
    assert ESIMSimulator(ESIMConfig(device="cpu"), width=4, height=4).device == "cpu"


def test_video_to_events_on_slider_clip(slider_frames, tmp_path):
    cv2 = pytest.importorskip("cv2")
    from evlib.simulation import VideoToEvents

    frames, _ = slider_frames
    path = tmp_path / "slider.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (240, 180), isColor=False
    )
    for f in frames[:30]:
        writer.write(f)
    writer.release()
    df = VideoToEvents(ESIMConfig(), VideoConfig(width=240, height=180)).process_video(
        path
    )
    assert isinstance(df, pl.DataFrame) and df.height > 10_000
    assert df.schema["t"] == pl.Duration("us")


def test_video_streaming_matches_batch(slider_frames, tmp_path):
    cv2 = pytest.importorskip("cv2")
    from evlib.simulation import VideoToEvents

    frames, _ = slider_frames
    path = tmp_path / "slider.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (240, 180), isColor=False
    )
    for f in frames[:30]:
        writer.write(f)
    writer.release()
    proc = VideoToEvents(ESIMConfig(), VideoConfig(width=240, height=180))
    batch = proc.process_video(path)
    chunks = list(proc.process_frames_streaming(path, chunk_frames=7))
    assert len(chunks) == 5
    # Equal-t ties have no defined order, so compare in a canonical (t, y, x, p) order.
    order = ["t", "y", "x", "polarity"]
    assert pl.concat(chunks).sort(order).equals(batch.sort(order))


def test_simulate_in_batches_matches_whole_stack(slider_frames):
    frames, t_ns = slider_frames
    cfg = ESIMConfig()
    sim = ESIMSimulator(cfg, width=240, height=180)
    parts = [
        sim.simulate(frames[a:b], t_ns[a:b], sort=False)
        for a, b in ((0, 8), (8, 9), (9, 20))
    ]
    keys = ["t", "y", "x", "polarity"]
    joined = pl.concat(parts).sort(keys)
    whole = simulate_frames(frames[:20], t_ns[:20], cfg).sort(keys)
    assert joined.equals(whole)
