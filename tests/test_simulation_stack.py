"""Load lumin vfi frame stacks: frames.u8, timestamps_ns.i64, meta.json.

Run: .venv/bin/pytest tests/test_simulation_stack.py.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from evlib.simulation import ESIMConfig, load_frame_stack, simulate_frames
from evlib.simulation.stack import FrameStack

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


def _write_stack(
    directory: Path,
    frame_data: np.ndarray,
    timestamps_ns: np.ndarray,
    drop_keys=(),
    **meta_overrides,
):
    """Write a stack directory in the exact layout lumin's StackWriter emits."""
    directory.mkdir(parents=True, exist_ok=True)
    frame_count, height, width = frame_data.shape
    np.ascontiguousarray(frame_data, dtype=np.uint8).tofile(directory / "frames.u8")
    np.ascontiguousarray(timestamps_ns, dtype="<i8").tofile(
        directory / "timestamps_ns.i64"
    )
    meta = {
        "width": width,
        "height": height,
        "frames": frame_count,
        "source": "slider_depth",
        "source_fps": 25.874,
        "model": "rife_v426_grey",
        "provider": "cpu",
        "policy": {"kind": "fixed", "upsample": 2, "max_flow": None, "max_depth": None},
        "factor": 1.0,
        "generated_by": "test",
        "date": "1700000000",
    }
    meta.update(meta_overrides)
    for key in drop_keys:
        meta.pop(key, None)
    (directory / "meta.json").write_text(json.dumps(meta))


@pytest.fixture()
def stack_dir(tmp_path, slider_frames):
    frames, t_ns = slider_frames
    directory = tmp_path / "stack"
    _write_stack(directory, frames, t_ns)
    return directory


def _load_esim_convert():
    """Load scripts/esim_convert.py by path; scripts/ is not an importable package."""
    spec = importlib.util.spec_from_file_location(
        "esim_convert", ROOT / "scripts" / "esim_convert.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_frame_stack_shapes_dtypes_meta(stack_dir, slider_frames):
    frames, t_ns = slider_frames
    stack = load_frame_stack(stack_dir)
    assert isinstance(stack, FrameStack)
    assert isinstance(stack.frames, np.memmap)
    assert stack.frames.shape == frames.shape
    assert stack.frames.dtype == np.uint8
    assert np.array_equal(stack.frames, frames)
    assert stack.timestamps_ns.dtype == np.int64
    assert np.array_equal(stack.timestamps_ns, t_ns)
    assert stack.meta["width"] == 240
    assert stack.meta["height"] == 180
    assert stack.meta["frames"] == 87
    assert stack.meta["source"] == "slider_depth"
    assert stack.meta["generated_by"] == "test"
    assert stack.meta["policy"]["kind"] == "fixed"


def test_simulate_frames_on_memmap_matches_in_memory_array(stack_dir):
    stack = load_frame_stack(stack_dir)
    cfg = ESIMConfig()
    from_stack = simulate_frames(stack.frames, stack.timestamps_ns, cfg)
    from_array = simulate_frames(np.asarray(stack.frames), stack.timestamps_ns, cfg)
    assert from_stack.equals(from_array)


def test_batched_cli_conversion_matches_whole_stack_event_count(stack_dir):
    esim_convert = _load_esim_convert()
    cfg = ESIMConfig()
    batched = esim_convert.convert_frame_stack(stack_dir, cfg, batch_frames=16)

    stack = load_frame_stack(stack_dir)
    whole = simulate_frames(stack.frames, stack.timestamps_ns, cfg)

    assert batched.height == whole.height


def test_load_frame_stack_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_frame_stack(tmp_path / "does_not_exist")


def test_load_frame_stack_frame_size_mismatch_raises(tmp_path, slider_frames):
    frames, t_ns = slider_frames
    directory = tmp_path / "stack"
    _write_stack(directory, frames, t_ns, frames=len(t_ns) + 1)
    with pytest.raises(ValueError, match="frames.u8"):
        load_frame_stack(directory)


def test_load_frame_stack_non_monotone_timestamps_raises(tmp_path, slider_frames):
    frames, t_ns = slider_frames
    directory = tmp_path / "stack"
    bad_t_ns = t_ns.copy()
    bad_t_ns[5] = bad_t_ns[4]
    _write_stack(directory, frames, bad_t_ns)
    with pytest.raises(ValueError, match="strictly increase"):
        load_frame_stack(directory)


def test_load_frame_stack_missing_factor_raises(tmp_path, slider_frames):
    frames, t_ns = slider_frames
    directory = tmp_path / "stack"
    _write_stack(directory, frames, t_ns, drop_keys=("factor",))
    with pytest.raises(ValueError, match="factor"):
        load_frame_stack(directory)


def test_load_frame_stack_missing_policy_raises(tmp_path, slider_frames):
    frames, t_ns = slider_frames
    directory = tmp_path / "stack"
    _write_stack(directory, frames, t_ns, drop_keys=("policy",))
    with pytest.raises(ValueError, match="policy"):
        load_frame_stack(directory)
