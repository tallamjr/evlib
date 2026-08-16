"""evlib kernel vs rpg_vid2e esim_torch reference digests for slider_depth.

The JSON under tests/conformance/reference/ is generated on a GPU host by
tests/conformance/vid2e_reference.py and is tracked, so this test always runs.
Run: .venv/bin/pytest tests/test_vid2e_conformance.py -q

Tolerances and why (measured 2026-08-16, esim_torch ecbb11a, 87 frames):
- total and positive counts within 0.25 % (measured 0.121 % and 0.120 %);
- per-frame counts within 1.5 % (measured max 0.75 %, mean 0.16 %);
- 10x10 block counts within 3 % (measured max 1.64 %, mean 0.13 %);
- first 200 events sorted by (t, y, x, p): same x, y, p; t within 1 ns.
per_pixel_counts_sha256 is recorded in the JSON but not asserted: exact-tie
flips change single pixels, so block_counts is the enforced spatial digest.
Both kernels are float32 ESIM. Nearly all divergence (98.4 % of the first
divergent pixel cells) is an exact tie: a uint8 pixel returns to a grey level
seen before, so L1 equals l_ref + k*c exactly, and float32 rounding resolves
the tie differently in each kernel. esim_torch also computes event times in
float32 (93 % of matched events differ by at most one float32 ulp of t), so
timestamps are only compared on the early sample where the ulp is below 1 ns.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import evlib
from evlib.simulation import ESIMConfig

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "tests" / "conformance" / "reference" / "vid2e_slider_depth_ct0.2.json"
SLIDER = ROOT / "data" / "slider_depth"

COUNT_TOL = 0.0025
FRAME_TOL = 0.015
BLOCK_TOL = 0.03


@pytest.fixture(scope="module")
def reference():
    return json.loads(REF.read_text())


@pytest.fixture(scope="module")
def evlib_events(reference):
    """Raw kernel output (x, y, t_ns, p) as int64 plus the frame timestamps in ns."""
    from PIL import Image

    frames, t_ns = [], []
    for line in (SLIDER / "images.txt").read_text().splitlines():
        secs, rel = line.split()
        frames.append(np.asarray(Image.open(SLIDER / rel).convert("L"), dtype=np.uint8))
        t_ns.append(round(float(secs) * 1e9))
    frames = np.stack(frames)
    t_ns = np.asarray(t_ns, dtype=np.int64)
    assert frames.shape == (
        reference["frames"],
        reference["height"],
        reference["width"],
    )
    cfg = ESIMConfig(
        positive_threshold=reference["c_pos"],
        negative_threshold=reference["c_neg"],
        refractory_period_ms=reference["refractory_ns"] / 1e6,
        log_eps=1e-3,
    )
    # The raw path keeps ns timestamps; the DataFrame path truncates to us.
    x, y, t, p = evlib.simulation_rs.simulate_frames(
        frames, t_ns, sort=True, **cfg.kernel_kwargs()
    )
    return tuple(a.astype(np.int64) for a in (x, y, t, p)), t_ns


def test_reference_metadata(reference):
    assert reference["source"] == "data/slider_depth/images"
    assert reference["log"] == "ln(I/255 + 1e-3) as float32"
    assert (reference["c_pos"], reference["c_neg"], reference["refractory_ns"]) == (
        0.2,
        0.2,
        0,
    )
    assert len(reference["per_frame_counts"]) == reference["frames"] - 1
    assert len(reference["sample"]) == 200
    assert reference["esim_torch_provenance"]["raw_polarity_values"] == [-1, 1]


def test_total_and_polarity_counts(reference, evlib_events):
    (_, _, _, p), _ = evlib_events
    total = len(p)
    pos = int((p == 1).sum())
    assert (
        abs(total - reference["total_events"]) <= COUNT_TOL * reference["total_events"]
    ), (
        total,
        reference["total_events"],
    )
    assert abs(pos - reference["pos_events"]) <= COUNT_TOL * reference["pos_events"], (
        pos,
        reference["pos_events"],
    )


def test_per_frame_counts(reference, evlib_events):
    (_, _, t, _), t_ns = evlib_events
    # Bin i holds events with t_ns[i] <= t < t_ns[i+1]; np.histogram closes the last bin.
    counts = np.histogram(t, bins=t_ns)[0]
    ref = np.asarray(reference["per_frame_counts"])
    assert counts.shape == ref.shape
    rel = np.abs(counts - ref) / ref
    assert rel.max() <= FRAME_TOL, (
        rel.argmax(),
        counts[rel.argmax()],
        ref[rel.argmax()],
    )


def test_block_counts(reference, evlib_events):
    (x, y, _, _), _ = evlib_events
    block = reference["block_px"]
    counts = np.zeros((reference["height"], reference["width"]), dtype=np.int64)
    np.add.at(counts, (y, x), 1)
    got = counts.reshape(
        counts.shape[0] // block, block, counts.shape[1] // block, block
    ).sum(axis=(1, 3))
    ref = np.asarray(reference["block_counts"])
    assert got.shape == ref.shape
    rel = np.abs(got - ref) / ref
    assert rel.max() <= BLOCK_TOL, rel.max()


def test_first_events_sample(reference, evlib_events):
    (x, y, t, p), _ = evlib_events
    order = np.lexsort((p, x, y, t))[:200]
    got = np.stack([x[order], y[order], t[order], p[order]], axis=1)
    ref = np.asarray(reference["sample"], dtype=np.int64)
    assert got.shape == ref.shape
    assert np.array_equal(got[:, [0, 1, 3]], ref[:, [0, 1, 3]])
    assert np.abs(got[:, 2] - ref[:, 2]).max() <= 1
