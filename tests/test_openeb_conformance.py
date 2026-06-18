import json
import os

import pytest

from tests.conformance import canonical
from tests.conformance.evlib_stream import evlib_digest

_BALLS = canonical.SAMPLES["80_balls.evt2"]["path"]


@pytest.mark.skipif(not os.path.exists(_BALLS), reason="EVT2 sample absent")
def test_evlib_digest_is_deterministic():
    a = evlib_digest(_BALLS)
    b = evlib_digest(_BALLS)
    assert a["n_events"] > 0
    assert a["stream_sha256"] == b["stream_sha256"]
    # polarity must be canonicalised to {0, 1}
    assert all(ev[2] in (0, 1) for ev in a["head"])


_REFERENCE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "conformance", "reference"
)


def _reference(key):
    with open(os.path.join(_REFERENCE_DIR, key + ".json")) as fh:
        return json.load(fh)


@pytest.mark.parametrize("key", list(canonical.SAMPLES))
def test_evlib_matches_openeb_reference(key):
    spec = canonical.SAMPLES[key]
    sample_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), spec["path"]
    )
    if not os.path.exists(sample_path):
        pytest.skip(f"sample absent: {spec['path']} (EVT3 is gitignored, runs locally)")

    ref = _reference(key)
    got = evlib_digest(sample_path)

    assert got["n_events"] == ref["n_events"], (
        f"{key}: event count {got['n_events']} != reference {ref['n_events']}"
    )
    assert got["geometry"] == ref["geometry"], f"{key}: geometry mismatch"
    if got["stream_sha256"] != ref["stream_sha256"]:
        first = next(
            (i for i, (a, b) in enumerate(zip(got["head"], ref["head"])) if a != b),
            None,
        )
        raise AssertionError(
            f"{key}: stream digest mismatch. "
            f"head diverges at index {first}: evlib {got['head'][first] if first is not None else 'n/a'} "
            f"vs reference {ref['head'][first] if first is not None else 'n/a'}. "
            f"Run `python tests/conformance/generate_reference.py --verify {key}` locally "
            f"to find the exact first divergent event."
        )
    assert got["head"] == ref["head"], f"{key}: head mismatch"
    assert got["tail"] == ref["tail"], f"{key}: tail mismatch"
