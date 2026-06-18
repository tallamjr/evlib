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
