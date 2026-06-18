"""Deterministic event-stream canonicalisation shared by the OpenEB conformance
generator and test. An event tuple is (x, y, polarity, timestamp) with polarity
in {0, 1} and timestamp in integer microseconds."""

import hashlib
import struct

HEAD_TAIL = 16
_RECORD = struct.Struct("<HHBq")  # x:u16, y:u16, pol:u8, t:i64, no padding

# Single source of truth for the conformance samples.
SAMPLES = {
    "80_balls.evt2": {
        "path": "data/prophesee/samples/evt2/80_balls.raw",
        "format": "EVT2",
    },
    "val_night_011.evt2": {
        "path": "tests/data/eTram/raw/val_2/val_night_011.raw",
        "format": "EVT2",
    },
    "pedestrians.evt3": {
        "path": "data/prophesee/samples/evt3/pedestrians.raw",
        "format": "EVT3",
    },
}


def canonical_sort(events):
    """Total order by (t, x, y, pol) so timestamp ties are unambiguous."""
    return sorted(events, key=lambda e: (e[3], e[0], e[1], e[2]))


def pack_stream(events):
    """Pack already-sorted events into the fixed-width little-endian byte stream."""
    return b"".join(_RECORD.pack(x, y, pol, t) for (x, y, pol, t) in events)


def compute_digest(events, geometry):
    """Reduce an event list to the committed digest dict."""
    ordered = canonical_sort(events)
    sha = hashlib.sha256(pack_stream(ordered)).hexdigest()
    head = [list(e) for e in ordered[:HEAD_TAIL]]
    tail = [list(e) for e in ordered[-HEAD_TAIL:]]
    return {
        "geometry": list(geometry),
        "n_events": len(ordered),
        "stream_sha256": sha,
        "head": head,
        "tail": tail,
    }
