"""Local-only: run the installed dv_processing (DV-framework) AEDAT 4.0 reader as
the reference oracle for evlib's AEDAT4 decoder, and parse its CSV output.

dv_processing MUST run in a SEPARATE subprocess: importing dv_processing and
evlib/polars in the same Python process triggers an OpenMP libomp double-init
crash (OMP Error #15). The subprocess below imports ONLY dv_processing; this
parent module never imports dv_processing (nor evlib/polars). Mirrors the
subprocess+tempfile structure of openeb_runner.py."""

import os
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Inline script run by the subprocess. Imports ONLY dv_processing, decodes the
# AEDAT4 recording, and writes "x,y,pol,t" CSV (polarity as 0/1, timestamp as
# integer microseconds) to the path given as argv[2].
_DV_SCRIPT = r"""
import sys
import dv_processing as dv

src = sys.argv[1]
out = sys.argv[2]
reader = dv.io.MonoCameraRecording(src)
with open(out, "w") as fh:
    while reader.isRunning():
        batch = reader.getNextEventBatch()
        if batch is None:
            continue
        for e in batch:
            fh.write("%d,%d,%d,%d\n" % (e.x(), e.y(), 1 if e.polarity() else 0, e.timestamp()))
"""


def parse_dv_csv(text):
    """Parse dv_processing CSV text into ([(x, y, pol, t), ...], geometry).

    Skips blank and ``%``-prefixed lines. geometry is the observed extent
    ``(max_x + 1, max_y + 1)``, or ``(0, 0)`` when there are no events."""
    events = []
    for line in text.splitlines():
        if not line or line.startswith("%"):
            continue
        x, y, pol, t = line.split(",")
        events.append((int(x), int(y), int(pol), int(t)))
    if not events:
        return events, (0, 0)
    max_x = max(e[0] for e in events)
    max_y = max(e[1] for e in events)
    return events, (max_x + 1, max_y + 1)


def run_dv_aedat4(raw_path):
    """Decode raw_path with the dv_processing reference reader; return (events, geometry).

    raw_path is resolved against REPO_ROOT when relative. Raises FileNotFoundError
    if the sample is missing, and RuntimeError if dv_processing is not importable
    in the subprocess."""
    raw_path = (
        os.path.join(REPO_ROOT, raw_path) if not os.path.isabs(raw_path) else raw_path
    )
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Sample missing: {raw_path}")
    env = {**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "out.csv")
        result = subprocess.run(
            [sys.executable, "-c", _DV_SCRIPT, raw_path, csv_path],
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            stderr = result.stderr or ""
            if "dv_processing" in stderr and (
                "ModuleNotFoundError" in stderr or "ImportError" in stderr
            ):
                raise RuntimeError("dv_processing not available")
            raise RuntimeError(
                f"dv_processing reader failed (exit {result.returncode}):\n{stderr}"
            )
        with open(csv_path) as fh:
            return parse_dv_csv(fh.read())
