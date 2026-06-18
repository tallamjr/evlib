"""Local-only: compile and run the OpenEB standalone EVT2/EVT3 decoders from the
developer's gitignored lib/openeb checkout, and parse their CSV output. Never
imported or run in CI (the conformance test only touches committed digests)."""

import hashlib
import os
import shutil
import subprocess
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OPENEB_ROOT = os.path.join(REPO_ROOT, "lib", "openeb")

_DECODER_DIR = {
    "EVT2": "metavision_evt2_raw_file_decoder",
    "EVT3": "metavision_evt3_raw_file_decoder",
}


def decoder_source(fmt):
    name = _DECODER_DIR[fmt]
    return os.path.join(OPENEB_ROOT, "standalone_samples", name, name + ".cpp")


def _compiler():
    for candidate in (os.environ.get("CXX"), "c++", "g++", "clang++"):
        if candidate and shutil.which(candidate):
            return candidate
    raise RuntimeError("No C++ compiler found (set CXX or install g++/clang++)")


def _compile_decoder(fmt):
    """Compile the decoder once, cached in the system temp dir by source hash."""
    src = decoder_source(fmt)
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"OpenEB decoder source missing: {src} (is lib/openeb checked out?)"
        )
    with open(src, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()[:16]
    out = os.path.join(tempfile.gettempdir(), f"evlib_openeb_{fmt.lower()}_{digest}")
    if not os.path.exists(out):
        subprocess.run(
            [_compiler(), "-std=c++17", "-O2", src, "-o", out],
            check=True,
        )
    return out


def parse_openeb_csv(text):
    """Parse OpenEB CSV text into ((x, y, pol, t), ...) and observed geometry."""
    events = []
    for line in text.splitlines():
        if not line or line.startswith("%"):
            continue
        x, y, pol, t = line.split(",")
        events.append((int(x), int(y), int(pol), int(t)))
    max_x = max((e[0] for e in events), default=-1)
    max_y = max((e[1] for e in events), default=-1)
    return events, (max_x + 1, max_y + 1)


def run_openeb(fmt, raw_path):
    """Decode raw_path with the OpenEB reference decoder; return (events, geometry)."""
    binary = _compile_decoder(fmt)
    raw_path = (
        os.path.join(REPO_ROOT, raw_path) if not os.path.isabs(raw_path) else raw_path
    )
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Sample missing: {raw_path}")
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "out.csv")
        subprocess.run([binary, raw_path, csv_path], check=True)
        with open(csv_path) as fh:
            return parse_openeb_csv(fh.read())
