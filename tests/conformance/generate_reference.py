"""Local-only generator for OpenEB conformance digests.

Compiles+runs the OpenEB reference decoder on each tracked/local sample and writes a
tiny digest JSON under tests/conformance/reference/. Requires a local lib/openeb
checkout and a C++ compiler. Never run in CI.

Usage:
  python tests/conformance/generate_reference.py --all [--update]
  python tests/conformance/generate_reference.py --sample 80_balls.evt2 [--update]
  python tests/conformance/generate_reference.py --verify 80_balls.evt2
"""

import argparse
import json
import os
import subprocess

from tests.conformance import canonical, openeb_runner
from tests.conformance.evlib_stream import evlib_events

REFERENCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference")


def _reference_path(key):
    return os.path.join(REFERENCE_DIR, key + ".json")


def _openeb_commit():
    try:
        return subprocess.run(
            ["git", "-C", openeb_runner.OPENEB_ROOT, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _build_digest(key):
    spec = canonical.SAMPLES[key]
    events, geometry = openeb_runner.run_openeb(spec["format"], spec["path"])
    digest = canonical.compute_digest(events, geometry=geometry)
    digest["sample"] = spec["path"]
    digest["format"] = spec["format"]
    digest["openeb_provenance"] = {
        "decoder": openeb_runner._DECODER_DIR[spec["format"]],
        "openeb_commit": _openeb_commit(),
        "generated_by": "tests/conformance/generate_reference.py",
    }
    return digest


def _write(key, digest, update):
    path = _reference_path(key)
    if os.path.exists(path) and not update:
        with open(path) as fh:
            old = json.load(fh)
        if old.get("stream_sha256") != digest["stream_sha256"]:
            raise SystemExit(
                f"{key}: digest changed (n_events {old.get('n_events')} -> {digest['n_events']}, "
                f"sha {old.get('stream_sha256', '')[:12]} -> {digest['stream_sha256'][:12]}). "
                f"Re-run with --update to accept."
            )
        print(f"{key}: unchanged")
        return
    os.makedirs(REFERENCE_DIR, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(digest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"{key}: wrote {path} ({digest['n_events']} events)")


def _verify(key):
    spec = canonical.SAMPLES[key]
    openeb, _ = openeb_runner.run_openeb(spec["format"], spec["path"])
    evl = evlib_events(spec["path"])
    o = canonical.canonical_sort(openeb)
    e = canonical.canonical_sort(evl)
    if len(o) != len(e):
        raise SystemExit(
            f"{key}: event count differs - OpenEB {len(o)} vs evlib {len(e)}"
        )
    for i, (a, b) in enumerate(zip(o, e)):
        if a != b:
            raise SystemExit(
                f"{key}: first divergence at index {i}: OpenEB {a} vs evlib {b}"
            )
    print(f"{key}: byte-identical across {len(o)} events")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true")
    p.add_argument("--sample")
    p.add_argument("--verify")
    p.add_argument("--update", action="store_true")
    args = p.parse_args()

    if args.verify:
        _verify(args.verify)
        return
    keys = list(canonical.SAMPLES) if args.all else [args.sample]
    if keys == [None]:
        p.error("pass --all, --sample KEY, or --verify KEY")
    for key in keys:
        spec = canonical.SAMPLES[key]
        path = os.path.join(openeb_runner.REPO_ROOT, spec["path"])
        if not os.path.exists(path):
            print(f"{key}: sample absent ({spec['path']}), skipping")
            continue
        _write(key, _build_digest(key), args.update)


if __name__ == "__main__":
    main()
