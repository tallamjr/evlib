#!/usr/bin/env python3
"""Generate captured-output blocks for documentation code samples.

Scans ``docs/**/*.md`` for fenced ```python blocks immediately followed
(blank lines allowed) by an ``<!-- evlib:output -->`` marker. Each such
block is executed (repo root as cwd, fresh namespace, stdout captured) and
the result is written back as a fenced ```text block between
``<!-- evlib:output:start -->``/``<!-- evlib:output:end -->`` markers.

Only blocks whose source references a tracked fixture
(``data/slider_depth/events.txt`` or
``data/prophesee/samples/evt2/80_balls.raw``) are executed, so the script
produces identical results on any machine and in CI.

Usage:
    python scripts/generate_doc_outputs.py [--check] [path...]

With no paths, scans all of docs/. ``--check`` diffs against what is on
disk without writing, exiting 1 if anything would change.
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import io
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_ROOT = REPO_ROOT / "docs"

TRACKED_FIXTURES = (
    "data/slider_depth/events.txt",
    "data/prophesee/samples/evt2/80_balls.raw",
)

MARKER = "<!-- evlib:output -->"
BLOCK_START = "<!-- evlib:output:start -->"
BLOCK_END = "<!-- evlib:output:end -->"

PYTHON_FENCE = re.compile(r"^```python\s*$")
CLOSE_FENCE = re.compile(r"^```\s*$")


class MalformedMarkerError(Exception):
    """Raised when an evlib:output:start marker has no matching end marker."""


def _find_blocks(lines: list[str]) -> list[dict]:
    """Locate python-fence/marker pairs. Each entry gives the source line
    range and, if present, the existing output block's line range."""
    blocks = []
    i = 0
    n = len(lines)
    while i < n:
        if not PYTHON_FENCE.match(lines[i]):
            i += 1
            continue
        source_start = i + 1
        j = source_start
        while j < n and not CLOSE_FENCE.match(lines[j]):
            j += 1
        if j >= n:
            break
        source_end = j
        k = source_end + 1
        while k < n and lines[k].strip() == "":
            k += 1
        if k >= n or lines[k].strip() != MARKER:
            i = source_end + 1
            continue
        marker_line = k
        m = marker_line + 1
        while m < n and lines[m].strip() == "":
            m += 1
        existing_range = None
        if m < n and lines[m].strip() == BLOCK_START:
            p = m + 1
            while p < n and lines[p].strip() != BLOCK_END:
                p += 1
            if p >= n:
                raise MalformedMarkerError(
                    f"{BLOCK_START} at line {m + 1} has no matching {BLOCK_END}"
                )
            existing_range = (m, p)
        blocks.append(
            {
                "source_start": source_start,
                "source_end": source_end,
                "marker_line": marker_line,
                "existing_range": existing_range,
            }
        )
        i = source_end + 1
    return blocks


@contextlib.contextmanager
def _repo_root_cwd():
    previous = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        yield
    finally:
        os.chdir(previous)


def _run_block(source: str, path: Path, source_line: int) -> str:
    buffer = io.StringIO()
    namespace = {"__name__": "__main__", "__file__": str(path)}
    with _repo_root_cwd():
        try:
            with contextlib.redirect_stdout(buffer):
                exec(compile(source, str(path), "exec"), namespace)
        except Exception:
            print(
                f"evlib:output block failed: {path} (source starts at line {source_line})",
                file=sys.stderr,
            )
            raise
    return buffer.getvalue()


def process_file(path: Path) -> tuple[str, list[str]]:
    """Return (regenerated text, notes about skipped blocks)."""
    original_text = path.read_text()
    trailing_newline = original_text.endswith("\n")
    lines = original_text.split("\n")
    if trailing_newline:
        lines = lines[:-1]

    blocks = _find_blocks(lines)
    notes = []
    new_lines = list(lines)
    offset = 0
    for block in blocks:
        source = "\n".join(lines[block["source_start"] : block["source_end"]])
        if not any(fixture in source for fixture in TRACKED_FIXTURES):
            notes.append(
                f"{path}: skipping block at line {block['source_start'] + 1} "
                "(no tracked-fixture reference)"
            )
            continue
        # Pad with leading blank lines so compiled line numbers match the
        # block's real position in the file; linecache keys tracebacks off
        # (filename, lineno) against the real file, not the block text.
        padded_source = ("\n" * block["source_start"]) + source
        output = _run_block(padded_source, path, block["source_start"] + 1)
        content = output.split("\n")
        if content and content[-1] == "":
            content.pop()
        replacement = [BLOCK_START, "```text"] + content + ["```", BLOCK_END]

        if block["existing_range"] is not None:
            start, end = block["existing_range"]
            insert_at = start + offset
            remove_count = end - start + 1
        else:
            insert_at = block["marker_line"] + 1 + offset
            remove_count = 0
        new_lines[insert_at : insert_at + remove_count] = replacement
        offset += len(replacement) - remove_count

    new_text = "\n".join(new_lines) + ("\n" if trailing_newline else "")
    return new_text, notes


def _resolve_targets(paths: list[Path]) -> list[Path]:
    if not paths:
        return sorted(DOCS_ROOT.rglob("*.md"))
    targets: set[Path] = set()
    for raw in paths:
        resolved = raw if raw.is_absolute() else REPO_ROOT / raw
        if resolved.is_dir():
            targets.update(resolved.rglob("*.md"))
        else:
            targets.add(resolved)
    return sorted(targets)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="diff only, do not write; exit 1 if any file would change",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="markdown files or directories to scan (default: docs/)",
    )
    args = parser.parse_args(argv)

    drifted = []
    for path in _resolve_targets(args.paths):
        new_text, notes = process_file(path)
        for note in notes:
            print(note, file=sys.stderr)

        original_text = path.read_text()
        if new_text == original_text:
            continue

        if args.check:
            drifted.append(path)
            diff = difflib.unified_diff(
                original_text.splitlines(keepends=True),
                new_text.splitlines(keepends=True),
                fromfile=str(path),
                tofile=f"{path} (generated)",
            )
            sys.stdout.writelines(diff)
        else:
            path.write_text(new_text)
            try:
                display = path.relative_to(REPO_ROOT)
            except ValueError:
                display = path
            print(f"updated {display}")

    if args.check and drifted:
        print(
            f"\n{len(drifted)} file(s) would change; run without --check to update.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
