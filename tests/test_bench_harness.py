"""Unit tests for the benchmark measurement harness (no heavy pipeline runs)."""

import sys
from pathlib import Path

# `benchmarks` is a repo-root package, not an installed one. Under pytest's prepend import mode
# the repo root is not always on sys.path (e.g. CI's `pytest tests/`), so add it explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.bench_rvt_pipeline import (  # noqa: E402
    ru_maxrss_bytes,
    summarize,
    time_and_memory,
)


def test_summarize_trivial_list():
    result = summarize([3.0, 1.0, 2.0])
    assert result["min"] == 1.0
    assert result["median"] == 2.0
    assert result["max"] == 3.0


def test_summarize_single_value():
    result = summarize([5.5])
    assert result["min"] == result["median"] == result["max"] == 5.5


def test_summarize_empty_raises():
    try:
        summarize([])
    except ValueError:
        return
    raise AssertionError("summarize([]) should raise ValueError")


def test_time_and_memory_sane_values():
    result = time_and_memory(lambda: sum(range(100_000)), repeats=3)
    assert len(result["times"]) == 3
    assert all(t >= 0.0 for t in result["times"])
    # peak RSS for any live Python process is comfortably above 1 MB
    assert result["peak_rss_bytes"] > 1_000_000


def test_ru_maxrss_unit_normalisation():
    # darwin reports bytes, linux reports KiB
    if sys.platform == "darwin":
        assert ru_maxrss_bytes(2048) == 2048
    else:
        assert ru_maxrss_bytes(2048) == 2048 * 1024
