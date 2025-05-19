import numpy as np

import evlib

# Direct access to the functions
events_to_block_py = evlib.evlib.core.events_to_block
merge_events = evlib.evlib.core.merge_events


def test_events_to_block_py():
    """Test the events_to_block_py function in core module"""
    # Create sample event data
    xs = np.array([1, 2, 3, 4], dtype=np.int64)
    ys = np.array([5, 6, 7, 8], dtype=np.int64)
    ts = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    ps = np.array([1, -1, 1, -1], dtype=np.int64)

    # Convert to block
    block = events_to_block_py(xs, ys, ts, ps)

    # Check shape and values
    assert block.shape == (4, 4)
    assert np.array_equal(block[:, 0], xs.astype(np.float64))
    assert np.array_equal(block[:, 1], ys.astype(np.float64))
    assert np.array_equal(block[:, 2], ts)
    assert np.array_equal(block[:, 3], ps.astype(np.float64))


def test_events_to_block_empty():
    """Test events_to_block_py with empty arrays"""
    # Create empty arrays
    xs = np.array([], dtype=np.int64)
    ys = np.array([], dtype=np.int64)
    ts = np.array([], dtype=np.float64)
    ps = np.array([], dtype=np.int64)

    # Convert to block
    block = events_to_block_py(xs, ys, ts, ps)

    # Check shape and type
    assert block.shape == (0, 4)
    assert block.dtype == np.float64


def test_merge_events_multiple():
    """Test merging multiple event tuples"""
    # Create multiple event tuples
    events1 = (
        np.array([1, 2], dtype=np.int64),
        np.array([3, 4], dtype=np.int64),
        np.array([0.1, 0.2], dtype=np.float64),
        np.array([1, -1], dtype=np.int64),
    )

    events2 = (
        np.array([5, 6], dtype=np.int64),
        np.array([7, 8], dtype=np.int64),
        np.array([0.3, 0.4], dtype=np.float64),
        np.array([-1, 1], dtype=np.int64),
    )

    events3 = (
        np.array([9, 10], dtype=np.int64),
        np.array([11, 12], dtype=np.int64),
        np.array([0.5, 0.6], dtype=np.float64),
        np.array([1, 1], dtype=np.int64),
    )

    # Merge events - the correct format is a tuple of tuples
    merged_xs, merged_ys, merged_ts, merged_ps = merge_events((events1, events2, events3))

    # Check lengths
    assert len(merged_xs) == 6
    assert len(merged_ys) == 6
    assert len(merged_ts) == 6
    assert len(merged_ps) == 6

    # Check content (order might be different)
    expected_xs = np.array([1, 2, 5, 6, 9, 10], dtype=np.int64)
    expected_ys = np.array([3, 4, 7, 8, 11, 12], dtype=np.int64)
    expected_ts = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64)
    expected_ps = np.array([1, -1, -1, 1, 1, 1], dtype=np.int64)

    # Check that all expected values are in the merged arrays
    for x in expected_xs:
        assert x in merged_xs
    for y in expected_ys:
        assert y in merged_ys
    for t in expected_ts:
        assert t in merged_ts
    for p in expected_ps:
        assert p in merged_ps


def test_merge_events_single():
    """Test merging a single event tuple"""
    # Create a single event tuple
    events = (
        np.array([1, 2, 3], dtype=np.int64),
        np.array([4, 5, 6], dtype=np.int64),
        np.array([0.1, 0.2, 0.3], dtype=np.float64),
        np.array([1, -1, 1], dtype=np.int64),
    )

    # Merge single event - use a tuple, not a list
    merged_xs, merged_ys, merged_ts, merged_ps = merge_events((events,))

    # Check that the result is identical to the input
    assert np.array_equal(merged_xs, events[0])
    assert np.array_equal(merged_ys, events[1])
    assert np.array_equal(merged_ts, events[2])
    assert np.array_equal(merged_ps, events[3])


def test_merge_events_empty():
    """Test merging empty event tuples"""
    # Create empty event tuples
    events1 = (
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64),
        np.array([], dtype=np.float64),
        np.array([], dtype=np.int64),
    )

    events2 = (
        np.array([], dtype=np.int64),
        np.array([], dtype=np.int64),
        np.array([], dtype=np.float64),
        np.array([], dtype=np.int64),
    )

    # Merge empty events - use a tuple, not a list
    merged_xs, merged_ys, merged_ts, merged_ps = merge_events((events1, events2))

    # Check that the result is empty
    assert len(merged_xs) == 0
    assert len(merged_ys) == 0
    assert len(merged_ts) == 0
    assert len(merged_ps) == 0

    # Check empty with non-empty
    events3 = (
        np.array([1, 2], dtype=np.int64),
        np.array([3, 4], dtype=np.int64),
        np.array([0.1, 0.2], dtype=np.float64),
        np.array([1, -1], dtype=np.int64),
    )

    merged_xs, merged_ys, merged_ts, merged_ps = merge_events((events1, events3))

    # Check that the result matches the non-empty input
    assert np.array_equal(merged_xs, events3[0])
    assert np.array_equal(merged_ys, events3[1])
    assert np.array_equal(merged_ts, events3[2])
    assert np.array_equal(merged_ps, events3[3])
