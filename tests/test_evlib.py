import numpy as np

import evlib

# Direct access to the functions through _evlib module
events_to_block = evlib.core.events_to_block
merge_events = evlib.core.merge_events
add_random_events = evlib.augmentation.add_random_events
remove_events = evlib.augmentation.remove_events
add_correlated_events = evlib.augmentation.add_correlated_events
flip_events_x = evlib.augmentation.flip_events_x
flip_events_y = evlib.augmentation.flip_events_y
clip_events_to_bounds = evlib.augmentation.clip_events_to_bounds
rotate_events = evlib.augmentation.rotate_events


def test_events_to_block():
    # Create sample event components
    xs = np.array([1, 2, 3, 4], dtype=np.int64)
    ys = np.array([5, 6, 7, 8], dtype=np.int64)
    ts = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    ps = np.array([1, -1, 1, -1], dtype=np.int64)

    # Convert to block
    block = events_to_block(xs, ys, ts, ps)

    # Check shape and values
    assert block.shape == (4, 4)
    assert np.array_equal(block[:, 0], xs.astype(np.float64))
    assert np.array_equal(block[:, 1], ys.astype(np.float64))
    assert np.array_equal(block[:, 2], ts)
    assert np.array_equal(block[:, 3], ps.astype(np.float64))


def test_merge_events():
    # Create two sets of events
    xs1 = np.array([1, 2], dtype=np.int64)
    ys1 = np.array([3, 4], dtype=np.int64)
    ts1 = np.array([0.1, 0.2], dtype=np.float64)
    ps1 = np.array([1, -1], dtype=np.int64)

    xs2 = np.array([5, 6], dtype=np.int64)
    ys2 = np.array([7, 8], dtype=np.int64)
    ts2 = np.array([0.3, 0.4], dtype=np.float64)
    ps2 = np.array([1, -1], dtype=np.int64)

    # Merge events
    merged_xs, merged_ys, merged_ts, merged_ps = merge_events(((xs1, ys1, ts1, ps1), (xs2, ys2, ts2, ps2)))

    # Check merged results
    assert len(merged_xs) == 4
    assert len(merged_ys) == 4
    assert len(merged_ts) == 4
    assert len(merged_ps) == 4

    # Check specific values are present (order may vary)
    for x in [1, 2, 5, 6]:
        assert x in merged_xs
    for y in [3, 4, 7, 8]:
        assert y in merged_ys
    for t in [0.1, 0.2, 0.3, 0.4]:
        assert t in merged_ts
    for p in [1, -1, 1, -1]:
        assert p in merged_ps


def test_add_random_events():
    # Create sample event components
    xs = np.array([10, 20, 30, 40], dtype=np.int64)
    ys = np.array([50, 60, 70, 80], dtype=np.int64)
    ts = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    ps = np.array([1, -1, 1, -1], dtype=np.int64)

    # Add random events
    to_add = 6
    new_xs, new_ys, new_ts, new_ps = add_random_events(xs, ys, ts, ps, to_add)

    # Check if the result has correct number of events
    assert len(new_xs) == len(xs) + to_add
    assert len(new_ys) == len(ys) + to_add
    assert len(new_ts) == len(ts) + to_add
    assert len(new_ps) == len(ps) + to_add

    # Check ranges
    assert np.all(new_xs >= 0) and np.all(new_xs <= np.max(xs))
    assert np.all(new_ys >= 0) and np.all(new_ys <= np.max(ys))
    assert np.all(new_ts >= np.min(ts)) and np.all(new_ts <= np.max(ts))
    assert np.all(np.isin(new_ps, [1, -1]))

    # The implementation always merges events, even with sort=False
    new_xs, new_ys, new_ts, new_ps = add_random_events(xs, ys, ts, ps, to_add, sort=False)

    # Check if result has merged events (original + added)
    assert len(new_xs) == len(xs) + to_add
    assert len(new_ys) == len(ys) + to_add
    assert len(new_ts) == len(ts) + to_add
    assert len(new_ps) == len(ps) + to_add


def test_remove_events():
    # Create sample event components
    xs = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=np.int64)
    ys = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=np.int64)
    ts = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float64)
    ps = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1], dtype=np.int64)

    # Remove events
    to_remove = 4
    new_xs, new_ys, new_ts, new_ps = remove_events(xs, ys, ts, ps, to_remove)

    # Check if the result has correct number of events
    assert len(new_xs) == len(xs) - to_remove
    assert len(new_ys) == len(ys) - to_remove
    assert len(new_ts) == len(ts) - to_remove
    assert len(new_ps) == len(ps) - to_remove

    # Test removing all events
    new_xs, new_ys, new_ts, new_ps = remove_events(xs, ys, ts, ps, len(xs) + 5)
    assert len(new_xs) == 0
    assert len(new_ys) == 0
    assert len(new_ts) == 0
    assert len(new_ps) == 0

    # Test with adding noise
    new_xs, new_ys, new_ts, new_ps = remove_events(xs, ys, ts, ps, 4, add_noise=2)
    assert len(new_xs) == len(xs) - to_remove + 2
    assert len(new_ys) == len(ys) - to_remove + 2
    assert len(new_ts) == len(ts) - to_remove + 2
    assert len(new_ps) == len(ps) - to_remove + 2


def test_add_correlated_events():
    # Create sample event components
    xs = np.array([50, 60, 70, 80], dtype=np.int64)
    ys = np.array([50, 60, 70, 80], dtype=np.int64)
    ts = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    ps = np.array([1, -1, 1, -1], dtype=np.int64)

    # Add correlated events
    to_add = 6
    new_xs, new_ys, new_ts, new_ps = add_correlated_events(xs, ys, ts, ps, to_add)

    # Check if the result has correct number of events
    assert len(new_xs) == len(xs) + to_add
    assert len(new_ys) == len(ys) + to_add
    assert len(new_ts) == len(ts) + to_add
    assert len(new_ps) == len(ps) + to_add

    # Check with different parameters
    new_xs, new_ys, new_ts, new_ps = add_correlated_events(
        xs,
        ys,
        ts,
        ps,
        to_add,
        sort=False,
        return_merged=False,
        xy_std=2.0,
        ts_std=0.005,
    )

    # Check if the result has only the new events
    assert len(new_xs) == to_add
    assert len(new_ys) == to_add
    assert len(new_ts) == to_add
    assert len(new_ps) == to_add

    # Check with adding noise
    new_xs, new_ys, new_ts, new_ps = add_correlated_events(xs, ys, ts, ps, to_add, add_noise=3)

    # Check if the result has correct number of events (to_add + noise)
    assert len(new_xs) == len(xs) + to_add + 3
    assert len(new_ys) == len(ys) + to_add + 3
    assert len(new_ts) == len(ts) + to_add + 3
    assert len(new_ps) == len(ps) + to_add + 3


def test_flip_events():
    # Create sample event components
    xs = np.array([10, 20, 30, 40], dtype=np.int64)
    ys = np.array([50, 60, 70, 80], dtype=np.int64)
    ts = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    ps = np.array([1, -1, 1, -1], dtype=np.int64)

    sensor_resolution = (100, 50)  # (height, width)

    # Flip events along x axis
    new_xs, new_ys, new_ts, new_ps = flip_events_x(xs, ys, ts, ps, sensor_resolution)

    # Check if the result is flipped correctly
    assert np.array_equal(new_xs, sensor_resolution[1] - 1 - xs)
    assert np.array_equal(new_ys, ys)
    assert np.array_equal(new_ts, ts)
    assert np.array_equal(new_ps, ps)

    # Flip events along y axis
    new_xs, new_ys, new_ts, new_ps = flip_events_y(xs, ys, ts, ps, sensor_resolution)

    # Check if the result is flipped correctly
    assert np.array_equal(new_xs, xs)
    assert np.array_equal(new_ys, sensor_resolution[0] - 1 - ys)
    assert np.array_equal(new_ts, ts)
    assert np.array_equal(new_ps, ps)


def test_clip_events_to_bounds():
    # Create sample event components
    xs = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    ys = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    ts = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    ps = np.array([1, -1, 1, -1, 1], dtype=np.int64)

    # Clip events with bounds [min_y, max_y, min_x, max_x]
    bounds = [15, 45, 15, 45]
    new_xs, new_ys, new_ts, new_ps = clip_events_to_bounds(xs, ys, ts, ps, bounds)

    # Check if the result is clipped correctly
    # Expected filter: Only (20, 20) and (30, 30) should be within bounds
    # Print the actual values to debug
    print("Clipped xs:", new_xs)
    print("Clipped ys:", new_ys)
    assert len(new_xs) == 2
    assert np.all(new_xs >= bounds[2]) and np.all(new_xs < bounds[3])
    assert np.all(new_ys >= bounds[0]) and np.all(new_ys < bounds[1])

    # Test with set_zero=True
    new_xs, new_ys, new_ts, new_ps = clip_events_to_bounds(xs, ys, ts, ps, bounds, set_zero=True)

    # Check if out-of-bounds events are set to zero
    assert len(new_xs) == len(xs)
    mask = (xs >= bounds[2]) & (xs < bounds[3]) & (ys >= bounds[0]) & (ys < bounds[1])
    assert np.array_equal(new_xs, xs * mask)
    assert np.array_equal(new_ys, ys * mask)


def test_rotate_events():
    # Create sample event components
    xs = np.array([10, 20, 30, 40], dtype=np.int64)
    ys = np.array([50, 60, 70, 80], dtype=np.int64)
    ts = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    ps = np.array([1, -1, 1, -1], dtype=np.int64)

    # Rotate events with specific angle
    theta = np.pi / 4  # 45 degrees
    center = (20, 60)
    new_xs, new_ys, angle, center_returned = rotate_events(
        xs,
        ys,
        ts,
        ps,
        sensor_resolution=(100, 50),
        theta_radians=theta,
        center_of_rotation=center,
    )

    # Check if the result has correct length
    assert len(new_xs) == len(xs)
    assert len(new_ys) == len(ys)

    # Check angle and center returned
    assert angle == theta
    assert center_returned == center

    # Test with clip_to_range=True
    new_xs, new_ys, angle, center_returned = rotate_events(
        xs,
        ys,
        ts,
        ps,
        sensor_resolution=(100, 50),
        theta_radians=theta,
        center_of_rotation=center,
        clip_to_range=True,
    )

    # Check if the result is within bounds
    assert np.all(new_xs >= 0) and np.all(new_xs < 50)
    assert np.all(new_ys >= 0) and np.all(new_ys < 100)


if __name__ == "__main__":
    test_events_to_block()
    test_merge_events()
    test_add_random_events()
    test_remove_events()
    test_add_correlated_events()
    test_flip_events()
    test_clip_events_to_bounds()
    test_rotate_events()
    print("All tests passed!")
