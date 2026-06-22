"""Self-contained numpy port of tonic's event-volume voxel grid.

This module PORTS the reference implementation verbatim from tonic so it can be
used as a conformance oracle without importing the ``tonic`` package (whose
``__init__`` pulls in scipy/librosa). The real tonic source lives at:

    lib/tonic/tonic/functional/to_voxel_grid.py
    (``to_voxel_grid_numpy``)

The algorithm is the event volume of:

    Zhu, A. Z., Yuan, L., Chaney, K., & Daniilidis, K. (2019).
    "Unsupervised Event-Based Learning of Optical Flow, Depth, and Egomotion."
    CVPR 2019.

tonic in turn adapted it from rpg_e2vid:
    https://github.com/uzh-rpg/rpg_e2vid/blob/master/utils/inference_utils.py#L431

The port is verified bit-identical against the real tonic source by
``tests/test_tonic_reference.py`` (which loads the real file by path).
"""

import numpy as np


def tonic_voxel_grid(events_struct, sensor_size_whp, n_time_bins):
    """Build a bilinear-interpolated event-volume voxel grid (tonic-compatible).

    Parameters
    ----------
    events_struct : np.ndarray
        Structured array with fields ``x``, ``y``, ``t``, ``p``. Events are
        assumed to be sorted by ``t`` (tonic uses ``t[0]`` / ``t[-1]`` for
        normalisation). ``p`` is mapped ``0 -> -1`` so polarity is +1/-1.
    sensor_size_whp : tuple
        ``(W, H, P)`` sensor size; ``P`` must equal 2.
    n_time_bins : int
        Number of temporal bins.

    Returns
    -------
    np.ndarray
        Dense voxel grid of shape ``(n_time_bins, 1, H, W)``, float64.
    """
    events = events_struct
    sensor_size = sensor_size_whp

    assert "x" and "y" and "t" and "p" in events.dtype.names
    assert sensor_size[2] == 2

    voxel_grid = np.zeros((n_time_bins, sensor_size[1], sensor_size[0]), float).ravel()

    # normalize the event timestamps so that they lie between 0 and n_time_bins
    ts = (
        n_time_bins
        * (events["t"].astype(float) - events["t"][0])
        / (events["t"][-1] - events["t"][0])
    )
    xs = events["x"].astype(int)
    ys = events["y"].astype(int)
    pols = events["p"].copy()
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(int)
    dts = ts - tis
    vals_left = pols * (1.0 - dts)
    vals_right = pols * dts

    valid_indices = tis < n_time_bins
    np.add.at(
        voxel_grid,
        xs[valid_indices]
        + ys[valid_indices] * sensor_size[0]
        + tis[valid_indices] * sensor_size[0] * sensor_size[1],
        vals_left[valid_indices],
    )

    valid_indices = (tis + 1) < n_time_bins
    np.add.at(
        voxel_grid,
        xs[valid_indices]
        + ys[valid_indices] * sensor_size[0]
        + (tis[valid_indices] + 1) * sensor_size[0] * sensor_size[1],
        vals_right[valid_indices],
    )

    voxel_grid = np.reshape(
        voxel_grid, (n_time_bins, 1, sensor_size[1], sensor_size[0])
    )

    return voxel_grid


def tonic_frame(events_struct, sensor_size_whp, n_time_bins):
    """Build a dense event frame by equal-width time bins (tonic-compatible).

    PORTS tonic's ``to_frame_numpy`` ``n_time_bins`` path together with the
    ``SliceByTimeBins`` slicer it delegates to. The real tonic source lives at:

        lib/tonic/tonic/functional/to_frame.py   (``to_frame_numpy``)
        lib/tonic/tonic/slicers.py               (``SliceByTimeBins``,
                                                   ``slice_events_by_time_bins``)

    The whole recording is sliced into ``n_time_bins`` equal-width TIME bins and
    events are counted per ``(polarity, y, x)`` per bin.

    Boundary handling reproduced verbatim from ``SliceByTimeBins`` (overlap=0):

        time_window  = (t[-1] - t[0]) // n_time_bins      # integer floor div
        stride       = time_window
        starts       = arange(n_time_bins) * stride + t[0]
        ends         = starts + time_window
        idx_start    = searchsorted(t, starts)            # side='left'
        idx_end      = searchsorted(t, ends)              # side='left'
        slice_i      = events[idx_start[i] : idx_end[i]]

    Consequences that MUST be matched by any Polars port:
      * ``time_window`` is integer-floored, so the bins span only
        ``[t0, t0 + n_time_bins*time_window]`` which is ``<= t_max``. Events at
        or beyond the last bin end (``t >= t0 + n_time_bins*time_window``) are
        DROPPED -- in particular the final max-time event is usually dropped.
      * ``searchsorted`` with ``side='left'`` makes each bin left-closed,
        right-open: an event exactly on a bin boundary belongs to the later bin.

    Parameters
    ----------
    events_struct : np.ndarray
        Structured array with fields ``x``, ``y``, ``t``, ``p``. Events are
        assumed sorted by ``t`` (tonic slices via ``searchsorted`` and uses
        ``t[0]`` / ``t[-1]``). ``p`` is mapped so negative -> channel 0,
        positive -> channel 1.
    sensor_size_whp : tuple
        ``(W, H, P)`` sensor size; ``P`` must equal 2.
    n_time_bins : int
        Number of equal-width temporal bins.

    Returns
    -------
    np.ndarray
        Dense frame of shape ``(n_time_bins, P, H, W)``, int64.
    """
    events = events_struct
    sensor_size = sensor_size_whp

    assert "x" and "y" and "t" and "p" in events.dtype.names
    assert sensor_size[2] == 2
    width, height, n_pols = (
        int(sensor_size[0]),
        int(sensor_size[1]),
        int(sensor_size[2]),
    )

    times = events["t"]

    # --- SliceByTimeBins.get_slice_metadata (overlap=0) ---
    # Integer floor division matches tonic exactly (// on integer timestamps).
    time_window = (times[-1] - times[0]) // n_time_bins
    stride = time_window  # overlap == 0
    window_start_times = np.arange(n_time_bins) * stride + times[0]
    window_end_times = window_start_times + time_window
    indices_start = np.searchsorted(times, window_start_times)
    indices_end = np.searchsorted(times, window_end_times)

    # to_frame_numpy maps polarity to channel index via astype(int). evlib data
    # is 0/1 or -1/1; tonic's array data is already 0/1. Match evlib by sending
    # any non-positive polarity to channel 0 and positive to channel 1.
    pol_idx = (events["p"] > 0).astype(int)

    frames = np.zeros((n_time_bins, n_pols, height, width), dtype=np.int64)
    for i in range(n_time_bins):
        start, end = int(indices_start[i]), int(indices_end[i])
        if end <= start:
            continue
        sl = slice(start, end)
        np.add.at(
            frames,
            (i, pol_idx[sl], events["y"][sl].astype(int), events["x"][sl].astype(int)),
            1,
        )
    return frames


def _slice_events_by_time_indices(times, dt):
    """Port of tonic ``SliceByTime.get_slice_metadata`` (overlap=0, no incomplete).

    Reproduces ``lib/tonic/tonic/slicers.py`` ``SliceByTime.get_slice_metadata``
    with ``time_window=dt``, ``overlap=0``, ``include_incomplete=False``,
    ``start_time=None``, ``end_time=None``. Returns the list of
    ``(idx_start, idx_end)`` searchsorted-left index pairs.

        stride       = dt
        start_time   = times[0]
        end_time     = times[-1]
        duration     = end_time - start_time
        n_slices     = floor((duration - dt) / dt) + 1   # include_incomplete=False
        n_slices     = max(n_slices, 1)
        starts       = arange(n_slices) * dt + times[0]
        ends         = starts + dt
        idx_start    = searchsorted(times, starts, side='left')
        idx_end      = searchsorted(times, ends,   side='left')

    Each slice i covers events with ``t`` in ``[i*dt + t0, (i+1)*dt + t0)``.
    """
    stride = dt
    start_time = times[0]
    end_time = times[-1]
    duration = end_time - start_time

    n_slices = int(np.floor((duration - dt) / stride) + 1)
    n_slices = max(n_slices, 1)

    window_start_times = np.arange(n_slices) * stride + start_time
    window_end_times = window_start_times + dt
    indices_start = np.searchsorted(times, window_start_times)[:n_slices]
    indices_end = np.searchsorted(times, window_end_times)[:n_slices]
    return list(zip(indices_start, indices_end))


def tonic_timesurface(events_struct, sensor_size_whp, dt, tau):
    """Build HOTS time surfaces (tonic-compatible). See Lagorce et al. 2016.

    PORTS tonic's ``to_timesurface_numpy`` together with the ``SliceByTime``
    slicer it delegates to. The real tonic source lives at:

        lib/tonic/tonic/functional/to_timesurface.py (``to_timesurface_numpy``)
        lib/tonic/tonic/slicers.py                    (``SliceByTime``,
                                                        ``slice_events_by_time``)

    Algorithm (overlap=0, include_incomplete=False):
      * Slice events by fixed time window ``dt`` (left-closed, right-open).
      * Maintain ``memory`` of shape ``(P, H, W)`` initialised to ``-inf``;
        ``start_t`` is the timestamp of the FIRST event.
      * For slice ``i`` (0-based): write ``memory[p, y, x] = t`` for each event
        in the slice (later events overwrite), then
        ``surf = exp(-((i+1)*dt + start_t - memory) / tau)`` and append it.
      * Pixels never seen keep ``memory = -inf`` so ``surf = exp(-inf) = 0``.

    Returns ``(n_slices, P, H, W)``, float64.

    Parameters
    ----------
    events_struct : np.ndarray
        Structured array with fields ``x``, ``y``, ``t``, ``p``. Events are
        assumed sorted by ``t`` (tonic slices via ``searchsorted`` and uses
        ``t[0]``). ``p`` is the channel index: tonic indexes ``memory`` by the
        raw ``p`` value, which for evlib data must be mapped negative -> channel
        0, positive -> channel 1 by the CALLER (this oracle indexes ``p`` as-is,
        matching tonic, so pass 0/1 channel indices).
    sensor_size_whp : tuple
        ``(W, H, P)`` sensor size.
    dt : float
        Time window for slicing / decay reference.
    tau : float
        Exponential decay time constant.
    """
    events = events_struct
    width, height, n_pols = (
        int(sensor_size_whp[0]),
        int(sensor_size_whp[1]),
        int(sensor_size_whp[2]),
    )

    times = events["t"].astype(np.float64)
    xs = events["x"].astype(int)
    ys = events["y"].astype(int)
    ps = events["p"].astype(int)

    metadata = _slice_events_by_time_indices(times, dt)

    # tonic init: np.ones(sensor_size[::-1]) * -inf -> shape (P, H, W).
    memory = np.ones((n_pols, height, width), dtype=float) * -np.inf
    start_t = times[0]
    all_surfaces = []
    for i, (start, end) in enumerate(metadata):
        start, end = int(start), int(end)
        if end > start:
            sl = slice(start, end)
            # later events in the slice overwrite earlier ones at the same pixel,
            # exactly like tonic's ``memory[tuple(indices)] = timestamps``.
            memory[ps[sl], ys[sl], xs[sl]] = times[sl]
        diff = -((i + 1) * dt + start_t - memory)
        surf = np.exp(diff / tau)
        all_surfaces.append(surf)
    return np.array(all_surfaces)
