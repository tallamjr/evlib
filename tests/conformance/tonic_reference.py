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
    """Build a bilinear-interpolated event-volume voxel grid (tonic-identical).

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
