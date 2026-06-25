"""Tests for the N-Caltech101 ATIS `.bin` event reader.

The fixtures are tiny, hand-encoded, genuine ATIS-format byte sequences
written to ``tmp_path``. Each 5-byte event group is ``(b0, b1, b2, b3, b4)``
with::

    x = b0
    y = b1
    polarity = b2 >> 7
    t_us = ((b2 & 0x7F) << 16) | (b3 << 8) | b4

so the expected decodes below are computed by hand.
"""

import numpy as np
import polars as pl
import pytest

from evlib.data.ncaltech import (
    NCALTECH_HEIGHT,
    NCALTECH_WIDTH,
    convert_ncaltech,
    read_atis_bin,
    representation_from_events,
)


def _encode_atis_event(x: int, y: int, polarity: int, t_us: int) -> bytes:
    """Encode one event as the 5 ATIS bytes (inverse of ``read_atis_bin``)."""
    b0 = x & 0xFF
    b1 = y & 0xFF
    b2 = ((polarity & 0x1) << 7) | ((t_us >> 16) & 0x7F)
    b3 = (t_us >> 8) & 0xFF
    b4 = t_us & 0xFF
    return bytes([b0, b1, b2, b3, b4])


def _write_atis_bin(path, events):
    """Write a list of ``(x, y, polarity, t_us)`` tuples to an ATIS ``.bin``."""
    raw = b"".join(_encode_atis_event(*ev) for ev in events)
    path.write_bytes(raw)


def _roundtrip_check():
    # Encoder must be the exact inverse of the decoder so fixtures are genuine.
    raw = _encode_atis_event(0x10, 0x20, 1, 0x7FABCD)
    assert raw == bytes([0x10, 0x20, 0xFF, 0xAB, 0xCD])


def test_sensor_constants():
    assert NCALTECH_WIDTH == 240
    assert NCALTECH_HEIGHT == 180


def test_single_event_all_timestamp_bytes(tmp_path):
    # b2 top bit set -> polarity 1; (b2 & 0x7F) == 0x7F.
    # t_us = (0x7F << 16) | (0xAB << 8) | 0xCD = 0x7FABCD = 8366541.
    b0, b1, b2, b3, b4 = 0x10, 0x20, 0xFF, 0xAB, 0xCD
    path = tmp_path / "single.bin"
    path.write_bytes(bytes([b0, b1, b2, b3, b4]))

    frame = read_atis_bin(path)

    assert frame.height == 1
    row = frame.row(0, named=True)
    assert row["x"] == 0x10
    assert row["y"] == 0x20
    assert row["polarity"] == 1
    assert row["t"] == (0x7F << 16) | (0xAB << 8) | 0xCD
    assert row["t"] == 0x7FABCD == 8367053


def test_top_bit_does_not_leak_into_timestamp(tmp_path):
    # Two events sharing the low 23 timestamp bits but differing top bit.
    # event 0: b2 = 0x80 -> polarity 1, (b2 & 0x7F) == 0 -> t high byte 0.
    # event 1: b2 = 0x00 -> polarity 0, (b2 & 0x7F) == 0 -> t high byte 0.
    # both: t_us = (0 << 16) | (0x12 << 8) | 0x34 = 0x1234 = 4660.
    raw = bytes(
        [
            0x05,
            0x06,
            0x80,
            0x12,
            0x34,
            0x07,
            0x08,
            0x00,
            0x12,
            0x34,
        ]
    )
    path = tmp_path / "polarity.bin"
    path.write_bytes(raw)

    frame = read_atis_bin(path)

    assert frame.height == 2
    first = frame.row(0, named=True)
    second = frame.row(1, named=True)
    assert first["polarity"] == 1
    assert second["polarity"] == 0
    assert first["t"] == 0x1234 == 4660
    assert second["t"] == 0x1234 == 4660


def test_multiple_events_file_order_and_dtypes(tmp_path):
    raw = bytes(
        [
            0x00,
            0x00,
            0x00,
            0x00,
            0x01,  # t=1, pol=0
            0xEF,
            0xB3,
            0x80,
            0x00,
            0x02,  # x=239, y=179, t=2, pol=1
            0x01,
            0x02,
            0x03,
            0x04,
            0x05,  # t=(3<<16)|(4<<8)|5
        ]
    )
    path = tmp_path / "multi.bin"
    path.write_bytes(raw)

    frame = read_atis_bin(path)

    assert frame.columns == ["x", "y", "t", "polarity"]
    assert frame.schema["x"] == pl.Int64
    assert frame.schema["y"] == pl.Int64
    assert frame.schema["t"] == pl.Int64
    assert frame.schema["polarity"] == pl.Int64

    assert frame["x"].to_list() == [0x00, 0xEF, 0x01]
    assert frame["y"].to_list() == [0x00, 0xB3, 0x02]
    assert frame["polarity"].to_list() == [0, 1, 0]
    assert frame["t"].to_list() == [
        1,
        2,
        (0x03 << 16) | (0x04 << 8) | 0x05,
    ]


def test_malformed_length_raises(tmp_path):
    path = tmp_path / "bad.bin"
    path.write_bytes(bytes([0x01, 0x02, 0x03, 0x04, 0x05, 0x06]))  # 6 bytes

    with pytest.raises(ValueError, match=str(path)):
        read_atis_bin(path)


# --- D2: representation building + N-Caltech tree converter ---

NBINS = 10
CHANNELS = 2 * NBINS


def test_encoder_is_inverse_of_decoder(tmp_path):
    _roundtrip_check()
    path = tmp_path / "rt.bin"
    _write_atis_bin(path, [(5, 6, 1, 4660), (7, 8, 0, 4660)])
    frame = read_atis_bin(path)
    assert frame["x"].to_list() == [5, 7]
    assert frame["y"].to_list() == [6, 8]
    assert frame["polarity"].to_list() == [1, 0]
    assert frame["t"].to_list() == [4660, 4660]


def test_representation_shape_and_dtype():
    events = pl.DataFrame(
        {
            "x": [10, 11, 12],
            "y": [20, 21, 22],
            "t": [100, 200, 300],
            "polarity": [0, 1, 0],
        },
        schema={k: pl.Int64 for k in ("x", "y", "t", "polarity")},
    )
    rep = representation_from_events(events, nbins=NBINS)
    assert rep.shape == (CHANNELS, NCALTECH_HEIGHT, NCALTECH_WIDTH)
    assert rep.dtype == np.uint8


def test_representation_known_event_channel():
    # Channel formula (read from build_sparse_histogram): polarity-major,
    #   channel = polarity * nbins + temporal_bin
    # An event sitting at t == t_min has t_norm == 0 -> temporal_bin 0.
    # Marker event: polarity 1 at t_min -> channel = 1*nbins + 0 = nbins.
    # A second (different) event provides the window span so binning is defined.
    marker_x, marker_y = 30, 40
    events = pl.DataFrame(
        {
            "x": [marker_x, 200],
            "y": [marker_y, 100],
            "t": [1000, 9000],
            "polarity": [1, 0],
        },
        schema={k: pl.Int64 for k in ("x", "y", "t", "polarity")},
    )
    rep = representation_from_events(events, nbins=NBINS)

    expected_channel = 1 * NBINS + 0
    assert rep[expected_channel, marker_y, marker_x] == 1
    # The same pixel must be empty on the polarity-0 half (no leakage).
    assert rep[0, marker_y, marker_x] == 0


def test_representation_degenerate_single_event():
    events = pl.DataFrame(
        {"x": [5], "y": [6], "t": [42], "polarity": [1]},
        schema={k: pl.Int64 for k in ("x", "y", "t", "polarity")},
    )
    rep = representation_from_events(events, nbins=NBINS)
    assert rep.shape == (CHANNELS, NCALTECH_HEIGHT, NCALTECH_WIDTH)
    assert rep.dtype == np.uint8
    # The single event must survive (not be divided away or dropped).
    assert int(rep.sum()) == 1
    assert rep[NBINS, 6, 5] == 1


def _build_two_class_tree(root):
    # Sorted class order is airplane < camera -> {airplane: 0, camera: 1}.
    airplane = root / "airplane"
    camera = root / "camera"
    airplane.mkdir(parents=True)
    camera.mkdir(parents=True)
    _write_atis_bin(
        airplane / "image_0001.bin",
        [(10, 20, 1, 1000), (30, 40, 0, 5000)],
    )
    _write_atis_bin(
        airplane / "image_0002.bin",
        [(11, 21, 0, 2000), (31, 41, 1, 6000)],
    )
    _write_atis_bin(
        camera / "image_0001.bin",
        [(12, 22, 1, 1500), (32, 42, 0, 5500)],
    )
    return ["airplane", "airplane", "camera"]


def test_convert_ncaltech_layout_and_labels(tmp_path):
    root = tmp_path / "ncaltech"
    expected_class_order = _build_two_class_tree(root)
    out_dir = tmp_path / "out"

    paths, labels = convert_ncaltech(root, out_dir, nbins=NBINS)

    assert len(paths) == 3
    assert labels == [0, 0, 1]  # airplane, airplane, camera (sorted classes)
    assert [p.parent for p in paths] == [out_dir] * 3

    saved_labels = np.load(out_dir / "labels.npy")
    assert saved_labels.dtype == np.int64
    assert saved_labels.tolist() == [0, 0, 1]

    for path in paths:
        assert path.exists()
        arr = np.load(path)
        assert arr.shape == (CHANNELS, NCALTECH_HEIGHT, NCALTECH_WIDTH)
        assert arr.dtype == np.uint8
        assert int(arr.sum()) > 0

    # The label map follows sorted class-directory names.
    assert expected_class_order == ["airplane", "airplane", "camera"]


def test_convert_ncaltech_empty_root_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no class"):
        convert_ncaltech(empty, tmp_path / "out")


def test_convert_ncaltech_class_without_bins_raises(tmp_path):
    root = tmp_path / "ncaltech"
    (root / "airplane").mkdir(parents=True)
    _write_atis_bin(root / "airplane" / "a.bin", [(1, 1, 1, 10)])
    (root / "camera").mkdir()  # no .bin files
    with pytest.raises(ValueError, match="camera"):
        convert_ncaltech(root, tmp_path / "out")
