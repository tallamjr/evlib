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

import polars as pl
import pytest

from evlib.data.ncaltech import (
    NCALTECH_HEIGHT,
    NCALTECH_WIDTH,
    read_atis_bin,
)


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
