from tests.conformance import canonical


def test_sort_is_by_t_then_x_y_pol():
    # events are (x, y, pol, t); canonical order is (t, x, y, pol)
    events = [(5, 1, 1, 100), (2, 9, 0, 100), (2, 9, 0, 50)]
    assert canonical.canonical_sort(events) == [
        (2, 9, 0, 50),
        (2, 9, 0, 100),
        (5, 1, 1, 100),
    ]


def test_pack_record_is_13_bytes_little_endian():
    # <HHBq = u16 x, u16 y, u8 pol, i64 t, no padding
    assert canonical.pack_stream([(1, 2, 1, 3)]) == bytes(
        [1, 0, 2, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0]
    )


def test_digest_is_deterministic_and_order_independent():
    a = [(5, 1, 1, 100), (2, 9, 0, 50)]
    b = [(2, 9, 0, 50), (5, 1, 1, 100)]
    da = canonical.compute_digest(a, geometry=(1280, 720))
    db = canonical.compute_digest(b, geometry=(1280, 720))
    assert da["stream_sha256"] == db["stream_sha256"]
    assert da["n_events"] == 2
    assert da["head"] == [[2, 9, 0, 50], [5, 1, 1, 100]]
    assert da["tail"] == [[2, 9, 0, 50], [5, 1, 1, 100]]


def test_head_tail_capped_at_16():
    events = [(i, 0, 0, i) for i in range(40)]
    d = canonical.compute_digest(events, geometry=(100, 100))
    assert len(d["head"]) == 16
    assert len(d["tail"]) == 16
    assert d["head"][0] == [0, 0, 0, 0]
    assert d["tail"][-1] == [39, 0, 0, 39]
