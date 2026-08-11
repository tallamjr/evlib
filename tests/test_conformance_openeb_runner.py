import os

import pytest

from tests.conformance import openeb_runner

_OPENEB = openeb_runner.OPENEB_ROOT


def test_parse_openeb_csv_maps_columns():
    csv = "%geometry:1280,720\n10,20,0,5\n11,21,1,7\n"
    events, geometry = openeb_runner.parse_openeb_csv(csv)
    assert events == [(10, 20, 0, 5), (11, 21, 1, 7)]
    # geometry reduced to observed extent (max+1), not the header value
    assert geometry == (12, 22)


@pytest.mark.skipif(not os.path.isdir(_OPENEB), reason="lib/openeb absent (local-only)")
def test_decoder_source_paths_exist():
    assert os.path.exists(openeb_runner.decoder_source("EVT2"))
    assert os.path.exists(openeb_runner.decoder_source("EVT3"))
