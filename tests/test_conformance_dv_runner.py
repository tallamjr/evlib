import os

import pytest

from tests.conformance import dv_aedat4_runner

_SAMPLE = os.path.join(
    dv_aedat4_runner.REPO_ROOT,
    "lib/dv-processing/tests/io/test_files/sample_data.aedat4",
)


def test_parse_dv_csv_maps_columns_and_geometry():
    csv = "10,20,0,5\n11,21,1,7\n"
    events, geometry = dv_aedat4_runner.parse_dv_csv(csv)
    assert events == [(10, 20, 0, 5), (11, 21, 1, 7)]
    assert geometry == (12, 22)


@pytest.mark.skipif(
    not os.path.exists(_SAMPLE), reason="dv-processing sample absent (local-only)"
)
def test_run_dv_aedat4_decodes_sample():
    # The sample ships with the local dv-processing checkout, but the dv_processing
    # Python module may not be installed. The runner reports that as a RuntimeError
    # from its subprocess; treat only that case as a local-only skip and let any
    # other decode failure surface.
    try:
        events, geometry = dv_aedat4_runner.run_dv_aedat4(
            "lib/dv-processing/tests/io/test_files/sample_data.aedat4"
        )
    except RuntimeError as exc:
        if "dv_processing not available" in str(exc):
            pytest.skip("dv_processing module not installed (local-only)")
        raise
    assert len(events) == 9193
    assert events[0] == (56, 16, 1, 1663249605734020)
