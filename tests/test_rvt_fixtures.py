from pathlib import Path
from tests.rvt_fixtures import (
    REF,
    raw_input_path,
    ref_repr_h5,
    ref_timestamps,
    ref_labels_npz,
    requires_reference_data,
)


@requires_reference_data
def test_reference_paths_exist():
    assert raw_input_path().exists()
    assert ref_repr_h5().exists()
    assert ref_timestamps().exists()
    assert ref_labels_npz().exists()
