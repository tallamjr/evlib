"""E2VID pretrained loading must fail loudly, matching the RVT policy (2026-08-08 P2 finding)."""

import pytest

pytest.importorskip("torch")

from evlib.models.e2vid import E2VID


def test_missing_weights_dir_raises(tmp_path):
    model = E2VID(pretrained=False)
    with pytest.raises(FileNotFoundError):
        model._load_pretrained_weights(weights_dir=tmp_path)  # empty dir, no *.pth*


def test_corrupt_weight_file_raises(tmp_path):
    (tmp_path / "e2vid.pth").write_bytes(b"not a torch checkpoint")
    model = E2VID(pretrained=False)
    with pytest.raises(RuntimeError):
        model._load_pretrained_weights(weights_dir=tmp_path)


def test_real_packaged_weights_still_load():
    # Real data check: the strict policy must not break loading the real
    # packaged weights. The file is gitignored, so skip when absent (CI).
    from pathlib import Path

    from evlib.models import e2vid as e2vid_module

    weights_dir = Path(e2vid_module.__file__).parent / "weights"
    if not sorted(weights_dir.glob("*.pth*")):
        pytest.skip("packaged E2VID weights not present (gitignored)")
    model = E2VID(pretrained=False)
    model._load_pretrained_weights()  # must not raise on the real file
