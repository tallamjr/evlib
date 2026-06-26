"""Verify the committed gen4 RVT-tiny checkpoint loads COMPLETELY into evlib's RVT.

The checkpoint (``python/evlib/models/weights/rvt-t.ckpt``) is loaded via
key-remapping with ``strict=False`` in ``RVT._load_pretrained_weights`` /
``RVT._convert_checkpoint_key``. A ``strict=False`` load silently tolerates
missing/unmapped weights, so a partial load would produce a garbage mAP
downstream. These tests prove the load is complete and correct before any
evaluation is built on top of it:

- every checkpoint key maps to a real model parameter (no unmapped
  load-bearing weights, no unexpected converted keys),
- every loaded tensor's shape matches the model parameter shape,
- the detection head is 3-class (gen4),
- representative backbone and head weights are actually populated from the
  checkpoint rather than left at random initialisation.
"""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from evlib.models.rvt import RVT

CHECKPOINT_PATH = (
    Path(__file__).resolve().parent.parent
    / "python"
    / "evlib"
    / "models"
    / "weights"
    / "rvt-t.ckpt"
)

# Keys in the raw Lightning checkpoint that are intentionally not model
# parameters (optimiser state, training bookkeeping, etc.). The loader returns
# ``None`` for these via ``_convert_checkpoint_key``. Enumerated explicitly so a
# newly introduced unmapped *load-bearing* key cannot hide behind a blanket
# "ignore everything that did not map" rule. The committed rvt-t.ckpt currently
# contains no such keys, but the allow-list documents the policy.
RAW_KEY_ALLOWLIST: frozenset = frozenset()


def _load_checkpoint_state_dict() -> dict:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def _convert_state_dict(model: RVT, raw_state_dict: dict) -> tuple[dict, list]:
    """Return (converted_state_dict, unmapped_raw_keys)."""
    converted: dict = {}
    unmapped: list = []
    for raw_key, value in raw_state_dict.items():
        new_key = model._convert_checkpoint_key(raw_key)
        if new_key is None:
            unmapped.append(raw_key)
        else:
            converted[new_key] = value
    return converted, unmapped


@pytest.fixture(scope="module")
def raw_state_dict() -> dict:
    assert CHECKPOINT_PATH.exists(), f"Checkpoint missing: {CHECKPOINT_PATH}"
    return _load_checkpoint_state_dict()


@pytest.fixture(scope="module")
def model() -> RVT:
    return RVT(variant="tiny", num_classes=3, pretrained=False)


def test_no_unmapped_loadbearing_keys(model, raw_state_dict):
    """Every checkpoint key either maps to a model param or is allow-listed."""
    _, unmapped = _convert_state_dict(model, raw_state_dict)
    offending = sorted(set(unmapped) - RAW_KEY_ALLOWLIST)
    assert not offending, (
        f"{len(offending)} checkpoint keys did not map to any model "
        f"parameter and are not allow-listed:\n  " + "\n  ".join(offending)
    )


def test_no_missing_keys(model, raw_state_dict):
    """Every model parameter is provided by the converted checkpoint."""
    converted, _ = _convert_state_dict(model, raw_state_dict)
    model_keys = set(model.state_dict().keys())
    missing = sorted(model_keys - set(converted.keys()))
    assert not missing, (
        f"{len(missing)} model parameters are not provided by the "
        f"checkpoint:\n  " + "\n  ".join(missing)
    )


def test_no_unexpected_keys(model, raw_state_dict):
    """Every converted checkpoint key corresponds to a real model parameter."""
    converted, _ = _convert_state_dict(model, raw_state_dict)
    model_keys = set(model.state_dict().keys())
    unexpected = sorted(set(converted.keys()) - model_keys)
    assert not unexpected, (
        f"{len(unexpected)} converted keys do not exist in the model:\n  "
        + "\n  ".join(unexpected)
    )


def test_all_shapes_match(model, raw_state_dict):
    """No silent shape-mismatch skips: converted tensors match model shapes."""
    converted, _ = _convert_state_dict(model, raw_state_dict)
    model_state = model.state_dict()
    mismatches = []
    for key, tensor in converted.items():
        if key not in model_state:
            continue
        if tuple(tensor.shape) != tuple(model_state[key].shape):
            mismatches.append(
                f"{key}: checkpoint {tuple(tensor.shape)} != "
                f"model {tuple(model_state[key].shape)}"
            )
    assert not mismatches, "Shape mismatches:\n  " + "\n  ".join(mismatches)


def test_full_coverage_counts(model, raw_state_dict):
    """The whole checkpoint loads: converted count == model param count."""
    converted, unmapped = _convert_state_dict(model, raw_state_dict)
    total_raw = len(raw_state_dict)
    total_model = len(model.state_dict())
    allow_listed = len(set(unmapped) & RAW_KEY_ALLOWLIST)
    # Every raw key is either converted or explicitly allow-listed.
    assert len(converted) + allow_listed == total_raw, (
        f"converted {len(converted)} + allow-listed {allow_listed} "
        f"!= total raw {total_raw}"
    )
    # Every model parameter is covered by a converted key.
    assert len(converted) == total_model, (
        f"converted {len(converted)} != model params {total_model}"
    )


def test_head_is_three_class(model, raw_state_dict):
    """The detection head is gen4 3-class (cls_preds output channel == 3)."""
    converted, _ = _convert_state_dict(model, raw_state_dict)
    cls_pred_weights = {
        k: v
        for k, v in converted.items()
        if k.startswith("head.cls_preds.") and k.endswith(".weight")
    }
    assert cls_pred_weights, "No head.cls_preds.*.weight keys found"
    for key, tensor in cls_pred_weights.items():
        assert tensor.shape[0] == 3, (
            f"{key} output channels = {tensor.shape[0]}, expected 3"
        )
    # The constructed model head must also expose 3 classes.
    model_state = model.state_dict()
    assert model_state["head.cls_preds.0.weight"].shape[0] == 3


def test_weights_actually_populated_from_checkpoint():
    """A pretrained load copies checkpoint tensors into the model.

    Loads with ``pretrained=True`` and asserts that a representative backbone
    conv weight and a head cls_pred weight equal the checkpoint tensors. This
    proves the remap actually applied rather than the model retaining its
    random initialisation.
    """
    raw_state_dict = _load_checkpoint_state_dict()
    reference = RVT(variant="tiny", num_classes=3, pretrained=False)
    converted, _ = _convert_state_dict(reference, raw_state_dict)

    loaded = RVT(variant="tiny", num_classes=3, pretrained=True)
    loaded_state = loaded.state_dict()

    backbone_key = next(
        k
        for k in converted
        if k.startswith("backbone.")
        and k.endswith(".weight")
        and converted[k].dim() >= 2
    )
    head_key = "head.cls_preds.0.weight"

    for key in (backbone_key, head_key):
        # Compare on CPU: on a CUDA box the model params may live on cuda
        # while the checkpoint tensors were loaded with map_location="cpu",
        # so an in-place device-agnostic comparison is required.
        assert torch.allclose(loaded_state[key].cpu(), converted[key].cpu()), (
            f"{key} was not populated from the checkpoint"
        )

    # Sanity: the freshly-loaded weights differ from the random init of a
    # separate un-pretrained model, confirming the load changed something.
    assert not torch.allclose(
        loaded_state[backbone_key].cpu(),
        reference.state_dict()[backbone_key].cpu(),
    ), "pretrained weights equal random init; load had no effect"


def test_pretrained_load_raises_on_torch_load_failure(monkeypatch):
    """A pretrained-load failure must RAISE, not fall back to random weights.

    Repo policy forbids silently continuing with randomly initialised weights
    when the caller explicitly asked for ``pretrained=True``: a downstream eval
    would then report a meaningless mAP. Simulate a corrupt/unreadable
    checkpoint by making ``torch.load`` raise, and assert the constructor
    propagates a clear ``RuntimeError`` rather than returning a random model.
    """

    def _boom(*args, **kwargs):
        raise OSError("simulated corrupt checkpoint")

    monkeypatch.setattr("evlib.models.rvt.torch.load", _boom)

    with pytest.raises(RuntimeError, match="failed to load pretrained weights"):
        RVT(variant="tiny", num_classes=3, pretrained=True)


def test_pretrained_load_raises_when_no_weights_found(monkeypatch):
    """Requesting ``pretrained=True`` with no checkpoint present must RAISE.

    The "no weights file found" branch previously printed a warning and left
    the model randomly initialised. When pretrained weights were explicitly
    requested that is the same dangerous silent fallback and must raise a
    ``FileNotFoundError`` naming the directory searched.
    """
    # Make every candidate checkpoint path appear absent.
    real_exists = Path.exists

    def _fake_exists(self):
        if self.suffix == ".ckpt":
            return False
        return real_exists(self)

    monkeypatch.setattr(Path, "exists", _fake_exists)

    with pytest.raises(FileNotFoundError, match="No pretrained weights"):
        RVT(variant="tiny", num_classes=3, pretrained=True)
