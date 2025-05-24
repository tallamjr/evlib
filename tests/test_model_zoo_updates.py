"""Test model zoo updates including real URLs and new architectures."""

import pytest
import evlib


def test_list_available_models():
    """Test that all models are listed correctly."""
    # Use the Rust function directly
    models = evlib.processing.list_available_models()

    # Check that we have at least the expected models
    expected_models = [
        "e2vid_unet",
        "firenet",
        "e2vid_plus",
        "firenet_plus",
        "spade_e2vid",
        "ssl_e2vid",
        "et_net",
        "hyper_e2vid",
    ]

    # The Rust function returns a list of model names
    for expected in expected_models:
        assert expected in models, f"Expected model {expected} not found"


def test_e2vid_model_info():
    """Test that E2VID model has correct updated information."""
    # Get model info directly from Rust
    e2vid_info = evlib.processing.get_model_info_py("e2vid_unet")

    assert e2vid_info is not None
    assert (
        e2vid_info["url"] == "https://download.ifi.uzh.ch/rpg/web/data/E2VID/models/E2VID_lightweight.pth.tar"
    )
    assert e2vid_info["checksum"] == "sha256:4cfeb2c850bf48fc9fa907e969cb8a04e3c51314da2d65bdb81145ac96574128"
    assert e2vid_info["size"] == 42_878_232
    assert e2vid_info["format"] == "pytorch"


def test_new_architectures_available():
    """Test that new architectures (ET-Net, HyperE2VID) are available."""
    models = evlib.processing.list_available_models()

    assert "et_net" in models
    assert "hyper_e2vid" in models

    # Check architecture types
    et_net_info = evlib.processing.get_model_info_py("et_net")
    hyper_info = evlib.processing.get_model_info_py("hyper_e2vid")

    assert et_net_info is not None
    assert et_net_info["architecture"] == "ETNet"

    assert hyper_info is not None
    assert hyper_info["architecture"] == "HyperE2Vid"


def test_python_model_imports():
    """Test that Python model classes can be imported."""
    from evlib.models import E2VID, FireNet, ETNet, HyperE2VID

    # Test that classes exist
    assert E2VID is not None
    assert FireNet is not None
    assert ETNet is not None
    assert HyperE2VID is not None


def test_model_instantiation():
    """Test that models can be instantiated."""
    from evlib.models import ETNet, HyperE2VID

    # Test ET-Net instantiation
    et_net = ETNet(pretrained=False)
    assert et_net is not None
    assert hasattr(et_net, "_model_type")
    assert et_net._model_type == "et_net"

    # Test HyperE2VID instantiation
    hyper = HyperE2VID(pretrained=False)
    assert hyper is not None
    assert hasattr(hyper, "_model_type")
    assert hyper._model_type == "hyper_e2vid"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
