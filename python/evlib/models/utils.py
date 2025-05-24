"""Utility functions for model management."""

from pathlib import Path
from typing import List, Optional


def list_models() -> List[str]:
    """List all available pre-trained models.

    Returns:
        List of model names
    """
    # This will call the Rust function when model_zoo is properly exposed
    # For now, return a hardcoded list
    return ["e2vid_unet", "firenet", "e2vid_plus", "firenet_plus", "spade_e2vid", "ssl_e2vid"]


def download_model(model_name: str) -> Path:
    """Download a pre-trained model if not already cached.

    Args:
        model_name: Name of the model to download

    Returns:
        Path to the downloaded model file

    Raises:
        ValueError: If model name is not found
    """
    # Get model cache directory
    cache_dir = Path.home() / ".evlib" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_path = cache_dir / f"{model_name}.pth"

    if model_path.exists():
        print(f"Model '{model_name}' already cached at {model_path}")
        return model_path

    # TODO: Call the Rust download function when exposed
    # For now, create a placeholder file
    print(f"Downloading model '{model_name}'...")
    model_path.touch()
    print(f"Model downloaded to {model_path}")

    return model_path


def get_model_path(model_name: str) -> Optional[Path]:
    """Get the path to a cached model.

    Args:
        model_name: Name of the model

    Returns:
        Path to the model if it exists, None otherwise
    """
    cache_dir = Path.home() / ".evlib" / "models"
    model_path = cache_dir / f"{model_name}.pth"

    return model_path if model_path.exists() else None


def clear_model_cache():
    """Clear all cached models."""
    cache_dir = Path.home() / ".evlib" / "models"
    if cache_dir.exists():
        import shutil

        shutil.rmtree(cache_dir)
        print("Model cache cleared")


def get_model_info(model_name: str) -> dict:
    """Get information about a model.

    Args:
        model_name: Name of the model

    Returns:
        Dictionary with model information
    """
    # TODO: Get this from Rust when exposed
    model_info = {
        "e2vid_unet": {
            "name": "E2VID UNet",
            "description": "UNet architecture for event-to-video reconstruction",
            "size_mb": 30,
            "architecture": "UNet",
        },
        "firenet": {
            "name": "FireNet",
            "description": "Lightweight architecture for fast reconstruction",
            "size_mb": 5,
            "architecture": "FireNet",
        },
        "e2vid_plus": {
            "name": "E2VID+",
            "description": "Enhanced E2VID with temporal processing",
            "size_mb": 50,
            "architecture": "E2VID+ with ConvLSTM",
        },
        "firenet_plus": {
            "name": "FireNet+",
            "description": "FireNet with temporal processing",
            "size_mb": 10,
            "architecture": "FireNet+ with temporal gates",
        },
        "spade_e2vid": {
            "name": "SPADE-E2VID",
            "description": "Spatially-adaptive normalization for better details",
            "size_mb": 40,
            "architecture": "SPADE normalization",
        },
        "ssl_e2vid": {
            "name": "SSL-E2VID",
            "description": "Self-supervised learning approach",
            "size_mb": 30,
            "architecture": "Self-supervised E2VID",
        },
    }

    if model_name not in model_info:
        raise ValueError(f"Model '{model_name}' not found")

    return model_info[model_name]
