"""
High-level API for event-to-video reconstruction models.

This module provides a unified interface for all reconstruction models
available in evlib, with automatic model downloading and easy-to-use APIs.
"""

from .e2vid import E2VID
from .firenet import FireNet
from .e2vid_plus import E2VIDPlus
from .firenet_plus import FireNetPlus
from .spade import SPADE
from .ssl import SSL
from .et_net import ETNet
from .hyper_e2vid import HyperE2VID
from .config import ModelConfig, ModelInfo
from .utils import list_models, download_model, get_model_path

__all__ = [
    # Models
    "E2VID",
    "FireNet",
    "E2VIDPlus",
    "FireNetPlus",
    "SPADE",
    "SSL",
    "ETNet",
    "HyperE2VID",
    # Configuration
    "ModelConfig",
    "ModelInfo",
    # Utilities
    "list_models",
    "download_model",
    "get_model_path",
]
