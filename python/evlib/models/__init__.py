"""
High-level API for event-to-video reconstruction models.

This module provides a unified interface for reconstruction models
available in evlib.
"""

try:
    from .config import ModelConfig

    _config_available = True
except ImportError:
    _config_available = False

# Only import models that don't have broken dependencies
__all__ = []

if _config_available:
    __all__.append("ModelConfig")

# Note: E2VID and other models are disabled due to missing dependencies
# They require additional dependencies (utils module, pretrained model loading, etc.)
# that are not currently implemented.
