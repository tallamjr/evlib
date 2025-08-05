"""
High-level API for event-to-video reconstruction models.

This module provides a unified interface for reconstruction models
available in evlib.
"""

# Always try to import config
try:
    from .config import ModelConfig

    _config_available = True
except ImportError:
    _config_available = False

# Try to import PyTorch-based models
try:
    import torch

    _torch_available = True
except ImportError:
    _torch_available = False

__all__ = []

if _config_available:
    __all__.append("ModelConfig")

if _torch_available:
    try:
        from .e2vid import E2VID

        __all__.append("E2VID")
    except ImportError as e:
        print(f"Warning: Could not import E2VID model: {e}")

    try:
        from .rvt import RVT, RVTModelConfig
        from .rvt_backbone import RVTBackbone, RVTConfig
        from .yolox_blocks import BaseConv, DWConv, CSPLayer, Focus, SPPBottleneck
        from .yolox_fpn import PAFPN, YoloXFPN, create_yolox_fpn
        from .yolox_head import YOLOXHead, postprocess

        __all__.extend(
            [
                "RVT",
                "RVTModelConfig",
                "RVTBackbone",
                "RVTConfig",
                "BaseConv",
                "DWConv",
                "CSPLayer",
                "Focus",
                "SPPBottleneck",
                "PAFPN",
                "YoloXFPN",
                "create_yolox_fpn",
                "YOLOXHead",
                "postprocess",
            ]
        )
    except ImportError as e:
        print(f"Warning: Could not import RVT/YOLOX models: {e}")

if not _torch_available:
    print("Warning: PyTorch not available. Deep learning models will not be available.")
    print("Install PyTorch with: pip install torch")
