"""
High-level API for event-to-video reconstruction models.

This module provides a unified interface for reconstruction models
available in evlib.
"""

import logging

logger = logging.getLogger(__name__)

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

# Names gated on torch (and, for RVT/YOLOX, their own sibling submodules).
# When unavailable, __getattr__ below raises a clear "install evlib[torch]"
# error the first time a caller actually touches one of these, instead of an
# unexplained AttributeError (P8).
_E2VID_NAMES = ("E2VID", "E2VIDRecurrent")
_RVT_NAMES = (
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
)

# Records, per unavailable name, the ImportError that made it unavailable
# (missing torch, or a genuine import bug in e2vid.py/rvt.py) so
# `__getattr__` can surface the real cause instead of a generic message.
_import_errors = {}

if _config_available:
    __all__.append("ModelConfig")

if _torch_available:
    try:
        from .e2vid import E2VID
        from .e2vid_recurrent import E2VIDRecurrent

        __all__.extend(["E2VID", "E2VIDRecurrent"])
    except ImportError as e:
        logger.warning("Could not import E2VID model: %s", e)
        for _name in _E2VID_NAMES:
            _import_errors[_name] = e

    try:
        from .rvt import RVT, RVTModelConfig
        from .rvt_backbone import RVTBackbone, RVTConfig
        from .yolox_blocks import BaseConv, DWConv, CSPLayer, Focus, SPPBottleneck
        from .yolox_fpn import PAFPN, YoloXFPN, create_yolox_fpn
        from .yolox_head import YOLOXHead, postprocess

        __all__.extend(_RVT_NAMES)
    except ImportError as e:
        logger.warning("Could not import RVT/YOLOX models: %s", e)
        for _name in _RVT_NAMES:
            _import_errors[_name] = e
else:
    logger.warning(
        "PyTorch not available; deep learning models will not be available. "
        "Install with: pip install evlib[torch]"
    )
    _no_torch_error = ImportError("PyTorch is not installed")
    for _name in _E2VID_NAMES + _RVT_NAMES:
        _import_errors[_name] = _no_torch_error


def __getattr__(name):
    if name in _import_errors:
        raise ImportError(
            f"evlib.models.{name} is unavailable: {_import_errors[name]}. "
            f"Install with: pip install evlib[torch]"
        ) from _import_errors[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
