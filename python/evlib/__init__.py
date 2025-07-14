# This file is required for Python to recognize this directory as a package

import numpy as np

# Import submodules (with graceful fallback)
try:
    from . import models
    _models_available = True
except ImportError:
    _models_available = False

try:
    from . import representations
    _representations_available = True
    
    # Import key representation functions directly
    from .representations import (
        stacked_histogram,
        create_voxel_grid,
        create_time_surface,
        create_event_histogram,
        smooth_voxel  # backward compatibility
    )
except ImportError:
    _representations_available = False

# Import data reading functions from Rust module
try:
    from .evlib import formats
    _formats_available = True
    
    # Make data reading functions directly accessible
    try:
        load_events = formats.load_events
        save_events_to_hdf5 = formats.save_events_to_hdf5
        save_events_to_text = formats.save_events_to_text
        detect_format = formats.detect_format
        get_format_description = formats.get_format_description
    except AttributeError:
        # Some functions might not be available in this build
        _formats_available = False
    
except ImportError:
    _formats_available = False

# Export the available functionality
__all__ = []

if _models_available:
    __all__.append("models")
if _representations_available:
    __all__.extend(["representations", "stacked_histogram", "create_voxel_grid", "create_time_surface", "create_event_histogram", "smooth_voxel"])
if _formats_available:
    __all__.extend(["formats", "load_events", "save_events_to_hdf5", "save_events_to_text", "detect_format", "get_format_description"])
