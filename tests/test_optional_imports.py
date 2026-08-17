"""Regression tests for evlib's optional-import behaviour (P7, P8).

P7: `import evlib` must not shell out to `nvidia-smi` or mutate Polars'
process-wide engine affinity as an import side effect (`configure_gpu_engine`
is the explicit opt-in for that).

P8: a submodule that fails to import because an extra is missing must raise a
clear "install evlib[X]" ImportError the first time a caller touches it,
rather than silently becoming `None` (which turns every use site into an
unexplained AttributeError).
"""

import importlib
import subprocess
import sys
from unittest import mock

import polars as pl
import pytest

import evlib


def test_import_does_not_call_nvidia_smi_or_mutate_polars_engine(monkeypatch):
    """P7 regression: reloading evlib must not call subprocess.run or mutate
    Polars' engine affinity; that is now opt-in via configure_gpu_engine()."""
    run_calls = []
    original_run = subprocess.run

    def _tracking_run(*args, **kwargs):
        run_calls.append(args)
        return original_run(*args, **kwargs)

    affinity_calls = []
    original_set_engine_affinity = pl.Config.set_engine_affinity

    def _tracking_set_engine_affinity(*args, **kwargs):
        affinity_calls.append(args)
        return original_set_engine_affinity(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", _tracking_run)
    monkeypatch.setattr(
        pl.Config,
        "set_engine_affinity",
        staticmethod(_tracking_set_engine_affinity),
    )

    importlib.reload(evlib)

    assert run_calls == [], f"import evlib called subprocess.run: {run_calls}"
    assert affinity_calls == [], (
        f"import evlib mutated Polars engine affinity: {affinity_calls}"
    )
    assert evlib._engine_type == "streaming"
    assert evlib._gpu_available is False


def test_configure_gpu_engine_is_explicit_opt_in():
    """configure_gpu_engine must exist, be callable, and never be invoked by
    import evlib itself (proven by the test above)."""
    assert hasattr(evlib, "configure_gpu_engine")
    assert callable(evlib.configure_gpu_engine)
    result = evlib.configure_gpu_engine()
    assert result in ("gpu", "streaming")
    assert evlib._engine_type == result


def test_streaming_utils_no_longer_exists():
    """The nonexistent streaming_utils import + dead __all__ append are gone."""
    assert not hasattr(evlib, "streaming_utils")
    assert "streaming_utils" not in evlib.__all__


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="visualization.py also imports h5py, which pyproject.toml excludes "
    "on win32 (HDF5 is unsupported there); evlib.visualization is genuinely "
    "unavailable on Windows regardless of the plot extra, so this platform "
    "cannot exercise the 'real module' path.",
)
def test_visualization_available_is_the_real_module():
    """In this environment cv2 and h5py are installed, so evlib.visualization
    must be the genuine module, not the lazy-error proxy (no false positives)."""
    assert not isinstance(evlib.visualization, evlib._LazyImportErrorModule)
    assert hasattr(evlib.visualization, "EventFrameRenderer")


def test_visualization_raises_clear_error_without_cv2(monkeypatch):
    """P8 regression: with cv2 unavailable, evlib.visualization must raise a
    clear install hint on first attribute access, not silently be None."""
    monkeypatch.delitem(sys.modules, "evlib.visualization", raising=False)
    monkeypatch.delitem(sys.modules, "cv2", raising=False)
    monkeypatch.setitem(sys.modules, "cv2", None)
    # importlib.reload() re-executes __init__.py into evlib's existing
    # __dict__ without clearing it first. CPython's import machinery
    # (_handle_fromlist) skips reimporting a submodule when the parent
    # already has that attribute, so the stale real module would be reused
    # unless it is also removed here.
    monkeypatch.delattr(evlib, "visualization", raising=False)
    try:
        importlib.reload(evlib)
        assert evlib.visualization is not None
        with pytest.raises(ImportError, match=r"evlib\[plot\]"):
            evlib.visualization.EventFrameRenderer
    finally:
        monkeypatch.undo()
        monkeypatch.delitem(sys.modules, "evlib.visualization", raising=False)
        importlib.reload(evlib)


def test_representations_and_rvt_are_unconditional_imports():
    """No None-swallow left for representations/rvt: both are always the
    real module (they have no optional dependencies)."""
    assert not isinstance(evlib.representations, evlib._LazyImportErrorModule)
    assert not isinstance(evlib.rvt, evlib._LazyImportErrorModule)
    assert callable(evlib.rvt.process_sequence)


def test_models_getattr_raises_clear_error_without_torch(monkeypatch):
    """P8 regression: evlib.models.RVT must raise a clear install hint, not
    AttributeError, when torch is unavailable."""
    import evlib.models as models_pkg

    monkeypatch.setitem(sys.modules, "torch", None)
    # importlib.reload() re-executes the module into its existing __dict__
    # without clearing it first, so a previously-imported RVT/E2VID class
    # would still satisfy attribute lookup and __getattr__ would never run,
    # unless those stale attributes are also removed here before reload.
    monkeypatch.delattr(models_pkg, "RVT", raising=False)
    monkeypatch.delattr(models_pkg, "E2VID", raising=False)
    try:
        reloaded = importlib.reload(models_pkg)
        with pytest.raises(ImportError, match=r"evlib\[torch\]"):
            reloaded.RVT
        with pytest.raises(ImportError, match=r"evlib\[torch\]"):
            reloaded.E2VID
    finally:
        monkeypatch.undo()
        importlib.reload(models_pkg)


def test_simulation_getattr_raises_clear_error_without_opencv(monkeypatch):
    """P8 regression: evlib.simulation.VideoToEvents must raise a clear
    install hint, not AttributeError, when opencv is unavailable."""
    import evlib.simulation as simulation_pkg

    monkeypatch.setitem(sys.modules, "cv2", None)
    # Same stale-attribute hazard as above: remove the previously-imported
    # real name before reload so __getattr__ is actually exercised.
    monkeypatch.delattr(simulation_pkg, "VideoToEvents", raising=False)
    try:
        reloaded = importlib.reload(simulation_pkg)
        with pytest.raises(ImportError, match=r"evlib\[plot\]"):
            reloaded.VideoToEvents
    finally:
        monkeypatch.undo()
        importlib.reload(simulation_pkg)


def test_models_and_simulation_unaffected_when_extras_present():
    """Sanity check: with torch/cv2 genuinely installed (this environment),
    every gated name resolves normally, __getattr__ is never invoked."""
    import evlib.models as models_pkg
    import evlib.simulation as simulation_pkg

    assert models_pkg.RVT is not None
    assert models_pkg.E2VID is not None
    assert simulation_pkg.VideoToEvents is not None
