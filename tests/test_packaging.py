import evlib


def test_evlib_imports_the_python_package_not_the_bare_extension():
    assert type(evlib.__loader__).__name__ == "SourceFileLoader"
    assert hasattr(evlib, "get_recommended_engine")


def test_filtering_is_the_python_polars_implementation():
    import evlib.filtering as f

    assert f.__name__ == "evlib.filtering"
    # Must be a real Python source file named filtering.py under the evlib package,
    # not the Rust extension (which would have no __file__ or a .so path). Normalise
    # the separator and match the install-agnostic suffix (editable vs site-packages).
    file_path = (getattr(f, "__file__", "") or "").replace("\\", "/")
    assert file_path.endswith("evlib/filtering.py"), file_path


def test_rvt_is_reachable():
    assert hasattr(evlib, "rvt")
    import importlib

    rvt = importlib.import_module("evlib.rvt")
    assert callable(rvt.process_sequence)


def test_configure_gpu_engine_is_exported():
    assert hasattr(evlib, "configure_gpu_engine")


def test_streaming_utils_is_gone():
    assert not hasattr(evlib, "streaming_utils")
