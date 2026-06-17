import evlib


def test_evlib_imports_the_python_package_not_the_bare_extension():
    assert type(evlib.__loader__).__name__ == "SourceFileLoader"
    assert hasattr(evlib, "get_recommended_engine")


def test_filtering_is_the_python_polars_implementation():
    import evlib.filtering as f

    assert f.__name__ == "evlib.filtering"
    assert getattr(f, "__file__", "").endswith("python/evlib/filtering.py")


def test_rvt_is_reachable():
    assert hasattr(evlib, "rvt")
    import importlib

    rvt = importlib.import_module("evlib.rvt")
    assert callable(rvt.process_sequence)
