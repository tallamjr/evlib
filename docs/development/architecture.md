# Architecture

evlib keeps a thin Rust core and does all DataFrame work in Polars from Python. Rust handles only what cannot be expressed as DataFrame operations: parsing binary event formats and building the Polars frame, plus one dense scatter-add kernel for RVT histograms. Everything else (filtering, representations, models, visualisation) lives in Python.

## Design Philosophy

### Core Principles

1. **Thin Rust core, Polars everywhere else**: Rust does binary decoding; Python Polars does the processing.
2. **Lazy and engine-selectable**: loaders return a Polars `LazyFrame`, so the same query runs on the CPU streaming engine or on the GPU via cudf-polars (`collect(engine="gpu")`) where CUDA is available.
3. **Real data validation**: format readers are tested against real event camera datasets.
4. **No dead weight**: the crate carries only the dependencies the two Rust modules actually use.

### Why this split

DataFrame work (filtering, windowing, grouping, representation maths) is exactly what Polars is built for, and expressing it as lazy Polars queries makes it GPU-collectable for free. Only the parts Polars cannot do, decoding camera-specific binary containers into columns and the dense scatter-add behind RVT stacked histograms, stay in Rust.

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Python (evlib)                           │
│  load_events  ·  filtering  ·  representations  ·  rvt      │
│  models (E2VID, RVT)  ·  visualization  ·  simulation       │
│            all processing as lazy Polars queries            │
└─────────────────────────────────────────────────────────────┘
           │  pl.LazyFrame [x, y, t, polarity]
           ▼
┌─────────────────────────────────────────────────────────────┐
│                  PyO3 boundary (evlib._evlib)               │
│         load_events · detect_format · arrow bridge          │
│         representations_rs (dense scatter-add)              │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│                      Rust core (src/)                       │
│  ev_formats         → binary decode + Polars frame build    │
│  ev_representations → dense scatter-add for RVT histograms  │
└─────────────────────────────────────────────────────────────┘
```

### Module Hierarchy

```
evlib/
├── src/                            # Rust source (compiled to evlib._evlib)
│   ├── ev_formats/                 # Binary parsing + frame construction
│   │   ├── mod.rs                  # Public API, load/save, PyO3 wrappers
│   │   ├── format_detector.rs      # Automatic format detection
│   │   ├── evt2_reader.rs          # Prophesee EVT2
│   │   ├── evt21_reader.rs         # Prophesee EVT2.1
│   │   ├── evt3_reader.rs          # Prophesee EVT3
│   │   ├── aedat_reader.rs         # AEDAT (iniVation)
│   │   ├── aedat4_reader.rs        # AEDAT 4.0 (flatbuffer + LZ4)
│   │   ├── aer_reader.rs           # AER
│   │   ├── hdf5_reader.rs          # HDF5 (opt-in, feature = "hdf5")
│   │   ├── ecf_codec.rs            # ECF codec
│   │   ├── prophesee_ecf_codec.rs  # Prophesee ECF variant
│   │   ├── evnt_tcp_reader.rs      # Streaming TCP reader (tokio)
│   │   ├── dataframe_builder.rs    # Decoded primitives → Polars frame
│   │   ├── arrow_builder.rs        # Apache Arrow zero-copy bridge
│   │   ├── polarity_handler.rs     # 0/1 ↔ -1/1 polarity encoding
│   │   └── streaming.rs            # Chunked reads for large files
│   ├── ev_representations/         # Dense scatter-add kernel
│   │   ├── mod.rs                  # PyO3 registration (representations_rs)
│   │   └── stacked_histogram_dense.rs  # RVT stacked-histogram scatter-add
│   ├── tracing_config.rs           # Structured logging (tracing)
│   ├── bin/                        # Small Rust binaries
│   └── lib.rs                      # PyO3 module definition
└── python/
    └── evlib/                      # Python package (the import entry point)
        ├── __init__.py             # load_events, engine config, re-exports
        ├── filtering.py            # Event filtering (pure Python Polars)
        ├── representations.py      # Representations (pure Python Polars)
        ├── rvt/                    # RVT-identical preprocessing pipeline
        ├── models/                 # E2VID and RVT (Python / PyTorch)
        ├── visualization.py        # Plotting helpers (Python-only)
        └── simulation.py           # ESIM video-to-events simulation
```

There is no `ev_core`, `ev_processing`, `ev_transforms`, or `ev_tracking` module: those were removed. The crate has no Rust machine-learning backend (no `tch`, `ort`/ONNX, or candle bindings); all models live in Python under `python/evlib/models/`.

## Data Flow

The path from a file on disk to a usable frame is short:

```
binary file ──► Rust decode (ev_formats) ──► Polars LazyFrame ──► Python Polars
  EVT2/3,        format detection,            columns:             filtering,
  AEDAT, AER,    container parsing,           [x, y, t, polarity]  representations,
  HDF5, text     polarity handling                                 models, plots
```

1. `evlib.load_events(path, ...)` calls the Rust loader, which detects the format, decodes the container, normalises polarity, and builds a Polars frame.
2. The Rust side returns a `LazyFrame` with columns `[x, y, t, polarity]`: `t` is a Duration in microseconds and `polarity` is already `-1`/`+1`.
3. `load_events` applies any time, spatial, or polarity filters as Polars expressions, so loading and filtering fuse into one lazy query, then optionally sorts by `t` (default `sort=True`).
4. Downstream work (`evlib.filtering`, `evlib.representations`, `evlib.rvt`, `evlib.models`) is all Polars, collected with a selectable engine.

For the RVT preprocessing path, `evlib.rvt` builds its lazy Polars query and hands the binned events to the Rust `representations_rs` dense scatter-add kernel for the final stacked-histogram accumulation.

## Core Components

### ev_formats: binary decode and frame construction

`ev_formats` is the only place that touches raw bytes. It provides automatic format detection (`detect_format`) and per-format readers for EVT2, EVT2.1, EVT3, AEDAT, AEDAT 4.0, AER, HDF5 (with the ECF codec), and text. Decoded primitives flow through `dataframe_builder.rs` to a Polars frame, or through `arrow_builder.rs` for a zero-copy Apache Arrow bridge. `polarity_handler.rs` normalises 0/1 and -1/1 encodings, and `streaming.rs` provides chunked reads for files too large to load whole.

HDF5 support is gated behind the `hdf5` Cargo feature (Linux and macOS only). When the feature is off, or on Windows, use `h5py` from Python for HDF5 I/O.

### ev_representations: the dense scatter-add kernel

`ev_representations` exposes a single dense scatter-add used by the RVT stacked-histogram path. It is registered with Python as `evlib.representations_rs` (named with the `_rs` suffix so it does not collide with the pure-Python `evlib.representations` module). All other representations (voxel grids, mixed density stacks, the general stacked histogram) are pure Python Polars in `evlib.representations`.

### Python processing layer

All processing is Python Polars:

- `evlib.filtering`: time, ROI, polarity, hot-pixel, and noise filters as lazy Polars expressions.
- `evlib.representations`: stacked histograms, voxel grids, mixed density stacks.
- `evlib.rvt`: the RVT-identical preprocessing pipeline (Polars plus the Rust scatter backend).
- `evlib.models`: E2VID and RVT models in Python/PyTorch with real pretrained weights.
- `evlib.visualization`: plotting helpers, pure Python.

## PyO3 Boundary

`src/lib.rs` defines the `_evlib` PyO3 module, built by maturin as `evlib._evlib`. It registers:

- top-level `load_events`, `detect_format`, and the Arrow bridge functions;
- a `formats` submodule (load/save, detection, ECF test, Arrow conversion);
- a `core` submodule with a handful of migrated helper functions (`merge_events`, `add_random_events`, `remove_events`, `events_to_block`);
- `representations_rs` (the dense scatter-add);
- `tracing_config` for logging control.

Frames cross the boundary as Polars objects via `pyo3-polars`; helpers in `lib.rs` (`extract_lazy_frame`, `lazy_frame_to_python`) convert between a Python Polars DataFrame and a Rust `LazyFrame`.

## Build System

### Maturin

evlib is built with maturin (PyO3 bindings). The Python package lives under `python/` and the compiled extension is `evlib._evlib`:

```toml
# pyproject.toml
[build-system]
requires = ["maturin>=1.3.2"]
build-backend = "maturin"

[tool.maturin]
python-source = "python"
module-name = "evlib._evlib"
strip = true
```

Python 3.11, 3.12, and 3.13 are supported (`requires-python = ">=3.11"`).

### Feature Flags

```toml
# Cargo.toml
[features]
default          = ["polars", "python", "arrow"]
python           = ["dep:pyo3", "dep:pyo3-ffi", "dep:numpy", "dep:pyo3-polars"]
extension-module = ["pyo3/extension-module"]
polars           = ["dep:polars"]
arrow            = ["dep:arrow", "dep:arrow-array", "dep:pyo3-arrow"]
zero-copy        = ["arrow"]   # alias for clarity
hdf5             = ["dep:hdf5-metno", "dep:hdf5-metno-sys"]  # Unix only
```

Key points:

- `extension-module` is deliberately **off** by default. With it off, PyO3's build script links the present libpython, so `cargo test` and `maturin develop` build and run without any `RUSTFLAGS` hack. Turn it on only for distributable wheels, e.g. `maturin build --release --features python,polars,arrow,extension-module`.
- `hdf5` is opt-in and Unix only. On Windows the underlying HDF5 dependencies are not available, so the feature compiles to no-op stubs there and Python `h5py` is used instead.
- `zero-copy` is an alias for `arrow`.

There are no `pytorch`, `cuda`, `mkl`, or `metal` Cargo features: GPU acceleration comes from cudf-polars at the Python layer (`collect(engine="gpu")`), not from Rust.

### Common commands

```bash
maturin develop                 # default minimal build (polars + python + arrow)
maturin develop --features hdf5 # opt-in HDF5 (Linux/macOS)
maturin develop --release       # release build for performance work
cargo test                      # Rust tests (no special flags needed)
pytest                          # Python tests
```

## Error Handling

Format readers return descriptive errors that include file-offset context where it helps debugging, and surface to Python as standard exceptions (for example `PyIOError` / `PyRuntimeError`). Polarity-encoding variations are handled rather than rejected, and very large files fall back to streaming reads. Structured logging is available via the `tracing` crate, controlled from Python through `evlib._evlib.tracing_config`.

## Related crates

`wasm/` is a separate tracked crate that builds a WebAssembly event-data demo. It is independent of the Python extension and is not part of the `evlib` Python package.

---

*The architecture keeps Rust minimal and pushes all DataFrame processing into Polars, so the same lazy queries scale from the CPU streaming engine to the GPU without rewriting the pipeline.*
