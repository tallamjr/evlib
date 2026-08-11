# Architecture

evlib keeps a thin Rust core and does all DataFrame work in Polars from Python. Rust handles only what cannot be expressed as DataFrame operations: parsing binary event formats and building the Polars frame, plus the dense scatter-add kernels (CPU, CUDA, and Metal variants) for RVT histograms. Everything else (filtering, representations, models, visualisation) lives in Python.

## Design Philosophy

### Core Principles

1. **Thin Rust core, Polars everywhere else**: Rust does binary decoding; Python Polars does the processing.
2. **Lazy and engine-selectable**: loaders return a Polars `LazyFrame`, so the same query runs on the CPU streaming engine or on the GPU via cudf-polars (`collect(engine="gpu")`) where CUDA is available.
3. **Query layer vs compute layer**: Polars is the query, filter, and transform layer (CPU, the cudf GPU engine, and CUDA unified managed memory for larger-than-VRAM work). The native Rust, CUDA, and Metal scatter-add kernels are the compute layer for the stacked histogram. Polars-GPU is not a free win for a single transfer-bound operation; the custom scatter-add kernels are where the GPU wins.
4. **Real data validation**: format readers are tested against real event camera datasets, and the RVT stacked histogram is validated bit-identical against RVT (torch), tonic, OpenEB, and dv_processing.
5. **No dead weight**: the crate carries only the dependencies the Rust modules actually use.

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
│         load_events · detect_format                         │
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
│   │   ├── prophesee_ecf_codec.rs  # Prophesee ECF variant
│   │   ├── dataframe_builder.rs    # Decoded primitives → Polars frame
│   │   └── streaming.rs            # Chunked reads for large files
│   ├── ev_representations/         # Dense scatter-add kernels (CPU / CUDA / Metal)
│   │   ├── mod.rs                  # PyO3 registration (representations_rs)
│   │   ├── stacked_histogram_dense.rs   # RVT stacked-histogram scatter-add (CPU)
│   │   ├── stacked_histogram_cuda.rs    # CUDA scatter-add (feature = "cuda")
│   │   └── stacked_histogram_metal.rs   # Metal scatter-add (feature = "metal")
│   ├── tracing_config.rs           # Structured logging (tracing)
│   ├── bin/                        # Small Rust binaries
│   └── lib.rs                      # PyO3 module definition
└── python/
    └── evlib/                      # Python package (the import entry point)
        ├── __init__.py             # load_events, engine config, re-exports
        ├── filtering.py            # Event filtering (pure Python Polars)
        ├── representations.py      # Representations (pure Python Polars)
        ├── rvt/                    # RVT-compatible preprocessing pipeline
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

For the RVT preprocessing path, `evlib.rvt.process_sequence` builds its lazy Polars query and hands the binned events to one of four backends, selected with `backend=`, for the final stacked-histogram accumulation:

- `"polars"`: the Polars query layer on the CPU, or on the GPU via cudf-polars when `engine=` selects the GPU. This is the default.
- `"rust"`: the Rust dense scatter-add kernel, CPU only.
- `"cuda"`: a custom CUDA scatter-add kernel, built with `nvcc` into `librvt_scatter.so` and loaded at runtime via `libloading` (located through the `EVLIB_CUDA_LIB` environment variable).
- `"metal"`: a Metal/MSL scatter-add kernel for Apple Silicon, via `metal-rs`.

The native kernels are exposed as `evlib.representations_rs.stacked_histogram_dense`, `stacked_histogram_dense_cuda`, and `stacked_histogram_dense_metal`.

## Core Components

### ev_formats: binary decode and frame construction

`ev_formats` is the only place that touches raw bytes. It provides automatic format detection (`detect_format`) and per-format readers for EVT2, EVT2.1, EVT3, AEDAT, AEDAT 4.0, AER, HDF5 (with the ECF codec), and text. Decoded primitives flow through `dataframe_builder.rs` to a Polars frame, which normalises polarity to -1/1 directly. `streaming.rs` provides chunked reads for files too large to load whole.

HDF5 support is gated behind the `hdf5` Cargo feature (Linux and macOS only). When the feature is off, or on Windows, use `h5py` from Python for HDF5 I/O.

### ev_representations: the dense scatter-add kernels

`ev_representations` exposes the dense scatter-add used by the RVT stacked-histogram path, in three variants: `stacked_histogram_dense` (CPU), `stacked_histogram_dense_cuda` (custom CUDA kernel), and `stacked_histogram_dense_metal` (Metal/MSL kernel for Apple Silicon). They are registered with Python as `evlib.representations_rs` (named with the `_rs` suffix so it does not collide with the pure-Python `evlib.representations` module). The CUDA and Metal variants are gated behind the `cuda` and `metal` Cargo features respectively. All other representations (voxel grids, mixed density stacks, the general stacked histogram) are pure Python Polars in `evlib.representations`.

### Python processing layer

All processing is Python Polars:

- `evlib.filtering`: time, ROI, polarity, hot-pixel, and noise filters as lazy Polars expressions.
- `evlib.representations`: stacked histograms, voxel grids, mixed density stacks.
- `evlib.rvt`: the RVT-compatible preprocessing pipeline (Polars plus the Rust scatter backend).
- `evlib.models`: E2VID and RVT models in Python/PyTorch with real pretrained weights.
- `evlib.visualization`: plotting helpers, pure Python.

## PyO3 Boundary

`src/lib.rs` defines the `_evlib` PyO3 module, built by maturin as `evlib._evlib`. It registers:

- top-level `load_events` and `detect_format`;
- a `formats` submodule (load/save, detection, ECF test);
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
default          = ["polars", "python"]
python           = ["dep:pyo3", "dep:pyo3-ffi", "dep:numpy", "dep:pyo3-polars"]
extension-module = ["pyo3/extension-module"]
polars           = ["dep:polars"]
hdf5             = ["dep:hdf5-metno", "dep:hdf5-metno-sys"]  # Unix only, dynamic link
hdf5-static      = ["hdf5", "hdf5-metno/static", "hdf5-metno/zlib"]  # Unix only, builds HDF5 from source
cuda             = ["dep:libloading"]              # runtime-loaded CUDA scatter-add kernel
metal            = ["dep:metal", "dep:objc"]       # Metal scatter-add kernel, macOS target only
```

Key points:

- `extension-module` is deliberately **off** by default. With it off, PyO3's build script links the present libpython, so `cargo test` and `maturin develop` build and run without any `RUSTFLAGS` hack. Turn it on only for distributable wheels, e.g. `maturin build --release --features python,polars,extension-module`.
- `hdf5` and `hdf5-static` are opt-in and Unix only. `hdf5` links dynamically against a system HDF5 install; `hdf5-static` builds HDF5 from source instead, so no system HDF5 is needed. Release wheels for macOS and Linux are built with `hdf5-static`, so `pip install evlib` already includes statically linked HDF5 (and ECF) support on those platforms since 0.13.1. On Windows neither feature's underlying HDF5 dependencies are available, so they compile to no-op stubs there and Python `h5py` is used instead.
- `cuda` enables the custom CUDA scatter-add backend. It pulls in `libloading` so the `nvcc`-built `librvt_scatter.so` can be loaded at runtime; there is no link-time CUDA dependency. Set `EVLIB_CUDA_LIB` to point at the built library.
- `metal` enables the Metal/MSL scatter-add backend on Apple Silicon. It pulls in `metal` and `objc` and is restricted to the macOS target. Build with `CC=clang`.

GPU acceleration comes from two complementary places: cudf-polars at the Python query layer (`collect(engine="gpu")`), and the native CUDA and Metal scatter-add kernels in the compute layer behind the `cuda` and `metal` features.

### Common commands

```bash
maturin develop                        # default minimal build (polars + python), no HDF5
maturin develop --features hdf5-static # HDF5 built from source, no system HDF5 needed (Linux/macOS)
maturin develop --features hdf5        # HDF5 dynamically linked against a system install (Linux/macOS)
maturin develop --release              # release build for performance work
cargo test                      # Rust tests (no special flags needed)
pytest                          # Python tests
```

## Error Handling

Format readers return descriptive errors that include file-offset context where it helps debugging, and surface to Python as standard exceptions (for example `PyIOError` / `PyRuntimeError`). Polarity-encoding variations are handled rather than rejected, and very large files fall back to streaming reads. Structured logging is available via the `tracing` crate, controlled from Python through `evlib._evlib.tracing_config`.

## Related crates

`wasm/` is a separate tracked crate that builds a WebAssembly event-data demo. It is independent of the Python extension and is not part of the `evlib` Python package.

---

*The architecture keeps Rust minimal and pushes all DataFrame processing into Polars, so the same lazy queries scale from the CPU streaming engine to the GPU without rewriting the pipeline.*
