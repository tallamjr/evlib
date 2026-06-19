# Release Notes

Release history and changelog for evlib.

## Release Philosophy

### Semantic Versioning

evlib follows [Semantic Versioning](https://semver.org/) (SemVer):

- **MAJOR**: Breaking changes to the public API
- **MINOR**: New features, backwards compatible
- **PATCH**: Bug fixes, backwards compatible

### Release Quality

- **Robust over rapid**: every release is tested against real data.
- **Real data validation**: format readers and the RVT pipeline are checked against real event camera datasets.
- **Performance verification**: the committed RVT pipeline benchmark (`benchmarks/bench_rvt_pipeline.py`) is the reproducible performance gate.

---

## Changelog

### 0.9.x (current)

The crate version is `0.9.0` (`Cargo.toml`). This series adds native GPU and Apple Silicon compute backends for the RVT stacked-histogram path and tightens the bit-identity guarantees against reference implementations.

**RVT preprocessing backends**

`evlib.rvt.process_sequence(...)` now exposes four backends through the `backend=` argument:

- `"polars"`: the Polars query layer on the CPU, or on the GPU via cudf-polars when `engine=` selects the GPU. This is the default.
- `"rust"`: the Rust dense scatter-add kernel, CPU only.
- `"cuda"`: a custom CUDA scatter-add kernel. The kernel is built with `nvcc` into `librvt_scatter.so` and loaded at runtime with `libloading`. The library is located via the `EVLIB_CUDA_LIB` environment variable.
- `"metal"`: a Metal/MSL scatter-add kernel for Apple Silicon, via `metal-rs`. Build with `CC=clang`.

The native kernels are exposed from the extension as `evlib.representations_rs.stacked_histogram_dense`, `stacked_histogram_dense_cuda`, and `stacked_histogram_dense_metal`.

**Cargo features**

The crate gained two optional Cargo features in addition to the existing set:

- `cuda` (pulls in `libloading` for runtime loading of the CUDA kernel).
- `metal` (pulls in `metal` and `objc`; macOS target only).

The full feature set is now `polars`, `python`, `arrow`, `hdf5`, `extension-module`, `zero-copy`, `cuda`, and `metal`. Default features remain `["polars", "python", "arrow"]`.

**Validated benchmark (RVT pipeline)**

Measured on the gen4_1mpx validation split (18 sequences, single pass), on an RTX 4090 for the GPU figures:

| Backend | Time | Note |
|---------|------|------|
| evlib CUDA | 283.6s | parity-plus with the RVT torch reference |
| RVT torch (GPU) | 286.3s | reference |
| evlib Rust (CPU) | 406.2s | 1.32x faster than the RVT torch CPU reference |
| RVT torch (CPU) | 534.2s | reference |

evlib CUDA is about 1.01x faster than RVT torch on the GPU, and about 1.88x faster than the RVT torch CPU reference. The evlib output is bit-identical to the RVT torch reference, bar a roughly 1e-10 float-binning boundary quirk that affects 3 of the 18 sequences.

**Bit-identity validation breadth**

The stacked-histogram output has been validated bit-identical against RVT (torch), tonic, OpenEB, and dv_processing.

### 0.8.x

The 0.8.x series trimmed the feature surface to keep the crate lean:

- Removed the Rust visualisation module and its dependency cascade (tokio, warp, reqwest, ratatui, gstreamer, and others).
- Removed the dead `tch` and `ort` (ONNX runtime) bindings from the Rust crate.
- Made HDF5 an opt-in Cargo feature (`--features hdf5`) on Linux and macOS; it is not available on Windows.

The Python API was unchanged. Deep-learning models (E2VID, RVT) remain available Python-side via PyTorch in `evlib.models`. `evlib.visualization` remains available as a pure-Python module.

!!! note "Removed modules"
    The Rust visualisation module, web visualisation, GStreamer integration, ETAP point tracking, ONNX runtime bindings, and the smooth voxel grid are no longer part of the crate. They are not referenced in the current API.

---

## Installation

### System Requirements

- **Python**: >= 3.11 (supported: 3.11, 3.12, 3.13; 3.12 recommended)
- **NumPy**: >= 1.24.0
- **Rust**: nightly toolchain (see `rust-toolchain.toml`)
- **Optional**: HDF5 system libraries (opt-in via `--features hdf5`, Linux and macOS only)
- **Optional**: PyTorch for the deep-learning models
- **Optional**: a CUDA toolchain (`nvcc`) for the `cuda` backend, or Apple Silicon with `CC=clang` for the `metal` backend

### Installation Commands

```bash
# Latest release from PyPI
pip install evlib

# Development version from source
pip install git+https://github.com/tallamjr/evlib.git

# With all optional dependencies
pip install evlib[all]
```

---

## Release Process

### Quality Gates

1. Python and Rust tests pass (`pytest`, `cargo test`).
2. The RVT pipeline benchmark runs and meets expectations.
3. Documentation is updated for new features.
4. Breaking changes are documented.

### Local-only gates

The `slow` and `integration` markered suites (including the RVT bit-identity acceptance test) are local-only gates. They depend on large gitignored datasets and are not run on hosted CI. Run them locally with:

```bash
.venv/bin/pytest -m "slow or integration" --run-slow --run-integration
```

---

## Support

- **GitHub Issues**: bug reports and feature requests.
- **Discussions**: general questions and community support.
- **Documentation**: guides, API reference, and examples.
