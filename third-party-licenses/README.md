# Third-party licences

These licence texts apply only to wheels built with the `hdf5-static` Cargo
feature (macOS and manylinux wheel builds in `.github/workflows/publish.yml`).
That feature statically links HDF5, blosc and zlib into the compiled
extension from source (see `vendor/hdf5-metno-sys-0.10.1-static-fix/`), so
those libraries' redistribution notices must ship with the resulting binary.
Wheels built without `hdf5-static` (Windows, or a local
`maturin develop`/`--features hdf5` dynamic-link build) do not embed these
libraries and do not need these notices, but the files are included in every
wheel unconditionally for simplicity.

- `HDF5-COPYING.txt`: HDF5 (Hierarchical Data Format 5), BSD-style licence,
  copied verbatim from `ext/hdf5/COPYING` in the `hdf5-metno-src` 0.9.5 crate
  (the exact vendored HDF5 source the static build compiles).
- `blosc-LICENSE.txt`: c-blosc, BSD 3-Clause licence, copied verbatim from
  `c-blosc/LICENSE.txt` in the `blosc-src` 0.3.6 crate.
- `zlib-LICENSE.txt`: zlib, zlib licence, copied verbatim from
  `src/zlib/LICENSE` in the `libz-sys` 1.1.22 crate (the `stock-zlib` variant,
  which is the default feature and the one evlib uses; not `zlib-ng`).
