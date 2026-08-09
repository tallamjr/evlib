# Provenance

Vendored verbatim from crates.io `hdf5-metno-sys` 0.10.1
(https://crates.io/crates/hdf5-metno-sys/0.10.1, upstream repo
https://github.com/metno/hdf5-rust), on 2026-08-09.

Every file under `src/` and `Cargo.toml` is byte-identical to the crates.io
0.10.1 source (verified with `diff -rq` against
`~/.cargo/registry/src/*/hdf5-metno-sys-0.10.1/`). `LICENSE-MIT` and
`LICENSE-APACHE` are copied unmodified from the upstream repository's `main`
branch (the crate is dual-licensed `MIT OR Apache-2.0`; crates.io downloads do
not include separate licence files, so they are not present in the registry
cache this vendor copy was diffed against, only in the upstream git repo).

## The one intentional change

`build.rs`, function `get_build_and_emit()` (the `static`-feature build path,
used only when the `hdf5-static` Cargo feature is enabled): on macOS only, the
two `cargo::rustc-link-lib=static=...` directives become
`cargo::rustc-link-lib=static:-bundle=...`.

Why: bundling the from-source `libhdf5.a` into an intermediate rlib (the
default behaviour for `static` link-kind directives from a build script) hits
a rustc archive-writer bug on macOS. Rustc appends the bundled static
library's raw archive members - including that library's own pre-existing
`__.SYMDEF` index - into the crate's rlib verbatim, without recomputing a
single combined index. The result has two `__.SYMDEF` entries, which neither
Apple's default linker nor the deprecated classic linker (`-ld_classic`) can
parse; the link fails with `ld: multiple SYMDEF member files found in an
archive`. `libhdf5.a` was confirmed to carry exactly one clean SYMDEF before
bundling, so the corruption is introduced by rustc, not by the CMake-built
HDF5 archive.

The `-bundle` modifier keeps `libhdf5.a` external instead of embedding it: it
is linked once, at final `cdylib` link time, via the `-L` search path the
build script already emits, so the double-index problem never arises. Linux
is unaffected (bundling links fine there), so only the `target_os = "macos"`
branch is patched; the Linux/other-Unix build behaviour is unchanged from
upstream.

There is no upstream issue filed for this (not checked in depth beyond
confirming the symptom); the fix was derived locally from inspecting the
`.rlib` archive layout with `ar -tv` and cross-referencing the general "ld:
multiple SYMDEF member files found in an archive" failure mode reported for
other projects that bundle C static libraries into Rust archives on macOS.

## Drop condition

Drop this vendored copy and go back to depending on `hdf5-metno-sys` from
crates.io directly (remove the `[patch.crates-io]` entry and this directory)
once either:

- upstream (https://github.com/metno/hdf5-rust) fixes the `static` feature's
  link-modifier choice on macOS, or exposes a way to opt into `-bundle`
  without a fork, or
- the underlying rustc archive-writer bug is fixed so that bundling a
  pre-indexed static library into an rlib produces one clean SYMDEF instead
  of two.

## Scope and operational notes (repo-only, do not publish as-is)

- `hdf5-static` is a **wheel-build/repo-only** feature on macOS. `cargo
  publish` strips `[patch]` sections from the published manifest, so a
  downstream crate consumer who enables `hdf5-static` against the plain
  crates.io `hdf5-metno-sys` 0.10.1 (i.e. without this patch) will hit the
  exact SYMDEF link failure this vendor copy works around. This feature is
  only meant to be built from this repository (CI wheel builds, or local
  `maturin`/`cargo build` runs against this checkout), never by a consumer of
  a published `evlib` crate.
- The `AR=/usr/bin/ar` / `RANLIB=/usr/bin/ranlib` pins in `publish.yml` and in
  local build instructions are **partial insurance only**. They guard against
  a GNU `ar` earlier on `PATH` (e.g. from a Homebrew binutils install)
  producing a GNU-format archive Apple's linker can't read. They do **not**
  guard against CMake picking a different `ar`/`ranlib` by itself: the `cmake`
  Rust crate does not read or forward an `AR`/`RANLIB` environment variable
  into `CMAKE_AR`/`CMAKE_RANLIB` (confirmed by reading its source - no such
  env var handling exists), so CMake's own `find_program`-based toolchain
  detection is what actually decides which `ar`/`ranlib` gets used. The env
  pins only help by removing the wrong tool from `PATH` before CMake looks;
  they are not a hard guarantee on a machine where a non-Apple `ar` happens to
  rank first in `CMAKE_AR`'s own search order for other reasons.
- **Never publish an sdist (`cargo publish` of `evlib` itself, or any process
  that produces a source tarball) while the `[patch.crates-io]` +
  `exclude = ["vendor/**/*"]` combination in the root `Cargo.toml` exists.**
  The exclude keeps `vendor/` out of the published `.crate` file, and `cargo
  publish` drops `[patch]` sections automatically, so a published sdist looks
  fine on the surface - but anyone trying to reproduce an `hdf5-static` build
  from that sdist alone (without this git checkout) cannot, since the patch
  and its target are both absent. `hdf5-static` builds must be done from this
  git repository, not from the published crate.
