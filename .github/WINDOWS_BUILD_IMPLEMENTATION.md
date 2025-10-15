# Windows Build Implementation Summary

## Overview

This document summarises the implementation of Windows wheel builds for evlib, enabling Windows users to install the library via `pip install evlib`.

## Key Changes

### 1. Build Configuration (Cargo.toml)

**Changed**: Default features to exclude HDF5 by default, with platform-specific overrides.

```toml
# Before
default = ["polars", "python", "tracing", "hdf5"]

# After
default = ["polars", "python", "tracing"]
default-unix = ["default", "hdf5"]  # Linux/macOS include HDF5
```

**Rationale**: HDF5 system libraries are difficult to build on Windows. By excluding from defaults, Windows builds work out of the box whilst Unix builds can explicitly add HDF5.

### 2. CI/CD Pipeline (.github/workflows/publish.yml)

**Added**: Windows build matrix entries for Python 3.10, 3.11, 3.12.

```yaml
# Windows builds (no HDF5 support)
- os: windows-latest
  platform: windows
  arch: x64
  python-version: "3.10"
# ... (3.11, 3.12)
```

**Modified**: Build steps to handle platform differences:

- Separate Unix/Windows setup steps
- Platform-specific maturin commands
- Windows builds use `--features python,polars` (no HDF5)
- Unix builds use `--features python,hdf5,polars`

### 3. Test Workflow (.github/workflows/test-windows-build.yml)

**Created**: Dedicated Windows testing workflow for validation before production.

**Features**:

- Tests all Python versions (3.10, 3.11, 3.12)
- Validates import functionality
- Verifies feature availability
- Confirms HDF5 save is unavailable
- Generates build artifacts

### 4. Local Testing Support

**Created**: `.actrc` configuration for local testing with act.

**Created**: `.github/scripts/test-windows-build.sh` helper script.

**Documentation**: `.github/workflows/README_windows_testing.md` comprehensive guide.

**Limitations**: Windows containers cannot run on macOS/Linux hosts due to Docker limitations.

### 5. Documentation Updates (CLAUDE.local.md)

**Added**:

- Platform-specific build requirements section
- Windows testing guide with act
- Platform-specific limitations section
- Windows feature flags documentation

## Feature Availability Matrix

| Feature               | Linux | macOS | Windows         |
| --------------------- | ----- | ----- | --------------- |
| Python bindings       | ✅    | ✅    | ✅              |
| Polars DataFrames     | ✅    | ✅    | ✅              |
| Event filtering       | ✅    | ✅    | ✅              |
| Data augmentation     | ✅    | ✅    | ✅              |
| Event representations | ✅    | ✅    | ✅              |
| HDF5 save             | ✅    | ✅    | ❌              |
| HDF5 read (via h5py)  | ✅    | ✅    | ✅              |
| ECF codec             | ✅    | ✅    | Via h5py plugin |
| EVT2/3 formats        | ✅    | ✅    | ✅              |
| Text formats          | ✅    | ✅    | ✅              |

## Build Commands by Platform

### Linux/macOS

```bash
maturin build --release --features python,hdf5,polars
```

### Windows

```bash
maturin build --release --features python,polars
```

## Code Changes

### Rust Source (src/lib.rs)

Existing `#[cfg(not(windows))]` guards remain:

- Line 113: `save_events_to_hdf5` excluded on Windows
- Line 155: Format module HDF5 save excluded on Windows

No changes needed - already Windows-compatible.

### HDF5 Dependencies (Cargo.toml)

HDF5 dependencies marked as optional:

```toml
hdf5-metno = { version = "0.10.1", features = ["blosc-all"], optional = true }
hdf5-metno-sys = { version = "0.10.1", optional = true }
```

Feature flag enables them on Unix builds only.

## Testing Strategy

### Local Testing (Limited)

```bash
# Syntax validation
./.github/scripts/test-windows-build.sh

# Dry run
act -W .github/workflows/test-windows-build.yml --dryrun
```

### GitHub Actions Testing (Full)

```bash
# Trigger test workflow
gh workflow run test-windows-build.yml

# Monitor results
gh run list --workflow=test-windows-build.yml
```

### Production Build

Windows wheels automatically built and published when:

1. Version bumped in Cargo.toml
2. Push to main/master branch
3. All platform builds succeed

## Known Limitations

### Windows-Specific

- No native HDF5 save functionality
- Use h5py for HDF5 operations
- ECF codec via hdf5plugin instead of native Rust

### act/Docker

- Windows containers cannot run on macOS/Linux
- Requires GitHub Actions for actual Windows testing
- Dry-run validation only on non-Windows hosts

## Rollout Plan

### Phase 1: Testing (Current)

- [x] Create test workflow
- [x] Validate build configuration
- [x] Document limitations
- [ ] Run test workflow on GitHub Actions
- [ ] Verify wheels build successfully

### Phase 2: Production

- [ ] Merge Windows builds to publish workflow
- [ ] Test publish to TestPyPI
- [ ] Verify Windows installation: `pip install evlib`
- [ ] Update README with Windows support

### Phase 3: Documentation

- [ ] Add Windows installation guide
- [ ] Document HDF5 alternatives
- [ ] Update platform compatibility matrix
- [ ] Add Windows-specific examples

## Verification Checklist

Before merging to production:

- [ ] Test workflow passes on GitHub Actions
- [ ] Windows wheels build successfully
- [ ] Wheel size reasonable (<50MB)
- [ ] Import test passes: `import evlib`
- [ ] Polars integration works
- [ ] HDF5 save gracefully unavailable
- [ ] h5py can read HDF5 files
- [ ] Documentation accurate

## References

- Issue: Windows builds excluded due to HDF5 complexity
- Solution: Optional HDF5 feature, platform-specific defaults
- Testing: act + GitHub Actions
- Build tool: maturin with PyO3
- Target: x86_64-pc-windows-msvc

## Contact

For issues with Windows builds:

- GitHub Issues: https://github.com/tallamjr/evlib/issues
- Label: `platform: windows`

---

_Generated: 2025-10-02_
_Implementation: Windows wheel builds with act testing support_
