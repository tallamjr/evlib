# Windows Build Testing Guide

## Overview

This document explains how to test Windows wheel builds for evlib using GitHub Actions locally with `act`.

## Prerequisites

- **act**: GitHub Actions runner for local testing
  ```bash
  # macOS
  brew install act

  # Linux
  curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
  ```

- **Docker**: Required for running containers
  ```bash
  # macOS: Install Docker Desktop
  # Linux: Install Docker Engine
  ```

## Important Limitations

### Windows Container Compatibility

**Windows containers cannot run natively on macOS or Linux hosts.**

This is a Docker limitation, not an `act` limitation. Windows containers require:
1. Windows host OS
2. Windows Server base images
3. Hyper-V or process isolation

### Recommended Testing Approaches

1. **Syntax validation**: Use `act --dryrun` to validate workflow syntax
2. **Linux substitute**: Test build process with Linux containers
3. **GitHub Actions**: Push to GitHub for actual Windows testing
4. **Local Windows**: Use a Windows machine with Docker Desktop

## Testing Commands

### Quick Start

```bash
# Run the test script (provides guidance and validation)
./.github/scripts/test-windows-build.sh
```

### Manual Testing

```bash
# List jobs in the Windows test workflow
act -l -W .github/workflows/test-windows-build.yml

# Dry run (validate syntax without execution)
act -W .github/workflows/test-windows-build.yml --dryrun

# Test specific job
act -W .github/workflows/test-windows-build.yml -j test-windows-wheels --dryrun

# Test with specific Python version
act -W .github/workflows/test-windows-build.yml \
    --matrix python-version:3.12 \
    --dryrun
```

### GitHub Actions Testing

```bash
# Trigger workflow manually
gh workflow run test-windows-build.yml

# Check workflow status
gh run list --workflow=test-windows-build.yml

# View logs from latest run
gh run view --log
```

## Windows Build Configuration

### Features Included
- ✅ Python bindings
- ✅ Polars DataFrame support
- ✅ Event filtering
- ✅ Data augmentation
- ✅ Event representations

### Features Excluded
- ❌ HDF5 save functionality (read-only via h5py)
- ❌ Native ECF codec (use h5py with hdf5plugin)

### Build Command
```bash
maturin build --release --features python,polars,arrow --interpreter python
```

### Cargo Features
```toml
# Default features (cross-platform)
default = ["polars", "python", "arrow"]

# Note: HDF5 and tracing are Unix-only dependencies
# (not feature flags, but platform-specific dependencies)
[target.'cfg(unix)'.dependencies]
hdf5-metno = { version = "0.10.1", features = ["blosc-all"] }
tracing = { version = "0.1.40" }
```

## Workflow Files

### `.github/workflows/test-windows-build.yml`
Dedicated Windows testing workflow for validating builds before adding to publish pipeline.

**Purpose**:
- Test Windows wheel builds
- Verify import functionality
- Validate feature availability
- Generate build artifacts

**Triggers**:
- Manual dispatch: `gh workflow run test-windows-build.yml`
- Pull requests affecting source code

### `.github/workflows/publish.yml`
Production workflow that builds and publishes wheels for all platforms.

**Windows Build Matrix**:
- Python 3.10, 3.11, 3.12
- x64 architecture
- MSVC toolchain

## Troubleshooting

### Container Architecture Warnings

If you see:
```
⚠ You are using Apple M-series chip and you have not specified container architecture
```

**Solution**: Add `--container-architecture linux/amd64` for Linux substitutes:
```bash
act --container-architecture linux/amd64 -W .github/workflows/test-windows-build.yml --dryrun
```

### Docker Not Running

If you see:
```
Error: Docker is not running
```

**Solution**: Start Docker Desktop and try again.

### Cannot Connect to Docker Daemon

If you see:
```
Cannot connect to the Docker daemon
```

**Solution**:
1. Verify Docker is running: `docker info`
2. Check Docker socket permissions
3. Restart Docker Desktop

## Platform-Specific Notes

### macOS (M-series)
- Windows containers not supported
- Use dry-run validation
- Test on GitHub Actions for actual Windows builds

### macOS (Intel)
- Windows containers not supported
- Use dry-run validation
- Test on GitHub Actions for actual Windows builds

### Linux
- Windows containers require Windows kernel
- Use dry-run validation
- Test on GitHub Actions for actual Windows builds

### Windows
- Can run Windows containers with Docker Desktop
- Requires Hyper-V or WSL 2 backend
- Full `act` support available

## CI/CD Integration

### Build Process
1. Checkout code
2. Setup Python and Rust
3. Install maturin
4. Build wheel with `--features python,polars`
5. Test imports and functionality
6. Upload wheel artifacts

### Artifact Storage
Windows wheels are stored as:
```
wheels-windows-x64-py3.10.whl
wheels-windows-x64-py3.11.whl
wheels-windows-x64-py3.12.whl
```

### Publishing
Windows wheels are automatically included in PyPI publish when:
1. Version number changes in Cargo.toml
2. All build jobs succeed (macOS, Linux, Windows)
3. Wheels pass validation tests

## Validation Checklist

- [ ] Workflow syntax is valid (`act --dryrun`)
- [ ] Python versions match project requirements (3.10, 3.11, 3.12)
- [ ] Features exclude HDF5 (`--features python,polars`)
- [ ] Build target is MSVC (`x86_64-pc-windows-msvc`)
- [ ] Import tests pass
- [ ] HDF5 save is correctly unavailable
- [ ] Wheel naming follows convention

## References

- [act documentation](https://github.com/nektos/act)
- [maturin documentation](https://www.maturin.rs/)
- [GitHub Actions Windows runners](https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners#supported-runners-and-hardware-resources)
- [PyO3 cross-compilation](https://pyo3.rs/v0.25.0/building_and_distribution)
