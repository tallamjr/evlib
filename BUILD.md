# Building evlib

This document provides instructions for building and developing the `evlib` package.

## Prerequisites

- Rust (latest stable version)
- Python 3.10 or later
- uv (recommended) or pip with venv

## Development Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/evlib.git
cd evlib
```

2. Set up a virtual environment:

```bash
uv venv --python <python-version> # 3.12 recommended
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install pip
```

3. Install the package in development mode:

Using uv (recommended):

```bash
uv pip install -e ".[dev]"
```

Or using pip:

```bash
pip install -e ".[dev]"
```

This will install the package in development mode along with development
dependencies like pytest.

Note: Using uv will automatically create/update a `uv.lock` file which locks all
dependency versions for reproducible installs. The uv.lock file should be
committed to version control to ensure consistent environments across all
development machines.

## Building with Cargo

For direct cargo builds, use the provided build script which handles Python integration:

```bash
# Make sure the script is executable
chmod +x build.sh

# Run the build script (uses activated virtual environment)
./build.sh
```

This script automatically:
1. Detects your Python installation from the active virtual environment
2. Finds the correct Python library and include paths
3. Sets up the necessary environment variables for building with PyO3
4. Builds the project with the python feature enabled

For development builds without the release optimization:

```bash
./build.sh --debug
```

## Building with Maturin

Alternatively, [Maturin](https://github.com/PyO3/maturin) can be used to build the Rust extension module.

### Development Build

For rapid development and testing:

```bash
maturin develop
```

This builds the Rust extension module and installs it in your current virtual
environment.

For a release build:

```bash
maturin develop --release
```

### Building a Wheel

To build a wheel for distribution:

```bash
maturin build --release
```

The built wheel will be in the `target/wheels/` directory.

## Running Tests

After building the extension module, you can run the tests:

```bash
pytest
```

Or with more verbosity:

```bash
pytest -v
```

To run a specific test:

```bash
pytest tests/test_evlib.py::test_name
```

## Using uv for Development Workflows

[uv](https://github.com/astral-sh/uv) provides several advantages for Python development:

1. **Faster Package Installation**: uv is significantly faster than pip for
   installing packages.

2. **Reproducible Environments**: The uv.lock file ensures that all developers
   use the exact same package versions.

3. **Common Commands**:

   ```bash
   # Install all dependencies
   uv pip install -e ".[all]"

   # Update a specific package
   uv pip install --upgrade package-name

   # Add a new development dependency
   uv pip install new-package
   ```

4. **Managing Virtual Environments**:

   ```bash
   # Create a new virtual environment
   uv venv

   # Activate the virtual environment
   source .venv/bin/activate
   ```

## Packaging

To build a wheel package for distribution:

```bash
maturin build --release
```

This will create a wheel in the `target/wheels/` directory that can be installed
with pip or uv.

For a universal2 wheel on macOS:

```bash
maturin build --release --universal2
```

For manylinux wheels:

```bash
maturin build --release --manylinux=2014
```

## Publishing to PyPI

First build the wheels and then use twine to upload:

```bash
maturin build --release
twine upload target/wheels/*
```

You can also install the built wheels locally using uv:

```bash
uv pip install target/wheels/*.whl
```
