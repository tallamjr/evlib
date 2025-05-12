# Building evlib

This document provides instructions for building and developing the `evlib` package.

## Prerequisites

- Rust (latest stable version)
- Python 3.8 or later
- Poetry or pip with venv

## Development Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/evlib.git
cd evlib
```

2. Set up a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the package in development mode:

```bash
pip install -e ".[dev]"
```

This will install the package in development mode along with development dependencies like pytest.

## Building with Maturin

[Maturin](https://github.com/PyO3/maturin) is used to build the Rust extension module.

### Development Build

For rapid development and testing:

```bash
maturin develop
```

This builds the Rust extension module and installs it in your current virtual environment.

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

## Packaging

To build a wheel package for distribution:

```bash
maturin build --release
```

This will create a wheel in the `target/wheels/` directory that can be installed with pip.

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