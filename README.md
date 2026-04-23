<table align="center">
  <tr>
    <td>
      <img src="./docs/evlogo.png" width="70" alt="evlib logo" />
    </td>
    <td>
      <h1 style="margin: 0;">
        <code>evlib</code>: Event Camera Data Processing Library
      </h1>
    </td>
  </tr>
</table>

<div style="text-align: center;" align="center">

[![PyPI Version](https://img.shields.io/pypi/v/evlib.svg)](https://pypi.org/project/evlib/)
[![Python Versions](https://img.shields.io/pypi/pyversions/evlib.svg)](https://pypi.org/project/evlib/)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://tallamjr.github.io/evlib/)
[![Python](https://github.com/tallamjr/evlib/actions/workflows/pytest.yml/badge.svg)](https://github.com/tallamjr/evlib/actions/workflows/pytest.yml)
[![Rust](https://github.com/tallamjr/evlib/actions/workflows/rust.yml/badge.svg)](https://github.com/tallamjr/evlib/actions/workflows/rust.yml)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-blue)](https://github.com/tallamjr/evlib)
[![License](https://img.shields.io/github/license/tallamjr/evlib)](https://github.com/tallamjr/evlib/blob/master/LICENSE.md)

</div>

An event camera processing library with a Rust backend and Python bindings,
designed for scalable data processing with real-world event camera datasets.

**Full documentation:** <https://tallamjr.github.io/evlib/>

<!-- mtoc-start -->

- [Quick Start](#quick-start)
  - [What are Event Cameras?](#what-are-event-cameras)
  - [Basic Usage](#basic-usage)
- [Installation](#installation)
- [Documentation](#documentation)
- [Examples](#examples)
- [Development](#development)
- [Community & Support](#community--support)
- [License](#license)

<!-- mtoc-end -->

## Quick Start

![xkcd](https://imgs.xkcd.com/comics/the_best_camera.png)

### What are Event Cameras?

Event cameras (also called neuromorphic or dynamic vision sensors) operate
asynchronously: each pixel independently reports brightness changes as they
occur, rather than sampling frames at a fixed rate.

Each **event** is represented as a 4-tuple:

$$e = (x, y, t, p)$$

Where:

- $x, y \in \mathbb{N}$: Pixel coordinates
- $t \in \mathbb{R}^+$: Timestamp (microsecond precision)
- $p \in \{-1, +1\}$ or $\{0, 1\}$: Polarity (brightness change direction)

An event fires when the logarithmic brightness change exceeds a threshold:

$$\log(L(x,y,t)) - \log(L(x,y,t_{\text{last}})) > \pm C$$

where $C$ is the contrast threshold. This yields microsecond temporal
resolution, 120 dB+ dynamic range, and data sparsity proportional to scene
motion.

For a deeper introduction, see the
[user guide](https://tallamjr.github.io/evlib/user-guide/loading-data/).

<p align="center">
  <img src="./wasm/wasm-sim.png" width="480" height="320" alt="event data visualisation">
</p>

### Basic Usage

```python
import evlib

# Automatic format detection — returns a Polars LazyFrame
events = evlib.load_events("data/prophesee/samples/evt2/80_balls.raw")

df = events.collect(engine="streaming")
print(f"Loaded {len(df):,} events")
print(f"Resolution: {df['x'].max()} x {df['y'].max()}")
print(f"Duration:   {df['t'].max() - df['t'].min()}")
```

Chain Polars expressions for efficient filtering and representation extraction:

```python notest
import evlib
import evlib.representations as evr
import polars as pl

events = evlib.load_events("data/prophesee/samples/hdf5/pedestrians.hdf5")

# Temporal + spatial + polarity filtering, lazily
filtered = events.filter(
    (pl.col("t").dt.total_microseconds() / 1_000_000).is_between(0.1, 0.5)
    & pl.col("x").is_between(100, 500)
    & (pl.col("polarity") == 1)
)

# Produce a stacked histogram ready for an RVT-style model
hist = evr.create_stacked_histogram(
    filtered.collect(),
    height=180, width=240,
    bins=5, window_duration_ms=50.0,
)
```

See the [representations guide](https://tallamjr.github.io/evlib/user-guide/representations/)
for voxel grids, time surfaces, and mixed density stacks.

## Installation

```bash
# Basic install
pip install evlib

# With PyTorch integration
pip install evlib[pytorch]
```

From source (requires Rust nightly and `maturin`):

```bash
git clone https://github.com/tallamjr/evlib.git
cd evlib
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
maturin develop                    # default minimal build
maturin develop --features hdf5    # opt-in HDF5 support (Linux/macOS)
```

HDF5 is opt-in on Linux/macOS and unavailable on Windows — use `h5py` directly
for HDF5 I/O on Windows. Full details and platform-specific notes live in
the [installation guide](https://tallamjr.github.io/evlib/getting-started/installation/).

## Documentation

Complete documentation is published at <https://tallamjr.github.io/evlib/>:

- [Quick Start](https://tallamjr.github.io/evlib/getting-started/quickstart/)
- [Loading Data](https://tallamjr.github.io/evlib/user-guide/loading-data/) — formats, polarity encoding, streaming
- [Event Representations](https://tallamjr.github.io/evlib/user-guide/representations/)
- [Polars Preprocessing](https://tallamjr.github.io/evlib/user-guide/polars-preprocessing/)
- [Performance Guide](https://tallamjr.github.io/evlib/getting-started/performance/) — benchmarks, memory monitoring, troubleshooting
- [API Reference](https://tallamjr.github.io/evlib/api/core/)
- [Platform Support](https://tallamjr.github.io/evlib/platform-support/windows/)

## Examples

Runnable examples live in [`examples/`](./examples):

```bash
python examples/simple_example.py
python examples/filtering_demo.py
python examples/stacked_histogram_demo.py

# Jupyter notebooks
pytest --nbmake examples/
```

Benchmarks and performance scripts live in [`benches/`](./benches).

## Development

```bash
# Tests
pytest                        # Python
cargo test                    # Rust
pytest --markdown-docs docs/  # doc examples

# Formatting / linting
black python/ tests/ examples/
cargo fmt
ruff check python/ tests/
cargo clippy -- -D warnings
```

See [CONTRIBUTING](./docs/development/contributing.md) and the
[architecture overview](./docs/development/architecture.md) for design details.

## Community & Support

- [**Issues**](https://github.com/tallamjr/evlib/issues): Report bugs and request features

![xkcd](https://imgs.xkcd.com/comics/infrastructures.png)

## License

MIT License — see [LICENSE.md](LICENSE.md) for details.
