# Data Loading API Reference

`evlib.data` is the PyTorch data-loading layer for event-vision training. It turns
preprocessed RVT sequences (or raw event h5 files) into batches of per-window event
representations with aligned object-detection labels, ready to feed a recurrent
backbone such as `evlib.models` RVT.

The package builds on three concepts:

- **Sources** read windows of event representations (and labels) from disk.
- **Datasets** wrap one or more sources and yield fixed-length window sequences.
- **Collate functions** stack a list of sequences into a model-ready batch dict.

All of this requires PyTorch (`pip install evlib[torch]`).

## The sequence contract

Every dataset item is a `SequenceSample`: an ordered list of `T` window tensors with
aligned per-window labels.

```python
from evlib.data import SequenceSample, DataKey

# SequenceSample fields:
#   ev_repr:        list[Tensor], each [C, H, W] (C == 2 * nbins), uint8 or float32
#   labels:         list[Optional[Tensor]], each [num_boxes, 5] (yolox) or None
#   is_first_sample: bool, marks the start of a continuous stream slot
#   is_padded_mask:  list[bool], True where a window is zero-padding

# DataKey holds the batch-dict keys the collate functions emit:
print(DataKey.EV_REPR, DataKey.OBJLABELS_SEQ, DataKey.IS_FIRST_SAMPLE, DataKey.IS_PADDED_MASK)
```

## Sources

A source implements the `ReprSource` protocol: `window_count()` and
`read_windows(lo, hi) -> (list[Tensor], list[Optional[Tensor]])`. Two concrete
sources ship with the package.

### `PreprocessedH5Source`

Reads one preprocessed RVT sequence directory. It expects the standard RVT layout
(`event_representations_v2/<repr_name>/event_representations_ds2_nearest.h5` plus the
`objframe_idx_2_repr_idx.npy` and `labels_v2/labels.npz` companions). The h5 handle is
opened lazily so the source is picklable and fork/spawn safe across DataLoader workers.

```python
def PreprocessedH5Source(
    seq_dir,
    repr_name="stacked_histogram_dt50_nbins10",
    downsample_by_2=True,
)
```

### `EvlibStreamSource`

Builds dense `[C, H, W]` windows on the fly from a raw event h5, reusing the
`evlib.rvt` window assignment and the Rust `stacked_histogram_dense` kernel instead of
reading a precomputed representation h5. Window-end timestamps and labels are taken
from the same processed sequence directory. Pass `gpu="cuda"` or `gpu="metal"` to
densify on the GPU.

```python
def EvlibStreamSource(
    raw_h5,
    seq_dir,
    repr_name="stacked_histogram_dt50_nbins10",
    downsample_by_2=True,
    nbins=10,
    count_cutoff=10,
    delta_t_us=50_000,
    height=720,
    width=1280,
    gpu=None,
)
```

`PreprocessedH5Source` and `EvlibStreamSource` are interchangeable behind the dataset
seam: the same dataset works with either.

## Datasets

All three are `torch.utils.data.Dataset` subclasses that yield `SequenceSample`s.

### `SequenceRandomDataset`

Map-style dataset where each item is one independent fixed-length window sequence.
Each source is tiled into non-overlapping windows of `sequence_length`; the final
sequence is zero-padded (tracked via `is_padded_mask`). Shuffle freely with a
`DataLoader`, since items carry no cross-item state.

```python
def SequenceRandomDataset(sources, sequence_length)
```

### `SequenceStreamDataset`

Iterable-style dataset that preserves temporal order within each stream slot, so a
recurrent model can carry LSTM state across consecutive batches. Use `is_first_sample`
to reset state at slot boundaries.

### `SampleDataset`

Map-style dataset for the single-sample (non-sequential) classification path: each
item is one window with its label, rather than a sequence.

## Collate functions

`custom_collate_random` stacks a list of `SequenceSample`s into a batch dict keyed by
`DataKey`. The event representation is returned as a **list of per-timestep tensors**,
each `[B, C, H, W]`, which is exactly the layout the RVT backbone consumes one timestep
at a time.

```python
# returned batch dict:
#   DataKey.EV_REPR:        list[Tensor], length T, each [B, C, H, W]
#   DataKey.OBJLABELS_SEQ:  list[list[Optional[Tensor]]], T x B labels
#   DataKey.IS_FIRST_SAMPLE: list[bool], length B
#   DataKey.IS_PADDED_MASK:  Tensor [T, B] bool
```

`custom_collate_stream` is the streaming-step collate: it takes a list of `batch_size`
slot-aligned `SequenceSample`s and produces the same dict shape.

## Lightning DataModule (optional)

`EventDataModule` wraps the datasets and collate functions in a
`pytorch_lightning.LightningDataModule`. It is exported only when PyTorch Lightning is
installed; if Lightning is missing, the import is skipped and the name is absent from
`evlib.data`.

## Minimal example

`PreprocessedH5Source` to `SequenceRandomDataset` to a `DataLoader` with the random
collate function. The per-timestep `DataKey.EV_REPR` tensor is `[B, C, H, W]` with
`C == 2 * nbins`.

```python
from torch.utils.data import DataLoader
from evlib.data import (
    PreprocessedH5Source,
    SequenceRandomDataset,
    custom_collate_random,
    DataKey,
)

source = PreprocessedH5Source("tests/data_fixtures/mini_seq")
dataset = SequenceRandomDataset([source], sequence_length=2)
loader = DataLoader(dataset, batch_size=1, collate_fn=custom_collate_random)

batch = next(iter(loader))
per_timestep = batch[DataKey.EV_REPR]  # list of [B, C, H, W] tensors, one per window
print(f"timesteps: {len(per_timestep)}")
print(f"first step [B, C, H, W]: {tuple(per_timestep[0].shape)}")
```

Feeding a batch into the model is covered by the RVT backbone forward pass
(`evlib.models.rvt_backbone.RVTBackbone.forward(x[B, C, H, W], previous_states)`),
which consumes one per-timestep tensor at a time and returns per-stage features plus
per-stage LSTM states. The tracked `tests/data_fixtures/mini_seq` fixture is 8x12
spatial (too small for the backbone's stride-32 downsampling); real training data uses
full-resolution windows. See the [processing API](processing.md) for the model layer
and `python/evlib/models/` for backbone configuration.
