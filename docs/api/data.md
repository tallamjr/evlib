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

**Training-ready memoisation.** `EvlibStreamSource` is now suitable for use inside a
`DataLoader` training loop. The first call to `read_windows` in each worker runs
`_ensure_time`, which reads the full raw uint32 time column once, corrects it to
non-decreasing in place, and `searchsorted`s the window-start/-end index arrays from the
grid. All three arrays (`_t_full`, `_starts`, `_ends`) are cached on the instance so
every subsequent `read_windows` call skips the file read and reuses the cached state.
Because `_t_full` can reach ~2 GB for a Gen4 sequence, the arrays are excluded from
pickle (`__getstate__` sets them to `None`); each DataLoader worker rebuilds its own copy
lazily after fork/spawn. Only the small `_grid` array and the `label_source` carry
through pickle intact.

`PreprocessedH5Source` and `EvlibStreamSource` are interchangeable behind the dataset
seam: the same dataset works with either.

## Augmentation

`SequenceAugmentor` applies RVT-matched spatial augmentation to a `SequenceSample`,
transforming both the `[C, H, W]` uint8 event-representation tensors and their aligned
yolox boxes together. The pipeline mirrors RVT's `RandomSpatialAugmentorGenX`: horizontal
flip, rotation, zoom-in, and zoom-out, in that order. Padded windows (where
`is_padded_mask[t]` is `True`) are returned unchanged. Boxes are stored in evlib's yolox
centre form `[class_id, cx, cy, w, h]`; the augmentor converts to RVT's top-left form
for each transform and converts back, so the arithmetic is byte-identical.

```python
class SequenceAugmentor:
    def __init__(
        self,
        *,
        sampler: str = "random",   # "random" (per-item) or "stream" (per-source)
        prob_hflip: float = 0.5,
        rotate_prob: float = 0.0,
        rotate_min_deg: float = 2.0,
        rotate_max_deg: float = 6.0,
        zoom_prob: Optional[float] = None,  # default 0.8 for "random", 0.5 for "stream"
        zoom_in_weight: int = 8,
        zoom_out_weight: int = 2,
        zoom_in_range: Tuple[float, float] = (1.0, 1.5),
        zoom_out_range: Tuple[float, float] = (1.0, 1.2),
        rng: Optional[np.random.Generator] = None,
    ) -> None: ...

    def __call__(self, sample: SequenceSample) -> SequenceSample: ...
    def for_source(self, first_sample: SequenceSample) -> _FrozenAugmentor: ...
```

Two entry points reflect RVT's two augmentation semantics.

`__call__` draws fresh parameters for each call, giving independent per-item
randomisation. This is the right choice for `SequenceRandomDataset`, where every item is
independent.

`for_source(first_sample)` draws parameters once from the first chunk of a source and
returns a `_FrozenAugmentor` callable. Every subsequent chunk from that source is
augmented with the same frozen parameters, matching RVT's streaming semantics where a
single random state is committed per source, not per chunk. This path requires zoom-in to
be disabled (`zoom_in_weight=0` or `sampler="stream"`) because zoom-in is label-aware and
cannot be computed without seeing the whole sequence.

The `"stream"` sampler preset sets `zoom_in_weight=0`, `zoom_prob=0.5`, and
`zoom_out_range=(1.0, 1.2)`.

### Wiring into datasets and DataModule

Pass `augmentor=` to either dataset class or to `EventDataModule`.

`SequenceRandomDataset` calls `augmentor(sample)` per `__getitem__`, so fresh parameters
are drawn for every item.

`SequenceStreamDataset` calls `augmentor.for_source(first_chunk)` on the first chunk from
each source and reuses the frozen result for all remaining chunks of that source.

`EventDataModule` passes the `augmentor` only to the train dataloader. Validation and test
dataloaders never receive it.

```python
from evlib.data import SequenceAugmentor, SequenceRandomDataset

augmentor = SequenceAugmentor(sampler="random", prob_hflip=0.5, zoom_prob=0.8)
dataset = SequenceRandomDataset([source], sequence_length=4, augmentor=augmentor)
```

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

## Label preprocessing from raw

`evlib.data.label_preprocess` reproduces RVT's offline label preprocessing pipeline,
turning a raw Prophesee `*_bbox.npy` structured array into the on-disk artifacts expected
by `PreprocessedH5Source` and `EvlibStreamSource`. Output is byte-identical to RVT's
`scripts/genx/preprocess_dataset.py` (verified by a local slow integration gate).

### Constants and types

```python
BBOX_DTYPE   # numpy structured dtype: (t, x, y, w, h, class_id, class_confidence, track_id)
LABEL_NPZ_FIELDS  # tuple of field names drawn from BBOX_DTYPE
RVT_REPR_DIR_NAME = "stacked_histogram_dt=50_nbins=10"  # on-disk repr directory name (RVT form)

class NoLabelsError(Exception): ...       # raised when all boxes are removed by filters
```

Note: `RVT_REPR_DIR_NAME` uses `=` separators to match RVT's on-disk layout. The
`PreprocessedH5Source` default `repr_name` uses the no-`=` form
`"stacked_histogram_dt50_nbins10"`; pass `repr_dir_name=RVT_REPR_DIR_NAME` when writing
with `preprocess_sequence` and the corresponding value when constructing the source.

### Core functions

```python
def read_raw_bbox(path: Union[str, Path]) -> np.ndarray:
    """Load a raw *_bbox.npy and validate its fields against BBOX_DTYPE."""

def apply_filters(
    labels: np.ndarray,
    *,
    dataset: str = "gen4",
    split: str,
    height: int,
    width: int,
    apply_psee_bbox_filter: bool = False,
    apply_faulty_bbox_filter: bool = True,
) -> np.ndarray:
    """Run RVT's filter chain in order.

    Steps: (1) gen4 class removal (pedestrian/two-wheeler/car only), (2) crop-to-FOV,
    (3) size filter (conservative w>=5 h>=5 for gen4, or Prophesee diag/side for gen1),
    (4) faulty-huge removal (train split only). Raises NoLabelsError if no boxes survive.
    """

def build_objframes_and_grid(
    filtered_labels: np.ndarray,
    *,
    dataset: str = "gen4",
    delta_t_us: int = 50000,
    align_t_us: int = 100000,
    ts_step_frame_ms: int = 100,
    ts_step_ev_repr_ms: int = 50,
    jitter_us: int = 2000,
) -> ObjframeGridResult:
    """Select object frames, build the event-repr window-end grid, and align the two.

    Returns an ObjframeGridResult dataclass with fields: labels, objframe_idx_2_label_idx,
    frame_timestamps_us, ev_repr_timestamps_us_end, objframe_idx_2_repr_idx.
    """

def write_preprocessed(
    out_dir: Union[str, Path],
    result: ObjframeGridResult,
    *,
    repr_dir_name: str = RVT_REPR_DIR_NAME,
) -> None:
    """Write the RVT directory tree for one sequence.

    Produces: labels_v2/labels.npz, labels_v2/timestamps_us.npy,
    event_representations_v2/<repr_dir_name>/objframe_idx_2_repr_idx.npy,
    event_representations_v2/<repr_dir_name>/timestamps_us.npy.
    """

def preprocess_sequence(
    bbox_path: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    dataset: str = "gen4",
    split: str = "val",
    height: int = 720,
    width: int = 1280,
    repr_dir_name: str = RVT_REPR_DIR_NAME,
) -> ObjframeGridResult:
    """End-to-end pipeline: read -> filter -> grid -> write.

    Ties read_raw_bbox -> apply_filters -> build_objframes_and_grid -> write_preprocessed.
    Supported datasets: "gen4" (1280x720, default) and "gen1" (304x240).
    """
```

### Typical usage

```python
from evlib.data import preprocess_sequence, RVT_REPR_DIR_NAME, PreprocessedH5Source

result = preprocess_sequence(
    "data/gen4/train/seq0/seq0_bbox.npy",
    out_dir="data/gen4/train/seq0/",
    dataset="gen4",
    split="train",
    height=720,
    width=1280,
)

# The processed sequence is now readable by PreprocessedH5Source:
source = PreprocessedH5Source(
    "data/gen4/train/seq0/",
    repr_name=RVT_REPR_DIR_NAME,
)
```

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
