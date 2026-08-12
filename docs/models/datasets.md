# Datasets & Training Data

`evlib.data` is the PyTorch data-loading layer that feeds `evlib.models`' RVT detector during training: it turns preprocessed event sequences into batches of per-window representations with aligned object-detection labels. This page is a narrative on-ramp; the full API (every class, every constructor argument) is documented in the [Data Loading API Reference](../api/data.md) and this page links into it rather than repeating it.

## The shape of the pipeline

Three layers, each with its own responsibility:

- **Sources** (`PreprocessedH5Source`, `EvlibStreamSource`) read windows of event representations and labels off disk, one sequence directory at a time.
- **Datasets** (`SequenceRandomDataset`, `SequenceStreamDataset`, `SampleDataset`) wrap one or more sources and yield fixed-length window sequences as `SequenceSample`s, ready for a `torch.utils.data.DataLoader`.
- **Collate functions** (`custom_collate_random`, `custom_collate_stream`) stack a batch of `SequenceSample`s into the dict of tensors an `evlib.models.RVT` backbone consumes, keyed by `DataKey`.

`SequenceRandomDataset` is the map-style dataset for shuffled training: every item is an independent window sequence. `SequenceStreamDataset` is the iterable-style counterpart that preserves temporal order within a stream slot, so a recurrent model can carry LSTM state across consecutive batches; `is_first_sample` marks where to reset it. `SampleDataset` is the non-sequential path used for single-window classification (see the N-Caltech101 walkthrough in the [Data Loading reference](../api/data.md#classification-n-caltech101)).

All of this needs PyTorch: `pip install evlib[torch]`.

## Augmentation

`SequenceAugmentor` applies RVT-matched spatial augmentation (horizontal flip, rotation, zoom-in, zoom-out) to a `SequenceSample`, transforming the event-representation tensors and their aligned bounding boxes together. It has two entry points for the two dataset semantics: `__call__` draws fresh parameters per item (for `SequenceRandomDataset`), and `for_source(first_sample)` freezes parameters for an entire stream (for `SequenceStreamDataset`), matching RVT's own streaming augmentation behaviour. See the [Augmentation section](../api/data.md#augmentation) of the data reference for the full constructor signature and the `"stream"` sampler preset.

## Label preprocessing from raw Prophesee data

Before any of the above can run, a raw `*_bbox.npy` label file has to be turned into the on-disk layout the sources expect. `evlib.data.label_preprocess` reproduces RVT's offline preprocessing exactly (byte-identical output, checked by a local slow integration gate): `read_raw_bbox` loads and validates the raw structured array, `apply_filters` runs RVT's class/crop/size/faulty-box filter chain, `build_objframes_and_grid` selects object frames and builds the aligned event-representation timestamp grid, and `preprocess_sequence` ties all three together plus `write_preprocessed` into one end-to-end call. Full signatures, the `BBOX_DTYPE` schema, and the `EVLIB_REPR_DIR_NAME`/`RVT_REPR_DIR_NAME` naming distinction are in [Label preprocessing from raw](../api/data.md#label-preprocessing-from-raw).

## `evlib-rvt-preprocess`: the console script

Building the event-representation side of a training sequence (as opposed to the labels above) is exposed as a standalone command, installed with the package (`evlib.rvt.pipeline:main` under `[project.scripts]`):

```bash
evlib-rvt-preprocess \
  --in-h5 raw/some_sequence_td.h5 \
  --out-dir out/some_sequence \
  --dataset gen4 \
  --height 720 --width 1280 \
  --labels-npy raw/some_sequence_bbox.npy \
  --split train
```

It requires exactly one of `--grid-npy` (a precomputed `ev_repr_timestamps_us` array) or `--labels-npy`; when only `--labels-npy` is given, the window grid is derived from the labels via `evlib.data.label_preprocess` (the same filter-and-grid pipeline described above), and the labels file is also passed through so the output carries both the event representations and the matching label artifacts. Other flags: `--dataset {gen1,gen4}`, `--no-downsample` (keep full spatial resolution instead of the default 2x downsample), and `--engine` (the Polars collection engine, `auto` by default). The console script always uses the `polars` scatter-add backend; to select `rust`, `cuda`, or `metal` instead, call `evlib.rvt.process_sequence(...)` directly, as covered in the [RVT Preprocessing Pipeline](../rvt/index.md) guide.

## Where to go next

- [Data Loading API Reference](../api/data.md): full constructor signatures for every source, dataset, augmentor, and collate function, plus a minimal `PreprocessedH5Source` -> `SequenceRandomDataset` -> `DataLoader` example.
- [RVT Preprocessing Pipeline](../rvt/index.md): the four scatter-add backends behind `process_sequence`, and how to pick one.
- [Evaluation](evaluation.md): scoring a trained RVT model against a preprocessed val split.
