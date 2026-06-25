"""PyTorch data loading for event-vision training (sequences + samples)."""

from evlib.data.augment import SequenceAugmentor
from evlib.data.collate import custom_collate_random, custom_collate_stream
from evlib.data.dataset_random import SequenceRandomDataset
from evlib.data.dataset_sample import SampleDataset
from evlib.data.dataset_stream import SequenceStreamDataset
from evlib.data.label_preprocess import (
    BBOX_DTYPE,
    LABEL_NPZ_FIELDS,
    NoLabelsError,
    apply_filters,
    read_raw_bbox,
)
from evlib.data.labels import LABEL_FIELDS, boxes_to_yolox
from evlib.data.sequence import DataKey, SequenceSample
from evlib.data.sources import EvlibStreamSource, PreprocessedH5Source, ReprSource

__all__ = [
    "SequenceSample",
    "SequenceAugmentor",
    "DataKey",
    "boxes_to_yolox",
    "LABEL_FIELDS",
    "BBOX_DTYPE",
    "LABEL_NPZ_FIELDS",
    "NoLabelsError",
    "read_raw_bbox",
    "apply_filters",
    "ReprSource",
    "PreprocessedH5Source",
    "EvlibStreamSource",
    "SequenceRandomDataset",
    "SequenceStreamDataset",
    "SampleDataset",
    "custom_collate_random",
    "custom_collate_stream",
]

try:
    from evlib.data.datamodule import EventDataModule  # noqa: F401

    __all__.append("EventDataModule")
except ImportError:
    pass
