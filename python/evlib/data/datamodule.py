"""Optional PyTorch Lightning DataModule wrapper (only used if lightning is installed)."""

from __future__ import annotations

from typing import Callable, List, Optional

from torch.utils.data import DataLoader

from evlib.data.collate import custom_collate_random, custom_collate_stream
from evlib.data.dataset_random import SequenceRandomDataset
from evlib.data.dataset_stream import SequenceStreamDataset
from evlib.data.sources import ReprSource

try:
    import pytorch_lightning as pl

    _HAS_LIGHTNING = True
except ImportError:  # lightning is optional; the torch datasets work without it
    _HAS_LIGHTNING = False


if _HAS_LIGHTNING:

    class EventDataModule(pl.LightningDataModule):
        def __init__(
            self,
            train_sources: List[ReprSource],
            val_sources: List[ReprSource],
            sequence_length: int,
            batch_size: int,
            num_workers: int = 4,
            sampling: str = "random",
            augmentor: Optional[Callable] = None,
        ) -> None:
            super().__init__()
            self.train_sources = train_sources
            self.val_sources = val_sources
            self.L = sequence_length
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.sampling = sampling
            # Augmentation is a TRAIN-only concern (RVT builds the augmentor for
            # the train split alone); val/test datasets never receive it.
            self.augmentor = augmentor

        def train_dataloader(self) -> DataLoader:
            if self.sampling == "stream":
                ds = SequenceStreamDataset(
                    self.train_sources,
                    self.L,
                    self.batch_size,
                    augmentor=self.augmentor,
                )
                return DataLoader(
                    ds,
                    batch_size=None,
                    num_workers=self.num_workers,
                    collate_fn=custom_collate_stream,
                )
            ds = SequenceRandomDataset(
                self.train_sources, self.L, augmentor=self.augmentor
            )
            return DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=custom_collate_random,
            )

        def val_dataloader(self) -> DataLoader:
            ds = SequenceRandomDataset(self.val_sources, self.L)
            return DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=custom_collate_random,
            )

        def test_dataloader(self) -> DataLoader:
            return self.val_dataloader()
