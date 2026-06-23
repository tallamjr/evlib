"""Map-style dataset: each item is one independent fixed-length window sequence."""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from evlib.data.sequence import SequenceSample
from evlib.data.sources import ReprSource


class SequenceRandomDataset(Dataset):
    def __init__(self, sources: List[ReprSource], sequence_length: int) -> None:
        if sequence_length < 1:
            raise ValueError("sequence_length must be >= 1")
        self.sources = sources
        self.L = sequence_length
        # index map: (source_idx, start_window) for each sequence
        self._index: List[Tuple[int, int]] = []
        for si, src in enumerate(sources):
            n = src.window_count()
            for start in range(0, n, self.L):
                self._index.append((si, start))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> SequenceSample:
        si, start = self._index[i]
        src = self.sources[si]
        n = src.window_count()
        hi = min(start + self.L, n)
        ev, labels = src.read_windows(start, hi)
        real = hi - start
        pad = self.L - real
        is_padded = [False] * real + [True] * pad
        if pad:
            zero = torch.zeros_like(ev[0])
            ev = ev + [zero.clone() for _ in range(pad)]
            labels = labels + [None] * pad
        return SequenceSample(
            ev, labels, is_first_sample=True, is_padded_mask=is_padded
        )
