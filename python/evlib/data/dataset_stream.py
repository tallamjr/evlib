"""Streaming dataset: persistent per-batch-slot ordered streams (plain IterableDataset)."""

from __future__ import annotations

from typing import Iterator, List

import torch
from torch.utils.data import IterableDataset, get_worker_info

from evlib.data.sequence import SequenceSample
from evlib.data.sources import ReprSource


def _stream_one_source(src: ReprSource, L: int) -> Iterator[SequenceSample]:
    n = src.window_count()
    first = True
    for start in range(0, n, L):
        hi = min(start + L, n)
        ev, labels = src.read_windows(start, hi)
        real = hi - start
        pad = L - real
        is_padded = [False] * real + [True] * pad
        if pad:
            zero = torch.zeros_like(ev[0])
            ev = ev + [zero.clone() for _ in range(pad)]
            labels = labels + [None] * pad
        yield SequenceSample(
            ev, labels, is_first_sample=first, is_padded_mask=is_padded
        )
        first = False


class SequenceStreamDataset(IterableDataset):
    def __init__(
        self, sources: List[ReprSource], sequence_length: int, batch_size: int
    ) -> None:
        self.sources = sources
        self.L = sequence_length
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[List[SequenceSample]]:
        info = get_worker_info()
        sources = self.sources
        if info is not None:
            sources = [
                s
                for idx, s in enumerate(self.sources)
                if idx % info.num_workers == info.id
            ]
        # assign sources round-robin to batch_size slots, concatenate each slot's streams
        slots: List[List[ReprSource]] = [[] for _ in range(self.batch_size)]
        for idx, s in enumerate(sources):
            slots[idx % self.batch_size].append(s)

        def slot_iter(slot_sources):
            for s in slot_sources:
                yield from _stream_one_source(s, self.L)

        iters = [slot_iter(s) for s in slots]
        active = True
        while active:
            batch_slot: List[SequenceSample] = []
            active = False
            for it in iters:
                try:
                    batch_slot.append(next(it))
                    active = True
                except StopIteration:
                    # pad an exhausted slot with an all-padded chunk to keep batch width
                    if batch_slot:
                        ref = batch_slot[0]
                        zeros = [torch.zeros_like(t) for t in ref.ev_repr]
                        batch_slot.append(
                            SequenceSample(
                                zeros, [None] * len(zeros), False, [True] * len(zeros)
                            )
                        )
            if active:
                yield batch_slot
