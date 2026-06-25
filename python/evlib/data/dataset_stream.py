"""Streaming dataset: persistent per-batch-slot ordered streams (plain IterableDataset)."""

from __future__ import annotations

from typing import Callable, Iterator, List, Optional

import torch
from torch.utils.data import IterableDataset, get_worker_info

from evlib.data.sequence import SequenceSample
from evlib.data.sources import ReprSource


def _stream_one_source(
    src: ReprSource,
    L: int,
    augmentor: Optional[Callable] = None,
) -> Iterator[SequenceSample]:
    """Yield this source's ordered chunks, optionally augmenting them.

    RVT stream semantics: augmentation params are drawn ONCE per source and the
    SAME params apply to every chunk yielded from it. The frozen applier is built
    from the first chunk via ``augmentor.for_source(first_chunk)`` and reused for
    all subsequent chunks of this source. The trailing padded chunk (when the
    source length is not a multiple of L) is real data plus padding, so it is
    augmented like any other; only the all-padded placeholder samples that
    ``__iter__`` synthesises for exhausted slots are left untouched.
    """
    n = src.window_count()
    first = True
    frozen = None
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
        sample = SequenceSample(
            ev, labels, is_first_sample=first, is_padded_mask=is_padded
        )
        if augmentor is not None:
            if frozen is None:
                # Draw the single per-source param state from the first chunk.
                frozen = augmentor.for_source(sample)
            sample = frozen(sample)
        yield sample
        first = False


class SequenceStreamDataset(IterableDataset):
    def __init__(
        self,
        sources: List[ReprSource],
        sequence_length: int,
        batch_size: int,
        augmentor: Optional[Callable] = None,
    ) -> None:
        self.sources = sources
        self.L = sequence_length
        self.batch_size = batch_size
        # Optional opt-in augmentor: params are drawn ONCE per source and reused
        # across all its chunks (RVT stream semantics); see _stream_one_source.
        self.augmentor = augmentor

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
                yield from _stream_one_source(s, self.L, self.augmentor)

        num_slots = self.batch_size
        iters = [slot_iter(s) for s in slots]
        # A pad template: the per-window zero tensors (shaped like a real
        # ev_repr) and their count. Captured from the first real sample of any
        # slot so trailing/empty slots can also be padded from step zero.
        pad_zeros: List[torch.Tensor] | None = None

        while True:
            batch_slot: List[SequenceSample | None] = [None] * num_slots
            any_real = False
            # Pass 1: pull one real sample per slot, fixed by slot index.
            for j in range(num_slots):
                try:
                    sample = next(iters[j])
                except StopIteration:
                    continue
                batch_slot[j] = sample
                any_real = True
                if pad_zeros is None:
                    pad_zeros = [torch.zeros_like(t) for t in sample.ev_repr]
            if not any_real:
                # Every stream is exhausted; nothing more to yield.
                return
            # Pass 2: fill exhausted/empty slots in place with a padded sample.
            if pad_zeros is None:
                raise RuntimeError("pad template unset despite a real sample")
            n = len(pad_zeros)
            for j in range(num_slots):
                if batch_slot[j] is None:
                    batch_slot[j] = SequenceSample(
                        [z.clone() for z in pad_zeros],
                        [None] * n,
                        is_first_sample=False,
                        is_padded_mask=[True] * n,
                    )
            yield batch_slot
