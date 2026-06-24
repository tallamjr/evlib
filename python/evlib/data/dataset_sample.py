"""Sample-level dataset for classification (one representation -> one label)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class SampleDataset(Dataset):
    def __init__(self, samples: Sequence[Path], labels: Sequence[int]) -> None:
        if len(samples) != len(labels):
            raise ValueError(
                f"samples ({len(samples)}) and labels ({len(labels)}) length mismatch"
            )
        self.samples: List[Path] = [Path(p) for p in samples]
        self.labels: List[int] = [int(v) for v in labels]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        arr = np.load(self.samples[i])
        return torch.from_numpy(np.ascontiguousarray(arr)), self.labels[i]
