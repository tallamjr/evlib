"""Generate a tiny classification fixture: 4 [C,H,W] arrays + integer labels."""

from pathlib import Path
import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parent / "mini_samples"
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    labels = [0, 1, 0, 2]
    for i, _ in enumerate(labels):
        np.save(
            root / f"sample_{i}.npy",
            rng.integers(0, 4, size=(20, 8, 12), dtype=np.uint8),
        )
    np.save(root / "labels.npy", np.array(labels, dtype=np.int64))
    print(f"wrote sample fixture to {root}")


if __name__ == "__main__":
    main()
