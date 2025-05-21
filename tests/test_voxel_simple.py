import evlib
import numpy as np

try:
    result = evlib.representations.events_to_voxel_grid(
        np.array([1], dtype=np.int64),
        np.array([1], dtype=np.int64),
        np.array([0.1], dtype=np.float64),
        np.array([1], dtype=np.int64),
        1,
        (10, 10),
    )
    print("Success")
    print(f"Shape: {result.shape}")
except Exception as e:
    print(f"Error: {e}")
