# evlib vs tonic representation benchmark

Single event stream, 20,000,000 events, eTram (1280x720). Identical events to both.
Wall-clock and peak RSS per (op, backend) in isolated subprocesses (single pass).

## voxel_grid

| backend | time (s) | peak RSS (GB) | events/s |
| --- | --- | --- | --- |
| tonic (NumPy) | 1.00 | 41.48 | 20.0M |
| evlib Polars (CPU) | 0.62 | 41.48 | 32.1M |
| evlib Polars (GPU / cudf UVM) | 0.94 | 41.48 | 21.2M |

evlib Polars (CPU) is 1.61x faster than tonic for voxel_grid (0.62s vs 1.00s).

evlib Polars (GPU / cudf UVM) is 1.06x faster than tonic for voxel_grid (0.94s vs 1.00s).

## event_frame

| backend | time (s) | peak RSS (GB) | events/s |
| --- | --- | --- | --- |
| tonic (NumPy) | 1.06 | 41.48 | 18.8M |
| evlib Polars (CPU) | 0.32 | 41.48 | 61.7M |
| evlib Polars (GPU / cudf UVM) | 0.40 | 41.48 | 49.9M |

evlib Polars (CPU) is 3.29x faster than tonic for event_frame (0.32s vs 1.06s).

evlib Polars (GPU / cudf UVM) is 2.66x faster than tonic for event_frame (0.40s vs 1.06s).

