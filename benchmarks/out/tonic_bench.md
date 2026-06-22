# evlib vs tonic representation benchmark

Single event stream, 20,000,000 events, eTram (1280x720). Same events to both.
Wall-clock and peak RSS per (op, backend) in isolated subprocesses (single pass).

## voxel_grid

| backend | time (s) | peak RSS (GB) | events/s |
| --- | --- | --- | --- |
| tonic (NumPy) | 0.84 | 41.48 | 23.8M |
| evlib Polars (CPU) | 0.62 | 41.48 | 32.1M |
| evlib Polars (GPU / cudf UVM) | 0.91 | 41.48 | 22.0M |

evlib Polars (CPU) is 1.35x faster than tonic for voxel_grid (0.62s vs 0.84s).

evlib Polars (GPU / cudf UVM) is 1.08x slower than tonic for voxel_grid (0.91s vs 0.84s).

## event_frame

| backend | time (s) | peak RSS (GB) | events/s |
| --- | --- | --- | --- |
| tonic (NumPy) | 0.92 | 41.48 | 21.8M |
| evlib Polars (CPU) | 0.32 | 41.48 | 61.6M |
| evlib Polars (GPU / cudf UVM) | 0.39 | 41.48 | 51.6M |

evlib Polars (CPU) is 2.83x faster than tonic for event_frame (0.32s vs 0.92s).

evlib Polars (GPU / cudf UVM) is 2.36x faster than tonic for event_frame (0.39s vs 0.92s).

## time_surface

| backend | time (s) | peak RSS (GB) | events/s |
| --- | --- | --- | --- |
| tonic (NumPy) | 0.55 | 41.48 | 36.2M |
| evlib Polars (CPU) | 0.26 | 41.48 | 76.5M |
| evlib Polars (GPU / cudf UVM) | 0.35 | 41.48 | 57.8M |

evlib Polars (CPU) is 2.12x faster than tonic for time_surface (0.26s vs 0.55s).

evlib Polars (GPU / cudf UVM) is 1.60x faster than tonic for time_surface (0.35s vs 0.55s).
