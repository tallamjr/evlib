# evlib vs RVT preprocessing benchmark

Full Gen4 validation sequence, raw h5 to stacked-histogram representation h5.
All outputs verified bit-identical to the committed reference (1198 windows,
shape (1198, 20, 360, 640) uint8).

| pipeline | min (s) | median (s) | max (s) | peak RSS (GB) |
| --- | --- | --- | --- | --- |
| evlib rust (dense scatter-add, raw h5) | 15.69 | 15.72 | 15.87 | 6.34 |
| evlib streaming (h5 to parquet + build) | 58.33 | 58.45 | 60.28 | 4.29 |
| evlib build only (from cached parquet) | 27.99 | 28.21 | 28.24 | 3.28 |
| RVT torch (reference) | 22.24 | 23.36 | 23.71 | 6.40 |

evlib rust backend is 1.49x faster than RVT torch reference (15.7s vs 23.4s median).
evlib rust backend uses 1.01x less peak memory than RVT torch reference (6.34 GB vs 6.40 GB).
evlib full pipeline is 2.50x slower than RVT torch reference (58.5s vs 23.4s median).
evlib build-only (cached parquet) is 1.21x slower than RVT torch reference (28.2s vs 23.4s median).
evlib full pipeline uses 1.49x less peak memory than RVT torch reference (4.29 GB vs 6.40 GB).
