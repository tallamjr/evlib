# evlib vs RVT preprocessing benchmark

Full Gen4 validation sequence, raw h5 to stacked-histogram representation h5.
All outputs verified bit-identical to the committed reference (1198 windows,
shape (1198, 20, 360, 640) uint8).

| pipeline | min (s) | median (s) | max (s) | peak RSS (GB) |
| --- | --- | --- | --- | --- |
| evlib streaming (h5 to parquet + build) | 57.91 | 58.55 | 58.55 | 4.29 |
| evlib build only (from cached parquet) | 28.67 | 28.69 | 28.77 | 3.37 |
| RVT torch (reference) | 22.06 | 22.15 | 22.16 | 6.40 |

evlib full pipeline is 2.64x slower than RVT torch reference (58.5s vs 22.2s median).
evlib build-only (cached parquet) is 1.30x slower than RVT torch reference (28.7s vs 22.2s median).
evlib full pipeline uses 1.49x less peak memory than RVT torch reference (4.29 GB vs 6.40 GB).
