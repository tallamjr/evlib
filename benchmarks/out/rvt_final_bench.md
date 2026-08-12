# evlib vs RVT preprocessing benchmark (full dataset)

Full gen4_1mpx validation split: 18 sequences, raw h5 -> stacked-histogram h5.
Every per-sequence output verified against the committed RVT reference.
Wall-clock and peak RSS measured per sequence in isolated subprocesses (single pass).

| pipeline | total time (s) | mean/seq (s) | peak RSS (GB) | total windows | diff vs ref (elems) |
| --- | --- | --- | --- | --- | --- |
| evlib CUDA (Rust scatter-add, GPU) | 283.6 | 15.76 | 15.48 | 21558 | 10 / 99339264000 (1.0e-10) |
| evlib polars (GPU / cudf, UVM managed) | 1169.2 | 64.96 | 15.48 | 21558 | 10 / 99339264000 (1.0e-10) |
| evlib rust (dense scatter-add) | 406.2 | 22.57 | 15.48 | 21558 | 10 / 99339264000 (1.0e-10) |
| RVT torch (GPU) | 286.3 | 15.91 | 15.48 | 21558 | 0 (exact match) |
| RVT torch (CPU, reference) | 534.2 | 29.68 | 15.48 | 21558 | 0 (exact match) |

evlib CUDA (Rust scatter-add, GPU) is 1.88x faster than RVT torch (CPU, reference) (total 283.6s vs 534.2s).
evlib polars (GPU / cudf, UVM managed) is 2.19x slower than RVT torch (CPU, reference) (total 1169.2s vs 534.2s).
evlib rust (dense scatter-add) is 1.32x faster than RVT torch (CPU, reference) (total 406.2s vs 534.2s).
RVT torch (GPU) is 1.87x faster than RVT torch (CPU, reference) (total 286.3s vs 534.2s).
