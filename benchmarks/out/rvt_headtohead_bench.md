# evlib vs RVT preprocessing benchmark (full dataset)

Full gen4_1mpx validation split: 18 sequences, raw h5 -> stacked-histogram h5.
Every per-sequence output verified bit-identical to the committed RVT reference.
Wall-clock and peak RSS measured per sequence in isolated subprocesses (single pass).

| pipeline | total time (s) | mean/seq (s) | peak RSS (GB) | total windows | diff vs ref (elems) |
| --- | --- | --- | --- | --- | --- |
| evlib polars (GPU / cudf, UVM managed) | 1169.2 | 64.96 | 15.48 | 21558 | 10 / 99339264000 (1.0e-10) |
| evlib rust (dense scatter-add) | 348.0 | 19.34 | 15.48 | 21558 | 10 / 99339264000 (1.0e-10) |
| RVT torch (GPU) | 269.9 | 15.00 | 15.48 | 21558 | 0 (bit-identical) |
| RVT torch (CPU, reference) | 514.9 | 28.61 | 15.48 | 21558 | 0 (bit-identical) |

evlib polars (GPU / cudf, UVM managed) is 2.27x slower than RVT torch (CPU, reference) (total 1169.2s vs 514.9s).
evlib rust (dense scatter-add) is 1.48x faster than RVT torch (CPU, reference) (total 348.0s vs 514.9s).
RVT torch (GPU) is 1.91x faster than RVT torch (CPU, reference) (total 269.9s vs 514.9s).
