# evlib vs RVT preprocessing benchmark (full dataset)

Full gen4_1mpx validation split: 18 sequences, raw h5 -> stacked-histogram h5.
Every per-sequence output verified bit-identical to the committed RVT reference.
Wall-clock and peak RSS measured per sequence in isolated subprocesses (single pass).

| pipeline | total time (s) | mean/seq (s) | peak RSS (GB) | total windows | diff vs ref (elems) |
| --- | --- | --- | --- | --- | --- |
| evlib polars (GPU / cudf, UVM managed) | 1846.8 | 102.60 | 15.49 | 21558 | 10 / 99339264000 (1.0e-10) |
| RVT torch (GPU) | 268.3 | 14.91 | 15.49 | 21558 | 0 (bit-identical) |
