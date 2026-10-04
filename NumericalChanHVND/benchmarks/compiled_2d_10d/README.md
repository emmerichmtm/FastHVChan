# Compiled numerical benchmark, 2D–10D

`benchmark_dimensions.py` compares the compiled numerical solver with the original numerical Python solver and both Python Chan references. Every run keeps compression enabled. The 45 standard datasets reproduce the previous 40 inputs and add five 2D inputs. Three additional forced-compression inputs exercise the new compression path in 4D, 7D, and 10D; they are excluded from performance aggregates.

Run the repository's compiled-solver verification first to warm all supported Numba paths and verify correctness. From the repository root, using a Python environment containing NumPy and Numba:

```powershell
python NumericalChanHVND/verify_compiled.py
python NumericalChanHVND/benchmarks/compiled_2d_10d/benchmark_dimensions.py
python NumericalChanHVND/benchmarks/current_3d_10d/verify_recorded_values.py --results NumericalChanHVND/benchmarks/compiled_2d_10d/results --max-points 12
python NumericalChanHVND/benchmarks/compiled_2d_10d/summarize_results.py
```

The default is three warm repetitions and a 45-second budget per algorithm/input worker. First use is measured separately. Benchmark workers run sequentially; other machine activity is not controlled. Inputs, every retained timing and value, counters, import time, Numba signatures, source hashes, and runtime versions are archived in `results/`. Existing timing files are never overwritten; use `--out other-directory` for another run. `--only substring` selects a smaller case set. The summary generator reads saved data without rerunning algorithms.

Chan d/3 is unavailable in 2D. Its direct engine is used to retain recursion/compression counters while performing the same public-wrapper coordinate transformation with prefiltering disabled. Every available first-call and warm value is checked against Chan d/2. Timeouts retain any available partial samples and cannot be interpreted as per-call lower bounds.

The new compiled solver uses an adaptive schedule, while numerical Python retains its original local block schedule. Ratios therefore compare complete implementations, not just compilation. The adaptive schedule has no asymptotic complexity claim here pending a separate proof audit.

The independent exact-rational oracle passed on 34 recorded inputs and 543
retained values, including the forced-compression cases. See
[`results/oracle_verification.json`](results/oracle_verification.json).
The remaining 14 inputs exceed its 12-point inclusion-exclusion limit.

The separate [natural-compression probe](NATURAL_COMPRESSION.md) shows actual
compression under the unchanged default settings on 1,552 boxes. Its single
call per solver is structural evidence, excluded from the warm timing ratios.
