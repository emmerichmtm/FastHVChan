# Whole-loop compilation: 4D benchmark

This harness compares the new compiled contraction backend, the earlier hybrid
Numba backend, the compiled prefix implementation from `HV4DMagnitude`, and the
two Python Chan references. It uses exactly the seven saved inputs from the
preceding `prefix4` audit. It does not change the compression schedule.

From a Python environment with the repository's NumPy/Numba dependencies:

```console
python NumericalChanHV4D/verify_whole_loop.py
python NumericalChanHV4D/benchmarks/whole_loop/benchmark_whole_loop.py --old /path/to/HV4DMagnitude
```

`--only sphere-s42-n20` selects one case; `--out`, `--repeats`, `--timeout`,
`--new`, and `--reference` override the corresponding defaults. `--old` must
point to the root of the older repository, containing `solver.py` and
`fastkernel.py`. The default output directory is `results` beside the harness.

Each algorithm/input pair runs in a fresh process. Workers run sequentially in
a reproducibly shuffled order. One first call, including any compilation or
cache loading, is recorded separately; the reported time is the median of
three subsequent calls. Inputs use positive minimization coordinates and a
common reference point. The conversion to upper corners for the newer engines
is included in their time. No dominance prefilter is used. A fresh engine is
constructed for every call, with base size 2 where configurable.

The default 45-second limit covers an entire worker, including startup,
imports, its first call, and all repetitions. A timeout is therefore **not** a
45-second lower bound on a single evaluation. Run the correctness suite first
to populate the compilation cache; timing comparisons assume an otherwise idle
machine. The script explicitly enables JIT in each worker.

`results/timings.json` preserves each repetition, values, counters, first-call
time, Numba nopython signature counts after first use, source hashes, versions,
and comparison against Chan d/2. `timings.csv` contains the summary rows, and
`datasets.json` records the selected inputs. Source hashes identify the actual
files, including uncommitted changes; recorded Git revisions alone do not.

The benchmark checks numerical agreement rather than proving exactness or an
asymptotic running-time bound. The dedicated correctness suite covers the
compiled contractions and magnitude handling separately.

`results_v1` preserves the first compiled version, whose sources are available
in commit `7ea05d21b67de869268740096fad32a1117b0430`. It repacked Python terms for
every base contraction and unboxed compression output. Its separate n20 profile
attributed 81.8% of profiled runtime to 41 calls to `pack()`. The final version
keeps native terms throughout recursion; `results` records its independent
benchmark run. Profile timings include profiler overhead and must not be used
as benchmark timings.

To reproduce a warm profile separately from all benchmark workers:

```console
python NumericalChanHV4D/benchmarks/whole_loop/profile_whole_loop.py --old /path/to/HV4DMagnitude --out NumericalChanHV4D/benchmarks/whole_loop/results/profile_n20.json
```

To measure first use with an empty compilation cache, run a new process and
choose a new or empty directory:

```console
python NumericalChanHV4D/benchmarks/whole_loop/cold_start.py --cache-dir /path/to/new-empty-cache
```

The script sets `NUMBA_CACHE_DIR` before importing Numba. It refuses a nonempty
directory and never deletes cache files. `results/cold_start.json` records
module import time separately from one first computation and one warm
computation on sphere/42/n20. First-computation time includes compilation and
execution; it is not a measurement of compiler time alone.
