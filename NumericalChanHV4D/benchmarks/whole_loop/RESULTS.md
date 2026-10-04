# Compiling the complete numerical contractions

The standalone [A4 report](../../paper/whole_loop_jit_4d_report.pdf) and
[LaTeX source](../../paper/whole_loop_jit_4d_report.tex) explain the numerical
contraction, recursion, correctness checks and the measurements below.

The final backend improves the earlier hybrid Numba implementation by
**1.47–8.13 times** on the five cases where both complete; the median of those
five ratios is **7.76 times**. It also completes the two 40-point cases whose
hybrid workers exceed the time budget. The default recursion and compression
schedule are retained.

This improves the new implementation, but the older compiled prefix solver
remains much faster on the larger spherical inputs. Compiling term processing
does not remove the term expansion caused by compression.

## Warm timings

Median milliseconds over three warm end-to-end calls, measured 4 October 2026.
Every row uses the same saved points and reference point for all algorithms.

| Input | Final whole loop | Earlier hybrid | Older prefix JIT | Python Chan d/3 | Python Chan d/2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sphere, seed 42, n=10 | 4.924 | 12.331 | 6.478 | 9.437 | 1.408 |
| Sphere, seed 42, n=20 | 252.106 | 2050.424 | 19.284 | 34.166 | 5.145 |
| Sphere, seed 42, n=40 | 7540.933 | budget | 55.884 | 134.478 | 18.260 |
| Sphere, seed 99, n=20 | 274.465 | 2172.687 | 21.673 | 32.210 | 5.612 |
| Sphere, seed 99, n=40 | 6929.245 | budget | 51.891 | 122.594 | 17.727 |
| Grid, seed 7, n=20 | 3.414 | 5.013 | 0.766 | 2.263 | 0.362 |
| Uniform, seed 17, n=20 | 115.610 | 897.588 | 14.112 | 18.851 | 2.529 |

`budget` means the complete worker exceeded 45 seconds, including startup,
imports, one untimed first call, and all repetitions. It is not a per-call
lower bound; no speed ratio is derived from a timeout. Python Chan d/2 remains
fastest on all seven measured inputs. These small cases do not establish an
asymptotic exponent or crossover.

All **33 completed algorithm/input combinations agree** with Chan d/2; the
largest scaled error is `7.89e-16`. Every whole-loop case exercised compiled
nopython `base_sum` and `apply_easy`; the five cases requiring compression also
exercised compiled `push_blocks`.

## Why native state matters

The first compiled version still converted Python terms into Numba containers
at every base contraction and converted compression results back into Python.
Keeping native terms throughout the geometric recursion avoids those repeated
conversions. The independent warm profile on sphere/42/n20 measures:

| Profile statistic | First compiled version | Final native-state version |
| --- | ---: | ---: |
| Profiled total time | 2.819 s | 0.265 s |
| Recorded function calls | 4,129,144 | 28,952 |
| `pack()` calls | 41 | 1 |
| `pack()` cumulative time | 2.306 s | 0.001027 s |

Profiles include profiler overhead and are separate from the timing table.
They identify where time is spent; their ratios are not benchmark speedups.
The final profile spends about 0.175 s in base contractions, 0.045 s in
compression, and 0.030 s applying easy constraints. The transfer overhead has
largely disappeared.

The amount of algebraic work is unchanged. On sphere/42/n20, hybrid and final
whole-loop versions both use 61 nodes, 10 compressions, 10,809 raw terms, 7,441
term eliminations, and a peak of 625 terms. On sphere/42/n40, the final version
uses 157 nodes, 13 compressions, 251,360 raw terms, 146,333 term eliminations,
and a peak of 7,501 terms. This growth is a remaining algorithmic limitation.

## First use

One separate run started a new process with an empty dedicated Numba cache on
sphere/42/n20. Module imports took **0.803 s**. The first computation took
**48.408 s**, including compilation and execution; the next computation in the
same process took **0.211 s**. Both returned the same value and counters, and
all three compiled entry points had nopython signatures. These are individual
observations, not medians or a measurement of compiler time alone. The warm
timing table above excludes this startup cost.

The first call therefore has a substantial cost. Persistent Numba caching is
enabled, and the cold-start script requires a new or empty cache directory so
this cost can be reproduced without deleting existing caches. Full metadata
and source hashes are in `results/cold_start.json`.

## Reproducibility

See [README.md](README.md) for commands and measurement details.
`results/timings.json` contains repetitions, counters, values, signature counts,
source hashes, versions, and validation. `results/profile_n20.json` contains the
separate warm profile. The first compiled version's measurements and profile
are retained in `results_v1`.

The benchmark used CPython 3.12.14, NumPy 2.5.3, and Numba 0.68.0 on Windows.
The final measured backend sources are preserved in commit
`1c2b5756477e4bc428b0b49acd4b7e9cc1526609`; the first compiled version is in
`7ea05d21b67de869268740096fad32a1117b0430`. Source hashes in each result file
identify the exact measured contents. The seven input records are copied into
each result directory.

A subsequent import-only fix gives the kernel one canonical package name,
`NumericalChanHV4D.compiled_terms`, so cached Numba types work with both package
and script imports. The numerical kernel is unchanged. The separate empty-cache
measurement uses this canonical import and records its current source hashes.
