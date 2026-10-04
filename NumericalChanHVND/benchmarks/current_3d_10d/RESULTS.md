# Current numerical hypervolume benchmarks, 3D–10D

This run records 40 inputs and 165 algorithm/input workers: 162 ok, 3 timeout. Validation errors recorded by the runner: **0**.

The benchmark selects a compiled sweep for its current Numba row in 3D, the new whole-contraction backend with persistent native term states in 4D, and the existing hybrid backend in 5D–10D. Geometric recursion remains Python in 4D–10D. This benchmark does not extend the whole-contraction implementation to higher dimensions.

## Key findings

The 3D compiled sweep has a median Python-time/Numba-time ratio of 1.20x over 4 completed ordinary-HV pairs; Numba is faster in 3 of those pairs.

The 4D whole-contraction backend has a median previous-hybrid/current time ratio of 5.12x over 4 completed ordinary-HV pairs (range 1.86–9.08x).

For the higher-dimensional hybrid, median Numba-time/Python-time ratios within each dimension are 5D: 1.78x (4 pairs); 6D: 1.84x (4 pairs); 7D: 1.94x (4 pairs); 8D: 2.10x (3 pairs); 9D: 1.51x (3 pairs); 10D: 2.52x (3 pairs). Here ratios above one mean the hybrid is slower than numerical Python.

The Python Chan d/2 reference has the lowest completed median on 28 of 28 ordinary-HV cases in 4D-10D.

These completed-pair summaries have selection bias: the slowest cases that time out are excluded. They do not describe the cost or success rate of the full workload; the status counts must be read alongside them.

## Method

One fresh worker runs at a time. Each implementation receives one untimed first call followed by 3 timed calls; tables report their median in milliseconds. The timed region includes input conversion and rank preprocessing, but excludes module imports. Every call constructs a fresh solver. The default compression schedule is retained and the Chan dominance prefilter is disabled. No fastmath or parallel numerical execution is used.

The 45.0-second limit applies to the complete worker, including startup, imports, the first call, and all timed repetitions. A timeout therefore gives no per-call runtime lower bound. No ratio is calculated from a timeout.

All implementations receive the same saved box union. Numerical solvers receive nonnegative anchored extents; the two Python Chan references receive the equivalent minimization representation. For magnitude the references receive the affine HV transformation `1 + extent/2`, with the transformation included in timing. The sphere inputs have mutually nondominated corners; the tied-grid inputs can contain dominated and duplicate points. Inputs are not filtered before timing.

The 3D sweep and the reference Chan algorithms differ algorithmically, so their timing ratio combines algorithm and implementation effects. Numerical Python versus Numba is the closer backend comparison. These are small, dimension-dependent test sizes; the timings do not estimate an asymptotic exponent or guarantee performance on large high-dimensional fronts.

Environment: Python `3.12.14 (main, Aug 25 2026, 14:01:42) [MSC v.1944 64 bit (AMD64)]`; NumPy `2.5.3`; Numba `0.68.0`; `Windows-11-10.0.26200-SP0`; CPU `AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD`.

## Ordinary hypervolume

All times below are milliseconds. Status words denote unfinished or failed workers rather than numeric measurements.

| Input | Numerical Python | Current Numba | Chan d/3 Python | Chan d/2 Python |
|---|---:|---:|---:|---:|
| sphere-d3-n32-hv | 0.578 | 0.185 | 10.435 | 2.797 |
| sphere-d3-n128-hv | 1.704 | 1.223 | 47.317 | 15.243 |
| sphere-d4-n12-hv | 27.549 | 9.904 | 12.245 | 2.370 |
| sphere-d4-n24-hv | 1495.842 | 255.693 | 26.757 | 6.006 |
| sphere-d5-n8-hv | 101.507 | 230.036 | 12.317 | 1.381 |
| sphere-d5-n16-hv | 1135.985 | 2106.774 | 44.488 | 5.979 |
| sphere-d6-n8-hv | 278.986 | 570.346 | 24.310 | 3.870 |
| sphere-d6-n12-hv | 1284.460 | 2577.558 | 55.926 | 8.985 |
| sphere-d7-n6-hv | 38.269 | 82.870 | 15.845 | 2.551 |
| sphere-d7-n8-hv | 733.066 | 1633.950 | 55.694 | 7.705 |
| sphere-d8-n6-hv | 598.112 | 1394.449 | 26.200 | 4.343 |
| sphere-d8-n10-hv | 5227.457 | timeout | 191.762 | 25.523 |
| sphere-d9-n4-hv | 10.794 | 16.269 | 9.434 | 1.095 |
| sphere-d9-n6-hv | 7201.031 | timeout | 37.379 | 3.864 |
| sphere-d10-n4-hv | 511.803 | 1319.259 | 16.438 | 0.871 |
| sphere-d10-n6-hv | 6025.325 | timeout | 91.463 | 9.667 |
| tied_grid-d3-n32-hv | 0.209 | 0.307 | 0.278 | 0.308 |
| tied_grid-d3-n128-hv | 0.497 | 0.490 | 1.247 | 1.217 |
| tied_grid-d4-n12-hv | 4.823 | 3.612 | 3.833 | 0.600 |
| tied_grid-d4-n24-hv | 42.552 | 14.118 | 7.702 | 1.371 |
| tied_grid-d5-n8-hv | 3.631 | 5.766 | 3.346 | 0.534 |
| tied_grid-d5-n16-hv | 254.779 | 432.337 | 24.355 | 1.773 |
| tied_grid-d6-n8-hv | 21.618 | 36.174 | 10.869 | 1.827 |
| tied_grid-d6-n12-hv | 31.886 | 53.409 | 14.819 | 2.316 |
| tied_grid-d7-n6-hv | 11.517 | 19.659 | 10.548 | 1.290 |
| tied_grid-d7-n8-hv | 19.880 | 29.367 | 23.639 | 3.736 |
| tied_grid-d8-n6-hv | 49.088 | 102.895 | 21.561 | 2.597 |
| tied_grid-d8-n10-hv | 1434.966 | 2935.382 | 66.429 | 8.355 |
| tied_grid-d9-n4-hv | 10.239 | 14.381 | 6.770 | 1.364 |
| tied_grid-d9-n6-hv | 45.334 | 98.065 | 23.289 | 3.615 |
| tied_grid-d10-n4-hv | 9.947 | 15.429 | 9.537 | 0.665 |
| tied_grid-d10-n6-hv | 1049.002 | 2647.617 | 32.620 | 4.024 |

## Dimension-by-dimension comparison

Ratios are baseline time divided by current Numba time, using only completed ordinary-HV pairs. A ratio above one means Numba was faster. Each cell reports the median of per-input ratios, followed by the number of matched inputs. Changing the dimension also changes the tested point counts; compare within a dimension rather than reading this table as a scaling curve.

| Dimension | Current backend | Current worker statuses | Python numerical / Numba | Chan d/3 / Numba | Chan d/2 / Numba |
|---|---|---|---:|---:|---:|
| 3D | compiled sweep | 4 ok | 1.2x (4) | 20.6x (4) | 7.47x (4) |
| 4D | whole-contraction JIT | 4 ok | 2.9x (4) | 0.803x (4) | 0.132x (4) |
| 5D | hybrid Numba | 4 ok | 0.564x (4) | 0.0549x (4) | 0.00505x (4) |
| 6D | hybrid Numba | 4 ok | 0.548x (4) | 0.16x (4) | 0.0251x (4) |
| 7D | hybrid Numba | 4 ok | 0.524x (4) | 0.364x (4) | 0.0482x (4) |
| 8D | hybrid Numba | 3 ok, 1 timeout | 0.477x (3) | 0.0226x (3) | 0.00311x (3) |
| 9D | hybrid Numba | 3 ok, 1 timeout | 0.663x (3) | 0.471x (3) | 0.0673x (3) |
| 10D | hybrid Numba | 3 ok, 1 timeout | 0.396x (3) | 0.0125x (3) | 0.00152x (3) |

## 4D check against the previous hybrid

This compares the new 4D backend with the generic ND hybrid on identical inputs. The older HV4DMagnitude prefix solver is outside this 3D–10D comparison; its earlier dedicated 4D comparison remains available [here](../../../NumericalChanHV4D/benchmarks/whole_loop/RESULTS.md).

| Input | Current whole-contraction Numba (ms) | Previous hybrid (ms) | Hybrid / current |
|---|---:|---:|---:|
| sphere-d4-n12-hv | 9.904 | 45.576 | 4.60x |
| sphere-d4-n12-mag | 9.681 | 50.891 | 5.26x |
| sphere-d4-n24-hv | 255.693 | 2320.770 | 9.08x |
| tied_grid-d4-n12-hv | 3.612 | 6.733 | 1.86x |
| tied_grid-d4-n24-hv | 14.118 | 79.625 | 5.64x |

## Magnitude

These additional sphere cases compare direct numerical magnitude with its equivalent affine-transformed HV in both Chan references. Times are milliseconds. They are excluded from the ordinary-HV ratio table above.

| Input | Numerical Python | Current Numba | Chan d/3 Python | Chan d/2 Python |
|---|---:|---:|---:|---:|
| sphere-d3-n32-mag | 0.418 | 0.236 | 9.189 | 2.588 |
| sphere-d4-n12-mag | 28.919 | 9.681 | 12.857 | 1.498 |
| sphere-d5-n8-mag | 108.132 | 191.156 | 10.820 | 1.501 |
| sphere-d6-n8-mag | 269.181 | 559.966 | 25.633 | 4.002 |
| sphere-d7-n6-mag | 37.239 | 74.575 | 16.411 | 2.217 |
| sphere-d8-n6-mag | 628.869 | 1433.465 | 25.219 | 4.283 |
| sphere-d9-n4-mag | 10.702 | 15.349 | 9.105 | 0.981 |
| sphere-d10-n4-mag | 547.877 | 1302.755 | 16.085 | 1.162 |

## First use and validation

The following ranges include every recorded first-call duration for current Numba, including a first call whose subsequent repetitions exceeded the worker budget. A first call may compile specializations or load existing cached code; these are **not fresh-cache compilation measurements**. The first-call durations exclude imports and include computation. They are not included in the warm medians.

| Dimension | Recorded first calls | First-call range (s) |
|---|---:|---:|
| 3D | 5 | 0.2285–2.6197 |
| 4D | 5 | 2.3638–2.8249 |
| 5D | 5 | 0.3284–2.3705 |
| 6D | 5 | 0.3364–2.9534 |
| 7D | 5 | 0.3071–1.9192 |
| 8D | 5 | 0.3705–12.3057 |
| 9D | 5 | 0.2871–18.6912 |
| 10D | 5 | 0.2560–16.0878 |

Across 162 completed rows with a recorded comparison, the largest scaled disagreement with Chan d/2 was **6.38e-16**. Scaled error is absolute disagreement divided by `max(1, abs(reference value))`. The runner checks the first call and all timed repetitions; incomplete workers are not counted as correctness passes. Agreement is a numerical cross-check, not a replacement for the existing exact-oracle test suites.

The independent [exact rational inclusion–exclusion audit](results/oracle_verification.json) reports **passed**: **31 inputs** with at most 12 points, **503 recorded values**, and maximum scaled error **6.38e-16**. The case count is checked against the saved input selection. The oracle uses the exact rational values of the binary64 input coordinates and shares no solver or compression code. Its value checks include any available samples from timed-out workers; they do not turn those workers into completed timing measurements.

## Reproduction records

The saved [input arrays](results/datasets.json), [full timing and counter records](results/timings.json), and [CSV timings](results/timings.csv) preserve the cases, per-call values, statuses, source hashes, and environment. Additional [hardware details](results/hardware.json) identify the measured host. [benchmark_dimensions.py](benchmark_dimensions.py) runs the experiment; [summarize_results.py](summarize_results.py) builds this document from those records without rerunning an algorithm.
