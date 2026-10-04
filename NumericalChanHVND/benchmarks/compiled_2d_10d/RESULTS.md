# Compiled numerical hypervolume, 2D–10D

Complete run: 48 datasets, 187 of 187 workers recorded; 186 ok, 1 timeout. Validation errors: **0**.

## Key findings

The compiled solver completed **48 of 48 recorded inputs**, including direct magnitude and all forced checks.

On ordinary-HV inputs in 4D–10D, compiled numerical was faster than Chan d/3 Python on **26/28 completed pairs**; the median baseline/compiled ratio was **2.02×**.

On ordinary-HV inputs in 4D–10D, compiled numerical was faster than the original numerical Python on **26/27 completed pairs**; the median baseline/compiled ratio was **6.29×**.

Chan d/2 Python was faster than compiled numerical on **28/28** completed ordinary-HV pairs in 4D–10D. The new implementation is therefore not the fastest solver across all tested dimensions and inputs.

Among the **35 completed default 4D–10D inputs**, **0** executed compression; there were 994 size-gate checks in total. Compression was enabled, but these states stayed below its threshold. Thus the default speedups do not measure the cost of compression when it is required. The separate forced cases demonstrate that the implemented path remains operational.

A separate [natural-compression structural probe](NATURAL_COMPRESSION.md) also demonstrates compression under the default settings. It is outside the 48-input suite, and its first-call measurements are excluded from all warm timing comparisons here.

## What is compared

The compiled solver uses compiled sweeps in 2D/3D and complete native term contractions in 4D–10D. Its geometric recursion stays in Python. It applies cell-local cropping, slab tightening, and an adaptive compression gate. Compression is enabled in every run. A default input can legitimately need no compression; separate forced cases verify that the compression path runs.

Numerical Python is the earlier finite-mass solver with its original local block schedule. Chan d/3 is the step-function implementation with its default adaptive schedule, except on the explicitly forced inputs. Chan d/2 is the reference implementation used to validate every available value. These are implementation comparisons, not a claim that compilation alone explains every difference.

**No asymptotic bound is claimed for the new adaptive implementation in this benchmark.** The adaptive gate needs a separate complexity proof. These small, dimension-dependent inputs do not estimate an asymptotic exponent.

## Method

Each worker has a 45-second total budget, including startup, imports, an untimed first call, and 3 timed repetitions. The table reports warm medians in milliseconds. Input transforms and preprocessing are included; imports are excluded from call timings. Workers run sequentially and construct a new solver on every call. Disk caches are retained and warmed by verification beforehand; first-use times are archived separately and are not empty-cache compilation times. No dominance prefilter, fastmath, or parallel numerical execution is used.

A timeout is a worker-budget result, not a lower bound on one call. No timing ratio uses a timeout. Completed-pair summaries can have selection bias; read them with the status counts. Raw samples, inputs, counters, source hashes, and Numba signatures are saved in [results](results/).

The ordinary box union is identical for every implementation. Direct numerical magnitude is checked against HV after the affine transformation `1 + extent/2`. Sphere corners are mutually nondominated; tied grids can have duplicate or dominated corners. The previous 3D–10D inputs are reproduced exactly, with five 2D inputs added.

Environment: python: `3.12.14 (main, Aug 25 2026, 14:01:42) [MSC v.1944 64 bit (AMD64)]`; platform: `Windows-11-10.0.26200-SP0`; cpu: `AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD`; logical_cpus: `16`; numpy: `2.5.3`; numba: `0.68.0`.

## Ordinary hypervolume

| Input | Compiled numerical | Numerical Python | Chan d/3 Python | Chan d/2 Python |
|---|---:|---:|---:|---:|
| sphere-d2-n32-hv | 0.242 | 0.160 | not applicable | 0.983 |
| sphere-d2-n128-hv | 0.474 | 0.770 | not applicable | 5.313 |
| sphere-d3-n32-hv | 0.190 | 0.352 | 9.458 | 2.505 |
| sphere-d3-n128-hv | 0.981 | 1.609 | 44.524 | 19.108 |
| sphere-d4-n12-hv | 6.863 | 27.013 | 11.508 | 1.656 |
| sphere-d4-n24-hv | 16.995 | 1877.982 | 28.632 | 6.216 |
| sphere-d5-n8-hv | 5.611 | 101.566 | 11.751 | 3.239 |
| sphere-d5-n16-hv | 16.971 | 1434.933 | 78.805 | 7.853 |
| sphere-d6-n8-hv | 13.361 | 280.798 | 26.910 | 3.426 |
| sphere-d6-n12-hv | 27.651 | 1634.712 | 54.249 | 9.206 |
| sphere-d7-n6-hv | 7.974 | 36.991 | 17.161 | 2.005 |
| sphere-d7-n8-hv | 20.201 | 759.785 | 66.189 | 6.179 |
| sphere-d8-n6-hv | 8.106 | 600.637 | 23.654 | 3.755 |
| sphere-d8-n10-hv | 130.842 | 11894.048 | 182.583 | 24.820 |
| sphere-d9-n4-hv | 4.864 | 17.431 | 16.203 | 0.998 |
| sphere-d9-n6-hv | 30.459 | timeout | 51.050 | 5.342 |
| sphere-d10-n4-hv | 10.176 | 913.106 | 22.798 | 1.730 |
| sphere-d10-n6-hv | 62.646 | 6737.424 | 96.593 | 11.206 |
| tied_grid-d2-n32-hv | 0.220 | 0.109 | not applicable | 0.156 |
| tied_grid-d2-n128-hv | 0.979 | 1.024 | not applicable | 2.245 |
| tied_grid-d3-n32-hv | 0.215 | 0.155 | 0.704 | 0.305 |
| tied_grid-d3-n128-hv | 1.375 | 0.504 | 2.188 | 1.218 |
| tied_grid-d4-n12-hv | 3.989 | 6.734 | 5.576 | 0.598 |
| tied_grid-d4-n24-hv | 6.580 | 41.378 | 7.947 | 1.086 |
| tied_grid-d5-n8-hv | 7.250 | 5.200 | 6.603 | 0.579 |
| tied_grid-d5-n16-hv | 13.276 | 789.458 | 29.147 | 2.082 |
| tied_grid-d6-n8-hv | 14.939 | 32.401 | 12.856 | 2.180 |
| tied_grid-d6-n12-hv | 12.431 | 33.237 | 31.002 | 1.942 |
| tied_grid-d7-n6-hv | 5.545 | 9.830 | 9.628 | 1.855 |
| tied_grid-d7-n8-hv | 10.033 | 21.050 | 18.958 | 2.524 |
| tied_grid-d8-n6-hv | 9.304 | 42.911 | 22.292 | 2.070 |
| tied_grid-d8-n10-hv | 30.948 | 1731.896 | 74.551 | 8.077 |
| tied_grid-d9-n4-hv | 4.482 | 8.526 | 9.038 | 0.832 |
| tied_grid-d9-n6-hv | 12.150 | 46.522 | 25.400 | 2.700 |
| tied_grid-d10-n4-hv | 4.518 | 10.668 | 9.036 | 0.659 |
| tied_grid-d10-n6-hv | 11.128 | 1209.791 | 36.693 | 3.619 |

Across the 48 completed compiled workers, the ratio of slowest to fastest warm repetition had median **1.28** and maximum **2.74**; 5 workers exceeded a ratio of two. These are three wall-clock samples per case under ordinary system load, without a dedicated machine or CPU isolation. The recorded variability limits the precision of small timing differences; the measurements do not supply confidence intervals.

## Completed-pair comparisons

Ratios are baseline time divided by compiled numerical time. Above one means compiled numerical is faster. Each entry gives the median per-input ratio and number of completed ordinary-HV pairs. Sizes change with dimension.

| Dimension | Compiled statuses | Numerical Python / compiled | Chan d/3 / compiled | Chan d/2 / compiled |
|---|---|---:|---:|---:|
| 2D | 4 ok | 0.855× (4) | not applicable | 3.18× (4) |
| 3D | 4 ok | 1.18× (4) | 24.3× (4) | 7.31× (4) |
| 4D | 4 ok | 5.11× (4) | 1.54× (4) | 0.203× (4) |
| 5D | 4 ok | 38.8× (4) | 2.14× (4) | 0.31× (4) |
| 6D | 4 ok | 11.8× (4) | 1.99× (4) | 0.206× (4) |
| 7D | 4 ok | 3.37× (4) | 2.02× (4) | 0.279× (4) |
| 8D | 4 ok | 65× (4) | 2.4× (4) | 0.242× (4) |
| 9D | 4 ok | 3.58× (3) | 2.05× (4) | 0.195× (4) |
| 10D | 4 ok | 98.6× (4) | 2.12× (4) | 0.174× (4) |

## Magnitude

| Input | Compiled numerical | Numerical Python | Chan d/3 Python | Chan d/2 Python |
|---|---:|---:|---:|---:|
| sphere-d2-n32-mag | 0.224 | 0.175 | not applicable | 1.639 |
| sphere-d3-n32-mag | 0.244 | 0.425 | 10.243 | 2.475 |
| sphere-d4-n12-mag | 6.076 | 31.790 | 15.215 | 1.707 |
| sphere-d5-n8-mag | 5.809 | 114.567 | 10.399 | 1.888 |
| sphere-d6-n8-mag | 12.317 | 269.159 | 24.251 | 3.168 |
| sphere-d7-n6-mag | 8.029 | 53.944 | 14.653 | 3.232 |
| sphere-d8-n6-mag | 16.749 | 604.221 | 29.762 | 3.875 |
| sphere-d9-n4-mag | 10.825 | 17.609 | 8.485 | 2.007 |
| sphere-d10-n4-mag | 5.715 | 523.388 | 17.685 | 1.025 |

## Forced compression checks

These extra inputs take the first eight corners of the saved 4D sphere and append constant unit extents to obtain 7D and 10D. Compiled numerical and Chan d/3 use `compress_every=2, compress_factor=0, use_compression=True`. Numerical Python retains its own schedule. These timings are excluded from the performance ratios above. A completed compiled row is accepted only if its compression counter is positive and its values agree with Chan d/2.

| Input | Compiled numerical | Numerical Python | Chan d/3 Python | Chan d/2 Python | Compiled compressions | Compiled peak terms |
|---|---:|---:|---:|---:|---:|---:|
| forced-padded-d4-n8-hv | 7.494 | 3.782 | 393.400 | 0.672 | 1 | 135 |
| forced-padded-d7-n8-hv | 9.968 | 5.388 | 559.963 | 0.866 | 1 | 135 |
| forced-padded-d10-n8-hv | 12.921 | 7.336 | 659.148 | 1.065 | 1 | 135 |

## Validation and interpretation

Maximum recorded scaled error against Chan d/2: 6.38e-16. Every first-call value and every retained warm value is checked with relative tolerance `2e-9` and absolute tolerance `2e-10`. Agreement between implementations complements the independent exact checks in the solver verification suite.

The [independent exact oracle](results/oracle_verification.json) **passed** on **34 saved datasets** and **543 recorded values**, including 3 forced-compression inputs. It evaluates exact rational inclusion–exclusion for the binary64 coordinates on inputs of at most 12 points. Its maximum scaled error was **6.38e-16**. The archived oracle input hashes match these timing and dataset files.

Counters distinguish a small recursion from a large contraction: compression lifts `d` axes to `2d`, and elimination can expand signed terms before merging. Native compilation removes Python work inside these contractions; it does not remove their mathematical expansion. Compare counters as well as elapsed time.

For an input that actually crosses the adaptive threshold without forcing it, see the separate [natural-compression report](NATURAL_COMPRESSION.md). Its structural evidence supplements the zero-compression default cases in this main suite; it is not added to their timing aggregates.

Previous results: [original 3D–10D hybrid comparison](../current_3d_10d/RESULTS.md) and [earlier dedicated 4D whole-contraction comparison](../../../NumericalChanHV4D/benchmarks/whole_loop/RESULTS.md).
