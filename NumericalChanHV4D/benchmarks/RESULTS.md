# Measured comparison with the unmodified Python Chan implementations

The same saved input arrays were used for all implementations. Timings include
rank preprocessing and API coordinate transformations, but exclude imports.
Every variant received an untimed first call; the table reports the median of
3 subsequent calls. For Numba that first call may include compilation
or cache loading. Fresh-cache probes are listed separately below. The benchmark
ran one worker at a time. The limit was 35 seconds for a worker's
whole first-call-plus-repeats sequence, including process startup; a timeout is
not a measured per-call duration. No speedup is inferred from a timeout.

The original Python implementations are `chan_orthant_dby3.py` (Chan Section 4.2)
and `chan_hypervolume.py` (Chan Section 2), at repository commit
`08539aa99e94e181d7230a8288885b3294621c9e`. Neither uses Numba. The dominance
prefilter was disabled in both. Section 4.2 does not support 2D. Magnitude was
passed to both references through HV corners `1+p/2`, including this conversion
in timing. Numerical Python was also run with floats, not Fraction arithmetic.

The Numba backend is hybrid: its recursive control and term construction are
Python. In 2D/3D the sweep is compiled; in 4D-10D numerical array kernels are
compiled. No fastmath or parallel execution is used. Speedups of the low-dimensional
sweeps over Chan therefore combine an algorithm change with implementation effects.
The numerical Python-versus-Numba comparison better isolates the backend change.

Environment: Python 3.12.14, NumPy
2.5.3, Numba 0.68.0,
Windows-11-10.0.26200-SP0; AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD.

For completed matched ordinary-HV cases in 4D, the median ratio Numba numerical / Python numerical was 1.60x (5 cases). The median ratio Numba numerical / the repository's Python Section-4.2 implementation was 11.60x (5 cases). Ratios above one mean the Numba version was slower. These are medians of per-case ratios for this small benchmark set. They do not support a practical speedup claim for this four-dimensional implementation.

| Input | Numerical Python (s) | Numerical Numba (s) | Chan 4.2 Python (s) | Chan 2 Python (s) |
|---|---:|---:|---:|---:|
| sphere-d4-n12-hv | 0.028069 | 0.041134 | 0.012223 | 0.0017601 |
| sphere-d4-n12-mag | 0.028301 | 0.043302 | 0.0126 | 0.0019805 |
| sphere-d4-n24-hv | 1.4637 | 2.3398 | 0.027249 | 0.0054424 |
| sphere-d4-n48-hv | timeout | timeout | 0.10325 | 0.018492 |
| tied_grid-d4-n12-hv | 0.0046516 | 0.0064536 | 0.0030063 | 0.0005476 |
| tied_grid-d4-n12-mag | 0.004033 | 0.006346 | 0.0034868 | 0.000561 |
| tied_grid-d4-n24-hv | 0.045366 | 0.076442 | 0.0065916 | 0.001393 |
| tied_grid-d4-n48-hv | 0.074685 | 0.1358 | 0.0018386 | 0.0009735 |

## First use with a fresh empty JIT cache

| Dimension | First use (s) | Second use (s) |
|---|---:|---:|
| 2 | 2.5352 | 9.72e-05 |
| 3 | 2.496 | 0.0001481 |
| 4 | 1.9391 | 0.0030841 |

These probes exclude module import time; the first measurement includes both compilation and computation. They are not isolated compiler-time measurements.

Completed implementations agreed within maximum scaled difference 3.31e-16; scaled difference means absolute disagreement divided by max(1, absolute reference value). Timeouts and noncompleted cases are excluded from this check.

Raw input data, per-repeat times, first-call times, statuses, source hashes, and environment are saved alongside this file. No asymptotic exponent was fitted from these small cases.
