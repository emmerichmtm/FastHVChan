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

For completed matched ordinary-HV cases in dimensions 4-10, the median ratio Numba numerical / Python numerical was 1.93x (20 cases). The median ratio Numba numerical / the repository's Python Section-4.2 implementation was 21.25x (20 cases). Ratios above one mean the Numba version was slower. These are medians of per-case ratios for this small benchmark set, not universal performance predictions. Array-level JIT compilation does not remove the cost of enumerating and managing many terms. The results do not support a high-dimensional speedup claim for this implementation.

| Input | Numerical Python (s) | Numerical Numba (s) | Chan 4.2 Python (s) | Chan 2 Python (s) |
|---|---:|---:|---:|---:|
| sphere-d2-n64-hv | 0.0004235 | 0.0004512 | --- | 0.0029573 |
| sphere-d2-n256-hv | 0.0014577 | 0.001104 | --- | 0.013813 |
| sphere-d3-n32-hv | 0.0003416 | 0.0002351 | 0.011185 | 0.0027285 |
| sphere-d3-n128-hv | 0.0017669 | 0.0006409 | 0.047406 | 0.015749 |
| sphere-d4-n12-hv | 0.028069 | 0.041134 | 0.012223 | 0.0017601 |
| sphere-d4-n12-mag | 0.028301 | 0.043302 | 0.0126 | 0.0019805 |
| sphere-d4-n24-hv | 1.4637 | 2.3398 | 0.027249 | 0.0054424 |
| sphere-d4-n48-hv | timeout | timeout | 0.10325 | 0.018492 |
| sphere-d5-n8-hv | 0.1066 | 0.21681 | 0.010083 | 0.0015649 |
| sphere-d5-n16-hv | 1.1589 | 2.2592 | 0.049136 | 0.0066998 |
| sphere-d6-n8-hv | 0.29042 | 0.64459 | 0.030688 | 0.0038519 |
| sphere-d6-n12-hv | 1.3261 | 2.8731 | 0.060131 | 0.0075414 |
| sphere-d7-n8-hv | 0.75676 | 1.7606 | 0.055029 | 0.0070965 |
| sphere-d7-n8-mag | 0.77396 | 1.7404 | 0.052417 | 0.0065812 |
| sphere-d8-n6-hv | 0.64398 | 1.462 | 0.036778 | 0.004028 |
| sphere-d8-n10-hv | 7.6582 | timeout | 0.19733 | 0.03268 |
| sphere-d9-n6-hv | timeout | timeout | 0.03161 | 0.0032998 |
| sphere-d10-n6-hv | 7.3794 | timeout | 0.087428 | 0.0094668 |
| sphere-d10-n6-mag | 7.1093 | timeout | 0.083935 | 0.01069 |
| sphere-d10-n8-hv | timeout | timeout | 0.20869 | 0.023293 |
| tied_grid-d2-n64-hv | 0.0001821 | 0.0003546 | --- | 0.0003574 |
| tied_grid-d2-n256-hv | 0.0011282 | 0.001448 | --- | 0.0011012 |
| tied_grid-d3-n32-hv | 0.0001543 | 0.0001992 | 0.0002508 | 0.0002783 |
| tied_grid-d3-n128-hv | 0.0009741 | 0.000561 | 0.0008789 | 0.0009561 |
| tied_grid-d4-n12-hv | 0.0046516 | 0.0064536 | 0.0030063 | 0.0005476 |
| tied_grid-d4-n12-mag | 0.004033 | 0.006346 | 0.0034868 | 0.000561 |
| tied_grid-d4-n24-hv | 0.045366 | 0.076442 | 0.0065916 | 0.001393 |
| tied_grid-d4-n48-hv | 0.074685 | 0.1358 | 0.0018386 | 0.0009735 |
| tied_grid-d5-n8-hv | 0.0039873 | 0.0043013 | 0.0024143 | 0.0005417 |
| tied_grid-d5-n16-hv | 0.23956 | 0.42104 | 0.019281 | 0.002114 |
| tied_grid-d6-n8-hv | 0.019569 | 0.039193 | 0.010568 | 0.0013638 |
| tied_grid-d6-n12-hv | 0.029085 | 0.045756 | 0.018626 | 0.0019664 |
| tied_grid-d7-n8-hv | 0.017597 | 0.033695 | 0.020898 | 0.0033299 |
| tied_grid-d7-n8-mag | 0.0194 | 0.025364 | 0.019358 | 0.0025012 |
| tied_grid-d8-n6-hv | 0.049372 | 0.089785 | 0.026469 | 0.0024849 |
| tied_grid-d8-n10-hv | 1.6024 | 3.5064 | 0.06902 | 0.0082419 |
| tied_grid-d9-n6-hv | 0.051739 | 0.10215 | 0.024152 | 0.0027103 |
| tied_grid-d10-n6-hv | 1.2636 | 3.4018 | 0.032566 | 0.0038714 |
| tied_grid-d10-n6-mag | 1.263 | 3.6763 | 0.034142 | 0.005999 |
| tied_grid-d10-n8-hv | timeout | timeout | 0.075188 | 0.01708 |

## First use with a fresh empty JIT cache

| Dimension | First use (s) | Second use (s) |
|---|---:|---:|
| 2 | 2.5352 | 9.72e-05 |
| 3 | 2.496 | 0.0001481 |
| 4 | 1.9391 | 0.0030841 |

These probes exclude module import time; the first measurement includes both compilation and computation. They are not isolated compiler-time measurements.

Completed implementations agreed within maximum scaled difference 3.31e-16; scaled difference means absolute disagreement divided by max(1, absolute reference value). Timeouts and noncompleted cases are excluded from this check.

Raw input data, per-repeat times, first-call times, statuses, source hashes, and environment are saved alongside this file. No asymptotic exponent was fitted from these small cases.
