# NumericalChanHV4D

Numerical four-dimensional Chan recursion for unions of origin-anchored boxes.
The exact backend stores cell masses and monotone integer cutoff arrays;
prefix sums replace integration. Its fixed-dimensional arithmetic bound is
`O(n^(4/3) polylog(n))`, with large hidden constants. This is a research
implementation, not a claim to beat the existing Chan code in wall time.

## Use

From this directory:

```python
from numerical_chan4 import hypervolume4
points = [(1, 4, 3, 2), (2, 3, 4, 1), (3, 2, 1, 4), (4, 1, 2, 3)]
assert hypervolume4(points) == 69
assert hypervolume4(points, magnitude=True) == 765 / 16
```

Nonnegative corners describe `[0,p]` boxes. Convert minimization points `a`
with reference `r` to extents `r-a`. Integer/Fraction input is exact in the
standard-library backend. Floating input can suffer cancellation.

## Numba

The newer [adaptive compiled engine](../NumericalChanHVND/ADAPTIVE_COMPILED.md)
adds Chan-style cell simplification and adaptive compression. Its 4D convenience
entry point is `numerical_chan4_adaptive.hypervolume4`; it also supports all
dimensions 2-10 through the generic interface. The scheduled backend below is
retained unchanged for comparison.

```bash
python -m pip install -r requirements.txt
```

```python
from numerical_chan4_numba import hypervolume4
value = hypervolume4(points)
```

The 4D entry point now uses [whole compiled contractions](numerical_chan4_compiled.py).
Term states stay native throughout recursion: constraint updates, elimination,
merging, base evaluation and compression execute inside Numba. The geometric
recursion stays readable Python and uses the same compression schedule.
The [implementation guide](WHOLE_LOOP_JIT.md) explains the small adapter and
the numerical kernel. No fastmath or parallelism is enabled.

Keep both directories together: the geometric engine and basic helpers are
shared with `../NumericalChanHVND`. That directory retains the earlier hybrid
backend, which remains available as a comparison. First use can include JIT
compilation; reported performance measurements distinguish it from warm calls.

## Report, tests, and timings

- [Adaptive compiled 2D-10D design](../NumericalChanHVND/ADAPTIVE_COMPILED.md)
  and [latest timings](../NumericalChanHVND/benchmarks/compiled_2d_10d/RESULTS.md).
  `python ../NumericalChanHVND/verify_compiled.py` validates this new backend.
- [Whole-contraction JIT report (PDF)](paper/whole_loop_jit_4d_report.pdf) /
  [self-contained LaTeX source](paper/whole_loop_jit_4d_report.tex): recursion,
  contraction proof, magnitude equivalence, verification and 4D performance.
- [LaTeX report](paper/numerical_chan4.tex) / [PDF](paper/numerical_chan4.pdf).
- `python verify_numerical_chan4.py` runs exact correctness tests.
- `python verify_whole_loop.py` checks the current compiled 4D backend,
  including direct magnitude, signed terms, and repeated compression.
- The shared Numba validation is `../NumericalChanHVND/verify_numba.py`.
- [Measured comparison](benchmarks/RESULTS.md) includes unmodified Python
  Chan Section 4.2 and Section 2, Python numerical, and Numba numerical.
- Full benchmark runner, datasets, and environment are in the ND directory;
  this directory contains the 4D rows and inputs.

The default block schedule is part of the complexity proof. `block_levels`
is a stress-test override and has no arbitrary-use complexity guarantee.

## Measured performance

The latest [adaptive compiled benchmark](../NumericalChanHVND/benchmarks/compiled_2d_10d/RESULTS.md)
includes a 24-point 4D sphere case: 17.0 ms for the new compiled solver,
28.6 ms for Python Chan d/3, and 1,878.0 ms for the original numerical Python.
Python Chan d/2 was faster at 6.2 ms. The new default compression gate stays
enabled; a separate [1,552-box structural probe](../NumericalChanHVND/benchmarks/compiled_2d_10d/NATURAL_COMPRESSION.md)
executes six compressions and matches both references. Repeated compression
is also covered by the new correctness suite. See the full report for timing
variability and the distinction between default and forced cases.

### Previous scheduled-backend measurements

The subsequent [3D-10D benchmark](../NumericalChanHVND/benchmarks/current_3d_10d/RESULTS.md)
includes further 4D whole-contraction versus hybrid comparisons, alongside the
current implementations in the other dimensions and direct magnitude cases.

The [current 4D benchmark](benchmarks/whole_loop/RESULTS.md) measures the new
whole-contraction backend against the earlier hybrid, older compiled prefix
solver, and Python Chan implementations on seven common inputs. It improves
the hybrid by 1.47-8.13x on completed matched cases (median 7.76x).
On the seed-42 twenty-point sphere, time drops from 2.050 s to 0.252 s.
The older prefix solver still takes only 0.019 s on that case; scheduled
compression continues to create many more terms in the new algorithm.

All seven new-backend cases finish, and all 33 completed algorithm/input
comparisons agree. Raw inputs, timings, counters and profiles are included.
The mathematical recursion and compression policy are unchanged by this JIT
work; no improved asymptotic exponent is claimed.

### Earlier audits of the hybrid backend

The [4D follow-up audit](benchmarks/prefix4/REPORT.md) also measures the older
compiled `HV4DMagnitude` prefix solver on common saved inputs. On spherical
fronts with 20 or 40 points, that solver is 1.59-2.11x faster than Python
Chan d/3; at 20 points it was 84.59-96.99x faster than the earlier hybrid backend.
Whole-loop compilation and the cost of the new compression policy explain
the difference. Chan d/2 remains fastest on these small test cases.

The audit includes [LaTeX](paper/prefix4_performance_audit.tex),
[PDF](paper/prefix4_performance_audit.pdf), raw repeats, inputs, profiles,
compilation and compression ablations, and a separate magnitude-kernel bug
reproducer. It establishes measured 4D performance, not an asymptotic improvement.

The earlier comparison below did not include the older compiled prefix solver:

On the five completed matched ordinary-HV 4D cases, the hybrid Numba backend
was a median 1.60x slower than numerical Python and 11.60x slower than
the existing Python Chan Section-4.2 implementation. The spherical n=48
numerical runs hit the 35-second whole-worker limit.

See the full methodology and raw data in [benchmarks/RESULTS.md](benchmarks/RESULTS.md).
