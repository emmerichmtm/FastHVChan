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

```bash
python -m pip install -r requirements.txt
```

```python
from numerical_chan4_numba import hypervolume4
value = hypervolume4(points)
```

This wrapper uses the shared float64 engine in `../NumericalChanHVND`.
Keep both directories together. Array scans, prefix sums, normalization,
suffix masks, and cutoff searches are compiled in Numba nopython mode;
geometric and term recursion remains Python. No fastmath or parallelism.

## Report, tests, and timings

- [LaTeX report](paper/numerical_chan4.tex) / [PDF](paper/numerical_chan4.pdf).
- `python verify_numerical_chan4.py` runs exact correctness tests.
- The shared Numba validation is `../NumericalChanHVND/verify_numba.py`.
- [Measured comparison](benchmarks/RESULTS.md) includes unmodified Python
  Chan Section 4.2 and Section 2, Python numerical, and Numba numerical.
- Full benchmark runner, datasets, and environment are in the ND directory;
  this directory contains the 4D rows and inputs.

The default block schedule is part of the complexity proof. `block_levels`
is a stress-test override and has no arbitrary-use complexity guarantee.

## Measured performance

On the five completed matched ordinary-HV 4D cases, the hybrid Numba backend
was a median 1.60x slower than numerical Python and 11.60x slower than
the existing Python Chan Section-4.2 implementation. The spherical n=48
numerical runs hit the 35-second whole-worker limit.

See the full methodology and raw data in [benchmarks/RESULTS.md](benchmarks/RESULTS.md).
