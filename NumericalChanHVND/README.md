# NumericalChanHVND: dimensions 2 through 10

Standalone numerical hypervolume and dominated-set L1 magnitude engines.
Every integer dimension from 2 to 10 is supported, including 5, 7, and 9.
The two original Chan implementations at the repository root are unchanged.

| Dimensions | Method | Arithmetic bound |
|---|---|---|
| 2 | Descending numerical sweep | O(n log n) |
| 3 | Sweep with weighted skyline segment tree | O(n log n) |
| 4-10 | Paired Chan cuts and numerical block-mass compression | O(n^(d/3) polylog(n)), fixed d |

Odd-dimensional rounds finish with one singleton. Compression uses 2d variables,
up to 20 in 10D, and retains constraints between different coordinate pairs.
Hidden dimension-dependent constants and logarithmic exponents can be large.

## Exact/standard-library backend

```python
from numerical_chan import hypervolume, magnitude, NumericalChan
p = [(2, 1, 1, 1, 1, 1, 1), (1, 2, 1, 1, 1, 1, 1)]
assert hypervolume(p) == 3
mag = magnitude(p)
solver = NumericalChan(dimension=7)
value = solver.compute(p)
print(solver.stats)
```

Corners are nonnegative extents of `[0,p]` boxes. For minimization data `a`
and a reference `r` satisfying `a <= r`, use `p = r-a`. Empty input is zero;
zero coordinates and duplicates are allowed. Zero-coordinate boxes must be
retained for magnitude. Dimension is inferred from nonempty input or checked
with `dimension=...`. Integer/Fraction arithmetic is exact; floats are supported.

## Float64 Numba backend

Tested with CPython 3.12.14, NumPy 2.5.3, Numba 0.68.0 on Windows.

```bash
python -m pip install -r requirements.txt
```

```python
from numerical_chan_numba import hypervolume, magnitude, NumericalChanNumba
value = hypervolume(p)
```

In 2D/3D the sweep and segment-tree kernels are compiled. In 4D-10D the
array scans, prefix sums, cutoff searches, normalization, and suffix masks
are compiled, while geometric recursion and term construction remain Python.
This is a **hybrid backend**, not a fully JIT-compiled solver. No fastmath or
parallelism is enabled. Conversion to float64 loses exact-rational semantics;
cancellation remains possible. Numba uses a disk cache, so first calls and
warm calls must be distinguished.

## Validate and reproduce the comparison

Run from this directory:

```bash
python verify_numerical_chan.py
python verify_numba.py
python benchmark_numerical_chan.py --reference-root .. --out benchmarks/reproduction
```

The exact suite checks 234 end-to-end cases across all nine dimensions, both
affine magnitude/HV transforms, 2d-variable compression and repeated masks.
The Numba suite independently checks its results against exact oracles.
High-dimensional compression regressions are small and structured; these are
not large dense 10D performance guarantees. Numerical array enumeration is
used only by independent test oracles, never by the solver.

- [Report PDF](paper/numerical_chan_2d_10d.pdf) and [LaTeX source](paper/numerical_chan_2d_10d.tex).
- [Benchmark findings](benchmarks/RESULTS.md), raw JSON/CSV, saved input datasets.
- Exact and floating validation records: `numerical_chan_results.json`,
  `numba_verification.json`.

The benchmark disables the Chan dominance prefilter, includes preprocessing
and API transforms in each timed call, records an untimed first call for all
variants, and reports medians of three warm calls. Timeouts remain explicit;
no speedup is inferred from them. The Section 4.2 reference is unsupported in
2D, where the Section 2 reference is used. For magnitude the references receive
the equivalent HV corners `1+p/2`, with that conversion included in timing.

The default local block schedule is necessary for the proof. Do not infer
that an arbitrary `block_levels` stress-test override has the same bound.

## Measured performance

On 20 completed matched ordinary-HV cases in dimensions 4-10, the hybrid
Numba backend was a median 1.93x slower than numerical Python and 21.25x
slower than the existing Python Chan Section-4.2 implementation. On the
3D spherical n=128 case, the compiled sweep took 0.64 ms versus 1.77 ms
for numerical Python. These are observed sample results, not general
speed guarantees. Eleven of 156 benchmark workers timed out.

See the full methodology and raw data in [benchmarks/RESULTS.md](benchmarks/RESULTS.md).
