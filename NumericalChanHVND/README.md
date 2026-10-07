# NumericalChanHVND: dimensions 2 through 10

Numerical hypervolume engines for every integer dimension from 2 to 10.
Use [`numerical_chan_local_compiled.py`](numerical_chan_local_compiled.py)
for the current implementation. Its numerical kernels are compiled with
Numba, and its local compression schedule is the one covered by the
[research report (PDF)](paper/numerical_product_chan.pdf) and
[LaTeX source](paper/numerical_product_chan.tex).

The report develops ordinary hypervolume directly from Lebesgue measure,
finite sums and products, retaining Chan's `O(n^(d/3) polylog(n))` arithmetic
bound for fixed dimension. The original Chan implementations at the
repository root are unchanged.

## Use the numerical implementation

From the repository root:

```sh
python -m pip install -r NumericalChanHVND/requirements.txt
```

```python
from NumericalChanHVND.numerical_chan_local_compiled import hypervolume

points = [(2, 1, 1, 1), (1, 2, 1, 1)]
assert hypervolume(points) == 3
```

Points are nonnegative extents of origin-anchored boxes `[0,p]`. For
minimization objectives `a` and a reference `r` satisfying `a <= r`, use
`p = r-a`. Empty input has hypervolume zero; duplicates and zero coordinates
are allowed. Dimension is inferred from nonempty input or can be supplied
with `dimension=...`.

| Dimensions | Method | Arithmetic bound |
|---|---|---|
| 2 | Descending numerical sweep | O(n log n) |
| 3 | Sweep with weighted skyline segment tree | O(n log n) |
| 4–10 | Chan cuts with numerical contractions and compression | O(n^(d/3) polylog(n)), fixed d |

Odd-dimensional rounds end with a single remaining coordinate. Compression
uses `2d` auxiliary variables and retains constraints between coordinate
pairs. Dimension-dependent constants and logarithmic factors can be large.
The proof requires the local schedule; an arbitrary compression schedule
does not inherit the same bound.

The implementation uses float64 arithmetic, without fast-math or numerical
parallelism. Converting exact rational inputs to float64 loses exact
arithmetic, and cancellation remains possible. Numba caches compiled kernels,
so distinguish first calls from warm calls. Tested with Python 3.12.14,
NumPy 2.5.3 and Numba 0.68.0 on Windows.

## Benchmarks

Use the **[4D–6D benchmark report](benchmarks/larger_4d_6d/RESULTS.md)** for the
current comparison. It has three columns: Numerical Chan (Numba), Chan d/3
(Python), and Chan d/2 (Python), on 25 nested spherical and simplex inputs.
It covers 64–1,024 points in 4D–6D and a 4,096-point 4D stress test. All
planned sizes remain visible, including timeouts and cases not run.

Numerical Chan (Numba) beat Chan d/3 (Python) on all 16 complete pairs, with a
median reference/numerical time ratio of 2.88. Chan d/2 (Python) was faster
on all 18 complete pairs. At 4,096 points in 4D, the numerical solver's
first call took 42.57 s versus 23.37 s for Chan d/3; neither completed the
warm protocol. These observations compare compiled and interpreted code,
exclude incomplete workers from ratios, and do not establish a universal
speedup or an improved complexity exponent.

[Reproduction and audit instructions](benchmarks/larger_4d_6d/README.md)
cover timings, saved inputs and source provenance. No historical data have
been deleted. The [full archived comparison](benchmarks/larger_4d_6d/RESULTS_ALL_VARIANTS.md)
retains the earlier compiled variant and diagnostic tables.

## Validate

From the repository root:

```sh
python NumericalChanHVND/verify_local_compiled.py
```

The exact standard-library engine remains available for integer and
`Fraction` arithmetic:

```python
from NumericalChanHVND.numerical_chan import hypervolume

assert hypervolume([(2, 1, 1, 1), (1, 2, 1, 1)]) == 3
```

Its [verification suite](verify_numerical_chan.py) checks small instances,
compression and repeated masks against independent exact calculations.
High-dimensional correctness tests are not large dense 10D performance
guarantees.

Magnitude motivated the algebraic viewpoint, but is not needed for ordinary
hypervolume. The `magnitude` function is available from the same modules.
The report's appendix explains its product measure and the transformations
between magnitude and hypervolume; zero-coordinate boxes must be retained
when computing magnitude.

## Earlier implementations and measurements

These remain available for reproduction; the current implementation and
comparison are linked above.

- [Earlier adaptive compiled design](ADAPTIVE_COMPILED.md) and
  [2D–10D measurements](benchmarks/compiled_2d_10d/RESULTS.md), using a
  different compression schedule.
- [Previous 3D–10D comparison](benchmarks/current_3d_10d/RESULTS.md).
- [Original numerical and hybrid comparison](benchmarks/RESULTS.md) and
  [original report](paper/numerical_chan_2d_10d.pdf).
