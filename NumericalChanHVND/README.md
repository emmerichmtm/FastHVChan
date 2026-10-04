# NumericalChanHVND: dimensions 2 through 10

Standalone numerical hypervolume and dominated-set L1 magnitude engines.
Every integer dimension from 2 to 10 is supported, including 5, 7, and 9.
The two original Chan implementations at the repository root are unchanged.

The [full research report (PDF)](paper/numerical_product_chan.pdf) and
[LaTeX source](paper/numerical_product_chan.tex) explain magnitude, both HV
transformations, finite contractions, and a proof retaining Chan's
`O(n^(d/3) polylog(n))` arithmetic bound. The new
[`numerical_chan_local_compiled.py`](numerical_chan_local_compiled.py)
implements its local checkpoint policy, including compression skips only
when the stored state is already small. Use this version when referring to
that complexity theorem; the archived adaptive benchmark below measures
a different policy. Run `python NumericalChanHVND/verify_local_compiled.py`
from the repository root to validate it.

```python
from NumericalChanHVND.numerical_chan_local_compiled import hypervolume, magnitude
```

The new [adaptive compiled backend](ADAPTIVE_COMPILED.md) supports 2D-10D,
including complete native contractions, cell pruning and enabled adaptive
compression in 4D-10D. Use `numerical_chan_compiled.hypervolume` for this version.
Its [benchmarks](benchmarks/compiled_2d_10d/RESULTS.md) compare it with the earlier
implementations. The schedule-specific bounds below describe the original
locally scheduled engine; this adaptive implementation has a separate policy.

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

For the latest implementation, see [adaptive compiled use and design](ADAPTIVE_COMPILED.md).
The interface described in this section is the earlier hybrid backend.

For 4D, the [dedicated entry point](../NumericalChanHV4D/README.md) now compiles
whole contractions and keeps term states native across recursion. The generic
ND interface below retains the hybrid backend used as its comparison.

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
- [Adaptive compiled 2D-10D comparison](benchmarks/compiled_2d_10d/RESULTS.md):
  complete native contractions in all dimensions, with compression enabled.
- [Previous 3D-10D comparison](benchmarks/current_3d_10d/RESULTS.md):
  two sizes and two input families in every dimension, plus direct magnitude;
  uses the new whole-contraction backend in 4D and labels the other backends.
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

The [adaptive compiled 2D-10D run](benchmarks/compiled_2d_10d/RESULTS.md)
records 48 inputs and 187 workers. All 48 compiled runs completed and agreed
with the available references. On ordinary-HV inputs in 4D-10D, it beat Python
Chan d/3 on 26/28 pairs (median speedup 2.02x), and original numerical Python
on 26/27 completed pairs (median 6.29x). Python Chan d/2 was faster on all 28
of those inputs. Timings use three warm repetitions and show substantial
variability; these small inputs do not establish an asymptotic exponent.

Compression is enabled. The main default benchmark inputs stay below its
size threshold; forced cases exercise it separately. A larger
[structured default-settings probe](benchmarks/compiled_2d_10d/NATURAL_COMPRESSION.md)
executes six compressions on 1,552 boxes and agrees with both Chan references.
That probe is a correctness and execution check, not a warm speed comparison.
The [new verification suite](verify_compiled.py) also checks repeated
compression and both HV and magnitude in all supported dimensions.

### Previous 3D-10D comparison

The [previous 3D-10D run](benchmarks/current_3d_10d/RESULTS.md) records 40 inputs
and 165 comparisons: 162 complete, three hybrid timeouts, no value mismatches.
An independent exact rational oracle also checks 31 inputs. Across four
ordinary-HV cases each, the 3D compiled sweep has median speedup 1.20x over
numerical Python, and the 4D whole-contraction backend has median speedup
5.12x over the previous hybrid. In 5D-10D, dimension-specific median hybrid
times are 1.51-2.52x slower than numerical Python on completed pairs.
At the time of that run, whole contractions were compiled only in 4D.

### Earlier hybrid comparison

On 20 completed matched ordinary-HV cases in dimensions 4-10, the hybrid
Numba backend was a median 1.93x slower than numerical Python and 21.25x
slower than the existing Python Chan Section-4.2 implementation. On the
3D spherical n=128 case, the compiled sweep took 0.64 ms versus 1.77 ms
for numerical Python. These are observed sample results, not general
speed guarantees. Eleven of 156 benchmark workers timed out.

See the full methodology and raw data in [benchmarks/RESULTS.md](benchmarks/RESULTS.md).
