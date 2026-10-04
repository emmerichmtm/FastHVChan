# Why the compiled 4D prefix implementation is faster

4 October 2026. This follow-up tests **only 4D** and leaves all solvers unchanged.

The user's recollection is confirmed. The compiled prefix solver in
`HV4DMagnitude` is **1.59-2.11x faster than the Python Chan d/3 implementation**
on the four spherical cases with 20 or 40 points. At 20 points it is
**84.59-96.99x faster than the new default hybrid Numba engine**.
The earlier numerical-Chan comparison omitted this compiled prefix solver;
it therefore did not establish that the earlier Numba approach was ineffective.

## Measurements

Times below are milliseconds, median of three warm end-to-end calls. The labels
give family, seed / n. `budget` means the **whole worker** exceeded 35 seconds,
including startup, imports, first call and repeats; it is not a 35-second
per-call lower bound. No speed ratio is inferred from a timeout.

| Family seed / n | Old JIT | Chan d/3 | New JIT | New no-comp. | Chan d/2 |
| --- | --- | --- | --- | --- | --- |
| Sphere 42 / 10 | 5.295 | 7.519 | 9.259 | 10.972 | 1.515 |
| Sphere 42 / 20 | 21.521 | 35.635 | 2087.270 | 45.190 | 5.167 |
| Sphere 42 / 40 | 62.343 | 111.593 | budget | 142.064 | 19.298 |
| Sphere 99 / 20 | 21.924 | 34.805 | 1854.674 | 37.848 | 5.983 |
| Sphere 99 / 40 | 52.861 | 111.451 | budget | 157.691 | 21.045 |
| Grid 7 / 20 | 0.834 | 2.225 | 5.549 | 5.907 | 0.323 |
| Uniform 17 / 20 | 14.546 | 18.634 | 832.086 | 29.710 | 2.405 |

Old JIT = `PrefixHV4D(fast=True)`. New JIT = the shared numerical Numba engine
at d=4, which is exactly the backend of `NumericalChanHV4D/numerical_chan4_numba.py`.
No-compression = that new engine with `block_levels=10**6`, a diagnostic override
with no claimed worst-case guarantee. Both Chan reference implementations run
without Numba. **Chan d/2 is fastest on all seven measured inputs.** These small
inputs do not rank asymptotic exponents or establish a crossover.

Additional Python measurements isolate compilation more carefully:

| Family seed / n | Old flat Python ms | Old object Python ms | Flat/JIT ratio | New Python ms |
| --- | --- | --- | --- | --- |
| Sphere 42 / 10 | 69.091 | 69.840 | 13.05 | 8.843 |
| Sphere 42 / 20 | 255.417 | 250.038 | 11.87 | 1291.028 |
| Sphere 42 / 40 | 804.383 | 785.652 | 12.90 | budget |
| Sphere 99 / 20 | 339.409 | 330.800 | 15.48 | 1147.316 |
| Sphere 99 / 40 | 695.438 | 715.985 | 13.16 | budget |
| Grid 7 / 20 | 1.596 | 2.260 | 1.91 | 3.424 |
| Uniform 17 / 20 | 139.630 | 129.447 | 9.60 | 500.031 |

Old flat Python disables JIT before importing the **same flat kernel**.
Old object Python uses `fast=False`, selecting a different decomposed evaluator.
New Python is `numerical_chan.NumericalChan(dimension=4)`; it is the counterpart
of the shared Numba engine, rather than the separate `numerical_chan4.py` file.
All timed numeric inputs are floats, including the exact-capable Python engine.

## Why it is faster

1. **The complete hot loop is compiled.** `fastkernel.k_prefix` runs the whole
   nested target-plane scan, binary searches, source-cap calculations, two-branch
   area formula and accumulation inside Numba. Packed numeric arrays replace
   Python work at every target cell. Disabling JIT for precisely this flat
   implementation makes the spherical n=20-40 cases 11.87-15.48x slower.
2. **The new default compression policy creates expensive term states.** In
   seed 42 / n=20, default and no-compression variants both make 61 recursive
   calls and 30 cuts. Default makes ten compressions, emits 10,809 raw terms,
   and reaches 625 intermediate terms. No-compression emits 709 raw terms and
   peaks at eleven. It reduces the measured time from 2,087.270 to 45.190 ms.
   This is a 46.19x improvement from the policy and its downstream effects.
3. **Small JIT helpers leave Python overhead in the new engine.** Winner
   enumeration, copying term dictionaries, constructing masks, normalizing
   states and hashing array-byte keys remain under Python control. On seed
   42 / n=20, a separate profile records 3,285,965 visible calls for new default,
   82,860 without compression, and 40,833 for old JIT. Native work inside Numba
   is not counted as individual Python calls, so these are overhead diagnostics,
   not arithmetic operation counts.

The default new profile spends about 15.3% cumulatively in `push_blocks` and
78.7% in base-case evaluation: most compression-related expense persists into
later evaluations. It would be inaccurate to attribute the entire slowdown to
time directly inside the compression routine. The existing Python Chan d/3 code
also gates compression on representation size; it performs none on seed 42 / n=20.
The ablation disables the new engine's scheduled compression; it does not
implement the reference solver's adaptive size trigger.

The old solver also shrinks cells after absorbing slabs, whereas the new engine
zeros unary entries. This is another structural difference, but this experiment
does not separately measure its contribution. Counters from different engines
need not have identical definitions. In particular, new node counts include
axis-skip calls and its peak-term statistic includes elimination temporaries.

## What the 2+2 implementation actually does

The old solver represents the easy constraints by six pairwise staircases.
For a fixed target cell in coordinates 3 and 4, it computes bounds A and B on
coordinates 1 and 2. The remaining source-plane area is obtained from stored
prefix records. The numerical sum has the shape

$$K=\sum_{u,v} w_3(u)w_4(v)\,\mathbf{1}_{v<f_{34}(u)}H_{12}(A(u,v),B(u,v)).$$

Here H12 is the capped source-plane area, evaluated using the staircase f12;
weights are target-cell widths for ordinary HV. This compact formula is useful
because the complete summation can stay in compiled code. It still explicitly
visits target-grid cells for each prefix query. On seed 42 / n=20 there are
81 box evaluations, each using 16 signed prefix queries, hence 1,296 queries.

The old solver retains live orthants and does not implement the new compression
schedule. Its current code and these timings **do not prove a complete
O(n^(4/3) polylog n) bound**. Neither an explicit target-cell scan nor a faster
finite benchmark supplies the missing global work bound. The old repository
itself marks its overall complexity claim as unverified.

## Correctness and magnitude scope

All 52 completed benchmark rows agree with the separate Chan d/2 evaluator;
the maximum scaled disagreement is 7.89e-16. The n=10 case also agrees with an
independent exact rational inclusion-exclusion calculation (maximum scaled
error 2.22e-16). Four new default runs time out: Python and Numba for each n=40
spherical case. This is empirical validation, not a numerical error theorem.

The old public solver explicitly uses Lebesgue measure. A separate direct
magnitude check finds an existing atom-handling bug in its packed kernel.
With all six staircases equal to 2 and every grid `[0,1]`, the query `(1,1,1,1)`
should yield `(1+1/2)^4 = 5.0625`, but `K_fast` returns 2.25. The decomposed
and brute evaluators return 5.0625. For `(1,1,0,0)`, expected magnitude is 2.25
but `K_fast` returns zero. Ordinary-Lebesgue versions of both checks pass.

The target-grid loop substitutes an anchor atom for the first interval instead
of retaining both. This issue is separate from ordinary-HV performance. Magnitude
can already use the correct ordinary-HV route through the affine identity

$$\operatorname{Mag}(D(A))=\operatorname{HV}_4(\{1+a/2:a\in A\}),\qquad A\subseteq\mathbb R_{\ge0}^4.$$

We check this route against exact inclusion-exclusion on two single-box cases
and a four-box union. It returns 81/16, 9/4, and 765/16 as expected.
The direct packed-kernel defect is documented here; no solver source was changed.

## Reproducibility

Machine: AMD Ryzen 7 5800H, 8 physical / 16 logical cores, about 16 GiB RAM;
Windows 11, CPython 3.12.14, NumPy 2.5.3, Numba 0.68.0. Algorithms run sequentially,
with no Numba parallelism or fastmath. Each algorithm/case uses a fresh worker;
one first call precedes three measured calls. The first call may load a disk
cache and is **not** a pure compilation-time measurement. Imports and first use
are excluded from the warm table; packing, preprocessing, coordinate conversion
and evaluation are included. No dominance prefilter is applied.

All algorithms receive the same saved point sets. Old prefix and reference Chan
use nonnegative minimization points p with reference r; the new engine uses
anchored extents r-p. Do not pass negative reflected coordinates to the old
prefix API: its zero-anchored grids omit negative cuts. Sphere cases use absolute
Gaussian vectors normalized to unit Euclidean length, with r=(1.2,...,1.2).
Seed 42 consumes sizes 5,10,20,40 in sequence, discarding n=5; seed 99 consumes
20,40. Grid and uniform cases have r=(1,...,1). The grid uses integers 0,...,3
divided by four. Profile runs are separate from timing runs.

The historical audit's exact timing datasets were not archived, so this is a
fresh comparison on the committed test family, not an exact rerun of its old table.
No confidence intervals or large-n scaling claims are made from three repeats.

Pinned sources:

- [HV4DMagnitude at 0c668e7](https://github.com/emmerichmtm/HV4DMagnitude/tree/0c668e76520201f41ac0f144caa8d468d6c84e86)
- [FastHVChan reference at 08539aa](https://github.com/emmerichmtm/FastHVChan/tree/08539aa99e94e181d7230a8288885b3294621c9e)
- New numerical engine: FastHVChan commit `97012f7`, shared files in `NumericalChanHVND`.

Run from the FastHVChan repository root after checking out the pinned older
repository at `../HV4DMagnitude` and installing the ND requirements:

```sh
python NumericalChanHV4D/benchmarks/prefix4/compare_prefix4.py --old ../HV4DMagnitude --new NumericalChanHVND --reference . --out prefix4-reproduction --profiles
python NumericalChanHV4D/benchmarks/prefix4/check_prefix4_magnitude.py --old ../HV4DMagnitude --out magnitude-checks.json
```

Files alongside this report include all inputs, raw repeats, first calls,
profiles, counters, source hashes, derived ratios, and the magnitude reproducer.

## Implementation direction justified by this audit

For practical 4D work, retain the old compiled 2+2 evaluator as the performance
baseline. A new implementation should compile a complete contraction loop and
control the growth of term states before expecting a benefit from Numba.
The large no-compression gain motivates investigating an adaptive policy, but
simply disabling the proved schedule is not an established n^(4/3) algorithm.
These are follow-up targets, not changes made or bounds proved in this audit.
