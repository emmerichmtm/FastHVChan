# Adaptive compiled numerical hypervolume, 2D-10D

The new `numerical_chan_compiled.py` keeps finite cell masses and moves complete
state operations into Numba. In 4D-10D its geometric driver follows the
simplification and adaptive-compression approach of this repository's
`chan_orthant_dby3.py`. In 2D and 3D it uses the existing compiled sweeps.
Compression is enabled by default. Ordinary HV and dominated-set L1 magnitude
use the same implementation.

## Use

From the repository root, after installing `NumericalChanHVND/requirements.txt`:

```python
from NumericalChanHVND.numerical_chan_compiled import hypervolume, magnitude

corners = [(1, 4, 3, 2), (2, 3, 4, 1), (3, 2, 1, 4), (4, 1, 2, 3)]
volume = hypervolume(corners, dimension=4)
mag = magnitude(corners, dimension=4)
assert abs(volume - 69) < 1e-9
assert abs(mag - 765 / 16) < 1e-9
```

Corners are nonnegative upper extents of origin-anchored boxes. Convert
minimization points `p` and reference `r` to `r-p`. Dimension is inferred from
nonempty input, or specified explicitly; every integer dimension 2 through 10
is supported. Empty input returns zero. Zero extents remain meaningful for
magnitude. The backend converts coordinates to float64, with no fastmath.
Use the original numerical Python module with `Fraction` inputs for exact
rational arithmetic.

For counters or compression settings:

```python
from NumericalChanHVND.numerical_chan_compiled import NumericalChanCompiled

solver = NumericalChanCompiled(dimension=7)  # use_compression=True
answer = solver.compute(points_7d, magnitude=False)
print(solver.stats)
```

`NumericalChanHV4D/numerical_chan4_adaptive.py` supplies a fixed-4D
`hypervolume4` convenience function. The earlier scheduled backends remain
available under their existing names for comparison and reproducibility.

## What changed

The earlier hybrid Numba backend compiles short array helpers while Python
creates, copies, normalizes and merges signed terms. That arrangement is often
slower than tuples and ordinary Python on short arrays. The new kernel retains
native typed term states throughout the recursion and compiles the complete
contractions, clipping, pair updates, merging and compression.

The larger performance issue was the representation carried through the tree.
The original numerical solver compressed whenever its local block expired,
even if the current state was small. Compression lifts `d` fine indices to
`2d` fine/block indices; elimination can create many signed terms. Those terms
then make later leaf calculations expensive. Its arrays also retained ranks
outside the current cell, represented by zero masks.

The new driver applies three changes together:

1. A slab tightens the cell's lower bound. Boxes are reclassified after the
   bounds change.
2. The native state is cropped to the surviving cell. Constraints that are
   constant, impossible or redundant on nonzero support are simplified, and
   equal states are merged. Simplification also runs after elimination.
3. Compression requires both enough recursion levels and enough stored state.
   The default level interval is `max(1, round(0.1*d*log2(max(n,2))))`.
   Compression then occurs when the stored array entries plus term count
   exceed `8*d*d*(hard_boxes+2)`.

The third rule follows the two-part trigger in the Chan d/3 reference. Its
size metric is deliberately explicit: the numerical engine counts array
entries, while the reference counts step-function pieces. Their compression
decisions need not coincide. Weighted-median cuts also use the numerical
engine's integer approximations to the face weights.

## Recursion and numerical preservation

```text
Node(boxes, state, cell, axis, levels_since_compression):
    discard disjoint boxes; tighten the cell using slabs
    crop and simplify the state; absorb pair constraints
    if the uncovered state is empty: return 0
    if at most two hard boxes remain: return compiled_base_sum(...)
    if both compression tests hold:
        state = compiled_block_sums(state, hard_box_grid)
        reindex the hard boxes; reset levels_since_compression
    choose a cyclic weighted-median cut
    return Node(left cell, ...) + Node(right cell, ...)
```

The node value is uncovered mass; the public result subtracts it from the
enclosing product mass. A term is a coefficient times unary cell masses times
monotone integer-cutoff predicates. Eliminating one variable evaluates a
prefix-sum difference on each disjoint first-winner region. There is no
quadrature or computer-algebra operation.

All grid endpoints are inclusive. Cropping from indices `lo` through `hi`
retains those masses, slices each cutoff on its source axis and subtracts the
target offset. A slab covering ranks through `k` leaves ranks starting at
`k+1`. These integer conventions preserve the atom of mass one at the origin
used for magnitude; cropping does not create a new atom at the new index zero.
Positive interval masses are their lengths for HV and half their lengths for
magnitude.

Compression partitions each fine index interval into disjoint consecutive
blocks, constrains every fine index to its selected block, and sums out the
fine indices. Thus the result preserves every block-aligned rectangle sum,
including signed intermediate states and the origin atom. It remains the
same finite-mass compression operation used by the scheduled implementation.

This document establishes the implementation's operations and validation,
not a new asymptotic bound. The earlier proof uses a particular local block
schedule. Transferring its bound to this adaptive gate needs a separate proof;
an observed speedup does not supply that proof.

## Verification and timings

Run `python NumericalChanHVND/verify_compiled.py`. It checks exact rational
inclusion-exclusion, signed native grids, boundary atoms, all cutoff
orientations, state immutability, affine magnitude equivalence and cached
imports. Forced cases exercise actual compression and repeated generations,
including 20-variable lifts in 10D. Forced settings are validation tools,
separate from the default performance measurements.

The [structured compression probe](benchmarks/compiled_2d_10d/NATURAL_COMPRESSION.md)
uses the unchanged defaults on 1,552 four-dimensional boxes. It executes
six compressions and agrees with both Chan implementations. The small default
benchmark cases execute none, illustrating why compression checks and actual
compressions are reported separately.

The [2D-10D benchmark](benchmarks/compiled_2d_10d/RESULTS.md) includes saved
inputs, all repetitions, first-use costs, algorithm counters and source hashes.
It compares complete implementations: the gains include simplification and
compression-policy changes as well as compilation. The reference Chan d/3
code's antiderivatives are themselves finite sums and linear arithmetic; the
speed difference is not a distinction between quadrature and finite sums.
