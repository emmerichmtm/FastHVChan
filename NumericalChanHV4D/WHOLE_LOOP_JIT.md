# Whole-contraction JIT for numerical Chan in 4D

The new backend applies the successful strategy of `HV4DMagnitude`: enter
Numba once for a complete numerical operation, and perform its inner loops
without returning to Python. It compiles the new algorithm's term algebra;
it does not replace it with the older target-grid sweep.

```python
from numerical_chan4_compiled import hypervolume4

corners = [(1, 4, 3, 2), (2, 3, 4, 1),
           (3, 2, 1, 4), (4, 1, 2, 3)]
assert hypervolume4(corners) == 69.0
assert hypervolume4(corners, magnitude=True) == 765 / 16
```

Inputs describe unions of `[0, corner]` boxes. For minimization points `p`
and reference `r`, use corners `r-p`. This is a float64 backend; use
`numerical_chan4.py` with rational inputs when exact arithmetic is needed.
Keep `NumericalChanHV4D` and `NumericalChanHVND` together and install
`requirements.txt`. First use includes compilation or loading the disk cache.

## Three parts

1. **The geometric recursion remains Python.** It classifies orthants,
   absorbs easy constraints, chooses a weighted median, and recurses on the
   two children. Its compression schedule is unchanged.
2. **The adapter creates the native state once.** `_initial_terms` packs the
   initial term. `_apply_easy` applies the classified constraints; `_base`
   evaluates a leaf; `_compress` sums fine cells into blocks. Descendants keep
   native term collections, so calls do not repeatedly pack or unpack them.
3. **The entire contraction stays in Numba.** `compiled_terms.py` performs
   bound construction, winner enumeration, masks, prefix sums, signed term
   emission, normalization, merging, and successive variable elimination.
   Base evaluation also keeps its small inclusion-exclusion loop in Numba.

The shared geometric engine exposes small methods for those operations. Its
default methods retain the previous hybrid implementation, while the 4D
subclass replaces the numerical operations. The geometric recursion is shared.

## The term representation

A term is a named tuple:

```text
(coefficient, unary, edges)
unary[axis] = array of cell masses
edges[source, target, upper, direction] = array of integer cutoffs
```

These are native Numba dictionaries containing contiguous arrays. They retain
the reference implementation's sparse, explicit structure. A copied term gets
new dictionaries but shares its arrays. Every mask or cutoff update allocates
a replacement array, so sibling branches cannot change one another's state.
There is no dense multidimensional grid or pair table.

## One elimination, numerically

For the variable being removed, collect its lower and upper bounds. Partition
the remaining index space according to which lower bound wins and which upper
bound wins. Strict comparisons break ties in the original first-winner order.
The sum over that variable is the difference of two prefix sums. Emit the
resulting signed terms, then merge identical normalized states. Repeat for the
next variable. In a 4D compression, four old variables are eliminated from a
state with eight variables, leaving four block indices.

The implementation preserves upper and lower cutoffs in both monotonicity
directions, inclusive block endpoints, and signed coefficients. Magnitude's
origin atom is an ordinary stored cell mass and survives compression by summing
the cell masses. No integration or symbolic expression system is used.

## Merging and correctness

Merging uses a canonical hash and complete array comparison. Floating arrays
are compared by their bits, matching the old backend's byte-string keys.
Dictionary insertion order does not affect the key, but candidate-bound order
is preserved because it determines tie breaking.

Each lookup examines at most sixteen collision-chain entries. If it does not
find a matching state, the incoming term is retained separately. Optional
merging can be missed without discarding any contribution; this bounds hash
collision work independently of the number of terms. Cancelled entries are
removed only after emission, so chain indices remain valid.

The small-base evaluator rejects 63 or more hard boxes, where a native bitmask
would overflow. The normal threshold is two. No fastmath, parallel execution,
or object-mode fallback is enabled.

## What this changes

The algorithm, recursion, compression schedule and measure are unchanged.
The compilation boundary moves outward to enclose complete contractions.
Python still performs geometry and builds small mask/block metadata arrays;
it packs the initial state once. This is not a claim that the entire solver
is compiled. Compilation does not improve the
asymptotic exponent, and it does not remove the intermediate terms produced
by the scheduled compression.

Run `python verify_whole_loop.py` for exact small-instance oracles, signed
contraction checks, cellwise and repeated compression checks, and comparisons
with the independent hybrid implementation. The benchmark script and saved
measurements are in `benchmarks/whole_loop`.
