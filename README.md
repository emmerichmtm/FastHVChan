# FastHVChan

Exact hypervolume indicator computation via **Timothy M. Chan, *Klee's Measure
Problem Made Easy*, FOCS 2013** ([author's preprint](https://tmc.web.engr.illinois.edu/easyklee8_13.pdf)).

*Co-created by Michael Emmerich (University of Jyväskylä) and Claude Fable 5
(Anthropic).*

Two standalone, dependency-free Python modules:

| Module | Algorithm | Worst case (paper) | As implemented | Use it for |
|---|---|---|---|---|
| [`chan_hypervolume.py`](chan_hypervolume.py) | Section 2: simple divide & conquer for arbitrary boxes | O(n^(d/2)), d ≥ 3 | O(n^(d/2) log n) | **practical computation** (also solves general Klee's measure problem) |
| [`chan_orthant_dby3.py`](chan_orthant_dby3.py) | Section 4.2: orthant algorithm, arbitrary orientations (hypervolume = grounded case) | O(n^(d/3) polylog n), d ≥ 4 | same exponent, large constants | **reference implementation** of the asymptotically fastest known hypervolume algorithm |

Plus classical exact baselines in [`hv_baselines.py`](hv_baselines.py) (and C99
ports in `c/hv_baselines.c`), included for comparison and benchmarking only:

- `hypervolume_ds` — **dimension sweep**: the classical O(n log n) 3-D sweep
  as base case, organised after Guerreiro, Fonseca & Emmerich (CCCG 2012,
  pp. 77–82) in higher dimensions. This implementation recomputes exclusive
  contributions, giving O(n² log n) at d = 4 (the paper's specialised
  bookkeeping achieves O(n²)).
- `hypervolume_wfg` — the **WFG algorithm** of While, Bradstreet & Barone
  (IEEE Trans. Evolutionary Computation 16(1):86–95, 2012): recursive
  exclusive hypervolumes over dominance-pruned limit sets; exponential worst
  case, very fast in practice.
- a vendored **Yıldız–Suri-style** anchored solver
  ([`benchmarks/vendor/`](benchmarks/vendor/README.md)), used as a benchmark
  contender via the transform `q = r − y`.

The hypervolume indicator of a point set `Y` w.r.t. a reference point `r` is
the volume of the union of the boxes `[y, r]`, `y ∈ Y` — a special case of
Klee's measure problem in which every box is a *grounded orthant*. Chan's
paper names this special case explicitly and gives the current best bound for
it. To our knowledge this repository contains the first implementation of the
Section 4.2 algorithm.

## Usage

```python
from chan_hypervolume import hypervolume            # practical
from chan_orthant_dby3 import hypervolume_dby3      # reference, d >= 3

points = [(0.2, 0.7, 0.3), (0.5, 0.2, 0.6), (0.8, 0.4, 0.1)]
ref    = (1.0, 1.0, 1.0)     # minimization by default

hv  = hypervolume(points, ref)         # -> 0.31
hv3 = hypervolume_dby3(points, ref)    # identical value
```

Both accept `maximize=True` (mirrors all coordinates) and `prefilter=False`
(skip the O(n²d) dominance filter when the front is known non-dominated).
General box unions (arbitrary Klee's measure problem):

```python
from chan_hypervolume import union_volume
union_volume([((0, 0), (1, 1)), ((5, 5), (7, 8))])   # -> 7.0
```

Unions of orthants of **arbitrary orientation** (full Section 4.2, d ≥ 3) —
each orthant is a `(vertex, signs)` pair, `signs[k] = +1` for `x_k ≥ v_k`,
`-1` for `x_k ≤ v_k`, measured inside a domain box:

```python
from chan_orthant_dby3 import orthant_union_volume
orthants = [((0.5, 0.5, 0.5), (+1, -1, +1)), ((0.3, 0.7, 0.2), (-1, -1, +1))]
orthant_union_volume(orthants, lo=(0, 0, 0), hi=(1, 1, 1))
```

Requirements: Python ≥ 3.9, standard library only.

## How the algorithms work

**Section 2 (d/2).** k-d-tree style divide and conquer over a cell, with a
twist: before each cut, every box that degenerates to a *slab* of the cell is
eliminated by collapsing the covered coordinate intervals to zero length
(a monotone re-coordination that preserves the uncovered volume). Every
surviving box then has a (d−2)-face crossing the cell; cutting at the
*weighted median* of those faces (weight `2^((i+j)/d)`, cut axis cycling
through dimensions) makes the total face weight drop by `2^(2/d)` per level,
which solves to O(n^(d/2)) leaves.

**Section 4.2 (d/3).** For orthants the simplification is strengthened:
boxes spanning the cell in all but *two* dimensions are also eliminated, by
absorbing their pairwise union — a staircase — into the integrand as a
condition `x_j ≤ f(x_i)` / `x_j ≥ f(x_i)` with `f` a monotone step function
(one condition per orientation class per axis pair; there are four classes,
which are exactly the four monotone boundaries `f⁻, g⁻, f⁺, g⁺` of the
paper's Section 4.1).
The integrand becomes a "basic function" (sums of products of step-function
densities and monotone step conditions), which Chan shows is closed under
one-variable integration (Lemma 4.4) and can be periodically *compressed*
back to complexity O(#surviving boxes) by averaging over the grid of the
surviving boxes' coordinates (Lemma 4.5). Cutting at weighted medians of
(d−3)-faces then gives O(n^(d/3)) leaves. The implementation realizes the
full symbolic machinery (step-function algebra, exactly-once tie handling on
plateaus, grid-averaging compression with an adaptive trigger).

The accompanying paper ([`paper/hypervolume_chan.pdf`](paper/hypervolume_chan.pdf))
contains a complete walkthrough, a transparent complexity analysis of the code
as written — including exactly where and why it loses a log factor against the
paper's bound — and the validation/scaling experiments.

## Correctness

Run the built-in suites (they are also what `tests/` invokes):

```bash
python chan_hypervolume.py
python chan_orthant_dby3.py
```

Validation is layered: randomized property tests for the step-function
algebra; the symbolic integrator checked against exact grid enumeration;
compression checked for integral preservation and on-grid breakpoints; both
algorithms checked against O(2^n) inclusion–exclusion, against each other,
with compression forced on/off, on tie-heavy integer instances, and — for
the orthant module — on random mixed-orientation orthant unions in d = 3–5.
Worst observed disagreement: ~4·10⁻¹⁴ relative.

## Performance snapshot

Spherical fronts (all points mutually non-dominated), Python 3.13, one core,
`chan_hypervolume.py`:

| d | n | time |
|---|-----|--------|
| 3 | 500 | 0.13 s |
| 4 | 200 | 0.22 s |
| 5 | 100 | 0.47 s |
| 6 | 60  | 0.76 s |
| 7 | 40  | 1.2 s  |

`chan_orthant_dby3.py` builds a *smaller recursion tree* — the d/3 exponent is
directly visible in node counts (e.g. d=5, n=32: 393 nodes vs 1823 for the
d/2 algorithm; fitted node exponents 1.70 vs 2.18) — but pays more per node
for the symbolic bookkeeping, ending up ~2–7× slower at reachable sizes and
far slower on instances that force its compression machinery. Use it as an
executable specification, not as the fast path.

The full cross-implementation picture (all Python variants, the C port, and
the three baselines on byte-identical datasets up to d=10) is in
[`paper/benchmarks.pdf`](paper/benchmarks.pdf); all variants agree to ~10⁻¹⁴
relative on every measured case. Notable: the dimension sweep is the fastest
*Python* variant on every dataset measured, and at high d even rivals the C
port of Chan's algorithm — worst-case guarantees and practical speed pull in
different directions on these instance families.

## Repository layout

```
chan_hypervolume.py     Section-2 algorithm + hypervolume/union_volume API
chan_orthant_dby3.py    Section-4.2 algorithm (arbitrary orthants) + API
hv_baselines.py         dimension sweep + WFG (comparison baselines)
tests/                  pytest wrappers around the self-test suites
benchmarks/             scaling experiments; cross-implementation benchmark
                        (incl. vendored Yıldız–Suri contender)
paper/                  LaTeX/PDF: walkthrough, complexity analysis, benchmarks
c/                      C99 ports: Section-2 algorithm + both baselines
```

The C code in `c/` is a standalone C99 library (no dependencies beyond libm)
covering `chan_hypervolume.py` and `hv_baselines.py`. The Chan port is
validated against brute-force inclusion-exclusion and against randomised cases
generated by the Python implementation; the baseline ports carry their own
suite (`test_hv_baselines`, run by `make test`) cross-checking them against
inclusion-exclusion and the Chan port. See `c/README.md`.

## Attribution

This repository — both implementations, the test suites, the benchmarks, and
the accompanying paper — was co-created by **Michael Emmerich** (University of
Jyväskylä) and **Claude Fable 5** (Anthropic) in an interactive agentic coding
session, working directly from Chan's FOCS 2013 paper.

## Citing

If you use this code, cite Chan's paper:

```bibtex
@inproceedings{Chan2013KMP,
  author    = {Timothy M. Chan},
  title     = {Klee's Measure Problem Made Easy},
  booktitle = {54th Annual {IEEE} Symposium on Foundations of Computer Science ({FOCS})},
  pages     = {410--419},
  year      = {2013},
  doi       = {10.1109/FOCS.2013.51}
}
```

## License

MIT — see [LICENSE](LICENSE).
