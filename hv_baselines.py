#!/usr/bin/env python3
"""Classical exact hypervolume baselines: dimension sweep and WFG.

These two algorithms complement the Chan (FOCS 2013) solvers in this
repository as comparison baselines:

* ``hypervolume_ds`` -- dimension sweep.  Points are processed in ascending
  order of the last objective; the volume is accumulated slab by slab while
  the (d-1)-dimensional volume of the swept front is maintained through
  exclusive contributions.  The d=3 base case is the classical O(n log n)
  sweep over a 2-D staircase; the d=4 organisation follows Guerreiro,
  Fonseca, and Emmerich, "A Fast Dimension-Sweep Algorithm for the
  Hypervolume Indicator in Four Dimensions", CCCG 2012, pp. 77-82.  Their
  specialised bookkeeping achieves O(n^2) in four dimensions; this
  implementation recomputes each exclusive contribution with a fresh
  (d-1)-dimensional computation, giving O(n^2 log n) for d=4 and
  O(n^{d-2} log n) in general -- same sweep structure, one log factor and
  some constant overhead away from the specialised bound.

* ``hypervolume_wfg`` -- the WFG algorithm of While, Bradstreet, and
  Barone, "A Fast Way of Calculating Exact Hypervolumes", IEEE Transactions
  on Evolutionary Computation 16(1):86-95, 2012.  Recursive
  exclusive-hypervolume computation over "limit sets": exponential worst
  case, but with dominance pruning it is very fast on practical fronts and
  dimensions.

Both share the API conventions of ``chan_hypervolume.hypervolume``
(minimisation by default, ``maximize=True`` mirrors, ``prefilter`` drops
dominated points first) and are exact.

Standard library only.  Run this file directly to execute its self-tests.

Co-created by Michael Emmerich (University of Jyvaskyla) and Claude Fable 5
(Anthropic).
"""

from __future__ import annotations

import itertools
import random
import sys
from bisect import bisect_left
from typing import Iterable, List, Sequence, Tuple

__all__ = ["hypervolume_ds", "hypervolume_wfg"]

Point = Tuple[float, ...]


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def _prepare(
    points: Iterable[Sequence[float]],
    reference: Sequence[float],
    maximize: bool,
    prefilter: bool,
) -> Tuple[List[Point], Point]:
    """Common front end: orientation, reference filtering, dominance filter."""
    sign = -1.0 if maximize else 1.0
    ref = tuple(sign * float(x) for x in reference)
    dim = len(ref)
    front: List[Point] = []
    for point in points:
        vector = tuple(sign * float(x) for x in point)
        if len(vector) != dim:
            raise ValueError("points and reference must have the same dimension")
        if all(x < r for x, r in zip(vector, ref)):
            front.append(vector)
    if prefilter:
        front = _nondominated(front)
    return front, ref


def _nondominated(points: Sequence[Point]) -> List[Point]:
    """Points not weakly dominated by another (minimisation; ties dropped)."""
    kept: List[Point] = []
    for point in sorted(points):
        if not any(all(a <= b for a, b in zip(other, point)) for other in kept):
            kept.append(point)
    return kept


def _box_volume(point: Point, ref: Point) -> float:
    volume = 1.0
    for a, b in zip(point, ref):
        volume *= b - a
    return volume


# --------------------------------------------------------------------------- #
# Dimension sweep
# --------------------------------------------------------------------------- #


def hypervolume_ds(
    points: Iterable[Sequence[float]],
    reference: Sequence[float],
    maximize: bool = False,
    prefilter: bool = True,
) -> float:
    """Hypervolume indicator by dimension sweep (see module docstring)."""
    front, ref = _prepare(points, reference, maximize, prefilter)
    if not front:
        return 0.0
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10_000))
    return _sweep(front, ref)


def _sweep(points: List[Point], ref: Point) -> float:
    """Exact hypervolume of a set of points strictly dominating ``ref``."""
    d = len(ref)
    if d == 1:
        return ref[0] - min(p[0] for p in points)
    if d == 2:
        return _hv2d(points, ref)
    if d == 3:
        return _hv3d(points, ref)

    # Sweep the last objective ascending; between successive sweep positions
    # the cross-section is constant, and its (d-1)-dimensional volume is
    # updated by the new point's exclusive contribution: its box volume minus
    # the volume already covered inside its box, which is a hypervolume of
    # the pointwise maxima with the previously swept points (Guerreiro,
    # Fonseca, Emmerich, CCCG 2012).
    pts = sorted(points, key=lambda p: p[-1])
    sub_ref = ref[:-1]
    swept: List[Point] = []
    cross_section = 0.0
    volume = 0.0
    for i, point in enumerate(pts):
        projection = point[:-1]
        clips = [
            tuple(max(a, b) for a, b in zip(projection, q)) for q in swept
        ]
        clips = [c for c in clips if all(x < r for x, r in zip(c, sub_ref))]
        exclusive = _box_volume(projection, sub_ref)
        if clips:
            exclusive -= _sweep(_nondominated(clips), sub_ref)
        cross_section += exclusive
        next_z = pts[i + 1][-1] if i + 1 < len(pts) else ref[-1]
        volume += cross_section * (next_z - point[-1])
        swept.append(projection)
    return volume


def _hv2d(points: Sequence[Point], ref: Point) -> float:
    """O(n log n) staircase area in two dimensions."""
    rx, ry = ref
    best = ry
    area = 0.0
    for x, y in sorted(points):
        if y < best:
            area += (rx - x) * (best - y)
            best = y
    return area


def _hv3d(points: Sequence[Point], ref: Point) -> float:
    """The classical O(n log n) 3-D dimension sweep over a 2-D staircase.

    Points are processed in ascending third coordinate; ``stairs`` holds the
    mutually non-dominated (x, y) projections sorted by x ascending (hence y
    descending), and the covered area is updated by each insertion's gain.
    (List insertion is O(n) worst case, so this Python version is O(n^2)
    worst case in list operations; the comparison structure is O(n log n).)
    """
    rx, ry, rz = ref
    pts = sorted(points, key=lambda p: p[2])
    stairs: List[Tuple[float, float]] = []
    xs: List[float] = []  # parallel array of stair x's, for bisect
    area = 0.0
    volume = 0.0
    for i, (x, y, z) in enumerate(pts):
        gain = _staircase_insert(stairs, xs, x, y, rx, ry)
        area += gain
        next_z = pts[i + 1][2] if i + 1 < len(pts) else rz
        volume += area * (next_z - z)
    return volume


def _staircase_insert(
    stairs: List[Tuple[float, float]],
    xs: List[float],
    x: float,
    y: float,
    rx: float,
    ry: float,
) -> float:
    """Insert (x, y) into the staircase; return the covered-area gain."""
    i = bisect_left(xs, x)
    cover = stairs[i - 1][1] if i > 0 else ry
    if y >= cover:
        return 0.0  # 2-D dominated: no new area
    gain = 0.0
    cur = x
    j = i
    while j < len(stairs) and stairs[j][1] >= y:
        nx = stairs[j][0]
        gain += (nx - cur) * (cover - y)
        cover = stairs[j][1]
        cur = nx
        j += 1
    nx = stairs[j][0] if j < len(stairs) else rx
    gain += (nx - cur) * (cover - y)
    del stairs[i:j]
    del xs[i:j]
    stairs.insert(i, (x, y))
    xs.insert(i, x)
    return gain


# --------------------------------------------------------------------------- #
# WFG
# --------------------------------------------------------------------------- #


def hypervolume_wfg(
    points: Iterable[Sequence[float]],
    reference: Sequence[float],
    maximize: bool = False,
    prefilter: bool = True,
) -> float:
    """Hypervolume indicator by the WFG algorithm (see module docstring)."""
    front, ref = _prepare(points, reference, maximize, prefilter)
    if not front:
        return 0.0
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 100_000))
    # Sorting by the last objective tends to make limit sets collapse fast.
    return _wfg(sorted(front, key=lambda p: p[-1]), ref)


def _wfg(front: List[Point], ref: Point) -> float:
    """Sum of exclusive hypervolumes, each via a recursive limit set."""
    total = 0.0
    for i, point in enumerate(front):
        total += _exclusive(point, front[i + 1 :], ref)
    return total


def _exclusive(point: Point, rest: List[Point], ref: Point) -> float:
    inclusive = _box_volume(point, ref)
    if not rest:
        return inclusive
    limit = [tuple(max(a, b) for a, b in zip(point, q)) for q in rest]
    limit = [c for c in limit if all(x < r for x, r in zip(c, ref))]
    if not limit:
        return inclusive
    return inclusive - _wfg(_nondominated(limit), ref)


# --------------------------------------------------------------------------- #
# Self-tests
# --------------------------------------------------------------------------- #


def _reference_hypervolume(points: Sequence[Point], reference: Point) -> float:
    """O(2^n) inclusion-exclusion, for tests only."""
    total = 0.0
    pts = [p for p in points if all(a < b for a, b in zip(p, reference))]
    for size in range(1, len(pts) + 1):
        for subset in itertools.combinations(pts, size):
            corner = [max(p[k] for p in subset) for k in range(len(reference))]
            volume = 1.0
            for a, b in zip(corner, reference):
                volume *= max(0.0, b - a)
            total += volume if size % 2 else -volume
    return total


def _check(name: str, got: float, expected: float) -> None:
    error = abs(got - expected) / max(1.0, abs(expected))
    status = "ok " if error <= 1e-9 else "FAIL"
    print(f"  [{status}] {name:<46} got {got:14.8f}  expected {expected:14.8f}")
    if status == "FAIL":
        raise AssertionError(f"{name}: {got} != {expected}")


def _self_test() -> None:
    print("hv_baselines self-tests")
    algorithms = [("dimension sweep", hypervolume_ds), ("WFG", hypervolume_wfg)]

    for label, algo in algorithms:
        _check(f"{label}: 1 point, d=3", algo([(0, 0, 0)], (1, 2, 3)), 6.0)
        _check(f"{label}: point on reference", algo([(1, 2)], (1, 2)), 0.0)
        _check(f"{label}: no points", algo([], (1, 1, 1)), 0.0)
        _check(f"{label}: duplicates", algo([(0.5, 0.5)] * 4, (1, 1)), 0.25)
        _check(f"{label}: L shape d=2", algo([(0, 1), (1, 0)], (2, 2)), 3.0)
        _check(
            f"{label}: maximisation", algo([(2, 2)], (0, 0), maximize=True), 4.0
        )

    rng = random.Random(20120808)  # CCCG 2012 opening day
    for label, algo in algorithms:
        for d in range(1, 7):
            for n in (1, 5, 10):
                ref = tuple(1.0 for _ in range(d))
                pts = [tuple(rng.random() for _ in range(d)) for _ in range(n)]
                _check(
                    f"{label}: random d={d}, n={n}",
                    algo(pts, ref),
                    _reference_hypervolume(pts, ref),
                )
        for d in (2, 3, 4, 5):  # tie-heavy integer grids
            ref = tuple(4.0 for _ in range(d))
            pts = [
                tuple(float(rng.randrange(0, 4)) for _ in range(d))
                for _ in range(10)
            ]
            _check(
                f"{label}: integer grid d={d}",
                algo(pts, ref),
                _reference_hypervolume(pts, ref),
            )

    try:
        from chan_hypervolume import hypervolume as hv_chan
    except ImportError:
        print("  [-- ] chan_hypervolume not importable, cross-check skipped")
    else:
        for d, n in ((3, 60), (4, 40), (5, 25), (6, 15)):
            ref = tuple(1.2 for _ in range(d))
            pts = []
            while len(pts) < n:  # spherical front: nothing dominated
                v = [abs(rng.gauss(0.0, 1.0)) + 1e-12 for _ in range(d)]
                s = sum(x * x for x in v) ** 0.5
                pts.append(tuple(x / s for x in v))
            want = hv_chan(pts, ref)
            for label, algo in algorithms:
                _check(f"{label}: sphere d={d}, n={n} vs Chan", algo(pts, ref), want)

    print("All self-tests passed.")


if __name__ == "__main__":
    _self_test()
