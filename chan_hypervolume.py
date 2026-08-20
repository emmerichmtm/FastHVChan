#!/usr/bin/env python3
"""Exact hypervolume / Klee's measure via Chan's "Klee's Measure Problem Made Easy".

Klee's measure problem: given ``n`` axis-parallel boxes in ``R^d``, compute the
volume of their union.  The *hypervolume indicator* of multi-objective
optimisation is the special case in which every box is the orthant ``[y_i, r]``
spanned by a solution point ``y_i`` and a reference point ``r``; the hypervolume
is then exactly the volume of the union of those orthants.

The algorithm
-------------
Timothy M. Chan, *Klee's Measure Problem Made Easy*, FOCS 2013.
https://tmc.web.engr.illinois.edu/easyklee8_13.pdf

Chan computes the measure of the **complement** of the union inside a box
domain (a "cell").  The recursion is a k-d-tree style divide and conquer::

    measure(B, cell):
        0. if |B| is tiny, answer directly (inclusion-exclusion)
        1. SIMPLIFY B
        2. CUT the cell into two subcells with an axis-parallel hyperplane
        3. return measure(B|left, left) + measure(B|right, right)

**Simplify** (step 1) is the twist that makes the analysis work.  A box that,
restricted to the cell, degenerates into a *slab* ``{a <= x_i <= b}`` -- i.e. it
spans the whole cell along every axis except ``i`` -- is eliminated by
*collapsing* that slab to zero thickness: the ``x_i`` coordinates of the cell and
of every box are pushed through a monotone map that deletes the covered
intervals.  Everything inside such a slab is covered by the union, so the
measure of the complement is unchanged.  Afterwards every surviving box fails to
span the cell along at least two axes, i.e. it has a ``(d-2)``-face crossing the
cell.

**Cut** (step 2) gives a ``(d-2)``-face orthogonal to axes ``i`` and ``j`` the
weight ``2^((i+j)/d)`` and cuts the cell at the *weighted median* of the
coordinates of the ``(d-2)``-faces orthogonal to axis 1.  The axis numbering is
then rotated, ``(1, 2, ..., d) -> (d, 1, ..., d-1)``, which just means the cut
axis cycles through the dimensions, k-d-tree style.  The rotation is what makes
the total weight ``N`` of the surviving ``(d-2)``-faces drop by a factor
``2^(2/d)`` in *each* subcell, so::

    T(N) <= 2 T(N / 2^(2/d)) + O(N)   =>   T(N) = O(N^(d/2)),

giving ``O(n^(d/2))`` time for constant ``d >= 3`` (and ``O(n log n)`` for
``d = 2``).  This straightforward transcription re-sorts at every node instead of
maintaining pre-sorted lists, which costs an extra logarithmic factor.

Public API
----------
``hypervolume(points, reference)`` -- hypervolume indicator of a point set.
``union_volume(boxes)``            -- volume of a union of arbitrary boxes.

Standard library only.  Run this file directly to execute its self-tests.

Co-created by Michael Emmerich (University of Jyvaskyla) and Claude Fable 5
(Anthropic).
"""

from __future__ import annotations

import itertools
import random
import time
from bisect import bisect_right
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

__all__ = ["Box", "ChanMeasure", "hypervolume", "union_volume", "nondominated"]

Point = Tuple[float, ...]
Interval = Tuple[float, float]


# --------------------------------------------------------------------------- #
# Geometry primitives
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Box:
    """An axis-parallel box ``[lo_0, hi_0] x ... x [lo_{d-1}, hi_{d-1}]``."""

    lo: Point
    hi: Point

    @property
    def dim(self) -> int:
        return len(self.lo)

    def volume(self) -> float:
        volume = 1.0
        for a, b in zip(self.lo, self.hi):
            if b <= a:
                return 0.0
            volume *= b - a
        return volume

    def clip_to(self, other: "Box") -> Optional["Box"]:
        """This box intersected with ``other``; ``None`` if the overlap is flat."""
        lo: List[float] = []
        hi: List[float] = []
        for a, b, c, d in zip(self.lo, self.hi, other.lo, other.hi):
            low = a if a > c else c
            high = b if b < d else d
            if high <= low:
                return None
            lo.append(low)
            hi.append(high)
        return Box(tuple(lo), tuple(hi))

    def split(self, axis: int, value: float) -> Tuple["Box", "Box"]:
        """Cut the box with the hyperplane ``x_axis = value``."""
        left_hi = list(self.hi)
        left_hi[axis] = value
        right_lo = list(self.lo)
        right_lo[axis] = value
        return Box(self.lo, tuple(left_hi)), Box(tuple(right_lo), self.hi)

    def map_axis(self, axis: int, transform: Callable[[float], float]) -> "Box":
        """Push both coordinates along ``axis`` through a monotone map."""
        lo = list(self.lo)
        hi = list(self.hi)
        lo[axis] = transform(lo[axis])
        hi[axis] = transform(hi[axis])
        return Box(tuple(lo), tuple(hi))

    def spans(self, cell: "Box", axis: int) -> bool:
        """True if the box covers ``cell`` along every axis other than ``axis``."""
        for k in range(len(self.lo)):
            if k != axis and (self.lo[k] > cell.lo[k] or self.hi[k] < cell.hi[k]):
                return False
        return True


def _merge_intervals(intervals: Iterable[Interval]) -> List[Interval]:
    """Union of 1-d intervals, as a sorted list of disjoint intervals."""
    merged: List[Interval] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def _collapse_map(covered: Sequence[Interval]) -> Callable[[float], float]:
    """Monotone map of the real line that shrinks ``covered`` to zero length.

    ``covered`` must be sorted and disjoint.  The returned map is the identity
    below the first interval and subtracts the total covered length seen so far
    everywhere else, so distances outside ``covered`` are preserved exactly.
    """
    starts = [start for start, _ in covered]
    removed_before: List[float] = []
    total = 0.0
    for start, end in covered:
        removed_before.append(total)
        total += end - start

    def collapse(x: float) -> float:
        index = bisect_right(starts, x) - 1
        if index < 0:
            return x
        start, end = covered[index]
        inside = (end if end < x else x) - start
        return x - removed_before[index] - inside

    return collapse


def _complement_by_inclusion_exclusion(boxes: Sequence[Box], cell: Box) -> float:
    """Measure of ``cell`` minus the union of ``boxes`` (already clipped to it).

    Exponential in ``len(boxes)`` -- only used as the base case of the recursion,
    for a constant number of boxes.
    """
    measure = cell.volume()
    for size in range(1, len(boxes) + 1):
        sign = -1.0 if size % 2 else 1.0
        for combination in itertools.combinations(boxes, size):
            common: Optional[Box] = combination[0]
            for box in combination[1:]:
                common = common.clip_to(box)
                if common is None:
                    break
            if common is not None:
                measure += sign * common.volume()
    return measure


# --------------------------------------------------------------------------- #
# Chan's divide and conquer
# --------------------------------------------------------------------------- #


class ChanMeasure:
    """Chan's recursion for the measure of the complement of a union of boxes.

    Instances are cheap; ``node_count`` reports how many recursive calls the last
    run took, which is handy when experimenting with ``base_case``.
    """

    def __init__(self, dim: int, base_case: int = 2) -> None:
        if dim < 1:
            raise ValueError("dimension must be at least 1")
        self.dim = dim
        self.base_case = max(0, base_case)
        self.node_count = 0

    # -- entry point ------------------------------------------------------- #

    def complement_measure(self, boxes: Sequence[Box], cell: Box) -> float:
        """Volume of ``cell`` not covered by any box of ``boxes``."""
        self.node_count = 0
        return self._measure(self._clip(boxes, cell), cell, depth=0)

    # -- the three steps of the paper -------------------------------------- #

    def _measure(self, boxes: List[Box], cell: Box, depth: int) -> float:
        """``boxes`` must already be clipped to ``cell`` and have positive volume."""
        self.node_count += 1

        # Step 0: trivial instances.
        if not boxes:
            return cell.volume()
        if len(boxes) <= self.base_case:
            return _complement_by_inclusion_exclusion(boxes, cell)

        # Step 1: simplify, i.e. collapse away every slab-shaped box.
        boxes, cell = self._simplify(boxes, cell)
        if not boxes:
            return cell.volume()

        # Step 2: cut at the weighted median of the (d-2)-faces of the cut axis.
        axis, depth, cuts = self._choose_cut(boxes, cell, depth)
        left, right = cell.split(axis, _weighted_median(cuts))

        # Step 3: recurse.  The paper's axis renumbering is the ``depth + 1``
        # below: the cut axis cycles through the dimensions.
        return self._measure(self._clip(boxes, left), left, depth + 1) + self._measure(
            self._clip(boxes, right), right, depth + 1
        )

    def _clip(self, boxes: Iterable[Box], cell: Box) -> List[Box]:
        clipped = (box.clip_to(cell) for box in boxes)
        return [box for box in clipped if box is not None]

    def _simplify(self, boxes: List[Box], cell: Box) -> Tuple[List[Box], Box]:
        """Remove every box that is a slab of ``cell`` by collapsing that slab.

        A box spanning the cell along all axes but ``axis`` covers a full slice of
        the cell, so the uncovered volume is unchanged if we shrink that slice to
        zero thickness.  Collapsing can turn a *surviving* box into a slab of a
        previously handled axis, so the sweep is repeated until it is stable.
        Afterwards every remaining box has a ``(d-2)``-face crossing the cell.
        """
        while boxes:
            collapsed_any = False
            for axis in range(self.dim):
                slabs: List[Box] = []
                rest: List[Box] = []
                for box in boxes:
                    (slabs if box.spans(cell, axis) else rest).append(box)
                if not slabs:
                    continue
                collapsed_any = True
                collapse = _collapse_map(
                    _merge_intervals((box.lo[axis], box.hi[axis]) for box in slabs)
                )
                cell = cell.map_axis(axis, collapse)
                boxes = [box.map_axis(axis, collapse) for box in rest]
                boxes = [box for box in boxes if box.lo[axis] < box.hi[axis]]
                if not boxes:
                    break
            if not collapsed_any:
                break
        return boxes, cell

    def _choose_cut(
        self, boxes: List[Box], cell: Box, depth: int
    ) -> Tuple[int, int, List[Tuple[float, float]]]:
        """Pick the cut axis (cycling) and collect its weighted cut candidates.

        An axis may carry no ``(d-2)``-face at all, in which case there is nothing
        to cut and we simply rotate the numbering again.  Since every simplified
        box has a ``(d-2)``-face crossing the cell, some axis always has one.
        """
        for _ in range(self.dim):
            axis = depth % self.dim
            cuts = self._cut_candidates(boxes, cell, axis, depth)
            if cuts:
                return axis, depth, cuts
            depth += 1
        raise AssertionError("simplified boxes must have a (d-2)-face in the cell")

    def _cut_candidates(
        self, boxes: List[Box], cell: Box, axis: int, depth: int
    ) -> List[Tuple[float, float]]:
        """Coordinates and weights of the ``(d-2)``-faces orthogonal to ``axis``.

        After ``depth`` cuts the paper's axis renumbering puts the original axis
        ``k`` at position ``((k - depth) mod d) + 1``, and the axis being cut sits
        at position 1.  A face orthogonal to positions ``i`` and ``j`` weighs
        ``2^((i+j)/d)``, so a face orthogonal to the cut axis and to the axis at
        position ``p`` weighs ``2^((1+p)/d)``.
        """
        dim = self.dim
        weights = [2.0 ** ((1 + (k - depth) % dim + 1) / dim) for k in range(dim)]

        cuts: List[Tuple[float, float]] = []
        for box in boxes:
            coordinates = [
                x
                for x in (box.lo[axis], box.hi[axis])
                if cell.lo[axis] < x < cell.hi[axis]
            ]
            if not coordinates:
                continue
            # Weight of the faces this box contributes at those coordinates: one
            # face per boundary of the box that crosses the cell along some other
            # axis.  A box with no such boundary is a slab and is already gone.
            weight = 0.0
            for k in range(dim):
                if k == axis:
                    continue
                crossings = (box.lo[k] > cell.lo[k]) + (box.hi[k] < cell.hi[k])
                if crossings:
                    weight += crossings * weights[k]
            if weight:
                cuts.extend((x, weight) for x in coordinates)
        return cuts


def _weighted_median(cuts: List[Tuple[float, float]]) -> float:
    """Smallest coordinate whose weight prefix reaches half of the total weight.

    Both open subcells then keep at most half of the weight: the faces sitting
    exactly on the cutting hyperplane belong to neither of them.
    """
    cuts.sort()
    half = sum(weight for _, weight in cuts) / 2.0
    accumulated = 0.0
    for coordinate, weight in cuts:
        accumulated += weight
        if accumulated >= half:
            return coordinate
    return cuts[-1][0]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def union_volume(
    boxes: Iterable[Sequence[Sequence[float]]], domain: Optional[Box] = None
) -> float:
    """Volume of the union of axis-parallel boxes (Klee's measure problem).

    Args:
        boxes: iterable of ``(lo, hi)`` pairs of ``d``-dimensional corners.
        domain: optional box the volume is restricted to; defaults to the
            bounding box of the input, which restricts nothing.

    Returns:
        The volume of the union, ``0.0`` for an empty or degenerate input.
    """
    normalised = [
        Box(tuple(float(x) for x in lo), tuple(float(x) for x in hi))
        for lo, hi in boxes
    ]
    normalised = [box for box in normalised if box.volume() > 0.0]
    if not normalised:
        return 0.0

    dim = normalised[0].dim
    if any(box.dim != dim for box in normalised):
        raise ValueError("all boxes must have the same dimension")

    if domain is None:
        domain = Box(
            tuple(min(box.lo[k] for box in normalised) for k in range(dim)),
            tuple(max(box.hi[k] for box in normalised) for k in range(dim)),
        )
    return domain.volume() - ChanMeasure(dim).complement_measure(normalised, domain)


def hypervolume(
    points: Iterable[Sequence[float]],
    reference: Sequence[float],
    maximize: bool = False,
    prefilter: bool = True,
) -> float:
    """Hypervolume indicator of ``points`` with respect to ``reference``.

    By default objectives are **minimised**: a point contributes the box between
    itself and the reference point, and only points strictly better than the
    reference in every objective contribute anything.

    Args:
        points: iterable of ``d``-dimensional objective vectors.
        reference: the reference point, dominated by (i.e. worse than) the points.
        maximize: set to ``True`` if larger objective values are better.
        prefilter: drop dominated points first.  This never changes the result and
            usually shrinks the input a lot; it costs ``O(n^2 d)`` in the worst
            case, so turn it off if the front is known to be non-dominated.

    Returns:
        The hypervolume, ``0.0`` if no point dominates the reference point.
    """
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
    if not front:
        return 0.0
    if prefilter:
        front = nondominated(front)

    boxes = [Box(point, ref) for point in front]
    domain = Box(tuple(min(p[k] for p in front) for k in range(dim)), ref)
    return domain.volume() - ChanMeasure(dim).complement_measure(boxes, domain)


def nondominated(points: Sequence[Point]) -> List[Point]:
    """Points not weakly dominated by another point (minimisation; ties dropped).

    Lexicographic order guarantees that any dominator of a point comes before it,
    so a single forward scan suffices.
    """
    kept: List[Point] = []
    for point in sorted(points):
        if not any(all(a <= b for a, b in zip(other, point)) for other in kept):
            kept.append(point)
    return kept


# --------------------------------------------------------------------------- #
# Self-tests and demo
# --------------------------------------------------------------------------- #


def _reference_union_volume(boxes: Sequence[Box]) -> float:
    """Independent O(2^n) inclusion-exclusion baseline, for tests only."""
    total = 0.0
    for size in range(1, len(boxes) + 1):
        sign = 1.0 if size % 2 else -1.0
        for combination in itertools.combinations(boxes, size):
            common: Optional[Box] = combination[0]
            for box in combination[1:]:
                common = common.clip_to(box)
                if common is None:
                    break
            if common is not None:
                total += sign * common.volume()
    return total


def _reference_hypervolume(points: Sequence[Point], reference: Point) -> float:
    boxes = [Box(tuple(point), tuple(reference)) for point in points]
    return _reference_union_volume([box for box in boxes if box.volume() > 0.0])


def _check(name: str, got: float, expected: float, tolerance: float = 1e-9) -> None:
    error = abs(got - expected)
    scale = max(1.0, abs(expected))
    status = "ok " if error <= tolerance * scale else "FAIL"
    print(f"  [{status}] {name:<40} got {got:14.8f}  expected {expected:14.8f}")
    if status == "FAIL":
        raise AssertionError(f"{name}: {got} != {expected}")


def _self_test() -> None:
    print("Hand-checked cases")
    _check("1 point, d=3", hypervolume([(0, 0, 0)], (1, 2, 3)), 6.0)
    _check("point on the reference", hypervolume([(1, 2, 3)], (1, 2, 3)), 0.0)
    _check("point behind the reference", hypervolume([(4, 4)], (1, 1)), 0.0)
    _check("no points", hypervolume([], (1, 1, 1)), 0.0)
    _check("duplicates collapse", hypervolume([(0, 0)] * 5, (2, 2)), 4.0)
    _check("dominated point ignored", hypervolume([(0, 0), (1, 1)], (2, 2)), 4.0)
    _check("two 2-d points, L shape", hypervolume([(0, 1), (1, 0)], (2, 2)), 3.0)
    _check("maximisation mirrors", hypervolume([(2, 2)], (0, 0), maximize=True), 4.0)
    _check(
        "disjoint boxes",
        union_volume([((0, 0), (1, 1)), ((5, 5), (7, 8))]),
        1.0 + 2.0 * 3.0,
    )
    _check(
        "nested boxes",
        union_volume([((0, 0, 0), (4, 4, 4)), ((1, 1, 1), (2, 2, 2))]),
        64.0,
    )

    rng = random.Random(20130810)

    print("\nRandom point sets vs. inclusion-exclusion")
    for dim in range(2, 6):
        for count in (1, 2, 5, 9):
            reference = tuple(1.0 for _ in range(dim))
            points = [tuple(rng.random() for _ in range(dim)) for _ in range(count)]
            _check(
                f"d={dim}, n={count}",
                hypervolume(points, reference),
                _reference_hypervolume(points, reference),
            )

    print("\nSpherical fronts (nothing is dominated) vs. inclusion-exclusion")
    for dim in range(2, 7):
        reference = tuple(1.5 for _ in range(dim))
        points = _spherical_front(10, dim, rng)
        _check(
            f"d={dim}, n=10 on the unit sphere",
            hypervolume(points, reference),
            _reference_hypervolume(points, reference),
        )

    print("\nCoarse integer grids (many ties) vs. inclusion-exclusion")
    for dim in range(2, 6):
        reference = tuple(4.0 for _ in range(dim))
        points = [
            tuple(float(rng.randrange(0, 4)) for _ in range(dim)) for _ in range(8)
        ]
        _check(
            f"d={dim}, n=8, integer coordinates",
            hypervolume(points, reference),
            _reference_hypervolume(points, reference),
        )

    print("\nRandom general boxes vs. inclusion-exclusion")
    for dim in range(2, 6):
        for count in (3, 7, 10):
            boxes = []
            for _ in range(count):
                lo = tuple(rng.random() for _ in range(dim))
                hi = tuple(x + rng.random() for x in lo)
                boxes.append((lo, hi))
            _check(
                f"d={dim}, n={count} boxes",
                union_volume(boxes),
                _reference_union_volume([Box(lo, hi) for lo, hi in boxes]),
            )

    print("\nAll tests passed.")


def _spherical_front(count: int, dim: int, rng: random.Random) -> List[Point]:
    """``count`` points on the positive unit sphere: no point dominates another.

    This is the standard hard case for hypervolume algorithms -- nothing can be
    pruned away before the real work starts.
    """
    front = []
    for _ in range(count):
        vector = [abs(rng.gauss(0.0, 1.0)) + 1e-12 for _ in range(dim)]
        norm = sum(x * x for x in vector) ** 0.5
        front.append(tuple(x / norm for x in vector))
    return front


def _demo() -> None:
    print("\nTiming on spherical fronts (all points mutually non-dominated)")
    print(f"  {'d':>2}  {'n':>5}  {'nodes':>8}  {'seconds':>8}  hypervolume")
    rng = random.Random(1)
    for dim, count in ((3, 500), (4, 200), (5, 100), (6, 60), (7, 40)):
        reference = tuple(1.5 for _ in range(dim))
        front = _spherical_front(count, dim, rng)
        solver = ChanMeasure(dim)
        boxes = [Box(point, reference) for point in front]
        domain = Box(tuple(min(p[k] for p in front) for k in range(dim)), reference)
        start = time.perf_counter()
        value = domain.volume() - solver.complement_measure(boxes, domain)
        elapsed = time.perf_counter() - start
        print(
            f"  {dim:>2}  {count:>5}  {solver.node_count:>8}  "
            f"{elapsed:>8.3f}  {value:.8f}"
        )


if __name__ == "__main__":
    _self_test()
    _demo()
