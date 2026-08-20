#!/usr/bin/env python3
"""Chan's O(n^(d/3) polylog n) orthant algorithm for the hypervolume indicator.

This module implements the algorithm of Section 4.2 of

    Timothy M. Chan, *Klee's Measure Problem Made Easy*, FOCS 2013,
    https://tmc.web.engr.illinois.edu/easyklee8_13.pdf

for *orthants of arbitrary orientation*: boxes with a single finite vertex,
``{x : x_k >= v_k or x_k <= v_k per axis}``, clipped to a domain box.  The
hypervolume indicator problem is the special case in which every orthant is
*grounded* (all constraints point the same way): each solution point spans a
box against the reference point.  The paper names this special case
explicitly.

The companion module ``chan_hypervolume.py`` implements the paper's simple
Section-2 algorithm, which runs in O(n^(d/2)) time for arbitrary boxes and is
the *practical* choice.  The algorithm here has a better exponent, d/3, but --
as expected for this line of theoretical work -- its constant factors are
large; it is a reference implementation whose value is fidelity to the paper,
not raw speed.

How it works
------------
Chan's recursion computes the measure of the complement of the union inside a
cell.  Section 2 eliminates boxes that degenerate to *slabs* (boxes spanning
the cell on all but one axis).  Section 4 additionally eliminates *2-sided
orthants* (boxes spanning the cell on all but two axes), so that every
surviving box has a (d-3)-face crossing the cell; cutting at weighted medians
of (d-3)-faces then yields O(N^(d/3)) leaves instead of O(N^(d/2)).

Absorbing 2-sided orthants is where the machinery lives.  The integrand is no
longer the constant 1 but a "basic function" (Definition 4.2)

    F(x) = sum of terms   coef * [E(x)] * h_1(x_1) * ... * h_d(x_d),

where each density ``h_i`` is a univariate step function and the predicate
``E`` is a conjunction of conditions ``x_j <= f(x_i)`` / ``x_j >= f(x_i)``
with *monotone* step functions ``f``.  For a pair of axes (i, j), 2-sided
orthants come in four orientation classes; the union of each class is a
staircase whose complement is a single monotone step condition (decreasing
for the ``(>=,>=)`` and ``(<=,<=)`` classes, increasing for the mixed ones,
with the sense of the inequality matching the x_j direction) -- so absorbing
a class multiplies one condition into every term, and conditions of the same
class merge by pointwise min/max.  These four classes are exactly the four
monotone functions ``f^-, g^-, f^+, g^+`` of the paper's Section 4.1.

Two symbolic operations make the recursion work:

* **Integration (Lemma 4.4).**  A basic function can be integrated over one
  axis by rewriting: every condition touching the axis becomes a lower or
  upper bound on the integration variable; a case split over which bound is
  the *active* one (first-minimizer-wins, so ties are counted exactly once)
  reduces the integral to ``H(U) - H(V)`` with ``H`` the antiderivative of the
  density -- and ``H`` composed with a step-function bound is again a step
  function, so the result is again a sum of basic-function terms.  Applying
  this d times integrates F over a box; this is the recursion's base case.

* **Compression (Lemma 4.5).**  Absorbed staircases make F's complexity grow,
  so F is periodically replaced by its average over the grid spanned by the
  coordinates of the *surviving* boxes:

      F~(x) = (integral of F over the grid cell containing x) / (cell volume).

  F~ integrates to the same value as F over every grid cell, hence over the
  complement of the union of the surviving boxes (which is grid-aligned), and
  every function in F~ has breakpoints only on the grid -- complexity O(|B|).
  F~ is computed with the same symbolic integration engine, integrating d
  fresh variables between the step-function bounds pred_i(x_i)/succ_i(x_i).

The recursion cuts cells at the weighted median of the (d-3)-face coordinates
(face orthogonal to axes i, j, k gets weight 2^((i+j+k)/d), axes renumbered
cyclically after each cut), giving the recurrence of Section 4.2:

    T(N) <= O(r^d) T(N/r^3) + O(r^d N)   =>   T(N) = O(N^(d/3) polylog N)

for d >= 4, when compression is applied every ~log2(r^d) levels.

Everything is measure-theoretically sloppy on measure-zero sets (values *at*
step-function breakpoints), which is harmless because the only consumer of
every predicate is a Lebesgue integral.  The one place where sloppiness is
*not* allowed -- preimages of plateaus under step functions, which have
positive measure -- is handled exactly by computing ray preimages from piece
values (``_ray_boundary``) instead of point evaluation.

Public API
----------
``hypervolume_dby3(points, reference)``       -- hypervolume indicator, exact.
``orthant_union_volume(orthants, lo, hi)``    -- union of arbitrary orthants.

Standard library only.  Run the file directly for its self-tests.
"""

from __future__ import annotations

import itertools
import math
import random
import sys
from bisect import bisect_left, bisect_right
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "hypervolume_dby3",
    "orthant_union_volume",
    "ChanOrthantMeasure",
    "Step",
    "Term",
]

INF = math.inf
Point = Tuple[float, ...]

# Condition senses: LE means "x_j <= f(x_i)", GE means "x_j >= f(x_i)".
LE, GE = "le", "ge"


# --------------------------------------------------------------------------- #
# Step functions
# --------------------------------------------------------------------------- #


class Step:
    """A piecewise-constant function on the real line.

    ``xs`` are the strictly increasing breakpoints and ``vs`` the values, with
    ``len(vs) == len(xs) + 1``: ``vs[t]`` is the value on the open interval
    ``(xs[t-1], xs[t])``.  Values may be ``+-inf``.  The value *at* a
    breakpoint is deliberately unspecified -- all uses are almost-everywhere.
    """

    __slots__ = ("xs", "vs")

    def __init__(self, xs: Sequence[float], vs: Sequence[float]) -> None:
        if len(vs) != len(xs) + 1:
            raise ValueError("need len(vs) == len(xs) + 1")
        # Canonicalize: drop breakpoints between equal values, so that equal
        # functions compare equal and term merging works.
        cxs: List[float] = []
        cvs: List[float] = [vs[0]]
        for x, v in zip(xs, vs[1:]):
            if v != cvs[-1]:
                cxs.append(x)
                cvs.append(v)
        self.xs = tuple(cxs)
        self.vs = tuple(cvs)

    # -- constructors ------------------------------------------------------- #

    @staticmethod
    def const(value: float) -> "Step":
        return Step((), (value,))

    # -- basics ------------------------------------------------------------- #

    def key(self) -> Tuple:
        """Hashable identity, used for term merging and bound deduplication."""
        return (self.xs, self.vs)

    def is_const(self) -> bool:
        return not self.xs

    def __call__(self, x: float) -> float:
        """Point evaluation (only safe almost everywhere -- tests, medians)."""
        return self.vs[bisect_left(self.xs, x)]

    def monotone_dir(self) -> Optional[int]:
        """+1 nondecreasing, -1 nonincreasing, 0 constant, None otherwise."""
        up = all(a <= b for a, b in zip(self.vs, self.vs[1:]))
        down = all(a >= b for a, b in zip(self.vs, self.vs[1:]))
        if up and down:
            return 0
        if up:
            return 1
        if down:
            return -1
        return None

    def all_zero(self) -> bool:
        return all(v == 0.0 for v in self.vs)

    # -- pointwise combination --------------------------------------------- #

    def _zip(self, other: "Step") -> Tuple[Tuple[float, ...], List[float], List[float]]:
        """Common refinement: merged breakpoints and both value sequences."""
        xs = tuple(sorted(set(self.xs) | set(other.xs)))
        va: List[float] = []
        vb: List[float] = []
        ia = ib = 0
        for t in range(len(xs) + 1):
            va.append(self.vs[ia])
            vb.append(other.vs[ib])
            if t < len(xs):
                if ia < len(self.xs) and self.xs[ia] == xs[t]:
                    ia += 1
                if ib < len(other.xs) and other.xs[ib] == xs[t]:
                    ib += 1
        return xs, va, vb

    def binary(self, other: "Step", op) -> "Step":
        xs, va, vb = self._zip(other)
        return Step(xs, [op(a, b) for a, b in zip(va, vb)])

    def minimum(self, other: "Step") -> "Step":
        return self.binary(other, min)

    def maximum(self, other: "Step") -> "Step":
        return self.binary(other, max)

    def times(self, other: "Step") -> "Step":
        def mul(a: float, b: float) -> float:
            if a == 0.0 or b == 0.0:  # convention 0 * inf = 0 (indicators)
                return 0.0
            return a * b

        return self.binary(other, mul)

    # -- comparisons yielding indicator step functions ---------------------- #

    def compare_const(self, c: float, op: str) -> "Step":
        """Indicator step of {x : self(x) op c}, op in '<', '<=', '>', '>='."""
        test = _CMP[op]
        return Step(self.xs, [1.0 if test(v, c) else 0.0 for v in self.vs])

    def compare(self, other: "Step", op: str) -> "Step":
        """Indicator step of {x : self(x) op other(x)} (same variable)."""
        test = _CMP[op]
        xs, va, vb = self._zip(other)
        return Step(xs, [1.0 if test(a, b) else 0.0 for a, b in zip(va, vb)])

    # -- clipping ----------------------------------------------------------- #

    def clamp(self, lo: float, hi: float) -> "Step":
        return Step(self.xs, [min(max(v, lo), hi) for v in self.vs])


_CMP = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
}


def _ray_boundary(f: Step, op: str, y: float) -> Tuple[str, float]:
    """The ray {t : f(t) op y} for a monotone step function ``f``, computed
    exactly from piece values (never by point evaluation, so plateaus -- which
    have positive-measure preimages -- land on the correct side).

    Returns ``(side, x)``: side ``"left"`` means the set is ``(-inf, x)`` and
    ``"right"`` means ``(x, +inf)``, always up to the measure-zero boundary
    point.  Full and empty sets are encoded with infinite ``x``.
    """
    test = _CMP[op]
    mask = [test(v, y) for v in f.vs]
    # The side depends only on f's direction and the comparison direction, so
    # that full/empty rays are encoded consistently across different y.
    direction = f.monotone_dir()
    if direction is None:
        raise AssertionError("non-monotone step passed to _ray_boundary")
    up = direction >= 0  # treat constants as nondecreasing
    side = "right" if up == (op in (">", ">=")) else "left"
    if side == "left":
        # Monotone f => mask is a contiguous prefix of the pieces.
        if all(mask):
            return ("left", INF)
        if not mask[0]:
            return ("left", -INF)
        last_true = mask.index(False) - 1
        if any(mask[last_true + 1 :]):
            raise AssertionError("non-monotone step passed to _ray_boundary")
        return ("left", f.xs[last_true])
    # Monotone f => mask is a contiguous suffix of the pieces.
    if all(mask):
        return ("right", -INF)
    if not mask[-1]:
        return ("right", INF)
    first_true = mask.index(True)
    if not all(mask[first_true:]):
        raise AssertionError("non-monotone step passed to _ray_boundary")
    return ("right", f.xs[first_true - 1])


# --------------------------------------------------------------------------- #
# Antiderivatives (continuous piecewise-linear functions)
# --------------------------------------------------------------------------- #


class _Antiderivative:
    """``H(x) = integral of h from lo to x`` on the clip range ``[lo, hi]``,
    for a step-function density ``h``.  Evaluation clamps into the range, so
    infinite bound values are safe once clipped by the caller.
    """

    __slots__ = ("xs", "ys", "slopes")

    def __init__(self, h: Step, lo: float, hi: float) -> None:
        xs = [lo] + [x for x in h.xs if lo < x < hi] + [hi]
        ys = [0.0]
        slopes = []
        for a, b in zip(xs, xs[1:]):
            slope = h((a + b) / 2.0)
            slopes.append(slope)
            ys.append(ys[-1] + slope * (b - a))
        self.xs, self.ys, self.slopes = xs, ys, slopes

    def __call__(self, x: float) -> float:
        x = min(max(x, self.xs[0]), self.xs[-1])
        t = max(0, min(len(self.slopes) - 1, bisect_right(self.xs, x) - 1))
        return self.ys[t] + self.slopes[t] * (x - self.xs[t])

    def compose(self, g: Step) -> Step:
        """H o g -- a step function again, evaluated exactly per piece."""
        return Step(g.xs, [self(v) for v in g.vs])


# --------------------------------------------------------------------------- #
# Bounds and comparisons between bounds
# --------------------------------------------------------------------------- #
#
# A *bound* on the integration variable is either a constant ("c", value) or a
# monotone step function of some other axis ("f", axis, Step).  The case split
# in the integration engine needs indicators like [bound_A <= bound_B]; the
# result is either trivially true/false, an indicator step folded into one
# axis's density, or a new binary condition between two axes.

BoundC = Tuple[str, float]
BoundF = Tuple[str, int, Step]
Bound = Tuple  # BoundC | BoundF

TRUE, FALSE = "true", "false"


def _bound_key(b: Bound) -> Tuple:
    return (b[0], b[1]) if b[0] == "c" else (b[0], b[1], b[2].key())


def _less_than(a: Bound, b: Bound, strict: bool):
    """The indicator of ``[a < b]`` (or ``[a <= b]`` if not strict).

    Returns one of:
      * ``"true"`` / ``"false"``           -- identically 1 / 0,
      * ``("dens", axis, indicator_step)`` -- a univariate indicator,
      * ``("cond", i, j, f, sense)``       -- the condition x_j sense f(x_i).
    """
    op = "<" if strict else "<="
    if a[0] == "c" and b[0] == "c":
        return TRUE if _CMP[op](a[1], b[1]) else FALSE
    if a[0] == "f" and b[0] == "c":  # [g(x) op c]
        return ("dens", a[1], a[2].compare_const(b[1], op))
    if a[0] == "c" and b[0] == "f":  # [c op g(x)]  <=>  [g(x) flipped-op c]
        flipped = ">" if strict else ">="
        return ("dens", b[1], b[2].compare_const(a[1], flipped))
    if a[1] == b[1]:  # same axis: pointwise comparison
        return ("dens", a[1], a[2].compare(b[2], op))
    # Cross-axis: [g(x_a) op g'(x_b)]  <=>  x_b in {t : g'(t) flipped-op v}
    # evaluated at v = g(x_a); the ray boundary as a function of x_a is a step
    # function theta with g's breakpoints.
    g, gp = a[2], b[2]
    if gp.is_const():
        return ("dens", a[1], g.compare_const(gp.vs[0], op))
    if g.is_const():
        flipped = ">" if strict else ">="
        return ("dens", b[1], gp.compare_const(g.vs[0], flipped))
    flipped = ">" if strict else ">="
    sides_vals = [_ray_boundary(gp, flipped, v) for v in g.vs]
    side = sides_vals[0][0]
    if any(s != side for s, _ in sides_vals):
        raise AssertionError("inconsistent ray sides for monotone bound")
    theta = Step(g.xs, [x for _, x in sides_vals])
    sense = GE if side == "right" else LE
    return ("cond", a[1], b[1], theta, sense)


# --------------------------------------------------------------------------- #
# Terms of a basic function
# --------------------------------------------------------------------------- #


class Term:
    """One term ``coef * prod_i h_i(x_i) * prod [x_j sense f(x_i)]``.

    ``dens`` maps an axis to its density step function (absent axis = density
    1).  ``conds`` maps a key ``(i, j, sense, bucket)`` to the monotone step
    function ``f`` of the condition ``x_j sense f(x_i)``; the bucket (+1
    nondecreasing / -1 nonincreasing) exists because two conditions of the
    same sense merge into one via pointwise min/max only when that preserves
    monotonicity.  Terms are immutable in style: operations return copies.
    """

    __slots__ = ("coef", "dens", "conds")

    def __init__(
        self,
        coef: float = 1.0,
        dens: Optional[Dict[int, Step]] = None,
        conds: Optional[Dict[Tuple[int, int, str, int], Step]] = None,
    ) -> None:
        self.coef = coef
        self.dens = dens or {}
        self.conds = conds or {}

    def clone(self) -> "Term":
        return Term(self.coef, dict(self.dens), dict(self.conds))

    def key(self) -> Tuple:
        return (
            tuple(sorted((a, s.key()) for a, s in self.dens.items())),
            tuple(sorted((k, f.key()) for k, f in self.conds.items())),
        )

    def is_dead(self) -> bool:
        """True if the term is identically zero almost everywhere."""
        if self.coef == 0.0:
            return True
        if any(s.all_zero() for s in self.dens.values()):
            return True
        for (i, j, sense, _), f in self.conds.items():
            if sense == LE and all(v == -INF for v in f.vs):
                return True
            if sense == GE and all(v == INF for v in f.vs):
                return True
        return False

    def mul_dens(self, axis: int, step: Step) -> None:
        self.dens[axis] = self.dens[axis].times(step) if axis in self.dens else step

    def add_cond(self, i: int, j: int, f: Step, sense: str) -> None:
        """Multiply the condition ``[x_j sense f(x_i)]`` into the term."""
        if i == j:
            raise AssertionError("conditions must relate two distinct axes")
        if f.is_const():  # univariate: fold as an indicator into x_j's density
            op = "<=" if sense == LE else ">="
            self.mul_dens(j, _identity_compare(f.vs[0], op))
            return
        direction = f.monotone_dir()
        if direction is None:
            raise AssertionError("condition function must be monotone")
        # Merge into an existing compatible bucket if possible (a constant-ish
        # direction 0 fits either bucket).
        buckets = (1, -1) if direction >= 0 else (-1, 1)
        for bucket in buckets:
            if direction != 0 and bucket != direction:
                continue
            key = (i, j, sense, bucket)
            if key in self.conds:
                old = self.conds[key]
                self.conds[key] = old.minimum(f) if sense == LE else old.maximum(f)
                return
        bucket = direction if direction != 0 else 1
        self.conds[(i, j, sense, bucket)] = f

    def apply(self, outcome) -> bool:
        """Multiply a ``_less_than`` outcome into the term.

        Returns False if the term became identically zero (caller drops it).
        """
        if outcome == FALSE:
            return False
        if outcome == TRUE:
            return True
        if outcome[0] == "dens":
            self.mul_dens(outcome[1], outcome[2])
        else:
            _, i, j, f, sense = outcome
            self.add_cond(i, j, f, sense)
        return True


def _identity_compare(c: float, op: str) -> Step:
    """Indicator step of {x : x op c}."""
    if op in ("<", "<="):
        return Step((c,), (1.0, 0.0))
    return Step((c,), (0.0, 1.0))


def _restrict_step(f: Step, a: float, b: float) -> Step:
    """Drop breakpoints outside the open interval (a, b); the result agrees
    with ``f`` on (a, b) and is constant beyond it.
    """
    first = bisect_right(f.xs, a)
    last = bisect_left(f.xs, b)
    return Step(f.xs[first:last], f.vs[first : last + 1])


def _prune_term(term: Term, range_of) -> bool:
    """Restrict a term's functions to the box given by ``range_of(axis)`` and
    simplify: conditions that are trivially true inside the box are removed,
    trivially false ones kill the term, out-of-range condition values clamp to
    +-inf (which merges adjacent staircase steps), unit densities disappear.

    Works in place; returns False if the term is identically zero on the box.
    This keeps representation sizes proportional to what is visible in the
    current cell -- without it the case splits of the integration engine blow
    up combinatorially.
    """
    for axis in list(term.dens):
        a, b = range_of(axis)
        step = _restrict_step(term.dens[axis], a, b)
        if step.all_zero():
            return False
        if step.is_const() and step.vs[0] == 1.0:
            del term.dens[axis]
        else:
            term.dens[axis] = step
    for key in list(term.conds):
        i, j, sense, _ = key
        ai, bi = range_of(i)
        aj, bj = range_of(j)
        f = _restrict_step(term.conds[key], ai, bi)
        if sense == LE:  # [x_j <= f(x_i)] with x_j in (aj, bj)
            vals = [INF if v >= bj else (-INF if v <= aj else v) for v in f.vs]
        else:  # [x_j >= f(x_i)]
            vals = [-INF if v <= aj else (INF if v >= bj else v) for v in f.vs]
        f = Step(f.xs, vals)
        trivial = INF if sense == LE else -INF
        impossible = -INF if sense == LE else INF
        if all(v == trivial for v in f.vs):
            del term.conds[key]
            continue
        if all(v == impossible for v in f.vs):
            return False
        term.conds[key] = f
    return True


def _prune_terms(terms: List[Term], range_of) -> List[Term]:
    return _merge_terms([t for t in terms if _prune_term(t, range_of)])


def _terms_complexity(terms: List[Term]) -> int:
    """Total number of pieces across all functions of all terms."""
    total = 0
    for term in terms:
        total += 1
        for step in term.dens.values():
            total += len(step.xs) + 1
        for f in term.conds.values():
            total += len(f.xs) + 1
    return total


def _merge_terms(terms: List[Term]) -> List[Term]:
    """Sum coefficients of structurally identical terms; drop dead terms."""
    table: Dict[Tuple, Term] = {}
    for term in terms:
        if term.is_dead():
            continue
        key = term.key()
        if key in table:
            table[key].coef += term.coef
        else:
            table[key] = term
    # merging only changed coefficients, so a plain coefficient check suffices
    return [t for t in table.values() if t.coef != 0.0]


# --------------------------------------------------------------------------- #
# The symbolic integration engine (Lemma 4.4)
# --------------------------------------------------------------------------- #


def _eliminate_axis(
    terms: List[Term],
    axis: int,
    lower: Bound,
    upper: Bound,
    clip_lo: float,
    clip_hi: float,
) -> List[Term]:
    """Integrate every term over ``x_axis`` between ``lower`` and ``upper``.

    Every condition touching the axis is turned into an additional lower or
    upper bound on the integration variable.  A case split enumerates which
    upper bound is the active minimum and which lower bound the active maximum
    (first-winner-wins ordering makes ties count exactly once), guarded by
    ``[V <= U]``; the inner integral is then ``H(U) - H(V)`` with ``H`` the
    antiderivative of the axis density.  ``H`` composed with a step-function
    bound is a step function on that bound's axis, so the output is again a
    list of terms -- with the axis fully eliminated.

    ``clip_lo``/``clip_hi`` bound the axis's true range; bound values are
    clamped into it before entering ``H`` so that infinities never meet ``H``.
    """
    out: List[Term] = []
    for term in terms:
        h = term.dens.pop(axis, None) or Step.const(1.0)
        anti = _Antiderivative(h, clip_lo, clip_hi)

        uppers: List[Bound] = []
        lowers: List[Bound] = []
        base = Term(term.coef, term.dens, {})
        dead = False
        for key, f in term.conds.items():
            i, j, sense, _ = key
            if axis not in (i, j):
                base.conds[key] = f
                continue
            if j == axis:  # [lambda sense f(x_i)]
                (uppers if sense == LE else lowers).append(("f", i, f))
                continue
            # i == axis: [x_j sense f(lambda)] -- solve for lambda.
            if f.monotone_dir() == 0:  # constant-valued: univariate in x_j
                op = "<=" if sense == LE else ">="
                base.mul_dens(j, _identity_compare(f.vs[0], op))
                continue
            want = ">=" if sense == LE else "<="  # {lambda : f(lambda) want x_j}
            theta, side = _ray_preimage_function(f, want)
            (uppers if side == "left" else lowers).append(("f", j, theta))
        uppers.append(upper)
        lowers.append(lower)
        uppers = _dedupe_bounds(uppers)
        lowers = _dedupe_bounds(lowers)

        for t, ub in enumerate(uppers):
            winner_u = base.clone()
            ok = True
            for t2, other in enumerate(uppers):
                if t2 == t:
                    continue
                # first-minimizer-wins: strict against earlier indices
                ok = winner_u.apply(_less_than(ub, other, strict=(t2 < t)))
                if not ok:
                    break
            if not ok:
                continue
            for s, lb in enumerate(lowers):
                winner = winner_u.clone()
                ok = True
                for s2, other in enumerate(lowers):
                    if s2 == s:
                        continue
                    # first-maximizer-wins
                    ok = winner.apply(_less_than(other, lb, strict=(s2 < s)))
                    if not ok:
                        break
                if ok:
                    ok = winner.apply(_less_than(lb, ub, strict=False))
                if not ok:
                    continue
                # integral = H(U) - H(V), as two terms
                for bound, sign in ((ub, 1.0), (lb, -1.0)):
                    piece = winner.clone()
                    piece.coef *= sign
                    if bound[0] == "c":
                        piece.coef *= anti(bound[1])
                    else:
                        piece.mul_dens(bound[1], anti.compose(bound[2].clamp(clip_lo, clip_hi)))
                    out.append(piece)
    return _merge_terms(out)


def _ray_preimage_function(f: Step, op: str) -> Tuple[Step, str]:
    """For monotone ``f``, the boundary of {lambda : f(lambda) op y} as a step
    function of ``y``, together with the ray side ('left': lambda <= theta(y),
    'right': lambda >= theta(y)).

    The returned step function is only ever applied to a *raw* integration
    variable, where breakpoint sloppiness is measure-zero-safe; compositions
    with other step functions go through ``_less_than``/``_ray_boundary``,
    which are piece-exact.
    """
    # theta changes value only where y crosses a value of f, so sampling one y
    # inside each open interval between consecutive distinct values is exact
    # almost everywhere (exact value points are measure-zero for the raw
    # variable theta is applied to).
    ys = sorted({v for v in f.vs if v not in (INF, -INF)})
    samples: List[float] = []
    if ys:
        samples.append(ys[0] - 1.0)
        for a, b in zip(ys, ys[1:]):
            samples.append((a + b) / 2.0)
        samples.append(ys[-1] + 1.0)
    else:
        samples.append(0.0)
    rays = [_ray_boundary(f, op, y) for y in samples]
    side = rays[0][0]
    return Step(ys, [x for _, x in rays]), side


def _dedupe_bounds(bounds: List[Bound]) -> List[Bound]:
    seen = set()
    unique = []
    for b in bounds:
        k = _bound_key(b)
        if k not in seen:
            seen.add(k)
            unique.append(b)
    return unique


def integrate_terms(terms: List[Term], lo: Sequence[float], hi: Sequence[float]) -> float:
    """Integrate a sum of basic-function terms over the box [lo, hi]."""
    if any(b <= a for a, b in zip(lo, hi)):
        return 0.0

    def range_of(axis: int) -> Tuple[float, float]:
        return lo[axis], hi[axis]

    live = _prune_terms([t.clone() for t in terms], range_of)
    for axis in range(len(lo)):
        a, b = lo[axis], hi[axis]
        live = _eliminate_axis(live, axis, ("c", a), ("c", b), a, b)
        live = _prune_terms(live, range_of)
        if not live:
            return 0.0
    return math.fsum(t.coef for t in live)


# --------------------------------------------------------------------------- #
# Compression (Lemma 4.5): grid-averaging of the basic function
# --------------------------------------------------------------------------- #


def compress_terms(
    terms: List[Term],
    grids: Sequence[Sequence[float]],
    lo: Sequence[float],
    hi: Sequence[float],
) -> List[Term]:
    """Replace F by its per-grid-cell average F~ (same integral over any
    grid-aligned region, complexity O(grid size) per function).

    ``grids[i]`` are the sorted grid coordinates on axis i, including the cell
    boundaries ``lo[i]`` and ``hi[i]``.  Implementation: rename all axes of F
    to fresh integration variables ``lambda_i = d + i``, integrate each
    ``lambda_i`` between the step-function bounds pred_i(x_i) / succ_i(x_i)
    (the grid neighbors of x_i), and divide by the grid-cell side lengths.
    """
    d = len(grids)

    def range_of(axis: int) -> Tuple[float, float]:
        return lo[axis % d], hi[axis % d]

    shifted: List[Term] = []
    for term in terms:
        dens = {axis + d: s for axis, s in term.dens.items()}
        conds = {(i + d, j + d, sense, bkt): f for (i, j, sense, bkt), f in term.conds.items()}
        shifted.append(Term(term.coef, dens, conds))

    live = _prune_terms(shifted, range_of)
    for i in range(d):
        g = list(grids[i])
        pred = Step(tuple(g), tuple([-INF] + g))          # largest grid value < x
        succ = Step(tuple(g), tuple(g + [INF]))           # smallest grid value > x
        live = _eliminate_axis(live, d + i, ("f", i, pred), ("f", i, succ), lo[i], hi[i])
        live = _prune_terms(live, range_of)
        if not live:
            return []

    for term in live:
        for i in range(d):
            g = grids[i]
            recip = Step(
                tuple(g),
                tuple([0.0] + [1.0 / (b - a) for a, b in zip(g, g[1:])] + [0.0]),
            )
            term.mul_dens(i, recip)
    return _merge_terms(live)


# --------------------------------------------------------------------------- #
# The divide-and-conquer driver (Section 4.2, grounded orthants)
# --------------------------------------------------------------------------- #


class ChanOrthantMeasure:
    """Measure of the complement of a union of orthants of arbitrary
    orientation inside a box domain, by Chan's Section-4.2 algorithm.

    An orthant is given as ``(vertex, signs)`` with ``signs[k] = +1`` for the
    constraint ``x_k >= vertex[k]`` and ``-1`` for ``x_k <= vertex[k]``; a
    bare point is accepted as shorthand for the grounded orthant
    ``{x >= p}`` (the hypervolume case).

    ``compress_every`` is the paper's "simplify only every ~log2(r^d) levels"
    knob (it throttles only the F~ compression; slab and staircase absorption
    are cheap and run at every node).  ``node_count`` and ``term_high_water``
    report statistics of the last run.
    """

    def __init__(
        self,
        dim: int,
        base_boxes: int = 2,
        compress_every: Optional[int] = None,
        compress_factor: float = 8.0,
        use_compression: bool = True,
    ) -> None:
        if dim < 3:
            raise ValueError("this algorithm needs dimension >= 3")
        self.dim = dim
        self.base_boxes = base_boxes
        self.compress_every = compress_every
        self.compress_factor = compress_factor
        self.use_compression = use_compression
        self.node_count = 0
        self.term_high_water = 0
        self.compress_count = 0

    # -- entry point -------------------------------------------------------- #

    def complement_measure(self, boxes: Sequence, lo: Point, hi: Point) -> float:
        normalized: List[Tuple[Point, Tuple[int, ...]]] = []
        for item in boxes:
            if len(item) == 2 and isinstance(item[0], (tuple, list)):
                vertex, signs = item
            else:  # bare point: grounded orthant {x >= p}
                vertex, signs = item, (1,) * self.dim
            if len(vertex) != self.dim or len(signs) != self.dim:
                raise ValueError("orthant dimension mismatch")
            normalized.append(
                (tuple(float(v) for v in vertex), tuple(int(s) for s in signs))
            )
        self.node_count = 0
        self.term_high_water = 0
        self.compress_count = 0
        n = max(2, len(normalized))
        if self.compress_every is None:
            # every ~log2(r^d) levels with r = n^0.1, i.e. 0.1 * d * log2 n
            self._interval = max(1, round(0.1 * self.dim * math.log2(n)))
        else:
            self._interval = self.compress_every
        sys.setrecursionlimit(max(sys.getrecursionlimit(), 50_000))
        return self._measure(normalized, [Term(1.0)], list(lo), list(hi), 0, 0)

    def _active(self, vertex: Point, signs, k: int, lo, hi) -> bool:
        """Does the orthant's k-constraint actually cut the cell?"""
        return vertex[k] > lo[k] if signs[k] > 0 else vertex[k] < hi[k]

    # -- one recursion node ------------------------------------------------- #

    def _measure(
        self,
        boxes: List[Tuple[Point, Tuple[int, ...]]],
        terms: List[Term],
        lo: List[float],
        hi: List[float],
        depth: int,
        since_compress: int,
    ) -> float:
        self.node_count += 1
        self.term_high_water = max(self.term_high_water, len(terms))

        # Step 1 (simplify, cheap part): absorb boxes that span the cell on
        # all axes (cover it), all but one axis (slabs -> shrink the cell), or
        # all but two axes (2-sided orthants -> staircase conditions).
        absorbed = self._absorb(boxes, terms, lo, hi)
        if absorbed is None:
            return 0.0  # the cell is entirely covered
        boxes, terms, lo, hi = absorbed
        terms = _prune_terms(terms, lambda a: (lo[a], hi[a]))
        if not terms:
            return 0.0

        # Base cases: no surviving boxes -> integrate F; few boxes ->
        # inclusion-exclusion over them (each intersection is a box).
        if len(boxes) <= self.base_boxes:
            total = 0.0
            for size in range(len(boxes) + 1):
                for subset in itertools.combinations(boxes, size):
                    sub_lo, sub_hi = list(lo), list(hi)
                    for vertex, signs in subset:
                        for k in range(self.dim):
                            if signs[k] > 0:
                                sub_lo[k] = max(sub_lo[k], vertex[k])
                            else:
                                sub_hi[k] = min(sub_hi[k], vertex[k])
                    value = integrate_terms(terms, sub_lo, sub_hi)
                    total += value if size % 2 == 0 else -value
            return total

        # Step 1 (simplify, expensive part): compress F back to complexity
        # O(|B|).  Compression is the only operation that can multiply the
        # number of terms (absorption only adds/merges conditions), so it is
        # doubly throttled: it must be at least `compress_every` levels since
        # the last one (the paper's r^d-block schedule, which bounds the
        # number of compressions per root-leaf path by a constant), and the
        # representation must actually have outgrown its post-compression
        # size O(|B|) by `compress_factor` -- running it more often than that
        # only compounds its constant-factor term blowup.
        if (
            self.use_compression
            and since_compress >= self._interval
            and _terms_complexity(terms)
            > self.compress_factor * self.dim**2 * (len(boxes) + 2)
        ):
            self.compress_count += 1
            grids = []
            for k in range(self.dim):
                grids.append(
                    sorted(
                        {lo[k], hi[k]}
                        | {
                            v[k]
                            for v, s in boxes
                            if self._active(v, s, k, lo, hi)
                        }
                    )
                )
            terms = compress_terms(terms, grids, lo, hi)
            since_compress = 0
            if not terms:
                return 0.0

        # Step 2 (cut): weighted median of the (d-3)-faces on the cycling cut
        # axis; face orthogonal to axes i, j, k weighs 2^((i'+j'+k')/d) in the
        # renumbering that puts the cut axis at position 1.
        for probe in range(self.dim):
            axis = (depth + probe) % self.dim
            cuts = self._cut_candidates(boxes, lo, hi, axis, depth + probe)
            if cuts:
                depth += probe
                break
        else:
            raise AssertionError("simplified boxes must expose a (d-3)-face")

        median = _weighted_median(cuts)
        left_hi = list(hi)
        left_hi[axis] = median
        right_lo = list(lo)
        right_lo[axis] = median
        # Clone before recursing: children prune their terms in place against
        # their own (smaller) cells, so they must not share Term objects.
        right_terms = [t.clone() for t in terms]
        return self._measure(
            boxes, terms, lo, left_hi, depth + 1, since_compress + 1
        ) + self._measure(
            boxes, right_terms, right_lo, hi, depth + 1, since_compress + 1
        )

    # -- simplification: absorb low-complexity boxes ------------------------ #

    def _absorb(self, boxes, terms, lo, hi):
        lo, hi = list(lo), list(hi)
        dim = self.dim
        while True:
            # Drop orthants whose intersection with the cell has no interior:
            # a >=-constraint at or above the ceiling (resp. a <=-constraint
            # at or below the floor) empties the box within the cell.
            live = []
            for vertex, signs in boxes:
                empty = any(
                    (vertex[k] >= hi[k]) if signs[k] > 0 else (vertex[k] <= lo[k])
                    for k in range(dim)
                )
                if not empty:
                    live.append((vertex, signs))
            boxes = live
            actives = [
                [k for k in range(dim) if self._active(v, s, k, lo, hi)]
                for v, s in boxes
            ]
            if any(not act for act in actives):
                return None  # a box covers the cell entirely
            # Slabs (exactly one active constraint) are halfspaces; their
            # complement chops the cell from one side.
            shrunk = False
            keep = []
            for (vertex, signs), act in zip(boxes, actives):
                if len(act) == 1:
                    i = act[0]
                    if signs[i] > 0:  # {x_i >= v}: complement is x_i < v
                        hi[i] = min(hi[i], vertex[i])
                    else:  # {x_i <= v}: complement is x_i > v
                        lo[i] = max(lo[i], vertex[i])
                    shrunk = True
                else:
                    keep.append(((vertex, signs), act))
            if any(hi[k] <= lo[k] for k in range(dim)):
                return None  # opposing slabs covered the cell
            boxes = [b for b, _ in keep]
            if not shrunk:
                actives = [act for _, act in keep]
                break
            # shrinking the cell can create new covers/slabs: loop again

        # 2-sided orthants: absorb each orientation class of each axis pair
        # as one monotone staircase condition on every term.
        pairs: Dict[Tuple[int, int, int, int], List[Tuple[float, float]]] = {}
        rest: List[Tuple[Point, Tuple[int, ...]]] = []
        for (vertex, signs), act in zip(boxes, actives):
            if len(act) == 2:
                i, j = act
                key = (i, j, signs[i], signs[j])
                pairs.setdefault(key, []).append((vertex[i], vertex[j]))
            else:
                rest.append((vertex, signs))
        if pairs:
            terms = [t.clone() for t in terms]
            for (i, j, si, sj), pts in pairs.items():
                f, sense = _staircase(pts, si, sj)
                for term in terms:
                    term.add_cond(i, j, f, sense)
            terms = _merge_terms(terms)
        return rest, terms, lo, hi

    # -- cutting ------------------------------------------------------------ #

    def _cut_candidates(self, boxes, lo, hi, axis, depth):
        d = self.dim
        pos = [((k - depth) % d) + 1 for k in range(d)]  # renumbered positions
        cuts: List[Tuple[float, float]] = []
        for vertex, signs in boxes:
            if not self._active(vertex, signs, axis, lo, hi):
                continue
            active = [
                k
                for k in range(d)
                if k != axis and self._active(vertex, signs, k, lo, hi)
            ]
            if len(active) < 2:
                continue
            weight = sum(
                2.0 ** ((pos[axis] + pos[j] + pos[k]) / d)
                for j, k in itertools.combinations(active, 2)
            )
            cuts.append((vertex[axis], weight))
        return cuts


def _staircase(
    points: Sequence[Tuple[float, float]], si: int, sj: int
) -> Tuple[Step, str]:
    """Complement boundary of a union of 2-sided orthants of one orientation
    class in the (i, j)-plane.

    Each point (a, b) is the vertex of the quadrant {x_i >=< a, x_j >=< b}
    with directions given by the signs ``si``/``sj`` (+1 for >=, -1 for <=).
    The complement of the union of one class is described by a single
    monotone step condition (up to measure zero):

        si  sj   covered iff                     complement        shape
        +   +    x_j >= min{b : a <= x_i}        x_j <= f(x_i)     decreasing
        -   -    x_j <= max{b : a >= x_i}        x_j >= f(x_i)     decreasing
        +   -    x_j <= max{b : a <= x_i}        x_j >= f(x_i)     increasing
        -   +    x_j >= min{b : a >= x_i}        x_j <= f(x_i)     increasing

    Returns the step function and the condition sense (LE or GE).
    """
    use_min = sj > 0
    agg = min if use_min else max
    neutral = INF if use_min else -INF
    sense = LE if use_min else GE

    best_at: Dict[float, float] = {}
    for a, b in points:
        best_at[a] = agg(best_at[a], b) if a in best_at else b
    xs = sorted(best_at)

    if si > 0:  # boxes with a <= t qualify: prefix aggregation
        vs = [neutral]
        run = neutral
        for a in xs:
            run = agg(run, best_at[a])
            vs.append(run)
    else:  # boxes with a >= t qualify: suffix aggregation
        rev = [neutral]
        run = neutral
        for a in reversed(xs):
            run = agg(run, best_at[a])
            rev.append(run)
        vs = list(reversed(rev))
    return Step(xs, vs), sense


def _weighted_median(cuts: List[Tuple[float, float]]) -> float:
    cuts.sort()
    half = sum(w for _, w in cuts) / 2.0
    acc = 0.0
    for x, w in cuts:
        acc += w
        if acc >= half:
            return x
    return cuts[-1][0]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def orthant_union_volume(
    orthants: Iterable[Tuple[Sequence[float], Sequence[int]]],
    lo: Sequence[float],
    hi: Sequence[float],
    **solver_options,
) -> float:
    """Volume of the union of arbitrary-orientation orthants within the box
    ``[lo, hi]``, by Chan's Section-4.2 algorithm (dimension >= 3).

    Each orthant is a pair ``(vertex, signs)``: ``signs[k] = +1`` selects the
    halfspace ``x_k >= vertex[k]`` and ``-1`` selects ``x_k <= vertex[k]``.
    The hypervolume indicator is the special case with all signs ``+1`` and
    ``hi`` the reference point (see ``hypervolume_dby3``).
    """
    lo = tuple(float(x) for x in lo)
    hi = tuple(float(x) for x in hi)
    dim = len(lo)
    volume = 1.0
    for a, b in zip(lo, hi):
        if b <= a:
            return 0.0
        volume *= b - a
    boxes = [
        (tuple(float(v) for v in vertex), tuple(int(s) for s in signs))
        for vertex, signs in orthants
    ]
    solver = ChanOrthantMeasure(dim, **solver_options)
    return volume - solver.complement_measure(boxes, lo, hi)


def hypervolume_dby3(
    points: Iterable[Sequence[float]],
    reference: Sequence[float],
    maximize: bool = False,
    prefilter: bool = True,
    **solver_options,
) -> float:
    """Hypervolume indicator via Chan's O(n^(d/3) polylog n) orthant algorithm.

    Same contract as ``chan_hypervolume.hypervolume`` (minimisation by
    default); requires dimension >= 3.  Extra keyword arguments are forwarded
    to ``ChanOrthantMeasure`` (``base_boxes``, ``compress_every``,
    ``use_compression``).
    """
    sign = -1.0 if maximize else 1.0
    ref = tuple(sign * float(x) for x in reference)
    dim = len(ref)
    front: List[Point] = []
    for point in points:
        vec = tuple(sign * float(x) for x in point)
        if len(vec) != dim:
            raise ValueError("points and reference must have the same dimension")
        if all(x < r for x, r in zip(vec, ref)):
            front.append(vec)
    if not front:
        return 0.0
    if prefilter:
        front = _nondominated(front)
    lo = tuple(min(p[k] for p in front) for k in range(dim))
    volume = 1.0
    for a, b in zip(lo, ref):
        volume *= b - a
    solver = ChanOrthantMeasure(dim, **solver_options)
    return volume - solver.complement_measure(front, lo, ref)


def _nondominated(points: Sequence[Point]) -> List[Point]:
    kept: List[Point] = []
    for point in sorted(points):
        if not any(all(a <= b for a, b in zip(other, point)) for other in kept):
            kept.append(point)
    return kept


# --------------------------------------------------------------------------- #
# Self-tests
# --------------------------------------------------------------------------- #


def _random_monotone_step(rng: random.Random, direction: int, m: int = 4) -> Step:
    xs = sorted(rng.sample([round(rng.uniform(-3, 3), 3) for _ in range(20)], m))
    xs = sorted(set(xs))
    vals = sorted(rng.uniform(-3, 3) for _ in range(len(xs) + 1))
    if direction < 0:
        vals = vals[::-1]
    return Step(xs, vals)


def _test_step_ops(rng: random.Random) -> None:
    for _ in range(300):
        f = _random_monotone_step(rng, rng.choice((1, -1)))
        g = _random_monotone_step(rng, rng.choice((1, -1)))
        x = rng.uniform(-4, 4)
        assert abs(f.minimum(g)(x) - min(f(x), g(x))) < 1e-12
        assert abs(f.times(g)(x) - f(x) * g(x)) < 1e-9
        c = rng.uniform(-3, 3)
        assert f.compare_const(c, "<=")(x) == (1.0 if f(x) <= c else 0.0)
    print("  [ok ] Step pointwise operations")


def _test_ray_boundary(rng: random.Random) -> None:
    for _ in range(2000):
        f = _random_monotone_step(rng, rng.choice((1, -1)))
        # include exact plateau values to exercise the positive-measure cases
        y = rng.choice([rng.uniform(-4, 4)] + list(f.vs[:-1]))
        if y in (INF, -INF):
            continue
        op = rng.choice(["<", "<=", ">", ">="])
        side, x0 = _ray_boundary(f, op, y)
        t = rng.uniform(-5, 5)
        inside = t < x0 if side == "left" else t > x0
        truth = _CMP[op](f(t), y)
        if abs(t - x0) > 1e-9 and all(abs(t - b) > 1e-9 for b in f.xs):
            assert inside == truth, (f.xs, f.vs, op, y, t, side, x0)
    print("  [ok ] _ray_boundary (incl. plateau levels)")


def _test_less_than(rng: random.Random) -> None:
    for _ in range(2000):
        def mk() -> Bound:
            if rng.random() < 0.25:
                return ("c", round(rng.uniform(-2, 2), 1))
            return ("f", rng.choice((0, 1)), _random_monotone_step(rng, rng.choice((1, -1))))

        a, b = mk(), mk()
        strict = rng.random() < 0.5
        res = _less_than(a, b, strict)
        x = {0: rng.uniform(-4, 4), 1: rng.uniform(-4, 4)}

        def val(bound: Bound) -> float:
            return bound[1] if bound[0] == "c" else bound[2](x[bound[1]])

        truth = val(a) < val(b) if strict else val(a) <= val(b)
        if res in (TRUE, FALSE):
            got = res == TRUE
        elif res[0] == "dens":
            got = res[2](x[res[1]]) == 1.0
        else:
            _, i, j, f, sense = res
            got = x[j] <= f(x[i]) if sense == LE else x[j] >= f(x[i])
        # allow disagreement only within a hair of a breakpoint (measure zero)
        near = any(
            any(abs(x[ax] - bp) < 1e-9 for bp in bound[2].xs)
            for bound in (a, b)
            if bound[0] == "f"
            for ax in (bound[1],)
        ) or abs(val(a) - val(b)) < 1e-9
        if not near:
            assert got == truth, (a, b, strict, res, x)
    print("  [ok ] _less_than bound comparisons")


def _grid_integral(terms: List[Term], lo, hi) -> float:
    """Exact integral of a sum of terms over [lo, hi] by grid enumeration
    (every function involved is a step function, so midpoint sampling on the
    combined breakpoint grid is exact).  Exponential; tests only.
    """
    d = len(lo)
    axes_breaks = [set([lo[k], hi[k]]) for k in range(d)]
    for term in terms:
        for a, s in term.dens.items():
            axes_breaks[a].update(b for b in s.xs if lo[a] < b < hi[a])
        for (i, j, _, _), f in term.conds.items():
            axes_breaks[i].update(b for b in f.xs if lo[i] < b < hi[i])
            axes_breaks[j].update(v for v in f.vs if lo[j] < v < hi[j])
    grids = [sorted(b) for b in axes_breaks]
    total = 0.0
    for cell in itertools.product(*[range(len(g) - 1) for g in grids]):
        mid = [
            (grids[k][cell[k]] + grids[k][cell[k] + 1]) / 2.0 for k in range(d)
        ]
        vol = 1.0
        for k in range(d):
            vol *= grids[k][cell[k] + 1] - grids[k][cell[k]]
        for term in terms:
            value = term.coef
            for a, s in term.dens.items():
                value *= s(mid[a])
            for (i, j, sense, _), f in term.conds.items():
                holds = mid[j] <= f(mid[i]) if sense == LE else mid[j] >= f(mid[i])
                if not holds:
                    value = 0.0
                    break
            total += value * vol
    return total


def _random_term(rng: random.Random, d: int) -> Term:
    term = Term(rng.choice((1.0, 1.0, -0.5)))
    for a in range(d):
        if rng.random() < 0.5:
            xs = sorted({round(rng.uniform(0, 1), 2) for _ in range(2)})
            term.dens[a] = Step(tuple(xs), tuple(rng.uniform(0, 2) for _ in range(len(xs) + 1)))
    npairs = rng.randrange(0, d)
    for _ in range(npairs):
        i, j = rng.sample(range(d), 2)
        f = _random_monotone_step(rng, rng.choice((1, -1)), m=3)
        f = Step(tuple(x * 0.2 + 0.5 for x in f.xs), tuple(v * 0.2 + 0.5 for v in f.vs))
        term.add_cond(i, j, f, rng.choice((LE, GE)))
    return term


def _test_integration(rng: random.Random) -> None:
    for d in (1, 2, 3, 4):
        for trial in range(40 if d < 4 else 15):
            terms = [_random_term(rng, d) for _ in range(rng.randrange(1, 3))]
            lo = [0.0] * d
            hi = [1.0] * d
            got = integrate_terms(terms, lo, hi)
            want = _grid_integral(terms, lo, hi)
            assert abs(got - want) < 1e-8 * max(1.0, abs(want)), (d, trial, got, want)
    print("  [ok ] symbolic integration vs exact grid enumeration")


def _test_compression(rng: random.Random) -> None:
    for d in (2, 3):
        for trial in range(25):
            terms = [_random_term(rng, d) for _ in range(rng.randrange(1, 3))]
            lo, hi = [0.0] * d, [1.0] * d
            pts = [
                tuple(rng.choice((0.2, 0.4, 0.6, 0.8)) for _ in range(d))
                for _ in range(rng.randrange(1, 4))
            ]
            grids = [sorted({lo[k], hi[k]} | {p[k] for p in pts}) for k in range(d)]
            packed = compress_terms(terms, grids, lo, hi)
            # integrals must agree over every region aligned to the point grid;
            # check the complement of the union of {x >= p} via cell sums
            for term_set in (terms, packed):
                pass
            def complement_integral(term_set):
                total = 0.0
                for cell in itertools.product(*[range(len(g) - 1) for g in grids]):
                    mids = [
                        (grids[k][cell[k]] + grids[k][cell[k] + 1]) / 2.0
                        for k in range(d)
                    ]
                    if any(all(mids[k] >= p[k] for k in range(d)) for p in pts):
                        continue
                    sub_lo = [grids[k][cell[k]] for k in range(d)]
                    sub_hi = [grids[k][cell[k] + 1] for k in range(d)]
                    total += _grid_integral(term_set, sub_lo, sub_hi)
                return total

            got = complement_integral(packed)
            want = complement_integral(terms)
            assert abs(got - want) < 1e-8 * max(1.0, abs(want)), (d, trial, got, want)
            # compressed functions must live on the grid
            for term in packed:
                for a, s in term.dens.items():
                    assert set(s.xs) <= set(grids[a]), "density off-grid"
    print("  [ok ] compression preserves grid-aligned integrals")


def _reference_hypervolume(points, reference) -> float:
    total = 0.0
    pts = list(points)
    for size in range(1, len(pts) + 1):
        for subset in itertools.combinations(pts, size):
            corner = [max(p[k] for p in subset) for k in range(len(reference))]
            vol = 1.0
            for a, b in zip(corner, reference):
                vol *= max(0.0, b - a)
            total += vol if size % 2 else -vol
    return total


def _reference_orthant_union(orthants, lo, hi) -> float:
    """O(2^n) inclusion-exclusion for arbitrary orthants; tests only."""
    d = len(lo)
    total = 0.0
    for size in range(1, len(orthants) + 1):
        for subset in itertools.combinations(orthants, size):
            vol = 1.0
            for k in range(d):
                a, b = lo[k], hi[k]
                for vertex, signs in subset:
                    if signs[k] > 0:
                        a = max(a, vertex[k])
                    else:
                        b = min(b, vertex[k])
                vol *= max(0.0, b - a)
            total += vol if size % 2 else -vol
    return total


def _test_staircase_classes(rng: random.Random) -> None:
    for _ in range(400):
        si, sj = rng.choice((1, -1)), rng.choice((1, -1))
        pts = [
            (rng.choice((0.2, 0.4, 0.6, 0.8)), rng.choice((0.2, 0.4, 0.6, 0.8)))
            for _ in range(rng.randrange(1, 6))
        ]
        f, sense = _staircase(pts, si, sj)
        for _ in range(20):
            t, y = rng.random(), rng.random()  # continuous: ties a.s. avoided
            covered = any(
                (t >= a if si > 0 else t <= a) and (y >= b if sj > 0 else y <= b)
                for a, b in pts
            )
            holds = y <= f(t) if sense == LE else y >= f(t)
            assert holds == (not covered), (si, sj, pts, t, y, f.xs, f.vs)
    print("  [ok ] staircase absorption, all four orientation classes")


def _test_mixed_orthants(rng: random.Random) -> None:
    lo3, hi3 = None, None  # silence linters
    for d in (3, 4, 5):
        lo = tuple(0.0 for _ in range(d))
        hi = tuple(1.0 for _ in range(d))
        cases = [(6, False), (10, False), (8, True)] if d < 5 else [(6, False), (6, True)]
        for n, integer_grid in cases:
            orthants = []
            for _ in range(n):
                if integer_grid:  # tie-heavy coordinates
                    vertex = tuple(rng.randrange(1, 4) / 4.0 for _ in range(d))
                else:
                    vertex = tuple(rng.random() for _ in range(d))
                signs = tuple(rng.choice((1, -1)) for _ in range(d))
                orthants.append((vertex, signs))
            got = orthant_union_volume(orthants, lo, hi)
            want = _reference_orthant_union(orthants, lo, hi)
            assert abs(got - want) < 1e-8 * max(1.0, want), (d, n, got, want)
    # forced compression with mixed orientations
    orthants = [
        (tuple(rng.random() for _ in range(4)), tuple(rng.choice((1, -1)) for _ in range(4)))
        for _ in range(8)
    ]
    lo, hi = (0.0,) * 4, (1.0,) * 4
    got = orthant_union_volume(orthants, lo, hi, compress_every=3, compress_factor=0.0)
    want = _reference_orthant_union(orthants, lo, hi)
    assert abs(got - want) < 1e-8 * max(1.0, want), (got, want)
    print("  [ok ] arbitrary-orientation orthants vs inclusion-exclusion (d=3..5)")


def _test_full_algorithm(rng: random.Random) -> None:
    sizes = {3: (1, 4, 8, 12), 4: (4, 8, 10), 5: (6,)}
    for d, ns in sizes.items():
        for n in ns:
            ref = tuple(1.0 for _ in range(d))
            pts = [tuple(rng.random() for _ in range(d)) for _ in range(n)]
            got = hypervolume_dby3(pts, ref)
            want = _reference_hypervolume(_nondominated(pts), ref)
            assert abs(got - want) < 1e-8 * max(1.0, want), (d, n, got, want)
    for d in (3, 4):  # coordinate-tie-heavy integer instances
        ref = tuple(4.0 for _ in range(d))
        pts = [
            tuple(float(rng.randrange(0, 4)) for _ in range(d)) for _ in range(10)
        ]
        got = hypervolume_dby3(pts, ref)
        want = _reference_hypervolume(_nondominated([p for p in pts if all(x < 4 for x in p)]), ref)
        assert abs(got - want) < 1e-8 * max(1.0, want), (d, "grid", got, want)
    print("  [ok ] full d/3 algorithm vs inclusion-exclusion (d=3..5)")


def _test_against_section2(rng: random.Random) -> None:
    try:
        from chan_hypervolume import hypervolume as hv_section2
    except ImportError:
        print("  [-- ] chan_hypervolume not importable, cross-check skipped")
        return
    for d, n in ((3, 25), (3, 40), (4, 15)):
        ref = tuple(1.0 for _ in range(d))
        pts = [tuple(rng.random() for _ in range(d)) for _ in range(n)]
        got = hypervolume_dby3(pts, ref)
        want = hv_section2(pts, ref)
        assert abs(got - want) < 1e-8 * max(1.0, want), (d, n, got, want)
    # compression on and off must agree
    for d, n in ((3, 20), (4, 10)):
        ref = tuple(1.0 for _ in range(d))
        pts = [tuple(rng.random() for _ in range(d)) for _ in range(n)]
        a = hypervolume_dby3(pts, ref, use_compression=True)
        b = hypervolume_dby3(pts, ref, use_compression=False)
        assert abs(a - b) < 1e-8 * max(1.0, a), (d, n, a, b)
    print("  [ok ] d/3 vs Section-2 algorithm; compression on == off")


def _test_forced_compression(rng: random.Random) -> None:
    """Force the F~ compression to fire inside the recursion (the adaptive
    default rarely needs it) and check the result is unchanged."""
    d, n = 4, 8
    ref = tuple(1.2 for _ in range(d))
    pts = []
    while len(pts) < n:  # points on the positive unit sphere: nothing dominated
        v = [abs(rng.gauss(0.0, 1.0)) + 1e-9 for _ in range(d)]
        s = math.sqrt(sum(x * x for x in v))
        pts.append(tuple(x / s for x in v))
    want = _reference_hypervolume(pts, ref)
    got = hypervolume_dby3(
        pts, ref, prefilter=False, compress_every=3, compress_factor=0.0
    )
    assert abs(got - want) < 1e-8 * max(1.0, want), (got, want)
    print("  [ok ] forced in-tree compression agrees with inclusion-exclusion")


def _self_test() -> None:
    rng = random.Random(20130811)
    print("chan_orthant_dby3 self-tests")
    _test_step_ops(rng)
    _test_ray_boundary(rng)
    _test_less_than(rng)
    _test_integration(rng)
    _test_compression(rng)
    _test_staircase_classes(rng)
    _test_full_algorithm(rng)
    _test_against_section2(rng)
    _test_forced_compression(rng)
    _test_mixed_orthants(rng)
    print("All self-tests passed.")


if __name__ == "__main__":
    _self_test()
