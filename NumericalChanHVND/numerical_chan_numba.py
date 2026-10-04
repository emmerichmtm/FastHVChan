"""Float64 hybrid backend: Numba array kernels, Python Chan/term recursion.

No fastmath; no parallelism. The exact backend remains numerical_chan.py.
Cold compilation/cache loading is separate from warm timings in benchmarks.
"""

from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, groupby, product
from math import isfinite, prod
import numpy as np
from numba import njit


@dataclass
class Term:
    coef: object
    unary: dict
    edges: dict

    def copy(self):
        return Term(self.coef, self.unary.copy(), self.edges.copy())

    def alive(self):
        return bool(self.coef) and all(nonzero(w) for w in self.unary.values())


@njit(cache=True)
def nonzero(values):
    for x in values:
        if x != 0:
            return True
    return False


@njit(cache=True)
def direction(values):
    inc, dec = True, True
    for i in range(1, len(values)):
        inc = inc and values[i - 1] <= values[i]
        dec = dec and values[i - 1] >= values[i]
    if inc:
        return 1
    if dec:
        return -1
    raise ValueError('Nonmonotone cutoff')


@njit(cache=True)
def prefix_sum(weights):
    result = np.zeros(len(weights) + 1)
    for i in range(len(weights)):
        result[i + 1] = result[i] + weights[i]
    return result


@njit(cache=True)
def search_array(values, queries, right, negate=False, offset=0):
    out = np.empty(len(queries), dtype=np.int64)
    for i in range(len(queries)):
        x = -queries[i] if negate else queries[i]
        lo, hi = 0, len(values)
        while lo < hi:
            mid = (lo + hi) // 2
            if values[mid] < x or (right and values[mid] == x):
                lo = mid + 1
            else:
                hi = mid
        out[i] = lo + offset
    return out


@njit(cache=True)
def normalize(values):
    factor = 1.0
    for x in values:
        if x != 0:
            factor = x
            break
    return factor, values / factor


@njit(cache=True)
def suffix_mask(points, size):
    result = np.full(size, -1, dtype=np.int64)
    for k in range(len(points)):
        a, b = points[k, 0], points[k, 1]
        result[a] = max(result[a], b)
    for k in range(size - 2, -1, -1):
        result[k] = max(result[k], result[k + 1])
    return result + 1


def unique_bounds(bounds):
    kept = {}
    for axis, values in bounds:
        kept.setdefault((axis, values if axis is None else values.tobytes()), (axis, values))
    return list(kept.values())


def ordered_unique(values):
    return [value for value, group in groupby(sorted(values))]


def multiply_unary(term, axis, values):
    term.unary[axis] = term.unary[axis] * values
    return term.alive()


def add_edge(term, source, target, upper, values):
    assert source < target and len(values) == len(term.unary[source])
    size = len(term.unary[target])
    low, high = (-1, size - 1) if upper else (0, size)
    values = np.clip(np.asarray(values, dtype=np.int64), low, high)
    key = (source, target, upper, direction(values))
    if key in term.edges:
        values = (np.minimum if upper else np.maximum)(values, term.edges.pop(key))
    if values.min() == values.max():
        indices = np.arange(size)
        return multiply_unary(term, target, indices <= values[0] if upper else indices >= values[0])
    term.edges[key] = values
    return multiply_unary(term, source, values >= 0 if upper else values < size)


def impose(term, first, second, strict=False):
    a, av = first
    b, bv = second
    if a is None and b is None:
        return (av < bv if strict else av <= bv) and term.alive()
    if a == b:
        return multiply_unary(term, a, av < bv if strict else av <= bv)
    if a is None:
        return multiply_unary(term, b, av < bv if strict else av <= bv)
    if b is None:
        return multiply_unary(term, a, av < bv if strict else av <= bv)
    if a < b:
        if direction(bv) == 1:
            return add_edge(term, a, b, False, search_array(bv, av, strict))
        return add_edge(term, a, b, True, search_array(-bv, av, not strict, True, -1))
    if direction(av) == 1:
        return add_edge(term, b, a, True, search_array(av, bv, not strict, False, -1))
    return add_edge(term, b, a, False, search_array(-av, bv, strict, True))


def merge(terms, stats):
    """Incremental exact-key merging; not needed for the raw emission bound."""
    kept = {}
    for term in terms:
        stats['raw_terms'] += 1
        if not term.alive():
            continue
        for axis, values in term.unary.items():
            factor, term.unary[axis] = normalize(values)
            term.coef *= factor
        key = (tuple((i, v.tobytes()) for i, v in sorted(term.unary.items())),
               tuple((i, v.tobytes()) for i, v in sorted(term.edges.items())))
        if key in kept:
            kept[key].coef += term.coef
        else:
            kept[key] = term
    answer = [term for term in kept.values() if term.coef]
    stats['peak_terms'] = max(stats['peak_terms'], len(answer))
    return answer


def eliminate_one(terms, axis, stats):
    """Sum one index by first-winner lower/upper bounds and prefix sums."""
    def emitted():
        for incoming in terms:
            stats['term_eliminations'] += 1
            term = incoming.copy()
            weights = term.unary.pop(axis)
            prefix = prefix_sum(weights)
            lowers, uppers = [(None, 0)], [(None, len(weights) - 1)]
            for key, values in list(term.edges.items()):
                i, j, upper, orient = key
                if axis not in (i, j):
                    continue
                del term.edges[key]
                if axis == j:
                    (uppers if upper else lowers).append((i, values))
                else:
                    targets = np.arange(len(term.unary[j]), dtype=np.int64)
                    if orient == 1:
                        cut = (search_array(values, targets, False) if upper
                               else search_array(values, targets, True, False, -1))
                        (lowers if upper else uppers).append((j, cut))
                    else:
                        negative = -values
                        cut = (search_array(negative, targets, True, True, -1) if upper
                               else search_array(negative, targets, False, True))
                        (uppers if upper else lowers).append((j, cut))
            lowers, uppers = unique_bounds(lowers), unique_bounds(uppers)
            stats['max_lower_candidates'] = max(stats['max_lower_candidates'], len(lowers))
            stats['max_upper_candidates'] = max(stats['max_upper_candidates'], len(uppers))
            for u, upper_bound in enumerate(uppers):
                top = term.copy()
                if not all(impose(top, upper_bound, other, strict=q < u)
                           for q, other in enumerate(uppers) if q != u):
                    continue
                for ell, lower_bound in enumerate(lowers):
                    winner = top.copy()
                    if not all(impose(winner, other, lower_bound, strict=q < ell)
                               for q, other in enumerate(lowers) if q != ell):
                        continue
                    if not impose(winner, lower_bound, upper_bound):
                        continue
                    a, av = upper_bound
                    b, bv = lower_bound
                    if a == b or a is None or b is None:
                        variable = a if a is not None else b
                        if variable is None:
                            winner.coef *= prefix[av + 1] - prefix[bv]
                        else:
                            size = len(winner.unary[variable])
                            high = np.full(size, prefix[av + 1]) if a is None else prefix[av + 1]
                            low = np.full(size, prefix[bv]) if b is None else prefix[bv]
                            multiply_unary(winner, variable, high - low)
                        yield winner
                        continue
                    for bound, offset, sign in ((upper_bound, 1, 1), (lower_bound, 0, -1)):
                        piece = winner.copy()
                        piece.coef *= sign
                        variable, values = bound
                        if variable is None:
                            piece.coef *= prefix[values + offset]
                        else:
                            multiply_unary(piece, variable,
                                           prefix[values + offset])
                        yield piece
    return merge(emitted(), stats)


def sum_out(terms, axes, stats):
    """Eliminate consecutive pairs, then an odd singleton; never build a pair table."""
    for offset in range(0, len(axes), 2):
        for axis in axes[offset:offset + 2]:
            terms = eliminate_one(terms, axis, stats)
    return terms


def value_at(terms, index):
    return sum(t.coef * prod(t.unary[i][index[i]] for i in t.unary)
               for t in terms if all(index[j] <= f[index[i]] if upper else index[j] >= f[index[i]]
                                     for (i, j, upper, _), f in t.edges.items()))


def push_blocks(terms, blocks, stats):
    """Lift d old axes to 2d variables, sum old axes, and keep d block axes."""
    d = len(blocks)
    lifted = []
    for old in terms:
        term = old.copy()
        for i in range(d):
            term.unary[i + d] = np.ones(len(blocks[i]))
        for i in range(d):
            identity = (i, np.arange(len(term.unary[i]), dtype=np.int64))
            lower = (i + d, np.array([a for a, b in blocks[i]], dtype=np.int64))
            upper = (i + d, np.array([b for a, b in blocks[i]], dtype=np.int64))
            if not impose(term, lower, identity) or not impose(term, identity, upper):
                break
        else:
            lifted.append(term)
    packed = sum_out(lifted, list(range(d)), stats)
    for term in packed:
        term.unary = {i - d: values for i, values in term.unary.items()}
        term.edges = {(i - d, j - d, upper, orient): values
                      for (i, j, upper, orient), values in term.edges.items()}
    stats['compressions'] += 1
    return packed


def rectangle_sum(terms, lo, hi, stats):
    d = len(lo)
    if any(a > b for a, b in zip(lo, hi)):
        return 0
    restricted = []
    for old in terms:
        term = old.copy()
        for i in range(d):
            multiply_unary(term, i, (np.arange(len(term.unary[i])) >= lo[i]) & (np.arange(len(term.unary[i])) <= hi[i]))
        if term.alive():
            restricted.append(term)
    return sum(t.coef for t in sum_out(restricted, list(range(d)), stats))


def ceil_root(value, degree):
    """Smallest integer r with r**degree >= value; no floating-point roots."""
    low, high = 0, 1 << ((value.bit_length() + degree - 1) // degree)
    while low < high:
        middle = (low + high) // 2
        if middle ** degree < value:
            low = middle + 1
        else:
            high = middle
    return low


class PrefixSkyline:
    """Nonincreasing heights with prefix-chmax updates and weighted total.

    A rectangle raises an initial interval to a height. Monotonicity makes
    the changed indices one interval: find its first index, then assign it.
    A lazy segment tree supports both operations in O(log n).
    """
    def __init__(self, masses):
        self.n = len(masses)
        self.prefix = [0]
        for mass in masses:
            self.prefix.append(self.prefix[-1] + mass)
        self.minimum, self.total, self.lazy = ([0] * (4 * self.n) for _ in range(3))

    def _set(self, node, lo, hi, height):
        self.minimum[node] = self.lazy[node] = height
        self.total[node] = height * (self.prefix[hi + 1] - self.prefix[lo])

    def _push(self, node, lo, hi):
        if self.lazy[node] is not None and lo != hi:
            mid = (lo + hi) // 2
            self._set(2 * node, lo, mid, self.lazy[node])
            self._set(2 * node + 1, mid + 1, hi, self.lazy[node])
            self.lazy[node] = None

    def _first_less(self, node, lo, hi, height):
        if self.minimum[node] >= height:
            return self.n
        if lo == hi:
            return lo
        self._push(node, lo, hi)
        mid = (lo + hi) // 2
        if self.minimum[2 * node] < height:
            return self._first_less(2 * node, lo, mid, height)
        return self._first_less(2 * node + 1, mid + 1, hi, height)

    def _assign(self, node, lo, hi, left, right, height):
        if left <= lo and hi <= right:
            self._set(node, lo, hi, height)
            return
        self._push(node, lo, hi)
        mid = (lo + hi) // 2
        if left <= mid:
            self._assign(2 * node, lo, mid, left, right, height)
        if right > mid:
            self._assign(2 * node + 1, mid + 1, hi, left, right, height)
        self.minimum[node] = min(self.minimum[2 * node], self.minimum[2 * node + 1])
        self.total[node] = self.total[2 * node] + self.total[2 * node + 1]

    def insert(self, last, height):
        first = self._first_less(1, 0, self.n - 1, height)
        if first <= last:
            self._assign(1, 0, self.n - 1, first, last, height)
        return self.total[1]


@njit(cache=True)
def tree_set(node, lo, hi, value, mass_prefix, minimum, total, lazy, pending):
    minimum[node] = lazy[node] = value
    total[node] = value * (mass_prefix[hi + 1] - mass_prefix[lo])
    pending[node] = True


@njit(cache=True)
def tree_push(node, lo, hi, mass_prefix, minimum, total, lazy, pending):
    if pending[node] and lo != hi:
        mid = (lo + hi) // 2
        tree_set(2 * node, lo, mid, lazy[node], mass_prefix, minimum, total, lazy, pending)
        tree_set(2 * node + 1, mid + 1, hi, lazy[node], mass_prefix, minimum, total, lazy, pending)
        pending[node] = False


@njit(cache=True)
def skyline_insert(last, height, mass_prefix, minimum, total, lazy, pending):
    n = len(mass_prefix) - 1
    if minimum[1] >= height:
        return total[1]
    node, lo, hi = 1, 0, n - 1
    while lo != hi:
        tree_push(node, lo, hi, mass_prefix, minimum, total, lazy, pending)
        mid = (lo + hi) // 2
        if minimum[2 * node] < height:
            node, hi = 2 * node, mid
        else:
            node, lo = 2 * node + 1, mid + 1
    first = lo
    if first > last:
        return total[1]
    stack = [(1, 0, n - 1, 0)]
    while stack:
        node, lo, hi, after = stack.pop()
        if after:
            minimum[node] = min(minimum[2 * node], minimum[2 * node + 1])
            total[node] = total[2 * node] + total[2 * node + 1]
        elif first <= lo and hi <= last:
            tree_set(node, lo, hi, height, mass_prefix, minimum, total, lazy, pending)
        else:
            tree_push(node, lo, hi, mass_prefix, minimum, total, lazy, pending)
            mid = (lo + hi) // 2
            stack.append((node, lo, hi, 1))
            if first <= mid:
                stack.append((2 * node, lo, mid, 0))
            if last > mid:
                stack.append((2 * node + 1, mid + 1, hi, 0))
    return total[1]


@njit(cache=True)
def sweep_ranked(events, xmass, ymass, lastmass):
    d, event, area, answer = events.shape[1], 0, 0., 0.
    prefix = prefix_sum(lastmass)
    yprefix = prefix_sum(ymass)
    minimum = np.zeros(4 * len(ymass))
    total = np.zeros_like(minimum)
    lazy = np.zeros_like(minimum)
    pending = np.ones(len(minimum), dtype=np.bool_)
    for x in range(len(xmass) - 1, -1, -1):
        while event < len(events) and events[event, 0] == x:
            height = prefix[events[event, d - 1] + 1]
            if d == 2:
                area = max(area, height)
            else:
                area = skyline_insert(events[event, 1], height, yprefix, minimum, total, lazy, pending)
            event += 1
        answer += xmass[x] * area
    return answer


def sweep_low_dimension(boxes, unary, stats):
    stats['sweep_insertions'] += len(boxes)
    stats['sweep_cells'] += len(unary[0])
    return sweep_ranked(np.array(sorted(boxes, reverse=True), dtype=np.int64),
                       unary[0], unary[1], unary[len(unary) - 1])


class NumericalChanNumba:
    def __init__(self, dimension=None, base_hard=2, block_levels=None):
        if dimension is not None and (type(dimension) is not int or not 2 <= dimension <= 10):
            raise ValueError('dimension must be an integer from 2 through 10.')
        self.dimension = dimension
        if type(base_hard) is not int or base_hard < 0 or (block_levels is not None and
                (type(block_levels) is not int or block_levels < 1)):
            raise ValueError('Nonnegative constant base_hard and positive block length required.')
        self.base_hard, self.block_levels = base_hard, block_levels
        self.stats = Counter()

    def compute(self, points, magnitude=False):
        points = [tuple(float(x) for x in p) for p in points]
        self.d = self.dimension if self.dimension is not None else (len(points[0]) if points else 2)
        if not 2 <= self.d <= 10:
            raise ValueError('Input dimension must be from 2 through 10.')
        if any(len(p) != self.d or any(x < 0 or (not isinstance(x, (int, Fraction)) and not isfinite(x))
                                 for x in p) for p in points):
            raise ValueError('Finite nonnegative corners of one consistent dimension required.')
        self.stats = Counter(input_points=len(points), dimension=self.d)
        if not points:
            return 0
        coords = [ordered_unique([0] + [p[i] for p in points]) for i in range(self.d)]
        atom, density = (1., .5) if magnitude else (0., 1.)
        unary = {i: np.array([atom] + [density * (b - a) for a, b in zip(c, c[1:])])
                 for i, c in enumerate(coords)}
        boxes = [tuple(bisect_left(coords[i], p[i]) for i in range(self.d)) for p in points]
        self.stats['initial_cells'] = sum(map(len, coords))
        if self.d <= 3:
            return sweep_low_dimension(boxes, unary, self.stats)
        # Integer weights approximate 2**(sum(cyclic_positions)/d) from above.
        scale = self.scale = (len(points) + 2) ** 2
        self.weights = {s: ceil_root(scale ** self.d * (1 << s), self.d)
                        for s in range(3, 3 * self.d - 5)}
        total = prod(sum(w) for w in unary.values())
        terms = [Term(1, unary, {})]
        self.stats['initial_cells'] = sum(map(len, coords))
        uncovered = self._node(boxes, terms, (0,) * self.d,
                               tuple(len(c) - 1 for c in coords), 0, None, 0)
        return total - uncovered

    def _absorb(self, boxes, terms, lo, hi):
        hard, slabs, pairs = [], {}, {}
        for p in boxes:
            self.stats['box_classifications'] += 1
            if any(p[i] < lo[i] for i in range(self.d)):
                continue
            active = [i for i in range(self.d) if p[i] < hi[i]]
            if not active:
                return [], []
            if len(active) == 1:
                i = active[0]
                slabs[i] = max(slabs.get(i, -1), p[i])
            elif len(active) == 2:
                i, j = active
                pairs.setdefault((i, j), []).append((p[i], p[j]))
            else:
                hard.append(p)
        masks = []
        for (i, j), points in pairs.items():
            f = suffix_mask(np.array(points, dtype=np.int64), len(terms[0].unary[i]))
            masks.append((i, j, f))
        updated = []
        for old in terms:
            term = old.copy()
            for i, cut in slabs.items():
                multiply_unary(term, i, np.arange(len(term.unary[i])) > cut)
            for i, j, f in masks:
                add_edge(term, i, j, False, f)
            if term.alive():
                updated.append(term)
        self.stats['absorbed_boxes'] += len(boxes) - len(hard)
        return hard, updated

    def _base(self, boxes, terms, lo, hi):
        answer = 0
        for bits in product((0, 1), repeat=len(boxes)):
            upper = tuple(min([hi[i]] + [p[i] for p, bit in zip(boxes, bits) if bit]) for i in range(self.d))
            answer += (-1) ** sum(bits) * rectangle_sum(terms, lo, upper, self.stats)
        self.stats['base_calls'] += 1
        return answer

    def _node(self, boxes, terms, lo, hi, axis, remaining, generation):
        self.stats['nodes'] += 1
        self.stats['max_generation'] = max(self.stats['max_generation'], generation)
        if not terms:
            return 0
        boxes, terms = self._absorb(boxes, terms, lo, hi)
        if not terms:
            return 0
        if len(boxes) <= self.base_hard:
            return self._base(boxes, terms, lo, hi)
        if remaining == 0:
            ends = [ordered_unique([hi[i]] + [p[i] for p in boxes if lo[i] <= p[i] < hi[i]]) for i in range(self.d)]
            blocks = [list(zip([lo[i]] + [x + 1 for x in ends[i][:-1]], ends[i])) for i in range(self.d)]
            terms = push_blocks(terms, blocks, self.stats)
            boxes = [tuple(bisect_right(ends[i], p[i]) - 1 for i in range(self.d)) for p in boxes]
            lo, hi = (0,) * self.d, tuple(len(e) - 1 for e in ends)
            generation += 1
            remaining = None
        if remaining is None:
            potential = sum(self.weights[sum((i - axis) % self.d for i in triple)]
                            for p in boxes for triple in combinations(
                                [i for i in range(self.d) if lo[i] <= p[i] < hi[i]], 3))
            size = (potential + self.scale - 1) // self.scale
            remaining = (self.block_levels if self.block_levels is not None
                         else self.d * max(1, (size.bit_length() - 1) // (4 * self.d)))
            self.stats['blocks'] += 1
        cuts = []
        for p in boxes:
            active = [i for i in range(self.d) if lo[i] <= p[i] < hi[i]]
            weight = sum(self.weights[sum((i - axis) % self.d for i in triple)]
                         for triple in combinations(active, 3) if axis in triple)
            if weight:
                cuts.append((p[axis], weight))
        if not cuts:
            self.stats['axis_skips'] += 1
            return self._node(boxes, terms, lo, hi, (axis + 1) % self.d, remaining - 1, generation)
        cuts.sort()
        whole, accumulated = sum(w for x, w in cuts), 0
        for cut, weight in cuts:
            accumulated += weight
            if 2 * accumulated >= whole:
                break
        left_hi, right_lo = list(hi), list(lo)
        left_hi[axis], right_lo[axis] = cut, cut + 1
        self.stats['cuts'] += 1
        args = ((axis + 1) % self.d, remaining - 1, generation)
        return (self._node(boxes, terms, lo, tuple(left_hi), *args)
                + self._node(boxes, terms, tuple(right_lo), hi, *args))


def hypervolume(points, *, dimension=None, magnitude=False, **options):
    """Measure the union of [0,p] boxes; infer d from nonempty input if omitted."""
    return NumericalChanNumba(dimension=dimension, **options).compute(points, magnitude=magnitude)


def magnitude(points, *, dimension=None, **options):
    """Dominated-set l1 magnitude, including the mass at the origin."""
    return hypervolume(points, dimension=dimension, magnitude=True, **options)
