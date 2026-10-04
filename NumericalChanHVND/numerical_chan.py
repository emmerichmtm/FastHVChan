"""Numerical Chan recursion on finite weighted coordinate chains.

No quadrature, antiderivatives, continuous step-function objects, or external
dependencies. The state contains numerical unary arrays and integer monotone
cutoff arrays. Compression pushes cell MASS forward; it never averages a
density or divides by cell size. Use Fraction input for exact rational output.

Dimensions 2 and 3 use numerical sweeps in O(n log n) arithmetic operations.
Dimensions 4 through 10 use the locally blocked Chan recursion in
O~_d(n**(d/3)) operations. Odd dimensions end each pair round with a singleton.
The default local block schedule is proved in numerical_chan_2d_10d.tex.
block_levels is a testing override; its arbitrary use has no claimed time bound.
Both base_hard and dimension are constants in the complexity statement.
"""

from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, groupby, product
from math import isfinite, prod


@dataclass
class Term:
    coef: object
    unary: dict
    edges: dict

    def copy(self):
        return Term(self.coef, self.unary.copy(), self.edges.copy())

    def alive(self):
        return bool(self.coef) and all(any(w) for w in self.unary.values())


def direction(values):
    if all(a <= b for a, b in zip(values, values[1:])):
        return 1
    if all(a >= b for a, b in zip(values, values[1:])):
        return -1
    raise ValueError("A cutoff must be monotone; opposite directions stay separate.")


def ordered_unique(values):
    return [value for value, group in groupby(sorted(values))]


def multiply_unary(term, axis, values):
    term.unary[axis] = tuple(a * b for a, b in zip(term.unary[axis], values))
    return term.alive()


def add_edge(term, source, target, upper, values):
    """Conjoin target <= f(source), or target >= f(source); source < target."""
    assert source < target and len(values) == len(term.unary[source])
    size = len(term.unary[target])
    low, high = (-1, size - 1) if upper else (0, size)
    values = tuple(max(low, min(high, x)) for x in values)
    key = (source, target, upper, direction(values))
    if key in term.edges:
        op = min if upper else max
        values = tuple(op(a, b) for a, b in zip(values, term.edges.pop(key)))
    if min(values) == max(values):
        cap = values[0]
        return multiply_unary(term, target,
                              tuple(int(j <= cap if upper else j >= cap) for j in range(size)))
    term.edges[key] = values
    return multiply_unary(term, source,
                          tuple(int(v >= 0 if upper else v < size) for v in values))


def impose(term, first, second, strict=False):
    """Conjoin A <= B (or A < B). Bound = (axis, tuple), or (None, constant).

    Same-variable comparisons become unary Boolean arrays, even when they
    alternate many times. Different-variable comparisons become one monotone
    cutoff by binary searches. This is the only comparison constructor.
    """
    a, av = first
    b, bv = second
    test = (lambda x, y: x < y) if strict else (lambda x, y: x <= y)
    if a is None and b is None:
        return test(av, bv) and term.alive()
    if a == b:
        return multiply_unary(term, a, tuple(int(test(x, y)) for x, y in zip(av, bv)))
    if a is None:
        return multiply_unary(term, b, tuple(int(test(av, y)) for y in bv))
    if b is None:
        return multiply_unary(term, a, tuple(int(test(x, bv)) for x in av))
    if a < b:  # B(target) >= A(source), strict if requested.
        if direction(bv) == 1:
            search = bisect_right if strict else bisect_left
            return add_edge(term, a, b, False, tuple(search(bv, x) for x in av))
        negative = tuple(-x for x in bv)
        search = bisect_left if strict else bisect_right
        return add_edge(term, a, b, True, tuple(search(negative, -x) - 1 for x in av))
    # A(target) <= B(source), strict if requested.
    if direction(av) == 1:
        search = bisect_left if strict else bisect_right
        return add_edge(term, b, a, True, tuple(search(av, y) - 1 for y in bv))
    negative = tuple(-x for x in av)
    search = bisect_right if strict else bisect_left
    return add_edge(term, b, a, False, tuple(search(negative, -y) for y in bv))


def merge(terms, stats):
    """Incremental exact-key merging; not needed for the raw emission bound."""
    kept = {}
    for term in terms:
        stats['raw_terms'] += 1
        if not term.alive():
            continue
        for axis, values in term.unary.items():
            factor = next(x for x in values if x)
            if factor != 1:
                term.coef *= factor
                divisor = Fraction(factor) if isinstance(factor, (int, Fraction)) else factor
                term.unary[axis] = tuple(x / divisor for x in values)
        key = (tuple(sorted(term.unary.items())), tuple(sorted(term.edges.items())))
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
            prefix = [0]
            for value in weights:
                prefix.append(prefix[-1] + value)
            lowers, uppers = [(None, 0)], [(None, len(weights) - 1)]
            for key, values in list(term.edges.items()):
                i, j, upper, orient = key
                if axis not in (i, j):
                    continue
                del term.edges[key]
                if axis == j:
                    (uppers if upper else lowers).append((i, values))
                else:
                    targets = range(len(term.unary[j]))
                    if orient == 1:
                        cut = (tuple(bisect_left(values, y) for y in targets) if upper
                               else tuple(bisect_right(values, y) - 1 for y in targets))
                        (lowers if upper else uppers).append((j, cut))
                    else:
                        negative = tuple(-x for x in values)
                        cut = (tuple(bisect_right(negative, -y) - 1 for y in targets) if upper
                               else tuple(bisect_left(negative, -y) for y in targets))
                        (uppers if upper else lowers).append((j, cut))
            lowers, uppers = list(dict.fromkeys(lowers)), list(dict.fromkeys(uppers))
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
                            high = [prefix[av + 1]] * size if a is None else [prefix[x + 1] for x in av]
                            low = [prefix[bv]] * size if b is None else [prefix[x] for x in bv]
                            multiply_unary(winner, variable, tuple(x - y for x, y in zip(high, low)))
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
                                           tuple(prefix[x + offset] for x in values))
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
            term.unary[i + d] = (1,) * len(blocks[i])
        for i in range(d):
            identity = (i, tuple(range(len(term.unary[i]))))
            lower = (i + d, tuple(a for a, b in blocks[i]))
            upper = (i + d, tuple(b for a, b in blocks[i]))
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
            multiply_unary(term, i, tuple(int(lo[i] <= j <= hi[i]) for j in range(len(term.unary[i]))))
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


def sweep_low_dimension(boxes, unary, stats):
    """Numerical descending-x sweep in 2D or 3D, including anchor atoms."""
    d = len(unary)
    prefix = [0]
    for mass in unary[d - 1]:
        prefix.append(prefix[-1] + mass)
    skyline = PrefixSkyline(unary[1]) if d == 3 else None
    events = sorted(boxes, reverse=True)
    event, area, answer = 0, 0, 0
    for x in reversed(range(len(unary[0]))):
        while event < len(events) and events[event][0] == x:
            p = events[event]
            height = prefix[p[-1] + 1]
            area = max(area, height) if d == 2 else skyline.insert(p[1], height)
            event += 1
            stats['sweep_insertions'] += 1
        answer += unary[0][x] * area
        stats['sweep_cells'] += 1
    return answer


class NumericalChan:
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
        points = [tuple(p) for p in points]
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
        atom, density = (1, Fraction(1, 2)) if magnitude else (0, 1)
        unary = {i: (atom,) + tuple(density * (b - a) for a, b in zip(c, c[1:]))
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
            f = [-1] * len(terms[0].unary[i])
            for a, b in points:
                f[a] = max(f[a], b)
            for a in range(len(f) - 2, -1, -1):
                f[a] = max(f[a], f[a + 1])
            masks.append((i, j, tuple(x + 1 for x in f)))
        updated = []
        for old in terms:
            term = old.copy()
            for i, cut in slabs.items():
                multiply_unary(term, i, tuple(int(j > cut) for j in range(len(term.unary[i]))))
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
    return NumericalChan(dimension=dimension, **options).compute(points, magnitude=magnitude)


def magnitude(points, *, dimension=None, **options):
    """Dominated-set l1 magnitude, including the mass at the origin."""
    return hypervolume(points, dimension=dimension, magnitude=True, **options)
