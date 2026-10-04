"""Native finite-mass contractions with a locally restarted Chan schedule.

The local weighted face potential determines each block's number of cyclic
axis steps. At every checkpoint we compress if the representation is large,
then start a new block even when compression was unnecessary. A skipped axis
uses one step, exactly as a cut does. Dimensions two and three use the existing
compiled sweep. The native term algebra is shared with the adaptive backend.

For the fixed-dimension analysis, base_hard and compress_factor are constants.
No global interval override or option to disable compression is exposed.
"""
from itertools import combinations

from NumericalChanHVND.numerical_chan_compiled import NumericalChanCompiled
from NumericalChanHVND.numerical_chan_numba import NumericalChanNumba
from NumericalChanHVND import compiled_terms as kernel


class NumericalChanLocalCompiled(NumericalChanCompiled):
    """Compiled numerical Chan recursion with local checkpoint budgets."""

    def __init__(self, dimension=None, base_hard=2, compress_factor=8.0):
        # The parent validates dimensions, base size and a finite factor >= 0.
        super().__init__(dimension=dimension, base_hard=base_hard,
                         compress_factor=compress_factor)

    def compute(self, points, magnitude=False):
        # Bypass the parent's global interval calculation, retaining its
        # initial native state and cell/base/compression methods.
        value = NumericalChanNumba.compute(self, points, magnitude=magnitude)
        self.stats['local_schedule'] = 1
        self.stats['compression_enabled'] = 1
        self.stats['compression_factor'] = self.compress_factor
        return value

    def _new_block(self, boxes, lo, hi, axis):
        """L = d max(1, floor(log2(Q)/(4d))), computed using integers."""
        potential = sum(
            self.weights[sum((i - axis) % self.d for i in triple)]
            for point in boxes
            for triple in combinations(
                [i for i in range(self.d) if lo[i] <= point[i] < hi[i]], 3))
        # self.scale is the common denominator of the rounded face weights.
        q = (potential + self.scale - 1) // self.scale
        if q <= 0:
            raise AssertionError('A nonterminal hard box must expose a triple face.')
        levels = self.d * max(1, (q.bit_length() - 1) // (4 * self.d))
        self.stats['blocks'] += 1
        self.stats['min_block_levels'] = min(self.stats.get('min_block_levels', levels), levels)
        self.stats['max_block_levels'] = max(self.stats['max_block_levels'], levels)
        self.stats['max_block_potential'] = max(self.stats['max_block_potential'], q)
        return levels

    def _node(self, boxes, terms, lo, hi, axis, remaining, generation):
        self.stats['nodes'] += 1
        self.stats['max_generation'] = max(self.stats['max_generation'], generation)
        if not terms:
            return 0.
        cell = self._cell_state(boxes, terms, lo, hi)
        if cell is None:
            return 0.
        boxes, terms, lo, hi = cell
        if not terms:
            return 0.
        if len(boxes) <= self.base_hard:
            return self._base(boxes, terms, lo, hi)

        if remaining == 0:
            self.stats['checkpoints'] += 1
            self.stats['compression_checks'] += 1
            size = int(kernel.complexity(terms))
            if size > self.compress_factor * self.d**2 * (len(boxes) + 2):
                boxes, terms, lo, hi = self._compress_state(boxes, terms, lo, hi)
                generation += 1
                self.stats['max_generation'] = max(self.stats['max_generation'], generation)
                if not terms:
                    return 0.
            else:
                self.stats['compression_skipped_small_state'] += 1
            # Reset after BOTH decisions. Reusing an expired budget after a
            # skip would check every subsequent level, changing the schedule.
            remaining = None
        if remaining is None:
            remaining = self._new_block(boxes, lo, hi, axis)

        cuts = []
        for point in boxes:
            active = [i for i in range(self.d) if lo[i] <= point[i] < hi[i]]
            weight = sum(self.weights[sum((i - axis) % self.d for i in triple)]
                         for triple in combinations(active, 3) if axis in triple)
            if weight:
                cuts.append((point[axis], weight))

        following = ((axis + 1) % self.d, remaining - 1, generation)
        if not cuts:
            self.stats['axis_skips'] += 1
            return self._node(boxes, terms, lo, hi, *following)
        cuts.sort()
        total, accumulated = sum(weight for _, weight in cuts), 0
        for cut, weight in cuts:
            accumulated += weight
            if 2 * accumulated >= total:
                break
        left_hi, right_lo = list(hi), list(lo)
        left_hi[axis], right_lo[axis] = cut, cut + 1
        self.stats['cuts'] += 1
        return (self._node(boxes, terms, lo, tuple(left_hi), *following)
                + self._node(boxes, terms, tuple(right_lo), hi, *following))


def hypervolume(points, *, dimension=None, magnitude=False, **options):
    """Measure the union of nonnegative anchored boxes using local scheduling."""
    return NumericalChanLocalCompiled(dimension=dimension, **options).compute(
        points, magnitude=magnitude)


def magnitude(points, *, dimension=None, **options):
    """Dominated-set l1 magnitude, retaining all mass on coordinate faces."""
    return hypervolume(points, dimension=dimension, magnitude=True, **options)
