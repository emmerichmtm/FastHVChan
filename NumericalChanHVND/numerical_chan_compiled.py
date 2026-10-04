"""Finite-mass Chan recursion with adaptive compression and native contractions.

Dimensions 2/3 retain the compiled sweep. In 4-10D, the geometric driver follows
the repository's Chan d/3 implementation: shrink slabs, prune to the current
cell, and compress only after both a depth and a representation-size test.
All numerical state operations run in Numba. No quadrature or fastmath is used.

The earlier locally scheduled engines remain available unchanged. Performance
of this adaptive policy is measured separately; its complexity is not inferred
from their block-schedule proof. All cell bounds below are inclusive integers,
so the same implementation preserves magnitude's positive mass at the origin.
"""
from bisect import bisect_right
from itertools import combinations
import math
from pathlib import Path
import sys

import numpy as np
from numba import types
from numba.typed import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from NumericalChanHVND.numerical_chan_numba import NumericalChanNumba, Term, suffix_mask
from NumericalChanHVND import compiled_terms as kernel


class NumericalChanCompiled(NumericalChanNumba):
    """Compiled finite-mass backend for 2D-10D, with compression enabled."""

    def __init__(self, dimension=None, base_hard=2, compress_every=None,
                 compress_factor=8.0, use_compression=True):
        super().__init__(dimension=dimension, base_hard=base_hard)
        if compress_every is not None and (type(compress_every) is not int or compress_every < 1):
            raise ValueError('compress_every must be a positive integer or None.')
        if not math.isfinite(compress_factor) or compress_factor < 0:
            raise ValueError('compress_factor must be finite and nonnegative.')
        self.compress_every = compress_every
        self.compress_factor = float(compress_factor)
        self.use_compression = bool(use_compression)

    def compute(self, points, magnitude=False):
        points = list(points)
        dimension = self.dimension if self.dimension is not None else (len(points[0]) if points else 2)
        self._interval = (self.compress_every if self.compress_every is not None else
                          max(1, round(.1 * dimension * math.log2(max(2, len(points))))))
        value = super().compute(points, magnitude=magnitude)
        self.stats['compression_interval'] = self._interval
        self.stats['compression_enabled'] = int(self.use_compression)
        return value

    def _initial_terms(self, unary):
        return kernel.pack([Term(1., unary, {})])

    def _counter_array(self):
        return np.zeros(len(kernel.COUNTERS), dtype=np.int64)

    def _base(self, boxes, terms, lo, hi):
        counters = self._counter_array()
        value = kernel.base_sum(terms, np.asarray(boxes, dtype=np.int64).reshape(-1, self.d),
                                np.asarray(lo, dtype=np.int64), np.asarray(hi, dtype=np.int64), counters)
        kernel.record_counters(self.stats, counters)
        self.stats['base_calls'] += 1
        return value

    def _cell_state(self, boxes, terms, lo, hi):
        """Shrink slab complements, crop arrays, then absorb pair constraints."""
        original_count = len(boxes)
        lo, hi = list(lo), list(hi)
        while True:
            remaining, shrunk = [], False
            for point in boxes:
                self.stats['box_classifications'] += 1
                if any(point[i] < lo[i] for i in range(self.d)):
                    continue
                active = [i for i in range(self.d) if point[i] < hi[i]]
                if not active:
                    return None
                if len(active) == 1:
                    axis = active[0]
                    # Exclude exactly the ranks covered by this anchored slab.
                    lo[axis] = max(lo[axis], point[axis] + 1)
                    shrunk = True
                else:
                    remaining.append(point)
            if any(a > b for a, b in zip(lo, hi)):
                return None
            boxes = remaining
            if not shrunk:
                break

        local_hi = tuple(b - a for a, b in zip(lo, hi))
        hard, pairs = [], {}
        for point in boxes:
            active = [i for i in range(self.d) if point[i] < hi[i]]
            local = tuple(min(point[i], hi[i]) - lo[i] for i in range(self.d))
            if len(active) == 2:
                i, j = active
                pairs.setdefault((i, j), []).append((local[i], local[j]))
            else:
                hard.append(local)
        counters = self._counter_array()
        terms = kernel.crop_terms(terms, np.asarray(lo, dtype=np.int64),
                                  np.asarray(hi, dtype=np.int64), counters)
        kernel.record_counters(self.stats, counters)
        if not terms:
            return None
        if pairs:
            masks = List.empty_list(kernel.MASK)
            for (i, j), points in pairs.items():
                cutoff = suffix_mask(np.asarray(points, dtype=np.int64), local_hi[i] + 1)
                masks.append((i, j, cutoff))
            terms = kernel.apply_easy(terms, np.empty((0, 2), dtype=np.int64), masks)
        self.stats['absorbed_boxes'] += original_count - len(hard)
        return hard, terms, (0,) * self.d, local_hi

    def _compress_state(self, boxes, terms, lo, hi):
        ends = [sorted({hi[i]} | {point[i] for point in boxes if lo[i] <= point[i] < hi[i]})
                for i in range(self.d)]
        blocks = List.empty_list(types.int64[:, ::1])
        for i in range(self.d):
            starts = [lo[i]] + [value + 1 for value in ends[i][:-1]]
            blocks.append(np.asarray(list(zip(starts, ends[i])), dtype=np.int64))
        counters = self._counter_array()
        terms = kernel.push_blocks(terms, blocks, counters)
        kernel.record_counters(self.stats, counters)
        boxes = [tuple(bisect_right(ends[i], point[i]) - 1 for i in range(self.d)) for point in boxes]
        return boxes, terms, (0,) * self.d, tuple(len(values) - 1 for values in ends)

    def _node(self, boxes, terms, lo, hi, axis, since_compression, generation):
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

        since_compression = 0 if since_compression is None else since_compression
        if self.use_compression and since_compression >= self._interval:
            self.stats['compression_checks'] += 1
            size = kernel.complexity(terms)
            if size > self.compress_factor * self.d**2 * (len(boxes) + 2):
                boxes, terms, lo, hi = self._compress_state(boxes, terms, lo, hi)
                since_compression = 0
                generation += 1
                self.stats['max_generation'] = max(self.stats['max_generation'], generation)
                if not terms:
                    return 0.
            else:
                self.stats['compression_skipped_small_state'] += 1

        # Chan's cyclic weighted-median cuts; skip axes without a hard face.
        for probe in range(self.d):
            current_axis = (axis + probe) % self.d
            cuts = []
            for point in boxes:
                if point[current_axis] >= hi[current_axis]:
                    continue
                others = [i for i in range(self.d) if i != current_axis and point[i] < hi[i]]
                weight = sum(self.weights[sum((i - current_axis) % self.d for i in (current_axis, j, k))]
                             for j, k in combinations(others, 2))
                if weight:
                    cuts.append((point[current_axis], weight))
            if cuts:
                break
            self.stats['axis_skips'] += 1
        else:
            raise AssertionError('A hard box must expose an eligible cut.')
        cuts.sort()
        total, accumulated = sum(weight for _, weight in cuts), 0
        for cut, weight in cuts:
            accumulated += weight
            if 2 * accumulated >= total:
                break
        left_hi, right_lo = list(hi), list(lo)
        left_hi[current_axis], right_lo[current_axis] = cut, cut + 1
        self.stats['cuts'] += 1
        following = ((current_axis + 1) % self.d, since_compression + 1, generation)
        return (self._node(boxes, terms, lo, tuple(left_hi), *following)
                + self._node(boxes, terms, tuple(right_lo), hi, *following))


def hypervolume(points, *, dimension=None, magnitude=False, **options):
    return NumericalChanCompiled(dimension=dimension, **options).compute(points, magnitude=magnitude)


def magnitude(points, *, dimension=None, **options):
    return hypervolume(points, dimension=dimension, magnitude=True, **options)
