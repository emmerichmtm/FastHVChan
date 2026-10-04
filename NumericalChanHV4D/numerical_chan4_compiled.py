"""4D Chan recursion with whole term contractions compiled by Numba.

The Python geometric recursion and its compression schedule are inherited.
Base evaluation and block compression each enter compiled code only once.
The float64 backend supports ordinary HV and direct dominated-set magnitude.
"""
from pathlib import Path
import sys
import numpy as np
from numba import types
from numba.typed import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from NumericalChanHVND.numerical_chan_numba import NumericalChanNumba, Term
try:
    from . import compiled_terms as kernel
except ImportError:  # Also support running directly from this directory.
    import compiled_terms as kernel


class NumericalChan4D(NumericalChanNumba):
    def __init__(self, base_hard=2, block_levels=None):
        super().__init__(dimension=4, base_hard=base_hard, block_levels=block_levels)

    def _base(self, boxes, terms, lo, hi):
        counters = np.zeros(len(kernel.COUNTERS), dtype=np.int64)
        answer = kernel.base_sum(kernel.pack(terms), np.asarray(boxes, dtype=np.int64).reshape(-1, 4),
                                 np.asarray(lo, dtype=np.int64), np.asarray(hi, dtype=np.int64), counters)
        kernel.record_counters(self.stats, counters)
        self.stats['base_calls'] += 1
        return answer

    def _compress(self, terms, blocks):
        counters = np.zeros(len(kernel.COUNTERS), dtype=np.int64)
        native_blocks = List.empty_list(types.int64[:, ::1])
        for axis_blocks in blocks:
            native_blocks.append(np.asarray(axis_blocks, dtype=np.int64))
        result = kernel.push_blocks(kernel.pack(terms), native_blocks, counters)
        kernel.record_counters(self.stats, counters)
        return [Term(term.coef, dict(term.unary.items()),
                     {(i, j, bool(upper), orient): values for (i, j, upper, orient), values in term.edges.items()})
                for term in result]


def hypervolume4(points, magnitude=False, **options):
    return NumericalChan4D(**options).compute(points, magnitude=magnitude)
