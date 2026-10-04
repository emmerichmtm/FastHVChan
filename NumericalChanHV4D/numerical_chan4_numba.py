"""4D entry point for the shared float64 Numba engine in NumericalChanHVND."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from NumericalChanHVND.numerical_chan_numba import NumericalChanNumba


class NumericalChan4Numba(NumericalChanNumba):
    def __init__(self, base_hard=2, block_levels=None):
        super().__init__(dimension=4, base_hard=base_hard, block_levels=block_levels)


def hypervolume4(points, magnitude=False, **options):
    return NumericalChan4Numba(**options).compute(points, magnitude=magnitude)
