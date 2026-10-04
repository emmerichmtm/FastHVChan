"""4D convenience entry point for the adaptive compiled numerical engine."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from NumericalChanHVND.numerical_chan_compiled import NumericalChanCompiled


def hypervolume4(points, magnitude=False, **options):
    return NumericalChanCompiled(dimension=4, **options).compute(points, magnitude=magnitude)
