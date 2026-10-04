"""Stable 4D entry point: native term states and complete compiled contractions."""
if __package__:
    from .numerical_chan4_compiled import NumericalChan4D
else:
    from numerical_chan4_compiled import NumericalChan4D


class NumericalChan4Numba(NumericalChan4D):
    """Backward-compatible class name for the whole-contraction 4D backend."""


def hypervolume4(points, magnitude=False, **options):
    return NumericalChan4Numba(**options).compute(points, magnitude=magnitude)
