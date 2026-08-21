"""Pytest wrappers around the modules' layered self-test suites, plus a few
independent cross-checks.  Run with:  python -m pytest tests/ -v
(or simply run each module directly for the same coverage).
"""

import itertools
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import chan_hypervolume as chv
import chan_orthant_dby3 as cod
import hv_baselines as hvb


def test_section2_suite():
    chv._self_test()


def test_baselines_suite():
    hvb._self_test()


def test_step_algebra():
    rng = random.Random(1)
    cod._test_step_ops(rng)
    cod._test_ray_boundary(rng)
    cod._test_less_than(rng)


def test_symbolic_integration():
    rng = random.Random(2)
    cod._test_integration(rng)


def test_compression():
    rng = random.Random(3)
    cod._test_compression(rng)
    cod._test_forced_compression(random.Random(4))


def test_dby3_end_to_end():
    rng = random.Random(5)
    cod._test_full_algorithm(rng)
    cod._test_against_section2(rng)


def test_mixed_orientation_orthants():
    cod._test_staircase_classes(random.Random(7))
    cod._test_mixed_orthants(random.Random(8))


def _reference_union(boxes):
    """O(2^n) inclusion-exclusion, independent of both modules' internals."""
    total = 0.0
    for size in range(1, len(boxes) + 1):
        for combo in itertools.combinations(boxes, size):
            lo = [max(b[0][k] for b in combo) for k in range(len(combo[0][0]))]
            hi = [min(b[1][k] for b in combo) for k in range(len(combo[0][0]))]
            vol = 1.0
            for a, b in zip(lo, hi):
                vol *= max(0.0, b - a)
            total += vol if size % 2 else -vol
    return total


def test_cross_module_agreement():
    rng = random.Random(6)
    for d in (3, 4):
        for n in (5, 9, 13):
            pts = [tuple(rng.random() for _ in range(d)) for _ in range(n)]
            ref = tuple(1.0 for _ in range(d))
            a = chv.hypervolume(pts, ref)
            b = cod.hypervolume_dby3(pts, ref)
            c = _reference_union([(p, ref) for p in pts])
            assert abs(a - b) < 1e-9 * max(1.0, c)
            assert abs(a - c) < 1e-9 * max(1.0, c)


def test_edge_cases():
    assert chv.hypervolume([], (1, 1, 1)) == 0.0
    assert cod.hypervolume_dby3([], (1, 1, 1)) == 0.0
    assert cod.hypervolume_dby3([(2, 2, 2)], (1, 1, 1)) == 0.0  # behind ref
    a = chv.hypervolume([(0, 0, 0)], (1, 2, 3))
    b = cod.hypervolume_dby3([(0, 0, 0)], (1, 2, 3))
    assert abs(a - 6.0) < 1e-12 and abs(b - 6.0) < 1e-12
    # duplicates and dominated points
    pts = [(0.5, 0.5, 0.5)] * 4 + [(0.6, 0.6, 0.6)]
    assert abs(cod.hypervolume_dby3(pts, (1, 1, 1)) - 0.125) < 1e-12


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            print(f"-- {name}")
            fn()
    print("all tests passed")
