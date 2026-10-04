"""Bounded, independent 4D checks for the whole-loop float64 JIT backend.

Run from any directory with a Python environment containing NumPy and Numba::

    python NumericalChanHV4D/verify_whole_loop.py

Small answers use exact rational inclusion--exclusion. A 20-point case checks
the normal compression schedule against the existing shared backend; two
12-point cases exercise repeated compression with a test-only shorter block.
This is a correctness suite, not a timing or asymptotic-complexity experiment.
"""

from collections import Counter
from fractions import Fraction
from itertools import combinations, product
from math import isclose, prod, sqrt
from pathlib import Path
import argparse
import json
import random
import time

import numpy as np
from numba import types
from numba.typed import Dict, List

import numerical_chan4_compiled as compiled
from NumericalChanHVND import numerical_chan_numba as shared


REL_TOL = 2e-9
ABS_TOL = 2e-10


def inclusion_exclusion(corners, magnitude=False):
    """Independent exact oracle, including the atom at the origin."""
    points = [tuple(Fraction(x) for x in point) for point in corners]
    answer = Fraction(0)
    for size in range(1, len(points) + 1):
        for subset in combinations(points, size):
            corner = [min(point[axis] for point in subset) for axis in range(4)]
            mass = prod(1 + x / 2 for x in corner) if magnitude else prod(corner)
            answer += mass if size % 2 else -mass
    return answer


def small_cases():
    rng = random.Random(94731)
    cases = [
        ("empty", []),
        ("origin", [(0, 0, 0, 0)]),
        ("zero-face", [(0, 2, 3, 4)]),
        ("several-zero-faces", [(0, 2, 3, 4), (3, 0, 4, 2), (2, 3, 0, 4)]),
        ("duplicate", [(1, 2, 3, 4)] * 3),
        ("nested", [(0, 0, 0, 0), (1, 2, 3, 4), (2, 3, 4, 5)]),
        ("overlap", [(1, 4, 3, 2), (3, 2, 1, 4)]),
        ("antichain-ties", [(1, 4, 3, 2), (2, 3, 4, 1),
                            (3, 2, 1, 4), (4, 1, 2, 3)]),
        ("mixed-scale", [(Fraction(1, 1000), 20, 3, 4),
                          (Fraction(1, 100), 2, 30, 4),
                          (Fraction(1, 10), 2, 3, 40)]),
    ]
    for n in (2, 3, 5, 8):
        for repeat in range(3):
            cases.append((f"tied-random-{n}-{repeat}",
                          [tuple(rng.randrange(7) for _ in range(4)) for _ in range(n)]))
    for repeat in range(4):
        cases.append((f"rational-random-{repeat}",
                      [tuple(Fraction(rng.randrange(20), rng.choice((3, 5, 7)))
                             for _ in range(4)) for _ in range(5)]))
    return cases


def sphere_corners():
    # Reproduce the positive-minimization benchmark family, then reflect once
    # into the nonnegative upper-corner convention of the numerical solvers.
    rng = random.Random(42)
    for n in (5, 10, 20):
        points = []
        for _ in range(n):
            point = [abs(rng.gauss(0, 1)) + 1e-12 for _ in range(4)]
            norm = sqrt(sum(x * x for x in point))
            points.append(tuple(1.2 - x / norm for x in point))
    return points


def repeated_compression_corners():
    rng = random.Random(1842)
    for n in (0, 1, 2, 4, 8, 12):
        points = [tuple(rng.randrange(8) for _ in range(4)) for _ in range(n)]
    return points


def explicit_value(terms, index):
    """Evaluate the defining product and predicates, without elimination."""
    answer = 0.0
    for term in terms:
        weight = term.coef * prod(values[index[axis]] for axis, values in term.unary.items())
        for (source, target, upper, _), values in term.edges.items():
            cut = values[index[source]]
            if (upper and index[target] > cut) or (not upper and index[target] < cut):
                weight = 0.0
                break
        answer += weight
    return answer


def native_blocks(blocks):
    result = List.empty_list(types.int64[:, ::1])
    for axis in blocks:
        result.append(np.asarray(axis, dtype=np.int64))
    return result


def check_term_algebra(check, checks):
    """Sixteen tiny signed terms; every output cell has an explicit-sum oracle."""
    rng = random.Random(331045)
    slots = ((True, 1), (True, -1), (False, 1), (False, -1))
    for repeat in range(16):
        dimension = (2, 3, 4)[repeat % 3]
        sizes = [4] * dimension
        unary = {axis: np.array([rng.choice((-2, -1, 1, 2)) for _ in range(4)], dtype=float)
                 for axis in range(dimension)}
        edges = {}
        active = [slots[repeat % 4]] if repeat < 8 else slots
        for upper, orientation in active:
            values = [1, 2, 3, 3] if upper else [0, 0, 1, 1]
            if orientation == -1:
                values = values[::-1]
            edges[(0, 1, upper, orientation)] = np.array(values, dtype=np.int64)
        for axis in range(1, dimension - 1):
            upper, orientation = slots[(repeat + axis) % 4]
            values = sorted([rng.randrange(-1, 5) for _ in range(4)], reverse=orientation == -1)
            edges[(axis, axis + 1, upper, orientation)] = np.clip(
                np.array(values, dtype=np.int64), -1 if upper else 0, 3 if upper else 4)
        term = shared.Term(float(rng.choice((-2, -1, 1, 2))), unary, edges)
        source = [term]
        packed = compiled.kernel.pack(source)
        fine = {q: explicit_value(source, q) for q in product(range(4), repeat=dimension)}
        rectangles = [([0] * dimension, [3] * dimension),
                      ([1] * dimension, [2] * dimension),
                      ([2] * dimension, [1] * dimension)]
        for lo, hi in rectangles:
            actual = compiled.kernel.rectangle_sum(packed, np.array(lo, dtype=np.int64),
                                                   np.array(hi, dtype=np.int64),
                                                   np.zeros(len(compiled.kernel.COUNTERS), dtype=np.int64))
            expected = sum(value for q, value in fine.items()
                           if all(lo[axis] <= q[axis] <= hi[axis] for axis in range(dimension)))
            check(actual, expected, ("signed rectangle", repeat, lo, hi))
            checks["signed_term_rectangle_sums"] += 1

        blocks = [[(0, 1), (2, 3)]] * dimension
        coarse = compiled.kernel.push_blocks(packed, native_blocks(blocks),
                                              np.zeros(len(compiled.kernel.COUNTERS), dtype=np.int64))
        coarse_reference = shared.push_blocks(source, blocks, Counter())
        for q in product(range(2), repeat=dimension):
            expected = sum(value for p, value in fine.items()
                           if tuple(x // 2 for x in p) == q)
            check(explicit_value(coarse, q), expected, ("compression cell", repeat, q))
            check(explicit_value(coarse_reference, q), expected, ("reference cell", repeat, q))
            checks["signed_first_compression_cells"] += 1

        # Add a changing coarse-grid mask, then coarsen again. The independent
        # oracle still sums original fine cells, including their signed weights.
        masked = []
        for old in coarse:
            new = shared.Term(old.coef, dict(old.unary.items()),
                              {(i, j, bool(upper), orient): values
                               for (i, j, upper, orient), values in old.edges.items()})
            if shared.add_edge(new, 0, dimension - 1, False, np.array([1, 0], dtype=np.int64)):
                masked.append(new)
        twice = compiled.kernel.push_blocks(compiled.kernel.pack(masked),
                                             native_blocks([[(0, 1)]] * dimension),
                                             np.zeros(len(compiled.kernel.COUNTERS), dtype=np.int64))
        expected = sum(value for p, value in fine.items() if p[-1] // 2 >= 1 - p[0] // 2)
        check(explicit_value(twice, (0,) * dimension), expected, ("second compression", repeat))
        checks["signed_repeated_compression_cells"] += 1
        # Array/dictionary copies inside contraction must not alter the input.
        for q, expected in fine.items():
            check(explicit_value(packed, q), expected, ("immutable packed input", repeat, q))
        checks["packed_input_immutability"] += 1

    # Exact cancellation must survive hashing, merging, and coefficient updates.
    negative = shared.Term(-term.coef, term.unary.copy(), term.edges.copy())
    cancelled = compiled.kernel.pack([term, negative])
    actual = compiled.kernel.rectangle_sum(cancelled, np.zeros(dimension, dtype=np.int64),
                                           np.full(dimension, 3, dtype=np.int64),
                                           np.zeros(len(compiled.kernel.COUNTERS), dtype=np.int64))
    check(actual, 0.0, "equal-state cancellation")
    checks["equal_state_cancellation"] += 1

    # Force a collision without searching for one: seed the hash bucket with an
    # unequal state, then verify exact comparison before adding a true duplicate.
    incoming = shared.Term(1.0, {0: np.array([1.0, 2.0])}, {})
    different = shared.Term(1.0, {0: np.array([1.0, 3.0])}, {})
    native_incoming = compiled.kernel.pack([incoming])[0]
    kept = compiled.kernel.pack([different])
    heads = Dict.empty(types.uint64, types.int64)
    code = np.uint64(compiled.kernel.state_hash(native_incoming))
    heads[code] = 0
    links = List.empty_list(types.int64)
    links.append(-1)
    counters = np.zeros(len(compiled.kernel.COUNTERS), dtype=np.int64)
    compiled.kernel.emit(compiled.kernel.copy_term(native_incoming), kept, heads, links, counters)
    assert len(kept) == 2
    compiled.kernel.emit(compiled.kernel.copy_term(native_incoming), kept, heads, links, counters)
    assert len(kept) == 2
    for index in ((0,), (1,)):
        check(explicit_value(kept, index), explicit_value([different, incoming, incoming], index),
              ("forced hash collision", index))
    checks["hash_collision_and_duplicate_merging"] += 1

    # A genuine duplicate beyond the 16-link search cap may remain separate;
    # preserving its numerical contribution is the required behavior.
    chain = [incoming] + [shared.Term(1.0, {0: np.array([1.0, float(k + 3)])}, {})
                          for k in range(16)]
    kept = compiled.kernel.pack(chain)
    heads[code] = 16
    links = List.empty_list(types.int64)
    for index in range(17):
        links.append(index - 1)
    compiled.kernel.emit(compiled.kernel.copy_term(native_incoming), kept, heads, links, counters)
    assert len(kept) == 18
    for index in ((0,), (1,)):
        check(explicit_value(kept, index), explicit_value(chain + [incoming], index),
              ("bounded hash collision chain", index))
    checks["bounded_collision_chain_preserves_sum"] += 1

    try:
        compiled.kernel.base_sum(compiled.kernel.pack([term]), np.zeros((63, 4), dtype=np.int64),
                                 np.zeros(dimension, dtype=np.int64),
                                 np.full(dimension, 3, dtype=np.int64), counters)
    except ValueError:
        checks["base_bitmask_overflow_guard"] += 1
    else:
        raise AssertionError("A 63-box base must reject int64 subset-mask overflow")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("whole_loop_verification.json"))
    args = parser.parse_args()
    start = time.perf_counter()
    checks, records = Counter(), []
    max_error = 0.0

    def check(actual, expected, label):
        nonlocal max_error
        actual, expected = float(actual), float(expected)
        assert isclose(actual, expected, rel_tol=REL_TOL, abs_tol=ABS_TOL), (
            label, actual, expected)
        max_error = max(max_error, abs(actual - expected) / max(1.0, abs(expected)))

    def record(label, solver, magnitude, actual):
        records.append({"case": label, "magnitude": magnitude, "value": float(actual),
                        "stats": {key: int(value) for key, value in solver.stats.items()}})

    check_term_algebra(check, checks)
    print("Signed contractions and cellwise repeated compression passed.", flush=True)

    cases = small_cases()
    # Reuse an instance across calls and measures to catch leaked recursion or
    # compiled workspace state, while still obtaining each case's own counters.
    solver = compiled.NumericalChan4D(base_hard=1)
    for label, points in cases:
        for magnitude in (False, True):
            actual = solver.compute(points, magnitude=magnitude)
            check(actual, inclusion_exclusion(points, magnitude), (label, magnitude))
            checks["exact_inclusion_exclusion_instances"] += 1
            record(label, solver, magnitude, actual)
    print("Small exact-oracle cases passed.", flush=True)

    # This identity detects lost atoms at zero coordinates without sharing the
    # magnitude-weight implementation with the independent ordinary-HV call.
    for label, points in cases[:9]:
        shifted = [tuple(1 + Fraction(x) / 2 for x in point) for point in points]
        direct = compiled.hypervolume4(points, magnitude=True)
        transformed = compiled.hypervolume4(shifted)
        check(direct, transformed, (label, "affine magnitude identity"))
        check(direct, inclusion_exclusion(points, True), (label, "public wrapper"))
        checks["magnitude_affine_equivalence"] += 1
        checks["public_wrapper_exact_checks"] += 1

    # Ordering and repeated generators must not change either measure.
    points = cases[-1][1]
    for magnitude in (False, True):
        actual = compiled.hypervolume4(list(reversed(points)) + points[:2], magnitude=magnitude)
        check(actual, inclusion_exclusion(points, magnitude), ("reorder-duplicate", magnitude))
        checks["reordering_and_duplicate_invariance"] += 1

    # Exercise the actual default schedule. The older backend is used only for
    # this larger case; it is not an oracle for the preceding small instances.
    points = sphere_corners()
    for magnitude in (False, True):
        solver = compiled.NumericalChan4D()
        reference = shared.NumericalChanNumba(dimension=4)
        actual = solver.compute(points, magnitude=magnitude)
        expected = reference.compute(points, magnitude=magnitude)
        check(actual, expected, ("default-compression-20", magnitude))
        assert solver.stats["compressions"] > 0, dict(solver.stats)
        assert reference.stats["compressions"] > 0, dict(reference.stats)
        checks["default_compression_shared_backend_comparisons"] += 1
        record("default-compression-20", solver, magnitude, actual)
    print("Default 20-point compression cases passed.", flush=True)

    # A deliberately aggressive schedule makes repeated compression observable
    # on a modest exact-oracle input. It is not the default theoretical schedule.
    points = repeated_compression_corners()
    for magnitude in (False, True):
        solver = compiled.NumericalChan4D(base_hard=1, block_levels=2)
        actual = solver.compute(points, magnitude=magnitude)
        check(actual, inclusion_exclusion(points, magnitude), ("repeated-compression-12", magnitude))
        assert solver.stats["max_generation"] >= 2, dict(solver.stats)
        checks["repeated_compression_exact_instances"] += 1
        record("repeated-compression-12", solver, magnitude, actual)
    print("Repeated compression cases passed.", flush=True)

    report = {
        "status": "All 4D whole-loop JIT checks passed.",
        "checks": dict(checks),
        "max_scaled_error": max_error,
        "tolerance": {"relative": REL_TOL, "absolute": ABS_TOL},
        "seconds_including_compilation": time.perf_counter() - start,
        "scope": "4D numerical correctness only; float64 arithmetic; no performance claim.",
        "cases": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
