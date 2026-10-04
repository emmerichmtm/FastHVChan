"""Bounded independent checks for the adaptive compiled 2--10D solver.

Run with NumPy and Numba installed::

    python NumericalChanHVND/verify_compiled.py

Small end-to-end answers use exact rational inclusion--exclusion. Kernel
checks explicitly enumerate small signed grids; this enumeration belongs only
to the oracle, never to the solver. Timings include compilation and are not
performance measurements.
"""

from collections import Counter
from fractions import Fraction
from itertools import combinations, product
from math import isclose, prod
from pathlib import Path
import argparse
import json
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from numba import types
from numba.typed import List

from NumericalChanHVND import compiled_terms as kernel
from NumericalChanHVND import numerical_chan_compiled as compiled
from NumericalChanHVND import numerical_chan as reference
from NumericalChanHVND.numerical_chan_numba import Term
from chan_hypervolume import hypervolume as chan_dby2


REL_TOL = 2e-9
ABS_TOL = 2e-10


def inclusion_exclusion(points, magnitude=False):
    """Exact anchored-box union; zero extents retain their magnitude atoms."""
    points = [tuple(Fraction(x) for x in point) for point in points]
    answer = Fraction(0)
    for size in range(1, len(points) + 1):
        for chosen in combinations(points, size):
            corner = [min(point[i] for point in chosen) for i in range(len(points[0]))]
            mass = prod(1 + x / 2 for x in corner) if magnitude else prod(corner)
            answer += mass if size % 2 else -mass
    return answer


def small_cases(dimension):
    rng = random.Random(721003 + dimension)
    cases = [
        ("empty", []),
        ("origin", [(0,) * dimension]),
        ("zero-faces", [(0,) + (3,) * (dimension - 1),
                         (3, 0) + (2,) * (dimension - 2)]),
        ("duplicates-nested", [(1,) * dimension, (3,) * dimension,
                                (3,) * dimension, (0,) * dimension]),
        ("tied-antichain", [tuple(1 + (i + j) % 5 for i in range(dimension))
                            for j in range(5)]),
        ("random-rational", [tuple(Fraction(rng.randrange(9), rng.choice((2, 3, 5)))
                                    for _ in range(dimension)) for _ in range(5)]),
    ]
    if dimension in (2, 4, 7, 10):
        cases.append(("mixed-scale", [tuple(Fraction(1, 1000) if i % 3 == 0 else
                                              (100 if i % 3 == 1 else 2)
                                              for i in range(dimension)),
                                        tuple(Fraction(1, 100) if i % 3 == 0 else
                                              (10 if i % 3 == 1 else 3)
                                              for i in range(dimension))]))
    return cases


def value_at(terms, index):
    """Direct defining product, with no prefix sums or variable elimination."""
    answer = 0.0
    for term in terms:
        value = term.coef * prod(weights[index[axis]] for axis, weights in term.unary.items())
        for (source, target, upper, _), values in term.edges.items():
            cut = values[index[source]]
            if (upper and index[target] > cut) or (not upper and index[target] < cut):
                value = 0.0
                break
        answer += value
    return answer


def native_blocks(blocks):
    result = List.empty_list(types.int64[:, ::1])
    for axis in blocks:
        result.append(np.asarray(axis, dtype=np.int64))
    return result


def fresh_counters():
    return np.zeros(len(kernel.COUNTERS), dtype=np.int64)


def kernel_checks(check, counts):
    # All four order/orientation classes, plus dimensions unused by any edge.
    # Unused axes still have nonunit signed mass and cannot be silently dropped.
    slots = ((True, 1), (True, -1), (False, 1), (False, -1))
    for dimension in (2, 4, 7, 10):
        active = min(dimension, 4)
        sizes = [4] * active + [1] * (dimension - active)
        for upper, orientation in slots:
            unary = {axis: np.array([1., -2., 0., 3.] if axis < active else [-2.])
                     for axis in range(dimension)}
            values = [-1, 0, 2, 3] if upper else [0, 1, 3, 4]
            if orientation == -1:
                values = values[::-1]
            edges = {(0, 1, upper, orientation): np.array(values, dtype=np.int64)}
            if active >= 4:
                edges[(1, 2, False, -1)] = np.array([3, 2, 1, 0], dtype=np.int64)
                edges[(2, 3, True, 1)] = np.array([0, 1, 3, 3], dtype=np.int64)
            original = [Term(-0.5, unary, edges)]
            packed = kernel.pack(original)
            fine = {q: value_at(original, q) for q in product(*(range(s) for s in sizes))}

            ranges = [([0] * dimension, [s - 1 for s in sizes]),
                      ([1] * active + [0] * (dimension - active),
                       [3] * active + [0] * (dimension - active)),
                      ([0] * dimension, [0] + [s - 1 for s in sizes[1:]]),
                      ([3] + [0] * (dimension - 1), [s - 1 for s in sizes])]
            for lo, hi in ranges:
                cropped = kernel.crop_terms(packed, np.array(lo, dtype=np.int64),
                                            np.array(hi, dtype=np.int64), fresh_counters())
                crop_sizes = [b - a + 1 for a, b in zip(lo, hi)]
                expected_sum = 0.0
                for q in product(*(range(s) for s in crop_sizes)):
                    original_index = tuple(x + offset for x, offset in zip(q, lo))
                    expected = fine[original_index]
                    check(value_at(cropped, q), expected,
                          ("inclusive crop", dimension, upper, orientation, lo, hi, q))
                    expected_sum += expected
                    counts["inclusive_signed_crop_cells"] += 1
                actual_sum = kernel.rectangle_sum(cropped, np.zeros(dimension, dtype=np.int64),
                                                  np.array(crop_sizes, dtype=np.int64) - 1,
                                                  fresh_counters())
                check(actual_sum, expected_sum,
                      ("cropped rectangle sum", dimension, upper, orientation, lo, hi))
                counts["cropped_signed_rectangle_sums"] += 1

            blocks = [[(0, 1), (2, 3)] if s == 4 else [(0, 0)] for s in sizes]
            coarse = kernel.push_blocks(packed, native_blocks(blocks), fresh_counters())
            expected_coarse = {}
            for q in product(*(range(len(b)) for b in blocks)):
                expected = sum(weight for p, weight in fine.items()
                               if tuple(x // 2 if i < active else 0
                                        for i, x in enumerate(p)) == q)
                expected_coarse[q] = expected
                check(value_at(coarse, q), expected,
                      ("first compression", dimension, upper, orientation, q))
                counts["signed_compression_cells"] += 1

            # A fresh mask between compression generations prevents testing
            # only the trivial associativity of unconstrained total mass.
            masks = List.empty_list(kernel.MASK)
            masks.append((0, 1, np.array([1, 0], dtype=np.int64)))
            masked = kernel.apply_easy(coarse, np.empty((0, 2), dtype=np.int64), masks)
            twice = kernel.push_blocks(masked, native_blocks(
                [[(0, len(b) - 1)] for b in blocks]), fresh_counters())
            expected = sum(value for q, value in expected_coarse.items() if q[1] >= 1 - q[0])
            check(value_at(twice, (0,) * dimension), expected,
                  ("masked repeated compression", dimension, upper, orientation))
            counts["signed_masked_repeated_compressions"] += 1
            for q, expected in fine.items():
                check(value_at(packed, q), expected, ("input immutability", dimension, q))
            counts["native_input_immutability"] += 1

    # A pure atom at index zero survives an inclusive singleton crop, even
    # when the zero-weight interval entries around it are removed by pruning.
    atom = kernel.pack([Term(1., {0: np.array([1., 0., 2.]),
                                  1: np.array([1., 0., 3.])}, {})])
    cropped = kernel.crop_terms(atom, np.array([0, 0], dtype=np.int64),
                                np.array([0, 0], dtype=np.int64), fresh_counters())
    check(value_at(cropped, (0, 0)), 1., "origin atom singleton crop")
    counts["origin_atom_boundary_crop"] += 1
    cropped = kernel.crop_terms(atom, np.array([2, 0], dtype=np.int64),
                                np.array([1, 2], dtype=np.int64), fresh_counters())
    assert len(cropped) == 0, "Empty inclusive crop must return an empty native list"
    check(kernel.rectangle_sum(atom, np.array([2, 0], dtype=np.int64),
                               np.array([1, 2], dtype=np.int64), fresh_counters()),
          0., "empty rectangle")
    counts["empty_crop_and_rectangle"] += 1

    # Cropping a decreasing edge can make it constant and fold it into the
    # increasing-key slot, which must not invalidate the simplifier iteration.
    opposite = [Term(1., {0: np.array([0., 2., -1., 0.]),
                           1: np.array([0., 1., -2., 0.])}, {
        (0, 1, True, -1): np.array([3, 2, 2, 0], dtype=np.int64),
        (0, 1, True, 1): np.array([0, 1, 2, 3], dtype=np.int64)})]
    packed = kernel.pack(opposite)
    cropped = kernel.crop_terms(packed, np.array([0, 0], dtype=np.int64),
                                np.array([3, 3], dtype=np.int64), fresh_counters())
    for q in product(range(4), repeat=2):
        check(value_at(cropped, q), value_at(opposite, q), ("opposite folded edges", q))
    counts["opposite_edge_simplification_regression"] += 1


def repeated_points():
    rng = random.Random(1842)
    for number in (0, 1, 2, 4, 8, 12):
        points = [tuple(rng.randrange(8) for _ in range(4)) for _ in range(number)]
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("compiled_verification.json"))
    args = parser.parse_args()
    started = time.perf_counter()
    counts, records = Counter(), []
    max_error = 0.0

    def check(actual, expected, label):
        nonlocal max_error
        actual, expected = float(actual), float(expected)
        assert isclose(actual, expected, rel_tol=REL_TOL, abs_tol=ABS_TOL), (
            label, actual, expected)
        max_error = max(max_error, abs(actual - expected) / max(1., abs(expected)))

    def record(label, dimension, mode, measure, solver, answer):
        records.append({"case": label, "dimension": dimension, "mode": mode,
                        "magnitude": measure, "value": float(answer),
                        "stats": {str(k): int(v) for k, v in solver.stats.items()}})

    kernel_checks(check, counts)
    print("Signed native crop, boundary atom and repeated compression checks passed.", flush=True)
    for dimension in range(2, 11):
        for mode, options in (("adaptive", {}),
                              ("forced", {"compress_every": 1, "compress_factor": 0.0})):
            # Reuse the engine across empty/nonempty inputs and both measures.
            solver = compiled.NumericalChanCompiled(dimension, base_hard=2, **options)
            for label, points in small_cases(dimension):
                # Dense 2d-variable elimination is deliberately not a stress
                # target here. Higher-dimensional repeated compression below
                # uses four interacting axes and common remaining extents.
                if mode == "forced" and dimension >= 6 and len(points) > 3:
                    points = points[:3]
                    label += "-bounded3"
                for measure in (False, True):
                    answer = solver.compute(points, magnitude=measure)
                    expected = inclusion_exclusion(points, measure)
                    check(answer, expected, (dimension, mode, label, measure))
                    counts["exact_end_to_end_instances"] += 1
                    record(label, dimension, mode, measure, solver, answer)

        points = small_cases(dimension)[4][1]
        # The old solver's scheduled compression can be expensive even on a
        # five-point high-dimensional input; the exact-oracle checks above
        # retain all five points for the new adaptive implementation.
        reference_points = points[:3] if dimension >= 7 else points
        for measure in (False, True):
            expected = inclusion_exclusion(reference_points, measure)
            answer = compiled.NumericalChanCompiled(dimension).compute(
                reference_points, magnitude=measure)
            # Existing numerical Python is a separately implemented finite-sum
            # recursion; Chan d/2 is an independent geometric cross-check.
            check(answer, reference.NumericalChan(dimension).compute(
                reference_points, magnitude=measure),
                  (dimension, "numerical Python", measure))
            transformed = ([tuple(1 + Fraction(x) / 2 for x in p) for p in reference_points]
                           if measure else reference_points)
            minimum_points = [tuple(-float(x) for x in p) for p in transformed]
            check(answer, chan_dby2(minimum_points, [0.] * dimension, prefilter=False),
                  (dimension, "Chan d/2", measure))
            check(answer, expected, (dimension, "cross-check exact", measure))
            counts["numerical_python_comparisons"] += 1
            counts["chan_dby2_comparisons_no_prefilter"] += 1

        atom_points = small_cases(dimension)[2][1]
        transformed = [tuple(1 + Fraction(x) / 2 for x in p) for p in atom_points]
        check(compiled.magnitude(atom_points, dimension=dimension),
              compiled.hypervolume(transformed, dimension=dimension),
              (dimension, "public affine magnitude identity"))
        check(compiled.hypervolume(points, dimension=dimension), inclusion_exclusion(points),
              (dimension, "public hypervolume"))
        check(compiled.magnitude(atom_points, dimension=dimension),
              inclusion_exclusion(atom_points, True), (dimension, "public magnitude"))
        counts["public_affine_identity"] += 1
        counts["public_wrapper_exact_checks"] += 2
        print(f"{dimension}D: default/forced exact cases, public APIs and independent references passed.",
              flush=True)

    deep = repeated_points()
    for dimension in (4, 5, 7, 10):
        points = [p + (2,) * (dimension - 4) for p in deep]
        for measure in (False, True):
            solver = compiled.NumericalChanCompiled(
                dimension, base_hard=1, compress_every=1, compress_factor=0.)
            answer = solver.compute(points, magnitude=measure)
            expected = inclusion_exclusion(deep, measure) * 2 ** (dimension - 4)
            check(answer, expected, (dimension, "repeated compression", measure))
            assert solver.stats.get("compressions", 0) >= 2, dict(solver.stats)
            assert solver.stats.get("max_generation", 0) >= 2, dict(solver.stats)
            counts["geometric_repeated_compression_exact_instances"] += 1
            record("repeated-compression-12", dimension, "forced", measure, solver, answer)
        print(f"{dimension}D: repeated geometric compression passed.", flush=True)

    # A bounded fully varying 6D case complements the padded high-dimensional
    # tests: all six coordinates interact, with genuine successive generations.
    points = [tuple(1 + (i + j) % 4 for i in range(6)) for j in range(4)]
    for measure in (False, True):
        solver = compiled.NumericalChanCompiled(
            6, base_hard=1, compress_every=1, compress_factor=0.)
        answer = solver.compute(points, magnitude=measure)
        check(answer, inclusion_exclusion(points, measure), (6, "fully varying forced", measure))
        assert solver.stats.get("compressions", 0) >= 1, dict(solver.stats)
        assert solver.stats.get("max_generation", 0) >= 2, dict(solver.stats)
        counts["fully_varying_6d_repeated_compression_exact_instances"] += 1
        record("fully-varying-6d-4", 6, "forced", measure, solver, answer)
    print("6D: fully varying repeated geometric compression passed.", flush=True)

    from NumericalChanHV4D.numerical_chan4_adaptive import hypervolume4
    points = [(1, 4, 3, 2), (2, 3, 4, 1), (3, 2, 1, 4), (4, 1, 2, 3)]
    for measure in (False, True):
        check(hypervolume4(points, magnitude=measure), inclusion_exclusion(points, measure),
              ("4D convenience adapter", measure))
        counts["adaptive_4d_convenience_adapter_exact_checks"] += 1

    # Reuse kernels cached by this process from the canonical package path.
    probe = """
import json
from NumericalChanHVND.numerical_chan_compiled import hypervolume, magnitude
p = [(1, 4, 3, 2, 2), (2, 3, 4, 1, 2), (3, 2, 1, 4, 2), (4, 1, 2, 3, 2)]
print(json.dumps({'hv': hypervolume(p), 'magnitude': magnitude(p)}))
"""
    result = subprocess.run([sys.executable, "-c", probe], cwd=ROOT,
                            capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 0, ("cached package import", result.stdout, result.stderr)
    values = json.loads(result.stdout)
    check(values["hv"], 138, "fresh-process cached HV")
    check(values["magnitude"], Fraction(765, 8), "fresh-process cached magnitude")
    counts["fresh_process_canonical_package_cache"] += 1
    signatures = {}
    for name in ("crop_terms", "push_blocks", "base_sum", "apply_easy"):
        signatures[name] = len(getattr(kernel, name).nopython_signatures)
        assert signatures[name] > 0, (name, "was not compiled in nopython mode")
    report = {
        "status": "All compiled 2--10D checks passed.",
        "dimensions": list(range(2, 11)), "checks": dict(counts),
        "max_scaled_error": max_error,
        "tolerance": {"relative": REL_TOL, "absolute": ABS_TOL},
        "nopython_signatures": signatures,
        "seconds_including_compilation": time.perf_counter() - started,
        "scope": "Small exact-oracle inputs, signed grid algebra, repeated compression and cache import. "
                 "Float64 correctness only; not a performance or asymptotic bound.",
        "cases": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
