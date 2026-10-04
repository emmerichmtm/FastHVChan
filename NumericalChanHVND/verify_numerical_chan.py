"""Exact, independent checks for dimensions 2 through 10. Run with Python 3.10+."""
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import combinations, product
from math import prod
from pathlib import Path
import json
import random
import time

from numerical_chan import (NumericalChan, PrefixSkyline, Term, add_edge, ceil_root,
                            hypervolume, magnitude, push_blocks, sum_out, value_at)


def inclusion_exclusion(points, mag=False):
    if not points:
        return 0
    d, answer = len(points[0]), 0
    for size in range(1, len(points) + 1):
        for chosen in combinations(points, size):
            corner = [min(p[i] for p in chosen) for i in range(d)]
            mass = prod(1 + Fraction(x, 2) for x in corner) if mag else prod(corner)
            answer += (-1) ** (size + 1) * mass
    return answer


def main():
    started = time.perf_counter()
    rng, counts, results = random.Random(20261004), Counter(), []
    # The 3D data structure is checked after every update against a dense 1D array.
    for length in (1, 2, 7, 31):
        masses = [Fraction(rng.randrange(4), 2) for _ in range(length)]
        tree, heights = PrefixSkyline(masses), [0] * length
        for repeat in range(80):
            last, height = rng.randrange(length), Fraction(rng.randrange(30), 3)
            for i in range(last + 1):
                heights[i] = max(heights[i], height)
            assert tree.insert(last, height) == sum(a * h for a, h in zip(masses, heights))
            counts['skyline_updates'] += 1
    for degree in range(2, 11):
        for number in (1, 2, 123, 2 ** 67 + 53, (2 ** 23 + 1) ** degree):
            root = ceil_root(number, degree)
            assert (root - 1) ** degree < number <= root ** degree
            counts['integer_roots'] += 1

    for d in range(2, 11):
        instances = [[], [(0,) * d], [(0,) + (3,) * (d - 1)],
                     [(2,) * d] * 2, [(1,) * d, (3,) * d],
                     [tuple(1 + (i + j) % 5 for i in range(d)) for j in range(5)]]
        instances += [[tuple(rng.randrange(1, 9) for _ in range(d)) for _ in range(n)]
                      for n in (2, 4, 7) for repeat in range(2)]
        instances += [[tuple(Fraction(rng.randrange(10), rng.choice((2, 3, 5)))
                             for _ in range(d)) for _ in range(5)]]
        local = Counter()
        for case, points in enumerate(instances):
            for mag in (False, True):
                engine = NumericalChan(d, base_hard=1)
                actual = engine.compute(points, magnitude=mag)
                expected = inclusion_exclusion(points, mag)
                assert actual == expected, (d, case, mag, actual, expected)
                counts['end_to_end_exact_instances'] += 1
                local.update(engine.stats)
        # The two affine transformations, including zero-coordinate magnitude.
        points = instances[6] + instances[2]
        transformed = [tuple(1 + Fraction(x, 2) for x in p) for p in points]
        assert magnitude(points) == hypervolume(transformed)
        positive = instances[7]
        scaled = [tuple(2 * (3 * x - 1) for x in p) for p in positive]
        assert hypervolume(positive) == magnitude(scaled) / (3 ** d)
        counts['affine_equivalences'] += 2
        # A full 2d-variable compression, with cross-pair interaction and extra axes.
        # Enumerating cells happens ONLY in this independent test oracle.
        sizes = [4 if i < 4 else 2 for i in range(d)]
        blocks = [[(0, 1), (2, 3)] if s == 4 else [(0, 1)] for s in sizes]
        for mag in (False, True):
            term = Term(1, {i: tuple(([1] if mag else [0]) +
                        [Fraction(k, 2) if mag else k for k in range(1, s)])
                        for i, s in enumerate(sizes)}, {})
            add_edge(term, 0, 1, True, (3, 2, 2, 1))
            if d >= 3:
                add_edge(term, 0, 2, True, (1, 2, 2, 3))
            if d >= 4:
                add_edge(term, 1, 3, False, (2, 1, 0, 0))
            if d >= 5:
                add_edge(term, 0, d - 1, True, (0, 0, 1, 1))
            expected = defaultdict(Fraction)
            for q in product(*(range(s) for s in sizes)):
                coarse = tuple(x // 2 if i < 4 else 0 for i, x in enumerate(q))
                expected[coarse] += value_at([term], dict(enumerate(q)))
            stats = Counter()
            packed = push_blocks([term], blocks, stats)
            for q, mass in expected.items():
                assert value_at(packed, dict(enumerate(q))) == mass, (d, mag, q)
                counts['first_compression_cells'] += 1
            # A NEW cross-axis mask, then a second compression generation.
            masked = []
            for old in packed:
                t = old.copy()
                if add_edge(t, 0, 1, False, (1, 0)):
                    masked.append(t)
            twice = push_blocks(masked, [[(0, len(b) - 1)] for b in blocks], stats)
            wanted = sum(mass for q, mass in expected.items() if q[1] >= 1 - q[0])
            assert value_at(twice, {i: 0 for i in range(d)}) == wanted
            counts['second_compressions'] += 1
            results.append({'dimension': d, 'lifted_variables': 2 * d, 'magnitude': mag,
                            'compression_stats': dict(stats)})
        print(f'{d}D: exact end-to-end, both transforms, and {2*d}-variable compression passed.', flush=True)
        results.append({'dimension': d, 'aggregate_end_to_end_stats': dict(local)})

    # Reuse a 4D instance with genuine repeated compression in odd and even d.
    deep_rng = random.Random(1842)
    for n in (0, 1, 2, 4, 8, 12):
        deep = [tuple(deep_rng.randrange(8) for _ in range(4)) for _ in range(n)]
    for d in (4, 5, 6, 7, 8, 9, 10):
        # Extra axes have common extent; their exact multiplicative effect is known.
        points = [p + (2,) * (d - 4) for p in deep]
        for mag in (False, True):
            solver = NumericalChan(d, base_hard=1, block_levels=2)
            answer = solver.compute(points, magnitude=mag)
            expected = inclusion_exclusion(deep, mag) * 2 ** (d - 4)
            assert answer == expected, (d, mag, answer, expected)
            assert solver.stats['max_generation'] >= 2
            counts['forced_repeated_recursion'] += 1
            results.append({'dimension': d, 'forced_blocks': 2, 'magnitude': mag,
                            'stats': dict(solver.stats)})
        print(f'{d}D: forced repeated geometric compression passed.', flush=True)

    for d in range(4, 11):
        solver = NumericalChan(d)
        solver.compute([(1,) * d] * 5)
        for s, weight in solver.weights.items():
            target = solver.scale ** d * (1 << s)
            assert (weight - 1) ** d < target <= weight ** d
        for repeat in range(25):
            axis = rng.randrange(d)
            lo, hi = (0,) * d, (5,) * d
            boxes = [tuple(rng.randrange(7) for _ in range(d)) for _ in range(9)]
            def potential(lower, upper, current):
                return sum(2 ** (sum((i - current) % d for i in triple) / d)
                           for p in boxes if all(p[i] >= lower[i] for i in range(d))
                           for triple in combinations([i for i in range(d) if p[i] < upper[i]], 3))
            candidates = []
            for p in boxes:
                weight = sum(solver.weights[sum((i - axis) % d for i in triple)]
                             for triple in combinations([i for i in range(d) if p[i] < hi[i]], 3)
                             if axis in triple)
                if weight:
                    candidates.append((p[axis], weight))
            parent = potential(lo, hi, axis)
            if candidates:
                total, partial = sum(w for _, w in candidates), 0
                for cut, weight in sorted(candidates):
                    partial += weight
                    if 2 * partial >= total:
                        break
                left, right = list(hi), list(lo)
                left[axis], right[axis] = cut, cut + 1
                cells = [(lo, left), (right, hi)]
            else:
                cells = [(lo, hi)]
            for lower, upper in cells:
                child = potential(lower, upper, (axis + 1) % d)
                assert child <= 2 ** (-3 / d) * (1 + 1 / solver.scale) * parent + 1e-9
                counts['face_contraction_checks'] += 1

    for kwargs, points in (({'dimension': 1}, []), ({'dimension': 11}, []),
                           ({'dimension': 3}, [(1, 2)]), ({}, [(1, -1)]),
                           ({}, [(1, float('nan'))]), ({}, [(1, float('inf'))])):
        try:
            hypervolume(points, **kwargs)
        except ValueError:
            counts['invalid_inputs'] += 1
        else:
            raise AssertionError((kwargs, points))
    report = {'status': 'All checks passed.', 'dimensions': list(range(2, 11)),
              'checks': dict(counts), 'seconds': round(time.perf_counter() - started, 3),
              'details': results,
              'scope': 'Exact arithmetic except floating diagnostics for face potential. '
                       'High-dimensional tests are small; no practical speed claim.'}
    Path(__file__).with_name('numerical_chan_results.json').write_text(
        json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'details'}, indent=2))


if __name__ == '__main__':
    main()
