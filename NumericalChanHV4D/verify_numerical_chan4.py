"""Independent finite-sum and inclusion--exclusion oracles, exact rational inputs."""

from collections import Counter
from fractions import Fraction
from itertools import combinations, product
from math import prod
from pathlib import Path
import json
import random
import sys
import time

from numerical_chan4 import (NumericalChan4, Term, add_edge, eliminate_one,
                            impose, push_blocks, sum_out, value_at)


def brute_hv(points, magnitude):
    answer = 0
    for k in range(1, len(points) + 1):
        for subset in combinations(points, k):
            corner = [min(p[d] for p in subset) for d in range(4)]
            answer += (-1) ** (k + 1) * prod(1 + x / 2 for x in corner) if magnitude else (
                (-1) ** (k + 1) * prod(corner))
    return answer


def random_term(rng, sizes, signed=True):
    unary = {i: tuple(rng.choice((-2, -1, 0, 1, 2, 3) if signed else (1, 2, 3))
                      for _ in range(n)) for i, n in enumerate(sizes)}
    term = Term(1, unary, {})
    for i, j in combinations(range(len(sizes)), 2):
        if rng.random() < .65:
            f = sorted([rng.randrange(-1, sizes[j] + 1) for _ in range(sizes[i])],
                       reverse=rng.choice((False, True)))
            add_edge(term, i, j, rng.choice((False, True)), f)
    return term


def table(terms, sizes):
    return {q: value_at(terms, dict(enumerate(q))) for q in product(*(range(n) for n in sizes))}


def main():
    start = time.perf_counter()
    rng = random.Random(640026)
    stats = Counter()
    checks = Counter()
    # All directions, strictness and argument order; compare every grid entry.
    for repeat in range(120):
        a = tuple(sorted([rng.randrange(-2, 5) for _ in range(4)], reverse=rng.choice((False, True))))
        b = tuple(sorted([rng.randrange(-2, 5) for _ in range(3)], reverse=rng.choice((False, True))))
        for strict, reverse in product((False, True), repeat=2):
            term = Term(1, {0: (1,) * 4, 1: (1,) * 3}, {})
            first, second = ((1, b), (0, a)) if reverse else ((0, a), (1, b))
            live = impose(term, first, second, strict)
            for i, j in product(range(4), range(3)):
                x, y = (b[j], a[i]) if reverse else (a[i], b[j])
                expected = int(x < y if strict else x <= y)
                actual = value_at([term], {0: i, 1: j}) if live else 0
                assert actual == expected
                checks['comparison_grid_entries'] += 1
    # Same-variable comparisons can oscillate: they must remain unary arrays.
    term = Term(1, {0: (1,) * 6}, {})
    a, b = (0, 2, 2, 4, 4, 6), (1, 1, 3, 3, 5, 5)
    assert impose(term, (0, a), (0, b))
    assert term.unary[0] == (1, 0, 1, 0, 1, 0) and not term.edges
    checks['alternating_unary_filter'] += 1

    for dimensions, repeats in ((2, 20), (3, 20), (4, 20), (5, 6), (6, 3), (8, 2)):
        for repeat in range(repeats):
            sizes = [rng.randrange(2, 5) if dimensions <= 4 else 2 for _ in range(dimensions)]
            term = random_term(rng, sizes)
            before = table([term], sizes)
            axis = rng.randrange(dimensions)
            after = eliminate_one([term], axis, stats)
            remaining = [i for i in range(dimensions) if i != axis]
            for values in product(*(range(sizes[i]) for i in remaining)):
                index = dict(zip(remaining, values))
                expected = sum(value for q, value in before.items()
                               if all(q[i] == index[i] for i in remaining))
                assert value_at(after, index) == expected
                checks['elimination_output_entries'] += 1
            scalar = sum(t.coef for t in sum_out([term], list(range(dimensions)), stats))
            assert scalar == sum(before.values())
            checks['complete_sums'] += 1

    compression_cases = []
    for magnitude in (False, True):
        axis_weights = (1, Fraction(1, 2), 1, Fraction(3, 2)) if magnitude else (0, 1, 2, 3)
        term = Term(1, {i: axis_weights for i in range(4)}, {})
        for i, j, upper, f in ((0, 1, True, (3, 2, 2, 1)),
                               (0, 2, True, (1, 2, 2, 3)),
                               (1, 3, False, (2, 1, 0, 0)),
                               (2, 3, True, (3, 3, 2, 1))):
            add_edge(term, i, j, upper, f)
        fine = table([term], [4] * 4)
        local = Counter()
        packed = push_blocks([term], [[(0, 1), (2, 3)]] * 4, local)
        for q in product(range(2), repeat=4):
            expected = sum(v for p, v in fine.items() if tuple(x // 2 for x in p) == q)
            assert value_at(packed, dict(enumerate(q))) == expected
            checks['first_compression_cells'] += 1
        # A new mask is aligned to the current hard grid, then it is coarsened.
        masked = []
        for old in packed:
            t = old.copy()
            if add_edge(t, 0, 3, False, (1, 0)):
                masked.append(t)
        twice = push_blocks(masked, [[(0, 1)]] * 4, local)
        expected = sum(v for p, v in fine.items() if p[3] // 2 >= 1 - p[0] // 2)
        assert value_at(twice, {i: 0 for i in range(4)}) == expected
        checks['second_compression_cells'] += 1
        compression_cases.append({'magnitude': magnitude, **dict(local)})

    cases = []
    fixed = [[], [(0, 0, 0, 0)], [(0, 2, 3, 4)], [(1, 2, 3, 4)] * 2,
             [(1, 4, 3, 2), (2, 3, 4, 1), (3, 2, 1, 4), (4, 1, 2, 3)]]
    instances = fixed + [[tuple(rng.randrange(7) for _ in range(4)) for _ in range(n)]
                         for n in (2, 3, 4, 5, 6, 8) for _ in range(3)]
    instances += [[tuple(Fraction(rng.randrange(20), rng.choice((3, 5, 7))) for _ in range(4))
                   for _ in range(5)] for _ in range(3)]
    deep_rng = random.Random(1842)
    for n in (0, 1, 2, 4, 8, 12):
        deep = [tuple(deep_rng.randrange(8) for _ in range(4)) for _ in range(n)]
    instances.append(deep)
    for case, points in enumerate(instances):
        points = [tuple(Fraction(x) for x in p) for p in points]
        for magnitude in (False, True):
            solver = NumericalChan4(base_hard=1)
            result = solver.compute(points, magnitude)
            expected = brute_hv(points, magnitude)
            assert result == expected, (case, magnitude, result, expected)
            checks['end_to_end_exact_instances'] += 1
            cases.append({'case': case, 'n': len(points), 'magnitude': magnitude,
                          'answer': str(result), **dict(solver.stats)})
    # Stress the closure with a deliberately aggressive TEST-ONLY block schedule.
    for magnitude in (False, True):
        solver = NumericalChan4(base_hard=1, block_levels=1)
        points = [tuple(Fraction(x) for x in p) for p in deep]
        assert solver.compute(points, magnitude) == brute_hv(points, magnitude)
        assert solver.stats['max_generation'] >= 3
        cases.append({'case': 'forced-deep', 'n': len(points), 'magnitude': magnitude,
                      'block_levels': 1, **dict(solver.stats)})
        checks['forced_deep_instances'] += 1
    assert sum(c.get('compressions', 0) for c in cases) > 0

    for points in instances[1:7]:
        original = [tuple(Fraction(x) for x in p) for p in points]
        shifted = [tuple(1 + x / 2 for x in p) for p in original]
        assert NumericalChan4().compute(original, True) == NumericalChan4().compute(shifted)
        checks['magnitude_affine_equivalence'] += 1

    # Independent check of the weighted-face decrease, including ties and skips.
    for repeat in range(180):
        n = rng.randrange(3, 20)
        solver = NumericalChan4()
        solver.compute([(1, 1, 1, 1)] * n)  # initializes exact integer type weights
        lo, hi = (0,) * 4, (7,) * 4
        boxes = [tuple(rng.randrange(9) for _ in range(4)) for _ in range(n)]
        boxes = [p for p in boxes if sum(p[i] < hi[i] for i in range(4)) >= 3]
        axis = rng.randrange(4)
        def potential(corners, lower, upper, current_axis):
            return sum(2 ** (sum((i - current_axis) % 4 for i in triple) / 4)
                       for p in corners if all(p[i] >= lower[i] for i in range(4))
                       for triple in combinations([i for i in range(4) if p[i] < upper[i]], 3))
        candidates = []
        for p in boxes:
            weight = sum(solver.weights[sum((i - axis) % 4 for i in triple)]
                         for triple in combinations([i for i in range(4) if p[i] < hi[i]], 3)
                         if axis in triple)
            if weight:
                candidates.append((p[axis], weight))
        parent = potential(boxes, lo, hi, axis)
        if candidates:
            candidates.sort()
            total, partial = sum(w for _, w in candidates), 0
            for cut, weight in candidates:
                partial += weight
                if 2 * partial >= total:
                    break
            left_hi, right_lo = list(hi), list(lo)
            left_hi[axis], right_lo[axis] = cut, cut + 1
            cells = [(lo, left_hi), (right_lo, hi)]
        else:
            cells = [(lo, hi)]
        for lower, upper in cells:
            child = potential(boxes, lower, upper, (axis + 1) % 4)
            assert child <= 2 ** (-.75) * (1 + 1 / solver.scale) * parent + 1e-10
            checks['face_weight_contraction_checks'] += 1

    # Independent repository implementation, only for a few ordinary-HV cases.
    legacy_dir = Path(__file__).resolve().parents[1]
    if not (legacy_dir / 'chan_orthant_dby3.py').exists():
        legacy_dir = legacy_dir / 'work/repository_sources/FastHVChan'
    if (legacy_dir / 'chan_orthant_dby3.py').exists():
        sys.path.insert(0, str(legacy_dir))
        from chan_orthant_dby3 import hypervolume_dby3
        for points in instances[-4:]:
            corners = [tuple(float(x) for x in p) for p in points]
            legacy = hypervolume_dby3([tuple(-x for x in p) for p in corners], (0.,) * 4,
                                      prefilter=False)
            expected = brute_hv([tuple(Fraction(x) for x in p) for p in points], False)
            assert abs(legacy - float(expected)) < 1e-8
            checks['FastHVChan_cross_checks'] += 1

    result = {'status': 'All exact checks passed.', 'checks': dict(checks),
              'seconds': round(time.perf_counter() - start, 3),
              'elimination_stats': dict(stats), 'compression_cases': compression_cases,
              'cases': cases,
              'scope': 'Numerical 4D solver and its finite-array compression; no measured asymptotic-speed claim.'}
    Path(__file__).with_name('numerical_chan4_results.json').write_text(
        json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('cases', 'elimination_stats', 'compression_cases')}, indent=2))


if __name__ == '__main__':
    main()
