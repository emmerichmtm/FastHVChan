"""Floating-point backend checks against independent exact inclusion--exclusion."""
from collections import Counter
from fractions import Fraction
from itertools import product
from pathlib import Path
import json
import math
import random
import time
import numpy as np
import numerical_chan as exact
import numerical_chan_numba as jit
from verify_numerical_chan import inclusion_exclusion


def close(actual, expected):
    assert math.isclose(float(actual), float(expected), rel_tol=2e-9, abs_tol=2e-10), (actual, expected)


def main():
    start = time.perf_counter()
    rng, checks, largest_error = random.Random(87234), Counter(), 0.0
    for d in range(2, 11):
        cases = [[], [(0,) * d], [(0,) + (2,) * (d - 1)], [(2,) * d] * 2]
        cases += [[tuple(rng.randrange(7) for _ in range(d)) for _ in range(n)] for n in (3, 6, 8)]
        cases += [[tuple(Fraction(rng.randrange(1, 20), 7) for _ in range(d)) for _ in range(5)]]
        for points in cases:
            for mag in (False, True):
                actual = jit.hypervolume(points, magnitude=mag, dimension=d)
                expected = inclusion_exclusion(points, mag)
                close(actual, expected)
                largest_error = max(largest_error, abs(actual - float(expected)) / max(1., abs(float(expected))))
                checks['end_to_end'] += 1
        # Cross-pair predicates with all four monotonicity/sense classes.
        sizes = [3] * d
        weights = {i: tuple(rng.choice((-2, -1, 1, 2)) for _ in range(3)) for i in range(d)}
        reference = exact.Term(1, weights, {})
        floating = jit.Term(1., {i: np.array(w, dtype=float) for i, w in weights.items()}, {})
        # Couple up to four axes; retain every remaining axis in the 2d lift.
        # Dense high-dimensional term expansions are a performance stress case,
        # not needed for this bounded correctness regression.
        for i in range(min(d - 1, 3)):
            for upper, values in ((True, (0, 1, 2)), (True, (2, 2, 1)),
                                  (False, (0, 0, 1)), (False, (1, 0, 0))):
                exact.add_edge(reference, i, i + 1, upper, values)
                jit.add_edge(floating, i, i + 1, upper, values)
        expected = sum(t.coef for t in exact.sum_out([reference], list(range(d)), Counter()))
        actual = sum(t.coef for t in jit.sum_out([floating], list(range(d)), Counter()))
        close(actual, expected)
        checks['signed_four_slot_sums'] += 1
        packed = jit.push_blocks([floating], [[(0, 1), (2, 2)]] * d, Counter())
        ref = exact.push_blocks([reference], [[(0, 1), (2, 2)]] * d, Counter())
        for q in product(range(2), repeat=d):
            close(jit.value_at(packed, dict(enumerate(q))), exact.value_at(ref, dict(enumerate(q))))
            checks['compression_cells'] += 1
        print(f'{d}D Numba backend passed.', flush=True)
    deep_rng = random.Random(1842)
    for n in (0, 1, 2, 4, 8, 12):
        deep = [tuple(deep_rng.randrange(8) for _ in range(4)) for _ in range(n)]
    for d in (4, 5, 10):
        for mag in (False, True):
            solver = jit.NumericalChanNumba(d, base_hard=1, block_levels=2)
            actual = solver.compute([p + (2,) * (d - 4) for p in deep], mag)
            close(actual, inclusion_exclusion(deep, mag) * 2 ** (d - 4))
            assert solver.stats['max_generation'] >= 2
            checks['repeated_compression_instances'] += 1
    for length in (1, 2, 7, 31):
        masses = np.array([rng.randrange(4) / 2 for _ in range(length)])
        prefix = jit.prefix_sum(masses)
        minimum, total, lazy = (np.zeros(4 * length) for _ in range(3))
        pending = np.ones(4 * length, dtype=np.bool_)
        heights = np.zeros(length)
        for repeat in range(80):
            last, height = rng.randrange(length), rng.randrange(30) / 3
            heights[:last + 1] = np.maximum(heights[:last + 1], height)
            close(jit.skyline_insert(last, height, prefix, minimum, total, lazy, pending),
                  np.sum(masses * heights))
            checks['compiled_skyline_updates'] += 1
    signatures = {name: len(getattr(jit, name).nopython_signatures)
                  for name in ('nonzero', 'direction', 'prefix_sum', 'search_array', 'normalize',
                               'suffix_mask', 'sweep_ranked', 'skyline_insert')}
    assert all(signatures.values()), signatures
    report = {'status': 'All checks passed.', 'checks': dict(checks),
              'max_scaled_error': largest_error, 'nopython_signature_counts': signatures,
              'seconds': time.perf_counter() - start,
              'tolerance': {'relative': 2e-9, 'absolute': 2e-10}}
    Path(__file__).with_name('numba_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
