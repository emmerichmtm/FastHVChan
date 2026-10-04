"""4D correctness spot checks for HV4DMagnitude at the audited commit.

The direct packed-kernel failures are recorded, not silently repaired.
The public ordinary-HV solver is separately checked after the affine transform.
"""
import argparse
from fractions import Fraction
from itertools import combinations
import json
import math
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--old', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.old.resolve()))
    from core import LEBESGUE, MAGNITUDE, Step
    from prefix4 import Prefix4, SixStaircases
    from solver import PrefixHV4D
    stairs = SixStaircases(*[Step((), (2.0,)) for _ in range(6)])
    direct = []
    for name, measure in (('Lebesgue', LEBESGUE), ('magnitude', MAGNITUDE)):
        prefix = Prefix4(stairs, [measure] * 4, [[0., 1.]] * 4)
        for query in ((1., 1., 1., 1.), (1., 1., 0., 0.)):
            expected = math.prod(measure.atom0 + measure.density * x for x in query)
            row = dict(measure=name, query=query, expected=expected,
                       packed=prefix.K_fast(*query),
                       decomposed=prefix.K_decomposed(*query),
                       brute=prefix.K_brute(*query))
            row['packed_matches'] = math.isclose(row['packed'], expected)
            assert row['decomposed'] == row['brute'] == expected
            direct.append(row)
    transformed = []
    cases = [[(1, 1, 1, 1)], [(1, 1, 0, 0)],
             [(1, 4, 3, 2), (2, 3, 4, 1), (3, 2, 1, 4), (4, 1, 2, 3)]]
    for corners in cases:
        expected = Fraction(0)
        for count in range(1, len(corners) + 1):
            for subset in combinations(corners, count):
                expected += (-1) ** (count + 1) * math.prod(
                    1 + Fraction(min(p[i] for p in subset), 2) for i in range(4))
        hv_corners = [[1. + x / 2 for x in p] for p in corners]
        ref = [max(p[i] for p in hv_corners) + 1. for i in range(4)]
        points = [[ref[i] - p[i] for i in range(4)] for p in hv_corners]
        value = PrefixHV4D(base_hard=2, fast=True).hypervolume(points, ref)
        assert math.isclose(value, float(expected), rel_tol=1e-12, abs_tol=1e-12)
        transformed.append(dict(corners=corners, exact_magnitude=str(expected),
                                transformed_hv=value))
    report = {'direct_kernel': direct, 'affine_transform': transformed}
    args.out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
