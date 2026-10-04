"""Check saved small benchmark values using exact rational inclusion-exclusion.

Run after timing has finished, to avoid competing with benchmark workers.
This independent oracle operates on exact binary64 input values, and uses no
solver, cutoff representation, compression, sweep or numerical integration.
"""
import argparse
from fractions import Fraction
import hashlib
import json
from math import isclose, prod
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent


def inclusion_exclusion(points, magnitude):
    corners = [tuple(Fraction(x) for x in point) for point in points]
    answer = Fraction(0)

    def visit(start, intersection, sign):
        nonlocal answer
        for index in range(start, len(corners)):
            current = corners[index] if intersection is None else tuple(
                min(x, y) for x, y in zip(intersection, corners[index]))
            answer += sign * prod(1 + x / 2 if magnitude else x for x in current)
            visit(index + 1, current, -sign)

    visit(0, None, 1)
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=HERE / 'results')
    parser.add_argument('--max-points', type=int, default=12)
    args = parser.parse_args()
    timings = args.results / 'timings.json'
    datasets = args.results / 'datasets.json'
    report = json.loads(timings.read_text())
    if 'finished_utc' not in report:
        parser.error('Wait until timing finishes before running this oracle')
    cases = json.loads(datasets.read_text())
    start, errors, checked_values = time.perf_counter(), [], 0
    checked = []
    for case in cases:
        if case['n'] > args.max_points:
            continue
        exact = inclusion_exclusion(case['points'], case['magnitude'])
        expected = float(exact)
        maximum = 0.
        algorithms = []
        for row in report['rows']:
            if row['id'] != case['id']:
                continue
            values = ([row['first_value']] if 'first_value' in row else [])
            values.extend(sample['value'] for sample in row.get('samples', []))
            if values:
                algorithms.append(row['algorithm'])
            for value in values:
                checked_values += 1
                maximum = max(maximum, abs(value - expected) / max(1., abs(expected)))
                if not isclose(value, expected, rel_tol=2e-9, abs_tol=2e-10):
                    errors.append(dict(id=case['id'], algorithm=row['algorithm'],
                                       value=value, expected=expected))
        checked.append(dict(id=case['id'], d=case['d'], n=case['n'],
                            exact_numerator=str(exact.numerator), exact_denominator=str(exact.denominator),
                            oracle_float=expected, algorithms_checked=algorithms,
                            max_scaled_error=maximum))
    output = dict(status='passed' if not errors else 'failed',
                  method='Exact rational inclusion-exclusion on the binary64 input coordinates',
                  max_points=args.max_points, checked_cases=len(checked), checked_values=checked_values,
                  max_scaled_error=max((row['max_scaled_error'] for row in checked), default=0.),
                  tolerance=dict(relative=2e-9, absolute=2e-10),
                  elapsed_seconds=time.perf_counter() - start,
                  input_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (timings, datasets)},
                  errors=errors, cases=checked)
    (args.results / 'oracle_verification.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({key: value for key, value in output.items() if key not in ('cases', 'input_sha256')}, indent=2))
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
