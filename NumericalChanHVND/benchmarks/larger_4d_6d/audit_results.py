"""Audit retained inputs, complete and partial timings, and source provenance."""
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from source_provenance import verify_sources

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    report = json.loads((HERE / 'results/timings.json').read_text(encoding='utf-8'))
    cases = json.loads((HERE / 'results/datasets.json').read_text(encoding='utf-8'))
    assert 'finished_utc' in report
    assert report['validation_errors'] == 0
    assert len(cases) == 25 and len(report['rows']) == 100
    index = {(row['id'], row['algorithm']): row for row in report['rows']}
    assert len(index) == 100
    with (HERE / 'results/timings.csv').open(encoding='utf-8', newline='') as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == len(index)
    for row in csv_rows:
        original = index[row['id'], row['algorithm']]
        assert row['status'] == original['status']
        for key in ('first_seconds', 'median_seconds'):
            expected = str(original[key]) if key in original else ''
            assert row[key] == expected
    verify_sources(report, HERE / 'results')
    series = {}
    for case in cases:
        assert len(case['points']) == case['n']
        for point in case['points']:
            assert len(point) == case['d'] and all(x > 0 for x in point)
            level = sum(x*x for x in point) if case['family'] == 'sphere' else sum(point)
            assert math.isclose(level, 1., rel_tol=2e-15, abs_tol=2e-15)
        series.setdefault((case['d'], case['family']), []).append(case)
    for group in series.values():
        largest = max(group, key=lambda c: c['n'])['points']
        assert all(case['points'] == largest[:case['n']] for case in group)
    checked_values = 0
    unverified_values = 0
    errors = []
    for case in cases:
        baseline = index[case['id'], 'chan_dby2'].get('first_value')
        for algorithm in report['algorithms']:
            row = index[case['id'], algorithm]
            assert row['status'] in ('ok', 'timeout', 'not_run_after_timeouts')
            values = ([row['first_value']] if 'first_value' in row else [])
            values += [sample['value'] for sample in row.get('samples', [])]
            if row['status'] == 'ok':
                times = [sample['seconds'] for sample in row['samples']]
                assert len(times) == report['repeats'] == 3
                assert times == row['seconds']
                assert statistics.median(times) == row['median_seconds']
            else:
                assert 'median_seconds' not in row
            if baseline is None:
                unverified_values += len(values)
                continue
            for value in values:
                assert math.isfinite(value)
                assert math.isclose(value, baseline, rel_tol=2e-9, abs_tol=2e-10)
                checked_values += 1
                errors.append(abs(value-baseline)/max(1., abs(baseline)))
    for group in series.values():
        for algorithm in report['algorithms']:
            streak = 0
            for case in sorted(group, key=lambda c: c['n']):
                row = index[case['id'], algorithm]
                if streak >= 2:
                    assert row['status'] == 'not_run_after_timeouts'
                else:
                    assert row['status'] != 'not_run_after_timeouts'
                    streak = streak+1 if row['status'] == 'timeout' else 0
    result = dict(status='passed', datasets=len(cases), rows=len(index),
                  statuses=dict(Counter(r['status'] for r in report['rows'])),
                  checked_returned_values=checked_values,
                  unverified_returned_values=unverified_values,
                  maximum_scaled_error=max(errors, default=0.),
                  input_checks='Cardinality, dimension, positivity, front normalization, and nested prefixes.',
                  timing_checks='Three samples and exact median for complete workers; no full median for partial workers.',
                  validation='Every retained first and warm value checked against available Chan d/2 first value.',
                  sources_checked=len(report['source_sha256']),
                  scope='Record and cross-implementation audit; not an exact large-instance oracle.',
                  sha256={name: hashlib.sha256((HERE / 'results' / name).read_bytes()).hexdigest()
                          for name in ('datasets.json', 'timings.json', 'timings.csv')})
    (HERE / 'results/audit.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
