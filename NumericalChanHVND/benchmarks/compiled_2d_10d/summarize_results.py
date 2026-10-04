"""Render saved benchmark records; never execute a solver."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ALGORITHMS = ('compiled_numba', 'numerical_python', 'chan_dby3', 'chan_dby2')
NAMES = ('Compiled numerical', 'Numerical Python', 'Chan d/3 Python', 'Chan d/2 Python')


def good(row):
    return row is not None and row.get('status') == 'ok'


def time_cell(row):
    if row is None:
        return 'not applicable'
    return f"{1000 * row['median_seconds']:.3f}" if good(row) else row['status'].replace('_', ' ')


def statuses(rows):
    return ', '.join(f'{count} {status}' for status, count in
                     sorted(Counter(row['status'] for row in rows).items()))


def key_findings(rows, indexed):
    compiled = [row for row in rows if row['algorithm'] == 'compiled_numba']
    ordinary = [row for row in compiled if not row['forced'] and not row['magnitude']
                and row['d'] >= 4]
    lines = [f"The compiled solver completed **{sum(good(row) for row in compiled)} of "
             f"{len(compiled)} recorded inputs**, including direct magnitude and all forced checks."]
    for algorithm, name in (('chan_dby3', 'Chan d/3 Python'),
                            ('numerical_python', 'the original numerical Python')):
        ratios = []
        for row in ordinary:
            baseline = indexed.get((row['id'], algorithm))
            if good(row) and good(baseline):
                ratios.append(baseline['median_seconds'] / row['median_seconds'])
        if ratios:
            lines.append(f"On ordinary-HV inputs in 4D–10D, compiled numerical was faster than "
                         f"{name} on **{sum(r > 1 for r in ratios)}/{len(ratios)} completed pairs**; "
                         f"the median baseline/compiled ratio was **{statistics.median(ratios):.2f}×**.")
    pairs = [(row, indexed.get((row['id'], 'chan_dby2'))) for row in ordinary]
    pairs = [(a, b) for a, b in pairs if good(a) and good(b)]
    if pairs:
        faster = sum(b['median_seconds'] < a['median_seconds'] for a, b in pairs)
        lines.append(f"Chan d/2 Python was faster than compiled numerical on **{faster}/{len(pairs)}** "
                     'completed ordinary-HV pairs in 4D–10D. The new implementation is therefore '
                     'not the fastest solver across all tested dimensions and inputs.')
    default = [row for row in compiled if not row['forced'] and row['d'] >= 4 and good(row)]
    positive = sum(row['counters'].get('compressions', 0) > 0 for row in default)
    checks = sum(row['counters'].get('compression_checks', 0) for row in default)
    lines.append(f"Among the **{len(default)} completed default 4D–10D inputs**, **{positive}** "
                 f"executed compression; there were {checks} size-gate checks in total. Compression "
                 'was enabled, but these states stayed below its threshold. Thus the default '
                 'speedups do not measure the cost of compression when it is required. The '
                 'separate forced cases demonstrate that the implemented path remains operational.')
    lines.append('A separate [natural-compression structural probe](NATURAL_COMPRESSION.md) '
                 'also demonstrates compression under the default settings. It is outside the '
                 '48-input suite, and its first-call measurements are excluded from all warm '
                 'timing comparisons here.')
    return lines


def render(report, cases, oracle=None):
    rows = report['rows']
    indexed = {(row['id'], row['algorithm']): row for row in rows}
    ordinary = [case for case in cases if not case['forced'] and not case['magnitude']]
    magnitude = [case for case in cases if not case['forced'] and case['magnitude']]
    forced = [case for case in cases if case['forced']]
    complete = 'Complete' if 'finished_utc' in report else 'Partial'
    expected = sum(3 if case['d'] == 2 else 4 for case in cases)
    lines = ['# Compiled numerical hypervolume, 2D–10D', '',
             f"{complete} run: {len(cases)} datasets, {len(rows)} of {expected} workers recorded; "
             f"{statuses(rows)}. Validation errors: **{report.get('validation_errors', 'pending')}**.", '',
             '## Key findings', '', '\n\n'.join(key_findings(rows, indexed)), '',
             '## What is compared', '',
             'The compiled solver uses compiled sweeps in 2D/3D and complete native term contractions '
             'in 4D–10D. Its geometric recursion stays in Python. It applies cell-local cropping, '
             'slab tightening, and an adaptive compression gate. Compression is enabled in every '
             'run. A default input can legitimately need no compression; separate forced cases '
             'verify that the compression path runs.', '',
             'Numerical Python is the earlier finite-mass solver with its original local block '
             'schedule. Chan d/3 is the step-function implementation with its default adaptive '
             'schedule, except on the explicitly forced inputs. Chan d/2 is the reference '
             'implementation used to validate every available value. These are implementation '
             'comparisons, not a claim that compilation alone explains every difference.', '',
             '**No asymptotic bound is claimed for the new adaptive implementation in this '
             'benchmark.** The adaptive gate needs a separate complexity proof. These small, '
             'dimension-dependent inputs do not estimate an asymptotic exponent.', '',
             '## Method', '',
             f"Each worker has a {report['worker_timeout_seconds']:g}-second total budget, including "
             f"startup, imports, an untimed first call, and {report['repeats']} timed repetitions. "
             'The table reports warm medians in milliseconds. Input transforms and preprocessing '
             'are included; imports are excluded from call timings. Workers run sequentially and '
             'construct a new solver on every call. Disk caches are retained and warmed by '
             'verification beforehand; first-use times are archived separately and are not '
             'empty-cache compilation times. No dominance prefilter, fastmath, or parallel '
             'numerical execution is used.', '',
             'A timeout is a worker-budget result, not a lower bound on one call. No timing ratio '
             'uses a timeout. Completed-pair summaries can have selection bias; read them with '
             'the status counts. Raw samples, inputs, counters, source hashes, and Numba signatures '
             'are saved in [results](results/).', '',
             'The ordinary box union is identical for every implementation. Direct numerical '
             'magnitude is checked against HV after the affine transformation `1 + extent/2`. '
             'Sphere corners are mutually nondominated; tied grids can have duplicate or '
             'dominated corners. The previous 3D–10D inputs are reproduced exactly, with five '
             '2D inputs added.', '',
             'Environment: ' + '; '.join(f'{key}: `{value}`' for key, value in
                 report['environment'].items()) + '.', '',
             '## Ordinary hypervolume', '',
             '| Input | ' + ' | '.join(NAMES) + ' |',
             '|---|' + '---:|' * len(NAMES)]
    for case in ordinary:
        lines.append('| ' + case['id'] + ' | ' + ' | '.join(
            time_cell(indexed.get((case['id'], name))) for name in ALGORITHMS) + ' |')
    spreads = [max(row['seconds']) / min(row['seconds']) for row in rows
               if row['algorithm'] == 'compiled_numba' and good(row) and min(row['seconds']) > 0]
    if spreads:
        lines += ['', f"Across the {len(spreads)} completed compiled workers, the ratio of slowest "
                  f"to fastest warm repetition had median **{statistics.median(spreads):.2f}** "
                  f"and maximum **{max(spreads):.2f}**; {sum(x > 2 for x in spreads)} workers "
                  'exceeded a ratio of two. These are three wall-clock samples per case under '
                  'ordinary system load, without a dedicated machine or CPU isolation. The '
                  'recorded variability limits the precision of small timing differences; '
                  'the measurements do not supply confidence intervals.']
    lines += ['', '## Completed-pair comparisons', '',
              'Ratios are baseline time divided by compiled numerical time. Above one means '
              'compiled numerical is faster. Each entry gives the median per-input ratio and '
              'number of completed ordinary-HV pairs. Sizes change with dimension.', '',
              '| Dimension | Compiled statuses | Numerical Python / compiled | Chan d/3 / compiled | Chan d/2 / compiled |',
              '|---|---|---:|---:|---:|']
    for dimension in range(2, 11):
        selected = [case for case in ordinary if case['d'] == dimension]
        if not selected:
            continue
        compiled = [indexed[case['id'], 'compiled_numba'] for case in selected
                    if (case['id'], 'compiled_numba') in indexed]
        cells = []
        for algorithm in ALGORITHMS[1:]:
            ratios = []
            for case in selected:
                a, b = indexed.get((case['id'], algorithm)), indexed.get((case['id'], 'compiled_numba'))
                if good(a) and good(b) and b['median_seconds'] > 0:
                    ratios.append(a['median_seconds'] / b['median_seconds'])
            cells.append(f'{statistics.median(ratios):.3g}× ({len(ratios)})' if ratios else 'not applicable')
        lines.append(f'| {dimension}D | {statuses(compiled)} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Magnitude', '', '| Input | ' + ' | '.join(NAMES) + ' |',
              '|---|' + '---:|' * len(NAMES)]
    for case in magnitude:
        lines.append('| ' + case['id'] + ' | ' + ' | '.join(
            time_cell(indexed.get((case['id'], name))) for name in ALGORITHMS) + ' |')
    lines += ['', '## Forced compression checks', '',
              'These extra inputs take the first eight corners of the saved 4D sphere and append '
              'constant unit extents to obtain 7D and 10D. Compiled numerical and Chan d/3 use '
              '`compress_every=2, compress_factor=0, use_compression=True`. Numerical Python '
              'retains its own schedule. These timings are excluded from the performance ratios '
              'above. A completed compiled row is accepted only if its compression counter is '
              'positive and its values agree with Chan d/2.', '',
              '| Input | ' + ' | '.join(NAMES) + ' | Compiled compressions | Compiled peak terms |',
              '|---|' + '---:|' * (len(NAMES) + 2)]
    for case in forced:
        row = indexed.get((case['id'], 'compiled_numba'), {})
        counters = row.get('counters', row.get('first_counters', {}))
        cells = [time_cell(indexed.get((case['id'], name))) for name in ALGORITHMS]
        cells += [str(counters.get(name, 'unavailable')) for name in ('compressions', 'peak_terms')]
        lines.append('| ' + case['id'] + ' | ' + ' | '.join(cells) + ' |')
    errors = [row['scaled_error'] for row in rows if 'scaled_error' in row]
    lines += ['', '## Validation and interpretation', '',
              f"Maximum recorded scaled error against Chan d/2: {max(errors, default=0.):.3g}. "
              'Every first-call value and every retained warm value is checked with relative '
              'tolerance `2e-9` and absolute tolerance `2e-10`. Agreement between implementations '
              'complements the independent exact checks in the solver verification suite.', '']
    if oracle is not None:
        forced_count = sum(case['id'].startswith('forced-') for case in oracle['cases'])
        forced_text = f", including {forced_count} forced-compression inputs" if forced_count else ''
        lines += [f"The [independent exact oracle](results/oracle_verification.json) "
                  f"**{oracle['status']}** on **{oracle['checked_cases']} saved datasets** "
                  f"and **{oracle['checked_values']} recorded values**{forced_text}. "
                  f"It evaluates exact rational inclusion–exclusion "
                  f"for the binary64 coordinates on inputs of at most {oracle['max_points']} "
                  f"points. Its maximum scaled error was **{oracle['max_scaled_error']:.3g}**. "
                  'The archived oracle input hashes match these timing and dataset files.', '']
    lines += [
              'Counters distinguish a small recursion from a large contraction: compression '
              'lifts `d` axes to `2d`, and elimination can expand signed terms before merging. '
              'Native compilation removes Python work inside these contractions; it does not '
              'remove their mathematical expansion. Compare counters as well as elapsed time.', '',
              'For an input that actually crosses the adaptive threshold without forcing it, '
              'see the separate [natural-compression report](NATURAL_COMPRESSION.md). Its '
              'structural evidence supplements the zero-compression default cases in this '
              'main suite; it is not added to their timing aggregates.', '',
              'Previous results: [original 3D–10D hybrid comparison](../current_3d_10d/RESULTS.md) '
              'and [earlier dedicated 4D whole-contraction comparison]'
              '(../../../NumericalChanHV4D/benchmarks/whole_loop/RESULTS.md).', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=HERE / 'results')
    parser.add_argument('--out', type=Path, default=HERE / 'RESULTS.md')
    args = parser.parse_args()
    report = json.loads((args.results / 'timings.json').read_text(encoding='utf-8'))
    cases = json.loads((args.results / 'datasets.json').read_text(encoding='utf-8'))
    oracle_path = args.results / 'oracle_verification.json'
    oracle = json.loads(oracle_path.read_text(encoding='utf-8')) if oracle_path.exists() else None
    if oracle is not None:
        for name, expected in oracle['input_sha256'].items():
            if hashlib.sha256((args.results / name).read_bytes()).hexdigest() != expected:
                parser.error('Oracle hashes do not match the current inputs; verify the new run first')
    args.out.write_text(render(report, cases, oracle), encoding='utf-8')
    print(args.out)


if __name__ == '__main__':
    main()
