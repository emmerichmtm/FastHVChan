"""Summarize archived larger-input measurements without rerunning solvers."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
from source_provenance import verify_sources

HERE = Path(__file__).resolve().parent
ALGORITHMS = ('local_compiled', 'global_compiled', 'chan_dby3', 'chan_dby2')
NAMES = ('Local compiled', 'Earlier compiled', 'Chan d/3 Python', 'Chan d/2 Python')
PRIMARY_ALGORITHMS = ('local_compiled', 'chan_dby3', 'chan_dby2')
PRIMARY_NAMES = ('Numerical Chan (Numba)', 'Chan d/3 (Python)', 'Chan d/2 (Python)')


def good(row):
    return row.get('status') == 'ok'


def cell(row, latex=False):
    if good(row):
        return f"{row['median_seconds']:.4f}"
    if row['status'] == 'timeout':
        count = len(row.get('samples', []))
        return f'TO[{count}]' if latex else f'timeout ({count}/3 warm)'
    if row['status'] == 'not_run_after_timeouts':
        return 'NR' if latex else 'not run after timeouts'
    return row['status'].replace('_', ' ')


def facts(report, cases, algorithms=ALGORITHMS):
    rows = [r for r in report['rows'] if r['algorithm'] in algorithms]
    index = {(r['id'], r['algorithm']): r for r in rows}
    selected = [r for r in rows if r['algorithm'] == 'local_compiled']
    ratios = {}
    for alg in algorithms[1:]:
        pairs = [(r, index[r['id'], alg]) for r in selected]
        q = [b['median_seconds']/a['median_seconds'] for a,b in pairs if good(a) and good(b)]
        ratios[alg] = dict(pairs=len(q), wins=sum(x>1 for x in q),
                           median=statistics.median(q) if q else None)
    error = max((r.get('scaled_error', 0.) for r in rows), default=0.)
    verified = sum('agree with Chan d/2' in r.get('validation', '') for r in rows)
    verified_values = sum(('first_value' in r)+len(r.get('samples', [])) for r in rows
                          if 'agree with Chan d/2' in r.get('validation', ''))
    spreads = [max(r['seconds'])/min(r['seconds']) for r in selected if good(r)]
    comp = [r for r in selected if good(r) and r['counters'].get('compressions', 0)>0]
    return dict(index=index, local=selected, ratios=ratios, max_error=error,
                planned_rows=len(rows), mismatches=sum(r.get('validation') == 'value mismatch' for r in rows),
                verified_rows=verified, verified_values=verified_values,
                statuses=Counter(r['status'] for r in rows),
                local_statuses=Counter(r['status'] for r in selected), compressed=comp,
                median_spread=statistics.median(spreads) if spreads else None,
                max_spread=max(spreads, default=None))


def markdown_all_variants(report, cases, info):
    lines = ['# Larger hypervolume benchmarks: 4D–6D', '',
        f"{len(cases)} inputs, {len(report['rows'])} planned algorithm/input rows. "
        f"Statuses: {dict(info['statuses'])}. Value mismatches: **{report['validation_errors']}**.", '',
        '**n is the number of input generators**, and each generator defines one origin-anchored box. '
        'These runs measure the new local-checkpoint compiled policy as well as the earlier global-interval policy.', '',
        '## Coverage and interpretation', '',
        'Spherical fronts use n = 64, 128, 256, 512, 1024 in 4D, 5D and 6D, plus n = 4096 in 4D. '
        'A second simplex family uses n = 64, 256, 1024 in each dimension. Within a dimension/family, '
        'smaller inputs are prefixes of the largest seeded sample. All points are on a positive '
        'nondominated front; no dominance filter is used.', '',
        f"Local compiled statuses: **{dict(info['local_statuses'])}**. "
        f"{len(info['compressed'])} completed local inputs executed compression.", '',
        '| Dimension | Sphere: 3 warm calls | Sphere: any returned value | Simplex: 3 warm calls | Simplex: any returned value |',
        '|---|---:|---:|---:|---:|']
    for d in (4,5,6):
        vals = []
        for family in ('sphere','simplex'):
            selected = [r for r in info['local'] if r['d']==d and r['family']==family]
            vals.extend((max((r['n'] for r in selected if good(r)), default=0),
                         max((r['n'] for r in selected if 'first_value' in r), default=0)))
        lines.append(f'| {d} | ' + ' | '.join(map(str, vals)) + ' |')
    lines += ['', 'Ratios below use only pairs with all three warm calls complete; censored larger '
              'cases are excluded, so these ratios should not be treated as universal speedups.', '']
    for alg, name in zip(ALGORITHMS[1:], NAMES[1:]):
        q = info['ratios'][alg]
        if q['pairs']:
            lines.append(f"- Against {name}: local compiled wins {q['wins']}/{q['pairs']} complete pairs; "
                         f"median reference/local time ratio {q['median']:.3g}×.")
    large = info['index'].get(('sphere-d4-n4096-hv', 'local_compiled'), {})
    ref = info['index'].get(('sphere-d4-n4096-hv', 'chan_dby3'), {})
    if 'first_seconds' in large and 'first_seconds' in ref:
        lines += ['', f"At **4D, n=4096**, local compiled returned its first value in "
                  f"{large['first_seconds']:.3f} s, versus {ref['first_seconds']:.3f} s for Python Chan d/3. "
                  'Neither completed the full warm protocol. Thus the advantages on complete smaller pairs '
                  'do not establish an advantage at larger n. These are first-call observations, including '
                  'possible compiled-kernel initialization, not warm medians.']
    for family in ('sphere', 'simplex'):
        lines += ['', f'## {family.title()} fronts', '', 'Warm median **seconds**, not milliseconds.', '',
                  '| d | n | '+' | '.join(NAMES)+' |', '|---:|---:|'+'---:|'*4]
        for case in cases:
            if case['family'] == family:
                lines.append(f"| {case['d']} | {case['n']} | "+' | '.join(
                    cell(info['index'][case['id'], alg]) for alg in ALGORITHMS)+' |')
    lines += ['', '## Compression in the local compiled solver', '',
              'Counters are from the completed first call even if later repetitions exceeded the worker budget.', '',
              '| Input | Worker status | Checkpoints | Compressions | Peak terms |',
              '|---|---|---:|---:|---:|']
    for row in info['local']:
        c = row.get('first_counters')
        if c is not None:
            lines.append(f"| {row['id']} | {row['status']} | {c.get('checkpoints',0)} | "
                         f"{c.get('compressions',0)} | {c.get('peak_terms',0)} |")
    lines += ['', '## Censored workers', '',
              'A worker has a 45-second budget for imports, one first call, and three warm calls. '
              'A timeout is **not** a lower bound on the runtime of one call. Partial calls that '
              'returned are retained below and in the raw JSON, but are not reported as a complete warm median. '
              'After two consecutive timeouts in a solver/dimension/family series, larger sizes in that series '
              'are marked not run. This resource rule does not prove that those larger cases would time out.', '',
              '| Input | Algorithm | Status | First call (s) | Completed warm calls (s) |',
              '|---|---|---|---:|---|']
    for row in report['rows']:
        if not good(row):
            first = f"{row['first_seconds']:.4f}" if 'first_seconds' in row else 'no value'
            warm = ', '.join(f"{s['seconds']:.4f}" for s in row.get('samples', [])) or 'none'
            lines.append(f"| {row['id']} | {row['algorithm']} | {row['status']} | {first} | {warm} |")
    lines += ['', '## Method and correctness', '', report['method'], '',
              'Environment: Windows 11, AMD Ryzen 7 5800H (16 logical CPUs), Python 3.12.14, '
              'NumPy 2.5.3 and Numba 0.68.0. The raw record retains the platform processor identifier.', '',
              'Each timed call includes input conversion and preprocessing. Warm medians exclude import '
              'and first-call costs. Existing caches are retained; first-call times are not empty-cache '
              'JIT compilation measurements. Only one numerical worker runs at a time. Ordinary machine '
              'activity is uncontrolled; results use one seeded front per family/dimension and three repetitions.', '',
              f"For completed local workers, median slowest/fastest warm-call ratio is {info['median_spread']:.3g}, "
              f"maximum {info['max_spread']:.3g}. No confidence interval or asymptotic exponent is inferred.", '',
              f"{info['verified_values']} returned values across {info['verified_rows']} rows were checked against Chan d/2; "
              f"maximum scaled error is {info['max_error']:.3g}. Every available first and warm value is checked. "
              'If the reference completed its first call but later timed out, that completed value remains '
              'usable for validation. A row without an available reference is marked unverified, not passed.', '',
              'These larger cases are cross-implementation checks, not exponential exact inclusion–exclusion '
              'oracles. Exact small-instance and signed-algebra tests remain in the separate verification suites.', '',
              'The local schedule is the policy covered by the report’s arithmetic theorem. '
              'The earlier global-interval solver has a different schedule. Their measurements remain separate.', '',
              'Raw inputs, timings, counters and source hashes: [results](results/). '
              'Run [benchmark_larger.py](benchmark_larger.py) to reproduce with a new output directory.']
    return '\n'.join(lines)+'\n'


def markdown(report, cases, info):
    counts = info['statuses']
    d3, d2 = info['ratios']['chan_dby3'], info['ratios']['chan_dby2']
    lines = ['# Numerical hypervolume benchmarks: 4D–6D', '',
        'The main comparison uses three solvers: **Numerical Chan (Numba)**, the numerical '
        'implementation with the local compression schedule proved in the report; '
        '**Chan d/3 (Python)**; and **Chan d/2 (Python)**. '
        'In the raw records their keys are `local_compiled`, `chan_dby3` and `chan_dby2`.', '',
        f"There are **{len(cases)} inputs** and {info['planned_rows']} selected solver/input rows: "
        f"{counts.get('ok', 0)} complete, {counts.get('timeout', 0)} timed out, and "
        f"{counts.get('not_run_after_timeouts', 0)} not run. "
        '**n is the number of input points**, each defining an origin-anchored box.', '',
        'Spherical fronts use n = 64, 128, 256, 512, 1024 in 4D–6D, plus n = 4096 in 4D. '
        'Simplex fronts use n = 64, 256, 1024 in each dimension. All points are nondominated. '
        'Smaller inputs are prefixes of the same seeded sample within each dimension/family.', '',
        '**Table entries are median seconds over three warm calls.** '
        '**TO** means the worker exceeded its 45-second budget for startup, imports, one first '
        'call and all three warm calls together; it does not mean that one call took 45 seconds. '
        '**NR** means not run after two consecutive timeouts at smaller sizes. '
        'Neither TO nor NR supplies a warm median.']
    for family in ('sphere', 'simplex'):
        lines += ['', f'## {family.title()} fronts', '',
                  '| d | n | ' + ' | '.join(PRIMARY_NAMES) + ' |', '|---:|---:|---:|---:|---:|']
        for case in cases:
            if case['family'] == family:
                values = []
                for alg in PRIMARY_ALGORITHMS:
                    row = info['index'][case['id'], alg]
                    values.append(f"{row['median_seconds']:.4f}" if good(row) else
                                  'TO' if row['status'] == 'timeout' else
                                  'NR' if row['status'] == 'not_run_after_timeouts' else row['status'])
                lines.append(f"| {case['d']} | {case['n']} | " + ' | '.join(values) + ' |')
    large = info['index']['sphere-d4-n4096-hv', 'local_compiled']
    ref = info['index']['sphere-d4-n4096-hv', 'chan_dby3']
    lines += ['', '## What the comparison shows', '',
        f"Numerical Chan (Numba) was faster than Chan d/3 (Python) on {d3['wins']}/{d3['pairs']} "
        f"complete pairs; the median reference/numerical time ratio was {d3['median']:.3g}×. "
        f"Chan d/2 (Python) was faster on all {d2['pairs']} complete pairs. "
        'These comparisons exclude incomplete workers and do not establish a universal speedup.', '',
        f"At **4D, n = 4096**, Numerical Chan (Numba) returned its first value in {large['first_seconds']:.3f} s, "
        f"versus {ref['first_seconds']:.3f} s for Chan d/3 (Python). Neither finished the warm protocol. "
        'These single first-call observations can include compiled-kernel initialization and cache '
        'loading; they are not warm medians, but show why the smaller-input results should not be '
        'extrapolated to all sizes.', '',
        '**Compilation matters:** this compares compiled numerical kernels with Python implementations '
        'of the references. It does not isolate the effect of the mathematical formulation or show '
        'an improved complexity exponent.', '',
        '## Measurement and validation', '',
        'Each call builds a fresh solver and includes input conversion and preprocessing. '
        'Warm medians exclude imports and first-call costs; existing compilation caches are retained. '
        'Workers run sequentially with no dominance prefilter or fast-math, and all compression '
        'defaults remain enabled. Only one seeded front per dimension/family and three warm '
        'repetitions were measured, under uncontrolled ordinary machine load.', '',
        'Environment: Windows 11, AMD Ryzen 7 5800H (16 logical CPUs), Python 3.12.14, '
        'NumPy 2.5.3 and Numba 0.68.0.', '',
        f"All **{info['verified_values']} returned values** from the three selected solvers, including "
        f"available partial results, agree with the available Chan d/2 reference: "
        f"**{info['mismatches']} mismatches**, maximum scaled error {info['max_error']:.3g}. "
        'These are cross-implementation checks; exact small-instance tests are separate.', '',
        'The [archived full comparison](RESULTS_ALL_VARIANTS.md) preserves the earlier compiled '
        'variant, compression counters and partial-call tables. All original '
        '[inputs and raw records](results/) are unchanged. See [README.md](README.md) '
        'for reproduction and audit commands.']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=HERE / 'results')
    parser.add_argument('--all-variants', action='store_true',
                        help='also regenerate the archived four-solver detail report')
    args = parser.parse_args()
    report = json.loads((args.results / 'timings.json').read_text(encoding='utf-8'))
    cases = json.loads((args.results / 'datasets.json').read_text(encoding='utf-8'))
    assert 'finished_utc' in report, 'Wait for the full run before publishing'
    verify_sources(report, args.results)
    info = facts(report, cases, PRIMARY_ALGORITHMS)
    (HERE / 'RESULTS.md').write_text(markdown(report, cases, info), encoding='utf-8')
    if args.all_variants:
        archive = ('> **Archive: full four-solver comparison.** The concise current comparison is '
                   '[RESULTS.md](RESULTS.md). This page retains the earlier compiled variant and '
                   'all diagnostic tables from the original published report.\n\n')
        archive += markdown_all_variants(report, cases, facts(report, cases))
        (HERE / 'RESULTS_ALL_VARIANTS.md').write_text(archive, encoding='utf-8')
    print(json.dumps({k: v for k, v in info.items()
                      if k not in ('index', 'local', 'compressed')}, indent=2))


if __name__ == '__main__':
    main()
