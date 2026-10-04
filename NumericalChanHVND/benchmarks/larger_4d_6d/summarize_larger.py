"""Summarize archived larger-input measurements without rerunning solvers."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
from source_provenance import verify_sources

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ALGORITHMS = ('local_compiled', 'global_compiled', 'chan_dby3', 'chan_dby2')
NAMES = ('Local compiled', 'Earlier compiled', 'Chan d/3 Python', 'Chan d/2 Python')


def good(row):
    return row.get('status') == 'ok'


def tex_scientific(value):
    mantissa, exponent = f'{value:.4e}'.split('e')
    return mantissa + r'\times10^{' + str(int(exponent)) + '}'


def cell(row, latex=False):
    if good(row):
        return f"{row['median_seconds']:.4f}"
    if row['status'] == 'timeout':
        count = len(row.get('samples', []))
        return f'TO[{count}]' if latex else f'timeout ({count}/3 warm)'
    if row['status'] == 'not_run_after_timeouts':
        return 'NR' if latex else 'not run after timeouts'
    return row['status'].replace('_', ' ')


def first_cell(row):
    if 'first_seconds' in row:
        return f"{row['first_seconds']:.3f}"
    return 'NR' if row['status'] == 'not_run_after_timeouts' else '---'


def facts(report, cases):
    rows = report['rows']
    index = {(r['id'], r['algorithm']): r for r in rows}
    selected = [r for r in rows if r['algorithm'] == 'local_compiled']
    ratios = {}
    for alg in ALGORITHMS[1:]:
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
                verified_rows=verified, verified_values=verified_values,
                statuses=Counter(r['status'] for r in rows),
                local_statuses=Counter(r['status'] for r in selected), compressed=comp,
                median_spread=statistics.median(spreads) if spreads else None,
                max_spread=max(spreads, default=None))


def markdown(report, cases, info):
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


def latex(report, cases, info):
    lines = [r'\section{Experiments with larger inputs in 4D--6D}', r'\label{sec:experiments}', '',
        'This additional experiment measures the local-checkpoint implementation covered by '
        'Theorem~\\ref{thm:main}, alongside the earlier global-interval compiled variant and both '
        'Python Chan references. The solvers were unchanged from source revision \\texttt{095fa73}; '
        'hashes identify the exact sources and inputs. Here $n$ denotes input generators, one per anchored box.', '',
        'The spherical inputs use $n=64,128,256,512,1024$ in each of dimensions four, five and six, '
        'and $n=4096$ in four dimensions. A second family on the positive simplex uses '
        '$n=64,256,1024$ in each dimension. Smaller inputs are nested prefixes within each family '
        'and dimension. Spherical coordinates are absolute Gaussian samples plus $0.01$, normalized '
        'in Euclidean norm; simplex coordinates are exponential samples plus $0.01$, normalized '
        'by their sum. Both constructions produce positive nondominated fronts.', '',
        'Each fresh sequential worker makes one first call and three warm calls, with a new solver '
        'on each call. Warm medians include input conversion and preprocessing but exclude imports '
        'and first use. Existing compilation caches are retained. The whole worker, including imports '
        'and all four calls, has a 45-second budget. A timeout therefore does not imply that one '
        'call took 45 seconds. Available partial timings and values are archived. After two consecutive '
        'timeouts in a solver/family/dimension series, larger sizes in that series are explicitly '
        'marked not run; no runtime is inferred for them.', '',
        'All default compression settings remain enabled. There is no dominance prefilter, fast-math '
        'or concurrent numerical worker. The processor is an AMD Ryzen 7 5800H with 16 logical CPUs. '
        'The environment is the same Windows/Python/Numba setup '
        'described in Section~\\ref{sec:archived}. Ordinary system load is uncontrolled.', '',
        'Tables~\\ref{tab:larger-sphere} and~\\ref{tab:larger-simplex} report complete warm medians. '
        'Table~\\ref{tab:larger-first} retains first-call observations on selected large inputs, '
        'including partial workers.', '']
    for family in ('sphere','simplex'):
        lines += [r'\begin{table}[tbp]', r'\centering\small', r'\setlength{\tabcolsep}{5pt}',
                  r'\begin{tabular}{rrrrrr}', r'\toprule',
                  r'$d$ & $n$ & \shortstack{Local\\compiled} & \shortstack{Earlier\\compiled} & '
                  r'\shortstack{Chan $d/3$\\Python} & \shortstack{Chan $d/2$\\Python} \\', r'\midrule']
        for case in cases:
            if case['family']==family:
                lines.append(f"{case['d']} & {case['n']} & "+' & '.join(
                    cell(info['index'][case['id'], alg], True) for alg in ALGORITHMS)+r' \\')
        lines += [r'\bottomrule',r'\end{tabular}',
                  '\\caption{'+family.title()+' fronts at larger input sizes: warm median \\emph{seconds}. '
                  'Local compiled uses the proved local-checkpoint policy; earlier compiled uses '
                  'the global-interval policy. TO[$k$] means worker timeout with $k$ complete warm '
                  'calls; NR means not run after two smaller-size timeouts.}',
                  r'\label{tab:larger-'+family+'}',r'\end{table}', '']
    lines += [r'\begin{table}[tbp]', r'\centering\small', r'\setlength{\tabcolsep}{4pt}',
              r'\begin{tabular}{lrrrrrr}', r'\toprule',
              r'Front & $d$ & $n$ & \shortstack{Local\\compiled} & \shortstack{Earlier\\compiled} & '
              r'\shortstack{Chan $d/3$\\Python} & \shortstack{Chan $d/2$\\Python} \\', r'\midrule']
    for case in cases:
        if case['n'] >= 1024 or (case['d'] == 6 and case['n'] == 512):
            lines.append(f"{case['family'].title()} & {case['d']} & {case['n']} & "+' & '.join(
                first_cell(info['index'][case['id'], alg]) for alg in ALGORITHMS)+r' \\')
    lines += [r'\bottomrule', r'\end{tabular}',
              r'\caption{Retained first-call seconds on the largest inputs, including workers that '
              r'later timed out. These are single observations, not warm medians; compiled first calls '
              r'can include kernel initialization and cache loading. Imports are excluded. '
              r'A dash means no first value returned within the worker budget; NR means not run.}',
              r'\label{tab:larger-first}', r'\end{table}', '']
    counts = info['statuses']
    lines += [f"There are {len(cases)} inputs and {len(report['rows'])} planned algorithm/input rows: "
              f"{counts.get('ok',0)} complete workers, {counts.get('timeout',0)} timeouts, and "
              f"{counts.get('not_run_after_timeouts',0)} not run by the escalation rule. "
              f"The local solver completes {info['local_statuses'].get('ok',0)} of its {len(cases)} rows. "
              'The tables retain all planned sizes so the missing large cases remain visible.', '']
    for alg,name in zip(ALGORITHMS[1:], ('the earlier compiled solver', 'Python Chan $d/3$', 'Python Chan $d/2$')):
        q=info['ratios'][alg]
        if q['pairs']:
            lines += [f"Against {name}, local compiled is faster on {q['wins']} of {q['pairs']} "
                      f"complete pairs, with median reference/local ratio {q['median']:.3g}. "
                      'These ratios omit censored pairs and are not universal speed guarantees.', '']
    large = info['index'].get(('sphere-d4-n4096-hv', 'local_compiled'), {})
    ref = info['index'].get(('sphere-d4-n4096-hv', 'chan_dby3'), {})
    if 'first_seconds' in large and 'first_seconds' in ref:
        lines += [f"For four dimensions and $n=4096$, local compiled returns its first value in "
                  f"{large['first_seconds']:.3f} seconds, versus {ref['first_seconds']:.3f} seconds "
                  'for Python Chan $d/3$. Neither completes all three warm calls. This first-call '
                  'observation shows why the complete-pair speedups should not be extrapolated to '
                  'larger inputs; it is not a warm-median comparison.', '']
    lines += [f"Compression is executed on {len(info['compressed'])} completed local inputs. "
              'Per-input checkpoint, compression and peak-term counts are retained with the raw data. '
              'The size rule is unchanged; absence of compression in an individual input is a recorded '
              'algorithm decision, not a disabled feature.', '',
              f"All {info['verified_values']} returned values in {info['verified_rows']} rows are checked against an available "
              f"Chan $d/2$ value; there are {report['validation_errors']} mismatches and the maximum "
              f"scaled error is ${tex_scientific(info['max_error'])}$. "
              'This checks every retained first and warm value, including partial workers. '
              'A completed first reference value remains valid even if that worker later exceeds its '
              'budget. These are cross-implementation comparisons; the larger sets were not evaluated '
              'by an exponential exact inclusion--exclusion oracle.', '',
              f"Among complete local workers, the ratio of slowest to fastest warm call has median "
              f"{info['median_spread']:.3g} and maximum {info['max_spread']:.3g}. "
              'The sample is one seeded front per dimension/family and three timed repetitions. '
              'It demonstrates performance and limits at larger $n$, not a fitted asymptotic exponent. '
              'The worst-case bound continues to rest on Theorem~\\ref{thm:main}. '
              'Full inputs, statuses and retained samples are in '
              '\\texttt{benchmarks/larger\\_4d\\_6d/results}.', '']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=HERE/'results')
    args=parser.parse_args()
    report=json.loads((args.results/'timings.json').read_text(encoding='utf-8'))
    cases=json.loads((args.results/'datasets.json').read_text(encoding='utf-8'))
    assert 'finished_utc' in report, 'Wait for the full run before publishing'
    verify_sources(report, args.results)
    info=facts(report,cases)
    (HERE/'RESULTS.md').write_text(markdown(report,cases,info),encoding='utf-8')
    (ROOT/'NumericalChanHVND/paper/numerical_product_chan_larger.tex').write_text(latex(report,cases,info),encoding='utf-8')
    print(json.dumps({k:v for k,v in info.items() if k not in ('index','local','compressed')},indent=2))


if __name__=='__main__':
    main()
