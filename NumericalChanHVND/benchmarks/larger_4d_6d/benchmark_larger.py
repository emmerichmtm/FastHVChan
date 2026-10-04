"""Larger, nested nondominated fronts in 4D--6D; sequential bounded workers.

The local-checkpoint solver is measured separately from the earlier adaptive
solver. Preserve every input, first call, repetition, counter and timeout.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / 'compiled_2d_10d'))
import benchmark_dimensions as previous

ALGORITHMS = ('local_compiled', 'global_compiled', 'chan_dby3', 'chan_dby2')


def make_cases():
    cases = []
    for family in ('sphere', 'simplex'):
        for dimension in (4, 5, 6):
            sizes = [64, 128, 256, 512, 1024] if family == 'sphere' else [64, 256, 1024]
            if family == 'sphere' and dimension == 4:
                sizes.append(4096)
            seed = 460000 + 100 * dimension + (family == 'simplex')
            rng = random.Random(seed)
            points = []
            for _ in range(max(sizes)):
                if family == 'sphere':
                    vector = [abs(rng.gauss(0., 1.)) + .01 for _ in range(dimension)]
                    divisor = math.sqrt(sum(x*x for x in vector))
                else:
                    vector = [rng.expovariate(1.) + .01 for _ in range(dimension)]
                    divisor = sum(vector)
                points.append([x/divisor for x in vector])
            for size in sizes:
                cases.append(dict(id=f'{family}-d{dimension}-n{size}-hv', family=family,
                                  d=dimension, n=size, seed=seed, magnitude=False,
                                  forced=False, points=points[:size]))
    return cases


def setup(algorithm, case):
    if algorithm == 'local_compiled':
        sys.path.insert(0, str(REPOSITORY))
        os.environ.pop('NUMBA_DISABLE_JIT', None)
        from NumericalChanHVND.numerical_chan_local_compiled import NumericalChanLocalCompiled

        def solve():
            engine = NumericalChanLocalCompiled(dimension=case['d'])
            value = engine.compute(case['points'], magnitude=case['magnitude'])
            return float(value), dict(engine.stats)
        return solve
    mapped = 'compiled_numba' if algorithm == 'global_compiled' else algorithm
    return previous.setup(mapped, case)


def worker(pipe, algorithm, case, repeats):
    try:
        started = time.perf_counter()
        solve = setup(algorithm, case)
        pipe.send(dict(event='imports', imports_seconds=time.perf_counter()-started))
        started = time.perf_counter()
        value, counters = solve()
        first = dict(first_seconds=time.perf_counter()-started,
                     first_value=value, first_counters=counters)
        pipe.send(dict(event='first', **first))
        samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            value, counters = solve()
            sample = dict(seconds=time.perf_counter()-started, value=value, counters=counters)
            samples.append(sample)
            pipe.send(dict(event='sample', sample=sample))
        signatures = previous.signatures('compiled_numba', case['d']) if algorithm.endswith('compiled') else {}
        pipe.send(dict(event='done', status='ok', **first, samples=samples,
                       seconds=[s['seconds'] for s in samples], value=value, counters=counters,
                       median_seconds=statistics.median(s['seconds'] for s in samples),
                       nopython_signatures=signatures))
    except Exception as error:
        pipe.send(dict(event='done', status='error', error=repr(error), traceback=traceback.format_exc()))
    finally:
        pipe.close()


def isolated(algorithm, case, repeats, timeout):
    reader, writer = mp.Pipe(duplex=False)
    process = mp.Process(target=worker, args=(writer, algorithm, case, repeats))
    started = time.monotonic()
    process.start()
    writer.close()
    partial = dict(samples=[])
    try:
        deadline = started + timeout
        while time.monotonic() < deadline:
            if not reader.poll(min(.1, max(0., deadline-time.monotonic()))):
                continue
            try:
                event = reader.recv()
            except EOFError:
                process.join(2)
                return dict(partial, status='process_failed', exit_code=process.exitcode)
            if event['event'] == 'done':
                process.join(2)
                return {**partial, **event}
            if event['event'] in ('imports', 'first'):
                partial.update({k:v for k,v in event.items() if k != 'event'})
            else:
                partial['samples'].append(event['sample'])
        return dict(partial, status='timeout', budget_seconds=timeout,
                    worker_seconds=time.monotonic()-started)
    finally:
        if process.is_alive():
            process.terminate()
        process.join(3)
        reader.close()


def available_values(row):
    return ([row['first_value']] if 'first_value' in row else []) + [s['value'] for s in row.get('samples', [])]


def validate(rows):
    reference = next(r for r in rows if r['algorithm'] == 'chan_dby2')
    values = available_values(reference)
    expected = values[0] if values else None
    failures = 0
    for row in rows:
        values = available_values(row)
        if not values:
            row['validation'] = 'no completed value'
        elif expected is None:
            row['validation'] = 'unverified: Chan d/2 produced no value'
        else:
            row['scaled_error'] = max(abs(v-expected)/max(1., abs(expected)) for v in values)
            agrees = all(math.isfinite(v) and math.isclose(v, expected, rel_tol=2e-9, abs_tol=2e-10) for v in values)
            row['validation'] = 'all available values agree with Chan d/2' if agrees else 'value mismatch'
            failures += not agrees
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=HERE/'results')
    parser.add_argument('--timeout', type=float, default=45.)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--only', default='')
    args = parser.parse_args()
    if args.timeout <= 0 or args.repeats < 1:
        parser.error('Positive timeout and repeat count required')
    if (args.out/'timings.json').exists():
        parser.error('Choose a new output directory; retained timings are never overwritten')
    args.out.mkdir(parents=True, exist_ok=True)
    cases = [case for case in make_cases() if args.only in case['id']]
    dataset = args.out/'datasets.json'
    dataset.write_text(json.dumps(cases, indent=2)+'\n', encoding='utf-8')
    files = [Path(__file__), Path(previous.__file__), dataset] + [REPOSITORY/p for p in (
        'NumericalChanHVND/numerical_chan_local_compiled.py',
        'NumericalChanHVND/numerical_chan_compiled.py', 'NumericalChanHVND/compiled_terms.py',
        'NumericalChanHVND/numerical_chan_numba.py', 'chan_orthant_dby3.py', 'chan_hypervolume.py')]
    revision = subprocess.check_output(['git', '-c', f'safe.directory={REPOSITORY.as_posix()}',
        '-C', str(REPOSITORY), 'rev-parse', 'HEAD'], text=True).strip()
    report = dict(started_utc=datetime.now(timezone.utc).isoformat(), source_revision=revision,
        source_sha256={str(p.relative_to(REPOSITORY)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        environment=dict(python=sys.version, platform=platform.platform(), cpu=platform.processor(),
                         logical_cpus=os.cpu_count(), numpy=importlib.metadata.version('numpy'),
                         numba=importlib.metadata.version('numba')),
        repeats=args.repeats, worker_timeout_seconds=args.timeout, algorithms=ALGORITHMS,
        method='Nested positive spherical and simplex fronts. Fresh sequential workers, one first call '
               f'and {args.repeats} warm calls, new solver per call. Existing compilation caches retained. '
               'Timeout includes startup/imports/all calls. No prefilter, fastmath or parallel solvers. '
               'All compression defaults retained. Partial values/times retained, never called a full median.',
        escalation_rule='After two consecutive timeouts in an algorithm/dimension/family, larger sizes '
                        'in that series are explicitly marked not_run_after_timeouts; no time is inferred.',
        rows=[])
    streak, errors = {}, 0
    for case in cases:
        order = list(ALGORITHMS)
        random.Random(case['id']).shuffle(order)
        current = []
        for algorithm in order:
            key = (algorithm, case['d'], case['family'])
            row = {k:v for k,v in case.items() if k != 'points'}
            row['algorithm'] = algorithm
            if streak.get(key, 0) >= 2:
                row.update(status='not_run_after_timeouts', samples=[], reason='two consecutive smaller-input worker timeouts')
            else:
                row.update(isolated(algorithm, case, args.repeats, args.timeout))
                streak[key] = streak.get(key, 0)+1 if row['status']=='timeout' else 0
            current.append(row)
            report['rows'].append(row)
            previous.save(report, args.out)
            print(case['id'], algorithm, row['status'], row.get('median_seconds'),
                  'first', row.get('first_seconds'), 'warm', len(row.get('samples', [])), flush=True)
        errors += validate(current)
        previous.save(report, args.out)
    report.update(finished_utc=datetime.now(timezone.utc).isoformat(), validation_errors=errors)
    previous.save(report, args.out)
    return int(errors != 0 or any(row['status'] in ('error','process_failed') for row in report['rows']))


if __name__ == '__main__':
    mp.freeze_support()
    raise SystemExit(main())
