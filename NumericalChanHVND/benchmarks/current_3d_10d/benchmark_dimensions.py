"""Benchmark the current numerical backends and Python Chan in dimensions 3-10.

One fresh worker at a time; first use and every warm repetition are archived.
The 4D dispatcher selects whole contractions; other dimensions retain their
existing sweep (3D) or hybrid (5-10D) backend. No algorithm code is modified.
"""
import argparse
from datetime import datetime, timezone
import csv
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

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
ALGORITHMS = ('numerical_python', 'current_numba', 'chan_dby3', 'chan_dby2')
PLAN = {3: (32, 128), 4: (12, 24), 5: (8, 16), 6: (8, 12),
        7: (6, 8), 8: (6, 10), 9: (4, 6), 10: (4, 6)}


def make_cases():
    cases = []
    for family in ('sphere', 'tied_grid'):
        for dimension, sizes in PLAN.items():
            for size in sizes:
                seed = 101000 + 1000 * dimension + 10 * size + (family == 'sphere')
                rng = random.Random(seed)
                points = []
                for _ in range(size):
                    if family == 'sphere':
                        vector = [abs(rng.gauss(0., 1.)) + .01 for _ in range(dimension)]
                        norm = math.sqrt(sum(x * x for x in vector))
                        points.append([x / norm for x in vector])
                    else:
                        points.append([rng.randrange(1, 8) / 8 for _ in range(dimension)])
                case = dict(id=f'{family}-d{dimension}-n{size}-hv', family=family,
                            d=dimension, n=size, seed=seed, magnitude=False, points=points)
                cases.append(case)
                if family == 'sphere' and size == sizes[0]:
                    cases.append({**case, 'id': case['id'][:-2] + 'mag', 'magnitude': True})
    return cases


def setup(algorithm, case):
    sys.path.insert(0, str(REPOSITORY))
    os.environ.pop('NUMBA_DISABLE_JIT', None)
    points, dimension, magnitude = case['points'], case['d'], case['magnitude']
    if algorithm in ('numerical_python', 'current_numba', 'hybrid_4d'):
        if algorithm == 'numerical_python':
            from NumericalChanHVND.numerical_chan import NumericalChan
            make_engine = lambda: NumericalChan(dimension=dimension)
        elif algorithm == 'current_numba' and dimension == 4:
            from NumericalChanHV4D.numerical_chan4_numba import NumericalChan4Numba
            make_engine = NumericalChan4Numba
        else:
            from NumericalChanHVND.numerical_chan_numba import NumericalChanNumba
            make_engine = lambda: NumericalChanNumba(dimension=dimension)

        def solve():
            engine = make_engine()
            value = engine.compute(points, magnitude=magnitude)
            return float(value), dict(engine.stats)
    else:
        if algorithm == 'chan_dby3':
            from chan_orthant_dby3 import hypervolume_dby3 as hypervolume
        else:
            from chan_hypervolume import hypervolume

        def solve():
            # The public Chan interfaces use minimization points and a reference.
            corners = [[-(1 + x / 2) if magnitude else -x for x in p] for p in points]
            return float(hypervolume(corners, (0.,) * dimension, prefilter=False)), {}
    return solve


def signatures(algorithm, dimension):
    if algorithm not in ('current_numba', 'hybrid_4d'):
        return {}
    if algorithm == 'current_numba' and dimension == 4:
        from NumericalChanHV4D import compiled_terms as kernel
        names = ('base_sum', 'push_blocks', 'apply_easy')
    else:
        from NumericalChanHVND import numerical_chan_numba as kernel
        names = ('sweep_ranked',) if dimension == 3 else (
            'nonzero', 'direction', 'prefix_sum', 'search_array', 'normalize', 'suffix_mask')
    return {name: len(getattr(kernel, name).nopython_signatures) for name in names}


def worker(pipe, algorithm, case, repeats):
    try:
        start = time.perf_counter()
        solve = setup(algorithm, case)
        imports_seconds = time.perf_counter() - start
        start = time.perf_counter()
        value, counters = solve()
        first = dict(first_seconds=time.perf_counter() - start, first_value=value,
                     imports_seconds=imports_seconds, first_counters=counters)
        pipe.send(dict(event='first', **first))
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            value, counters = solve()
            sample = dict(seconds=time.perf_counter() - start, value=value, counters=counters)
            samples.append(sample)
            pipe.send(dict(event='sample', sample=sample))
        seconds = [sample['seconds'] for sample in samples]
        pipe.send(dict(event='done', status='ok', **first, value=value, counters=counters,
                       samples=samples, seconds=seconds, median_seconds=statistics.median(seconds),
                       nopython_signatures=signatures(algorithm, case['d'])))
    except Exception as error:
        pipe.send(dict(event='done', status='error', error=repr(error)))
    finally:
        pipe.close()


def isolated(algorithm, case, repeats, timeout):
    reader, writer = mp.Pipe(duplex=False)
    process = mp.Process(target=worker, args=(writer, algorithm, case, repeats))
    process.start()
    writer.close()
    deadline, first, samples = time.monotonic() + timeout, {}, []
    try:
        while time.monotonic() < deadline:
            if not reader.poll(min(.1, max(0., deadline - time.monotonic()))):
                continue
            try:
                result = reader.recv()
            except EOFError:
                process.join(2)
                return dict(first, status='process_failed', exit_code=process.exitcode, samples=samples)
            if result['event'] == 'done':
                process.join(2)
                return result
            if result['event'] == 'first':
                first = {key: value for key, value in result.items() if key != 'event'}
            else:
                samples.append(result['sample'])
        return dict(first, status='timeout', budget_seconds=timeout, samples=samples)
    finally:
        if process.is_alive():
            process.terminate()
        process.join(3)
        reader.close()


def save(report, out):
    (out / 'timings.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    with (out / 'timings.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, extrasaction='ignore', fieldnames=(
            'id', 'd', 'n', 'family', 'magnitude', 'algorithm', 'backend', 'status',
            'value', 'first_seconds', 'imports_seconds', 'median_seconds', 'scaled_error'))
        writer.writeheader()
        writer.writerows(report['rows'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=HERE / 'results')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=45.)
    parser.add_argument('--only', default='')
    args = parser.parse_args()
    if args.repeats < 1 or args.timeout <= 0:
        parser.error('repeats and timeout must be positive')
    cases = [case for case in make_cases() if args.only in case['id']]
    if not cases:
        parser.error('No matching inputs')
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'timings.json').exists():
        parser.error('Choose a new output directory; existing timings are never overwritten')
    dataset = args.out / 'datasets.json'
    dataset.write_text(json.dumps(cases, indent=2) + '\n', encoding='utf-8')
    files = [Path(__file__), dataset] + [REPOSITORY / relative for relative in (
        'NumericalChanHVND/numerical_chan.py', 'NumericalChanHVND/numerical_chan_numba.py',
        'NumericalChanHV4D/numerical_chan4_numba.py', 'NumericalChanHV4D/numerical_chan4_compiled.py',
        'NumericalChanHV4D/compiled_terms.py', 'chan_orthant_dby3.py', 'chan_hypervolume.py')]
    try:
        revision = subprocess.check_output(['git', '-c', f'safe.directory={REPOSITORY.as_posix()}',
            '-C', str(REPOSITORY), 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    backends = {str(d): ('compiled sweep' if d == 3 else
                          'whole-contraction JIT' if d == 4 else 'hybrid Numba') for d in PLAN}
    report = dict(started_utc=datetime.now(timezone.utc).isoformat(),
        environment=dict(python=sys.version, platform=platform.platform(), cpu=platform.processor(),
                         logical_cpus=os.cpu_count(), numpy=importlib.metadata.version('numpy'),
                         numba=importlib.metadata.version('numba')),
        source_revision=revision,
        source_sha256={str(path.relative_to(REPOSITORY)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in files},
        method='Sequential fresh processes; separate first call, then three warm end-to-end calls. '
               'Imports excluded from call durations; included in total worker budget. '
               'Existing disk caches retained; first use is not an empty-cache measurement. '
               'API transforms and preprocessing included. No dominance prefilter, fastmath or parallelism.',
        backend_by_dimension=backends, repeats=args.repeats, worker_timeout_seconds=args.timeout,
        base_hard=2, compression_schedule='default', tolerance=dict(relative=2e-9, absolute=2e-10),
        rows=[])
    errors = 0
    for case in cases:
        order = list(ALGORITHMS) + (['hybrid_4d'] if case['d'] == 4 else [])
        random.Random(case['id']).shuffle(order)
        current = []
        for algorithm in order:
            row = {key: value for key, value in case.items() if key != 'points'}
            backend = backends[str(case['d'])] if algorithm == 'current_numba' else algorithm
            row.update(algorithm=algorithm, backend=backend)
            row.update(isolated(algorithm, case, args.repeats, args.timeout))
            current.append(row)
            report['rows'].append(row)
            save(report, args.out)
            print(case['id'], algorithm, row['status'], row.get('median_seconds', ''), flush=True)
        baseline = next(row for row in current if row['algorithm'] == 'chan_dby2')
        for row in current:
            if row['status'] in ('error', 'process_failed'):
                errors += 1
            values = ([row['first_value']] if 'first_value' in row else [])
            values.extend(sample['value'] for sample in row.get('samples', []))
            if not values:
                continue
            if baseline['status'] != 'ok':
                row['validation'] = 'unavailable: Chan d/2 did not complete'
                errors += 1
                continue
            expected = baseline['value']
            row['scaled_error'] = max(abs(value - expected) / max(1., abs(expected)) for value in values)
            row['validation'] = 'all available values agree with Chan d/2'
            if not all(math.isclose(value, expected, rel_tol=2e-9, abs_tol=2e-10) for value in values):
                row['status'], row['validation'] = 'value_mismatch', 'failed'
                errors += 1
        save(report, args.out)
    report.update(validation_errors=errors, finished_utc=datetime.now(timezone.utc).isoformat())
    save(report, args.out)
    return int(errors != 0)


if __name__ == '__main__':
    mp.freeze_support()
    raise SystemExit(main())
