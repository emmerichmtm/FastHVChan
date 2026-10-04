"""Reproducible, isolated-process timings against unmodified FastHVChan.

Run: python benchmark_numerical_chan.py --reference-root PATH --out DIRECTORY
Each case/algorithm gets one untimed first call, then repeated warm calls.
Timeouts are recorded, never treated as measured speedups. No dominance filter.
"""
import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time


def make_cases():
    cases = []
    plan = {2: (64, 256), 3: (32, 128), 4: (12, 24, 48), 5: (8, 16),
            6: (8, 12), 7: (8,), 8: (6, 10), 9: (6,), 10: (6, 8)}
    for family in ('sphere', 'tied_grid'):
        for d, counts in plan.items():
            for n in counts:
                rng = random.Random(101000 + 1000 * d + 10 * n + (family == 'sphere'))
                points = []
                for _ in range(n):
                    if family == 'sphere':
                        v = [abs(rng.gauss(0., 1.)) + .01 for _ in range(d)]
                        norm = math.sqrt(sum(x * x for x in v))
                        points.append([x / norm for x in v])
                    else:
                        points.append([rng.randrange(1, 8) / 8 for _ in range(d)])
                cases.append({'id': f'{family}-d{d}-n{n}-hv', 'family': family,
                              'd': d, 'n': n, 'magnitude': False, 'points': points})
                if (d, n) in ((4, 12), (7, 8), (10, 6)):
                    cases.append({**cases[-1], 'id': f'{family}-d{d}-n{n}-mag', 'magnitude': True})
    return cases


def worker(connection, algorithm, case, reference_root, repeats):
    try:
        points, d, mag = case['points'], case['d'], case['magnitude']
        if algorithm == 'numerical_python':
            from numerical_chan import hypervolume
            solve = lambda: hypervolume(points, magnitude=mag)
        elif algorithm == 'numerical_numba':
            from numerical_chan_numba import hypervolume
            solve = lambda: hypervolume(points, magnitude=mag)
        else:
            sys.path.insert(0, reference_root)
            if algorithm == 'chan_dby3':
                from chan_orthant_dby3 import hypervolume_dby3 as hypervolume
            else:
                from chan_hypervolume import hypervolume
            def solve():
                corners = [[-(1 + x / 2) if mag else -x for x in p] for p in points]
                return hypervolume(corners, (0.,) * d, prefilter=False)
        start = time.perf_counter()
        value = solve()
        first_call = time.perf_counter() - start
        connection.send({'event': 'warmup_done', 'first_call_seconds': first_call, 'value': float(value)})
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            value = solve()
            times.append(time.perf_counter() - start)
        connection.send({'event': 'finished', 'status': 'ok', 'value': float(value),
                         'first_call_seconds': first_call, 'seconds': times,
                         'median_seconds': statistics.median(times)})
    except Exception as error:
        connection.send({'event': 'finished', 'status': 'error', 'error': repr(error)})
    finally:
        connection.close()


def isolated(algorithm, case, root, repeats, timeout):
    receiving, sending = mp.Pipe(duplex=False)
    process = mp.Process(target=worker, args=(sending, algorithm, case, root, repeats))
    process.start()
    sending.close()
    deadline, warmup = time.monotonic() + timeout, {}
    try:
        while time.monotonic() < deadline:
            if receiving.poll(min(.2, max(0., deadline - time.monotonic()))):
                message = receiving.recv()
                if message['event'] == 'finished':
                    process.join(2)
                    return message
                warmup = message
            if not process.is_alive():
                break
        if process.is_alive():
            process.terminate()
            process.join(5)
            return {'status': 'timeout', 'timeout_seconds': timeout, **warmup}
        return {'status': 'process_failed', 'exit_code': process.exitcode, **warmup}
    finally:
        receiving.close()
        if process.is_alive():
            process.terminate()
            process.join(5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('benchmark_results'))
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=45.)
    parser.add_argument('--only', default='')
    parser.add_argument('--algorithms', default='numerical_python,numerical_numba,chan_dby3,chan_dby2')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cases = [c for c in make_cases() if args.only in c['id']]
    (args.out / 'datasets.json').write_text(json.dumps(cases, indent=2) + '\n')
    import numpy, numba
    report = {'environment': {'python': sys.version, 'platform': platform.platform(),
              'cpu': platform.processor(), 'numpy': numpy.__version__, 'numba': numba.__version__,
              'logical_cpus': os.cpu_count()}, 'repeats': args.repeats, 'timeout_seconds': args.timeout,
              'reference_sha256': {name: hashlib.sha256((args.reference_root / name).read_bytes()).hexdigest()
                                   for name in ('chan_orthant_dby3.py', 'chan_hypervolume.py')},
              'timing_scope': 'End-to-end float64 call including rank preprocessing and API transforms; '
                              'imports excluded; first call recorded separately; warm median reported. '
                              'Numba backend is hybrid, not a fully compiled recursion. '
                              'All algorithms have one untimed first call; dominance prefilter disabled.',
              'rows': []}
    for case in cases:
        for algorithm in args.algorithms.split(','):
            if case['d'] == 2 and algorithm == 'chan_dby3':
                continue
            row = {k: v for k, v in case.items() if k != 'points'}
            row['algorithm'] = algorithm
            row.update(isolated(algorithm, case, str(args.reference_root.resolve()), args.repeats, args.timeout))
            report['rows'].append(row)
            (args.out / 'timings.json').write_text(json.dumps(report, indent=2) + '\n')
            print(case['id'], algorithm, row['status'], row.get('median_seconds', ''), flush=True)
        values = [r for r in report['rows'] if r['id'] == case['id'] and r['status'] == 'ok']
        if values:
            reference = values[0]['value']
            for row in values:
                row['scaled_difference'] = abs(row['value'] - reference) / max(1., abs(reference))
                if not math.isclose(row['value'], reference, rel_tol=2e-8, abs_tol=1e-9):
                    row['status'] = 'value_mismatch'
    (args.out / 'timings.json').write_text(json.dumps(report, indent=2) + '\n')
    fields = ['id', 'd', 'n', 'family', 'magnitude', 'algorithm', 'status', 'value',
              'first_call_seconds', 'median_seconds', 'scaled_difference']
    with (args.out / 'timings.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(report['rows'])


if __name__ == '__main__':
    mp.freeze_support()
    main()
