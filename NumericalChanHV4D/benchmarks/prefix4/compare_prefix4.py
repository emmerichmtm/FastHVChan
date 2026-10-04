"""4D-only comparison of the published prefix solver and numerical Chan engines.

All inputs use positive minimization coordinates. The older prefix solver must
not be fed the negative reflected coordinates used by generic Chan adapters.
Each process has its own imports/JIT toggle; timed processes run sequentially.
"""
import argparse
import cProfile
import csv
from fractions import Fraction
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import pstats
import random
import statistics
import sys
import time

ALGORITHMS = ('old_prefix_numba', 'old_prefix_flat_python', 'old_prefix_object_python',
              'new_python', 'new_numba', 'new_numba_no_compression', 'chan_dby3', 'chan_dby2')


def sphere(n, rng):
    points = []
    for _ in range(n):
        v = [abs(rng.gauss(0., 1.)) + 1e-12 for _ in range(4)]
        norm = math.sqrt(sum(x * x for x in v))
        points.append([x / norm for x in v])
    return points


def datasets():
    cases = []
    for seed, sizes in ((42, (5, 10, 20, 40)), (99, (20, 40))):
        rng = random.Random(seed)
        for n in sizes:
            points = sphere(n, rng)
            if n == 5:
                continue
            cases.append({'id': f'sphere-s{seed}-n{n}', 'family': 'sphere', 'seed': seed,
                          'n': n, 'points': points, 'reference': [1.2] * 4})
    rng = random.Random(7)
    cases.append({'id': 'grid-s7-n20', 'family': 'grid', 'seed': 7, 'n': 20,
                  'points': [[rng.randrange(4) / 4 for _ in range(4)] for _ in range(20)],
                  'reference': [1.] * 4})
    rng = random.Random(17)
    cases.append({'id': 'uniform-s17-n20', 'family': 'uniform', 'seed': 17, 'n': 20,
                  'points': [[rng.random() for _ in range(4)] for _ in range(20)],
                  'reference': [1.] * 4})
    return cases


def setup(algorithm, case, paths):
    old, new, reference = paths
    sys.path[:0] = [old, new, reference]
    # Must be set before any Numba import in this fresh worker.
    if algorithm == 'old_prefix_flat_python':
        os.environ['NUMBA_DISABLE_JIT'] = '1'
    else:
        os.environ.pop('NUMBA_DISABLE_JIT', None)
    points, ref = case['points'], tuple(case['reference'])
    if algorithm.startswith('old_'):
        from solver import PrefixHV4D
        def solve():
            engine = PrefixHV4D(base_hard=2, fast=algorithm != 'old_prefix_object_python')
            value = engine.hypervolume(points, ref)
            return value, dict(engine.counters)
    elif algorithm.startswith('new_'):
        if algorithm == 'new_python':
            from numerical_chan import NumericalChan as Engine
        else:
            from numerical_chan_numba import NumericalChanNumba as Engine
        def solve():
            options = {'block_levels': 10 ** 6} if algorithm.endswith('no_compression') else {}
            engine = Engine(dimension=4, base_hard=2, **options)
            corners = [[ref[i] - p[i] for i in range(4)] for p in points]
            value = engine.compute(corners)
            return value, dict(engine.stats)
    elif algorithm == 'chan_dby3':
        from chan_orthant_dby3 import ChanOrthantMeasure
        def solve():
            # The public API's float conversion and filtering, with no dominance filter.
            front = [tuple(float(x) for x in p) for p in points if all(x < r for x, r in zip(p, ref))]
            if not front:
                return 0., {}
            lo = tuple(min(p[i] for p in front) for i in range(4))
            engine = ChanOrthantMeasure(4, base_boxes=2)
            value = math.prod(r - l for l, r in zip(lo, ref)) - engine.complement_measure(front, lo, ref)
            return value, {'nodes': engine.node_count, 'peak_terms': engine.term_high_water,
                           'compressions': engine.compress_count}
    else:
        from chan_hypervolume import hypervolume
        def solve():
            return hypervolume(points, ref, prefilter=False), {}
    return solve


def worker(pipe, algorithm, case, paths, repeats, profile=False):
    try:
        solve = setup(algorithm, case, paths)
        start = time.perf_counter()
        value, counters = solve()
        first = time.perf_counter() - start
        pipe.send({'event': 'first', 'value': value, 'first_seconds': first, 'counters': counters})
        if profile:
            profiler = cProfile.Profile()
            profiler.enable()
            value, counters = solve()
            profiler.disable()
            stats = pstats.Stats(profiler)
            records = []
            for (filename, line, function), (primitive, calls, own, cumulative, callers) in stats.stats.items():
                records.append({'file': Path(filename).name, 'line': line, 'function': function,
                                'primitive_calls': primitive, 'calls': calls,
                                'self_seconds': own, 'cumulative_seconds': cumulative})
            records.sort(key=lambda x: x['self_seconds'], reverse=True)
            pipe.send({'event': 'done', 'status': 'ok', 'value': value, 'counters': counters,
                       'profile_seconds': stats.total_tt, 'calls': stats.total_calls, 'profile': records[:70]})
        else:
            times = []
            for _ in range(repeats):
                start = time.perf_counter()
                value, counters = solve()
                times.append(time.perf_counter() - start)
            metadata = {}
            if algorithm.startswith('old_prefix'):
                import fastkernel
                metadata['kernel_nopython_signatures'] = len(getattr(fastkernel.k_prefix, 'nopython_signatures', ()))
                metadata['numba_disabled'] = os.environ.get('NUMBA_DISABLE_JIT', '0')
            pipe.send({'event': 'done', 'status': 'ok', 'value': value, 'counters': counters,
                       'seconds': times, 'median_seconds': statistics.median(times),
                       'first_seconds': first, **metadata})
    except Exception as error:
        pipe.send({'event': 'done', 'status': 'error', 'error': repr(error)})
    finally:
        pipe.close()


def isolated(algorithm, case, paths, repeats, timeout, profile=False):
    reader, writer = mp.Pipe(duplex=False)
    process = mp.Process(target=worker, args=(writer, algorithm, case, paths, repeats, profile))
    process.start()
    writer.close()
    deadline, first = time.monotonic() + timeout, {}
    try:
        while time.monotonic() < deadline:
            if reader.poll(.1):
                try:
                    result = reader.recv()
                except EOFError:
                    break
                if result['event'] == 'done':
                    process.join(2)
                    return result
                first = result
            # Drain the pipe even after a successful worker exit: the final
            # result can already be buffered behind its first-call message.
        if process.is_alive():
            process.terminate()
            process.join(3)
            return {**first, 'status': 'timeout', 'budget_seconds': timeout}
        return {**first, 'status': 'process_failed', 'exit_code': process.exitcode}
    finally:
        reader.close()
        if process.is_alive():
            process.terminate()
            process.join(3)


def exact_hv(points, ref):
    from itertools import combinations
    points = [[Fraction(x) for x in p] for p in points]
    ref = [Fraction(x) for x in ref]
    result = Fraction(0)
    for size in range(1, len(points) + 1):
        for subset in combinations(points, size):
            result += (-1) ** (size + 1) * math.prod(max(Fraction(0), ref[i] - max(p[i] for p in subset)) for i in range(4))
    return float(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--old', required=True, type=Path)
    parser.add_argument('--new', required=True, type=Path)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--repeats', default=3, type=int)
    parser.add_argument('--timeout', default=35., type=float)
    parser.add_argument('--only', default='')
    parser.add_argument('--profiles', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    paths = tuple(str(x.resolve()) for x in (args.old, args.new, args.reference))
    cases = [case for case in datasets() if args.only in case['id']]
    (args.out / 'datasets.json').write_text(json.dumps(cases, indent=2) + '\n')
    import importlib.metadata
    report = {'environment': {'python': sys.version, 'platform': platform.platform(),
              'cpu': platform.processor(), 'numba': importlib.metadata.version('numba'),
              'numpy': importlib.metadata.version('numpy')}, 'repeats': args.repeats,
              'timeout_seconds': args.timeout, 'rows': [],
              'source_hashes': {str(Path(p).name) + '/' + name: hashlib.sha256((Path(p) / name).read_bytes()).hexdigest()
                 for p, files in zip(paths, [('solver.py', 'fastkernel.py'), ('numerical_chan.py', 'numerical_chan_numba.py'), ('chan_orthant_dby3.py', 'chan_hypervolume.py')]) for name in files}}
    for case in cases:
        # Rotate execution order reproducibly; still exactly one timed process at a time.
        order = list(ALGORITHMS)
        random.Random(case['id']).shuffle(order)
        for algorithm in order:
            row = {k: v for k, v in case.items() if k not in ('points', 'reference')}
            row.update(algorithm=algorithm)
            row.update(isolated(algorithm, case, paths, args.repeats, args.timeout))
            report['rows'].append(row)
            (args.out / 'timings.json').write_text(json.dumps(report, indent=2) + '\n')
            print(case['id'], algorithm, row['status'], row.get('median_seconds', ''), flush=True)
        completed = [r for r in report['rows'] if r['id'] == case['id'] and r['status'] == 'ok']
        reference = next(r['value'] for r in completed if r['algorithm'] == 'chan_dby2')
        for row in completed:
            row['scaled_error'] = abs(row['value'] - reference) / max(1., abs(reference))
            if not math.isclose(row['value'], reference, rel_tol=2e-9, abs_tol=2e-10):
                row['status'] = 'value_mismatch'
    report['independent_small_oracle'] = []
    small = next(c for c in datasets() if c['id'] == 'sphere-s42-n10')
    wanted = exact_hv(small['points'], small['reference'])
    for row in report['rows']:
        if row['id'] == small['id'] and row['status'] == 'ok':
            assert math.isclose(row['value'], wanted, rel_tol=2e-9, abs_tol=2e-10)
            report['independent_small_oracle'].append({'algorithm': row['algorithm'], 'scaled_error': abs(row['value'] - wanted) / max(1., abs(wanted))})
    (args.out / 'timings.json').write_text(json.dumps(report, indent=2) + '\n')
    with (args.out / 'timings.csv').open('w', newline='') as f:
        fields = ['id','n','family','seed','algorithm','status','value','first_seconds','median_seconds','scaled_error']
        writer = csv.DictWriter(f, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(report['rows'])
    if args.profiles:
        case = next(c for c in datasets() if c['id'] == 'sphere-s42-n20')
        profiles = []
        for algorithm in ('old_prefix_numba', 'old_prefix_object_python', 'new_numba', 'new_numba_no_compression', 'chan_dby3'):
            row = {'algorithm': algorithm, 'case': case['id'], **isolated(algorithm, case, paths, 1, args.timeout, True)}
            profiles.append(row)
            print('profile', algorithm, row['status'], flush=True)
        (args.out / 'profiles.json').write_text(json.dumps(profiles, indent=2) + '\n')


if __name__ == '__main__':
    mp.freeze_support()
    main()
