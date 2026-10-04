"""Bounded probe of default adaptive compression on common-extent boxes.

This is a structural demonstration, not a speed benchmark. Each solver gets
one fresh, sequential worker. Imports, cache loading and its one computation
all count against the worker budget. No compression parameter is overridden.
"""
import argparse
from datetime import datetime, timezone
from itertools import combinations
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
from pathlib import Path
import platform
import sys
import time
import traceback


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]


def make_case(steps, hard_count):
    """Six planar staircases share their other coordinates' common extent.

    In each axis pair the staircase corners are (t, steps+1-t), for
    t=1,...,steps. The remaining two coordinates equal M. These boxes are
    therefore absorbed as pair constraints at the root, retaining long rank
    arrays. The additional corners are genuinely active on all four axes.
    Their coordinate sum is constant, so none dominates another. Their
    coordinates are above (steps+1)/2, so no planar staircase covers them.
    """
    extent = steps + hard_count + 28
    points = []
    for first, second in combinations(range(4), 2):
        for value in range(1, steps + 1):
            point = [extent] * 4
            point[first], point[second] = value, steps + 1 - value
            points.append(point)
    base = max((steps + 1) // 2 + 1, round(.63 * steps))
    stride = next(value for value in range(3, hard_count + 2, 2)
                  if math.gcd(value, hard_count) == 1)
    for i in range(hard_count):
        j = stride * i % hard_count
        points.append([base + i, base + hard_count - 1 - i,
                       base + j, base + hard_count - 1 - j])
    if max(max(point) for point in points[-hard_count:]) >= extent:
        raise ValueError('Choose more staircase steps for this hard-corner count.')
    return dict(id=f'common-extent-stairs-{steps}-hard-{hard_count}',
                steps=steps, hard_corners=hard_count, d=4, n=len(points),
                common_extent=extent, staircase_boxes=6 * steps,
                points=points)


def worker(pipe, algorithm, case):
    sys.path.insert(0, str(REPOSITORY))
    try:
        import_started = time.perf_counter()
        points = case['points']
        if algorithm == 'compiled_default':
            from NumericalChanHVND.numerical_chan_compiled import NumericalChanCompiled
            from NumericalChanHVND import compiled_terms

            class ObservedDefault(NumericalChanCompiled):
                def _compress_state(self, boxes, terms, lo, hi):
                    started = time.perf_counter()
                    # This is the whole run's high-water counter, not the
                    # current branch's generation. Final max_generation is
                    # the recorded test for repeated compression on a path.
                    pipe.send(dict(event='compression_started',
                                   max_generation_seen=self.stats['max_generation'],
                                   hard_boxes=len(boxes), terms=len(terms),
                                   array_entries_and_terms=int(compiled_terms.complexity(terms)),
                                   threshold=self.compress_factor * self.d**2 * (len(boxes) + 2),
                                   rank_sizes=[b - a + 1 for a, b in zip(lo, hi)]))
                    result = super()._compress_state(boxes, terms, lo, hi)
                    pipe.send(dict(event='compression_completed',
                                   seconds=time.perf_counter() - started,
                                   terms=len(result[1]),
                                   array_entries_and_terms=int(compiled_terms.complexity(result[1]))))
                    return result

            engine = ObservedDefault(dimension=4)
            solve = lambda: float(engine.compute(points))
            counters = lambda: dict(engine.stats)
        elif algorithm == 'chan_dby3':
            from chan_orthant_dby3 import ChanOrthantMeasure
            corners = [tuple(-float(x) for x in point) for point in points]
            lo = tuple(min(point[axis] for point in corners) for axis in range(4))
            engine = ChanOrthantMeasure(4)
            solve = lambda: math.prod(-x for x in lo) - engine.complement_measure(corners, lo, (0.,) * 4)
            counters = lambda: dict(nodes=engine.node_count, compressions=engine.compress_count,
                                    peak_terms=engine.term_high_water)
        else:
            from chan_hypervolume import hypervolume
            corners = [tuple(-float(x) for x in point) for point in points]
            solve = lambda: float(hypervolume(corners, (0.,) * 4, prefilter=False))
            counters = lambda: {}
        pipe.send(dict(event='imports_complete', seconds=time.perf_counter() - import_started))
        started = time.perf_counter()
        value = solve()
        pipe.send(dict(event='done', status='ok', value=value,
                       computation_seconds=time.perf_counter() - started,
                       counters=counters()))
    except Exception as error:
        pipe.send(dict(event='done', status='error', error=repr(error),
                       traceback=traceback.format_exc()))
    finally:
        pipe.close()


def isolated(algorithm, case, timeout):
    reader, writer = mp.Pipe(duplex=False)
    process = mp.Process(target=worker, args=(writer, algorithm, case))
    started = time.monotonic()
    process.start()
    writer.close()
    events = []
    try:
        deadline = started + timeout
        while time.monotonic() < deadline:
            if not reader.poll(min(.1, max(0., deadline - time.monotonic()))):
                continue
            try:
                event = reader.recv()
            except EOFError:
                process.join(2)
                return dict(algorithm=algorithm, status='process_failed',
                            exit_code=process.exitcode, events=events)
            if event['event'] == 'done':
                process.join(2)
                return dict(algorithm=algorithm, events=events,
                            worker_seconds=time.monotonic() - started, **event)
            events.append(event)
        return dict(algorithm=algorithm, status='timeout',
                    budget_seconds=timeout, events=events)
    finally:
        if process.is_alive():
            process.terminate()
        process.join(3)
        reader.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, nargs='+', default=[256])
    parser.add_argument('--hard', type=int, default=16)
    parser.add_argument('--timeout', type=float, default=45.)
    parser.add_argument('--out', type=Path, default=HERE / 'results' / 'natural_compression_probe.json')
    args = parser.parse_args()
    if min(args.steps) < 16 or args.hard < 3 or args.timeout <= 0:
        parser.error('steps must be at least 16, hard at least 3, and timeout positive')
    if args.out.exists():
        parser.error('Choose a new --out path; existing measurements are never overwritten')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    source_files = [Path(__file__)] + [REPOSITORY / name for name in (
        'NumericalChanHVND/numerical_chan_compiled.py',
        'NumericalChanHVND/compiled_terms.py',
        'NumericalChanHVND/numerical_chan_numba.py',
        'chan_orthant_dby3.py', 'chan_hypervolume.py')]
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  method='Fresh sequential workers, one computation each; defaults unchanged. '
                         'Compiled compression boundaries observed by a subclass without changing decisions. '
                         'Timeout includes imports, cache loading, and computation. '
                         'These are structural probes, not warm timing benchmarks.',
                  worker_timeout_seconds=args.timeout,
                  environment=dict(python=sys.version, platform=platform.platform(),
                                   numpy=importlib.metadata.version('numpy'),
                                   numba=importlib.metadata.version('numba')),
                  source_sha256={path.relative_to(REPOSITORY).as_posix():
                                 hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in source_files},
                  cases=[])
    for steps in args.steps:
        case = make_case(steps, args.hard)
        entry = dict(case, rows=[])
        report['cases'].append(entry)
        for algorithm in ('compiled_default', 'chan_dby3', 'chan_dby2'):
            row = isolated(algorithm, case, args.timeout)
            entry['rows'].append(row)
            args.out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
            print(case['id'], algorithm, row['status'], row.get('computation_seconds'),
                  row.get('counters', {}), flush=True)
        baseline = next(row for row in entry['rows'] if row['algorithm'] == 'chan_dby2')
        if baseline['status'] == 'ok':
            for row in entry['rows']:
                if row['status'] == 'ok':
                    row['scaled_error'] = abs(row['value'] - baseline['value']) / max(1., abs(baseline['value']))
                    row['value_agrees'] = math.isclose(row['value'], baseline['value'], rel_tol=2e-9, abs_tol=2e-10)
        args.out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    mp.freeze_support()
    main()
