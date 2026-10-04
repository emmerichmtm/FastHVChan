"""Independent numerical and checkpoint checks for the local compiled policy.

Run with NumPy and Numba installed::

    python NumericalChanHVND/verify_local_compiled.py

The exact rational oracle and input families are shared with verify_compiled;
no implementation algebra is used by that oracle. This is a bounded
correctness suite, not a benchmark or an empirical complexity proof.
"""
from collections import Counter
from fractions import Fraction
from pathlib import Path
import argparse
import hashlib
import json
from math import isclose
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from NumericalChanHVND import numerical_chan_local_compiled as compiled
from NumericalChanHVND.verify_compiled import inclusion_exclusion, small_cases, repeated_points
from NumericalChanHVND.benchmarks.compiled_2d_10d.natural_compression_probe import make_case
from chan_hypervolume import hypervolume as chan_dby2

REL_TOL = 2e-9
ABS_TOL = 2e-10


class AuditedLocal(compiled.NumericalChanLocalCompiled):
    """Observe recursion arguments without changing state or scheduling."""

    def compute(self, points, magnitude=False):
        self.frames, self.block_events = [], []
        self.advance_checks = 0
        self.max_checkpoint_generation = 0
        return super().compute(points, magnitude=magnitude)

    def _node(self, boxes, terms, lo, hi, axis, remaining, generation):
        if self.frames:
            parent = self.frames[-1]
            # This covers both real cuts and skipped axes. It also detects
            # an expired checkpoint budget reused after a cheapness skip.
            assert remaining == parent['budget'] - 1, (remaining, parent)
            assert axis == (parent['axis'] + 1) % self.d, (axis, parent)
            self.advance_checks += 1
            checkpoint_generation = parent['checkpoint_generation']
        else:
            assert remaining is None
            checkpoint_generation = 0
        assert remaining is None or remaining >= 0, remaining
        self.frames.append(dict(axis=axis, budget=remaining,
                                checkpoint_generation=checkpoint_generation))
        try:
            return super()._node(boxes, terms, lo, hi, axis, remaining, generation)
        finally:
            self.frames.pop()

    def _new_block(self, boxes, lo, hi, axis):
        frame = self.frames[-1]
        assert frame['budget'] in (None, 0), frame
        at_checkpoint = frame['budget'] == 0
        levels = super()._new_block(boxes, lo, hi, axis)
        assert levels >= self.d and levels % self.d == 0, (self.d, levels)
        frame['budget'] = levels
        frame['checkpoint_generation'] += int(at_checkpoint)
        self.max_checkpoint_generation = max(self.max_checkpoint_generation,
                                              frame['checkpoint_generation'])
        self.block_events.append(dict(levels=levels, hard_boxes=len(boxes),
                                      at_checkpoint=at_checkpoint,
                                      checkpoint_generation=frame['checkpoint_generation']))
        return levels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).with_name('local_compiled_verification.json'))
    args = parser.parse_args()
    started = time.perf_counter()
    checks, records = Counter(), []
    max_error = 0.

    def check(actual, expected, label):
        nonlocal max_error
        actual, expected = float(actual), float(expected)
        assert isclose(actual, expected, rel_tol=REL_TOL, abs_tol=ABS_TOL), (
            label, actual, expected)
        max_error = max(max_error, abs(actual - expected) / max(1., abs(expected)))

    def record(label, dimension, mode, measure, solver, answer):
        checks['cyclic_advance_and_budget_checks'] += solver.advance_checks
        assert not solver.frames
        assert len(solver.block_events) == solver.stats.get('blocks', 0)
        assert solver.stats.get('blocks', 0) <= solver.stats.get('checkpoints', 0) + 1
        records.append(dict(case=label, dimension=dimension, mode=mode, magnitude=measure,
                            value=float(answer), stats=dict(solver.stats),
                            max_checkpoint_generation=solver.max_checkpoint_generation,
                            block_events=solver.block_events))

    for dimension in range(2, 11):
        for mode, factor in (('default', 8.), ('compress-at-every-checkpoint', 0.)):
            solver = AuditedLocal(dimension=dimension, compress_factor=factor)
            for label, points in small_cases(dimension):
                if factor == 0 and dimension >= 6 and len(points) > 3:
                    points = points[:3]
                    label += '-bounded3'
                for measure in (False, True):
                    answer = solver.compute(points, magnitude=measure)
                    check(answer, inclusion_exclusion(points, measure),
                          (dimension, mode, label, measure))
                    checks['exact_end_to_end_instances'] += 1
                    record(label, dimension, mode, measure, solver, answer)
        points = small_cases(dimension)[2][1]
        transformed = [tuple(1 + Fraction(x) / 2 for x in p) for p in points]
        check(compiled.magnitude(points, dimension=dimension),
              compiled.hypervolume(transformed, dimension=dimension),
              (dimension, 'forward magnitude transformation'))
        positive = small_cases(dimension)[4][1][:3]
        inverse = [tuple(2 * (3 * x - 1) for x in p) for p in positive]
        check(compiled.hypervolume(positive, dimension=dimension),
              compiled.magnitude(inverse, dimension=dimension) / 3 ** dimension,
              (dimension, 'inverse magnitude transformation'))
        checks['two_way_magnitude_transformations'] += 2
        print(f'{dimension}D: default and forced checkpoints, exact values and transformations passed.',
              flush=True)

    # Retain four interacting axes while varying the actual lifted dimension.
    # Compare all modes against the same exact 4D rational oracle times the
    # common extra-axis mass, and observe successive checkpoint generations.
    deep = repeated_points()
    for dimension in (4, 5, 7, 10):
        points = [p + (2,) * (dimension - 4) for p in deep]
        for mode, factor in (('default', 8.), ('forced-compression', 0.),
                             ('cheapness-skips', 1e12)):
            for measure in (False, True):
                solver = AuditedLocal(dimension, base_hard=1, compress_factor=factor)
                answer = solver.compute(points, magnitude=measure)
                check(answer, inclusion_exclusion(deep, measure) * 2 ** (dimension - 4),
                      (dimension, mode, 'local checkpoint recursion', measure))
                assert solver.stats.get('checkpoints', 0) > 0, (dimension, mode, dict(solver.stats))
                if factor == 0:
                    assert solver.stats.get('compressions', 0) > 0, dict(solver.stats)
                    checks['positive_local_compression_instances'] += 1
                elif factor == 1e12:
                    assert solver.stats.get('compressions', 0) == 0, dict(solver.stats)
                    assert solver.stats.get('compression_skipped_small_state', 0) > 0
                    checks['positive_cheapness_skip_instances'] += 1
                checks['exact_local_checkpoint_recursion_instances'] += 1
                checks['instances_with_repeated_checkpoint_generations'] += int(
                    solver.max_checkpoint_generation >= 2)
                record('local-checkpoint-recursion-12', dimension, mode, measure, solver, answer)
        print(f'{dimension}D: compression and cheapness-skip checkpoints passed.', flush=True)
    # A larger but inexpensive antichain gives two checkpoints on one path.
    # Use an independent geometric solver instead of exponential 64-box IE.
    permutation = list(range(1, 65))
    random.Random(68419).shuffle(permutation)
    points = [(i, 65 - i, j, 65 - j)
              for i, j in zip(range(1, 65), permutation)]
    for measure in (False, True):
        transformed = ([tuple(1 + x / 2 for x in p) for p in points]
                       if measure else points)
        expected = chan_dby2([tuple(-x for x in p) for p in transformed],
                             (0.,) * 4, prefilter=False)
        for mode, factor in (('forced-compression', 0.), ('cheapness-skips', 1e12)):
            solver = AuditedLocal(4, base_hard=1, compress_factor=factor)
            answer = solver.compute(points, magnitude=measure)
            check(answer, expected, ('successive local checkpoints', mode, measure))
            assert solver.max_checkpoint_generation >= 2, dict(solver.stats)
            if factor == 0:
                assert solver.stats.get('max_generation', 0) >= 2, dict(solver.stats)
            else:
                assert solver.stats.get('compressions', 0) == 0, dict(solver.stats)
                assert solver.stats.get('compression_skipped_small_state', 0) >= 2
            checks['successive_checkpoints_reference_comparisons'] += 1
            checks['instances_with_repeated_checkpoint_generations'] += 1
            record('permuted-antichain-64', 4, mode, measure, solver, answer)
    print('Successive local checkpoints pass with compression and with cheapness skips.', flush=True)

    # The already published common-extent construction naturally crosses the
    # default threshold. Its size is large because its many planar boxes are
    # immediately absorbed; only sixteen corners enter geometric recursion.
    case = make_case(256, 16)
    points = case['points']
    for measure in (False, True):
        solver = AuditedLocal(4)
        answer = solver.compute(points, magnitude=measure)
        transformed = ([tuple(1 + x / 2 for x in p) for p in points]
                       if measure else points)
        expected = chan_dby2([tuple(-x for x in p) for p in transformed],
                             (0.,) * 4, prefilter=False)
        check(answer, expected, ('natural default compression', measure))
        if not measure:
            check(answer, 6372330876, 'published natural-probe HV')
        assert solver.stats.get('compressions', 0) > 0, dict(solver.stats)
        checks['natural_default_compression_reference_comparisons'] += 1
        record(case['id'], 4, 'default', measure, solver, answer)
    print('Natural default local compression agrees with independent Chan d/2.', flush=True)

    probe = '''
import json
from NumericalChanHVND.numerical_chan_local_compiled import hypervolume, magnitude
p = [(1,4,3,2,2),(2,3,4,1,2),(3,2,1,4,2),(4,1,2,3,2)]
print(json.dumps({'hv':hypervolume(p),'magnitude':magnitude(p)}))
'''
    result = subprocess.run([sys.executable, '-c', probe], cwd=ROOT,
                            capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 0, (result.stdout, result.stderr)
    result = json.loads(result.stdout)
    check(result['hv'], 138, 'fresh-process local HV')
    check(result['magnitude'], Fraction(765, 8), 'fresh-process local magnitude')
    checks['fresh_process_canonical_package_cache'] += 1

    files = [Path(__file__), ROOT / 'NumericalChanHVND/numerical_chan_local_compiled.py',
             ROOT / 'NumericalChanHVND/compiled_terms.py',
             ROOT / 'NumericalChanHVND/numerical_chan_compiled.py',
             ROOT / 'NumericalChanHVND/numerical_chan_numba.py']
    report = dict(status='All local compiled 2--10D checks passed.',
                  dimensions=list(range(2, 11)), checks=dict(checks), max_scaled_error=max_error,
                  tolerance=dict(relative=REL_TOL, absolute=ABS_TOL),
                  seconds_including_compilation=time.perf_counter() - started,
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in files},
                  scope='Exact small-instance oracles and observed checkpoint invariants; '
                        'natural default compression compared with independent Chan d/2. '
                        'Float64 checks, not a benchmark or an asymptotic proof.',
                  cases=records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'cases'}, indent=2))


if __name__ == '__main__':
    main()
