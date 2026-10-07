"""Reproduce the report's eight-point example; no dependencies unless --compiled.

Michael Emmerich's FastHVChan implementation project, October 2026.
The exhaustive checks here are tiny-instance oracles, not fast HV algorithms.
"""
import argparse
from itertools import combinations, product
import json
from math import prod
from pathlib import Path
import sys

POINTS = [(5, 1, 5, 1), (5, 2, 3, 2), (4, 3, 4, 1), (3, 4, 2, 3),
          (2, 5, 1, 4), (1, 5, 2, 4), (2, 2, 5, 3), (1, 3, 3, 5)]
EXPECTED_TABLE = [[21, 21, 16, 8, 8], [16, 16, 9, 7, 4],
                  [10, 9, 8, 6, 0], [8, 7, 4, 0, 0], [8, 6, 0, 0, 0]]
REGIONS = {
    'root': [(1, 5)] * 4,
    'L': [(1, 3), (1, 5), (1, 5), (1, 5)],
    'LL': [(1, 3), (1, 3), (1, 5), (1, 5)],
    'LLL': [(1, 3), (1, 3), (1, 2), (1, 5)],
    'LLR': [(1, 3), (1, 3), (3, 5), (1, 5)],
    'LR': [(1, 3), (4, 5), (1, 5), (1, 5)],
    'LRL': [(1, 3), (4, 5), (1, 2), (1, 5)],
    'LRR': [(1, 3), (4, 5), (3, 5), (1, 5)],
    'R': [(4, 5), (1, 5), (1, 5), (1, 5)],
}
EXPECTED_UNCOVERED = dict(zip(REGIONS, [433, 216, 99, 21, 78, 117, 27, 90, 217]))


def covered(k):
    return any(all(x <= y for x, y in zip(k, a)) for a in POINTS)


def check(compiled=False):
    assert all(sum(a) == 12 and all(1 <= x <= 5 for x in a) for a in POINTS)
    assert all(not all(x <= y for x, y in zip(a, b))
               for a in POINTS for b in POINTS if a != b)
    hv = sum(covered(k) for k in product(range(1, 6), repeat=4))
    intersections = [sum(prod(min(a[i] for a in subset) for i in range(4))
                         for subset in combinations(POINTS, r)) for r in range(1, 9)]
    ie = sum((-1)**r * value for r, value in enumerate(intersections))
    table = [[sum(covered((i, j, k, l)) for k, l in product(range(1, 6), repeat=2))
              for j in range(1, 6)] for i in range(1, 6)]
    projections = [sum(any(i <= a[s] and j <= a[s+1] for a in POINTS)
                       for i, j in product(range(1, 6), repeat=2)) for s in (0, 2)]
    assert hv == ie == sum(map(sum, table)) == 192
    assert table == EXPECTED_TABLE and projections == [19, 21]
    uncovered = {name: sum(not covered(k) for k in product(
        *(range(lo, hi+1) for lo, hi in bounds))) for name, bounds in REGIONS.items()}
    assert uncovered == EXPECTED_UNCOVERED
    def leaf_mass(count_x, upper_y, upper_z, upper_t):
        return count_x * upper_z * sum(max(0, upper_t - (2 if j == 1 else 1) + 1)
                                       for j in range(1, upper_y + 1))
    leaf = [leaf_mass(*args) for args in ((2, 5, 5, 5), (2, 2, 3, 2),
                                         (1, 3, 4, 1), (1, 2, 3, 1))]
    assert leaf == [240, 18, 8, 3] and leaf[0]-leaf[1]-leaf[2]+leaf[3] == 217
    blocks_y, blocks_t = [(1, 2), (3,), (4, 5)], [(1,), (2,), (3, 4, 5)]
    compressed = [[sum(j > 1 or t > 1 for j, t in product(ys, ts))
                   for ts in blocks_t] for ys in blocks_y]
    assert compressed == [[1, 2, 6], [1, 1, 3], [2, 2, 6]]
    coarse_leaf = [2*5*sum(map(sum, compressed)),
                   2*3*sum(compressed[0][:2]),
                   4*(compressed[0][0]+compressed[1][0]), 3*compressed[0][0]]
    assert coarse_leaf == leaf
    record = dict(points=POINTS, hypervolume=hv, projection_areas=projections,
                  planar_areas=table, row_sums=list(map(sum, table)),
                  intersection_sums_by_subset_size=intersections,
                  uncovered_by_region=uncovered, right_leaf_masses=leaf,
                  illustrative_compressed_pair_masses=compressed,
                  compiled_checked=False)
    if compiled:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from NumericalChanHVND.numerical_chan import hypervolume
        from NumericalChanHVND.numerical_chan_local_compiled import NumericalChanLocalCompiled

        class Trace(NumericalChanLocalCompiled):
            def __init__(self):
                super().__init__(dimension=4)
                self.stack, self.trace = [], {}

            def _node(self, boxes, terms, lo, hi, axis, remaining, generation):
                if self.stack:
                    parent = self.stack[-1]
                    path = ('' if parent['path'] == 'root' else parent['path']) + 'LR'[parent['children']]
                    parent['children'] += 1
                else:
                    path = 'root'
                frame = dict(path=path, children=0)
                self.stack.append(frame)
                value = super()._node(boxes, terms, lo, hi, axis, remaining, generation)
                self.stack.pop()
                self.trace[path] = value
                return value

        solver = Trace()
        exact_value, compiled_value = hypervolume(POINTS), solver.compute(POINTS)
        assert exact_value == compiled_value == hv
        assert solver.trace == EXPECTED_UNCOVERED
        assert all(solver.stats[key] == val for key, val in
                   [('nodes', 9), ('cuts', 4), ('base_calls', 5), ('compressions', 0)])
        record.update(compiled_checked=True, exact_solver_hv=int(exact_value),
                      compiled_solver_hv=compiled_value, compiled_trace=solver.trace,
                      compiled_stats=dict(solver.stats))
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiled', action='store_true', help='Also check exact and Numba solvers.')
    parser.add_argument('--output', type=Path, help='Optional path for the verification record.')
    options = parser.parse_args()
    result = json.dumps(check(options.compiled), indent=2) + '\n'
    if options.output:
        options.output.write_text(result, encoding='utf-8')
    print(result)
